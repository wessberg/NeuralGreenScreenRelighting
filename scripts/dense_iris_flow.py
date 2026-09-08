"""Experimental pretrained correspondence on native same-eye crops.

Frozen RAFT predicts coordinates only. RGB estimates are bilinear samples of
original donor observations. Neither flow nor round-trip consistency establishes
visible iris material, clean radiance, or permission to replace a reflection.
"""
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

from iris_correspondence import disjoint_footprints
from temporal_iris import footprint_valid

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = Raft_Large_Weights.C_T_SKHT_V2


def prepare(rgb, excluded=None, pixel_aspect=1.):
    a = np.asarray(rgb, np.float32).copy()
    if a.ndim != 3 or a.shape[-1] != 3 or min(a.shape[:2]) < 2:
        raise ValueError('Expected native HxWx3 display RGB')
    if excluded is not None:
        hidden = np.asarray(excluded, bool)
        if hidden.shape != a.shape[:2]:
            raise ValueError('Native exclusion shape mismatch')
        a[hidden] = 0  # before any resize/normalization, also for reverse flow
    if not np.isfinite(a).all() or a.min() < 0 or a.max() > 255:
        raise ValueError('Expected finite display RGB in 0..255')
    if not np.isfinite(pixel_aspect) or pixel_aspect <= 0:
        raise ValueError('Invalid pixel aspect')
    h, w = a.shape[:2]
    scale = 128 / min(h, w * pixel_aspect)
    nh = int(np.ceil(h * scale / 8) * 8)
    nw = int(np.ceil(w * pixel_aspect * scale / 8) * 8)
    tensor = torch.from_numpy(a).permute(2, 0, 1)[None]
    tensor = F.interpolate(tensor, (nh, nw), mode='bilinear', align_corners=False,
                           antialias=False)
    return tensor / 127.5 - 1


def native_flow(flow, shape):
    h, w = shape
    nh, nw = flow.shape[-2:]
    out = F.interpolate(flow, (h, w), mode='bilinear', align_corners=False)
    out = out * out.new_tensor([w / nw, h / nh])[None, :, None, None]
    return out[0].permute(1, 2, 0).cpu().numpy()


class DenseEyeFlow:
    def __init__(self, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint = ROOT / 'dependencies/raft-weights' / WEIGHTS.url.rsplit('/', 1)[-1]
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        if not self.checkpoint.exists():
            torch.hub.download_url_to_file(WEIGHTS.url, str(self.checkpoint),
                                           hash_prefix=self.checkpoint.stem.rsplit('-', 1)[-1])
        checkpoint_hash = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        if not checkpoint_hash.startswith(self.checkpoint.stem.rsplit('-', 1)[-1]):
            raise ValueError('Checkpoint hash does not match the released RAFT weights')
        self.model = raft_large(weights=None).eval().to(self.device)
        self.model.load_state_dict(torch.load(self.checkpoint, map_location='cpu', weights_only=True))
        self.provenance = dict(model='torchvision RAFT large', weights=WEIGHTS.name,
            url=WEIGHTS.url, sha256=checkpoint_hash,
            torch=torch.__version__, device=self.device, updates=12, dtype='float32')

    @torch.inference_mode()
    def pair(self, target, donor, *, target_excluded=None, donor_excluded=None,
             pixel_aspect=1.):
        if np.shape(target) != np.shape(donor):
            raise ValueError('Pair must share native crop shape and pixel aspect')
        a = prepare(target, target_excluded, pixel_aspect).to(self.device)
        b = prepare(donor, donor_excluded, pixel_aspect).to(self.device)
        forward = self.model(a, b, num_flow_updates=12)[-1]
        backward = self.model(b, a, num_flow_updates=12)[-1]
        shape = np.shape(target)[:2]
        return native_flow(forward, shape), native_flow(backward, shape)


def sample_native(array, xy):
    """Exact bilinear sampler with explicit floor footprints, no OpenCV 1/32 rounding.

    Caller must reject incomplete footprints. Out-of-bounds rows return zero.
    """
    a, xy = np.asarray(array), np.asarray(xy, np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError('Expected finite Nx2 coordinates')
    out = np.zeros((len(xy),) + a.shape[2:], np.float32)
    base = np.floor(xy).astype(int)
    good = ((base >= 0) & (base < np.array(a.shape[1::-1]) - 1)).all(1)
    x, y = base[good].T
    dx, dy = (xy[good] - base[good]).T
    tail = (slice(None),) + (None,) * (a.ndim - 2)
    out[good] = (a[y,x] * ((1-dx)*(1-dy))[tail] +
                 a[y,x+1] * (dx*(1-dy))[tail] +
                 a[y+1,x] * ((1-dx)*dy)[tail] + a[y+1,x+1] * (dx*dy)[tail])
    return out


def query_flow(forward, backward, xy):
    mapped = xy + sample_native(forward, xy)
    inbound = footprint_valid(np.ones(backward.shape[:2], bool), mapped)
    residual = np.linalg.norm(mapped + sample_native(backward, mapped) - xy, axis=1)
    residual[~inbound] = np.inf
    return mapped, inbound, residual


def assess_pair(target_rgb, donor_rgb, target_ob, donor_ob, query, forward, backward):
    """Fixed-map diagnostic, with all-arm query/context footprint separation.

    No RGB-driven optimization or donor selection. Context is an observed-pixel
    score, not held out from the pretrained network's receptive field.
    """
    query = np.asarray(query, np.float32)
    qraft, _, fb = query_flow(forward, backward, query)
    qmaps = {'fixed': query.copy(), 'raft': qraft}
    delta = np.array([3., 0.], np.float32)
    query_union = np.concatenate([q for xy in qmaps.values() for q in (xy, xy+delta)])
    yy, xx = np.where(target_ob.material)
    context = np.column_stack([xx, yy]).astype(np.float32)
    context = context[disjoint_footprints(context, query)]
    cmaps = {'fixed': context, 'raft': context + sample_native(forward, context)}
    common = np.ones(len(context), bool)
    for xy in cmaps.values():
        for mapped in (xy, xy+delta):
            common &= footprint_valid(donor_ob.material, mapped)
            common &= disjoint_footprints(mapped, query_union)
    values = sample_native(target_rgb, context)
    result = {}
    for name in ('fixed', 'raft'):
        mapped, cxy = qmaps[name], cmaps[name]
        err = np.abs(sample_native(donor_rgb, cxy) - values).mean(-1)
        wrong = np.abs(sample_native(donor_rgb, cxy+delta) - values).mean(-1)
        mae = float(err[common].mean()) if common.any() else None
        margin = float((wrong[common] - err[common]).mean()) if common.any() else None
        context_pass = bool(common.sum() >= 8 and mae <= 3 and margin >= 1)
        inbound = footprint_valid(np.ones(donor_rgb.shape[:2], bool), mapped)
        candidate = inbound & footprint_valid(donor_ob.material, mapped)
        # Fixed map has no learned visibility: round trip is identically zero.
        cycle = fb if name == 'raft' else np.zeros(len(query), np.float32)
        gated = candidate & (cycle <= .5) & context_pass & target_ob.candidate_ok & donor_ob.candidate_ok
        for xy in (cxy[common], (cxy+delta)[common]):
            assert disjoint_footprints(xy, query_union).all()
        result[name] = dict(rgb=sample_native(donor_rgb, mapped), mapped=mapped,
            inbound=inbound, candidate=candidate, gated=gated, roundtrip=cycle,
            context_pixels=int(common.sum()), context_mae=mae, wrong_map_margin=margin,
            context_pass=context_pass, target_candidate_ok=target_ob.candidate_ok,
            donor_candidate_ok=donor_ob.candidate_ok,
            footprints=dict(query_union=query_union, context=cxy[common],
                            wrong=(cxy+delta)[common]))
    return result


def aggregate(pairs, method, support):
    colors = np.stack([p[method]['rgb'] for p in pairs])
    valid = np.stack([p[method][support] for p in pairs])
    data = np.ma.array(colors, mask=np.broadcast_to(~valid[..., None], colors.shape))
    rgb = np.ma.median(data, axis=0).filled(0).astype(np.float32)
    mad = np.ma.median(np.ma.abs(data-rgb).mean(-1), axis=0).filled(np.inf)
    count = valid.sum(0)
    return dict(rgb=rgb, count=count, supported=count >= 3, mad=mad,
                agreement=(count >= 3) & (mad <= 2))
