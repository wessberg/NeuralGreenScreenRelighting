"""`eye_soft_removal.py` on the GPU: the same construction, flags and outputs, batched.

Round 39. The CPU pass ran single-threaded numpy for 21 minutes per 1200 frames with the RTX
5080 at 1 % (round 38 section 5); the artist: *"Never run stuff on the CPU when it can be
avoided"* and *"Port it to the GPU instead of optimizing it on the CPU."*

What moved, and how it is laid out:

* **Every per-pixel pass runs in torch on CUDA, batched over eye-frames.** An eye window is
  42-43 px, so the eye-frames of one side are grouped by window SHAPE and each group is one
  dense `(B, h, w)` batch -- no padding, so every border rule (cv2's reflect-101 blur, the
  distance transform's image edge) is the same as on one window. Cross-frame steps (the
  atlas, the temporal shading median) gather across groups.
* **Frames are held as eye windows, not frames.** Each EXR is read once (threaded), checked
  half-exact and turned into its 8-bit source PNG ON THE GPU, and only its two eye windows
  (float32) stay on the device. The 8-bit frames stay in RAM for the written outputs.
* **The atlas registration search is one batched remap per chunk of frames** over all
  845 / 605 / 121 candidates, scored in float64.
* **What stays in numpy, on purpose:** per-frame SCALAR bookkeeping (limbus gate, lid
  bridging, gain series, lash-line curves of 101 samples, the once-per-shot canonical
  rebuild on its 121x121 grid) -- exact, and a few ms over the clip. `cv2.fillPoly`
  rasterises the landmark eyelid polygon (a drawing call); its distance transform is GPU.

**Reproduction.** Primitives are exact or float-epsilon (`eye_gpu.py --selftest`). End to
end, `validate_eye_gpu.py` compares every written pixel, weight and record field against the
CPU reference run with the same flags and reports what differs; see rounds/round39.md.

Supported flags are the round 37/38 construction (`--fill-iris` path, `--lid-line parabola`,
`--specular-reach`, `--top-shading` with `--shading-sides all`, `--shading-temporal`,
`--region-rows`, `--gain-ring lower`, `--gain-temporal`, `--lid-temporal`, `--limbus-gate`,
`--canonical-votes[-frac]`, `--fill-residual-gate`, `--pupil-target/--pupil-colour`,
`--grain 0`). Anything else is refused with a pointer to the CPU reference rather than run
half-ported.

    dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/eye_removal_gpu.py \\
        --shot music_videos/reporter_test/shots/reporter_full --name r39_gpu --count 1200 ...
"""
import hashlib
import json
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from color_pipeline import linear_to_srgb
from shot_paths import from_args
import eye_reflection_repair as r32
import eye_sequence_atlas as atl
import eye_soft_removal as esr
from limbus_fit import Track as LimbusTrack
import eye_gpu as G
from eye_gpu import DEV, F32, F64, as_t, div

T0 = time.time()


def log(*a):
    print('[%7.1fs]' % (time.time() - T0), *a, flush=True)


# =========================================================================== flags
def check_supported(args):
    """Refuse any flag the GPU port does not implement, instead of silently ignoring it."""
    bad = []
    if args.fill_iris <= 0:
        bad.append('--fill-iris must be > 0 (the round 34+ construction)')
    if args.atlas_frames:
        bad.append('--atlas-frames')
    if args.canonical_from:
        bad.append('--canonical-from')
    if args.catchlight:
        bad.append('--catchlight')
    if args.grain > 0:
        bad.append('--grain > 0')
    if args.gain_model != 'local':
        bad.append('--gain-model ramp')
    if args.lid_shadow > 0:
        bad.append('--lid-shadow')
    if args.lid_line != 'parabola':
        bad.append("--lid-line ceiling")
    if args.soft_lids:
        bad.append('--soft-lids')
    if args.shading_lid:
        bad.append('--shading-lid')
    if args.region_heal and not args.region_rows:
        bad.append('--region-heal without --region-rows')
    if args.top_shading > 0 and args.shading_sides != 'all':
        bad.append("--shading-sides top")
    if not args.freeze_registration:
        bad.append('--no-freeze-registration')
    if args.atlas_cap > 0:
        bad.append('--atlas-cap')
    if args.lash_exempt:
        bad.append('lash exemptions (pass --no-lash-exempt)')
    if not args.lash_guard:
        bad.append('--no-lash-guard')
    if not args.lid_temporal:
        bad.append('--lid-temporal is required')
    if args.keep_rim:
        bad.append('--keep-rim')
    if args.lid_extend != 0:
        bad.append('--lid-extend')
    if not args.limbus_track or not args.eyelid_track:
        bad.append('a limbus track and an eyelid track are required')
    if bad:
        raise SystemExit('eye_removal_gpu.py does not implement: %s. Run the CPU reference, '
                         'eye_soft_removal.py, for these.' % '; '.join(bad))


# =========================================================================== I/O
def read_exr(path):
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None or raw.dtype != np.float32 or raw.ndim != 3:
        raise ValueError('Expected scene-linear keyed RGB EXR at %s' % path)
    return np.ascontiguousarray(raw[..., ::-1]), digest


class Writer:
    """PNG encoding on worker threads; encoding is compression, not image maths."""

    def __init__(self, workers):
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.futures = []

    def write(self, path, image, params=None):
        def job():
            ok = cv2.imwrite(str(path), image, params or [])
            if not ok:
                raise IOError('could not write %s' % path)
        self.futures.append(self.pool.submit(job))
        if len(self.futures) > 256:
            self.drain(keep=64)

    def drain(self, keep=0):
        while len(self.futures) > keep:
            self.futures.pop(0).result()

    def close(self):
        self.drain()
        self.pool.shutdown()


# =========================================================================== geometry
class Eye:
    """One eye in one frame: the scalar geometry of r32.eye_frame, before any pixel."""
    __slots__ = ('i', 'side', 'limbus', 'ax', 'ay', 'x0', 'y0', 'h', 'w', 'lid',
                 'lid_inset', 'poly', 'x_range', 'group', 'slot')

    @property
    def window(self):
        return (slice(self.y0, self.y0 + self.h), slice(self.x0, self.x0 + self.w))


def scalar_eye(i, side, override, lid, lid_inset, poly, shape, margin=6):
    """The window and anchor of `r32.eye_frame`, or None where it abstains on geometry."""
    limbus, ax, ay = override
    limbus = np.asarray(limbus, float)
    if not np.isfinite([ax, ay]).all() or min(ax, ay) < 4:
        return None
    height, width = shape
    reach = max(ax, ay) * r32.RHO_MAX + r32.CENTRE_SEARCH + margin
    x0 = int(np.floor(limbus[0] - reach)); x1 = int(np.ceil(limbus[0] + reach)) + 1
    y0 = int(np.floor(limbus[1] - reach)); y1 = int(np.ceil(limbus[1] + reach)) + 1
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        return None
    e = Eye()
    e.i, e.side, e.limbus, e.ax, e.ay = i, side, limbus, ax, ay
    e.x0, e.y0, e.h, e.w = x0, y0, y1 - y0, x1 - x0
    e.lid, e.lid_inset = lid, lid_inset
    e.poly = poly
    e.x_range = (float(poly[:, 0].min()), float(poly[:, 0].max()))
    return e


class Group:
    """All eyes of one side whose windows share a shape: one dense GPU batch."""

    def __init__(self, side, eyes):
        self.side = side
        self.eyes = eyes
        self.h, self.w = eyes[0].h, eyes[0].w
        for n, e in enumerate(eyes):
            e.slot = n
        B = len(eyes)
        self.B = B
        ys = torch.arange(self.h, device=DEV, dtype=F64)
        xs = torch.arange(self.w, device=DEV, dtype=F64)
        self.x0 = as_t(np.array([e.x0 for e in eyes], np.float64))
        self.y0 = as_t(np.array([e.y0 for e in eyes], np.float64))
        # xx, yy: the int64 mgrid of the CPU code, held as exact float64
        self.xx = (xs.view(1, 1, -1) + self.x0.view(B, 1, 1)).expand(B, self.h, self.w)
        self.yy = (ys.view(1, -1, 1) + self.y0.view(B, 1, 1)).expand(B, self.h, self.w)
        self.lx = as_t(np.array([e.limbus[0] for e in eyes], np.float64)).view(B, 1, 1)
        self.ly = as_t(np.array([e.limbus[1] for e in eyes], np.float64)).view(B, 1, 1)
        self.ax = as_t(np.array([e.ax for e in eyes], np.float64)).view(B, 1, 1)
        self.ay = as_t(np.array([e.ay for e in eyes], np.float64)).view(B, 1, 1)
        self.rows = torch.arange(self.h, device=DEV).view(1, -1, 1).expand(B, self.h, self.w)
        self.cols = torch.arange(self.w, device=DEV).view(1, 1, -1).expand(B, self.h, self.w)
        self.patch = None

    def subset(self, keep):
        """A new group of the eyes where `keep` (bool list) is true, tensors sliced."""
        idx = [n for n, k in enumerate(keep) if k]
        g = Group.__new__(Group)
        g.side, g.h, g.w = self.side, self.h, self.w
        g.eyes = [self.eyes[n] for n in idx]
        for n, e in enumerate(g.eyes):
            e.slot = n
        g.B = len(idx)
        sel = torch.as_tensor(idx, device=DEV, dtype=torch.int64)
        for name in ('x0', 'y0'):
            setattr(g, name, getattr(self, name)[sel])
        for name, val in self.__dict__.items():
            if name in ('side', 'h', 'w', 'eyes', 'B', 'x0', 'y0'):
                continue
            if isinstance(val, torch.Tensor) and val.dim() >= 1 and val.shape[0] == self.B:
                setattr(g, name, val[sel])
            elif isinstance(val, list) and len(val) == self.B:
                setattr(g, name, [val[n] for n in idx])
            else:
                setattr(g, name, val)
        return g


def lid_distance(group, poly_scale=4):
    """r32.lid_distance for the group: polygon rasterised at 4x by cv2.fillPoly (drawing),
    exact Euclidean distance and the INTER_AREA 4x reduction on the GPU."""
    h, w = group.h, group.w
    fine = np.zeros((group.B, h * poly_scale, w * poly_scale), np.uint8)
    for n, e in enumerate(group.eyes):
        polygon = e.poly - [e.x0, e.y0]
        cv2.fillPoly(fine[n], [np.rint((polygon + .5) * poly_scale - .5).astype(np.int32)], 1)
    dist = G.edt_precise(as_t(fine.astype(bool)))
    dist = dist / G.dev_scalar(float(poly_scale))
    return G.resize_area_int(dist, poly_scale)


def filled_poly(group):
    """The 1x, shift-2 raster `signed_lid_distance` uses for its outside half."""
    h, w = group.h, group.w
    filled = np.zeros((group.B, h, w), np.uint8)
    for n, e in enumerate(group.eyes):
        cv2.fillPoly(filled[n], [np.rint((e.poly - [e.x0, e.y0]) * 4).astype(np.int32)],
                     1, cv2.LINE_8, 2)
    return as_t(filled.astype(bool))


def parabola_opening(group, keys, inset, feather=0.8):
    """r32.parabola_opening for every eye of the group whose lid entry exists.

    `keys` lists, per eye, the lid dict or None (then the opening is all ones and unused).
    `inset` is a per-eye list of python floats. float64 throughout, as numpy promotes it.
    """
    B = group.B
    ups, los, cen, has_up, has_lo = [], [], [], [], []
    xr0, xr1, ins = [], [], []
    for e, lid, k in zip(group.eyes, keys, inset):
        up = (lid or {}).get('upper')
        lo = (lid or {}).get('lower')
        has_up.append(up is not None)
        has_lo.append(lo is not None)
        ups.append(up if up is not None else (0.0, 0.0, 0.0))
        los.append(lo if lo is not None else (0.0, 0.0, 0.0))
        cen.append((lid or {}).get('centre', (0.0, 0.0))[0])
        xr0.append(e.x_range[0]); xr1.append(e.x_range[1]); ins.append(float(k))
    up = as_t(np.array(ups, np.float64)).view(B, 3, 1, 1)
    lo = as_t(np.array(los, np.float64)).view(B, 3, 1, 1)
    c0 = as_t(np.array(cen, np.float64)).view(B, 1, 1)
    has_up = as_t(np.array(has_up)).view(B, 1, 1)
    has_lo = as_t(np.array(has_lo)).view(B, 1, 1)
    inset_t = as_t(np.array(ins, np.float64)).view(B, 1, 1)
    xr0 = as_t(np.array(xr0, np.float64)).view(B, 1, 1)
    xr1 = as_t(np.array(xr1, np.float64)).view(B, 1, 1)
    u = group.xx - c0
    fe = max(feather, 1e-6)
    m = torch.ones((B, group.h, group.w), dtype=F64, device=DEV)
    for coef, sign, have in ((up, 1.0, has_up), (lo, -1.0, has_lo)):
        edge = coef[:, 0] * u ** 2 + coef[:, 1] * u + coef[:, 2] + sign * inset_t
        val = torch.clamp(div(sign * (group.yy - edge), fe) + 0.5, 0.0, 1.0)
        m = torch.where(have, torch.minimum(m, val), m)
    inside_x = torch.clamp(div(group.xx - (xr0 + inset_t), fe) + 0.5, 0.0, 1.0)
    inside_x = torch.minimum(inside_x, torch.clamp(div(xr1 - inset_t - group.xx, fe) + 0.5,
                                                   0.0, 1.0))
    return (m * inside_x).to(F32)


def eye_frame(group):
    """Per-pixel half of r32.eye_frame: `visible`, `opening`, and its < 90 px gate."""
    lid = lid_distance(group)
    visible = lid > 0.85
    lids = [e.lid for e in group.eyes]
    has = torch.as_tensor([l is not None for l in lids], device=DEV).view(-1, 1, 1)
    opening = parabola_opening(group, lids, [e.lid_inset for e in group.eyes])
    visible = torch.where(has, opening > 0.5, visible)
    rho_l = torch.hypot(div(group.xx - group.lx, group.ax), div(group.yy - group.ly, group.ay))
    vis_px = (visible & (rho_l < r32.RHO_MAX)).sum((1, 2))
    group.visible = visible
    group.opening = opening
    group.has_lid = has.view(-1)
    group.lid_dist = lid
    return (vis_px >= 90).tolist()


def solve_eye(group):
    """r32.solve_eye on the limbus (offset 0): rho, theta, inside; < 60 px abstains."""
    px = div(group.xx - group.lx, group.ax)
    py = div(group.yy - group.ly, group.ay)
    group.rho = torch.hypot(px, py)
    group.theta = torch.atan2(py, px)
    group.inside = group.visible & (group.rho < r32.RHO_MAX)
    return (group.inside.sum((1, 2)) >= 60).tolist()


# =========================================================================== detect_anomaly
_EDGES = r32.EDGES
_CENTRES = (_EDGES[:-1] + _EDGES[1:]) / 2
_NODES = np.array([-np.pi + (k + 0.5) * 2 * np.pi / 16 for k in range(16)])
_NODE_BASIS = r32.angular_basis(_NODES)


def _fill_profile(table, good, centres):
    """np.interp(centres, centres[good], table[good]) per item: (B, K, C) float64."""
    B, K, C = table.shape
    k = torch.arange(K, device=DEV).view(1, K).expand(B, K)
    left = torch.where(good, k, torch.full_like(k, -1)).cummax(1).values
    right = torch.where(good, k, torch.full_like(k, K)).flip(1).cummin(1).values.flip(1)
    first = torch.where(good, k, torch.full_like(k, K)).min(1).values.view(B, 1).expand(B, K)
    last = torch.where(good, k, torch.full_like(k, -1)).max(1).values.view(B, 1).expand(B, K)
    lidx = left.clamp(0, K - 1)
    ridx = right.clamp(0, K - 1)
    c = as_t(centres)
    tl = torch.gather(table, 1, lidx[..., None].expand(B, K, C))
    tr = torch.gather(table, 1, ridx[..., None].expand(B, K, C))
    slope = (tr - tl) / (c[ridx] - c[lidx])[..., None]
    val = slope * (c.view(1, K, 1) - c[lidx][..., None]) + tl
    val = torch.where((left < 0)[..., None],
                      torch.gather(table, 1, first.clamp(0, K - 1)[..., None].expand(B, K, C)), val)
    val = torch.where((right >= K)[..., None],
                      torch.gather(table, 1, last.clamp(0, K - 1)[..., None].expand(B, K, C)), val)
    return torch.where(good[..., None], table, val)


def detect_anomaly(values, rho, theta, inside, iterations=4, sigmas=4.0, floor=0.012, grow=1,
                   reach=7.0):
    """r32.detect_anomaly, batched. Returns (anomaly (B,h,w) bool, ok (B,) bool).

    values float32 (B,h,w,3); rho/theta float64; inside bool. Same model (per-bin median
    radial profile, 16-bin median angular gain with its ridge solve), the same frozen first
    threshold, the same hysteresis growth near the seed.
    """
    B, h, w, _ = values.shape
    anomaly = torch.zeros_like(inside)
    alive = torch.ones(B, dtype=torch.bool, device=DEV)
    threshold = low = None
    flat_v = values.reshape(B, -1, 3)
    flat_rho = rho.reshape(B, -1)
    E = [float(x) for x in _EDGES]
    luma = values.max(-1).values
    ones3 = torch.ones((3, 3), dtype=torch.bool).numpy()
    rows = torch.arange(h, device=DEV).view(1, h, 1)
    cols = torch.arange(w, device=DEV).view(1, 1, w)
    theta_basis = torch.stack([torch.ones_like(theta), torch.cos(theta), torch.sin(theta),
                               torch.cos(2 * theta), torch.sin(2 * theta)], -1)
    for _ in range(iterations):
        usable = inside & ~anomaly
        alive &= usable.sum((1, 2)) >= 40
        fu = usable.reshape(B, -1)
        table = torch.full((B, 16, 3), float('nan'), dtype=F64, device=DEV)
        for k in range(16):
            band = fu & (flat_rho >= E[k]) & (flat_rho < E[k + 1])
            med, n = G.median(flat_v, band[..., None].expand(-1, -1, 3), dim=1)
            table[:, k] = torch.where((n >= 4), med.to(F64), torch.full_like(med, float('nan'), dtype=F64))
        good = torch.isfinite(table[:, :, 0])
        alive &= good.sum(1) >= 4
        profile = _fill_profile(torch.nan_to_num(table), good, _CENTRES)
        radial = torch.stack([G.interp(rho, as_t(_CENTRES), profile[:, :, c])
                              for c in range(3)], -1)
        # angular gain
        lp = radial.max(-1).values
        valid = usable & (lp > 1e-4)
        nvalid = valid.sum((1, 2))
        ratio = luma.to(F64) / torch.where(valid, lp, torch.ones_like(lp))
        index = (div(theta + np.pi, 2 * np.pi) * 16).to(torch.int64) % 16
        meds = torch.zeros((B, 16), dtype=F64, device=DEV)
        cnts = torch.zeros((B, 16), dtype=torch.int64, device=DEV)
        fr = ratio.reshape(B, -1)
        fidx = index.reshape(B, -1)
        fval = valid.reshape(B, -1)
        for k in range(16):
            blk = fval & (fidx == k)
            med, n = G.median(fr, blk, dim=1)
            meds[:, k] = med
            cnts[:, k] = n
        inc = cnts >= 3
        nnodes = inc.sum(1)
        wts = torch.where(inc, cnts.to(F64), torch.zeros_like(meds))
        A = as_t(_NODE_BASIS).view(1, 16, 5).expand(B, 16, 5)
        sw = torch.sqrt(wts)
        aw = A * sw[..., None]
        tot = wts.sum(1)
        pen = 0.05 * tot
        M = aw.transpose(1, 2) @ aw + pen.view(B, 1, 1) * torch.eye(5, dtype=F64, device=DEV)
        rhs = (aw.transpose(1, 2) @ (torch.where(inc, meds, torch.zeros_like(meds)) * sw)[..., None])[..., 0]
        rhs[:, 0] = rhs[:, 0] + pen
        use_fit = (nvalid >= 40) & (nnodes >= 5)
        M = torch.where(use_fit.view(B, 1, 1), M, torch.eye(5, dtype=F64, device=DEV).expand(B, 5, 5))
        sol = torch.linalg.solve(M, rhs)
        gain = torch.clamp((theta_basis @ sol.view(B, 1, 5, 1))[..., 0], 0.35, 2.5)
        gain = torch.where(use_fit.view(B, 1, 1), gain, torch.ones_like(gain))
        base = radial * gain[..., None]
        signed = (values.to(F64) - base).max(-1).values
        if threshold is None:
            sm = inside.reshape(B, -1)
            ss = signed.reshape(B, -1)
            med0, _ = G.median(ss, sm, dim=1)
            spread = G.median((ss - med0[:, None]).abs(), sm, dim=1)[0] * 1.4826
            threshold = torch.clamp_min(sigmas * spread, floor)
            low = torch.clamp_min(2.5 * spread, floor / 2)
        peak = torch.where(inside, signed, torch.full_like(signed, -float('inf')))
        pmax = peak.reshape(B, -1).max(1)
        alive &= torch.isfinite(pmax.values) & (pmax.values > threshold)
        seed = pmax.indices
        sr, sc = seed // w, seed % w
        near = torch.hypot((cols - sc.view(B, 1, 1)).to(F64),
                           (rows - sr.view(B, 1, 1)).to(F64)) <= reach
        strong = inside & near & (signed > threshold.view(B, 1, 1))
        alive &= strong.any((1, 2))
        core = strong
        done = torch.zeros(B, dtype=torch.bool, device=DEV)
        for _g in range(2):
            grown = G.dilate(core, ones3) & inside & near & (signed > low.view(B, 1, 1))
            same = grown.sum((1, 2)) == core.sum((1, 2))
            core = torch.where((done | same).view(B, 1, 1), core, grown)
            done |= same
        anomaly = torch.where(alive.view(B, 1, 1), core, anomaly)
    if grow:
        k = np.ones((2 * grow + 1, 2 * grow + 1), bool)
        anomaly = G.dilate(anomaly, k) & inside
    return anomaly & alive.view(B, 1, 1), alive


# =========================================================================== atlas
def sample_eye(group, valid, u, v):
    """atl.sample_eye: the window resampled onto the canonical grid. The map is built the
    way numpy builds it -- `u * ax` in float32 (ax a Python float), then the float64 limbus
    added -- so the float32 map is the same map."""
    B = group.B
    mx = ((u[None] * group.ax.to(F32)).to(F64) + group.lx - group.x0.view(B, 1, 1)).to(F32)
    my = ((v[None] * group.ay.to(F32)).to(F64) + group.ly - group.y0.view(B, 1, 1)).to(F32)
    rgb = G.remap_linear(group.patch, mx, my)
    w = G.remap_linear(valid.to(F32), mx, my)
    return rgb, w


def _atlas_maps(u, v, s, tu, tv):
    """((u * s + tu) / ATLAS_STEP + (N - 1) / 2) in float32 for candidate arrays s, tu, tv (K,)."""
    s = as_t(np.float32(s)).view(-1, 1, 1)
    tu = as_t(np.float32(tu)).view(-1, 1, 1)
    tv = as_t(np.float32(tv)).view(-1, 1, 1)
    step = G.dev_scalar(atl.ATLAS_STEP)
    half = float(np.float32((atl.ATLAS_N - 1) / 2))
    mx = (u[None] * s + tu) / step + half
    my = (v[None] * s + tv) / step + half
    return mx, my


def register_frames(luma, wgt, ref_luma, ref_w, core, shift, step, scales, centres,
                    chunk_elems=48_000_000):
    """`atl.register` for every frame at once: the same candidate list per frame (same
    float32 offsets, same bound test), NCC on log radiance over cells valid in both, scored
    in float64, first maximum in the CPU's loop order. Returns per frame (ncc, du, dv, s)
    or None."""
    N = luma.shape[0]
    offs = np.arange(-shift, shift + 1e-9, step, dtype=np.float32)
    ref = np.log(np.maximum(ref_luma, 1e-4))
    refmask = as_t((ref_w > 0.98) & core)
    ref_t = as_t(ref.astype(np.float64))
    src = torch.stack([luma, wgt], -1)                         # (N, H, W, 2)
    out = [None] * N
    # candidate lists: shared when centre and scales are (per construction) shared
    cand = []
    for f in range(N):
        cx, cy = centres[f]
        lst = []
        for s in scales[f]:
            for du in offs:
                for dv in offs:
                    tu, tv = cx + float(du), cy + float(dv)
                    ok = not (max(abs(tu), abs(tv)) > atl.SHIFT_BOUND + 1e-9)
                    lst.append((float(s), tu, tv, ok))
        cand.append(lst)
    K = len(cand[0])
    assert all(len(c) == K for c in cand)
    per = max(1, int(chunk_elems // (K * atl.ATLAS_N * atl.ATLAS_N)))
    # The canonical grid is a meshgrid of ONE float32 axis, so every candidate's map is
    # separable -- column-only in x, row-only in y -- and remap_separable samples it
    # without an index per sample.
    tax = as_t(np.linspace(-atl.ATLAS_REACH, atl.ATLAS_REACH, atl.ATLAS_N).astype(np.float32))
    u_chk, _ = atl.atlas_grid()
    assert np.array_equal(u_chk[0], tax.cpu().numpy())
    step_t = G.dev_scalar(atl.ATLAS_STEP)
    half = float(np.float32((atl.ATLAS_N - 1) / 2))
    y2 = ref_t * ref_t
    for s0 in range(0, N, per):
        idx = list(range(s0, min(N, s0 + per)))
        Bc = len(idx)
        S = as_t(np.float32([[c[0] for c in cand[f]] for f in idx])).view(Bc, K, 1)
        TU = as_t(np.float32([[c[1] for c in cand[f]] for f in idx])).view(Bc, K, 1)
        TV = as_t(np.float32([[c[2] for c in cand[f]] for f in idx])).view(Bc, K, 1)
        OK = as_t(np.array([[c[3] for c in cand[f]] for f in idx]))
        mx = (tax.view(1, 1, -1) * S + TU) / step_t + half
        my = (tax.view(1, 1, -1) * S + TV) / step_t + half
        smp = G.remap_separable(src[idx], mx, my)
        a, aw = smp[..., 0], smp[..., 1]
        m = (aw > 0.98) & refmask
        n = m.sum((2, 3))
        mf = m.to(F64)
        x = torch.log(torch.clamp_min(a, 1e-4).to(F64)) * mf
        sx = x.sum((2, 3))
        sxx = (x * x).sum((2, 3))
        sy = (mf * ref_t).sum((2, 3))
        syy = (mf * y2).sum((2, 3))
        sxy = (x * ref_t).sum((2, 3))
        nf = n.to(F64).clamp_min(1)
        vxx = sxx - sx * sx / nf
        vyy = syy - sy * sy / nf
        vxy = sxy - sx * sy / nf
        d = torch.sqrt(torch.clamp_min(vxx, 0) * torch.clamp_min(vyy, 0))
        c = vxy / torch.where(d > 0, d, torch.ones_like(d))
        valid = OK & (n >= 400) & (d > 0)
        c = torch.where(valid, c, torch.full_like(c, -float('inf')))
        best = c.max(1)
        bv = best.values.cpu().numpy()
        bi = best.indices.cpu().numpy()
        for r, f in enumerate(idx):
            if np.isfinite(bv[r]):
                s_, tu_, tv_, _ = cand[f][bi[r]]
                out[f] = (float(bv[r]), tu_, tv_, s_)
    return out


def robust_gain(rgb_w, atlas, mask, basis, floor=1e-4, iterations=4):
    """atl.robust_gain for every frame: per-channel log-domain [1, u, v] field, IRLS.

    Weighted normal equations stand in for numpy's SVD lstsq (same minimiser, float64);
    medians are numpy's. Returns (N, 3, 3) float64 and a per-frame ok flag."""
    N = rgb_w.shape[0]
    msk = mask.reshape(N, -1)
    X = basis.reshape(-1, 3)                                    # float64
    coef = torch.zeros((N, 3, 3), dtype=F64, device=DEV)
    ok_frame = msk.sum(1) >= 60
    for c in range(3):
        a = rgb_w[..., c].reshape(N, -1)
        b = atlas[..., c].reshape(1, -1).expand(N, -1)
        ok = msk & (b > floor) & (a > floor)
        ok_frame &= ok.sum(1) >= 60
        la = torch.log(a.clamp_min(floor).to(F64)).to(F32)
        if atlas.dtype == F32:
            lb = torch.log(b.clamp_min(floor).to(F64)).to(F32)
            y = (la - lb).to(F64)
        else:
            y = la.to(F64) - torch.log(b.clamp_min(floor))
        y = torch.where(ok, y, torch.zeros_like(y))
        w = ok.to(F64)
        sol = torch.zeros((N, 3), dtype=F64, device=DEV)
        for _ in range(iterations):
            Xw = X[None] * w[..., None]
            M = Xw.transpose(1, 2) @ X[None].expand(N, -1, -1)
            rhs = (Xw * y[..., None]).sum(1)
            Mi = torch.where(ok_frame.view(N, 1, 1), M, torch.eye(3, dtype=F64, device=DEV).expand(N, 3, 3))
            sol = torch.linalg.solve(Mi, rhs)
            r = y - (X[None] @ sol[..., None])[..., 0]
            med, _ = G.median(r, ok, dim=1)
            mad, _ = G.median((r - med[:, None]).abs(), ok, dim=1)
            s = 1.4826 * mad + 1e-6
            w = torch.where(ok, 1.0 / (1.0 + (r / (2.0 * s[:, None])) ** 2), torch.zeros_like(r))
        coef[:, c] = sol
    return coef, ok_frame


def apply_gain(coef, basis):
    """atl.apply_gain: exp(clip(basis . coef, +/-2.5)), (N, H, W, 3) float64."""
    lin = (basis[None, ..., 0:1] * coef[:, None, None, :, 0]
           + basis[None, ..., 1:2] * coef[:, None, None, :, 1]) \
        + basis[None, ..., 2:3] * coef[:, None, None, :, 2]
    return torch.exp(torch.clamp(lin, -2.5, 2.5))


def sharpness(atlas, mask):
    """atl.sharpness (reported only)."""
    luma = np.log(np.maximum(atlas.max(-1), 1e-4)).astype(np.float32)
    gy, gx = np.gradient(luma, atl.ATLAS_STEP)
    inner = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    if not inner.any():
        return None
    return float(np.hypot(gx, gy)[inner].mean())


def build_atlas(rgb, wts, iterations=5, gain_window=5):
    """atl.build on the GPU: register, refit per-frame gain, rebuild the median atlas.

    rgb (N,121,121,3) f32, wts (N,121,121) f32, in FRAME ORDER (the gain series is smoothed
    over list order). Returns the same quantities `eye_soft_removal.py` uses, per frame as
    arrays."""
    N = rgb.shape[0]
    u, v = atl.atlas_grid()
    rho = np.hypot(u, v)
    core = rho < r32.RHO_MAX
    basis = as_t(atl.gain_basis(u, v))
    ws = (wts > 0.98).to(F32)
    atlas = ((rgb * ws[..., None]).sum(0) / torch.clamp_min(ws.sum(0), 1e-6)[..., None])
    aw = ws.sum(0) >= atl.MIN_CLEAN_OBSERVATIONS
    atlas_np = atlas.cpu().numpy()
    aw_np = aw.cpu().numpy()
    seed_sharp = sharpness(atlas_np, aw_np & core)
    du = np.zeros(N); dv = np.zeros(N); sc = np.ones(N); ncc = np.zeros(N)
    registered = np.zeros(N, bool)
    luma = rgb.max(-1).values
    core_t = as_t(core)
    history = []
    for it in range(iterations):
        ref_luma = atlas_np.max(-1).astype(np.float32)
        ref_w = aw_np.astype(np.float32)
        first = it == 0
        shift, step = ((atl.COARSE_SHIFT, atl.COARSE_STEP) if first
                       else (atl.FINE_SHIFT, atl.FINE_STEP))
        scales = [atl.SCALES if it <= 1 else (sc[f],) for f in range(N)]
        centres = [(0.0, 0.0) if first else (du[f], dv[f]) for f in range(N)]
        best = register_frames(luma, wts, ref_luma, ref_w, core, shift, step, scales, centres)
        unregistered = 0
        for f, b in enumerate(best):
            if b is None:
                registered[f] = False
                unregistered += 1
                continue
            ncc[f], du[f], dv[f], sc[f] = b[0], b[1], b[2], b[3]
            registered[f] = True
        if registered.any():
            mu = np.median(np.stack([du[registered], dv[registered]], 1), axis=0)
            du = np.array([float(np.clip(x - mu[0], -atl.SHIFT_BOUND, atl.SHIFT_BOUND)) for x in du])
            dv = np.array([float(np.clip(x - mu[1], -atl.SHIFT_BOUND, atl.SHIFT_BOUND)) for x in dv])
        mx, my = _atlas_maps(as_t(u), as_t(v), sc, du, dv)
        rgb_w = G.remap_linear(rgb, mx, my)
        w_w = G.remap_linear(wts, mx, my)
        valid = w_w > 0.98
        atlas_t = as_t(atlas_np)
        mask = valid & as_t(aw_np)[None] & core_t[None]
        g, okg = robust_gain(rgb_w, atlas_t, mask, basis)
        raw = [np.zeros((3, 3)) if not okg[f] else gg for f, gg in
               enumerate(g.cpu().numpy())]
        coefs = atl.smooth_series(raw, gain_window)
        gain = apply_gain(as_t(coefs), basis)
        stack = rgb_w.to(F64) / torch.clamp_min(gain, 1e-3)
        masked = torch.where(valid[..., None], stack, torch.full_like(stack, float('nan')))
        atlas_new = torch.empty((atl.ATLAS_N, atl.ATLAS_N, 3), dtype=F64, device=DEV)
        for r0 in range(0, atl.ATLAS_N, 16):
            atlas_new[r0:r0 + 16] = G.nanmedian0(masked[:, r0:r0 + 16])
        atlas_np = torch.nan_to_num(atlas_new).cpu().numpy()
        aw_np = (valid.to(F32).sum(0) >= atl.MIN_CLEAN_OBSERVATIONS).cpu().numpy()
        nccs = [ncc[f] for f in range(N) if registered[f]] or [float('nan')]
        history.append(dict(
            iteration=it, unregistered_frames=unregistered,
            mean_ncc=round(float(np.mean(nccs)), 5), min_ncc=round(float(np.min(nccs)), 5),
            atlas_cells=int((aw_np & core).sum()),
            sharpness=round(sharpness(atlas_np, aw_np & core), 5),
            shift_rms_norm=round(float(np.sqrt(np.mean(du ** 2 + dv ** 2))), 5)))
        log('R33_ATLAS iter', history[-1])
    return dict(atlas=atlas_np, aw=aw_np, history=history, rho=rho, seed_sharpness=seed_sharp,
                du=du, dv=dv, scale=sc, ncc=ncc, registered=registered, gain_coef=coefs,
                gain=gain, rgb_w=rgb_w, valid=valid)


def predict(group, atlas, aw, du, dv, scale, coef=None):
    """eye_atlas_repair.predict for the group; du/dv/scale are per-eye Python floats (the
    frozen registration), `coef` per-eye (3, 3) or None for zeros. base is float64 as in
    numpy (float32 atlas times a float64 gain of ones)."""
    B = group.B
    p = div(group.xx - group.lx, group.ax).to(F32)
    q = div(group.yy - group.ly, group.ay).to(F32)
    du_t = as_t(np.float32(du)).view(B, 1, 1)
    dv_t = as_t(np.float32(dv)).view(B, 1, 1)
    sc_t = as_t(np.float32(scale)).view(B, 1, 1)
    step = G.dev_scalar(atl.ATLAS_STEP)
    half = float(np.float32((atl.ATLAS_N - 1) / 2))
    ai = ((p - du_t) / sc_t) / step + half
    aj = ((q - dv_t) / sc_t) / step + half
    at = as_t(atlas).to(F32) if atlas.dtype == np.float32 else as_t(atlas)
    if at.dtype != F32:
        raise ValueError('the painted canonical is float32 on this path')
    base = G.remap_linear(at[None].expand(B, -1, -1, -1).contiguous(), ai, aj)
    support = G.remap_linear(as_t(aw.astype(np.float32))[None].expand(B, -1, -1).contiguous(),
                             ai, aj) > 0.98
    if coef is None:
        return torch.clamp_min(base.to(F64), 0.0), support
    basis = torch.stack([torch.ones_like(p), p, q], -1).to(F64)
    gain = torch.exp(torch.clamp(
        (basis[..., 0:1] * coef[:, None, None, :, 0] + basis[..., 1:2] * coef[:, None, None, :, 1])
        + basis[..., 2:3] * coef[:, None, None, :, 2], -2.5, 2.5))
    return torch.clamp_min(base.to(F64) * gain, 0.0), support


def local_gain(patch, base, ring, lo=0.5, hi=2.0):
    """esr.local_gain per eye: (B, 3) float32 and an ok flag."""
    B = patch.shape[0]
    ok_all = ring.sum((1, 2)) >= 40
    k = torch.zeros((B, 3), dtype=F32, device=DEV)
    for c in range(3):
        a = patch[..., c].reshape(B, -1)
        b = base[..., c].reshape(B, -1)
        r = ring.reshape(B, -1)
        ok = r & (a > 1e-4) & (b > 1e-4)
        ok_all &= ok.sum(1) >= 40
        med, _ = G.median(a.to(F64) / torch.where(ok, b, torch.ones_like(b)), ok, dim=1)
        k[:, c] = torch.clamp(med, lo, hi).to(F32)
    return k, ok_all


def residual_scale(signed, inside):
    """esr.residual_scale per eye (float64), None where fewer than 40 pixels."""
    B = signed.shape[0]
    s = signed.reshape(B, -1)
    m = inside.reshape(B, -1)
    med, n = G.median(s, m, dim=1)
    mad, _ = G.median((s - med[:, None]).abs(), m, dim=1)
    return mad * 1.4826, n >= 40


def signed_lid_positive(group):
    """`signed_lid_distance(...) > 0`, which is all `--lid-extend 0` reads: inside the 1x
    shift-2 raster AND a positive 4x distance."""
    return filled_poly(group) & (group.lid_dist > 0)


# =========================================================================== lash line
def display_lum(patch):
    """clip(linear_to_srgb(clip(patch, 0, 1)), 0, 1).max(-1), float32."""
    return torch.clamp(G.srgb(torch.clamp(patch, 0, 1)), 0, 1).max(-1).values


def lash_ceiling_raw(group, lum, lash_run=3, pupil_rho=0.42, spec_level=0.45,
                     spec_reach=1.05, ring_px=3.0, ring_level=0.10):
    """esr.lash_ceiling(..., raw=True) per eye: (B, w) float32 rows, inf where a column
    reported nothing; and whether any column did.

    Walks each column down from the lid skin to the first run that reaches lash level --
    vectorised over columns and eyes: the searchable rows of a column are compacted in
    order, and "the first row n whose value is under `entry` and whose 9-row run reaches
    `floor_v`" is found for every n at once.
    """
    B, h, w = lum.shape
    rho = group.rho
    spec = (lum > spec_level) & (rho < spec_reach)
    k = int(2 * np.ceil(ring_px) + 1)
    near = G.dilate(spec, G.ellipse_kernel(k))
    blocked = spec | (near & (lum < ring_level)) | (rho < pupil_rho)
    floor_row = []
    for e in group.eyes:
        centre_row = float(e.limbus[1] - e.y0)
        floor_row.append(int(np.ceil(centre_row + 0.45 * e.ay)))
    band = group.rows <= as_t(np.array(floor_row)).view(B, 1, 1)
    searchable = band & ~blocked
    p = div(group.xx[:, 0, :] - group.lx[:, :, 0], group.ax[:, :, 0])      # (B, w)
    colok = p.abs() <= 1.0
    key = torch.where(searchable, group.rows, group.rows + h)
    order = key.sort(dim=1).indices                                        # (B, h, w)
    cnt = searchable.sum(1)                                                # (B, w)
    lumc = torch.gather(lum, 1, order)
    nidx = torch.arange(h, device=DEV).view(1, h, 1)
    inrange = nidx < cnt[:, None, :]
    inf = torch.tensor(float('inf'), dtype=F32, device=DEV)
    lumc = torch.where(inrange, lumc, inf)
    colok &= cnt >= 6
    first3 = lumc[:, :3].sort(dim=1).values[:, 1]                          # median of 3
    skin = first3.to(F64)
    colok &= skin > 0.15
    entry = (skin * 0.60)
    floor_v = (skin * 0.45)
    pad = torch.cat([lumc, inf.expand(B, 8, w)], 1)
    win = pad.unfold(1, 9, 1)                                              # (B, h, w, 9)
    runmin = win.min(-1).values
    c1 = lumc < entry.to(F32)[:, None, :]
    c2 = runmin.to(F64) <= floor_v[:, None, :]
    hit = c1 & c2 & inrange
    anyhit = hit.any(1)
    nstar = torch.where(hit, nidx.expand(B, h, w), torch.full_like(order, h)).min(1).values
    nstar_c = nstar.clamp(max=h - 1)
    vals = torch.gather(win, 1, nstar_c.view(B, 1, w, 1).expand(B, 1, w, 9))[:, 0]   # (B, w, 9)
    runlen = (cnt - nstar_c).clamp(max=9)
    j = vals.argmin(-1)
    kk = j.clone()
    fv = floor_v
    for _ in range(lash_run):
        nxt = (kk + 1).clamp(max=8)
        v_next = torch.gather(vals, 2, nxt[..., None])[..., 0].to(F64)
        step_ok = ((kk + 1) < runlen) & ((kk - j) < lash_run) & (v_next < fv)
        kk = torch.where(step_ok, kk + 1, kk)
    got_n = (nstar_c + kk).clamp(max=h - 1)
    got = torch.gather(order, 1, got_n[:, None, :])[:, 0]
    found = colok & anyhit
    ceiling = torch.where(found, got.to(F32), inf)
    return ceiling, found.any(1)


def lid_line(raw, excl, group, curvature, min_span=0.6, tukey=2.0, iterations=6):
    """esr.lid_line per eye: (B, w) float32 and ok. `curvature` (B,) float64."""
    B, w = raw.shape
    max_row = as_t(np.float32([float(e.limbus[1] - e.y0) for e in group.eyes])).view(B, 1)
    ok = torch.isfinite(raw) & (raw <= max_row)
    if excl is not None:
        ok &= ~excl
    good = ok.sum(1) >= 5
    x_native = torch.arange(w, device=DEV, dtype=F64).view(1, w) + group.x0.view(B, 1)
    l0 = as_t(np.array([float(e.limbus[0]) for e in group.eyes])).view(B, 1)
    u_all = x_native - l0
    big = torch.tensor(1e300, dtype=F64, device=DEV)
    umax = torch.where(ok, u_all, -big).max(1).values
    umin = torch.where(ok, u_all, big).min(1).values
    ax = group.ax.view(B)
    good &= ~((umax - umin) < min_span * ax)
    y = torch.where(ok, raw.to(F64), torch.zeros_like(u_all))
    cv = curvature.view(B, 1)
    target = y - cv * u_all ** 2
    wgt = ok.to(F64)
    beta = torch.zeros((B, 2), dtype=F64, device=DEV)
    for _ in range(iterations):
        s_uu = (wgt * u_all * u_all).sum(1)
        s_u = (wgt * u_all).sum(1)
        s_1 = wgt.sum(1)
        r_u = (wgt * u_all * target).sum(1)
        r_1 = (wgt * target).sum(1)
        M = torch.stack([torch.stack([s_uu, s_u], -1), torch.stack([s_u, s_1], -1)], -2)
        det = s_uu * s_1 - s_u * s_u
        # numpy raises on an exactly zero LU pivot, which a single weighted sample gives;
        # the GPU determinant of that rank-1 matrix is tiny but not zero (track_gpu.py).
        good &= (det != 0) & ((wgt > 0).sum(1) >= 2)
        Mi = torch.where(good.view(B, 1, 1), M, torch.eye(2, dtype=F64, device=DEV).expand(B, 2, 2))
        beta = torch.linalg.solve(Mi, torch.stack([r_u, r_1], -1))
        resid = target - (beta[:, :1] * u_all + beta[:, 1:])
        med, _ = G.median(resid, ok, dim=1)
        mad, _ = G.median((resid - med[:, None]).abs(), ok, dim=1)
        scale = torch.clamp_min(1.4826 * mad, 0.5)
        t = torch.clamp(resid / (tukey * scale[:, None]), -1, 1)
        wgt = torch.where(ok, (1 - t ** 2) ** 2, torch.zeros_like(t))
        good &= wgt.sum(1) >= 3
    line = cv * u_all ** 2 + beta[:, :1] * u_all + beta[:, 1:]
    return line.to(F32), good


# =========================================================================== specular
_TIERS = ((0.45, 0.40, 0.18), (0.30, 0.40, 0.18), (0.20, 0.20, 0.10))


def corneal_specular(lum, rho, rho_seed=0.85, max_area=90):
    """esr.corneal_specular per eye: the one compact bright blob with a dark surround (and
    dark above), trying the three level tiers in order. (B, h, w) bool and found (B,)."""
    B, h, w = lum.shape
    core = torch.zeros((B, h, w), dtype=torch.bool, device=DEV)
    found = torch.zeros(B, dtype=torch.bool, device=DEV)
    kern5 = G.ellipse_kernel(5)
    rows = torch.arange(h, device=DEV).view(1, h, 1)
    for level, ring_max, contrast in _TIERS:
        todo = ~found
        if not todo.any():
            break
        hot = (lum > level) & (rho < 1.0) & todo.view(B, 1, 1)
        cand = hot & (rho < rho_seed)
        labels = G.connected_components(hot)
        key = torch.where(cand, labels, torch.zeros_like(labels))
        items = torch.arange(B, device=DEV).view(B, 1, 1).expand(B, h, w)
        pairs = torch.unique(torch.stack([items[cand], key[cand]], 1), dim=0)
        if pairs.numel() == 0:
            continue
        P = pairs.shape[0]
        pi, pl = pairs[:, 0], pairs[:, 1]
        comp = labels[pi] == pl.view(P, 1, 1)
        area = comp.sum((1, 2))
        ring = G.dilate(comp, kern5) & ~comp
        lp = lum[pi]
        ring_med, nring = G.median(lp.reshape(P, -1), ring.reshape(P, -1), dim=1)
        peak = torch.where(comp, lp, torch.full_like(lp, -float('inf'))).reshape(P, -1).max(1).values
        rm = ring_med.to(F64)
        pk = peak.to(F64)
        ok = (area <= max_area) & (nring >= 6) & ~(rm > ring_max) & ~(pk - rm < contrast)
        top_row = torch.where(comp, rows.expand(P, h, w), torch.full_like(comp, h, dtype=torch.int64)).reshape(P, -1).min(1).values
        above = ring & (rows <= top_row.view(P, 1, 1))
        amed, nab = G.median(lp.reshape(P, -1), above.reshape(P, -1), dim=1)
        ok &= ~((nab >= 2) & (amed.to(F64) > ring_max))
        # best per item: highest peak, first in label order on a tie
        score = torch.where(ok, pk, torch.full_like(pk, -float('inf')))
        order = torch.argsort(pl, stable=True)
        best_val = torch.full((B,), -float('inf'), dtype=F64, device=DEV)
        best_val.scatter_reduce_(0, pi, score, reduce='amax')
        is_best = ok & (score == best_val[pi])
        # first (lowest label) among ties
        lab_best = torch.full((B,), torch.iinfo(torch.int64).max, device=DEV, dtype=torch.int64)
        lab_best.scatter_reduce_(0, pi[is_best], pl[is_best], reduce='amin')
        chosen = is_best & (pl == lab_best[pi])
        got = torch.zeros((B,), dtype=torch.bool, device=DEV)
        got[pi[chosen]] = True
        got &= todo
        sel_items = pi[chosen]
        sel_labels = pl[chosen]
        if sel_items.numel():
            mask_new = torch.zeros((B, h, w), dtype=torch.bool, device=DEV)
            mask_new[sel_items] = labels[sel_items] == sel_labels.view(-1, 1, 1)
            core = torch.where(got.view(B, 1, 1), mask_new, core)
        found |= got
        del order
    return core, found


def specular_region(core, found, rho, reach_px=2.0, rho_max=1.02):
    """esr.specular_region: the blob and a disc of `reach_px` around it (float32)."""
    dist = G.distance_chamfer(~core, 5)
    w = torch.clamp((reach_px + 1.0) - dist, 0.0, 1.0)
    w = w * (rho < rho_max).to(F32)
    return torch.where(found.view(-1, 1, 1), w, torch.zeros_like(w))


# =========================================================================== weights
def quantize(w):
    """np.rint(np.clip(w, 0, 1) * 255).astype(float32) / 255 -- the written weight."""
    return torch.round(torch.clamp(w, 0, 1) * 255.0) / G.dev_scalar(255.0)


def soft_weight(signed, under, inside, support, sigma, k_lo, k_hi, reach, blur=0.7,
                rim_reach=4.0, rim_region=None):
    """esr.soft_weight per eye (rim on). Returns (w float32, ok, top float64, seed (B,2))."""
    B, h, w = signed.shape
    peak = torch.where(inside, signed, torch.full_like(signed, -float('inf')))
    pm = peak.reshape(B, -1).max(1)
    top = pm.values
    ok = torch.isfinite(top) & ~(top <= k_lo * sigma)
    seed = pm.indices
    sr, sc = seed // w, seed % w
    rows = torch.arange(h, device=DEV).view(1, h, 1)
    cols = torch.arange(w, device=DEV).view(1, 1, w)
    near = torch.hypot((cols - sc.view(B, 1, 1)).to(F64), (rows - sr.view(B, 1, 1)).to(F64)) <= reach
    span = max((k_hi - k_lo) * sigma, 1e-6)
    wt = torch.clamp(div(signed - k_lo * sigma, span), 0.0, 1.0).to(F32)
    wt = wt * (inside & near).to(F32)
    core = wt > 0.5
    k = int(2 * np.ceil(rim_reach) + 1)
    region = inside if rim_region is None else (inside & rim_region)
    collar = G.dilate(core, G.ellipse_kernel(k)) & region & near
    wn = torch.clamp(div(under - k_lo * sigma, span), 0.0, 1.0).to(F32) * collar.to(F32)
    wt = torch.maximum(wt, wn)
    if blur > 0:
        wt = torch.maximum(G.gaussian_blur(wt, blur), wt)
        wt = wt * (inside & near).to(F32)
    s = torch.clamp(G.distance_chamfer(support, 3) / G.dev_scalar(2.0), 0.0, 1.0)
    wt = wt * s
    wt = quantize(wt)
    return wt, ok, top, torch.stack([sr, sc], 1)


# =========================================================================== shading
_COEFF = torch.tensor([0.2126, 0.7152, 0.0722], dtype=F32)


def _lum_sum(x):
    c = _COEFF.to(DEV)
    if x.dtype == F64:
        c = c.to(F64)
    p = x * c
    return (p[..., 0] + p[..., 1]) + p[..., 2]


def _log32(x):
    """numpy float32 log, as the correctly rounded value."""
    return torch.log(x.to(F64)).to(F32)


def _exp32(x):
    return torch.exp(x.to(F64)).to(F32)


def top_shading(patch, base, w, rho, sigma=1.5, decay=2.5, lo=-1.6, hi=0.35, exclude=None):
    """esr.top_shading with sides='all': (field float32, boundary px, ok)."""
    fill = w >= 0.5
    below = G.dilate(fill, np.ones((3, 3), bool))
    top = (~fill) & below & (rho < 0.97)
    if exclude is not None:
        ex, has = exclude
        disp = display_lum(patch)
        top = torch.where(has.view(-1, 1, 1), top & ~ex & (disp <= 0.45), top)
    n = top.sum((1, 2))
    ok = n >= 3
    ls = torch.clamp_min(_lum_sum(patch), 1e-4)
    lb = torch.clamp_min(_lum_sum(base), 1e-4)
    d = torch.clamp(_log32(ls).to(F64) - torch.log(lb), lo, hi).to(F32)
    topf = top.to(F32)
    num = G.gaussian_blur(d * topf, sigma)
    den = G.gaussian_blur(topf, sigma)
    avg = torch.where(den > 1e-3, num / torch.clamp_min(den, 1e-6), torch.zeros_like(num))
    dist = G.distance_chamfer(~top, 5)
    m = avg * _exp32(-div(torch.clamp_min(dist - 1.0, 0.0), max(decay, 1e-3)))
    field = _exp32(m)
    return torch.where(ok.view(-1, 1, 1), field, torch.ones_like(field)), n, ok


def region_shading(patch, base, inside_fill, region, lo=-1.6, hi=0.35, iterations=400,
                   hot=None):
    """esr.region_shading with closed_top=True: the membrane, (field float32, ok)."""
    omega = region & ~inside_fill
    ls = torch.clamp_min(_lum_sum(patch), 1e-4)
    lb = torch.clamp_min(_lum_sum(base), 1e-4)
    d = torch.clamp(torch.log(ls) - torch.log(lb), lo, hi).to(F32) if patch.dtype == F64 else \
        torch.clamp(_log32(ls).to(F64) - torch.log(lb), lo, hi).to(F32)
    B, h, w = omega.shape
    m = torch.zeros((B, h, w), dtype=F32, device=DEV)
    ring = G.dilate(omega, np.ones((3, 3), bool)) & ~omega
    if hot is not None:
        alt = ring & ~hot
        ring = torch.where(alt.any((1, 2)).view(B, 1, 1), alt, ring)
    rc = ring.sum((1, 2))
    rmean = (torch.where(ring, d, torch.zeros_like(d)).sum((1, 2)) / rc.clamp_min(1).to(F32))
    m = torch.where(omega & (rc > 0).view(B, 1, 1), rmean.view(B, 1, 1).expand(B, h, w), m)
    fixed = ~omega
    nan = torch.tensor(float('nan'), dtype=F32, device=DEV)
    hotfixed = (hot & fixed) if hot is not None else None
    for _ in range(iterations):
        acc = torch.zeros((B, h, w), dtype=F32, device=DEV)
        cnt = torch.zeros((B, h, w), dtype=F32, device=DEV)
        nb_val = torch.where(fixed, d, m)
        for dy, dx in ((1, 0), (0, 1), (0, -1), (-1, 0)):
            src = G.shift2d(nb_val, dy, dx, float('nan'))
            ok = torch.isfinite(src)
            if hotfixed is not None:
                ok &= ~G.shift2d(hotfixed, dy, dx, False)
            acc = acc + torch.where(ok, src, torch.zeros_like(src))
            cnt = cnt + ok.to(F32)
        upd = torch.where(cnt > 0, acc / torch.clamp_min(cnt, 1), m)
        m = torch.where(omega, upd, m)
    field = torch.where(omega, _exp32(m), torch.ones_like(m))
    return field, omega.any((1, 2))


def region_rows(patch, base, inside_fill, region, lo=-1.6, hi=0.35, hot=None):
    """esr.region_rows: each run of region pixels in a row takes the interpolation of
    log(photograph / canonical) between two usable pixels at each end. (field, ok)."""
    omega = region & ~inside_fill
    B, h, w = omega.shape
    ok_any = omega.any((1, 2))
    ls = torch.clamp_min(_lum_sum(patch), 1e-4)
    lb = torch.clamp_min(_lum_sum(base), 1e-4)
    d = torch.clamp(torch.log(ls) - torch.log(lb), lo, hi).to(F32)
    usable = ~omega if hot is None else (~omega & ~hot)
    xs = torch.arange(w, device=DEV).view(1, 1, w).expand(B, h, w)
    # run start a and end b for every omega pixel
    a = torch.where(~omega, xs, torch.full_like(xs, -1)).cummax(2).values + 1
    b = torch.where(~omega, xs, torch.full_like(xs, w)).flip(2).cummin(2).values.flip(2) - 1

    def take(col):
        inb = (col >= 0) & (col < w)
        cc = col.clamp(0, w - 1)
        val = torch.gather(d, 2, cc)
        us = torch.gather(usable, 2, cc) & inb
        return val, us

    l1, u1 = take(a - 1)
    l2, u2 = take(a - 2)
    r1, v1 = take(b + 1)
    r2, v2 = take(b + 2)
    zero = torch.zeros_like(l1)
    lsum = torch.where(u1, l1, zero) + torch.where(u2, l2, zero)
    ln = u1.to(F32) + u2.to(F32)
    left = torch.where(ln == 2, lsum / G.dev_scalar(2.0), lsum)
    rsum = torch.where(v1, r1, zero) + torch.where(v2, r2, zero)
    rn = v1.to(F32) + v2.to(F32)
    right = torch.where(rn == 2, rsum / G.dev_scalar(2.0), rsum)
    hasl, hasr = ln > 0, rn > 0
    t = ((xs - (a - 1)).to(F64) / (b + 1 - (a - 1)).to(F64))
    both = ((1 - t) * left.to(F64) + t * right.to(F64)).to(F32)
    nan = torch.tensor(float('nan'), dtype=F32, device=DEV)
    m = torch.where(hasl & hasr, both, torch.where(hasl, left, torch.where(hasr, right, nan)))
    m = torch.where(omega, m, nan)
    need = (omega & torch.isnan(m)).any((1, 2))
    if need.any():
        idx = torch.nonzero(need)[:, 0]
        fb, fok = region_shading(patch[idx], base[idx], inside_fill[idx], region[idx], lo, hi,
                                 hot=None if hot is None else hot[idx])
        fill = _log32(torch.clamp_min(fb, 1e-4))
        mi = m[idx]
        mi = torch.where(torch.isnan(mi) & omega[idx] & fok.view(-1, 1, 1), fill, mi)
        m[idx] = mi
    m = torch.where(omega, torch.nan_to_num(m), torch.zeros_like(m))
    up = torch.roll(m, 1, 1); dn = torch.roll(m, -1, 1)
    up_ok = torch.roll(omega, 1, 1); dn_ok = torch.roll(omega, -1, 1)
    up = torch.where(up_ok, up, m); dn = torch.where(dn_ok, dn, m)
    med3 = torch.stack([up, m, dn]).sort(0).values[1]
    m = torch.where(omega, med3, m)
    field = torch.where(omega, _exp32(m), torch.ones_like(m))
    return field, ok_any


# =========================================================================== temporal
def filter_shading(store, half):
    """esr.filter_shading (symmetric, not lid-registered): the 5-frame median of each
    frame's shading field at the same place on her eye, painted frames voting only.

    `store[(i, side)] = dict(log=(h,w) f32 tensor, valid=(h,w) bool tensor, eye=Eye)`.
    Every (frame, neighbour) pair is one batched remap; neighbours of another window
    shape are padded to a common canvas -- edge-replicated for the BORDER_REPLICATE
    remap and zero for the BORDER_CONSTANT one, which leaves both remaps unchanged.
    """
    keys = sorted(store)
    if not keys:
        return {}
    Hm = max(store[k]['log'].shape[0] for k in keys)
    Wm = max(store[k]['log'].shape[1] for k in keys)
    pairs = []                       # (target key, source key)
    radius = {}
    for (i, side) in keys:
        r = 0
        for k in range(1, half + 1):
            if all((i + s * m, side) in store for m in range(1, k + 1) for s in (-1, 1)):
                r = k
            else:
                break
        radius[(i, side)] = r
        for j in range(i - r, i + r + 1):
            if (j, side) in store:
                pairs.append(((i, side), (j, side)))
    # canvases
    def canvas(key):
        rec = store[key]
        lg = rec['log']
        h, w = lg.shape
        lp = torch.nn.functional.pad(lg[None, None], (0, Wm - w, 0, Hm - h), mode='replicate')[0, 0]
        vp = torch.zeros((Hm, Wm), dtype=F32, device=DEV)
        vp[:h, :w] = rec['valid'].to(F32)
        return lp, vp
    canv = {k: canvas(k) for k in keys}
    out = {}
    stacks = {k: [] for k in keys}
    CH = 4096
    for s0 in range(0, len(pairs), CH):
        chunk = pairs[s0:s0 + CH]
        # group by target shape so maps stack
        by_shape = {}
        for n, (tk, sk) in enumerate(chunk):
            by_shape.setdefault(store[tk]['log'].shape, []).append((tk, sk))
        for (h, w), lst in by_shape.items():
            P = len(lst)
            e_t = [store[tk]['eye'] for tk, _ in lst]
            e_s = [store[sk]['eye'] for _, sk in lst]
            ys = torch.arange(h, device=DEV, dtype=F32).view(1, h, 1)
            xs = torch.arange(w, device=DEV, dtype=F32).view(1, 1, w)
            tx0 = as_t(np.float32([e.x0 for e in e_t])).view(P, 1, 1)
            ty0 = as_t(np.float32([e.y0 for e in e_t])).view(P, 1, 1)
            tl0 = as_t(np.float32([float(e.limbus[0]) for e in e_t])).view(P, 1, 1)
            tl1 = as_t(np.float32([float(e.limbus[1]) for e in e_t])).view(P, 1, 1)
            tax = as_t(np.float32([e.ax for e in e_t])).view(P, 1, 1)
            tay = as_t(np.float32([e.ay for e in e_t])).view(P, 1, 1)
            xx = (xs + tx0).expand(P, h, w)
            yy = (ys + ty0).expand(P, h, w)
            u = (xx - tl0) / tax
            v = (yy - tl1) / tay
            sl0 = as_t(np.float32([float(e.limbus[0]) for e in e_s])).view(P, 1, 1)
            sl1 = as_t(np.float32([float(e.limbus[1]) for e in e_s])).view(P, 1, 1)
            sax = as_t(np.float32([e.ax for e in e_s])).view(P, 1, 1)
            say = as_t(np.float32([e.ay for e in e_s])).view(P, 1, 1)
            sx0 = as_t(np.float32([e.x0 for e in e_s])).view(P, 1, 1)
            sy0 = as_t(np.float32([e.y0 for e in e_s])).view(P, 1, 1)
            mx = (sl0 + u * sax) - sx0
            my = (sl1 + v * say) - sy0
            src_l = torch.stack([canv[sk][0] for _, sk in lst])
            src_v = torch.stack([canv[sk][1] for _, sk in lst])
            got = G.remap_linear(src_l, mx, my, border='replicate')
            okv = G.remap_nearest(src_v, mx, my) > 0.5
            for n, (tk, sk) in enumerate(lst):
                g = got[n]
                if sk != tk:
                    g = torch.where(okv[n], g, torch.tensor(float('nan'), dtype=F32, device=DEV))
                stacks[tk].append((sk[0], g))
    for k in keys:
        layers = [g for _, g in sorted(stacks[k], key=lambda t: t[0])]
        own = store[k]['log']
        if len(layers) >= 3:
            st = torch.stack(layers)
            s = st.sort(0).values
            n = (~torch.isnan(st)).sum(0)
            lo = ((n - 1).clamp(min=0) // 2)[None]
            hi = (n // 2).clamp(max=st.shape[0] - 1)[None]
            med = (torch.gather(s, 0, lo)[0] + torch.gather(s, 0, hi)[0]) / G.dev_scalar(2.0)
            med = torch.where(n > 0, med, torch.full_like(med, float('nan')))
            out[k] = torch.where(torch.isfinite(med), med, own)
        else:
            out[k] = own
    return out


# =========================================================================== review tile
def eye_tile8(src8, out8, weight, points, topology, zoom, source_frame, eyes, half=17):
    """esr.eye_tile from the 8-bit frames already made (rgb8 is elementwise, so a crop of
    the converted frame is the conversion of the crop)."""
    cols = []
    H, W = src8.shape[:2]
    for side, rec in zip(r32.SIDES, eyes):
        centre, _, _ = r32.limbus_ellipse(points, topology['iris_' + side])
        cx, cy = int(round(centre[0])), int(round(centre[1]))
        y0, y1 = max(0, cy - half), min(H, cy + half)
        x0, x1 = max(0, cx - half), min(W, cx + half)
        s = src8[y0:y1, x0:x1, ::-1].copy()
        w = weight[y0:y1, x0:x1]
        edge = cv2.dilate((w > 0.02).astype(np.uint8), np.ones((3, 3), np.uint8))
        edge = (edge > 0) & ~(w > 0.02)
        s[edge] = (0, 200, 255)
        pair = np.concatenate([s, out8[y0:y1, x0:x1, ::-1]], axis=0)
        cols.append(cv2.resize(pair, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST))
    h = max(c.shape[0] for c in cols)
    cols = [np.pad(c, ((0, h - c.shape[0]), (0, 4), (0, 0))) for c in cols]
    tile = np.concatenate(cols, axis=1)
    label = np.zeros((22, tile.shape[1], 3), np.uint8)
    text = '%d %s' % (source_frame, ''.join('.' if e.get('repaired') else 'x' for e in eyes))
    cv2.putText(label, text, (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1,
                cv2.LINE_AA)
    return np.concatenate([label, tile], axis=0)


def _srgb_scalar(x):
    """round(float(linear_to_srgb(clip(x, 0, 1))), 4) on the CPU value -- the reported
    peaks are single numbers and numpy computes them exactly as the reference does."""
    return round(float(linear_to_srgb(np.clip(x, 0, 1))), 4)


# =========================================================================== limbus placement
def refine_limbus(eyes, cut, fitted, args):
    """Re-place an eye where the CANONICAL EYE says the limbus fit slid off her iris.

    Round 38's e2 (source 243 left, the "double iris") and the audit's class B: the limbus
    fit converges on the wrong circle for runs of frames -- +8 px sideways at 493-497
    left, 3.5 px vertical wander over 933-955 right -- confidently enough that no temporal
    rule can tell a slide from a saccade, and round 37's one-frame gate then "repairs" the
    good neighbours toward the slid frames (241, 242, 246, 247 left).

    The pass already holds the one model of her eye that does not look at the limbus edge:
    the canonical iris, pupil and limbal ring, built once from ~1000 frames. So every eye
    is matched against it over +/-`--refine-span` px around its tracked centre (NCC of
    log display luminance against the canonical rendered at each candidate, frozen
    registration), and moved only when the canonical fits CLEARLY better somewhere else:
    at least `--refine-min-shift` px away and `--refine-min-gain` higher correlation.

    **Only the lower part of the eye is compared** (below `--refine-low` radii above the
    centre). Compared over the whole opening, the canonical -- which has no lid shadow --
    is pulled UP into the dark lash band at a turned head: at source 238/240 left it sat
    4 px high on the lid while the track was right. The lower limbus, the sclera either
    side and the lower pupil carry the position without the lash.
    """
    offs = np.arange(-args.refine_span, args.refine_span + 1e-9, args.refine_step)
    n1 = len(offs)
    DX, DY = np.meshgrid(offs, offs)
    K = DX.size
    c0k = K // 2
    assert DX.ravel()[c0k] == 0 and DY.ravel()[c0k] == 0
    dxt, dyt = as_t(DX.ravel()), as_t(DY.ravel())
    step = G.dev_scalar(atl.ATLAS_STEP)
    half = float(np.float32((atl.ATLAS_N - 1) / 2))
    moved, report = {}, []
    for side in r32.SIDES:
        fit = fitted[side]
        if fit is None:
            continue
        reg = fit['frozen_registration']
        atlas = as_t(fit['atlas'])
        aw = as_t(fit['aw'].astype(np.float32))[..., None]
        du, dv = np.float32(reg['du']), np.float32(reg['dv'])
        sc = G.dev_scalar(float(np.float32(reg['scale'])))
        shapes = {}
        for (i, sd), e in sorted(eyes.items()):
            if sd == side:
                shapes.setdefault((e.h, e.w), []).append(e)
        for shape, lst in sorted(shapes.items()):
            g = Group(side, lst)
            g.patch = torch.stack([cut(e) for e in lst])
            eye_frame(g)
            solve_eye(g)
            lum = display_lum(g.patch)
            core, _ = corneal_specular(lum, g.rho)
            spec = G.dilate(core, np.ones((9, 9), bool))
            vrel = div(g.yy - g.ly, g.ay)
            mask = g.visible & (g.rho < 1.2) & ~spec & (vrel > args.refine_low)
            ls = torch.log(torch.clamp_min(lum, 1e-3)).to(F64)
            B, h, w = g.B, g.h, g.w
            per = max(1, int(args.refine_chunk // (K * h * w)))
            for s0 in range(0, B, per):
                s1 = min(B, s0 + per)
                b = s1 - s0
                cx = g.lx[s0:s1].view(b, 1, 1, 1) + dxt.view(1, K, 1, 1)
                cy = g.ly[s0:s1].view(b, 1, 1, 1) + dyt.view(1, K, 1, 1)
                p = ((g.xx[s0:s1][:, None] - cx) / g.ax[s0:s1].view(b, 1, 1, 1)).to(F32)
                q = ((g.yy[s0:s1][:, None] - cy) / g.ay[s0:s1].view(b, 1, 1, 1)).to(F32)
                ai = ((p - du) / sc) / step + half
                aj = ((q - dv) / sc) / step + half
                pred = G.remap_single(atlas, ai, aj)
                sup = G.remap_single(aw, ai, aj)[..., 0] > 0.98
                disp = torch.pow(torch.clamp(pred, 0, 1), 1 / 2.4) * 1.055 - 0.055
                pl = torch.log(torch.clamp(disp.max(-1).values, 1e-3, 1.0)).to(F64)
                mf = (mask[s0:s1][:, None] & sup).to(F64)
                n = mf.sum((2, 3))
                a = ls[s0:s1][:, None] * mf
                bb = pl * mf
                nn = n.clamp_min(1)
                sa, sb = a.sum((2, 3)), bb.sum((2, 3))
                vab = (a * bb).sum((2, 3)) - sa * sb / nn
                vaa = (a * a).sum((2, 3)) - sa * sa / nn
                vbb = (bb * bb).sum((2, 3)) - sb * sb / nn
                c = vab / torch.sqrt(torch.clamp_min(vaa * vbb, 1e-18))
                valid = n >= 0.6 * n.max(1, keepdim=True).values
                c = torch.where(valid, c, torch.full_like(c, -2.0))
                cn = c.cpu().numpy()
                nn_np = n.cpu().numpy()
                for r in range(b):
                    e = g.eyes[s0 + r]
                    k = int(np.argmax(cn[r]))
                    iy, ix = divmod(k, n1)
                    surf = cn[r].reshape(n1, n1)
                    dx, dy = float(offs[ix]), float(offs[iy])
                    for axis in (0, 1):
                        j = ix if axis == 0 else iy
                        if 0 < j < n1 - 1:
                            if axis == 0:
                                l_, m_, r_ = surf[iy, j - 1], surf[iy, j], surf[iy, j + 1]
                            else:
                                l_, m_, r_ = surf[j - 1, ix], surf[j, ix], surf[j + 1, ix]
                            den = l_ - 2 * m_ + r_
                            if den < 0:
                                sft = float(np.clip(0.5 * (l_ - r_) / den, -0.5, 0.5)) * args.refine_step
                                if axis == 0:
                                    dx += sft
                                else:
                                    dy += sft
                    best, at0 = float(cn[r, k]), float(cn[r, c0k])
                    shift = float(np.hypot(dx, dy))
                    ok = (shift >= args.refine_min_shift and best - at0 >= args.refine_min_gain
                          and best >= args.refine_min_ncc and nn_np[r, k] >= args.refine_min_px)
                    if ok:
                        moved[(e.i, side)] = (float(e.limbus[0]) + dx, float(e.limbus[1]) + dy)
                        report.append(dict(frame=e.i, side=side, dx=round(dx, 2), dy=round(dy, 2),
                                           ncc=round(best, 3), ncc_at_track=round(at0, 3),
                                           mask_px=int(nn_np[r, k])))
    return moved, report


class _Centres:
    """A limbus track with some centres replaced, for `esr.gate_limbus`."""

    def __init__(self, track, fixed):
        self.track, self.fixed = track, fixed

    def get(self, frame, side):
        got = self.track.get(frame, side)
        if got is None:
            return None
        key = (int(frame), side)
        if key in self.fixed:
            return (tuple(self.fixed[key]), got[1], got[2])
        return got


def gate_limbus_trusted(track, corners, se, rate, gate=1.0, ratio=2.5, window_s=1.0,
                        max_gap_s=0.25):
    """Round 37's limbus gate, made to respect the fit's OWN uncertainty.

    Round 38's e2 (the "double iris", source 241-247 left) was two failures stacked. The
    limbus fit slid +6..+8 px on 243-245 -- and said so: its centre standard error rose to
    0.24-0.31 px against 0.04 either side. Then the one-frame gate, which replaces a frame
    that breaks from the MEAN OF ITS TWO NEIGHBOURS, took those slid frames as the
    neighbours of 242 and 246 and "repaired" the good frames toward the slide by 3.8 px,
    and 241 and 247 by 1.9 -- seven frames of double iris from three bad fits.

    So first, a frame whose standard error exceeds `ratio` times the median of its eye
    over +/-`window_s` seconds is not trusted: its centre is re-derived from the nearest
    TRUSTED frames either side (their offset from the eye corners, interpolated in time,
    carried by the corners here), when both lie within `max_gap_s` seconds. Then the
    one-frame gate runs on that track, so a distrusted frame can never be anyone's
    reference. Returns the gate's usual dict, entries carrying 'trust' or 'gate'.
    """
    half = max(1, int(round(window_s * float(rate))))
    gap = max(1, int(round(max_gap_s * float(rate))))
    fixed = {}
    info = {}
    for side in r32.SIDES:
        frames = sorted(i for (i, s) in se if s == side and track.get(i, side) is not None)
        vals = {i: se[(i, side)] for i in frames if se[(i, side)] is not None}
        trusted = {}
        for i in frames:
            s = vals.get(i)
            near = [vals[j] for j in range(i - half, i + half + 1) if j in vals]
            med = float(np.median(near)) if near else None
            trusted[i] = (s is not None and med is not None and s <= ratio * med
                          and (i, side) in corners)
        good = [i for i in frames if trusted[i]]
        for i in frames:
            if trusted[i] or (i, side) not in corners:
                continue
            prev = [j for j in good if i - gap <= j < i]
            nxt = [j for j in good if i < j <= i + gap]
            if not prev or not nxt:
                continue
            a, b = prev[-1], nxt[0]
            oa = np.asarray(track.get(a, side)[0], float) - corners[(a, side)]
            ob = np.asarray(track.get(b, side)[0], float) - corners[(b, side)]
            t = (i - a) / float(b - a)
            pred = corners[(i, side)] + (1 - t) * oa + t * ob
            here = np.asarray(track.get(i, side)[0], float)
            moved = float(np.hypot(*(pred - here)))
            if moved > gate:
                fixed[(i, side)] = pred
                info[(i, side)] = moved
    wrapped = _Centres(track, fixed)
    out = esr.gate_limbus(wrapped, corners, 0, gate)
    result = {}
    for k, c in fixed.items():
        got = track.get(*k)
        result[k] = (c, got[1], got[2], info[k], 'trust')
    for k, v in out.items():
        result[k] = (v[0], v[1], v[2], v[3], 'gate')
    return result


def lid_height(lid, limbus, ay):
    """The fitted upper lid's height above the limbus centre, in limbus radii, or None."""
    if not lid or lid.get('upper') is None:
        return None
    a, b, c = lid['upper']
    u = float(limbus[0]) - float(lid['centre'][0])
    return (float(limbus[1]) - (a * u * u + b * u + c)) / ay


# =========================================================================== final pass
def skin_runs(counts, high, low, max_neutral=2):
    """Which eyes the skin gate takes, with HYSTERESIS IN TIME.

    A per-frame count threshold toggles inside a run: at source 933-955 right (sideways
    gaze, the circular canonical larger than her foreshortened iris) the count moves
    between 3 and 21 frame to frame, and gating only the frames above 6 switched her eye
    between the canonical and her own iris every few frames. So an eye at or above `high`
    gates, and the gate spreads to the frames either side for as long as they still show
    at least `low` skin pixels under the fill -- one decision per run. Frames with no
    count (untreated, or already softbox-only) are passed over, up to `max_neutral` in a
    row.
    """
    gate = set()
    for side in r32.SIDES:
        c = {i: v for (i, s), v in counts.items() if s == side}
        for i, v in c.items():
            if v < high:
                continue
            gate.add((i, side))
            for step in (-1, 1):
                j, neutral = i + step, 0
                while True:
                    if j in c:
                        if c[j] < low:
                            break
                        gate.add((j, side))
                        neutral = 0
                    else:
                        neutral += 1
                        if neutral > max_neutral:
                            break
                    j += step
    return gate


def load_floors(payload, ref=None):
    """Every eye's `lower_iris` fit (eyelid_fit.py --lower-iris-edge) in the eye's own
    coordinates: height relative to `ref` (the eye corners, as load_lids measures the lids)
    and u relative to the lid fit's centre; plus, for EVERY eye with a centre, the centre
    and reference a neighbour's floor is carried by."""
    out = {}
    for row in payload['rows']:
        for e in row['eyes']:
            if not e.get('centre'):
                continue
            key = (int(row['frame']), e['side'])
            ref_y = float(ref[key][1]) if ref is not None and key in ref else float(e['centre'][1])
            rec = dict(centre=float(e['centre'][0]), ref_y=ref_y, coef=None)
            li = e.get('lower_iris')
            if li:
                a, b, c = li['coef']
                rec.update(coef=(a, b, c - ref_y), span=tuple(li['u_span']))
            out[key] = rec
    return out


def floor_rows(g, side, floors, margin, release=1.0):
    """--lid-floor: per eye, the row of the lower lid over her iris in each window column
    (float32, inf where there is none), and which eyes have one.

    **A per-column median over the frame and its two neighbours, a missing floor counting
    as no bound.** So an ISOLATED fit is never a floor (round 40, 858 right: 13 columns on
    a blink), a one-frame gap takes the lower of its neighbours' floors, and nothing is
    interpolated across a run of frames without one -- load_lids' bridging carried 858's
    and 866's lone fits across 862 right and cut into an open iris. Each fit holds only
    over the columns its samples span and releases `release` px per px beyond them: past
    them a tilted parabola is an extrapolation and climbs into the disc (21-22, 386,
    836-837 left)."""
    rows, has = [], []
    for e in g.eyes:
        here = floors.get((e.i, side))
        if here is None:
            rows.append(np.full(g.w, np.inf, np.float32))
            has.append(False)
            continue
        u = np.arange(e.x0, e.x0 + g.w, dtype=np.float64) - here['centre']
        cand = []
        for j in (e.i - 1, e.i, e.i + 1):
            f = floors.get((j, side))
            if f is None or f['coef'] is None:
                cand.append(np.full(g.w, np.inf))
                continue
            a, b, h = f['coef']
            uu = np.clip(u, f['span'][0], f['span'][1])
            cand.append(a * uu * uu + b * uu + h + here['ref_y'] + release * np.abs(u - uu))
        y = np.median(np.stack(cand), axis=0)
        rows.append((y - e.y0 - margin).astype(np.float32))
        has.append(bool(np.isfinite(y).any()))
    return as_t(np.stack(rows)), torch.as_tensor(has, device=DEV)


def final_weights(g, side, fit, reg, sg, args, ceil_dict, lids, store, measure=False,
                  gate_set=None, floors=None):
    """The per-eye body of eye_soft_removal's last loop, up to the shading, batched: the
    whole-iris fill bounded by the lids, the excess-based removal, the lash line, the
    specular region replaced whole, the gates, and the edge / region shading. Eyes that
    reach the shading are put in `store` for the temporal median; the blend is
    `blend_group`. Returns {(i, side): result}."""
    B, h, w = g.B, g.h, g.w
    out = {}
    du, dv, scale = reg['du'], reg['dv'], reg['scale']
    recs = []
    active = []
    for n, e in enumerate(g.eyes):
        # An eye first collected after --refine-limbus moved it was never registered: it
        # carries the build's initialiser, as an unregistered frame does.
        ncc = fit['ncc'].get(e.i, 0.0)
        fs = g.frame_scale[n]
        rec = dict(side=side, limbus_centre=[round(float(c), 3) for c in e.limbus],
                   limbus_axes=[round(e.ax, 3), round(e.ay, 3)],
                   registration_shift_px=[round(float(du * e.ax), 4), round(float(dv * e.ay), 4)],
                   registration_ncc=round(float(ncc), 5),
                   frame_residual_scale=None if fs is None else round(fs, 6))
        if getattr(args, '_moved', None) and (e.i, side) in args._moved:
            rec['limbus_refined_px'] = args._moved[(e.i, side)]
        if fit['registered'].get(e.i, False) and ncc < args.min_ncc_frozen:
            rec.update(repaired=False, reason='registration_below_threshold',
                       registration_frozen=True)
            active.append(False)
        else:
            active.append(True)
        recs.append(rec)
    # the painted disc in the canonical's registered coordinates
    p_ = div(g.xx - g.lx, g.ax)
    q_ = div(g.yy - g.ly, g.ay)
    au = div(p_ - du, scale)
    av = div(q_ - dv, scale)
    rho = torch.hypot(au, av)
    edge = as_t(np.array([1.5 / max(min(e.ax, e.ay), 1e-6) for e in g.eyes])).view(B, 1, 1)
    edge = torch.clamp_min(edge, 1e-6)
    wt = torch.clamp((args.fill_iris - rho) / edge, 0.0, 1.0).to(F32)
    wt = wt * (g.visible & g.support).to(F32)
    ceils, has_ceil, nub = [], [], []
    for n, e in enumerate(g.eyes):
        c = ceil_dict.get(e.i)
        has_ceil.append(c is not None)
        ceils.append(np.full(w, np.inf, np.float32) if c is None else c)
        lid_here = lids.get((e.i, side))
        no_up = (args.lid_temporal and c is None
                 and (lid_here is None or lid_here.get('upper') is None))
        # A vetoed eye never reaches the fill in the reference, so it gets no note.
        if no_up and active[n]:
            recs[n]['fill_abstained'] = 'no_upper_bound'
        # **A lid fit the image does not confirm is not a bound.** Round 38's e1/e3 and
        # every whole-disc-over-a-squint frame the round 39 audit found (137, 293, 301-304,
        # 617, 624-626, 651-653, 859, 1099 left, 302 right): the lash walk finds no lash
        # line, while the fitted upper lid -- on the lid crease, or bridged from frames
        # where it was -- claims more than `--lid-claim` limbus radii of open iris above
        # the centre. An eye that open has a lash above its iris for the walk to find, so
        # the two contradict each other, and the fill painted the whole disc over her lid.
        # Measured on the clip: of 128 treated eyes with no lash line, exactly these 15
        # have a lid above 0.8 radii, and all 15 paint lid skin.
        if (not no_up and args.lid_claim > 0 and c is None):
            hgt = lid_height(lid_here, e.limbus, e.ay)
            if hgt is not None and hgt > args.lid_claim:
                no_up = True
                if active[n]:
                    recs[n]['fill_abstained'] = 'lid_unconfirmed_%.2fr' % hgt
        fs = g.frame_scale[n]
        off = (args.fill_residual_gate > 0 and fs is not None
               and fs > args.fill_residual_gate * sg)
        if off and not no_up:
            if active[n]:
                recs[n]['fill_abstained'] = 'residual_%.1fx_shot' % (fs / sg)
            no_up = True
        nub.append(no_up)
    nub_t = torch.as_tensor(nub, device=DEV)
    wt = torch.where(nub_t.view(B, 1, 1), torch.zeros_like(wt), wt)
    sup = torch.clamp(G.distance_chamfer(g.support, 3) / G.dev_scalar(2.0), 0.0, 1.0)
    wt = wt * sup
    lum = g.lum
    ring_out = (rho > 1.15) & (rho < 1.55)
    pct, cnt = G.percentile(lum, ring_out, 90)
    sclera = [float(x) if c > 40 else 0.5 for x, c in zip(pct.cpu().numpy(), cnt.cpu().numpy())]
    thr = as_t(np.float32([max(0.45, s + 0.06) for s in sclera])).view(B, 1, 1)
    hot = (lum > thr) & (rho < 1.05)
    labels = G.connected_components(hot)
    inside_opening = g.visible & g.support
    touch = torch.where(hot & inside_opening, labels, torch.zeros_like(labels))
    keep = torch.zeros((B, h * w + 2), dtype=torch.bool, device=DEV)
    keep.scatter_(1, touch.reshape(B, -1), torch.ones_like(touch.reshape(B, -1), dtype=torch.bool))
    keep[:, 0] = False
    blob = torch.gather(keep, 1, labels.reshape(B, -1)).view(B, h, w)
    blob = G.dilate(blob, G.ellipse_kernel(5)) & (rho < 1.05)
    wide = ((rho < args.fill_iris) & g.support & g.lid_pos) | (blob & g.support)
    wide &= g.support
    ws, wsok, top, seed = soft_weight(g.signed, g.under, wide, g.support, sg, args.k_lo,
                                      args.k_hi, args.reach, rim_reach=args.rim_reach,
                                      rim_region=g.visible)
    wsok = wsok & ~nub_t
    above_lid = ((ws > 0) & ~(g.visible & g.support)).sum((1, 2))
    wt = torch.where(wsok.view(B, 1, 1), torch.maximum(wt, ws), wt)
    wt = quantize(wt)
    peak_excess = torch.where(wide, g.signed, torch.full_like(g.signed, -float('inf'))).reshape(B, -1).max(1).values
    # the lash line
    ceil_t = as_t(np.stack(ceils))
    has_ceil_t = torch.as_tensor(has_ceil, device=DEV)
    rows_f = torch.arange(h, device=DEV, dtype=F32).view(1, h, 1)
    allow = torch.clamp(rows_f - ceil_t[:, None, :], 0.0, 1.0)
    lash_px = ((wt > 0) & (allow < 1.0)).sum((1, 2))
    wt = torch.where(has_ceil_t.view(B, 1, 1), quantize(wt * allow), wt)
    lash_px = torch.where(has_ceil_t, lash_px, torch.zeros_like(lash_px))
    # **The lower lid over her iris bounds the fill from below** (--lid-floor, round 40) --
    # the mirror of the lash line. Where her lower lid covers the iris the lid parabola was
    # an extrapolation of the sclera columns and sat below it: 76 % of every pixel of lid
    # skin the fill painted, clip-wide, was there (933-955 right, 866-867, 11, 584, 977),
    # and the skin gate handed those eyes the softbox region only. The floor is measured in
    # her image (eyelid_fit.iris_edge_samples) and filtered in time like the lids.
    floor_px = torch.zeros(B, dtype=torch.int64, device=DEV)
    if floors is not None:
        fl_t, has_fl = floor_rows(g, side, floors, args.floor_margin)
        allow_f = torch.clamp(fl_t[:, None, :] - rows_f, 0.0, 1.0)
        floor_px = torch.where(has_fl, ((wt > 0) & (allow_f < 1.0)).sum((1, 2)), floor_px)
        wt = torch.where(has_fl.view(B, 1, 1), quantize(wt * allow_f), wt)
        g.floor_rows = fl_t
    # the specular's own region (replaced whole below)
    spec_px = torch.zeros(B, dtype=torch.int64, device=DEV)
    wb = torch.zeros_like(wt)
    sfound = torch.zeros(B, dtype=torch.bool, device=DEV)
    core_s = torch.zeros_like(g.visible)
    if args.specular_reach > 0:
        core_s, sfound = corneal_specular(lum, g.rho)
        wb = specular_region(core_s, sfound, g.rho, args.specular_reach)
        wb = wb * sup
    # **Never paint iris over skin or sclera.** The safety net under --lid-claim and
    # --refine-limbus: an eye whose fill would still cover `--skin-gate` pixels that HER
    # photograph shows brighter than halfway between her iris and her lid skin (the
    # softbox's own region excepted) gets only the softbox's region, like a frame with no
    # upper bound. Iris, pupil and lash are dark on this plate and skin and sclera
    # bright, so painting canonical iris over one is a wrong bound (class A) or a slid
    # limbus (class B) -- measured, every eye at >= 6 px was one; the eyes the artist
    # accepted in round 37 reach 5.
    skin_px = torch.zeros(B, dtype=torch.int64, device=DEV)
    if args.skin_gate > 0:
        near_s = G.dilate(core_s, np.ones((5, 5), bool))
        iris_sel = g.visible & (g.rho > 0.45) & (g.rho < 0.85) & ~near_s
        iris_lv, n_iris = G.median(lum.reshape(B, -1), iris_sel.reshape(B, -1), dim=1)
        band = (g.rho > 1.3) & (g.rho < 1.65) & (g.yy < g.ly)
        skin_lv, n_band = G.percentile(lum, band, 75)
        iris_lv = torch.where(n_iris > 20, iris_lv, torch.full_like(iris_lv, 0.25))
        skin_lv = torch.where(n_band > 20, skin_lv, torch.full_like(skin_lv, 0.7))
        thr = (0.5 * (iris_lv + skin_lv)).view(B, 1, 1)
        skin_px = ((wt >= 0.5) & ~(wb > 0) & (lum > thr)).sum((1, 2))
        act_t = torch.as_tensor(active, device=DEV)
        if measure:
            eligible = (~nub_t & act_t).cpu().numpy()
            sp = skin_px.cpu().numpy()
            return {(e.i, side): int(sp[n]) for n, e in enumerate(g.eyes) if eligible[n]}
        if gate_set is None:
            gate = (skin_px >= args.skin_gate) & ~nub_t & act_t
        else:
            gate = torch.as_tensor([(e.i, side) in gate_set for e in g.eyes], device=DEV)
            gate = gate & ~nub_t & act_t
        if gate.any():
            gv = gate.view(B, 1, 1)
            wt = torch.where(gv, torch.zeros_like(wt), wt)
            wsok = wsok & ~gate
            lash_px = torch.where(gate, torch.zeros_like(lash_px), lash_px)
            for n in torch.nonzero(gate)[:, 0].tolist():
                tag = '' if gate_set is None or skin_px[n] >= args.skin_gate else '_run'
                recs[n]['fill_abstained'] = 'skin_%dpx%s' % (int(skin_px[n]), tag)
        g.skin_px = skin_px.cpu().numpy()
    elif measure:
        return {}
    w_fill = wt
    if args.specular_reach > 0:
        spec_px = torch.where(sfound, (wb > wt).sum((1, 2)), spec_px)
        wt = torch.where(sfound.view(B, 1, 1), quantize(torch.maximum(wt, wb)), wt)
    # gates
    wmax = wt.reshape(B, -1).max(1).values
    heavy = wt > 0.5
    nheavy = heavy.sum((1, 2))
    bright = (g.signed > args.k_hi * sg) & (wt > 0.5)
    bright = torch.where((bright.sum((1, 2)) < 3).view(B, 1, 1), heavy, bright)
    neg = torch.full_like(g.base, -float('inf'))
    bmax = torch.where(bright[..., None], g.base, neg).reshape(B, -1).max(1).values
    pmax = torch.where(bright[..., None], g.patch, torch.full_like(g.patch, -float('inf'))).reshape(B, -1).max(1).values
    wmax_np, nheavy_np = wmax.cpu().numpy(), nheavy.cpu().numpy()
    darker = (bmax < pmax.to(F64)).cpu().numpy()
    wsum = wt.sum((1, 2)).cpu().numpy()
    status = []
    for n in range(B):
        rec = recs[n]
        if not active[n]:
            status.append(None)
        elif not (wmax_np[n] > 0):
            rec.update(repaired=False, reason='no_reflection_found')
            status.append(None)
        elif nheavy_np[n] < 3:
            rec.update(repaired=False, reason='no_confident_pixels',
                       weighted_pixels=round(float(np.sum(wt[n].cpu().numpy())), 3))
            status.append(None)
        elif not darker[n]:
            rec.update(repaired=False, reason='prediction_not_darker_than_source',
                       weighted_pixels=round(float(np.sum(wt[n].cpu().numpy())), 3))
            status.append(None)
        else:
            status.append('shade')
    # shading
    go = torch.as_tensor([s == 'shade' for s in status], device=DEV)
    comp_log = torch.zeros((B, h, w), dtype=F32, device=DEV)
    shade_range = [None] * B
    region_range = [None] * B
    shade_boundary = [0] * B
    if args.top_shading > 0 and go.any():
        shade, nb, shok = top_shading(g.patch, g.base, w_fill, g.rho, decay=args.top_shading,
                                      exclude=((wb > 0), sfound) if args.region_rows else None)
        top_log = torch.where(shok.view(B, 1, 1), _log32(torch.clamp_min(shade, 1e-4)),
                              torch.zeros_like(shade))
        sel = w_fill > 0
        inf = torch.tensor(float('inf'), dtype=F32, device=DEV)
        tmin = torch.where(sel, top_log, inf).reshape(B, -1).min(1).values.cpu().numpy()
        tmax = torch.where(sel, top_log, -inf).reshape(B, -1).max(1).values.cpu().numpy()
        anysel = sel.any((1, 2)).cpu().numpy()
        nb_np, shok_np = nb.cpu().numpy(), shok.cpu().numpy()
        comp = top_log.clone()
        omega = (wb > 0) & (w_fill < 0.5)
        base_top = g.base * _exp32(top_log)[..., None]
        wf = w_fill[..., None]
        compimg = (g.patch * (1 - wf)).to(F64) + base_top * wf
        hot2 = (display_lum(g.patch) > 0.35) & ~(w_fill >= 0.5)
        rf, rfok = region_rows(compimg, g.base, w_fill >= 0.5, wb > 0, hot=hot2)
        rfok = rfok & sfound
        comp = torch.where(omega & rfok.view(B, 1, 1), _log32(torch.clamp_min(rf, 1e-4)), comp)
        rmin = torch.where(omega, rf, inf).reshape(B, -1).min(1).values.cpu().numpy()
        rmax = torch.where(omega, rf, -inf).reshape(B, -1).max(1).values.cpu().numpy()
        rfok_np = rfok.cpu().numpy()
        for n in range(B):
            if status[n] != 'shade':
                continue
            shade_boundary[n] = int(nb_np[n]) if shok_np[n] else 0
            if anysel[n]:
                shade_range[n] = [round(float(np.exp(np.float32(tmin[n]))), 4),
                                  round(float(np.exp(np.float32(tmax[n]))), 4)]
            if rfok_np[n]:
                region_range[n] = [round(float(rmin[n]), 4), round(float(rmax[n]), 4)]
        comp_log = comp
    valid = wt > 0
    for n, e in enumerate(g.eyes):
        if status[n] == 'shade' and args.top_shading > 0 and args.shading_temporal > 0:
            store[(e.i, side)] = dict(log=comp_log[n], valid=valid[n], eye=e)
    g.w_final, g.w_fill, g.comp_log, g.wb = wt, w_fill, comp_log, wb
    g.status = status
    g.recs = recs
    g.info = dict(top=top.cpu().numpy(), above_lid=above_lid.cpu().numpy(),
                  lash_px=lash_px.cpu().numpy(), spec_px=spec_px.cpu().numpy(),
                  wsok=wsok.cpu().numpy(), peak_excess=peak_excess.cpu().numpy(),
                  floor_px=floor_px.cpu().numpy() if floors is not None else None)
    g.bright = bright
    g.shade_range, g.region_range, g.shade_boundary = shade_range, region_range, shade_boundary
    for n, e in enumerate(g.eyes):
        out[(e.i, side)] = dict(record=recs[n], eye=e, group=g, slot=n)
    return out


def blend_group(g, side, sg, args, results, shade_filtered):
    """The temporal shading median applied, the blend, and the reported statistics."""
    B, h, w = g.B, g.h, g.w
    go = [s == 'shade' for s in g.status]
    if not any(go):
        return
    comp = g.comp_log.clone()
    tchange = [None] * B
    if shade_filtered is not None:
        filt_l, idx_l = [], []
        for n, e in enumerate(g.eyes):
            f = shade_filtered.get((e.i, side))
            if go[n] and f is not None:
                filt_l.append(f)
                idx_l.append(n)
        if idx_l:
            idx = torch.as_tensor(idx_l, device=DEV)
            filt = torch.stack(filt_l)
            diff = (filt - comp[idx]).abs()
            p95, _ = G.percentile(diff, g.w_final[idx] > 0, 95)
            for k, n in enumerate(idx_l):
                tchange[n] = round(float(p95[k].cpu()), 4)
            comp[idx] = filt
    wt = g.w_final
    base_used = g.base * _exp32(comp)[..., None]
    wf = wt[..., None]
    blended = (g.patch * (1 - wf)).to(F64) + base_used * wf
    blended = torch.clamp_min(blended, 0.0)
    out_win = torch.where((wt > 0)[..., None], blended.to(F32), g.patch)
    after = blended.max(-1).values - g.base.max(-1).values
    near = G.dilate(wt > 0.5, np.ones((7, 7), bool)) & g.inside
    heavy = wt > 0.5
    # per-eye reductions
    neg = -float('inf')
    near_peak = torch.where((g.inside & near)[..., None], blended, torch.full_like(blended, neg)).reshape(B, -1).max(1).values
    sur_peak = torch.where((g.inside & ~near)[..., None], blended, torch.full_like(blended, neg)).reshape(B, -1).max(1).values
    any_near = (g.inside & near).any((1, 2))
    any_sur = (g.inside & ~near).any((1, 2))
    br = g.bright
    psrc = torch.where(br[..., None], g.patch, torch.full_like(g.patch, neg)).reshape(B, -1).max(1).values
    prem = torch.where(br[..., None], blended, torch.full_like(blended, neg)).reshape(B, -1).max(1).values
    hsum = (g.base * heavy[..., None]).sum((1, 2))
    hcnt = heavy.sum((1, 2)).to(F64).clamp_min(1)
    fill_mean = hsum / hcnt[:, None]
    rim = ((g.under > args.k_lo * sg) & heavy).sum((1, 2))
    floor_px = (((g.patch <= 1e-6).sum(-1) >= 2) & heavy).sum((1, 2))
    out8 = G.rgb8(out_win)
    fsc = g.frame_scale
    r33 = [None if fs is None else max(4.0 * fs, 0.012) for fs in fsc]
    r33_t = as_t(np.array([np.inf if x is None else x for x in r33])).view(B, 1, 1)
    above_r33 = (g.inside & near & (after > r33_t)).sum((1, 2))
    above_2s = (near & (after > 2 * sg)).sum((1, 2))
    touched = (wt > 0).sum((1, 2))
    full = (wt > 0.99).sum((1, 2))
    wt_np = wt.cpu().numpy()
    out8_np = out8.cpu().numpy()
    vals = {k: v.cpu().numpy() for k, v in dict(
        near_peak=near_peak, sur_peak=sur_peak, any_near=any_near, any_sur=any_sur,
        psrc=psrc, prem=prem, fill_mean=fill_mean, rim=rim, floor_px=floor_px,
        above_r33=above_r33, above_2s=above_2s, touched=touched, full=full).items()}
    info = g.info
    for n, e in enumerate(g.eyes):
        if not go[n]:
            continue
        rec = g.recs[n]
        if tchange[n] is not None:
            rec['shading_temporal_change_p95'] = tchange[n]
        rec.update(
            repaired=True, reason=None,
            sequence_residual_scale=round(sg, 6),
            r33_threshold_linear=None if r33[n] is None else round(float(r33[n]), 6),
            residual_pixels_above_r33_threshold=(None if r33[n] is None
                                                 else int(vals['above_r33'][n])),
            peak_excess_sigmas=round(float(info['peak_excess'][n] / sg), 3),
            weighted_pixels=round(float(np.sum(wt_np[n])), 3),
            pixels_touched=int(vals['touched'][n]),
            pixels_full_weight=int(vals['full'][n]),
            residual_pixels_above_2sigma=int(vals['above_2s'][n]),
            residual_peak_display=(_srgb_scalar(np.float64(vals['near_peak'][n]))
                                   if vals['any_near'][n] else None),
            surround_peak_display=(_srgb_scalar(np.float64(vals['sur_peak'][n]))
                                   if vals['any_sur'][n] else None),
            peak_source_display=_srgb_scalar(np.float32(vals['psrc'][n])),
            peak_removed_display=_srgb_scalar(np.float64(vals['prem'][n])),
            highlight_pixels_above_lid=int(info['above_lid'][n]) if info['wsok'][n] else 0,
            lash_guard_pixels=int(info['lash_px'][n]),
            specular_region_added=int(info['spec_px'][n]),
            top_shading_range=g.shade_range[n],
            region_shading_range=g.region_range[n],
            top_shading_boundary_px=g.shade_boundary[n],
            rim_pixels_removed=int(vals['rim'][n]),
            floor_pixels_removed=int(vals['floor_px'][n]),
            grain_scale=None,
            fill_display=[int(round(float(linear_to_srgb(np.clip(c, 0, 1)) * 255)))
                          for c in vals['fill_mean'][n]])
        if info.get('floor_px') is not None:
            rec['floor_pixels'] = int(info['floor_px'][n])
        res = results[(e.i, side)]
        res['out8'] = out8_np[n]
        res['w'] = wt_np[n]


def write_record(root, shot, args, rows, sheets, sigma, fitted, lids, lid_curvature,
                 lid_curv_frame, limbus_fix, refine_report=None):
    """removal.json with the reference's keys, plus which implementation made it."""
    repaired = [r for r in rows if r['changed_pixels']]
    here = Path(__file__).resolve().parent
    record = dict(
        rows=rows, contact_sheets=sheets,
        frames=len(rows), frames_repaired=len(repaired),
        frames_exact_source=len(rows) - len(repaired),
        total_changed_pixels=int(sum(r['changed_pixels'] for r in rows)),
        sequence_residual_scale={s: sigma.get(s) for s in r32.SIDES},
        weight_ramp_sigmas=[args.k_lo, args.k_hi], reach_px=args.reach,
        gain_model=args.gain_model, union_footprint=bool(args.union_footprint),
        fill_iris=args.fill_iris, pupil_target=args.pupil_target,
        pupil_level={sd: (None if fitted[sd] is None else
                          [fitted[sd].get('pupil_before'), fitted[sd].get('pupil_after')])
                     for sd in r32.SIDES},
        grain_fraction=args.grain,
        atlas={s: (None if fitted[s] is None else dict(
            frames=fitted[s]['frames'], iterations=fitted[s]['history'],
            sharpness=round(fitted[s]['sharpness'], 5),
            unregistered_seed_sharpness=round(fitted[s]['seed_sharpness'], 5),
            holes_interpolated=fitted[s]['atlas_holes_interpolated'],
            holes_remaining=fitted[s]['atlas_holes_remaining']))
            for s in r32.SIDES},
        method='round 33 sequence atlas, with a CONTINUOUS removal weight on the excess '
               'over it and ONE residual scale per eye for the whole shot; no mask',
        atlas_frames=None, atlas_mode=args.atlas_mode,
        generated_cells={sd: (None if fitted[sd] is None else 0) for sd in r32.SIDES},
        material_source='observed clean pixels of the same eye, pooled over the shot',
        generated_material_used=False, scene_reflection_added=False,
        catchlight_render=None, catchlight_gain=None, catchlight_gain_source=None,
        catchlight_matte='tracked eyelid polygon AND the limbus, both feathered ~1 px',
        dark_rim_repaired=not args.keep_rim, rim_reach_px=args.rim_reach,
        production_enabled=False, artist_accepted=False,
        source_outside_weight_exact=True,
        lid_extend_px=args.lid_extend, lid_inset_px=args.lid_inset,
        atlas_lid_inset_px=args.atlas_lid_inset,
        lash_guard=bool(args.lash_guard), lash_bright=args.lash_bright,
        lash_exempt=bool(args.lash_exempt), lid_line=args.lid_line,
        lid_line_margin_px=args.lid_line_margin, lid_line_curvature=lid_curvature,
        lid_line_curvature_per_frame=len(lid_curv_frame),
        specular_reach_px=args.specular_reach, top_shading_decay_px=args.top_shading,
        shading_sides=args.shading_sides, canonical_votes=args.canonical_votes,
        canonical_votes_frac=args.canonical_votes_frac,
        gain_ring=args.gain_ring, gain_temporal=bool(args.gain_temporal),
        level_window=args.level_window, shading_temporal=args.shading_temporal,
        shading_lid=bool(args.shading_lid), region_heal=bool(args.region_heal),
        region_rows=bool(args.region_rows), pupil_colour=bool(args.pupil_colour),
        limbus_gate_px=args.limbus_gate,
        limbus_gated={'%d_%s' % (i + int(shot.source_start_frame), sd): round(v[3], 3)
                      for (i, sd), v in limbus_fix.items()},
        lid_line_window=args.lid_line_window, soft_lids=bool(args.soft_lids),
        lid_window=args.lid_window, lid_temporal=bool(args.lid_temporal),
        canonical_symmetry={k: (v.get('symmetry') if v else None) for k, v in fitted.items()},
        fill_residual_gate=args.fill_residual_gate, canonical_from=None,
        lid_shadow_strength=args.lid_shadow, lid_shadow=None,
        atlas_grid=atl.configure(args.atlas_reach, args.atlas_cells),
        sample_reach=args.sample_reach, atlas_cap_percentile=args.atlas_cap,
        atlas_cap_level={k: None for k in fitted},
        ncc_floor=args.min_ncc_frozen,
        eyelid_track=args.eyelid_track, eyelid_track_sha256=r32.sha(shot.relight / args.eyelid_track),
        eyelid_source='image_fit' if lids else 'mediapipe_polygon',
        eyelid_fitted_eyes=len(lids),
        freeze_registration=bool(args.freeze_registration),
        frozen_registration={k: (v['frozen_registration'] if v else None)
                             for k, v in fitted.items()},
        limbus_track=args.limbus_track,
        limbus_track_sha256=r32.sha(shot.relight / args.limbus_track),
        limbus_source='image_fit',
        lid_floor=bool(args.lid_floor), floor_margin_px=args.floor_margin,
        lid_claim_radii=args.lid_claim, skin_gate_px=args.skin_gate,
        skin_gate_low_px=args.skin_gate_low,
        skin_counts=getattr(args, '_skin_counts', None) or None,
        limbus_trust_ratio=args.limbus_trust,
        limbus_gate_kind={'%d_%s' % (i + int(shot.source_start_frame), sd):
                          (v[4] if len(v) > 4 else 'gate') for (i, sd), v in limbus_fix.items()},
        refine_limbus=None if refine_report is None else dict(
            span_px=args.refine_span, step_px=args.refine_step, low_radii=args.refine_low,
            min_shift_px=args.refine_min_shift, min_gain=args.refine_min_gain,
            min_ncc=args.refine_min_ncc, min_px=args.refine_min_px, moved=refine_report),
        implementation='gpu', device=(torch.cuda.get_device_name(0) if DEV.type == 'cuda'
                                      else 'cpu'),
        code_sha256=r32.sha(__file__), gpu_primitives_sha256=r32.sha(here / 'eye_gpu.py'),
        reference_code_sha256=r32.sha(here / 'eye_soft_removal.py'),
        atlas_code_sha256=r32.sha(atl.__file__),
        limbus_code_sha256=r32.sha(here / 'limbus_fit.py'),
        track_sha256=r32.sha(shot.relight / 'face_landmarks.npz'),
        seconds=round(time.time() - T0, 1))
    r32.write_json(root / 'removal.json', record)
    r32.write_json(shot.log(args.name + '.json'), record)


def dump_state(root, groups, fitted, sigma, ceilings, lids):
    """diag.npz: what the pass believed about every eye it treated (see --dump)."""
    keys, win, ceil, lidc, status, floor = [], [], [], [], [], []
    planes = {k: [] for k in ('w_final', 'w_fill', 'wb', 'visible', 'support')}
    codes = {None: 0, 'shade': 1}
    for side in r32.SIDES:
        if fitted[side] is None or sigma[side] is None:
            continue
        for g in groups[side]:
            if not hasattr(g, 'w_final'):
                continue
            wf = (quantize(g.w_final) * 255).round().to(torch.uint8).cpu().numpy()
            wl = (quantize(g.w_fill) * 255).round().to(torch.uint8).cpu().numpy()
            wbb = (torch.clamp(g.wb, 0, 1) * 255).round().to(torch.uint8).cpu().numpy()
            vis = g.visible.cpu().numpy()
            sup = g.support.cpu().numpy()
            for n, e in enumerate(g.eyes):
                keys.append((e.i, 0 if side == 'left' else 1))
                win.append((e.x0, e.y0, e.w, e.h, float(e.limbus[0]), float(e.limbus[1]), e.ax))
                c = ceilings[side].get(e.i)
                ceil.append(np.full(e.w, np.nan, np.float32) if c is None else c)
                lid = lids.get((e.i, side)) or {}
                up, lo = lid.get('upper'), lid.get('lower')
                lidc.append(list(up if up is not None else (np.nan,) * 3)
                            + list(lo if lo is not None else (np.nan,) * 3)
                            + list(lid.get('centre', (np.nan, np.nan))))
                status.append(codes.get(g.status[n], 0))
                fr = getattr(g, 'floor_rows', None)
                fl = (np.full(e.w, np.nan, np.float32) if fr is None
                      else fr[n].cpu().numpy().astype(np.float32))
                floor.append(np.where(np.isfinite(fl), fl, np.nan).astype(np.float32))
                for name, arr in (('w_final', wf), ('w_fill', wl), ('wb', wbb),
                                  ('visible', vis), ('support', sup)):
                    planes[name].append(arr[n].ravel())
    np.savez_compressed(
        root / 'diag.npz', keys=np.array(keys, np.int32), window=np.array(win, np.float64),
        ceiling=np.concatenate(ceil), status=np.array(status, np.int8),
        lids=np.array(lidc, np.float64), floor_rows=np.concatenate(floor),
        **{k: np.concatenate(v) for k, v in planes.items()})
    log('dumped', len(keys), 'eyes to diag.npz')


def load_diag(path):
    """Read diag.npz back into {(frame, side): dict of per-eye arrays}."""
    z = np.load(path)
    out = {}
    po = co = 0
    for n, (i, s) in enumerate(z['keys']):
        x0, y0, w, h, lx, ly, ax = z['window'][n]
        w, h = int(w), int(h)
        rec = dict(x0=int(x0), y0=int(y0), w=w, h=h, limbus=(lx, ly), ax=ax,
                   ceiling=z['ceiling'][co:co + w], lids=z['lids'][n],
                   status=int(z['status'][n]),
                   floor_rows=(z['floor_rows'][co:co + w] if 'floor_rows' in z.files
                               else np.full(w, np.nan, np.float32)))
        for k in ('w_final', 'w_fill', 'wb', 'visible', 'support'):
            rec[k] = z[k][po:po + w * h].reshape(h, w)
        po += w * h
        co += w
        out[(int(i), 'left' if s == 0 else 'right')] = rec
    return out


# =========================================================================== main
def main():
    ap = esr.build_parser()
    ap.add_argument('--io-workers', type=int, default=12,
                    help='threads for EXR reading and PNG writing (I/O, not maths)')
    ap.add_argument('--dump', action='store_true',
                    help="write diag.npz: every treated eye's window, lid fits, lash line, "
                         "fill / softbox / final weights and status, for the audits")
    ap.add_argument('--lid-claim', type=float, default=0.0,
                    help="limbus radii of open iris above the centre that a fitted upper lid "
                         "may claim when the image shows NO lash line; above it the lid is "
                         "not a bound and the eye gets only the softbox's region. 0 is off. "
                         "See final_weights().")
    ap.add_argument('--skin-gate', type=int, default=0,
                    help="pixels of the fill over HER skin or sclera (source brighter than "
                         "halfway between her iris and her lid skin, outside the softbox's "
                         "region) at which an eye gets only the softbox's region. 0 is off.")
    ap.add_argument('--skin-gate-low', type=int, default=0,
                    help="with --skin-gate: the gate spreads in time from a gated eye to its "
                         "neighbours while they still show at least this many skin pixels "
                         "under the fill (hysteresis, so a run is treated one way). 0 is a "
                         "per-frame gate. See skin_runs().")
    ap.add_argument('--lid-floor', action='store_true',
                    help="round 40: bound the fill from below by the lower lid where it "
                         "covers her iris -- the eyelid track's `lower_iris` fits "
                         "(eyelid_fit.py --lower-iris-edge), bridged and median-of-3 filtered "
                         "in time like the lids. The mirror of the lash line.")
    ap.add_argument('--floor-margin', type=float, default=0.0,
                    help='native px to pull the floor up (positive) or let it down')
    ap.add_argument('--limbus-trust', type=float, default=0.0,
                    help="with --limbus-gate: a frame whose limbus fit reports a centre "
                         "standard error above this multiple of its eye's local median is "
                         "re-derived from the trusted frames either side before the one-frame "
                         "gate runs, and never serves as a neighbour. 0 is round 37's gate. "
                         "See gate_limbus_trusted().")
    ap.add_argument('--refine-limbus', action='store_true',
                    help="after building the canonical eye, re-place every eye where the "
                         "canonical fits her iris clearly better than at the tracked limbus, "
                         "then run the passes again (the canonical is kept). See "
                         "refine_limbus().")
    ap.add_argument('--refine-span', type=float, default=8.0, help='search half-width, px')
    ap.add_argument('--refine-step', type=float, default=0.5, help='search step, px')
    ap.add_argument('--refine-low', type=float, default=-0.3,
                    help='compare only rows below this many radii above the centre')
    ap.add_argument('--refine-min-shift', type=float, default=1.5,
                    help='px the canonical must want to move an eye before it is moved')
    ap.add_argument('--refine-min-gain', type=float, default=0.10,
                    help='NCC the canonical must gain there over the tracked position')
    ap.add_argument('--refine-min-ncc', type=float, default=0.6)
    ap.add_argument('--refine-min-px', type=int, default=100)
    ap.add_argument('--refine-chunk', type=float, default=16e6,
                    help='candidate samples per GPU batch (memory)')
    args = ap.parse_args()
    if args.canonical_votes <= 0 and args.canonical_votes_frac <= 0:
        raise SystemExit('eye_removal_gpu.py needs --canonical-votes or --canonical-votes-frac')
    check_supported(args)
    shot = from_args(args)
    log('device', torch.cuda.get_device_name(0) if DEV.type == 'cuda' else 'cpu')
    # ---- scalar setup: identical to eye_soft_removal.main -------------------------------
    track = LimbusTrack.load(shot.relight / args.limbus_track)
    meta = json.loads((shot.relight / 'face_landmarks.json').read_text())
    topology = meta['topology']
    with np.load(shot.relight / 'face_landmarks.npz') as z:
        landmarks = {k: z[k] for k in ('points', 'found', 'eyelid_aperture')}
    corners = esr.eye_corners_mid(landmarks['points'], landmarks['found'], topology)
    open_arc = {}
    limbus_payload = json.loads((shot.relight / args.limbus_track).read_text(encoding='utf-8'))
    for row in limbus_payload['rows']:
        for e in row['eyes']:
            if e.get('open_arc') is not None:
                open_arc[(int(row['frame']), e['side'])] = float(e['open_arc'])
    limbus_fix = {}
    if args.limbus_gate > 0 and args.limbus_trust > 0:
        se = {}
        for row in limbus_payload['rows']:
            for e in row['eyes']:
                if e.get('fitted'):
                    se[(int(row['frame']), e['side'])] = e.get('centre_se_px')
        limbus_fix = gate_limbus_trusted(track, corners, se, shot.frame_rate, args.limbus_gate,
                                         args.limbus_trust)
        log('R39_LIMBUS_TRUST', {s: sum(1 for k, v in limbus_fix.items()
                                        if k[1] == s and v[4] == 'trust') for s in r32.SIDES},
            'distrusted frames re-derived')
        log('R37_LIMBUS_GATE', {s: sorted((i + int(shot.source_start_frame), round(v[3], 2), v[4][0])
                                          for (i, sd), v in limbus_fix.items() if sd == s)
                                for s in r32.SIDES})
    elif args.limbus_gate > 0:
        limbus_fix = esr.gate_limbus(track, corners, shot.frame_count, args.limbus_gate)
        log('R37_LIMBUS_GATE', {s: sorted((i + int(shot.source_start_frame), round(v[3], 2))
                                          for (i, sd), v in limbus_fix.items() if sd == s)
                                for s in r32.SIDES})
    lid_payload = json.loads((shot.relight / args.eyelid_track).read_text(encoding='utf-8'))
    lid_curvature = {s: (lid_payload.get('curvature') or {}).get('%s_upper' % s)
                     for s in r32.SIDES}
    lid_curv_frame = {}
    for row in lid_payload['rows']:
        for e in row['eyes']:
            got = (e.get('curvature') or {}).get('upper')
            if got is not None:
                lid_curv_frame[(int(row['frame']), e['side'])] = float(got)
    lids = esr.load_lids(lid_payload, temporal=True, half=args.lid_window,
                         ref=corners if args.limbus_gate > 0 else None)
    log('R37_LIDS temporal', {s: sum(1 for k in lids if k[1] == s) for s in r32.SIDES})
    floors = None
    if args.lid_floor:
        if not lid_payload.get('lower_iris_edge'):
            raise SystemExit('--lid-floor needs an eyelid track fitted with '
                             'eyelid_fit.py --lower-iris-edge')
        floors = load_floors(lid_payload, corners if args.limbus_gate > 0 else None)
        log('R40_FLOOR', {s: sum(1 for k, v in floors.items() if k[1] == s and v.get('coef'))
                          for s in r32.SIDES}, 'eyes with a lower lid over the iris fitted')
    if Path(args.name).name != args.name:
        raise ValueError('Expected a directory name, not a path')
    root = shot.composite / args.name
    for sub in ('source', 'removal', 'weight', 'sheets'):
        (root / sub).mkdir(parents=True, exist_ok=True)
    indices = list(range(args.start, args.start + args.count))
    if min(indices) < 0 or max(indices) >= shot.frame_count:
        raise ValueError('Requested interval leaves the shot')
    atlas_grid_config = atl.configure(args.atlas_reach, args.atlas_cells)
    log('R35_ATLAS_GRID', atlas_grid_config)
    u_np, v_np = atl.atlas_grid()
    u_t, v_t = as_t(u_np), as_t(v_np)

    # ---- geometry: every eye's window before any pixel is read -------------------------
    probe = cv2.imread(str(shot.fg(indices[0])), cv2.IMREAD_UNCHANGED)
    H, W = probe.shape[:2]

    def make_eyes(moved):
        """Every eye's scalar geometry; `moved[(i, side)]` re-places a limbus centre."""
        eyes = {}
        for i in indices:
            if not (landmarks['found'][i] and np.isfinite(landmarks['points'][i]).all()):
                continue
            for side in r32.SIDES:
                override = track.get(i, side)
                if (i, side) in limbus_fix:
                    override = limbus_fix[(i, side)][:3]
                if override is None:
                    continue
                if (i, side) in moved:
                    override = (tuple(moved[(i, side)]), override[1], override[2])
                lid = lids.get((i, side))
                poly = np.asarray(landmarks['points'][i])[topology['eyelid_' + side], :2]
                e = scalar_eye(i, side, override, lid, esr.lid_inset_for(lid, args), poly, (H, W))
                if e is not None:
                    eyes[(i, side)] = e
        for i in indices:
            lst = [eyes[(i, s)] for s in r32.SIDES if (i, s) in eyes]
            if len(lst) == 2:
                a, b = lst
                if not (a.x0 + a.w <= b.x0 or b.x0 + b.w <= a.x0
                        or a.y0 + a.h <= b.y0 or b.y0 + b.h <= a.y0):
                    raise ValueError('eye windows overlap at frame %d; the port assumes they '
                                     'do not' % i)
        return eyes

    eyes = make_eyes({})
    log('geometry', len(eyes), 'eye windows')

    # ---- frames: read, check, 8-bit source on the GPU, keep the eye windows -------------
    # Each eye's window is kept with a MARGIN px border, so an eye re-placed later
    # (--refine-limbus) is cut from memory rather than read again.
    MARGIN = 12
    writer = Writer(args.io_workers)
    frames8 = {}
    shas = {}
    bigs = {}
    with ThreadPoolExecutor(max_workers=args.io_workers) as pool:
        futures = {i: pool.submit(read_exr, shot.fg(i)) for i in indices}
        for i in indices:
            rgb, digest = futures.pop(i).result()
            shas[i] = digest
            t = torch.from_numpy(rgb).to(DEV)
            h16 = t.half()
            if not torch.equal(h16.float(), t):
                raise ValueError('frame %d is not exactly representable in float16' % i)
            src8 = G.rgb8_half(h16).cpu().numpy()
            frames8[i] = src8
            writer.write(root / 'source' / ('%06d.png' % i), src8[..., ::-1])
            for side in r32.SIDES:
                e = eyes.get((i, side))
                if e is None:
                    continue
                bx0, by0 = max(0, e.x0 - MARGIN), max(0, e.y0 - MARGIN)
                bx1, by1 = min(W, e.x0 + e.w + MARGIN), min(H, e.y0 + e.h + MARGIN)
                bigs[(i, side)] = (h16[by0:by1, bx0:bx1].clone(), bx0, by0)
            del t, h16
    log('frames read', len(indices), 'eye crops', len(bigs))

    def cut(e):
        big, bx0, by0 = bigs[(e.i, e.side)]
        ox, oy = e.x0 - bx0, e.y0 - by0
        if ox < 0 or oy < 0 or ox + e.w > big.shape[1] or oy + e.h > big.shape[0]:
            raise ValueError('eye %d %s moved outside its kept crop' % (e.i, e.side))
        return big[oy:oy + e.h, ox:ox + e.w].float()

    dropped = {}

    def collect_groups(eyes):
        """Eye frame, polar frame and footprint for every eye, batched per window shape.
        Every eye dropped here is recorded in `dropped` with the stage that dropped it."""
        groups = {s: [] for s in r32.SIDES}
        for side in r32.SIDES:
            shapes = {}
            for (i, sd), e in sorted(eyes.items()):
                if sd == side:
                    shapes.setdefault((e.h, e.w), []).append(e)
            for shape, lst in sorted(shapes.items()):
                g = Group(side, lst)
                g.patch = torch.stack([cut(e) for e in lst])
                for stage, fn in (('lid_opening_under_90px', eye_frame),
                                  ('inside_under_60px', solve_eye),
                                  ('anomaly_not_found', None)):
                    if fn is None:
                        anomaly, alive = detect_anomaly(g.patch, g.rho, g.theta, g.inside)
                        g.anomaly = anomaly
                        keep = alive.tolist()
                    else:
                        keep = fn(g)
                    for e, k in zip(g.eyes, keep):
                        if not k:
                            dropped[(e.i, side)] = stage
                    g = g.subset(keep)
                    if not g.B:
                        break
                if g.B:
                    groups[side].append(g)
        return groups

    groups = collect_groups(eyes)
    collected = {}
    for side in r32.SIDES:
        rgbs, ws, order = [], [], []
        for g in groups[side]:
            collar = G.dilate(g.anomaly, np.ones((5, 5), bool))
            strict = g.visible & (g.rho < args.sample_reach)
            if args.atlas_lid_inset > 0:
                lidl = [e.lid for e in g.eyes]
                op2 = parabola_opening(g, lidl, [args.lid_inset + args.atlas_lid_inset] * g.B) > 0.5
                has = g.has_lid.view(-1, 1, 1)
                strict = strict & torch.where(has, op2, torch.ones_like(op2))
                no_up = torch.as_tensor([(e.lid is not None and e.lid.get('upper') is None)
                                         for e in g.eyes], device=DEV).view(-1, 1, 1)
                strict = strict & torch.where(no_up, g.yy > g.ly, torch.ones_like(strict))
            usable = strict & ~collar
            rgb, w = sample_eye(g, usable, u_t, v_t)
            rgbs.append(rgb); ws.append(w); order += [e.i for e in g.eyes]
        if not order:
            collected[side] = None
            continue
        perm = np.argsort(order, kind='stable')
        collected[side] = dict(frames=[order[p] for p in perm],
                               rgb=torch.cat(rgbs)[as_t(perm)], w=torch.cat(ws)[as_t(perm)])
        log('R34_COLLECT', side, len(order), 'frames')

    # ---- the canonical eye, once per side ------------------------------------------------
    fitted = {}
    for side in r32.SIDES:
        col = collected[side]
        if col is None or len(col['frames']) < 20:
            fitted[side] = None
            continue
        frames_list = col['frames']
        N = len(frames_list)
        b = build_atlas(col['rgb'], col['w'], args.iterations, args.gain_window)
        atlas, aw, rho = b['atlas'], b['aw'], b['rho']
        registration = dict(du=float(np.median(b['du'])), dv=float(np.median(b['dv'])),
                            scale=float(np.median(b['scale'])),
                            du_p95=float(np.percentile(b['du'], 95)),
                            dv_p95=float(np.percentile(b['dv'], 95)))
        log('R35_FREEZE', side, {k: round(v, 5) for k, v in registration.items()})
        core = rho < min(args.sample_reach, atl.ATLAS_REACH - 0.02)
        holes_before = int((core & ~aw).sum())
        atlas, aw = atl.fill_atlas_holes(atlas, aw, core)
        hi_votes = args.canonical_votes
        lo_votes = 10
        if args.canonical_votes_frac > 0:
            hi_votes = max(int(round(args.canonical_votes_frac * N)), 2)
            lo_votes = max(int(round(hi_votes / 6.0)), 1)
        votes = b['valid'].sum(0).cpu().numpy()
        atlas, symmetry = esr.symmetric_canonical(atlas, votes, rho, v_np, lo_votes=lo_votes,
                                                  hi_votes=hi_votes)
        log('R37_CANONICAL', side, symmetry)
        pupil_before = pupil_after = None
        if args.pupil_target > 0:
            rgb0 = col['rgb']
            lum = rgb0.max(-1).values
            selg = as_t(np.hypot(*atl.atlas_grid()) < args.pupil_target)
            sel = (col['w'] > 0.98) & selg[None]
            cnt = sel.sum((1, 2))
            p5, _ = G.percentile(lum, sel, 5)
            p10, _ = G.percentile(lum, sel, 10)
            use = cnt >= 8
            obs = [float(x) for x in p5[use].cpu().numpy()]
            dark = sel & (lum <= p10.view(-1, 1, 1)) & use.view(-1, 1, 1)
            px = rgb0[dark].cpu().numpy()
            if obs:
                target = float(np.percentile(obs, 5))
                selr = rho < args.pupil_target
                cur = float(np.median(atlas[selr].max(-1))) if selr.any() else 0.0
                if cur > 1e-5 and target < cur:
                    ramp = np.clip((args.pupil_target - rho) /
                                   max(args.pupil_target * 0.45, 1e-6), 0.0, 1.0)
                    k = target / cur
                    if args.pupil_colour and px.size:
                        hue = np.median(px, axis=0)
                        hue = hue / max(float(hue.max()), 1e-6)
                        cur_rgb = np.median(atlas[selr], axis=0)
                        k = np.clip(target * hue / np.maximum(cur_rgb, 1e-6), 0.0, 1.0)
                        pupil_rgb = [round(float(x), 5) for x in target * hue]
                        atlas = atlas * (1.0 - ramp[..., None] * (1.0 - k[None, None, :]))
                    else:
                        pupil_rgb = None
                        atlas = atlas * (1.0 - ramp[..., None] * (1.0 - k))
                    pupil_before, pupil_after = round(cur, 5), round(target, 5)
                    log('R34_PUPIL', side, 'atlas centre', pupil_before, '->', pupil_after,
                        'rgb', pupil_rgb)
        shown = np.clip(linear_to_srgb(np.clip(atlas, 0, 1)), 0, 1)
        shown = np.where(aw[..., None], shown, 0.0)
        cv2.imwrite(str(root / ('canonical_%s.png' % side)),
                    cv2.resize((shown[..., ::-1] * 255).astype(np.uint8), None, fx=4, fy=4,
                               interpolation=cv2.INTER_NEAREST))
        valid = b['valid']
        lum_f = torch.clamp(G.srgb(torch.clamp((b['rgb_w'].to(F64) / torch.clamp_min(
            b['gain'], 1e-3)).max(-1).values, 0, 1)), 0, 1)
        lum_f = torch.where(valid, lum_f, torch.full_like(lum_f, float('nan')))
        med0 = G.nanmedian0(lum_f)
        spread = G.nanmedian0((lum_f - med0[None]).abs())
        np.savez_compressed(root / ('canonical_%s_votes.npz' % side),
                            count=valid.sum(0).cpu().numpy(),
                            spread=np.nan_to_num(spread.cpu().numpy()).astype(np.float32),
                            atlas=atlas.astype(np.float32), aw=aw)
        if atlas.dtype != np.float32:
            raise ValueError('canonical is %s, expected float32' % atlas.dtype)
        fitted[side] = dict(
            atlas=atlas, aw=aw, history=b['history'], symmetry=symmetry,
            frozen_registration=registration,
            atlas_holes_interpolated=holes_before - int((core & ~aw).sum()),
            atlas_holes_remaining=int((core & ~aw).sum()),
            seed_sharpness=b['seed_sharpness'], generated_cells=0,
            pupil_before=pupil_before, pupil_after=pupil_after,
            sharpness=atl.sharpness(atlas, aw & (rho < r32.RHO_MAX)),
            frames=N,
            ncc={f: float(b['ncc'][n]) for n, f in enumerate(frames_list)},
            registered={f: bool(b['registered'][n]) for n, f in enumerate(frames_list)},
            gain_coef={f: b['gain_coef'][n] for n, f in enumerate(frames_list)})
        log('R34_ATLAS_READY', side, N, 'frames')
        del b

    # ---- optional: re-place slid eyes on the canonical, then run the passes on them ------
    refine_report = None
    args._moved = {}
    if args.refine_limbus:
        moved, refine_report = refine_limbus(eyes, cut, fitted, args)
        log('R39_REFINE moved', {s: sum(1 for k in moved if k[1] == s) for s in r32.SIDES},
            'eyes')
        if moved:
            args._moved = {k: [round(v[0] - float(eyes[k].limbus[0]), 2),
                               round(v[1] - float(eyes[k].limbus[1]), 2)]
                           for k, v in moved.items()}
            eyes = make_eyes(moved)
            groups = collect_groups(eyes)

    # ---- prepass: prediction and the per-frame level --------------------------------------
    for side in r32.SIDES:
        fit = fitted[side]
        if fit is None:
            groups[side] = []
            continue
        reg = fit['frozen_registration']
        for g in groups[side]:
            B = g.B
            g.base0, g.support = predict(g, fit['atlas'], fit['aw'], [reg['du']] * B,
                                         [reg['dv']] * B, [reg['scale']] * B)
            near_foot = G.dilate(g.anomaly, np.ones((9, 9), bool))
            ring = g.inside & g.support & ~near_foot & (g.rho < 0.85)
            if args.gain_ring == 'lower':
                lower = ring & (g.yy > g.ly) & (g.rho > 0.40)
                ring = torch.where((lower.sum((1, 2)) >= 40).view(B, 1, 1), lower, ring)
            g.ring = ring
            k0, k0ok = local_gain(g.patch, g.base0, ring)
            g.k0 = k0.cpu().numpy()
            g.k0ok = k0ok.cpu().numpy()
    gain_series = {s: {} for s in r32.SIDES}
    if args.gain_temporal:
        for side in r32.SIDES:
            raw_k = {}
            allf = []
            for g in groups[side]:
                for n, e in enumerate(g.eyes):
                    allf.append(e.i)
                    if g.k0ok[n]:
                        raw_k[e.i] = g.k0[n]
            have = sorted(raw_k)
            for i in sorted(allf):
                if i in raw_k:
                    radius = 0
                    for k_ in range(1, args.level_window + 1):
                        if all(i + s * m in raw_k for m in range(1, k_ + 1) for s in (-1, 1)):
                            radius = k_
                        else:
                            break
                    near = list(range(i - radius, i + radius + 1))
                    gain_series[side][i] = np.median(np.stack([raw_k[j] for j in near]), 0)
                    continue
                prev = [j for j in have if i - 6 <= j < i]
                nxt = [j for j in have if i < j <= i + 6]
                if prev or nxt:
                    pick = [raw_k[prev[-1]]] if prev else []
                    pick += [raw_k[nxt[0]]] if nxt else []
                    gain_series[side][i] = np.mean(np.stack(pick), 0)
            log('R37_GAIN_SERIES', side, 'measured', len(raw_k), 'of', len(allf), 'filled',
                len(gain_series[side]))

    # ---- pass A: the prediction, the residual and ONE sigma per eye -----------------------
    sigma = {}
    for side in r32.SIDES:
        fit = fitted[side]
        scales = []
        for g in groups[side]:
            B = g.B
            ks, use_ramp = [], []
            for n, e in enumerate(g.eyes):
                if args.gain_temporal:
                    k = gain_series[side].get(e.i)
                else:
                    k = g.k0[n] if g.k0ok[n] else None
                ks.append(np.zeros(3, np.float32) if k is None else np.asarray(k, np.float32))
                use_ramp.append(k is None)
            kt = as_t(np.stack(ks))
            base = torch.clamp_min(g.base0 * kt.view(B, 1, 1, 3), 0.0)
            ur = torch.as_tensor(use_ramp, device=DEV)
            g.gain_used = ['ramp_fallback' if r else 'local' for r in use_ramp]
            g.local_gain = [None if r else [round(float(x), 4) for x in k]
                            for r, k in zip(use_ramp, ks)]
            if ur.any():
                reg = fit['frozen_registration']
                coef = as_t(np.stack([fit['gain_coef'].get(e.i, np.zeros((3, 3)))
                                      for e in g.eyes]))
                rb, rs = predict(g, fit['atlas'], fit['aw'], [reg['du']] * B, [reg['dv']] * B,
                                 [reg['scale']] * B, coef=coef)
                base = torch.where(ur.view(B, 1, 1, 1), rb, base)
                g.support = torch.where(ur.view(B, 1, 1), rs, g.support)
            g.base = base
            p64 = g.patch.to(F64)
            g.signed = (p64 - base).max(-1).values
            g.under = (base - p64).max(-1).values
            fs, fsok = residual_scale(g.signed, g.inside)
            g.frame_scale = [float(x) if o else None
                             for x, o in zip(fs.cpu().numpy(), fsok.cpu().numpy())]
            scales += [x for x in g.frame_scale if x is not None]
            g.lid_pos = signed_lid_positive(g)
        sigma[side] = float(np.median(scales)) if scales else None
        log('R34_SIGMA', side, sigma[side],
            'p5', None if not scales else round(float(np.percentile(scales, 5)), 6),
            'p95', None if not scales else round(float(np.percentile(scales, 95)), 6))

    # ---- pass A2: the lash line, then ONE temporal median ----------------------------------
    ceilings = {s: {} for s in r32.SIDES}
    grid_u = np.linspace(-1.25, 1.25, 101)
    for side in r32.SIDES:
        curves, geoms_by = {}, {}
        line_stats = dict(fitted=0, abstained=0, bridged=0)
        detail_keys = []
        for g in groups[side]:
            B = g.B
            lum = display_lum(g.patch)
            g.lum = lum
            raw, rawok = lash_ceiling_raw(g, lum)
            core, cfound = corneal_specular(lum, g.rho)
            colany = core.any(1)
            excl = G.dilate(colany[:, None, :], np.ones((1, 5), bool))[:, 0] & cfound.view(B, 1)
            curv = []
            for e in g.eyes:
                c = lid_curv_frame.get((e.i, side), lid_curvature.get(side))
                if c is None:
                    raise ValueError('no lid curvature for %s %d' % (side, e.i))
                curv.append(c)
            line, lok = lid_line(raw, excl, g, as_t(np.array(curv, np.float64)))
            lok &= rawok
            line = line + float(args.lid_line_margin)
            line_np = line.cpu().numpy()
            lok_np = lok.cpu().numpy()
            for n, e in enumerate(g.eyes):
                detail_keys.append(e.i)
                if not lok_np[n]:
                    line_stats['abstained'] += 1
                    continue
                line_stats['fitted'] += 1
                c = line_np[n]
                u = ((np.arange(e.x0, e.x0 + e.w) - e.limbus[0]) / e.ax).astype(np.float64)
                v = ((c + e.y0 - e.limbus[1]) / e.ay).astype(np.float64)
                ok = np.isfinite(v)
                if ok.sum() < 8:
                    continue
                curve = np.interp(grid_u, u[ok], v[ok], left=np.nan, right=np.nan)
                curve[(grid_u < u[ok].min()) | (grid_u > u[ok].max())] = np.nan
                curves[e.i] = curve
                geoms_by[e.i] = e
        eye_of = {e.i: e for g in groups[side] for e in g.eyes}
        have = sorted(curves)
        for i in sorted(detail_keys):
            if i in curves:
                continue
            prev = [j for j in have if i - 8 <= j < i]
            nxt = [j for j in have if i < j <= i + 8]
            if prev and nxt:
                a, b_ = prev[-1], nxt[0]
                t = (i - a) / float(b_ - a)
                curves[i] = (1 - t) * curves[a] + t * curves[b_]
            elif prev or nxt:
                src_j = prev[-1] if prev else nxt[0]
                here_arc = open_arc.get((i, side))
                src_arc = open_arc.get((src_j, side))
                if (here_arc is None or src_arc is None or src_arc > here_arc + 0.02):
                    continue
                curves[i] = curves[src_j]
            else:
                continue
            geoms_by[i] = eye_of[i]
            line_stats['bridged'] += 1
        log('R37_LID_LINE', side, line_stats)
        for i, curve in sorted(curves.items()):
            radius = 0
            for k_ in range(1, args.lid_line_window + 1):
                if all(i + s * m in curves for m in range(1, k_ + 1) for s in (-1, 1)):
                    radius = k_
                else:
                    break
            stack = [curves[j] for j in range(i - radius, i + radius + 1) if j in curves]
            med = (np.nanmedian(np.vstack(stack), axis=0) if len(stack) > 1 else curve)
            med = np.where(np.isfinite(med), med, curve)
            e = geoms_by[i]
            u = ((np.arange(e.x0, e.x0 + e.w) - e.limbus[0]) / e.ax).astype(np.float64)
            good = np.isfinite(med)
            if good.sum() < 4:
                continue
            back = np.interp(u, grid_u[good], med[good])
            ceilings[side][i] = (back * e.ay + e.limbus[1] - e.y0).astype(np.float32)
        steps = []
        ks = sorted(curves)
        for x, y in zip(ks, ks[1:]):
            if y == x + 1:
                dd = np.abs(curves[y] - curves[x])
                if np.isfinite(dd).any():
                    steps.append(float(np.nanmax(dd)))
        log('R36_CEILING', side, 'frames', len(curves),
            'raw frame-to-frame max (iris radii) p95',
            None if not steps else round(float(np.percentile(steps, 95)), 4),
            'max', None if not steps else round(max(steps), 4))

    # ---- the weight, the gates and the shading, for every eye at once -------------------
    results = {}          # (i, side) -> record, and for repaired eyes the window + weight
    store = {}
    gate_set, skin_counts = None, {}
    if args.skin_gate > 0 and args.skin_gate_low > 0:
        for side in r32.SIDES:
            fit = fitted[side]
            if fit is None or sigma[side] is None:
                continue
            for g in groups[side]:
                skin_counts.update(final_weights(g, side, fit, fit['frozen_registration'],
                                                 sigma[side], args, ceilings[side], lids, {},
                                                 measure=True, floors=floors))
        gate_set = skin_runs(skin_counts, args.skin_gate, args.skin_gate_low)
        log('R39_SKIN_GATE', {s: sum(1 for k in gate_set if k[1] == s) for s in r32.SIDES},
            'eyes gated (hysteresis %d/%d)' % (args.skin_gate, args.skin_gate_low))
    for side in r32.SIDES:
        fit = fitted[side]
        if fit is None or sigma[side] is None:
            continue
        reg = fit['frozen_registration']
        sg = sigma[side]
        for g in groups[side]:
            results.update(final_weights(g, side, fit, reg, sg, args, ceilings[side], lids,
                                         store, gate_set=gate_set, floors=floors))
    log('dry pass', len(store), 'eye-frames stored for the temporal shading')
    shade_filtered = filter_shading(store, args.shading_temporal) if (
        args.shading_temporal > 0 and args.top_shading > 0) else None
    if shade_filtered is not None:
        log('R37_SHADING_FILTERED', len(shade_filtered), 'eye-frames, half-window',
            args.shading_temporal)
    for side in r32.SIDES:
        fit = fitted[side]
        if fit is None or sigma[side] is None:
            continue
        for g in groups[side]:
            blend_group(g, side, sigma[side], args, results, shade_filtered)
    if args.dump:
        dump_state(root, groups, fitted, sigma, ceilings, lids)

    # ---- write -----------------------------------------------------------------------------
    rows, sheet_tiles = [], []
    for i in indices:
        src8 = frames8.pop(i)
        out8 = src8.copy()
        # The weight is only ever non-zero inside the eye windows, so the full-frame arrays
        # are byte copies plus the windows. `combined` is still a full frame because the
        # reported weighted_pixels is numpy's pairwise sum over the frame, reproduced exactly.
        combined = np.zeros((H, W), np.float32)
        weight8 = np.zeros((H, W), np.uint8)
        eyes_rec = []
        for side in r32.SIDES:
            res = results.get((i, side))
            if res is None:
                eyes_rec.append(dict(side=side, repaired=False, reason='no_eye_frame',
                                     drop_stage=dropped.get((i, side), (
                                         'no_limbus_or_window' if (i, side) not in eyes
                                         else 'no_canonical'))))
                continue
            eyes_rec.append(res['record'])
            if res.get('out8') is not None:
                e = res['eye']
                wv = res['w']
                win = e.window
                touched = wv > 0
                out8[win] = np.where(touched[..., None], res['out8'], src8[win])
                combined[win] = np.maximum(combined[win], wv)
                weight8[win] = np.rint(np.clip(combined[win], 0, 1) * 255).astype(np.uint8)
                assert np.array_equal(weight8[win] > 0, combined[win] > 0), \
                    'the written weight and the applied weight disagree'
                assert np.array_equal(out8[win][~touched], src8[win][~touched]), \
                    'source changed outside the weight'
        writer.write(root / 'removal' / ('%06d.png' % i), out8[..., ::-1])
        writer.write(root / 'weight' / ('%06d.png' % i), weight8)
        if landmarks['found'][i] and np.isfinite(landmarks['points'][i]).all():
            sheet_tiles.append(eye_tile8(src8, out8, combined, landmarks['points'][i],
                                         topology, args.sheet_zoom,
                                         shot.source_start_frame + i, eyes_rec))
        else:
            sheet_tiles.append(np.zeros((22 + 34 * 2 * args.sheet_zoom,
                                         2 * 34 * args.sheet_zoom + 8, 3), np.uint8))
        rows.append(dict(frame=i, source_frame=shot.source_start_frame + i,
                         source_sha256=shas[i],
                         changed_pixels=int((combined > 0).sum()),
                         weighted_pixels=round(float(combined.sum()), 3), eyes=eyes_rec))
    sheets = []
    for k in range(0, len(sheet_tiles), args.sheet_frames):
        path = root / 'sheets' / ('contact_%03d.jpg' % indices[k])
        writer.write(path, r32.contact_sheet(sheet_tiles[k:k + args.sheet_frames]),
                     [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        sheets.append(path.name)
    writer.close()
    log('written', len(rows), 'frames')
    args._skin_counts = {'%d_%s' % k: v for k, v in sorted(skin_counts.items())}
    write_record(root, shot, args, rows, sheets, sigma, fitted, lids, lid_curvature,
                 lid_curv_frame, limbus_fix, refine_report)
    log('R34_REMOVE_DONE', len(rows), 'frames,', sum(1 for r in rows if r['changed_pixels']),
        'repaired')


if __name__ == '__main__':
    main()
