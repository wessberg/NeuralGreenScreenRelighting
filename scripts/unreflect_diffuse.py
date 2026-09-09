"""Run UnReflectAnything's DIFFUSE reconstruction on our own tracked highlight mask.

    dependencies/unreflect-runtime/.venv/Scripts/python.exe -B scripts/unreflect_diffuse.py \\
        --frames <dir of RGB png> --masks <dir of 8-bit mask png> --out <dir>

Round 19 loaded this checkpoint, verified all 601 state entries against the author's
release, and ran ONLY its highlight head -- `unreflect_highlights.py` records
`diffuse_decoder_executed: False` in its own provenance. It then reported that none of
the 12 sparse catchlight anchors passed the author's 0.3 threshold, and stopped there.
That was the wrong conclusion to stop on: the model's own detector missing a 9-pixel
corneal glint says nothing about whether its RECONSTRUCTION can remove one, and the
checkpoint carries both decoders plus the token inpainter.

`UnReflect_Model_TokenInpainter.forward` accepts `inpaint_mask_override`, so the
detector can be replaced wholesale by this project's own tracked footprint -- which is
the point. Rounds 18-34 built a specular tracker for a 24-pixel iris; a general-purpose
highlight head trained on rooms and endoscopy is not going to beat it on this, and it
already measurably does not. The model supplies what we cannot: a learned diffuse
appearance under the highlight.

**This is the right CLASS of model, unlike the two tried before it.** SVOR and
EffectErase are OBJECT removal: the mask tells them "an object is here, delete it and
rebuild the background", so given a mask on an iris they delete the iris -- SVOR
returned blank sclera, EffectErase returned a lump of skin. UnReflectAnything is trained
to take a surface with a specular highlight on it and return the same surface without
it. The mask means "highlight here", not "object here".

It is single-image, so it carries no temporal guarantee at all. That is supplied
downstream by pooling its output into the round-33 registered sequence atlas
(`eye_soft_removal.py --atlas-frames`), which makes the fill one image for the whole
shot. Do not composite its per-frame output directly; that is the crawl the artist
caught in round 34's ProPainter pass.

Writes no production default.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from dotmap import DotMap
import unreflectanything
from unreflectanything.model_ import UnReflectModel

ROOT = Path(__file__).resolve().parents[1]
# The author's CURRENT release. The repository also carries a 1.99 GB
# `full_model_weights_old.pt`; this is the 3.44 GB one, and the hash is asserted rather
# than assumed so a silently swapped checkpoint cannot pass.
CHECKPOINT_SHA256 = '12c0cf4518a39325b7bfe2bed04a96395189f2643c2f90b7b79360a659dc739b'


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_model(device='cuda'):
    weights = ROOT / 'dependencies/unreflect-weights/weights/full_model_weights.pt'
    if file_hash(weights) != CHECKPOINT_SHA256:
        raise ValueError('UnReflectAnything checkpoint differs from the recorded release')
    with torch.serialization.safe_globals([DotMap]):
        checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
    checkpoint.pop('optimizer_state_dict', None)
    config = UnReflectModel._sanitize_embedded_config(checkpoint['config'], False)
    config['MODEL']['RGB_ENCODER']['ENCODER'] = str(ROOT / 'dependencies/unreflect-weights/encoder')
    wrapper = unreflectanything.model(weights=checkpoint, config=config, device='cpu', strict=True)
    model = wrapper._model
    expected = UnReflectModel._strip_key_prefixes(model, checkpoint['model_state_dict'], False)
    actual = model.state_dict()
    assert actual.keys() == expected.keys(), 'state dict keys differ from the release'
    assert all(torch.equal(actual[k], expected[k]) for k in expected), 'weights differ'
    names = sorted(getattr(model, 'decoder_names', []))
    if 'diffuse' not in names:
        raise ValueError('This checkpoint has no diffuse decoder; got %s' % names)
    return model.to(device).eval(), dict(
        checkpoint_sha256=CHECKPOINT_SHA256, strict_state_entries=len(actual),
        exact_loaded_tensors=True, decoders=names,
        model_class=type(model).__name__, torch=torch.__version__)


@torch.inference_mode()
def run(model, rgb, mask, device, size, dilation, threshold=0.3):
    """One frame. `rgb` is HxWx3 uint8, `mask` HxW uint8; returns HxWx3 uint8 diffuse."""
    h, w = rgb.shape[:2]
    x = torch.from_numpy(rgb[..., ::-1].copy()).permute(2, 0, 1)[None].float().to(device) / 255.0
    m = torch.from_numpy((mask > 0).astype(np.float32))[None, None].to(device)
    if (h, w) != (size, size):
        x = F.interpolate(x, size=(size, size), mode='bicubic', align_corners=False).clamp(0, 1)
        m = F.interpolate(m, size=(size, size), mode='nearest')
    out = model({'rgb': x, 'inpaint_mask_override': m,
                 'inpaint_mask_dilation': dilation,
                 'inpaint_mask_threshold': threshold})
    d = out['diffuse'].clamp(0, 1)
    if d.shape[-2:] != (h, w):
        d = F.interpolate(d, size=(h, w), mode='bicubic', align_corners=False).clamp(0, 1)
    arr = (d[0].permute(1, 2, 0).float().cpu().numpy() * 255.0)
    return np.rint(arr).astype(np.uint8)[..., ::-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--frames', required=True)
    ap.add_argument('--masks', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--size', type=int, default=896,
                    help="the model's native input size; the round-19 provenance records 896")
    ap.add_argument('--dilation', type=int, default=5,
                    help='patch-space dilation of the override mask, in pixels')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--blend', type=float, default=2.0,
                    help="feather radius, in pixels, for compositing the model's diffuse "
                         "ONLY inside our mask. The decoder re-renders the WHOLE frame, "
                         "not just the masked region -- skin and lashes come back softer "
                         "and it leaves a grey patch on the lower lid -- so taking its "
                         "output wholesale would trade a highlight for a soft face. 0 "
                         "writes the raw decoder output.")
    args = ap.parse_args()
    frames = sorted(Path(args.frames).glob('*.png'))
    masks = sorted(Path(args.masks).glob('*.png'))
    if len(frames) != len(masks) or not frames:
        raise SystemExit('Need matching frame and mask counts, got %d and %d'
                         % (len(frames), len(masks)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model, prov = load_model(args.device)
    print('UNREFLECT_MODEL', json.dumps(prov), flush=True)
    for k, (f, m) in enumerate(zip(frames, masks)):
        rgb = cv2.imread(str(f))
        msk = cv2.imread(str(m), cv2.IMREAD_GRAYSCALE)
        if rgb is None or msk is None:
            raise SystemExit('Could not read %s / %s' % (f, m))
        if f.name != m.name:
            raise SystemExit('Frame/mask names differ: %s vs %s' % (f.name, m.name))
        diffuse = run(model, rgb, msk, args.device, args.size, args.dilation)
        if args.blend > 0:
            w = (msk > 0).astype(np.float32)
            w = cv2.GaussianBlur(w, (0, 0), args.blend)
            w = np.maximum(w, (msk > 0).astype(np.float32))[..., None]
            diffuse = np.rint(rgb * (1 - w) + diffuse * w).astype(np.uint8)
        cv2.imwrite(str(out / f.name), diffuse)
        if k % 20 == 0:
            print('UNREFLECT_DIFFUSE', k, f.name, flush=True)
    prov.update(frames=len(frames), size=args.size, dilation=args.dilation,
                blend_feather_px=args.blend,
                mask_source='this project\'s tracked highlight footprint, via '
                            'inpaint_mask_override -- NOT the model\'s own highlight head',
                diffuse_decoder_executed=True, production_enabled=False)
    (out / 'unreflect.json').write_text(json.dumps(prov, indent=2, default=str) + '\n',
                                        encoding='utf-8')
    print('UNREFLECT_DONE', len(frames), '->', out, flush=True)


if __name__ == '__main__':
    main()
