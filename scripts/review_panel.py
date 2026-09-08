"""Labelled side-by-side STILLS at a named facial region, cropped from the landmarks.

    dependencies/CorridorKey/.venv/Scripts/python.exe scripts/review_panel.py --shot <dir> \\
        --frame 21 --region mouth --tiles plate,composite/ratio,composite/ratio_r5 --zoom 6

`review_video.py` is stage 12c and is what the artist actually judges: the artefacts that matter
here are temporal and a still cannot show them. This is its companion for the other half of the
job -- arguing about a specific detail in a written round, where a still at 1:1 is worth more
than a video and a whole round can turn on one crop.

It exists because rounds 1-4 built these by hand every time. Round 4's `r4_spill/000021_mouth.png`
had to be RE-REGISTERED by template matching in round 5 before its own numbers could be
recomputed, because nothing recorded which crop it was. Here the crop is derived from the
landmark store and written into the manifest beside the image.

Regions are anatomical, not pixel boxes, so the same command works on a different performer, a
different framing and a different resolution:

    eyes    both eye openings, padded
    mouth   the lips and the perioral skin
    hair    the crown above the brow line
    face    the face oval
    all     the whole matte

Tiles are `plate` (the keyed foreground), `diffuse`/`specular` (the layers from
`separate_specular.py`), or any path under the shot directory holding `%06d.png` or `.exr`.
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import sys, json, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np, cv2
from shot_paths import add_shot_argument, from_args
from color_pipeline import linear_to_srgb, srgb_to_linear
from separate_specular import load_layers

OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
        377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LIPS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
        291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
EYES = [33, 133, 263, 362, 159, 145, 386, 374]
BROW = [105, 334, 107, 336]


def region_box(P, region, alpha, pad=0.35):
    h, w = alpha.shape
    if region == 'all':
        ys, xs = np.nonzero(alpha > 0.5)
        return int(xs.min()), int(ys.min()), int(np.ptp(xs) + 1), int(np.ptp(ys) + 1)
    idx = {'eyes': EYES, 'mouth': LIPS, 'face': OVAL}.get(region)
    if region == 'hair':
        oval = P[OVAL]
        x0, x1 = oval[:, 0].min(), oval[:, 0].max()
        y1 = float(P[BROW][:, 1].mean())
        ys, xs = np.nonzero(alpha > 0.5)
        y0 = float(ys.min())
        pts = np.array([[x0, y0], [x1, y1]])
    elif idx is None:
        raise SystemExit('Unknown --region %s' % region)
    else:
        pts = P[idx][:, :2]
    x0, y0 = pts[:, 0].min(), pts[:, 1].min()
    x1, y1 = pts[:, 0].max(), pts[:, 1].max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bw, bh = (x1 - x0) * (1 + 2 * pad), (y1 - y0) * (1 + 2 * pad)
    # Square-ish tiles read better side by side, and a crop that is 20:1 is unusable.
    bw = bh = max(bw, bh * 1.5)
    bh = bw / 1.5
    X = int(round(max(cx - bw / 2, 0)))
    Y = int(round(max(cy - bh / 2, 0)))
    return X, Y, int(round(min(bw, w - X))), int(round(min(bh, h - Y)))


def load_tile(shot, spec, f, shape):
    if spec in ('plate', 'diffuse', 'specular'):
        D, S = load_layers(shot, f)
        return {'plate': D + S, 'diffuse': D, 'specular': S}[spec]
    d = shot.dir / spec
    for ext, conv in (('.png', True), ('.exr', False)):
        p = d / ('%06d%s' % (f, ext))
        if p.exists():
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            if img.dtype == np.uint8:
                return srgb_to_linear(img[..., 2::-1].astype(np.float32) / 255.0)
            if img.dtype == np.uint16:
                return srgb_to_linear(img[..., 2::-1].astype(np.float32) / 65535.0)
            return img[..., 2::-1].astype(np.float32)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_shot_argument(ap)
    ap.add_argument('--frame', type=int, default=21)
    ap.add_argument('--region', default='face',
                    help='eyes | mouth | hair | face | all, or X,Y,W,H in plate pixels')
    ap.add_argument('--tiles', default='plate,composite/ratio',
                    help='comma-separated: plate, diffuse, specular, or a dir under the shot')
    ap.add_argument('--labels', default='', help='comma-separated; defaults to the tile specs')
    ap.add_argument('--zoom', type=float, default=0,
                    help='0 picks a zoom that makes each tile about --tile-width across')
    ap.add_argument('--tile-width', type=int, default=1000)
    ap.add_argument('--gain', default='', help='comma-separated per-tile linear gain, e.g. for '
                                               'a specular layer that is dark by construction')
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    shot = from_args(args)

    lmk = np.load(shot.relight / 'face_landmarks.npz')
    if not lmk['found'][args.frame]:
        raise SystemExit('No face detected on frame %d; use --region all or X,Y,W,H'
                         % args.frame)
    P = lmk['points'][args.frame]
    alpha = cv2.imread(str(shot.alpha(args.frame)), -1)
    alpha = alpha[..., 0] if alpha.ndim == 3 else alpha
    if ',' in args.region:
        X, Y, W, H = (int(v) for v in args.region.split(','))
    else:
        X, Y, W, H = region_box(P, args.region, alpha)
    z = args.zoom if args.zoom > 0 else max(1.0, round(args.tile_width / max(W, 1)))

    specs = [t.strip() for t in args.tiles.split(',') if t.strip()]
    labels = [t.strip() for t in args.labels.split(',')] if args.labels else specs
    gains = [float(g) for g in args.gain.split(',')] if args.gain else [1.0] * len(specs)
    gains += [1.0] * (len(specs) - len(gains))
    tiles, missing = [], []
    for spec, lab, g in zip(specs, labels, gains):
        img = load_tile(shot, spec, args.frame, alpha.shape)
        if img is None:
            missing.append(spec)
            continue
        crop = img[Y:Y + H, X:X + W] * g
        srgb = (np.clip(linear_to_srgb(np.clip(crop, 0, None)), 0, 1) * 255).astype(np.uint8)
        srgb = cv2.resize(srgb, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)
        txt = lab + ('' if g == 1.0 else '  x%g' % g)
        cv2.putText(srgb, txt, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(srgb, txt, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 1,
                    cv2.LINE_AA)
        tiles.append(srgb)
    if not tiles:
        raise SystemExit('No tiles could be read. Missing: %s' % missing)
    panel = np.concatenate(tiles, 1)
    out = Path(args.out) if args.out else (shot.composite / 'panels' /
                                           ('%06d_%s.png' % (args.frame, args.region)))
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), panel[..., ::-1])
    manifest = {'shot': shot.name, 'frame': args.frame, 'region': args.region,
                'crop_xywh': [X, Y, W, H], 'zoom': z, 'tiles': specs, 'labels': labels,
                'gains': gains, 'missing': missing, 'output': str(out),
                'size': [int(panel.shape[1]), int(panel.shape[0])],
                'why_recorded': 'round 4 shipped panels whose crop was nowhere written down, and '
                                'round 5 had to recover it by template matching before it could '
                                'recompute their numbers'}
    out.with_suffix('.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
