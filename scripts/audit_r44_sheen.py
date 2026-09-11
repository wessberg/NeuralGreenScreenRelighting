"""Round 44: the shine the artist still sees on her LIPS and in her HAIR -- whose light is it?

    dependencies/CorridorKey/.venv/Scripts/python.exe scripts/audit_r44_sheen.py --shot <dir> \\
        --dirs r42_wet,r44_meso --out r44_sheen.json

Same test as audit_r43_shine.py, on two regions round 43 never covered: the lip vermilion
(mediapipe lips_outer minus lips_inner) and the crown of the hair (the matte above the brow,
outside the skin mask, within 0.9 IPD of the forehead centre). Per frame and run, ``sheen`` =
p95 / p50 of luminance in the region, and the correlation of sheen with log E_target there
across the tour. A highlight that belongs to the capture room's softbox is flat across the
tour whatever the bulb does (the ratio scales it with the skin around it); one that belongs to
the new light rises at the bright stations. Also reported: the position of the region's
brightest tenth relative to the region centre (px, in IPD units) at the darkest and brightest
quintile -- a photographed gloss does not move, a rendered one follows the source.
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import sys, json, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np, cv2
from shot_paths import add_shot_argument, from_args
from color_pipeline import srgb_to_linear

LUMA = np.array([.2126, .7152, .0722], np.float32)


def poly_mask(pts, ids, shape):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(pts[ids][:, :2]).astype(np.int32)], 255)
    return m > 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_shot_argument(ap)
    ap.add_argument('--dirs', required=True)
    ap.add_argument('--render-dir', default='ratio_render')
    ap.add_argument('--out', default='r44_sheen.json')
    args = ap.parse_args()
    shot = from_args(args)
    topo = json.loads((shot.relight / 'face_landmarks.json').read_text())['topology']
    with np.load(shot.relight / 'face_landmarks.npz') as z:
        pts_all, found = z['points'], z['found']
    runs = args.dirs.split(',')
    frames = sorted(int(p.stem) for p in (shot.composite / runs[0]).glob('[0-9]*.png'))
    rows = []
    for f in frames:
        if not found[f]:
            continue
        p = pts_all[f]
        ipd = float(np.linalg.norm(p[33, :2] - p[263, :2]))
        alpha = cv2.imread(str(shot.alpha(f)), -1)
        alpha = alpha[..., 0] if alpha.ndim == 3 else alpha
        h, w = alpha.shape
        skin = cv2.imread(str(shot.relight / 'skin_mask' / ('%06d.png' % f)), -1)
        skin = skin[..., 0] if skin.ndim == 3 else skin
        lips = poly_mask(p, topo['lips_outer'], (h, w)) & ~poly_mask(p, topo['lips_inner'], (h, w))
        lips = cv2.erode(lips.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        yy, xx = np.mgrid[0:h, 0:w]
        fc = (p[10, :2] + p[151, :2]) / 2
        brow_y = min(p[105, 1], p[334, 1])
        hair = (alpha > 0.99) & (skin < 32767) & (yy < brow_y - 0.15 * ipd) & (np.hypot(xx - fc[0], yy - fc[1]) < 0.9 * ipd)
        regions = {'lips': (lips, p[13, :2]), 'hair': (hair, fc)}
        E = cv2.imread(str(shot.composite / args.render_dir / ('%06d.exr' % f)), -1)[..., 2::-1] @ LUMA
        row = {'frame': f}
        for name, (m, centre) in regions.items():
            if m.sum() < 40:
                continue
            row['E_%s' % name] = float(np.median(E[m]))
            for run in runs:
                a = cv2.imread(str(shot.composite / run / ('%06d.png' % f)))
                if a is None:
                    continue
                if a.shape[:2] != (h, w):
                    a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
                lum = srgb_to_linear(a[..., ::-1].astype(np.float32) / 255.0) @ LUMA
                v = lum[m]
                med = float(np.median(v))
                top = v >= np.percentile(v, 90)
                cx = float((xx[m][top].mean() - centre[0]) / ipd)
                cy = float((yy[m][top].mean() - centre[1]) / ipd)
                row['%s_%s' % (run, name)] = {'level': med, 'sheen': float(np.percentile(v, 95) / max(med, 1e-6)),
                                              'bright_offset_ipd': [cx, cy]}
        rows.append(row)
    summary = {}
    for name in ('lips', 'hair'):
        Ev = np.array([r['E_%s' % name] for r in rows if 'E_%s' % name in r])
        if len(Ev) < 5:
            continue
        lo, hi = np.percentile(Ev, 20), np.percentile(Ev, 80)
        for run in runs:
            recs = [r['%s_%s' % (run, name)] for r in rows if '%s_%s' % (run, name) in r]
            if len(recs) != len(Ev):
                continue
            sh = np.array([r['sheen'] for r in recs])
            off = np.array([r['bright_offset_ipd'] for r in recs])
            summary['%s_%s' % (run, name)] = {
                'sheen_p50': float(np.median(sh)), 'sheen_dark_stations': float(np.median(sh[Ev <= lo])),
                'sheen_bright_stations': float(np.median(sh[Ev >= hi])),
                'corr_sheen_logE': float(np.corrcoef(sh, np.log(np.maximum(Ev, 1e-6)))[0, 1]),
                'bright_offset_dark_ipd': np.median(off[Ev <= lo], 0).round(3).tolist(),
                'bright_offset_bright_ipd': np.median(off[Ev >= hi], 0).round(3).tolist(),
                'bright_offset_spread_ipd': float(np.std(off, 0).mean())}
    out = {'summary': summary, 'per_frame': rows, 'runs': runs}
    (shot.logs / args.out).write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
