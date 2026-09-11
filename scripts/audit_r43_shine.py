"""Round 43: does the shine on her face follow the NEW light, or the old room's?

    dependencies/CorridorKey/.venv/Scripts/python.exe scripts/audit_r43_shine.py --shot <dir> \\
        --dirs r42_wet,r43_delight_comp --out r43_shine_audit.json

Reads the composites only (the artist's acceptance test is the picture), plus the ratio render
for the light level at each station. Per frame, per run, over the forehead, the nose bridge and
the camera-left cheek (mediapipe boxes, skin-eroded):

* ``sheen``  -- p95 / p50 of skin luminance inside the box: how far the brightest tenth stands
  above the region. On a matte face this is texture only and barely moves with the light; a
  specular lobe raises it whatever the level.
* ``level``  -- p50 of the box, and the rendered irradiance ``E_t`` there.

The claim being tested: with the room's shine still in the plate, ``sheen`` is flat across
the tour whatever ``E_t`` does (the ratio scales lobe and skin alike). After removal, the
sheen that remains is the additive term's, which is gated by the rendered irradiance, so it
should track ``E_t``. Reported as the correlation of ``sheen`` with ``log E_t`` per run, and
the sheen at the darkest and brightest quintile of stations.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2
import numpy as np
from shot_paths import add_shot_argument, from_args
from color_pipeline import srgb_to_linear

LUMA = np.array([.2126, .7152, .0722], np.float32)


def boxes(pts, ipd):
    def box(c, rw, rh):
        return (slice(int(c[1] - rh * ipd), int(c[1] + rh * ipd)), slice(int(c[0] - rw * ipd), int(c[0] + rw * ipd)))
    return dict(forehead=box((pts[10] + pts[151]) / 2, 0.18, 0.10), nose_bridge=box(pts[6], 0.06, 0.08),
                cheek_l=box(pts[50], 0.09, 0.07), chin=box(pts[199], 0.10, 0.06))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_shot_argument(ap)
    ap.add_argument('--dirs', required=True)
    ap.add_argument('--render-dir', default='ratio_render')
    ap.add_argument('--out', default='r43_shine_audit.json')
    args = ap.parse_args()
    shot = from_args(args)
    with np.load(shot.relight / 'face_landmarks.npz') as z:
        pts_all, found = z['points'], z['found']
    runs = args.dirs.split(',')
    frames = sorted(int(p.stem) for p in (shot.composite / runs[0]).glob('[0-9]*.png'))
    rows = []
    for f in frames:
        if not found[f]:
            continue
        p = pts_all[f][:, :2]
        ipd = float(np.linalg.norm(p[33] - p[263]))
        bx = boxes(p, ipd)
        E = cv2.imread(str(shot.composite / args.render_dir / ('%06d.exr' % f)), -1)
        E = E[..., 2::-1] @ LUMA
        skin = cv2.imread(str(shot.relight / 'skin_mask' / ('%06d.png' % f)), -1)
        skin = cv2.erode((skin > 32767).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        row = dict(frame=f)
        for name, sl in bx.items():
            m = skin[sl]
            if m.sum() < 30:
                continue
            row['E_%s' % name] = float(np.median(E[sl][m]))
            for run in runs:
                a = cv2.imread(str(shot.composite / run / ('%06d.png' % f)))
                if a is None:
                    continue
                if a.shape[:2] != E.shape[:2]:
                    a = cv2.resize(a, E.shape[1::-1], interpolation=cv2.INTER_AREA)
                lum = srgb_to_linear(a[..., ::-1].astype(np.float32) / 255.0) @ LUMA
                v = lum[sl][m]
                med = float(np.median(v))
                row['%s_%s' % (run, name)] = dict(level=med, sheen=float(np.percentile(v, 95) / max(med, 1e-6)))
        rows.append(row)
    summary = {}
    for name in ('forehead', 'nose_bridge', 'cheek_l', 'chin'):
        Ev = np.array([r['E_%s' % name] for r in rows if 'E_%s' % name in r])
        if len(Ev) < 5:
            continue
        lo, hi = np.percentile(Ev, 20), np.percentile(Ev, 80)
        for run in runs:
            sh = np.array([r['%s_%s' % (run, name)]['sheen'] for r in rows if '%s_%s' % (run, name) in r])
            if len(sh) != len(Ev):
                continue
            summary['%s_%s' % (run, name)] = dict(
                sheen_p50=float(np.median(sh)), sheen_p95=float(np.percentile(sh, 95)),
                sheen_dark_stations=float(np.median(sh[Ev <= lo])), sheen_bright_stations=float(np.median(sh[Ev >= hi])),
                corr_sheen_logE=float(np.corrcoef(sh, np.log(np.maximum(Ev, 1e-6)))[0, 1]),
                E_range_stops=float(np.log2(max(Ev.max(), 1e-6) / max(Ev.min(), 1e-6))))
    out = dict(summary=summary, per_frame=rows, runs=runs)
    (shot.logs / args.out).write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
