"""Round 44: face-scale stills of several runs side by side, one row per frame, labelled.

    dependencies/CorridorKey/.venv/Scripts/python.exe scripts/review_r44_face.py --shot <dir> \\
        --dirs r43_nospec,r44_meso --labels "R43 nospec|R44" --frames 73,30,97 --out r44_face_73.png

The crop is centred on the mediapipe landmarks' centroid (a fixed size in IPD units, so the
same crop follows her), magnified nearest-neighbour so a pixel stays a pixel. Round 36's rule:
the previous result and the new one in the SAME tile, at high zoom, and step frames.
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import sys, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np, cv2
from shot_paths import add_shot_argument, from_args


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_shot_argument(ap)
    ap.add_argument('--dirs', required=True)
    ap.add_argument('--labels', default='')
    ap.add_argument('--frames', required=True)
    ap.add_argument('--half-w', type=float, default=1.6, help='crop half-width, IPDs')
    ap.add_argument('--half-h', type=float, default=1.9, help='crop half-height, IPDs')
    ap.add_argument('--zoom', type=float, default=2.0)
    ap.add_argument('--out', required=True, help='PNG path; relative = under composite/panels')
    args = ap.parse_args()
    shot = from_args(args)
    dirs = args.dirs.split(',')
    labels = args.labels.split('|') if args.labels else dirs
    with np.load(shot.relight / 'face_landmarks.npz') as z:
        pts_all = z['points']
    rows = []
    for f in (int(x) for x in args.frames.split(',')):
        p = pts_all[f][:, :2]
        ipd = float(np.linalg.norm(p[33] - p[263]))
        cx, cy = p.mean(0)
        x0, x1 = int(cx - args.half_w * ipd), int(cx + args.half_w * ipd)
        y0, y1 = int(cy - args.half_h * ipd), int(cy + args.half_h * ipd)
        tiles = []
        for d, lab in zip(dirs, labels):
            img = cv2.imread(str(shot.composite / d / ('%06d.png' % f)))
            if img is None:
                raise SystemExit('missing %s/%06d.png' % (d, f))
            h, w = img.shape[:2]
            crop = img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
            crop = cv2.resize(crop, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_NEAREST)
            cv2.putText(crop, '%s  f%d' % (lab, f), (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(crop, '%s  f%d' % (lab, f), (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(crop)
        hmin = min(t.shape[0] for t in tiles)
        rows.append(np.concatenate([t[:hmin] for t in tiles], 1))
    wmin = min(r.shape[1] for r in rows)
    sheet = np.concatenate([r[:, :wmin] for r in rows], 0)
    out = Path(args.out)
    if not out.is_absolute():
        out = shot.composite / 'panels' / out
        out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)
    print(out, sheet.shape)


if __name__ == '__main__':
    main()
