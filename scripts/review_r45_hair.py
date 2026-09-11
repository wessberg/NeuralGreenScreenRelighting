"""Round 45: look at the HAIR at hair scale -- several runs side by side, stepped by frame.

    dependencies/CorridorKey/.venv/Scripts/python.exe scripts/review_r45_hair.py --shot <dir> \\
        --dirs r44_m50_s05,r45_rmonly,r45_hair35 --labels "R44|removal only|R45" \\
        --frames 73,97,120 --out r45_crown.png

The crown is a small part of a face-scale sheet, and every defect this round chased -- a band
surviving on the silhouette arc, a bright strip down the parting, a grey haze where a fibre
highlight should be streaks -- is invisible until the crop is the hair and the zoom is 2x or
more. The crop follows HER: it is centred on the face oval and sized in face radii, so the same
frame of hair is compared across runs and across stations.

`--fields <hair_sheen run>` adds a second sheet of what the measurement sees -- the strand
texture, the structure-tensor coherence and the tangent angle -- which is where to look when the
highlight is blotchy rather than streaky.
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import sys, json, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np, cv2
from shot_paths import add_shot_argument, from_args
import hair_region as hr


def label(img, text):
    cv2.putText(img, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_shot_argument(ap)
    ap.add_argument('--dirs', required=True)
    ap.add_argument('--labels', default='')
    ap.add_argument('--frames', required=True)
    ap.add_argument('--side', choices=('crown', 'left', 'right'), default='crown',
                    help='crown spans the whole top of the head; left/right isolate one side, '
                         'which is where a light off the camera axis puts its band')
    ap.add_argument('--zoom', type=float, default=2.2)
    ap.add_argument('--fields', default='', help='a hair_sheen.py run: add a measurement sheet')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    shot = from_args(args)
    topo = json.loads((shot.relight / 'face_landmarks.json').read_text())['topology']
    with np.load(shot.relight / 'face_landmarks.npz') as z:
        pts_all = z['points']
    dirs = args.dirs.split(',')
    labels = args.labels.split('|') if args.labels else dirs
    span = {'crown': (-1.15, 1.15), 'left': (-1.2, -0.05), 'right': (0.05, 1.2)}[args.side]

    rows = []
    for f in (int(x) for x in args.frames.split(',')):
        c, rad, _ = hr.face_frame(pts_all[f], topo)
        x0, x1 = int(c[0] + span[0] * rad), int(c[0] + span[1] * rad)
        y0, y1 = int(c[1] - 1.5 * rad), int(c[1] - 0.2 * rad)
        tiles = []
        for d, lab in zip(dirs, labels):
            img = cv2.imread(str(shot.composite / d / ('%06d.png' % f)))
            if img is None:
                raise SystemExit('missing %s/%06d.png' % (d, f))
            h, w = img.shape[:2]
            crop = img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
            tiles.append(label(cv2.resize(crop, None, fx=args.zoom, fy=args.zoom,
                                          interpolation=cv2.INTER_NEAREST), '%s  f%d' % (lab, f)))
        if args.fields:
            root = shot.composite / args.fields
            F = cv2.imread(str(root / 'field' / ('%06d.png' % f)), -1)
            if F is not None:
                F = F[..., ::-1].astype(np.float32) / 65535.0
                h, w = F.shape[:2]
                for ch, name, hi in ((1, 'strand texture', 0.5), (2, 'coherence', 1.0),
                                     (0, 'tangent angle', 1.0)):
                    v = np.clip(F[..., ch] / hi, 0, 1)[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
                    cm = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
                    tiles.append(label(cv2.resize(cm, None, fx=args.zoom, fy=args.zoom,
                                                  interpolation=cv2.INTER_NEAREST), name))
        hm = min(t.shape[0] for t in tiles)
        rows.append(np.concatenate([t[:hm] for t in tiles], 1))
    wm = min(r.shape[1] for r in rows)
    sheet = np.concatenate([r[:, :wm] for r in rows], 0)
    out = Path(args.out)
    if not out.is_absolute():
        out = shot.composite / 'panels' / out
        out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)
    print(out, sheet.shape)


if __name__ == '__main__':
    main()
