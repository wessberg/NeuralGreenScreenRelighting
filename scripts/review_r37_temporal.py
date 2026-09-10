"""Consecutive frames, STABILISED on the fitted limbus, with their frame-to-frame change.

Flicker is invisible on a contact sheet: every tile is a different moment and her head
moves between them, so a fill that changes colour between two frames reads exactly like
her eye moving. This resamples each frame into the eye's own coordinates -- sub-pixel,
on the fitted limbus -- so consecutive tiles show the same piece of eye, and puts the
AMPLIFIED difference to the previous frame underneath, for the source and for each
result. Change in a result row where the source row is quiet is the repair moving on
its own.

    dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/review_r37_temporal.py \
        --shot music_videos/reporter_test/shots/reporter_a --side left --frames 36-47 \
        --dirs r36g/source,r36g/removal,r37e/removal --labels "SRC|R36g|R37e"
"""
import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shot_paths import add_shot_argument, from_args
import limbus_fit as lf


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_shot_argument(ap)
    ap.add_argument('--side', default='left')
    ap.add_argument('--frames', required=True, help='first-last shot frame indices')
    ap.add_argument('--dirs', required=True)
    ap.add_argument('--labels', default='')
    ap.add_argument('--reach', type=float, default=1.35, help='crop half-size, limbus radii')
    ap.add_argument('--px', type=int, default=150, help='tile size in output pixels')
    ap.add_argument('--gain', type=float, default=5.0, help='difference amplification')
    ap.add_argument('--mode', choices=('rgb', 'chroma', 'lightness'), default='rgb',
                    help="what the difference rows show: the max-channel change, the change "
                         "in CIELAB chroma only (hue/colour flicker), or in L* only")
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    shot = from_args(args)
    track = lf.Track.load(shot.relight / 'limbus_track.json')
    lo, hi = (int(x) for x in args.frames.split('-'))
    dirs = [d.strip() for d in args.dirs.split(',')]
    labels = args.labels.split('|') if args.labels else dirs
    t = np.linspace(-args.reach, args.reach, args.px).astype(np.float32)
    gu, gv = np.meshgrid(t, t)
    prev = {}
    cols = []
    for i in range(lo, hi + 1):
        got = track.get(i, args.side)
        if got is None:
            continue
        (cx, cy), ax, ay = got[0], float(got[1]), float(got[2])
        mx, my = (cx + gu * ax).astype(np.float32), (cy + gv * ay).astype(np.float32)
        rows, diffs = [], []
        for d, lab in zip(dirs, labels):
            img = cv2.imread(str(shot.composite / d / ('%06d.png' % i)))
            tile = cv2.remap(img, mx, my, cv2.INTER_LINEAR)
            if args.mode == 'rgb':
                rep = tile.astype(np.float32)
            else:
                cie = cv2.cvtColor(tile.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
                rep = cie[..., 1:] if args.mode == 'chroma' else cie[..., :1]
                rep = rep * (255.0 / 100.0)
            if d in prev:
                delta = rep - prev[d]
                diff = (np.sqrt((delta ** 2).sum(-1)) if args.mode == 'chroma'
                        else np.abs(delta).max(-1))
                dimg = cv2.applyColorMap(np.clip(diff * args.gain, 0, 255).astype(np.uint8),
                                         cv2.COLORMAP_INFERNO)
            else:
                dimg = np.zeros_like(tile)
            prev[d] = rep
            cv2.putText(tile, '%s %d' % (lab, i + int(shot.source_start_frame)), (3, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
            rows.append(tile)
            diffs.append(dimg)
        cols.append(np.pad(np.vstack(rows + diffs), ((0, 0), (0, 3), (0, 0))))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, np.hstack(cols))
    print(args.out)


if __name__ == '__main__':
    main()
