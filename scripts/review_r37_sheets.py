"""Tight eye crops, several results STACKED in one column per frame, many frames per image.

Round 37's main looking tool. Round 36 section 1 found that a defect invisible in one
result alone is obvious stacked under another; this puts SOURCE and any number of results
(and optionally a weight map) in one column per frame, cropped tight on the fitted limbus
at a zoom where a single pixel is visible, and lays N frames side by side -- small enough
images to actually be read, every frame of the shot covered.

    dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/review_r37_sheets.py \
        --shot music_videos/reporter_test/shots/reporter_a --side left \
        --dirs r37w/source,r37k/removal,r37w/removal --labels "SRC|R37k|R37w" \
        --out composite/r37_sheets/left
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
    ap.add_argument('--dirs', required=True, help='comma-separated <composite dir>/<sub>; a '
                                                  'single-channel dir (a weight) is colour-mapped')
    ap.add_argument('--labels', default='', help='PIPE-separated row labels')
    ap.add_argument('--side', default='left')
    ap.add_argument('--frames', default='', help='first-last shot frame indices; default all')
    ap.add_argument('--half', type=int, default=13, help='crop half-size, native px')
    ap.add_argument('--zoom', type=int, default=9)
    ap.add_argument('--cols', type=int, default=8)
    ap.add_argument('--dy', type=int, default=0, help='move the crop down by this many px')
    ap.add_argument('--out', required=True, help='output prefix, relative to the shot dir')
    args = ap.parse_args()
    shot = from_args(args)
    track = lf.Track.load(shot.relight / 'limbus_track.json')
    if args.frames:
        lo, hi = (int(x) for x in args.frames.split('-'))
    else:
        lo, hi = 0, shot.frame_count - 1
    dirs = [shot.composite / d.strip() for d in args.dirs.split(',')]
    labels = args.labels.split('|') if args.labels else [d.parent.name for d in dirs]
    h, z = args.half, args.zoom
    tiles, first = [], []
    for i in range(lo, hi + 1):
        got = track.get(i, args.side)
        if got is None:
            continue
        cx, cy = float(got[0][0]), float(got[0][1])
        col = []
        for d, lab in zip(dirs, labels):
            im = cv2.imread(str(d / ('%06d.png' % i)), cv2.IMREAD_UNCHANGED)
            if im is None:
                t = np.zeros((2 * h * z, 2 * h * z, 3), np.uint8)
            else:
                if im.ndim == 2:
                    im = cv2.applyColorMap(im, cv2.COLORMAP_INFERNO)
                x0, y0 = int(round(cx)) - h, int(round(cy)) - h + args.dy
                crop = im[max(0, y0):y0 + 2 * h, max(0, x0):x0 + 2 * h]
                crop = cv2.copyMakeBorder(crop, 0, 2 * h - crop.shape[0], 0,
                                          2 * h - crop.shape[1], cv2.BORDER_CONSTANT)
                t = cv2.resize(crop, None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)
            cv2.rectangle(t, (0, 0), (t.shape[1], 14), (0, 0, 0), -1)
            cv2.putText(t, '%s %d' % (lab, i + int(shot.source_start_frame)), (3, 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            col.append(t)
        tiles.append(np.vstack(col))
        first.append(i)
    if not tiles:
        raise SystemExit('no tracked frames for %s in %d-%d' % (args.side, lo, hi))
    out = shot.dir / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    for k in range(0, len(tiles), args.cols):
        blk = tiles[k:k + args.cols]
        blk += [np.zeros_like(blk[0])] * (args.cols - len(blk))
        sheet = np.hstack([np.pad(t, ((0, 0), (0, 3), (0, 0))) for t in blk])
        p = out.parent / ('%s_%s_%04d.png' % (out.name, args.side,
                                               first[k] + int(shot.source_start_frame)))
        cv2.imwrite(str(p), sheet)
    print('%d frames, %d sheets -> %s' % (len(tiles), -(-len(tiles) // args.cols), out.parent))


if __name__ == '__main__':
    main()
