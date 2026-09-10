"""Two results in the SAME tile, one above the other, at eye scale, for every frame.

**This is the tool that found round 36's defects, and none of them showed up in a number.**
The artist, after three rounds of being told the metrics said a change was neutral:

    You tend to use your instruments, but I would wish you LOOKED instead. Just split out
    all frames of that video, isolate the eyes, and LOOK.

The point is the SAME TILE. A frame judged on its own reads as fine; the same frame stacked
directly under the previous round's result shows a white block in the corner of her eye, a
grey speck in the pupil, or a washed band across the iris at a glance. Round 36 called
frames clean at 18x in isolation that were plainly wrong in comparison, which is recorded
in `rounds/round36.md` §1 as a calibration warning.

No thresholds, no masks, no overlays -- it crops on the fitted limbus so the eye stays put
while she moves, and otherwise gets out of the way.

    dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/review_r36_ab.py         --shot music_videos/reporter_test/shots/reporter_a --side left         --dirs "r35_final/removal,r36g/removal" --out sheets/ab
"""
import argparse, os, sys
from pathlib import Path
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2, numpy as np
sys.path.insert(0, 'scripts')
from shot_paths import from_args, add_shot_argument
import limbus_fit as lf

ap = argparse.ArgumentParser()
add_shot_argument(ap)
ap.add_argument('--side', default='left')
ap.add_argument('--zoom', type=int, default=9)
ap.add_argument('--cols', type=int, default=10)
ap.add_argument('--rows', type=int, default=3)
ap.add_argument('--dirs', default='r35_final/removal,r36_final/removal')
ap.add_argument('--out', required=True)
a = ap.parse_args()
shot = from_args(a)
track = lf.Track.load(shot.relight / 'limbus_track.json')
dirs = a.dirs.split(',')

tiles = []
for i in range(150):
    got = track.get(i, a.side)
    if got is None:
        continue
    (cx, cy), ax, ay = got[0], float(got[1]), float(got[2])
    cx, cy = float(cx), float(cy); r = max(ax, ay)
    y0, y1 = int(cy - 1.9 * r), int(cy + 1.5 * r)
    x0, x1 = int(cx - 1.8 * r), int(cx + 1.8 * r)
    stack = []
    for d in dirs:
        img = cv2.imread(str(shot.composite / d / ('%06d.png' % i)), cv2.IMREAD_UNCHANGED)
        if img is None or y0 < 0 or x0 < 0 or y1 > img.shape[0] or x1 > img.shape[1]:
            stack = []
            break
        t_ = img[y0:y1, x0:x1]
        if t_.ndim == 2:
            t_ = cv2.cvtColor(t_, cv2.COLOR_GRAY2BGR)
        stack.append(t_)
    if not stack:
        continue
    t = cv2.resize(np.vstack(stack), None, fx=a.zoom, fy=a.zoom,
                   interpolation=cv2.INTER_NEAREST)
    t = cv2.copyMakeBorder(t, 13, 2, 2, 2, cv2.BORDER_CONSTANT, value=(30, 30, 30))
    cv2.putText(t, '%d' % (i + int(shot.source_start_frame)), (3, 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
    tiles.append(t)

h = max(t.shape[0] for t in tiles); w = max(t.shape[1] for t in tiles)
tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, w - t.shape[1],
                            cv2.BORDER_CONSTANT, value=(0, 0, 0)) for t in tiles]
per = a.cols * a.rows
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
for n, k in enumerate(range(0, len(tiles), per)):
    ch = tiles[k:k + per]
    g = [np.hstack(ch[j:j + a.cols]) for j in range(0, len(ch), a.cols)]
    g = [cv2.copyMakeBorder(x, 0, 0, 0, g[0].shape[1] - x.shape[1],
                            cv2.BORDER_CONSTANT, value=(0, 0, 0)) for x in g]
    p = '%s_%s_%02d.jpg' % (a.out, a.side, n)
    cv2.imwrite(p, np.vstack(g), [int(cv2.IMWRITE_JPEG_QUALITY), 96])
    print(p)
