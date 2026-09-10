"""Draw the LASH CEILING on the photograph, so the guard can be judged by eye.

The artist's instruction for round 36: *"the softbox reflection will never be present in
the eyelashes or along the eyelid itself ... we should prevent us from drawing on her
eyelid and lashes"*, and *"it's also correct and important that we do not draw our own
pixels ON or above that eyelid arc"*.

`eye_soft_removal.lash_ceiling()` is the guard that enforces it. This puts it on every
frame at eye scale, in green, beside the fitted lid margin in yellow, so the two can be
compared against her lashes rather than against each other. Round 35's mistake was
sending a video whose frames had not been looked at; this is the looking.

Nothing is written but a contact sheet.

    dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/review_r36_lash.py \
        --shot music_videos/reporter_test/shots/reporter_a
"""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from color_pipeline import linear_to_srgb
from shot_paths import add_shot_argument, from_args
import eye_reflection_repair as r32
from limbus_fit import Track as LimbusTrack
from eye_soft_removal import lash_ceiling, lid_inset_for


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_shot_argument(ap)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--count', type=int, default=150)
    ap.add_argument('--frames', default=None, help='comma-separated indices instead')
    ap.add_argument('--side', default='left')
    ap.add_argument('--zoom', type=int, default=10)
    ap.add_argument('--columns', type=int, default=6)
    ap.add_argument('--lid-inset', type=float, default=2.0)
    ap.add_argument('--lid-inset-rms', type=float, default=2.0)
    ap.add_argument('--out', default='r36_lash')
    args = ap.parse_args()
    shot = from_args(args)
    track = LimbusTrack.load(shot.relight / 'limbus_track.json')
    lids = {}
    for row in json.loads((shot.relight / 'eyelid_track.json')
                          .read_text(encoding='utf-8'))['rows']:
        for e in row['eyes']:
            if e.get('fitted'):
                lids[(int(row['frame']), e['side'])] = dict(
                    centre=e['centre'], upper=(e['upper'] or {}).get('coef'),
                    lower=(e['lower'] or {}).get('coef'),
                    rms=max((e['upper'] or {}).get('rms', 0.0),
                            (e['lower'] or {}).get('rms', 0.0)))
    meta = json.loads((shot.relight / 'face_landmarks.json').read_text())
    topology = meta['topology']
    with np.load(shot.relight / 'face_landmarks.npz') as z:
        points, found = z['points'], z['found']

    indices = ([int(x) for x in args.frames.split(',')] if args.frames
               else list(range(args.start, args.start + args.count)))
    tiles, missing = [], 0
    for i in indices:
        if not found[i]:
            continue
        raw = cv2.imread(str(shot.fg(i)), cv2.IMREAD_UNCHANGED)
        if raw is None:
            continue
        values = raw[..., ::-1].astype(np.float32)
        geom, _ = r32.eye_frame(values, points[i], topology, args.side,
                                limbus_override=track.get(i, args.side),
                                eyelid_override=lids.get((i, args.side)),
                                lid_inset=lid_inset_for(lids.get((i, args.side)), args))
        if geom is None:
            continue
        patch = values[geom['window']]
        state = r32.solve_eye(patch, geom)
        if state is None:
            continue
        ceil = lash_ceiling(patch, geom, state)
        img = np.clip(linear_to_srgb(np.clip(patch, 0, 1)), 0, 1)
        img = (img[..., ::-1] * 255).astype(np.uint8)
        img = cv2.resize(img, None, fx=args.zoom, fy=args.zoom,
                         interpolation=cv2.INTER_NEAREST)
        cfg = lids.get((i, args.side))
        ys, xs = geom['window']
        if cfg and cfg.get('upper'):
            a, b, c = cfg['upper']
            for px in range(img.shape[1]):
                u = xs.start + px / args.zoom - cfg['centre'][0]
                py = int((a * u * u + b * u + c - ys.start) * args.zoom)
                if 0 <= py < img.shape[0]:
                    img[py, px] = (0, 220, 220)
        if ceil is None:
            missing += 1
        else:
            for cx in range(ceil.size):
                if not np.isfinite(ceil[cx]):
                    continue
                py = int(ceil[cx] * args.zoom)
                for px in range(cx * args.zoom, (cx + 1) * args.zoom):
                    if 0 <= py < img.shape[0] and px < img.shape[1]:
                        img[py, px] = (0, 255, 0)
        cv2.putText(img, '%d' % (i + int(shot.source_start_frame)), (3, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    if not tiles:
        raise SystemExit('no tiles')
    h = max(t.shape[0] for t in tiles); w = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0] + 3, 0, w - t.shape[1] + 3,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0)) for t in tiles]
    cols = args.columns
    rows = [np.hstack(tiles[k:k + cols]) for k in range(0, len(tiles), cols)]
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, rows[0].shape[1] - r.shape[1],
                               cv2.BORDER_CONSTANT, value=(0, 0, 0)) for r in rows]
    dest = shot.composite / args.out
    dest.mkdir(parents=True, exist_ok=True)
    per = 5
    for n, k in enumerate(range(0, len(rows), per)):
        p = dest / ('lash_%s_%02d.jpg' % (args.side, n))
        cv2.imwrite(str(p), np.vstack(rows[k:k + per]),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        print('R36_LASH_SHEET', p, flush=True)
    print('R36_LASH', args.side, 'tiles', len(tiles), 'no_ceiling', missing, flush=True)


if __name__ == '__main__':
    main()
