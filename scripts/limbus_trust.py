"""Re-derive the limbus centres the fit itself distrusts, before anything is fitted to them.

Round 39. Round 38's e2 (the "double iris", source 241-247 left) and the audit's class B:
the limbus fit (`limbus_fit.py`) converges on the wrong circle for runs of frames -- +6..+8
px sideways at 243-245 left, +8 px at 493-499 left, 5 px at 1050-1052 right -- and reports
it: its centre standard error rises to 3-12x its eye's local median while the frames either
side sit at the median. Everything downstream is anchored on that centre, including
`eyelid_fit.py`, which searches for the lids around it: at 494-499 left the lids fitted
around the slid centre collapsed and the eye pass dropped an open eye for six frames with
the softbox standing (round 38's e4).

A frame whose standard error exceeds `--ratio` times the median of its eye over +/-
`--window` seconds is re-derived from the nearest TRUSTED frames either side (within
`--max-gap` seconds): their offset from the eye corners, interpolated in time, carried by
this frame's corners. The corners move with her head and not with her gaze or lids
(round 37); the offset from them is her gaze, which is smooth over the few frames bridged.
Nothing else changes: a trusted frame keeps its fit exactly, and a distrusted one with no
trusted frame close enough on both sides keeps its fit (the eye pass's gates still see it).

Writes a track in `limbus_fit.py`'s format (`smoothed_centre` replaced, the fit's own
centre kept, `trust` recording what was done), so every consumer reads it unchanged.

    dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/limbus_trust.py \
        --shot music_videos/reporter_test/shots/reporter_full --out limbus_track_t.json
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shot_paths import add_shot_argument, from_args
import eye_reflection_repair as r32
import eye_soft_removal as esr


def rederive(rows, corners, rate, ratio=2.5, window_s=1.0, max_gap_s=0.25, min_move=1.0):
    """Mutates `rows` (limbus_fit rows). Returns {side: [(frame, moved_px), ...]}."""
    half = max(1, int(round(window_s * float(rate))))
    gap = max(1, int(round(max_gap_s * float(rate))))
    done = {}
    for side in r32.SIDES:
        eye = {}
        for row in rows:
            for e in row['eyes']:
                if e['side'] == side and e.get('fitted') and e.get('smoothed_centre'):
                    eye[int(row['frame'])] = e
        frames = sorted(eye)
        se = {i: eye[i].get('centre_se_px') for i in frames}
        trusted = {}
        for i in frames:
            near = [se[j] for j in range(i - half, i + half + 1)
                    if j in se and se[j] is not None]
            med = float(np.median(near)) if near else None
            trusted[i] = (se[i] is not None and med is not None and se[i] <= ratio * med
                          and (i, side) in corners)
        good = [i for i in frames if trusted[i]]
        centre = {i: np.asarray(eye[i]['smoothed_centre'], float) for i in frames}
        done[side] = []
        for i in frames:
            if trusted[i] or (i, side) not in corners:
                continue
            prev = [j for j in good if i - gap <= j < i]
            nxt = [j for j in good if i < j <= i + gap]
            if not prev or not nxt:
                eye[i]['trust'] = dict(distrusted=True, rederived=False,
                                       reason='no trusted frame within %d either side' % gap)
                continue
            a, b = prev[-1], nxt[0]
            oa = centre[a] - corners[(a, side)]
            ob = centre[b] - corners[(b, side)]
            t = (i - a) / float(b - a)
            pred = corners[(i, side)] + (1 - t) * oa + t * ob
            moved = float(np.hypot(*(pred - centre[i])))
            if moved <= min_move:
                eye[i]['trust'] = dict(distrusted=True, rederived=False, moved_px=round(moved, 3))
                continue
            eye[i]['trust'] = dict(distrusted=True, rederived=True, from_frames=[a, b],
                                   fit_centre=[round(float(c), 4) for c in centre[i]],
                                   moved_px=round(moved, 3))
            eye[i]['smoothed_centre'] = [round(float(pred[0]), 4), round(float(pred[1]), 4)]
            done[side].append((i, round(moved, 2)))
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_shot_argument(ap)
    ap.add_argument('--track', default='limbus_track.json')
    ap.add_argument('--out', default='limbus_track_t.json')
    ap.add_argument('--ratio', type=float, default=2.5)
    ap.add_argument('--window', type=float, default=1.0, help='seconds')
    ap.add_argument('--max-gap', type=float, default=0.25, help='seconds')
    args = ap.parse_args()
    shot = from_args(args)
    data = json.loads((shot.relight / args.track).read_text(encoding='utf-8'))
    meta = json.loads((shot.relight / 'face_landmarks.json').read_text())
    with np.load(shot.relight / 'face_landmarks.npz') as z:
        corners = esr.eye_corners_mid(z['points'], z['found'], meta['topology'])
    done = rederive(data['rows'], corners, shot.frame_rate, args.ratio, args.window,
                    args.max_gap)
    data['trust'] = dict(source_track=args.track, source_sha256=r32.sha(shot.relight / args.track),
                         ratio=args.ratio, window_s=args.window, max_gap_s=args.max_gap,
                         rederived={s: len(v) for s, v in done.items()},
                         code_sha256=r32.sha(Path(__file__)))
    r32.write_json(shot.relight / args.out, data)
    for s, v in done.items():
        print(s, len(v), 'rederived:', v)
    print('wrote', shot.relight / args.out)


if __name__ == '__main__':
    main()
