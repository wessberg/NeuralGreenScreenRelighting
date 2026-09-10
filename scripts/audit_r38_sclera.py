"""Where does a capture-room reflection sit on the WHITE of her eye, outside the iris?

The artist, after round 37: *"there's also a secondary visible reflection in at least one
of her eyes at some angles, here visible outside the iris, in the white of her eye."* The
round 37 construction replaces the iris disc only (the fill stops at 0.99 limbus radii and
the softbox region at ~1.05), so a glint on the sclera is untouched by design. This finds
it in the SOURCE, frame by frame, so the next round knows how often and where.

A glint on the sclera is not "bright": the sclera already reads 0.6-0.8 display. It is a
small blob brighter than the sclera AROUND it -- the same local-contrast test the round 37
audit uses for a leftover softbox, turned outward. Searched only between 1.10 and 2.8
limbus radii and inside the fitted eye opening (the lid fit, pulled in 1 px), so her lid
skin and the tear line's specular at the corner are counted separately: a blob touching
the corner of the opening is reported as `corner`, because a wet caruncle glints too and
it is a different surface.

Reports per eye and frame: blob count, largest area, peak excess over the local sclera,
and position in limbus radii and clock angle. Diagnostic only; writes no pixels.

    dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/audit_r38_sclera.py \
        --shot music_videos/reporter_test/shots/reporter_full
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_shot_argument(ap)
    ap.add_argument('--eyelid-track', default='eyelid_track.json')
    ap.add_argument('--excess', type=float, default=0.10,
                    help='display above the local (9x9 median) sclera to count as a glint')
    ap.add_argument('--floor', type=float, default=0.80, help='minimum display level')
    ap.add_argument('--out', default='r38_sclera_audit.json')
    args = ap.parse_args()
    shot = from_args(args)
    track = LimbusTrack.load(shot.relight / 'limbus_track.json')
    lid_rows = json.loads((shot.relight / args.eyelid_track).read_text())['rows']
    lids = {}
    for row in lid_rows:
        for e in row['eyes']:
            if e.get('upper') and e.get('lower'):
                lids[(int(row['frame']), e['side'])] = e
    topology = json.loads((shot.relight / 'face_landmarks.json').read_text())['topology']
    with np.load(shot.relight / 'face_landmarks.npz') as z:
        points, found = z['points'], z['found']

    rows = []
    for i in range(shot.frame_count):
        if not found[i]:
            continue
        raw = cv2.imread(str(shot.fg(i)), cv2.IMREAD_UNCHANGED)
        lum = np.clip(linear_to_srgb(np.clip(raw[..., ::-1].astype(np.float32), 0, 1)),
                      0, 1).max(-1)
        for side in r32.SIDES:
            got = track.get(i, side)
            if got is None:
                continue
            (cx, cy), r = got[0], float(got[1])
            poly = np.asarray(points[i])[topology['eyelid_' + side], :2]
            x0 = int(np.floor(min(poly[:, 0].min(), cx - 3 * r))) - 2
            x1 = int(np.ceil(max(poly[:, 0].max(), cx + 3 * r))) + 3
            y0, y1 = int(cy - 2.2 * r), int(cy + 2.2 * r) + 1
            if x0 < 0 or y0 < 0 or x1 > lum.shape[1] or y1 > lum.shape[0]:
                continue
            win = (slice(y0, y1), slice(x0, x1))
            lw = lum[win]
            yy, xx = np.mgrid[y0:y1, x0:x1]
            rho = np.hypot(xx - cx, yy - cy) / r
            lid = lids.get((i, side))
            if lid is not None:
                opening = r32.parabola_opening(
                    lid['upper']['coef'], lid['lower']['coef'], lid['centre'], win,
                    (float(poly[:, 0].min()), float(poly[:, 0].max())), inset=1.0) > 0.5
            else:
                opening = r32.lid_distance(points[i], topology['eyelid_' + side], win) > 1.0
            region = opening & (rho > 1.10) & (rho < 2.8)
            if region.sum() < 20:
                continue
            med = cv2.medianBlur((lw * 255).astype(np.uint8), 9).astype(np.float32) / 255.
            hot = region & (lw - med > args.excess) & (lw > args.floor)
            if not hot.any():
                continue
            n, lab, stats, cent = cv2.connectedComponentsWithStats(hot.astype(np.uint8))
            edge = cv2.dilate((~opening).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            blobs = []
            for k in range(1, n):
                area = int(stats[k, cv2.CC_STAT_AREA])
                if area > 60:
                    continue
                m = lab == k
                bx, by = cent[k][0] + x0, cent[k][1] + y0
                ang = float(np.degrees(np.arctan2(-(by - cy), bx - cx)))
                blobs.append(dict(area=area, excess=round(float((lw - med)[m].max()), 3),
                                  peak=round(float(lw[m].max()), 3),
                                  rho=round(float(np.hypot(bx - cx, by - cy) / r), 2),
                                  angle_deg=round(ang, 1),
                                  corner=bool((m & edge).any()),
                                  xy=[round(float(bx), 1), round(float(by), 1)]))
            if blobs:
                rows.append(dict(frame=i, source_frame=i + int(shot.source_start_frame),
                                 side=side, blobs=blobs))
        if i % 100 == 0:
            print('R38_SCLERA frame', i, flush=True)

    summary = {}
    for side in r32.SIDES:
        mine = [r for r in rows if r['side'] == side]
        open_ = [r for r in mine if any(not b['corner'] for b in r['blobs'])]
        frames = sorted(r['source_frame'] for r in open_)
        runs = []
        for f in frames:
            if runs and f <= runs[-1][1] + 2:
                runs[-1][1] = f
            else:
                runs.append([f, f])
        summary[side] = dict(frames_with_scleral_glint=len(open_),
                             frames_with_corner_glint_only=len(mine) - len(open_),
                             runs=[r for r in runs if r[1] - r[0] >= 2],
                             isolated_frames=sum(1 for r in runs if r[1] - r[0] < 2))
    dest = shot.logs / args.out
    dest.write_text(json.dumps(dict(summary=summary, rows=rows, excess=args.excess,
                                    floor=args.floor, eyelid_track=args.eyelid_track),
                               indent=1), encoding='utf-8')
    print(json.dumps(summary, indent=1))
    print('R38_SCLERA_WROTE', dest)


if __name__ == '__main__':
    main()
