"""Find EVERY frame with a brightened lid or a surviving softbox, from the source alone.

The artist's acceptance test, in his words: *"Look at all frames, and fix all of those
where we either see a brighter lid or part of the softbox reflection."* Eyeballing three
hundred eyes finds the ones that happen to be on screen when you look; this finds all of
them and ranks them, so the fixing is driven by the list rather than by the last
screenshot.

**Both tests are computed from the SOURCE photograph and the RESULT only.** They use none
of the repair's own masks -- not the limbus fit, not the eyelid fit, not the removal
weight. A test that shared those would be blind in exactly the places they are wrong,
which is the mistake this project has made repeatedly and recorded in AGENTS.md.

*Brightened lid.* The repair is allowed to brighten the eye in one place and one place
only: the black undershoot ringing beside the highlight, which round 34 established must
go WITH the highlight because it is that highlight's own coding artefact. So the defect is
"the result is brighter than the source, and the source there is neither the specular nor
its dark ring". Both of those are located in the source by level, independently of what
the repair thought it was doing.

*Surviving softbox.* Where the source carries the specular, the result must be near the
level of the iris around it. Anything still standing well above that iris is the softbox,
whether it is a whole blob or the sliver the mask's edge cut off.

Writes `logs/r35_defects.json` and a contact sheet of the worst offenders, source beside
result, at eye scale.

    dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/audit_r35_defects.py \
        --shot music_videos/reporter_test/shots/reporter_a --name r35_cleanpool
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
from shot_paths import add_shot_argument, from_args
import eye_reflection_repair as r32
import limbus_fit as lf

BRIGHTER = 0.06          # display levels; a rise above this is not noise
RING_REACH = 5.0         # native px the coding undershoot may sit from the specular
RING_LEVEL = 0.22        # display; the undershoot is near legal black
SURVIVE = 0.15           # display above the surrounding iris before a specular counts


def specular_and_ring(luma, disc, ring_ratio):
    """Locate the softbox and its dark ringing in the SOURCE, by level only.

    `disc` stops INSIDE the limbus on purpose. Reaching to 1.05 radii swept in the wet
    caruncle at the nasal corner -- her own tear film, bright, and correctly left alone --
    and reported it as thirty eyes of surviving softbox on frames where nothing survived.
    The corneal specular is on the cornea; the corner is not."""
    outer = ring_ratio & ~disc
    sclera = float(np.percentile(luma[outer], 90)) if outer.sum() > 40 else 0.5
    spec = disc & (luma > max(0.45, sclera + 0.06))
    k = int(2 * np.ceil(RING_REACH) + 1)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    near = cv2.dilate(spec.astype(np.uint8), kern) > 0
    ring = near & (luma < RING_LEVEL)
    return spec, (ring | spec), sclera


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_shot_argument(ap)
    ap.add_argument('--name', default='r35_cleanpool')
    ap.add_argument('--track', default='limbus_track.json')
    ap.add_argument('--out', default='r35_defects.json')
    ap.add_argument('--worst', type=int, default=24)
    ap.add_argument('--zoom', type=int, default=10)
    args = ap.parse_args()
    shot = from_args(args)
    track = lf.Track.load(shot.relight / args.track)
    log = json.loads((shot.logs / (args.name + '.json')).read_text(encoding='utf-8'))
    repaired = {(row['frame'], e['side']): bool(e.get('repaired'))
                for row in log['rows'] for e in row['eyes']}
    root = shot.composite / args.name

    rows, untreated = [], []
    for row in log['rows']:
        i = row['frame']
        src = cv2.imread(str(root / 'source' / ('%06d.png' % i)))
        res = cv2.imread(str(root / 'removal' / ('%06d.png' % i)))
        if src is None or res is None:
            continue
        s = src.max(-1).astype(np.float32) / 255.0
        r = res.max(-1).astype(np.float32) / 255.0
        for side in r32.SIDES:
            got = track.get(i, side)
            if got is None:
                continue
            centre, radius = got[0], got[1]
            yy, xx = np.mgrid[:s.shape[0], :s.shape[1]]
            rho = np.hypot(xx - centre[0], yy - centre[1]) / max(radius, 1e-6)
            disc = rho < 0.98
            spec, protected, sclera = specular_and_ring(s, disc, rho < 1.7)
            if not repaired.get((i, side)):
                if spec.sum() >= 3:
                    untreated.append(dict(frame=i, source_frame=row['source_frame'],
                                          side=side, specular_pixels=int(spec.sum()),
                                          peak=round(float(s[spec].max()), 4),
                                          reason=next((e.get('reason')
                                                       for e in row['eyes']
                                                       if e['side'] == side), None)))
                continue
            # --- brightened lid -----------------------------------------------------
            # "Brighter than the source" alone is not the defect: filling the black
            # ringing beside the highlight brightens it on purpose, and the drawn pupil
            # sits at a level of its own. What the artist is pointing at is specifically
            # **a dark thing in the outer part of the eye that got lighter** -- a lash,
            # a lid margin, the lid's own shadow. Those are darker than the iris around
            # them and they are not near the centre, which separates them from the pupil
            # without using any mask from the repair.
            rise = (r - s) * (rho < 1.25)
            body0 = disc & (rho > 0.45) & ~protected
            level = float(np.median(s[body0])) if body0.sum() > 40 else 0.0
            lash_like = (s < level - 0.08) & (rho > 0.55) & ~protected
            defect = (rise > BRIGHTER) & lash_like
            any_rise = int(((rise > BRIGHTER) & ~protected).sum())
            # --- surviving softbox --------------------------------------------------
            body = disc & (rho > 0.45) & ~protected
            iris = float(np.median(r[body])) if body.sum() > 40 else 0.0
            left = spec & (r > iris + SURVIVE)
            rec = dict(frame=i, source_frame=row['source_frame'], side=side,
                       brightened_pixels=int(defect.sum()),
                       any_brightened_pixels=any_rise,
                       source_iris_level=round(level, 4),
                       brightened_max=round(float(rise[defect].max()), 4)
                       if defect.any() else 0.0,
                       specular_pixels=int(spec.sum()),
                       surviving_pixels=int(left.sum()),
                       surviving_max=round(float((r[left] - iris).max()), 4)
                       if left.any() else 0.0,
                       iris_level=round(iris, 4), sclera_level=round(sclera, 4))
            rec['score'] = round(rec['brightened_max'] * rec['brightened_pixels'] ** 0.5
                                 + rec['surviving_max'] * rec['surviving_pixels'] ** 0.5, 4)
            rows.append(rec)

    bright = [x for x in rows if x['brightened_pixels'] > 0]
    surv = [x for x in rows if x['surviving_pixels'] > 0]
    worst = sorted(rows, key=lambda x: -x['score'])[:args.worst]

    tiles = []
    for rec in worst:
        i, side = rec['frame'], rec['side']
        got = track.get(i, side)
        c = got[0]
        h = 24
        x0, y0 = int(round(c[0])) - h, int(round(c[1])) - h
        pair = []
        for label, sub in (('src', 'source'), ('res', 'removal')):
            im = cv2.imread(str(root / sub / ('%06d.png' % i)))
            crop = im[y0:y0 + 2 * h, x0:x0 + 2 * h]
            if crop.shape[:2] != (2 * h, 2 * h):
                crop = np.zeros((2 * h, 2 * h, 3), np.uint8)
            pair.append(cv2.resize(crop, None, fx=args.zoom, fy=args.zoom,
                                   interpolation=cv2.INTER_NEAREST))
        tile = np.hstack(pair)
        cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, 16), (0, 0, 0), -1)
        cv2.putText(tile, '%d %s  bright %d/%.2f  softbox %d/%.2f'
                    % (rec['source_frame'], side[0].upper(), rec['brightened_pixels'],
                       rec['brightened_max'], rec['surviving_pixels'],
                       rec['surviving_max']), (3, 12), 0, 0.34, (255, 255, 255), 1,
                    cv2.LINE_AA)
        tiles.append(tile)
    if tiles:
        cols = 3
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
        sheet = np.vstack([np.hstack(tiles[k:k + cols])
                           for k in range(0, len(tiles), cols)])
        out = root / 'defects.jpg'
        cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94])
        print('worst %d -> %s' % (len(worst), out))

    def stat(items, key):
        v = np.array([x[key] for x in items], float)
        return (float(np.median(v)), float(np.percentile(v, 95)), float(v.max())) if len(v) else (0., 0., 0.)

    print('%d treated eyes' % len(rows))
    print('  brightened lid : %d eyes  pixels median %.0f p95 %.0f max %.0f  '
          'rise median %.3f p95 %.3f max %.3f'
          % ((len(bright),) + stat(bright, 'brightened_pixels')
             + stat(bright, 'brightened_max')))
    print('  softbox left   : %d eyes  pixels median %.0f p95 %.0f max %.0f  '
          'above iris median %.3f p95 %.3f max %.3f'
          % ((len(surv),) + stat(surv, 'surviving_pixels') + stat(surv, 'surviving_max')))
    print('  untreated eyes that still carry a specular: %d' % len(untreated))
    for u in untreated:
        print('     source %d %-5s %d px peak %.3f  (%s)'
              % (u['source_frame'], u['side'], u['specular_pixels'], u['peak'],
                 u['reason']))
    print('  worst offenders:')
    for rec in worst[:12]:
        print('     source %d %-5s score %.2f  bright %d px max %.3f  softbox %d px '
              'max %.3f' % (rec['source_frame'], rec['side'], rec['score'],
                            rec['brightened_pixels'], rec['brightened_max'],
                            rec['surviving_pixels'], rec['surviving_max']))

    r32.write_json(shot.logs / args.out, dict(
        name=args.name, rows=rows, untreated=untreated,
        brightened_eyes=len(bright), surviving_eyes=len(surv),
        thresholds=dict(brighter=BRIGHTER, ring_reach_px=RING_REACH,
                        ring_level=RING_LEVEL, survive_above_iris=SURVIVE),
        independence='specular and its ringing are located in the SOURCE by level; no '
                     'mask from the repair is used by either test',
        production_enabled=False, artist_accepted=False,
        code_sha256=r32.sha(Path(__file__))))


if __name__ == '__main__':
    main()
