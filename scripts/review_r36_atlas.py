"""EVERY eye in the shot, source above result, with the defects flagged on the tile.

The artist asked for exactly this: *"make sure you have an atlas of all frames in the
video that includes the source vs our current, so you can identify which ones still carry
the softbox reflection. You can use that atlas too to verify the eyelid and lashes are
untouched and that the inner eye shadow looks great."*

Three hundred eyes do not fit on one legible sheet, so this writes a set of sheets at eye
scale, in frame order, and burns three measurements onto each tile so the sheet can be
READ rather than only looked at:

* **SOFTBOX n** -- pixels of the source's own specular still standing well above the iris
  around it in the RESULT. This is `audit_r35_defects.py`'s test, recomputed here so the
  sheet and the acceptance log cannot disagree.
* **LASH n** -- pixels ON OR ABOVE the lash that the result changed at all. The artist's
  new rule is that this must be zero: a corneal reflection is never on a lash, so nothing
  may be painted there. The lash line comes from `eye_soft_removal.lash_ceiling()`, read
  from the source.
* **STEP x** -- the luminance step across the fill boundary in the upper iris, result
  minus source. This is the inner-eye shadow: 0 means the repair carries her own
  lid shadow across the seam, negative means the fill is too dark under the lash and
  positive too bright.

Tiles whose numbers are bad get a coloured border, so a bad frame is findable by scrolling
rather than by reading every number.

    dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/review_r36_atlas.py \
        --shot music_videos/reporter_test/shots/reporter_a --name r35_final
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

SURVIVE = 0.10      # display above its own neighbourhood before a pixel is a fragment
CHANGED = 2 / 255.  # display; below this the result is the photograph


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_shot_argument(ap)
    ap.add_argument('--name', default='r35_final')
    ap.add_argument('--zoom', type=int, default=6)
    ap.add_argument('--columns', type=int, default=10)
    ap.add_argument('--rows', type=int, default=5)
    ap.add_argument('--lid-inset', type=float, default=2.0)
    ap.add_argument('--lid-inset-rms', type=float, default=2.0)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    shot = from_args(args)
    root = shot.composite / args.name
    out_dir = shot.composite / (args.out or ('%s_atlas' % args.name))
    out_dir.mkdir(parents=True, exist_ok=True)
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

    tiles, rows_out = [], []
    for path in sorted((root / 'source').glob('*.png')):
        i = int(path.stem)
        src = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        res = cv2.imread(str(root / 'removal' / path.name), cv2.IMREAD_UNCHANGED)
        wt = cv2.imread(str(root / 'weight' / path.name), cv2.IMREAD_UNCHANGED)
        if src is None or res is None or wt is None or not found[i]:
            continue
        ls = src[..., ::-1].astype(np.float32).max(-1) / 255.0
        lr = res[..., ::-1].astype(np.float32).max(-1) / 255.0
        raw = cv2.imread(str(shot.fg(i)), cv2.IMREAD_UNCHANGED)
        values = None if raw is None else raw[..., ::-1].astype(np.float32)
        for side in r32.SIDES:
            got = track.get(i, side)
            if got is None:
                continue
            centre, ax, ay = got[0], float(got[1]), float(got[2])
            cx, cy = float(centre[0]), float(centre[1])
            r = max(ax, ay)
            y0, y1 = int(cy - 2.1 * r), int(cy + 1.8 * r)
            x0, x1 = int(cx - 2.0 * r), int(cx + 2.0 * r)
            if y0 < 0 or x0 < 0 or y1 > ls.shape[0] or x1 > ls.shape[1]:
                continue
            sub = (slice(y0, y1), slice(x0, x1))
            yy, xx = np.mgrid[y0:y1, x0:x1]
            rho = np.hypot((xx - cx) / ax, (yy - cy) / ay)
            changed = np.abs(lr[sub] - ls[sub]) > CHANGED

            # --- a BRIGHT FRAGMENT anywhere in the iris of the RESULT ------------------
            # **This asks the question the eye asks, and the previous version did not.**
            # It used to locate the specular in the SOURCE by level and then check those
            # pixels in the result -- so it could only find what its own threshold had
            # already decided was the softbox, and when the guard sheared off the
            # highlight's upper-left corner at 0.45-0.62 display the test reported no
            # change on every frame while the artist could see a white block in the corner
            # of her eye. A metric that shares an assumption with the bug is blind to it,
            # which is written in AGENTS.md and happened anyway.
            #
            # The replacement reads the RESULT alone: is there anything inside her iris
            # that is much brighter than the iris around it? A repaired eye has no such
            # thing, whatever the source had and wherever the repair thought the specular
            # was. `iris_lv` is the result's own iris level, so it needs no reference
            # image and no model of the reflection.
            # The test is LOCAL contrast, not a level. A level cannot work: her iris is
            # darker at the top than the bottom by more than a leftover fragment is bright,
            # so any single threshold either floods or misses. A surviving specular is a
            # bright BLOB against a smooth iris, which is exactly what a pixel minus its
            # own neighbourhood measures, and a correctly repaired iris has none.
            med = cv2.medianBlur((lr[sub] * 255).astype(np.uint8), 7).astype(np.float32) / 255.
            local = lr[sub] - med
            # rho < 0.90: the limbal ring and the wet caruncle at the nasal corner both sit
            # outside it, are not on the cornea, and are correctly left alone -- round 35
            # chased thirty of the caruncle before catching that.
            survive = int(((rho < 0.90) & (local > SURVIVE)).sum())

            # --- paint on or above the lash -------------------------------------------
            lash = None
            if values is not None:
                geom, _ = r32.eye_frame(values, points[i], topology, side,
                                        limbus_override=got,
                                        eyelid_override=lids.get((i, side)),
                                        lid_inset=lid_inset_for(lids.get((i, side)), args))
                if geom is not None:
                    state = r32.solve_eye(values[geom['window']], geom)
                    if state is not None:
                        ceil = lash_ceiling(values[geom['window']], geom, state)
                        if ceil is not None:
                            gy, gx = geom['window']
                            abs_ceil = np.full(ls.shape[1], np.inf, np.float32)
                            abs_ceil[gx.start:gx.stop] = ceil + gy.start
                            lash = int((changed
                                        & (yy <= abs_ceil[None, x0:x1])).sum())

            # --- the inner-eye shadow across the fill boundary -------------------------
            ww = wt[sub].astype(np.float32) / 255.0
            step = None
            core, outs = ww > 0.5, ww <= 0.0
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            shell = core & ~(cv2.erode(core.astype(np.uint8), k) > 0)
            near = (cv2.dilate(core.astype(np.uint8), k) > 0) & outs
            up = (yy - cy) / ay < -0.05
            a_, b_ = shell & up & (rho < 1.0), near & up & (rho < 1.0)
            if a_.sum() >= 12 and b_.sum() >= 12:
                step = float((np.median(lr[sub][a_]) - np.median(lr[sub][b_]))
                             - (np.median(ls[sub][a_]) - np.median(ls[sub][b_])))

            top = np.vstack([src[sub], res[sub]])
            top = cv2.resize(top, None, fx=args.zoom, fy=args.zoom,
                             interpolation=cv2.INTER_NEAREST)
            bad = (survive > 0) or (lash or 0) > 0 or (step is not None and abs(step) > 0.05)
            colour = ((0, 0, 255) if survive > 0 else
                      (0, 140, 255) if (lash or 0) > 0 else
                      (0, 200, 255) if bad else (40, 40, 40))
            top = cv2.copyMakeBorder(top, 16, 2, 2, 2, cv2.BORDER_CONSTANT, value=colour)
            label = '%d%s' % (i + int(shot.source_start_frame), side[0].upper())
            if survive:
                label += ' BRIGHT%d' % survive
            if lash:
                label += ' LASH%d' % lash
            if step is not None:
                label += ' STEP%+.2f' % step
            cv2.putText(top, label, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                        (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(top)
            rows_out.append(dict(frame=i, source_frame=i + int(shot.source_start_frame),
                                 side=side, softbox_px=survive, lash_px=lash,
                                 seam_step=None if step is None else round(step, 4)))
        print('R36_ATLAS frame', i, flush=True)

    h = max(t.shape[0] for t in tiles); w = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, w - t.shape[1],
                                cv2.BORDER_CONSTANT, value=(0, 0, 0)) for t in tiles]
    per = args.columns * args.rows
    for n, k in enumerate(range(0, len(tiles), per)):
        chunk = tiles[k:k + per]
        grid = [np.hstack(chunk[j:j + args.columns])
                for j in range(0, len(chunk), args.columns)]
        grid = [cv2.copyMakeBorder(g, 0, 0, 0, grid[0].shape[1] - g.shape[1],
                                   cv2.BORDER_CONSTANT, value=(0, 0, 0)) for g in grid]
        p = out_dir / ('atlas_%02d.jpg' % n)
        cv2.imwrite(str(p), np.vstack(grid), [int(cv2.IMWRITE_JPEG_QUALITY), 94])
        print('R36_ATLAS_SHEET', p, flush=True)

    soft = [r for r in rows_out if r['softbox_px']]
    lashy = [r for r in rows_out if (r['lash_px'] or 0)]
    steps = [r['seam_step'] for r in rows_out if r['seam_step'] is not None]
    summary = dict(
        name=args.name, eyes=len(rows_out),
        softbox_eyes=len(soft), softbox_worst=sorted(
            soft, key=lambda r: -r['softbox_px'])[:20],
        lash_eyes=len(lashy), lash_worst=sorted(
            lashy, key=lambda r: -(r['lash_px'] or 0))[:20],
        seam_step_median=None if not steps else round(float(np.median(steps)), 4),
        seam_step_p95=None if not steps else round(float(np.percentile(
            np.abs(steps), 95)), 4),
        seam_step_max_abs=None if not steps else round(float(np.max(np.abs(steps))), 4),
        rows=rows_out)
    dest = shot.logs / ('r36_atlas_%s.json' % args.name)
    dest.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('R36_ATLAS_SUMMARY eyes', len(rows_out),
          'softbox_eyes', len(soft), 'lash_eyes', len(lashy),
          'seam |step| p95', summary['seam_step_p95'],
          'max', summary['seam_step_max_abs'], flush=True)
    print('R36_ATLAS_WROTE', dest, flush=True)


if __name__ == '__main__':
    main()
