"""Four numbers per eye, over all 300 eyes, read from the SOURCE and the RESULT.

Round 36's handoff says its numbers are reproducible and its adjectives are not; this is
the numbers half of round 37's review, and it exists because two of round 36's own
instruments were blind in exactly the place the defect was. Everything here is measured
on the photograph and the result; the repair's weight is used only to say which pixels
it claims to have replaced.

* **FRAGMENTS** -- bright blobs left in the iris of the RESULT: a pixel more than 0.10
  display above the median of its own 7x7 neighbourhood, inside rho < 0.90. No reference,
  no model of the reflection. (`review_r36_atlas.py`'s test, recomputed.)
* **UPPER / LOWER LEVEL** -- median display luma of the result minus the source over the
  fully replaced iris above the pupil (rho .40-.95, above the centre) and below it, with
  the source's own highlight and a 3 px collar excluded. The upper number is the LID
  SHADOW: round 36 was rejected on sight for a "light washed band across the top of the
  iris", and a clean canonical iris is lighter than her shadowed one unless the shadow is
  carried. The lower number is the control -- the fill is levelled there.
* **LASH** -- changed pixels above her lash, where the lash is read per column by
  `lash_ceiling()`'s walk ONLY in the columns it can be trusted: not under the highlight
  (it steps over it and lands in the lower iris -- round 36's own lash count was measured
  against that and counted the filled pupil as lash), and not below the eye centre.
  `brightened` is the subset the result made lighter by more than 0.04: a painted lash
  reads as a lighter lash.
* **FLICKER** -- the result sampled on a fixed grid in the eye's own coordinates, frame
  to frame, over cells replaced in both frames, against the SOURCE's own frame-to-frame
  change at the same cells where the source is clean. p95 and max, never a median alone.
  The grid rides the limbus TRACK, so on a frame where the track slid a still fill
  scores as a jump; `departure` (result change minus her change) and
  `audit_r37_flicker.py` on a steady stretch are the better reads.
* **REGION STANDOUT** -- the softbox region's upper half (located in the SOURCE) against
  the same rows just beside it, in the RESULT: how much the repaired spot stands out
  from the lash shadow around it. This is the test that reproduced the artist's R37k
  screenshot at once -- 717 right at +0.22 with the next worst at +0.10.

    dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/audit_r37_eyes.py \
        --shot music_videos/reporter_test/shots/reporter_a --names r36g,r37e
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
from eye_soft_removal import lash_ceiling, corneal_specular, lid_inset_for

SURVIVE = 0.10
CHANGED = 2 / 255.
BRIGHTENED = 0.04
GRID = np.stack(np.meshgrid(np.linspace(-0.9, 0.9, 25), np.linspace(-0.9, 0.9, 25)), -1)


def luma8(img):
    return img[..., ::-1].astype(np.float32).max(-1) / 255.0


def trusted_lash_rows(values, geom, state):
    """Per-column lash rows (patch coordinates) from the walk, only where trustworthy."""
    patch = values[geom['window']]
    raw = lash_ceiling(patch, geom, state, raw=True)
    if raw is None:
        return None
    lum = np.clip(linear_to_srgb(np.clip(patch, 0, 1)), 0, 1).max(-1)
    core = corneal_specular(lum, state['rho'])
    ok = np.isfinite(raw) & (raw <= float(geom['limbus'][1] - geom['window'][0].start))
    if core is not None:
        cols = cv2.dilate(core.any(axis=0).astype(np.uint8)[None, :],
                          np.ones((1, 5), np.uint8))[0] > 0
        ok &= ~cols
    return np.where(ok, raw, np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_shot_argument(ap)
    ap.add_argument('--names', required=True, help='comma-separated composite run names')
    ap.add_argument('--out', default='r37_eyes_audit.json')
    ap.add_argument('--lid-inset', type=float, default=2.0)
    ap.add_argument('--lid-inset-rms', type=float, default=2.0)
    ap.add_argument('--eyelid-track', default='eyelid_track.json',
                    help='relight/<file>: the lid track the audited run was painted with')
    args = ap.parse_args()
    shot = from_args(args)
    names = [n.strip() for n in args.names.split(',') if n.strip()]
    track = LimbusTrack.load(shot.relight / 'limbus_track.json')
    lids = {}
    for row in json.loads((shot.relight / args.eyelid_track).read_text())['rows']:
        for e in row['eyes']:
            if e.get('fitted'):
                lids[(int(row['frame']), e['side'])] = dict(
                    centre=e['centre'], upper=(e['upper'] or {}).get('coef'),
                    lower=(e['lower'] or {}).get('coef'),
                    rms=max((e['upper'] or {}).get('rms', 0.0),
                            (e['lower'] or {}).get('rms', 0.0)))
    topology = json.loads((shot.relight / 'face_landmarks.json').read_text())['topology']
    with np.load(shot.relight / 'face_landmarks.npz') as z:
        points, found = z['points'], z['found']

    frames = sorted(int(p.stem) for p in (shot.composite / names[0] / 'source').glob('*.png'))
    per = {n: [] for n in names}
    samples = {n: {} for n in names}
    for i in frames:
        if not found[i]:
            continue
        raw = cv2.imread(str(shot.fg(i)), cv2.IMREAD_UNCHANGED)
        values = raw[..., ::-1].astype(np.float32)
        ls = np.clip(linear_to_srgb(np.clip(values, 0, 1)), 0, 1).max(-1)
        results = {n: luma8(cv2.imread(str(shot.composite / n / 'removal' / ('%06d.png' % i))))
                   for n in names}
        weights = {n: cv2.imread(str(shot.composite / n / 'weight' / ('%06d.png' % i)),
                                 cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
                   for n in names}
        src8 = luma8(cv2.imread(str(shot.composite / names[0] / 'source' / ('%06d.png' % i))))
        for side in r32.SIDES:
            got = track.get(i, side)
            if got is None:
                continue
            (cx, cy), ax, ay = got[0], float(got[1]), float(got[2])
            r = max(ax, ay)
            y0, y1 = int(cy - 2.1 * r), int(cy + 1.8 * r)
            x0, x1 = int(cx - 2.0 * r), int(cx + 2.0 * r)
            if y0 < 0 or x0 < 0 or y1 > ls.shape[0] or x1 > ls.shape[1]:
                continue
            sub = (slice(y0, y1), slice(x0, x1))
            yy, xx = np.mgrid[y0:y1, x0:x1]
            pu, pv = (xx - cx) / ax, (yy - cy) / ay
            rho = np.hypot(pu, pv)
            core = corneal_specular(ls[sub], rho)
            spec = np.zeros(rho.shape, bool) if core is None else (
                cv2.dilate(core.astype(np.uint8), cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (7, 7))) > 0)
            # The softbox region's UPPER half and the same rows just beside it, for the
            # standout test (see the module docstring).
            reg_up = reg_beside = None
            if core is not None:
                B = cv2.dilate(core.astype(np.uint8), cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (5, 5))) > 0
                brows = np.nonzero(B.any(1))[0]
                reg_up = np.zeros_like(B)
                reg_beside = np.zeros_like(B)
                for yrow in brows[:max(1, len(brows) // 2)]:
                    reg_up[yrow] = B[yrow]
                    cols = np.nonzero(B[yrow])[0]
                    reg_beside[yrow, max(0, cols.min() - 4):max(0, cols.min() - 1)] = True
                    reg_beside[yrow, cols.max() + 2:cols.max() + 5] = True
                reg_beside &= ~B
                if reg_up.sum() < 3 or reg_beside.sum() < 4:
                    reg_up = reg_beside = None
            geom, _ = r32.eye_frame(values, points[i], topology, side, limbus_override=got,
                                    eyelid_override=lids.get((i, side)),
                                    lid_inset=lid_inset_for(lids.get((i, side)), args))
            lash_rows = None
            if geom is not None:
                state = r32.solve_eye(values[geom['window']], geom)
                if state is not None:
                    lr_ = trusted_lash_rows(values, geom, state)
                    if lr_ is not None:
                        gy, gx = geom['window']
                        lash_rows = np.full(ls.shape[1], np.nan, np.float32)
                        lash_rows[gx.start:gx.stop] = lr_ + gy.start
            # fixed eye grid for flicker
            gxp = (cx + GRID[..., 0] * ax).astype(np.float32)
            gyp = (cy + GRID[..., 1] * ay).astype(np.float32)
            src_grid = cv2.remap(src8, gxp, gyp, cv2.INTER_LINEAR)
            spec_full = np.zeros(ls.shape, np.float32)
            spec_full[sub] = spec.astype(np.float32)
            spec_grid = cv2.remap(spec_full, gxp, gyp, cv2.INTER_LINEAR) > 0.01
            for n in names:
                lr = results[n][sub]
                ww = weights[n][sub]
                changed = np.abs(lr - src8[sub]) > CHANGED
                med = cv2.medianBlur((lr * 255).astype(np.uint8), 7).astype(np.float32) / 255.
                # A leftover reflection is bright ON THE IRIS, so its neighbourhood is
                # dark. On a squint the lid skin sits inside rho < 0.90 and its pores are
                # bright on bright; without this the test counted them as fragments.
                frag = int(((rho < 0.90) & (lr - med > SURVIVE) & (med < 0.40)).sum())
                full = (ww >= 0.99) & ~spec
                up = full & (rho > 0.40) & (rho < 0.95) & (pv < -0.10)
                lo = full & (rho > 0.40) & (rho < 0.95) & (pv > 0.30)
                upper = (float(np.median(lr[up]) - np.median(src8[sub][up]))
                         if up.sum() >= 12 else None)
                lower = (float(np.median(lr[lo]) - np.median(src8[sub][lo]))
                         if lo.sum() >= 12 else None)
                lash = bright = None
                if lash_rows is not None:
                    rows_here = lash_rows[None, x0:x1]
                    above = np.isfinite(rows_here) & (yy <= rows_here)
                    lash = int((changed & above).sum())
                    bright = int((above & (lr - src8[sub] > BRIGHTENED)).sum())
                rw = cv2.remap(weights[n], gxp, gyp, cv2.INTER_LINEAR)
                res_grid = cv2.remap(results[n], gxp, gyp, cv2.INTER_LINEAR)
                samples[n][(i, side)] = (res_grid, rw >= 0.99, src_grid, spec_grid)
                standout = (None if reg_up is None else
                            float(np.median(lr[reg_up]) - np.median(lr[reg_beside])))
                per[n].append(dict(frame=i, source_frame=i + int(shot.source_start_frame),
                                   side=side, fragments=frag, upper_level=upper,
                                   lower_level=lower, lash_px=lash, lash_brightened=bright,
                                   region_standout=standout,
                                   treated=bool((ww > 0).any())))
        print('R37_AUDIT frame', i, flush=True)

    def stats(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        a = np.asarray(vals, float)
        return dict(n=int(a.size), median=round(float(np.median(a)), 4),
                    p95_abs=round(float(np.percentile(np.abs(a), 95)), 4),
                    max_abs=round(float(np.max(np.abs(a))), 4))

    summary = {}
    pairs = {}
    for n in names:
        rows = per[n]
        treated = [r for r in rows if r['treated']]
        flick_res, flick_src, follow = [], [], []
        for (i, side), (rg, rok, sg, sspec) in samples[n].items():
            nxt = samples[n].get((i + 1, side))
            if nxt is None:
                continue
            rg2, rok2, sg2, sspec2 = nxt
            m = rok & rok2
            if m.sum() >= 20:
                flick_res.append(float(np.percentile(np.abs(rg2 - rg)[m], 95)))
                pairs.setdefault(n, []).append(dict(frame=i + int(shot.source_start_frame),
                                                    side=side, flicker_p95=round(
                                                        flick_res[-1], 4)))
            c = (~sspec) & (~sspec2)
            if c.sum() >= 20:
                flick_src.append(float(np.percentile(np.abs(sg2 - sg)[c], 95)))
            # How far the result's change departs from HER change, on cells it replaced in
            # both frames and where her photograph is clean in both. A fill frozen while
            # her lid shadow moves scores her whole change here; a fill that crawls on its
            # own scores its crawl. Either is a departure from the photograph.
            mc = m & c
            if mc.sum() >= 20:
                follow.append(float(np.percentile(
                    np.abs((rg2 - rg) - (sg2 - sg))[mc], 95)))
        summary[n] = dict(
            eyes=len(rows), treated=len(treated),
            fragment_eyes=sum(1 for r in rows if r['fragments']),
            fragment_px=sum(r['fragments'] for r in rows),
            fragment_max=max((r['fragments'] for r in rows), default=0),
            upper_level=stats([r['upper_level'] for r in treated]),
            lower_level=stats([r['lower_level'] for r in treated]),
            lash_eyes=sum(1 for r in rows if r['lash_px']),
            lash_px=sum(r['lash_px'] or 0 for r in rows),
            lash_max=max((r['lash_px'] or 0 for r in rows), default=0),
            lash_brightened_eyes=sum(1 for r in rows if r['lash_brightened']),
            lash_brightened_px=sum(r['lash_brightened'] or 0 for r in rows),
            region_standout=stats([r['region_standout'] for r in rows]),
            region_standout_over_005=sum(1 for r in rows
                                         if (r['region_standout'] or 0) > 0.05),
            region_standout_worst=[(r['source_frame'], r['side'],
                                    round(r['region_standout'], 3))
                                   for r in sorted(rows, key=lambda r: -(r['region_standout']
                                                                         or -9))[:8]],
            flicker_result_p95_per_pair=stats(flick_res),
            flicker_source_p95_per_pair=stats(flick_src),
            departure_from_source_change_p95_per_pair=stats(follow))
        print('R37_AUDIT_SUMMARY', n, json.dumps(summary[n]), flush=True)
    dest = shot.logs / args.out
    dest.write_text(json.dumps(dict(summary=summary, rows=per, flicker_pairs=pairs),
                               indent=1), encoding='utf-8')
    print('R37_AUDIT_WROTE', dest, flush=True)


if __name__ == '__main__':
    main()
