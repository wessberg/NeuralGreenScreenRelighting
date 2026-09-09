"""Score round 33's sequence atlas on withheld pixels and measure its temporal stability.

Two questions, because round 32 failed on the second while passing the first.

**Withheld accuracy.** Reuses round 32's frozen rotation design unchanged so the numbers
are comparable on identical pixels: each repaired eye's real footprint is rotated about
the limbus in the eye's own normalized polar frame, which preserves its size, shape and
radius exactly, and the rotated copy is then withheld and predicted. The controls are
round 32's own -- its completion, the radial profile alone, an annulus median, round
31's flat ring gain -- plus the plate's own angular texture variation as a floor.

The atlas is scored across a TEMPORAL SPLIT: the atlas that predicts a site in the
second half of the shot is built only from the first half, and vice versa. This matters.
Scoring an atlas on frames it pooled would be close to self-prediction, and the
assumption actually being tested is the load-bearing one -- that ONE appearance estimate
is valid across the whole shot rather than only near the frames it came from.

**Temporal stability.** The same measure round 32 was rejected on
(`logs/r32_flicker.json`), so the two are directly comparable, plus a mask-shape measure
the artist's rejection specifically named: he said the result was unstable "in terms of
neither shape nor color". Round 32 reported a median adjacent-frame step BELOW the
source's own control while the result was plainly flickering, so p95, max and the full
range are reported here and the median is never quoted alone.

    dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/audit_r33_repair.py \\
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
import eye_sequence_atlas as atl
import eye_atlas_repair as rep
from audit_r32_repair import (ROTATIONS, annulus_median, display_error, fit_model,
                              ring_gain, rotate_mask)


def roughness(series, window=2):
    """Each frame's departure from a rolling median of the same series.

    Isolates frame-to-frame jitter from smooth change: a fill that drifts as the eye
    turns has low roughness, one that jumps for a single frame has high roughness.
    """
    a = np.asarray(series, float)
    if len(a) < 2 * window + 1:
        return []
    out = []
    for i in range(len(a)):
        lo, hi = max(0, i - window), min(len(a), i + window + 1)
        out.append(float(np.abs(a[i] - np.median(a[lo:hi], axis=0)).mean()))
    return out


def summarize(values):
    a = np.asarray(values, float)
    if not a.size:
        return None
    return dict(n=int(a.size), median=round(float(np.median(a)), 4),
                p95=round(float(np.percentile(a, 95)), 4),
                max=round(float(a.max()), 4))


def collect(shot, topology, landmarks, indices):
    """Per-frame eye crops, footprints and atlas samples -- the repair's own pass 1."""
    u, v = atl.atlas_grid()
    per_side = {s: [] for s in r32.SIDES}
    context = {}
    for i in indices:
        raw = cv2.imread(str(shot.fg(i)), cv2.IMREAD_UNCHANGED)
        if raw is None:
            continue
        values = raw[..., ::-1].astype(np.float32)
        if not (landmarks['found'][i] and np.isfinite(landmarks['points'][i]).all()):
            continue
        for side in r32.SIDES:
            geom, _ = r32.eye_frame(values, landmarks['points'][i], topology, side)
            if geom is None:
                continue
            patch = values[geom['window']]
            state = r32.solve_eye(patch, geom)
            if state is None:
                continue
            anomaly, model, _ = r32.detect_anomaly(patch, state['rho'], state['theta'],
                                                   state['inside'])
            if anomaly is None:
                continue
            collar = cv2.dilate(anomaly.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            rgb, w = atl.sample_eye(patch, geom, state['inside'] & ~collar, u, v)
            per_side[side].append(dict(frame=i, rgb=rgb, w=w,
                                       luma=rgb.max(-1).astype(np.float32),
                                       du=0.0, dv=0.0, scale=1.0, ncc=0.0,
                                       ax=geom['ax'], ay=geom['ay']))
            context[(i, side)] = dict(geom=geom, patch=patch, state=state,
                                      footprint=anomaly, model=model, collar=collar)
        print('R33_AUDIT collect', i, flush=True)
    return per_side, context


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_shot_argument(ap)
    ap.add_argument('--name', default='r33_repair')
    ap.add_argument('--count', type=int, default=150)
    args = ap.parse_args()
    shot = from_args(args)
    root = shot.composite / args.name
    report = json.loads((root / 'repair.json').read_text())
    meta = json.loads((shot.relight / 'face_landmarks.json').read_text())
    topology = meta['topology']
    with np.load(shot.relight / 'face_landmarks.npz') as z:
        landmarks = {k: z[k] for k in ('points', 'found')}
    indices = list(range(args.count))

    per_side, context = collect(shot, topology, landmarks, indices)

    # ---- withheld pixels, across a temporal split -----------------------------------
    keys = ('atlas_split', 'r32_repair', 'profile', 'annulus_median',
            'single_gain', 'texture_floor')
    samples = {s: {k: [] for k in keys} for s in r32.SIDES}
    stats = dict(attempted=0, scored=0, rejected=0, reasons={})
    folds = {}
    for side in r32.SIDES:
        frames = per_side[side]
        if len(frames) < 40:
            continue
        cut = frames[len(frames) // 2]['frame']
        folds[side] = cut
        built = {}
        for tag, subset in (('first', [f for f in frames if f['frame'] < cut]),
                            ('second', [f for f in frames if f['frame'] >= cut])):
            work = [dict(f) for f in subset]
            atlas, aw, hist, _, rho, _ = atl.build(work, 5, 5)
            core = rho < r32.RHO_MAX
            atlas, aw = atl.fill_atlas_holes(atlas, aw, core)
            built[tag] = dict(atlas=atlas, aw=aw)
            print('R33_AUDIT fold', side, tag, len(work), 'frames', flush=True)
        # A frame in the first half is predicted by the atlas built from the second.
        for f in frames:
            other = built['second' if f['frame'] < cut else 'first']
            ctx = context[(f['frame'], side)]
            geom, patch, state = ctx['geom'], ctx['patch'], ctx['state']
            rho, theta, inside = state['rho'], state['theta'], state['inside']
            footprint, collar = ctx['footprint'], ctx['collar']
            # The held atlas has no registration or gain for this frame, because the
            # frame was not in its build. Both are re-fitted against the held atlas from
            # this frame's CLEAN pixels only -- never from the withheld site.
            probe = dict(f)
            probe.update(du=0.0, dv=0.0, scale=1.0)
            best = atl.register(probe['luma'], probe['w'], other['atlas'].max(-1).astype(np.float32),
                                other['aw'].astype(np.float32), rho_core(other), atl.COARSE_SHIFT,
                                atl.COARSE_STEP, atl.SCALES)
            if best is None:
                continue
            probe.update(du=best[1], dv=best[2], scale=best[3])
            for angle in ROTATIONS:
                stats['attempted'] += 1
                site = rotate_mask(footprint, rho, theta, state['centre'], geom['ax'],
                                   geom['ay'], angle,
                                   (geom['window'][1].start, geom['window'][0].start))
                if site is None or not site.any():
                    stats['rejected'] += 1
                    stats['reasons']['empty'] = stats['reasons'].get('empty', 0) + 1
                    continue
                if (site & collar).any() or not inside[site].all():
                    stats['rejected'] += 1
                    stats['reasons']['overlap'] = stats['reasons'].get('overlap', 0) + 1
                    continue
                hidden = footprint | site
                refit = fit_model(patch, rho, theta, inside & ~hidden)
                if refit is None:
                    stats['rejected'] += 1
                    stats['reasons']['nomodel'] = stats['reasons'].get('nomodel', 0) + 1
                    continue
                base = refit['base']
                donor = inside & ~(cv2.dilate(hidden.astype(np.uint8),
                                              np.ones((5, 5), np.uint8)) > 0)
                if donor.sum() < 60:
                    stats['rejected'] += 1
                    stats['reasons']['nodonor'] = stats['reasons'].get('nodonor', 0) + 1
                    continue
                filled, supported = r32.angular_fill(patch - base, rho, theta, site, donor)
                if not supported.all():
                    stats['rejected'] += 1
                    stats['reasons']['unsupported'] = stats['reasons'].get('unsupported', 0) + 1
                    continue
                # Gain for the held atlas, fitted on clean pixels with BOTH the real
                # footprint and the withheld site excluded, so no truth leaks in.
                pred_base, support, (pp, qq) = rep.predict(
                    geom, other['atlas'], other['aw'], probe['du'], probe['dv'],
                    probe['scale'], np.zeros((3, 3)))
                fitmask = inside & ~cv2.dilate(hidden.astype(np.uint8),
                                               np.ones((5, 5), np.uint8)).astype(bool) & support
                coef = atl.robust_gain(patch, pred_base, fitmask, atl.gain_basis(pp, qq))
                if coef is None or not support[site].all():
                    stats['rejected'] += 1
                    stats['reasons']['no_atlas_support'] = \
                        stats['reasons'].get('no_atlas_support', 0) + 1
                    continue
                atlas_pred = (pred_base * atl.apply_gain(coef, atl.gain_basis(pp, qq)))[site]
                flat = ring_gain(patch, site, donor)
                if flat is None:
                    stats['rejected'] += 1
                    stats['reasons']['noring'] = stats['reasons'].get('noring', 0) + 1
                    continue
                truth = patch[site]
                floor = annulus_median(patch, rho, site, donor)
                if not np.isfinite(floor).all():
                    stats['rejected'] += 1
                    stats['reasons']['nofloor'] = stats['reasons'].get('nofloor', 0) + 1
                    continue
                preds = {
                    'atlas_split': atlas_pred,
                    'r32_repair': np.maximum(base[site] + filled, 0.0),
                    'profile': base[site],
                    'annulus_median': floor,
                    'single_gain': np.broadcast_to(flat, truth.shape),
                    # The plate's own angular variation at the same radii: the same
                    # annulus median scored against the truth it did not come from.
                    'texture_floor': floor,
                }
                for k in keys:
                    if k == 'texture_floor':
                        continue
                    samples[side][k].append(display_error(preds[k], truth))
                # The plate's own angular texture variation at the same radii, computed
                # exactly as round 32 did. NOT a second annulus median: an earlier
                # version here passed `donor & ~site`, but `donor` already excludes the
                # site, so it reproduced the annulus_median column digit for digit and
                # reported a floor of 13.3 that the atlas appeared to beat.
                donor_rho, donor_residual = rho[donor], (patch - base)[donor]
                floor_scale = []
                for r0 in rho[site]:
                    band = np.abs(donor_rho - r0) < 0.075
                    if band.sum() >= 4:
                        block = donor_residual[band]
                        floor_scale.append(float(np.median(
                            np.abs(block - np.median(block, axis=0))).mean()))
                if floor_scale:
                    reference = base[site].mean(axis=0)
                    samples[side]['texture_floor'].append(display_error(
                        reference[None] + float(np.mean(floor_scale)), reference[None]))
                stats['scored'] += 1

    withheld = {}
    for side in r32.SIDES:
        if not samples[side]['atlas_split']:
            withheld[side] = None
            continue
        withheld[side] = {k: dict(pixels=int(np.concatenate(samples[side][k]).size),
                                  median_display_error=round(float(np.median(
                                      np.concatenate(samples[side][k]))), 4),
                                  p95=round(float(np.percentile(
                                      np.concatenate(samples[side][k]), 95)), 4))
                          for k in keys}

    # ---- temporal stability, on what the artist actually watches --------------------
    temporal = {}
    for side in r32.SIDES:
        fills, src_fills, steps, controls, ious, sizes = [], [], [], [], [], []
        prev = None
        for row in report['rows']:
            i = row['frame']
            entry = next((e for e in row['eyes'] if e['side'] == side), None)
            if entry is None or not entry.get('repaired'):
                prev = None
                continue
            rep_img = cv2.imread(str(root / 'repair' / ('%06d.png' % i)))
            src_img = cv2.imread(str(root / 'source' / ('%06d.png' % i)))
            mask = cv2.imread(str(root / 'mask' / ('%06d.png' % i)), cv2.IMREAD_GRAYSCALE)
            if rep_img is None or mask is None:
                prev = None
                continue
            ctx = context.get((i, side))
            if ctx is None:
                prev = None
                continue
            window = ctx['geom']['window']
            local = (mask[window] > 0)
            if local.sum() < 3:
                prev = None
                continue
            ys, xs = np.where(local)
            ry, rx = ys + window[0].start, xs + window[1].start
            fill = rep_img[ry, rx][:, ::-1].astype(float).mean(axis=0)
            fills.append(fill)
            src_fills.append(src_img[ry, rx][:, ::-1].astype(float).mean(axis=0))
            if prev is not None and prev['frame'] == i - 1:
                # Same pixel coordinates in both frames, repair against source control.
                a = rep_img[prev['ry'], prev['rx']][:, ::-1].astype(float).mean(axis=0)
                steps.append(float(np.abs(a - prev['fill']).mean()))
                b = src_img[prev['ry'], prev['rx']][:, ::-1].astype(float).mean(axis=0)
                controls.append(float(np.abs(b - prev['src']).mean()))
                inter = np.logical_and(prev['cells'], ctx_cells(ctx, local)).sum()
                union = np.logical_or(prev['cells'], ctx_cells(ctx, local)).sum()
                ious.append(float(inter / max(union, 1)))
                sizes.append(abs(int(local.sum()) - prev['size']))
            prev = dict(frame=i, ry=ry, rx=rx, fill=fill,
                        src=src_img[ry, rx][:, ::-1].astype(float).mean(axis=0),
                        cells=ctx_cells(ctx, local), size=int(local.sum()))
        block = np.array(fills)
        # An adjacent-frame STEP conflates real eye motion with flicker, which is how
        # round 32 reported a median below the source's own control while the result was
        # plainly flickering. The jitter figure below removes the smooth part: it is the
        # fill colour's residual against a five-frame rolling median of itself, so a
        # steady drift as the eye turns costs nothing and a one-frame excursion costs
        # everything. The source's own jitter is the paired control.
        jitter = roughness(fills)
        src_jitter = roughness(src_fills)
        temporal[side] = dict(
            adjacent_frame_pairs=len(steps),
            repair_step=summarize(steps), source_step_control=summarize(controls),
            repair_jitter=summarize(jitter), source_jitter_control=summarize(src_jitter),
            mask_iou_between_adjacent_frames=(None if not ious else dict(
                n=len(ious), median=round(float(np.median(ious)), 4),
                p5=round(float(np.percentile(ious, 5)), 4),
                min=round(float(np.min(ious)), 4))),
            mask_size_change_px=summarize(sizes),
            fill_display_min=[round(float(v), 2) for v in block.min(0)],
            fill_display_max=[round(float(v), 2) for v in block.max(0)],
            fill_display_median=[round(float(v), 2) for v in np.median(block, axis=0)])

    record = dict(
        code_sha256=r32.sha(__file__), repair_code_sha256=report['code_sha256'],
        atlas_code_sha256=report['atlas_code_sha256'],
        rotations_deg=list(ROTATIONS), temporal_split_cut_frame=folds,
        withheld_sites=stats, withheld=withheld, temporal=temporal,
        note=('atlas_split is scored ONLY on frames outside the half its atlas was '
              'built from. r32_repair, profile, annulus_median and single_gain are '
              'round 32\'s controls, recomputed here on identical pixels. '
              'texture_floor is the plate\'s own angular variation at the same radii '
              'and is not a target any completion can beat. Temporal figures are read '
              'from the 8-bit review frames the artist watches, so they are quantized '
              'at one display level.'))
    r32.write_json(shot.log('r33_withheld.json'), record)
    print(json.dumps({'withheld_sites': stats, 'withheld': withheld,
                      'temporal': temporal}, indent=1), flush=True)


def rho_core(_built):
    u, v = atl.atlas_grid()
    return np.hypot(u, v) < r32.RHO_MAX


def ctx_cells(ctx, local):
    """The footprint as a coarse cell set in the eye's normalized frame, for IoU."""
    geom = ctx['geom']
    ys, xs = np.where(local)
    p = (xs + geom['window'][1].start - geom['limbus'][0]) / geom['ax']
    q = (ys + geom['window'][0].start - geom['limbus'][1]) / geom['ay']
    grid = np.zeros((64, 64), np.uint8)
    ai = np.clip(np.rint((p + 1) / 2 * 63).astype(int), 0, 63)
    aj = np.clip(np.rint((q + 1) / 2 * 63).astype(int), 0, 63)
    grid[aj, ai] = 1
    # The scatter is sparser than the cell grid, so close it before it is compared.
    # Without this the cells of two adjacent frames essentially never coincide and the
    # IoU reads 0.0 for every pair, which measures the sampling, not the mask.
    return cv2.dilate(grid, np.ones((3, 3), np.uint8)) > 0


if __name__ == '__main__':
    main()
