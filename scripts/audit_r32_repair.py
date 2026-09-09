"""Score round 32's completion on pixels it never saw, against paired controls.

The design is frozen in `docs/r32-eye-protocol.md`, written before any error number
below existed. Each repaired eye's real footprint is rotated about the limbus in the
eye's own normalized polar frame, which preserves its size, shape and radius exactly,
so a rotated copy poses the same completion problem at an angle where the true pixels
ARE observed. Those pixels are then withheld and predicted.

The `single_gain` control is round 31's failure reproduced on the identical pixels: one
flat linear RGB fitted on a surrounding ring, which is what put a light brown spot in
the pupil. `texture_floor` is the plate's own angular texture variation at the same
radii -- no completion can beat it, so it says how much of the remaining error is the
iris being a real, varying material rather than the method being wrong.

    dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/audit_r32_repair.py \\
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
from eye_reflection_repair import (EDGES, SIDES, angular_fill, angular_gain, detect_anomaly,
                                   evaluate_profile, eye_frame, pupil_radius, radial_profile,
                                   sha, solve_eye, write_json)

ROTATIONS = (60.0, 120.0, 180.0, 240.0, 300.0)


def display_error(prediction, truth):
    """Per-pixel mean absolute sRGB difference on a 0-255 scale."""
    a = np.clip(linear_to_srgb(np.clip(prediction, 0, None).astype(np.float32)), 0, 1)
    b = np.clip(linear_to_srgb(np.clip(truth, 0, None).astype(np.float32)), 0, 1)
    return np.abs(a - b).mean(axis=-1) * 255.0


def fit_model(values, rho, theta, usable):
    profile, centres = radial_profile(values[usable], rho[usable], EDGES)
    if profile is None:
        return None
    radial = evaluate_profile(profile, centres, rho)
    gain = angular_gain(values.max(axis=-1), radial.max(axis=-1), theta, usable)
    return dict(base=radial * gain[..., None], profile=profile, centres=centres)


def rotate_mask(mask, rho, theta, centre, ax, ay, angle, origin):
    """Rotate a footprint about the limbus in the eye's normalized polar frame.

    `centre` and the polar grids are in FULL-FRAME coordinates, because that is how
    the repair builds them; `mask` is indexed in the eye window. `origin` converts.
    """
    ys, xs = np.where(mask)
    turned = theta[ys, xs] + np.deg2rad(angle)
    radius = rho[ys, xs]
    px = np.rint(centre[0] + ax * radius * np.cos(turned)).astype(int) - origin[0]
    py = np.rint(centre[1] + ay * radius * np.sin(turned)).astype(int) - origin[1]
    out = np.zeros(mask.shape, bool)
    inside = (px >= 0) & (py >= 0) & (px < mask.shape[1]) & (py < mask.shape[0])
    if not inside.all():
        return None
    out[py, px] = True
    return out


def annulus_median(values, rho, site, donor, tolerance=0.075):
    out = np.zeros((int(site.sum()), values.shape[-1]))
    donor_rho, donor_values = rho[donor], values[donor]
    for j, r0 in enumerate(rho[site]):
        band = np.abs(donor_rho - r0) < tolerance
        if band.sum() < 2:
            band = np.abs(donor_rho - r0) < tolerance * 2
        out[j] = np.median(donor_values[band], axis=0) if band.any() else np.nan
    return out


def ring_gain(values, site, donor):
    """One flat linear RGB from a ring around the site -- round 31's fill shape."""
    away = cv2.distanceTransform((~site).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    ring = donor & (away >= 3) & (away < 7)
    if ring.sum() < 12:
        ring = donor & (away >= 2) & (away < 9)
    if ring.sum() < 6:
        return None
    return np.median(values[ring], axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_shot_argument(ap)
    ap.add_argument('--name', default='r32_repair')
    args = ap.parse_args()
    shot = from_args(args)
    report = json.loads((shot.composite / args.name / 'repair.json').read_text())
    meta = json.loads((shot.relight / 'face_landmarks.json').read_text())
    topology = meta['topology']
    with np.load(shot.relight / 'face_landmarks.npz') as z:
        landmarks = {k: z[k] for k in ('points', 'found')}

    samples = {side: {key: [] for key in ('repair', 'profile', 'annulus_median',
                                          'single_gain', 'texture_floor')}
               for side in SIDES}
    inner_outer = {side: {'inner': [], 'outer': []} for side in SIDES}
    sites = 0
    rejected = 0
    attempted = 0
    why = {}
    for row in report['rows']:
        i = row['frame']
        if not any(e.get('repaired') for e in row['eyes']):
            continue
        raw = cv2.imread(str(shot.fg(i)), cv2.IMREAD_UNCHANGED)
        values_full = raw[..., ::-1].astype(np.float32)
        for entry in row['eyes']:
            if not entry.get('repaired'):
                continue
            side = entry['side']
            geom, _ = eye_frame(values_full, landmarks['points'][i], topology, side)
            if geom is None:
                continue
            patch = values_full[geom['window']]
            state = solve_eye(patch, geom)
            if state is None:
                continue
            rho, theta, inside = state['rho'], state['theta'], state['inside']
            footprint, model, reason = detect_anomaly(patch, rho, theta, inside)
            if footprint is None:
                continue
            collar = cv2.dilate(footprint.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            step = pupil_radius(model['profile'], model['centres'])
            for angle in ROTATIONS:
                attempted += 1
                site = rotate_mask(footprint, rho, theta, state['centre'],
                                   geom['ax'], geom['ay'], angle,
                                   (geom['window'][1].start, geom['window'][0].start))
                if site is None or not site.any():
                    rejected += 1
                    why['empty'] = why.get('empty', 0) + 1
                    continue
                if (site & collar).any() or not inside[site].all():
                    rejected += 1
                    why['overlap'] = why.get('overlap', 0) + 1
                    continue
                hidden = footprint | site
                refit = fit_model(patch, rho, theta, inside & ~hidden)
                if refit is None:
                    rejected += 1
                    why['nomodel'] = why.get('nomodel', 0) + 1
                    continue
                base = refit['base']
                donor = inside & ~(cv2.dilate(hidden.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
                if donor.sum() < 60:
                    rejected += 1
                    why['nodonor'] = why.get('nodonor', 0) + 1
                    continue
                filled, supported = angular_fill(patch - base, rho, theta, site, donor)
                if not supported.all():
                    rejected += 1
                    why['unsupported'] = why.get('unsupported', 0) + 1
                    continue
                truth = patch[site]
                predictions = {
                    'repair': np.maximum(base[site] + filled, 0.0),
                    'profile': base[site],
                    'annulus_median': annulus_median(patch, rho, site, donor),
                }
                flat = ring_gain(patch, site, donor)
                if flat is None:
                    rejected += 1
                    why['noring'] = why.get('noring', 0) + 1
                    continue
                predictions['single_gain'] = np.repeat(flat[None], int(site.sum()), axis=0)
                sites += 1
                for key, prediction in predictions.items():
                    if not np.isfinite(prediction).all():
                        continue
                    samples[side][key].append(display_error(prediction, truth))
                # The plate's own angular texture variation at the same radii.
                floor_rho = rho[site]
                donor_rho, donor_residual = rho[donor], (patch - base)[donor]
                floor = []
                for r0 in floor_rho:
                    band = np.abs(donor_rho - r0) < 0.075
                    if band.sum() >= 4:
                        block = donor_residual[band]
                        floor.append(float(np.median(np.abs(block - np.median(block, axis=0))
                                                     ).mean()))
                if floor:
                    reference = base[site].mean(axis=0)
                    scale = float(np.mean(floor))
                    samples[side]['texture_floor'].append(
                        display_error(reference[None] + scale, reference[None]))
                errors = display_error(predictions['repair'], truth)
                if step is not None:
                    within = rho[site] < step
                    if within.any():
                        inner_outer[side]['inner'].append(errors[within])
                    if (~within).any():
                        inner_outer[side]['outer'].append(errors[~within])

    def summarize(chunks):
        if not chunks:
            return None
        joined = np.concatenate(chunks)
        return dict(pixels=int(joined.size), median=round(float(np.median(joined)), 4),
                    p95=round(float(np.percentile(joined, 95)), 4),
                    mean=round(float(joined.mean()), 4))

    results = {side: {key: summarize(chunks) for key, chunks in samples[side].items()}
               for side in SIDES}
    split = {side: {key: summarize(chunks) for key, chunks in inner_outer[side].items()}
             for side in SIDES}

    # Temporal stability of the fill RELATIVE TO ITS OWN SURROUNDINGS. Round 31 fitted
    # a flat linear gain on a ring every frame and it swung 1.73x in red between source
    # 701 and 722, with R/B warmth 2.162 against 1.307; that is what the artist saw as
    # one frame reading wrong and the next acceptable. The comparable quantity here is
    # the ratio of the mean filled radiance to the mean of the surrounding annulus,
    # because nothing in this round fits a per-frame gain at all.
    temporal = {}
    ratios = {side: [] for side in SIDES}
    for row in report['rows']:
        i = row['frame']
        if not row['changed_pixels']:
            continue
        repaired = cv2.imread(str(shot.composite / args.name / 'linear' / ('%06d.exr' % i)),
                              cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(shot.composite / args.name / 'mask' / ('%06d.png' % i)),
                          cv2.IMREAD_GRAYSCALE)
        if repaired is None or mask is None:
            continue
        repaired = repaired[..., ::-1].astype(np.float32)
        edited = mask > 0
        for entry in row['eyes']:
            if not entry.get('repaired'):
                continue
            side = entry['side']
            geom, _ = eye_frame(repaired, landmarks['points'][i], topology, side)
            if geom is None:
                continue
            window = geom['window']
            local = edited[window]
            if local.sum() < 3:
                continue
            patch = repaired[window]
            ring = (cv2.dilate(local.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0) & ~local
            ring &= geom['visible']
            if ring.sum() < 12:
                continue
            fill = patch[local].mean(axis=0)
            around = np.maximum(patch[ring].mean(axis=0), 1e-6)
            ratios[side].append((i, fill / around))
    for side in SIDES:
        rows = ratios[side]
        if len(rows) < 2:
            temporal[side] = None
            continue
        block = np.array([r[1] for r in rows])
        warmth = block[:, 0] / np.maximum(block[:, 2], 1e-9)
        temporal[side] = dict(
            frames=len(rows),
            fill_over_surround_median=[round(float(v), 4) for v in np.median(block, axis=0)],
            fill_over_surround_max_over_min=[round(float(v), 4) for v in
                                             (block.max(axis=0) / np.maximum(block.min(axis=0), 1e-9))],
            fill_over_surround_iqr=[round(float(v), 4) for v in
                                    (np.percentile(block, 75, axis=0) - np.percentile(block, 25, axis=0))],
            red_blue_ratio_median=round(float(np.median(warmth)), 4),
            red_blue_ratio_max_over_min=round(float(warmth.max() / max(warmth.min(), 1e-9)), 4))

    record = dict(
        protocol_sha256=sha(Path(__file__).resolve().parents[1] / 'docs/r32-eye-protocol.md'),
        code_sha256=sha(__file__),
        repair_code_sha256=report['code_sha256'],
        rotations_deg=list(ROTATIONS),
        sites_attempted=attempted, sites_scored=sites, sites_rejected=rejected,
        rejection_reasons=why,
        withheld=results, repair_split_by_profile_step=split, temporal_fill=temporal,
        note=('Rotated-footprint withheld pixels in the same eyes. Controls are paired '
              'on identical pixels. texture_floor is the plate\'s own angular variation '
              'at the same radii, not an achievable target to beat.'))
    write_json(shot.log('r32_withheld.json'), record)
    print(json.dumps({k: record[k] for k in ('sites_attempted', 'sites_scored',
                                             'sites_rejected', 'rejection_reasons')}, indent=1))
    for side in SIDES:
        print(side, json.dumps(results[side]))
    for side in SIDES:
        print(side, 'split', json.dumps(split[side]))
    for side in SIDES:
        print(side, 'temporal', json.dumps(temporal[side]))
    print('R32_WITHHELD_DONE', flush=True)


if __name__ == '__main__':
    main()
