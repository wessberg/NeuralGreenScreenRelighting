"""Exhaustive donor-context probe at three fixed target frames; no repaired pixels.

This tests whether the 12-donor shortlist alone explains missing evidence. It is a
coverage probe, not a second recovery algorithm or a calibrated confidence test.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from audit_r8_reflections import iris_patch, reflection_candidates, LUMA, read, write
from audit_r9_iris import (display, bright_exclusion, sampling_maps, footprint_guard,
                          fit, warp, MIN_CONTEXT, tile)
from color_pipeline import srgb_to_linear
from shot_paths import add_shot_argument, from_args


def panel(patch, mask, text):
    im = tile(patch)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(im, [c*6 for c in contours], -1, (0, 0, 255), 1)
    im = cv2.copyMakeBorder(im, 24, 0, 0, 0, cv2.BORDER_CONSTANT)
    cv2.putText(im, text, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1)
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_shot_argument(ap)
    shot = from_args(ap.parse_args())
    if shot.name != 'reporter_a' or shot.video_dir.name != 'reporter_test':
        raise SystemExit('Reporter-only diagnostic')
    cv2.setNumThreads(1)
    out = shot.composite / 'r10_iris'
    lm = np.load(out / 'source_landmarks.npz')
    topo = json.loads((out / 'source_tracking.json').read_text())['topology']
    report = {'target_local_frames_fixed_before_results': [21, 79, 120], 'eyes': {},
              'method': 'all nonadjacent donors; same initial context-overlap requirement, bounded affine and masks; individual context MAE <=2',
              'limitation': 'unmasked observations are not certified clean iris; coverage is not a recovery or confidence result'}
    for side in ('left', 'right'):
        frames = []
        for i in range(len(lm['found'])):
            rgb = srgb_to_linear(read(out / 'source_crops' / ('%06d.png' % i))[..., ::-1].astype(np.float32)/255)
            points = lm['points'][i].copy()
            points[:, :2] -= lm['origins'][i]
            p, v, _ = iris_patch(rgb, points, topo, side)
            if v.sum() < 150 or not lm['found'][i]:
                v[:] = False
            frames.append((p, v))
        patches, valid = np.stack([p for p, v in frames]), np.stack([v for p, v in frames])
        dv = valid & ~np.stack([bright_exclusion(p, v) for p, v in frames])
        rows = []
        for k in report['target_local_frames_fixed_before_results']:
            index = shot.config['selection']['start_frame'] + k
            target = patches[index].copy()
            hole = reflection_candidates(target @ LUMA, valid[index])
            row = {'local_frame': k, 'source_frame': index, 'candidate_pixels': int(hole.sum())}
            if not hole.any():
                row['status'] = 'no compact candidate; no invented target mask'
                rows.append(row)
                continue
            mx, my = sampling_maps(lm['points'][index], topo, side)
            guard = footprint_guard(mx, my, hole)
            target[guard] = 0
            tv = valid[index] & ~guard
            tv &= ~bright_exclusion(target, tv)
            counts, outside = np.zeros(hole.shape, int), np.zeros(hole.shape, int)
            fits, eligible, bounded_fits, usable_fits = [], 0, 0, 0
            if tv.sum() >= MIN_CONTEXT:
                for j, donor in enumerate(patches):
                    if abs(j-index) <= 3 or (dv[j] & tv).sum() < max(MIN_CONTEXT, .6*tv.sum()):
                        continue
                    eligible += 1
                    matrix = fit(target, donor, tv, dv[j])
                    if matrix is None:
                        continue
                    bounded_fits += 1
                    aligned = warp(donor, matrix)
                    usable = warp(dv[j].astype(np.float32), matrix) >= .9999
                    context = usable & tv
                    if context.sum() < max(MIN_CONTEXT, .6*tv.sum()):
                        continue
                    usable_fits += 1
                    error = float(np.abs(display(aligned[context])-display(target[context])).mean())
                    if error > 2:
                        continue
                    supported = usable & hole
                    counts += supported
                    if not 700 <= j < 850:
                        outside += supported
                    fits.append({'source_frame': j, 'context_display_mae': error,
                                 'candidate_usable_pixels': int(supported.sum()),
                                 'matrix': matrix.tolist()})
            fits.sort(key=lambda f: (-f['candidate_usable_pixels'], f['context_display_mae'], f['source_frame']))
            row.update({'initial_context_eligible_donors': eligible, 'bounded_fits': bounded_fits,
                        'usable_context_fits': usable_fits, 'context_mae_le_2_donors': len(fits),
                        'candidate_samples_with_1_3_donors': [int((counts[hole]>=v).sum()) for v in (1, 3)],
                        'candidate_samples_with_1_3_outside_donors': [int((outside[hole]>=v).sum()) for v in (1, 3)],
                        'max_donors_at_candidate': int(counts[hole].max()),
                        'context_accepted_donors': fits})
            # Show donors with most candidate coverage first; no substitution into target.
            tiles = [panel(patches[index], hole, 'target source %d' % index)]
            for f in fits[:7]:
                matrix = np.array(f['matrix'], np.float32)
                p = warp(patches[f['source_frame']], matrix)
                usable = warp(dv[f['source_frame']].astype(np.float32), matrix) >= .9999
                im = panel(p, hole, '%d err %.2f usable %d' % (f['source_frame'],
                           f['context_display_mae'], f['candidate_usable_pixels']))
                # Orange dots identify excluded grid locations within the target hole.
                y, x = np.where(hole & ~usable)
                im[y*6+27, x*6+3] = [0, 165, 255]
                tiles.append(im)
            while len(tiles) < 8:
                tiles.append(np.zeros_like(tiles[0]))
            path = out / ('donors_%s_%04d.jpg' % (side, index))
            write(path, np.concatenate([np.concatenate(tiles[:4], 1), np.concatenate(tiles[4:], 1)], 0))
            row['inspection_panel'] = str(path.relative_to(shot.dir))
            rows.append(row)
            print(side, k, 'candidate', row['candidate_pixels'], 'accepted', len(fits),
                  'coverage', row['candidate_samples_with_1_3_donors'], flush=True)
        report['eyes'][side] = rows
    (out / 'exhaustive_donors_detail.json').write_text(json.dumps(report), encoding='utf-8')
    for rows in report['eyes'].values():
        for row in rows:
            fits = row.pop('context_accepted_donors', [])
            row['shown_donors'] = [{k: v for k, v in f.items() if k != 'matrix'} for f in fits[:7]]
    report['inspection'] = 'pending'
    shot.log('r10_donor_probe.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
