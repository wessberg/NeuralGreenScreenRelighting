"""Compare two eye-pass runs pixel by pixel and field by field: the GPU port against the CPU
reference (`eye_soft_removal.py`) with the same flags on the same frames.

Round 39's gate: the GPU port may not move a pixel the artist has seen without saying
exactly which and why. This reads only what both runs WROTE -- `removal/`, `weight/`,
`source/`, the canonical eyes and `removal.json` -- so it shares no code with either.

Reports, per frame and over the run: pixels whose 8-bit value differs (and by how much),
split into inside / outside either run's weight; weight differences in 1/255 steps; per-eye
decisions (repaired, reason, fill_abstained) that differ; numeric record fields' largest
difference; the canonical atlases' largest difference. Lists the worst frames so they
can be LOOKED at.

    dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/validate_eye_gpu.py \
        --shot music_videos/reporter_test/shots/reporter_full --ref r38_full_own --test r39_gpu
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shot_paths import add_shot_argument, from_args

NUMERIC = ('frame_residual_scale', 'registration_ncc', 'sequence_residual_scale',
           'peak_excess_sigmas', 'weighted_pixels', 'pixels_touched', 'pixels_full_weight',
           'residual_pixels_above_2sigma', 'residual_peak_display', 'peak_removed_display',
           'lash_guard_pixels', 'specular_region_added', 'top_shading_boundary_px',
           'rim_pixels_removed', 'shading_temporal_change_p95', 'highlight_pixels_above_lid')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_shot_argument(ap)
    ap.add_argument('--ref', required=True)
    ap.add_argument('--test', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--worst', type=int, default=12)
    args = ap.parse_args()
    shot = from_args(args)
    A = shot.composite / args.ref
    B = shot.composite / args.test
    ra = json.loads((A / 'removal.json').read_text(encoding='utf-8'))
    rb = json.loads((B / 'removal.json').read_text(encoding='utf-8'))
    rows_a = {r['frame']: r for r in ra['rows']}
    rows_b = {r['frame']: r for r in rb['rows']}
    frames = sorted(set(rows_a) & set(rows_b))
    report = dict(ref=args.ref, test=args.test, frames=len(frames))
    # ---- global fields
    report['sigma'] = {'ref': ra['sequence_residual_scale'], 'test': rb['sequence_residual_scale']}
    report['frozen_registration'] = {'ref': ra['frozen_registration'],
                                     'test': rb['frozen_registration']}
    report['canonical_symmetry'] = {'ref': ra['canonical_symmetry'], 'test': rb['canonical_symmetry']}
    report['pupil_level'] = {'ref': ra['pupil_level'], 'test': rb['pupil_level']}
    canon = {}
    for side in ('left', 'right'):
        pa, pb = A / ('canonical_%s_votes.npz' % side), B / ('canonical_%s_votes.npz' % side)
        if pa.exists() and pb.exists():
            with np.load(pa) as za, np.load(pb) as zb:
                canon[side] = dict(
                    atlas_max_abs=float(np.abs(za['atlas'] - zb['atlas']).max()),
                    atlas_max_abs_display=float(np.abs(
                        np.clip(za['atlas'], 0, 1) ** (1 / 2.4) - np.clip(zb['atlas'], 0, 1) ** (1 / 2.4)).max()),
                    aw_equal=bool(np.array_equal(za['aw'], zb['aw'])),
                    count_equal=bool(np.array_equal(za['count'], zb['count'])),
                    spread_max_abs=float(np.abs(za['spread'] - zb['spread']).max()))
    report['canonical'] = canon
    # ---- per frame pixels
    per = []
    tot = dict(px_diff=0, px_diff_outside_weights=0, weight_diff_px=0, max_level=0,
               frames_identical=0, source_diff_px=0)
    hist = np.zeros(256, np.int64)
    for i in frames:
        sa = cv2.imread(str(A / 'source' / ('%06d.png' % i)))
        sb = cv2.imread(str(B / 'source' / ('%06d.png' % i)))
        tot['source_diff_px'] += int((sa != sb).any(-1).sum())
        xa = cv2.imread(str(A / 'removal' / ('%06d.png' % i))).astype(np.int16)
        xb = cv2.imread(str(B / 'removal' / ('%06d.png' % i))).astype(np.int16)
        wa = cv2.imread(str(A / 'weight' / ('%06d.png' % i)), cv2.IMREAD_UNCHANGED)
        wb = cv2.imread(str(B / 'weight' / ('%06d.png' % i)), cv2.IMREAD_UNCHANGED)
        d = np.abs(xa - xb).max(-1)
        dw = np.abs(wa.astype(np.int16) - wb.astype(np.int16))
        outside = (wa == 0) & (wb == 0)
        n = int((d > 0).sum())
        hist += np.bincount(d.ravel(), minlength=256)[:256]
        rec = dict(frame=i, px_diff=n, max_level=int(d.max()),
                   px_diff_outside_weights=int((d[outside] > 0).sum()),
                   weight_diff_px=int((dw > 0).sum()), weight_max_step=int(dw.max()))
        per.append(rec)
        tot['px_diff'] += n
        tot['px_diff_outside_weights'] += rec['px_diff_outside_weights']
        tot['weight_diff_px'] += rec['weight_diff_px']
        tot['max_level'] = max(tot['max_level'], rec['max_level'])
        tot['frames_identical'] += int(n == 0 and rec['weight_diff_px'] == 0)
    report['pixels'] = tot
    report['level_histogram_nonzero'] = {int(k): int(v) for k, v in enumerate(hist) if v and k}
    diffs = np.array([r['px_diff'] for r in per])
    levels = np.array([r['max_level'] for r in per])
    report['per_frame'] = dict(px_diff_p95=float(np.percentile(diffs, 95)),
                               px_diff_max=int(diffs.max()),
                               max_level_p95=float(np.percentile(levels, 95)),
                               max_level_max=int(levels.max()))
    report['worst_frames'] = sorted(per, key=lambda r: (-r['max_level'], -r['px_diff']))[:args.worst]
    # ---- per eye records
    decisions, numeric = [], {k: [] for k in NUMERIC}
    for i in frames:
        ea = {e['side']: e for e in rows_a[i]['eyes']}
        eb = {e['side']: e for e in rows_b[i]['eyes']}
        for side in ('left', 'right'):
            a, b = ea.get(side, {}), eb.get(side, {})
            key = lambda e: (e.get('repaired'), e.get('reason'), e.get('fill_abstained'))
            if key(a) != key(b):
                decisions.append(dict(frame=i, side=side, ref=key(a), test=key(b)))
            for k in NUMERIC:
                va, vb = a.get(k), b.get(k)
                if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                    numeric[k].append(abs(va - vb))
                elif (va is None) != (vb is None):
                    numeric[k].append(float('inf'))
    report['decision_differences'] = decisions
    report['numeric_field_max_abs'] = {k: (max(v) if v else None) for k, v in numeric.items()}
    report['numeric_field_n_differ'] = {k: int(sum(1 for x in v if x > 0)) for k, v in numeric.items()}
    out = Path(args.out) if args.out else shot.log('validate_%s_vs_%s.json' % (args.test, args.ref))
    out.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    brief = {k: report[k] for k in ('frames', 'pixels', 'per_frame', 'canonical',
                                     'numeric_field_max_abs')}
    brief['decision_differences'] = len(decisions)
    print(json.dumps(brief, indent=1))
    print('worst:', [(r['frame'], r['max_level'], r['px_diff']) for r in report['worst_frames']])
    print('wrote', out)


if __name__ == '__main__':
    main()
