"""Reporter-only short versus full-source iris evidence; never produces a reflection repair.

Run track_r10_source.py in the facetrack venv first, then this in the pipeline venv.
Both arms use identical raw-source RGB and target geometry. Round 9's fixed gate,
native-footprint guard, affine bounds, shortlisting and four hidden discs are reused.
"""
import argparse
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from audit_r8_reflections import iris_patch, reflection_candidates, LUMA, read, write
from audit_r9_iris import (SIZE, CENTRES, display, dilate, bright_exclusion, sampling_maps,
                          footprint_guard, recover, measure, summarise, temporal_error,
                          tile, digest, code_checks)
from color_pipeline import srgb_to_linear
from shot_paths import add_shot_argument, from_args


def label(frame, lines):
    frame = cv2.copyMakeBorder(frame, 48, 0, 0, 0, cv2.BORDER_CONSTANT)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (8, 19 + i*20), cv2.FONT_HERSHEY_SIMPLEX,
                    .43, (255, 255, 255), 1)
    return frame


def mask_tile(patch, valid, candidate, excluded):
    bgr = display(patch)[..., ::-1].astype(np.uint8)
    bgr[~valid] = (bgr[~valid].astype(float) * .3).astype(np.uint8)
    bgr[excluded & valid] = (bgr[excluded & valid]*.4 + np.array([0, 165, 255])*.6).astype(np.uint8)
    bgr[candidate] = [0, 0, 255]
    return cv2.resize(bgr, (246, 246), interpolation=cv2.INTER_NEAREST)


def encode(shot, name, filename, count, per_sheet):
    subprocess.run([sys.executable, str(Path(__file__).with_name('review_video.py')),
                    '--shot', str(shot.dir), '--name', 'r10_iris/' + name,
                    '--fps', 'shot', '--out', filename], check=True)
    cap = cv2.VideoCapture(str(shot.final / filename))
    fps = cap.get(cv2.CAP_PROP_FPS)
    decoded, tiles, sheets = 0, [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        thumb = cv2.resize(frame, (246, 180), interpolation=cv2.INTER_AREA)
        cv2.putText(thumb, str(decoded), (4, 176), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
        tiles.append(thumb)
        decoded += 1
        if len(tiles) == per_sheet or decoded == count:
            while len(tiles) % 6:
                tiles.append(np.zeros_like(tiles[0]))
            sheet = np.concatenate([np.concatenate(tiles[j:j+6], 1)
                                    for j in range(0, len(tiles), 6)], 0)
            path = shot.composite / 'r10_iris' / ('%s_review_%04d.jpg' % (name, decoded))
            write(path, sheet)
            sheets.append(str(path.relative_to(shot.dir)))
            tiles = []
    cap.release()
    if decoded != count or abs(fps-float(shot.frame_rate)) > .001:
        raise RuntimeError('Incomplete or mistimed encode')
    return {'video': filename, 'decoded_frames': decoded, 'fps': fps, 'sheets': sheets,
            'inspection': 'pending', 'artist_verdict': 'pending'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_shot_argument(ap)
    args = ap.parse_args()
    shot = from_args(args)
    if shot.name != 'reporter_a' or shot.video_dir.name != 'reporter_test':
        raise SystemExit('Reporter-only diagnostic')
    cv2.setNumThreads(1)
    out = shot.composite / 'r10_iris'
    lm = np.load(out / 'source_landmarks.npz')
    tracking = json.loads((out / 'source_tracking.json').read_text())
    topo = tracking['topology']
    start, length = shot.config['selection']['start_frame'], shot.frame_count
    n = len(lm['found'])
    protected = [shot.plate, shot.dir / 'shot.json', shot.relight / 'face_landmarks.npz',
                 Path(__file__).with_name('composite_ratio.py')]
    protected += [p for i in range(length) for p in
                  (shot.fg(i), shot.composite / 'ratio' / ('%06d.png' % i))]
    hashes = {str(p): digest(p) for p in protected}
    checks = code_checks()
    data = {side: [] for side in ('left', 'right')}
    key_differences = {side: [] for side in data}
    for i in range(n):
        crop = read(out / 'source_crops' / ('%06d.png' % i))
        # Deliberately match the existing keyer's sRGB numerical convention here.
        # This is not a new validated BT.709 scene-linear ingest transform.
        rgb = srgb_to_linear(crop[..., ::-1].astype(np.float32)/255)
        points = lm['points'][i].copy()
        points[:, :2] -= lm['origins'][i]
        fg = read(shot.fg(i-start))[..., ::-1] if start <= i < start+length else None
        for side in data:
            if lm['found'][i]:
                patch, valid, radius = iris_patch(rgb, points, topo, side)
                mx, my = sampling_maps(lm['points'][i], topo, side)
                if valid.sum() < 150:
                    valid[:] = False
                if fg is not None:
                    keyed, kv, _ = iris_patch(fg, lm['points'][i], topo, side)
                    common = kv & valid
                    key_differences[side].append(measure(patch, keyed, common))
            else:
                patch = np.zeros((SIZE, SIZE, 3), np.float32)
                valid = np.zeros((SIZE, SIZE), bool)
                mx = my = np.zeros((SIZE, SIZE), np.float32)
                radius = 0
            data[side].append((patch, valid, mx, my, radius))
        if i % 100 == 0:
            print('loaded', i, flush=True)
    mask_seq, hidden_seq = out / 'mask_review', out / 'holdout_review'
    mask_seq.mkdir(exist_ok=True)
    hidden_seq.mkdir(exist_ok=True)
    yy, xx = np.mgrid[:SIZE, :SIZE]
    holes = [(xx-x)**2 + (yy-y)**2 <= 9 for x, y in CENTRES]
    reports, mask_panels, hidden_panels = {}, {}, {}
    invariance = []
    for side, frames in data.items():
        patches = np.stack([f[0] for f in frames])
        valid = np.stack([f[1] for f in frames])
        excluded = np.stack([bright_exclusion(p, v) for p, v in zip(patches, valid)])
        dv = valid & ~excluded
        candidates = np.stack([reflection_candidates(p @ LUMA, v) for p, v in zip(patches, valid)])
        old_excluded = np.stack([dilate(c, 5) for c in candidates])
        short = (patches[start:start+length], valid[start:start+length], dv[start:start+length])
        wide = (patches, valid, dv)
        masks = []
        for i in range(n):
            masks.append(np.concatenate([tile(patches[i]),
                mask_tile(patches[i], valid[i], candidates[i], excluded[i])], 1))
        mask_panels[side] = masks
        rows, candidate_rows, panels = [], [], []
        temporal = {m: [None]*length for m in ('short', 'wide')}
        for k in range(length):
            i = start+k
            review = [tile(patches[i]), tile(patches[i], holes[3]), tile(patches[i], holes[3])]
            for h, hole in enumerate(holes):
                if not valid[i][hole].all() or old_excluded[i][hole].any():
                    continue
                guard = footprint_guard(frames[i][2], frames[i][3], hole)
                result = {'short': recover(*short, k, hole, guard, True),
                          'wide': recover(*wide, i, hole, guard, True)}
                common = result['short'][1] & result['wide'][1]
                row = {'frame': k, 'source_frame': i, 'centre': CENTRES[h],
                       'hole_pixels': int(hole.sum()), 'guard_pixels': int(guard.sum())}
                for m, (pred, support, conf, info) in result.items():
                    row[m] = {'all': measure(pred, patches[i], support),
                              'paired': measure(pred, patches[i], common),
                              'confident': measure(pred, patches[i], conf), 'fit': info}
                rows.append(row)
                if h == 3:
                    for j, m in enumerate(('short', 'wide'), 1):
                        pred, support, conf, _ = result[m]
                        shown = patches[i].copy()
                        shown[support] = pred[support]
                        review[j] = tile(shown, hole, support)
                        temporal[m][k] = (pred, common)
                if k in (21, 79, 120) and h == 3:
                    changed = patches.copy()
                    changed[i][guard] = [.99, .001, .7]
                    cdv = dv.copy()
                    cdv[i] = valid[i] & ~bright_exclusion(changed[i], valid[i])
                    for m, triple, index in [('short', (changed[start:start+length],
                            valid[start:start+length], cdv[start:start+length]), k),
                            ('wide', (changed, valid, cdv), i)]:
                        test = recover(*triple, index, hole, guard, True)
                        ref = result[m]
                        if not (all(np.array_equal(a, b) for a, b in zip(test[:3], ref[:3]))
                                and test[3] == ref[3]):
                            raise AssertionError('Hidden RGB leaked into prediction')
                        invariance.append({'side': side, 'frame': k, 'method': m,
                                           'supported_pixels': int(ref[1].sum()), 'passed': True})
            glint = candidates[i]
            if glint.any():
                guard = footprint_guard(frames[i][2], frames[i][3], glint)
                row = {'frame': k, 'source_frame': i, 'candidate_pixels': int(glint.sum())}
                for m, triple, index in [('short', short, k), ('wide', wide, i)]:
                    _, support, conf, info = recover(*triple, index, glint, guard, True)
                    row[m] = {'supported_pixels': int(support.sum()),
                              'confident_pixels': int(conf.sum()), 'fit': info}
                candidate_rows.append(row)
            panels.append(np.concatenate(review, 1))
            if k % 25 == 0:
                print(side, k, 'hidden patches', len(rows), flush=True)
        hidden_panels[side] = panels
        missing = recover(patches, np.zeros_like(valid), np.zeros_like(dv), start,
                          holes[3], holes[3], True)
        if missing[1].any() or missing[2].any():
            raise AssertionError('Missing evidence produced support')
        wide_ids = [j for row in rows for j in row['wide']['fit'].get('donor_indices', [])]
        reports[side] = {'eligible_patches': len(rows),
            'wide_hidden_selected_donor_slots': len(wide_ids),
            'wide_hidden_outside_selected_interval_slots': sum(not start <= j < start+length for j in wide_ids),
            'summary': {m: {s: summarise(rows, m, s) for s in ('all', 'paired', 'confident')}
                        for m in ('short', 'wide')},
            'candidate_totals': {m: {field: sum(r[m][field] for r in candidate_rows)
                for field in ('supported_pixels', 'confident_pixels')} for m in ('short', 'wide')},
            'candidate_frames': len(candidate_rows),
            'candidate_pixels': int(candidates[start:start+length].sum()),
            'temporal_lower_hole': {m: temporal_error(p, patches[start:start+length])
                                    for m, p in temporal.items()},
            'full_source_mask_counts': [{'source_frame': i, 'visible': int(valid[i].sum()),
                'candidate': int(candidates[i].sum()), 'excluded': int((excluded[i]&valid[i]).sum()),
                'radius_native_px': frames[i][4]} for i in range(n)],
            'raw_vs_keyed_target_display_mae_p50_p95': np.percentile(
                [r['display_mae'] for r in key_differences[side] if r], [50, 95]).tolist(),
            'per_patch': rows, 'candidate_glints': candidate_rows}
    for i in range(n):
        frame = np.concatenate([mask_panels[s][i] for s in ('left', 'right')], 0)
        frame = label(frame, ['source %04d  raw | masks; TOP=subject-left' % i,
                             'RED=candidate ORANGE=excluded DIM=outside. No repair.'])
        write(mask_seq / ('%06d.png' % i), frame)
    for k in range(length):
        frame = np.concatenate([hidden_panels[s][k] for s in ('left', 'right')], 0)
        frame = label(frame, ['source %04d  truth | 150-frame donors | 1200-frame donors' % (start+k),
                             'Hidden lower patch ONLY. Purple=unavailable. No reflection repair.'])
        write(hidden_seq / ('%06d.png' % k), frame)
    review = [encode(shot, 'mask_review', 'r10_full_source_masks.mp4', n, 60),
              encode(shot, 'holdout_review', 'r10_iris_withheld_comparison.mp4', length, 30)]
    if hashes != {str(p): digest(p) for p in protected}:
        raise AssertionError('Protected production file changed')
    if not any(r['method'] == 'wide' and r['supported_pixels'] for r in invariance):
        raise AssertionError('No supported wide-pool independence check')
    checks.update({'hidden_rgb_invariance': invariance, 'missing_support': 'passed',
                   'protected_files_sha256_unchanged': len(protected),
                   'protected_manifest': hashes})
    report = {'status': 'diagnostic only; no reflection removal', 'tracking': tracking,
              'method': {'target_source_frames': [start, start+length-1],
                  'short_donor_source_frames': [start, start+length-1],
                  'wide_donor_source_frames': [0, n-1],
                  'donor_indices_convention': 'short indices are local to selection; wide indices are original source frame numbers',
                  'shared': 'round9 recover: affine, 12 shortlisted, best 7, >=3 donors, target +/-3 excluded',
                  'confidence': 'unchanged <=2 display context MAE and <=2 donor MAD',
                  'colour': '8-bit OpenCV original decode, sRGB numerical inverse matching existing keyer; NOT validated BT.709 linear ingest',
                  'target_landmarks': 'cached; outside selected interval freshly tracked'},
              'eyes': reports, 'checks': checks, 'review': review,
              'limitations': ['Conservative masks are neither material labels nor removal masks.',
                  'No clean reflection ground truth; confident agreement may still be wrong.',
                  'Wider shortlisting is not exhaustive and may select nearby similar donors.',
                  'No photometric fitting; source decode differs from keyed half-float RGB.',
                  'Fresh donor tracking and cached target tracking may have different biases.',
                  'Affine model cannot resolve refraction, pupil dilation, lids or motion blur.',
                  'Hair/lips/source occlusion remain unresolved; production defaults unchanged.']}
    (out / 'correspondence_detail.json').write_text(json.dumps(report), encoding='utf-8')
    for eye in report['eyes'].values():
        for row in eye['per_patch'] + eye['candidate_glints']:
            for m in ('short', 'wide'):
                row[m]['fit'].pop('matrices_target_to_donor', None)
    # Store the full hash manifest with the regenerable detailed evidence.
    report['checks'].pop('protected_manifest')
    shot.log('r10_iris_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({s: {'summary': r['summary'], 'candidate_totals': r['candidate_totals']}
                      for s, r in reports.items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()
