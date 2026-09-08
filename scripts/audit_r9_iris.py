"""Reporter-only masked iris correspondence experiment; never writes a repaired plate.

Compare centre/scale donors with masked affine ECC on identical withheld observations.
Native bilinear sampling footprints are excluded from target context before fitting.
All brightness statistics are recomputed without that context. Confidence is a hypothesis
tested against hidden RGB, not a claim of clean texture or permission to remove a glint.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from audit_r8_reflections import iris_patch, read, write, reflection_candidates, LUMA
from color_pipeline import linear_to_srgb
from shot_paths import add_shot_argument, from_args
import cv2
import numpy as np

SIZE = 41
CENTRES = [(14, 20), (26, 20), (20, 14), (20, 26)]
IDENTITY = np.eye(2, 3, dtype=np.float32)
MIN_CONTEXT = 80
MAX_CONTEXT_ERROR = 2.0  # display levels, fixed before evaluating this sequence
MAX_DONOR_MAD = 2.0


def display(rgb):
    return np.clip(linear_to_srgb(rgb), 0, 1) * 255


def dilate(mask, width):
    return cv2.dilate(mask.astype(np.uint8), np.ones((width, width), np.uint8)) > 0


def bright_exclusion(rgb, valid):
    """Exclude ALL bright pixels, including boundary-connected components, from fitting.

    This is a conservative exclusion, not a material label or a removal mask.
    The caller passes a target validity mask already excluding the withheld footprint.
    """
    lum = rgb @ LUMA
    if valid.sum() < MIN_CONTEXT:
        return np.ones(valid.shape, bool)
    cutoff = max(2.5 * np.median(lum[valid]), np.percentile(lum[valid], 85))
    return dilate(valid & (lum > cutoff), 5)


def sampling_maps(points, topo, side):
    ids = topo['iris_' + side]
    centre = points[ids[0], :2]
    radius = np.linalg.norm(points[ids[1:], :2] - centre, axis=1).mean()
    axis = np.linspace(-1.1, 1.1, SIZE, dtype=np.float32)
    yy, xx = np.meshgrid(axis, axis, indexing='ij')
    return centre[0] + xx * radius, centre[1] + yy * radius


def footprint_guard(mx, my, hole):
    """Exclude every grid sample sharing any of four native contributors with the hole.

    Including zero-weight neighbours at integer coordinates is conservative. Geometry
    alone defines this guard; neither RGB nor reconstruction error enters it.
    """
    x, y = np.floor(mx).astype(np.int64), np.floor(my).astype(np.int64)
    # Coordinates are nonnegative on this reporter crop; encode pairs without collisions.
    stride = int(x.max() + 3)
    footprints = np.stack([(y + dy) * stride + x + dx
                           for dy, dx in [(0, 0), (0, 1), (1, 0), (1, 1)]], -1)
    contributors = np.unique(footprints[hole])
    return np.isin(footprints, contributors).any(-1)


def warp(a, matrix):
    return cv2.warpAffine(a, matrix, (SIZE, SIZE),
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)


def bounded(matrix):
    if not np.isfinite(matrix).all() or np.linalg.det(matrix[:, :2]) <= 0:
        return False
    scales = np.linalg.svd(matrix[:, :2], compute_uv=False)
    corners = np.array([[5, 5, 1], [35, 5, 1], [5, 35, 1], [35, 35, 1]], np.float32)
    displacement = np.linalg.norm(corners @ matrix.T - corners[:, :2], axis=1)
    return bool(scales.min() >= .90 and scales.max() <= 1.10 and displacement.max() <= 4)


def fit(target, donor, target_valid, donor_valid):
    """ECC maps target coordinates to donor coordinates; no target-hole RGB is present."""
    target_luma = (target @ LUMA).astype(np.float32)
    donor_luma = (donor @ LUMA).astype(np.float32)
    try:
        _, matrix = cv2.findTransformECCWithMask(
            target_luma, donor_luma, target_valid.astype(np.uint8),
            donor_valid.astype(np.uint8), IDENTITY.copy(), cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 60, 1e-5), 1)
    except cv2.error:
        return None
    return matrix if bounded(matrix) else None


def recover(patches, valid, donor_valid, index, hole, guard, aligned):
    # Scrub BEFORE statistics, fitting, ranking or confidence. All target reads use this copy.
    target = patches[index].copy()
    target[guard] = 0
    tv = valid[index] & ~guard
    tv &= ~bright_exclusion(target, tv)
    empty = (np.zeros_like(target), np.zeros_like(hole), np.zeros_like(hole))
    if tv.sum() < MIN_CONTEXT:
        return (*empty, {'failure': 'insufficient_target_context', 'context_pixels': int(tv.sum())})
    common = donor_valid & tv
    counts = common.sum((1, 2))
    errors = (np.abs(display(patches) - display(target)).mean(-1) * common).sum((1, 2))
    errors /= np.maximum(counts, 1)
    errors[counts < max(MIN_CONTEXT, .6 * tv.sum())] = np.inf
    errors[max(0, index-3):index+4] = np.inf
    shortlist = np.argsort(errors, kind='stable')[:12]
    shortlist = shortlist[np.isfinite(errors[shortlist])]
    candidates = []
    for j in shortlist:
        matrix = fit(target, patches[j], tv, donor_valid[j]) if aligned else IDENTITY.copy()
        if matrix is None:
            continue
        warped = warp(patches[j], matrix)
        # All interpolation contributors must be usable; nearest-neighbour masks leak glints.
        usable = warp(donor_valid[j].astype(np.float32), matrix) >= .9999
        context = usable & tv
        if context.sum() < max(MIN_CONTEXT, .6 * tv.sum()):
            continue
        error = float(np.abs(display(warped[context]) - display(target[context])).mean())
        candidates.append((error, int(j), warped, usable, matrix))
    candidates.sort(key=lambda c: (c[0], c[1]))
    fit_count = len(candidates)
    candidates = candidates[:7]
    if len(candidates) < 3:
        return (*empty, {'failure': 'fewer_than_three_usable_fits',
                        'shortlisted': len(shortlist), 'usable_fits': fit_count})
    donors = np.stack([c[2] for c in candidates])
    usable = np.stack([c[3] for c in candidates])
    mask = np.broadcast_to(~usable[..., None], donors.shape)
    prediction = np.ma.median(np.ma.array(donors, mask=mask), axis=0).filled(0).astype(np.float32)
    deviation = np.abs(display(donors) - display(prediction)).mean(-1)
    mad = np.ma.median(np.ma.array(deviation, mask=~usable), axis=0).filled(np.inf)
    supported = hole & valid[index] & (usable.sum(0) >= 3)
    context_error = float(np.median([c[0] for c in candidates]))
    confident = supported & (mad <= MAX_DONOR_MAD) & (context_error <= MAX_CONTEXT_ERROR)
    info = {'donor_indices': [c[1] for c in candidates], 'context_display_mae': context_error,
            'shortlisted': len(shortlist), 'usable_fits': fit_count,
            'matrices_target_to_donor': [c[4].tolist() for c in candidates],
            'median_donor_mad': float(np.median(mad[supported])) if supported.any() else None}
    return prediction, supported, confident, info


def measure(prediction, truth, mask):
    if not mask.any():
        return None
    a, b = prediction[mask], truth[mask]
    return {'pixels': int(mask.sum()),
            'relative_rgb_absolute_error': float(np.abs(a-b).sum() / max(b.sum(), 1e-8)),
            'display_mae': float(np.abs(display(a)-display(b)).mean())}


def summarise(rows, method, subset):
    samples = [r[method][subset] for r in rows if r[method][subset] is not None]
    return {'patches': len(samples), 'pixels': sum(r['pixels'] for r in samples),
            'relative_rgb_error_p50_p95': np.percentile(
                [r['relative_rgb_absolute_error'] for r in samples], [50, 95]).tolist() if samples else None,
            'display_mae_p50_p95': np.percentile(
                [r['display_mae'] for r in samples], [50, 95]).tolist() if samples else None,
            'patches_above_5_display_mae': sum(r['display_mae'] > 5 for r in samples)}


def tile(rgb, hole=None, supported=None):
    bgr = display(rgb)[..., ::-1].astype(np.uint8)
    if hole is not None:
        unsupported = hole if supported is None else hole & ~supported
        bgr[unsupported] = [150, 0, 150]
    return cv2.resize(bgr, (246, 246), interpolation=cv2.INTER_NEAREST)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_checks():
    """Known translations in several directions; do not mistake a diagonal-only pass for truth."""
    rng = np.random.default_rng(9)
    texture = cv2.GaussianBlur(rng.uniform(.03, .3, (SIZE, SIZE, 3)).astype(np.float32), (0, 0), 1)
    yy, xx = np.mgrid[:SIZE, :SIZE]
    hole = (xx-20)**2 + (yy-20)**2 <= 3**2
    valid = (xx > 5) & (xx < 35) & (yy > 5) & (yy < 35)
    target = texture.copy()
    target[hole] = 0
    checks = []
    for dx, dy in [(1.2, -.7), (-1.1, .8), (.3, 1.3), (-.6, -1.1)]:
        forward = IDENTITY.copy()
        forward[:, 2] = [dx, dy]
        donor = cv2.warpAffine(texture, forward, (SIZE, SIZE))
        matrix = fit(target, donor, valid & ~dilate(hole, 5), valid)
        if matrix is None:
            raise AssertionError('Synthetic known-motion registration rejected')
        p = np.array([20, 20, 1], np.float32)
        error = float(np.linalg.norm(matrix @ p - forward @ p))
        if error > .3:
            raise AssertionError('Wrong warp convention or failed registration: %s' % error)
        checks.append({'translation_xy': [dx, dy], 'centre_error_grid_px': error})
    # Verify the guard at the NATIVE-image boundary, rather than only on an interpolated grid.
    native = rng.uniform(0, 1, (100, 100, 3)).astype(np.float32)
    my, mx = np.meshgrid(np.linspace(38.1, 64.9, SIZE, dtype=np.float32),
                         np.linspace(37.9, 65.1, SIZE, dtype=np.float32), indexing='ij')
    guard = footprint_guard(mx, my, hole)
    changed = native.copy()
    x, y = np.floor(mx[hole]).astype(int), np.floor(my[hole]).astype(int)
    for dy, dx in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        changed[y+dy, x+dx] = [1, 0, 1]
    a = cv2.remap(native, mx, my, cv2.INTER_LINEAR)
    b = cv2.remap(changed, mx, my, cv2.INTER_LINEAR)
    if not np.array_equal(a[~guard], b[~guard]) or np.array_equal(a[hole], b[hole]):
        raise AssertionError('Native hidden pixels leaked through the sampling guard')
    return {'synthetic_direction_checks': checks, 'native_footprint_independence': 'passed'}


def temporal_error(predictions, truths):
    """Adjacent change in reconstruction residual on the fixed lower hole, where both exist.

    Subtract the true observed change first. This still includes iris-grid tracking error;
    it is not a decomposition of algorithmic flicker or a result on actual catchlights.
    """
    rows = []
    for i in range(1, len(predictions)):
        if predictions[i] is None or predictions[i-1] is None:
            continue
        p, valid = predictions[i]
        previous, previous_valid = predictions[i-1]
        common = valid & previous_valid
        if not common.any():
            continue
        residual = display(p[common]) - display(truths[i][common])
        old_residual = display(previous[common]) - display(truths[i-1][common])
        rows.append({'frame': i, 'pixels': int(common.sum()),
                     'residual_change_display_mae': float(np.abs(residual-old_residual).mean())})
    return {'adjacent_pairs': len(rows), 'p50_p95_display_levels': np.percentile(
        [r['residual_change_display_mae'] for r in rows], [50, 95]).tolist() if rows else None,
        'per_pair': rows}


def write_report(report, shot):
    # Full transforms are useful for debugging but would dominate the versioned report.
    detail_path = shot.composite / 'r9_iris' / 'correspondence_detail.json'
    detail_path.write_text(json.dumps(report, separators=(',', ':')), encoding='utf-8')
    compact = json.loads(json.dumps(report))
    for eye in compact['eyes'].values():
        fits = [r['fit'] for r in eye['candidate_glints']]
        fits += [r[m]['fit'] for r in eye['per_patch'] for m in ('fixed', 'aligned')]
        for fit_info in fits:
            fit_info.pop('matrices_target_to_donor', None)
    compact['full_correspondence_detail'] = str(detail_path.relative_to(shot.dir))
    shot.log('r9_iris_audit.json').write_text(json.dumps(compact, indent=2), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_shot_argument(ap)
    ap.add_argument('--checks-only', action='store_true')
    args = ap.parse_args()
    shot = from_args(args)
    if shot.name != 'reporter_a' or shot.video_dir.name != 'reporter_test':
        raise SystemExit('Reporter-only diagnostic; not a production repair stage')
    cv2.setNumThreads(1)
    checks = code_checks()
    if args.checks_only:
        print(json.dumps(checks, indent=2))
        return
    lm = np.load(shot.relight / 'face_landmarks.npz')
    topo = json.loads((shot.relight / 'face_landmarks.json').read_text())['topology']
    tour = json.loads(shot.log('stress_proxy.json').read_text())['per_frame']
    if [r['shot_frame'] for r in tour] != list(range(len(tour))):
        raise RuntimeError('Native-motion diagnostic requires consecutive shot frames')
    out = shot.composite / 'r9_iris'
    out.mkdir(exist_ok=True)
    data = {s: [] for s in ('left', 'right')}
    protected = [shot.dir / 'shot.json', Path(__file__).with_name('composite_ratio.py')]
    for i in range(len(tour)):
        protected.extend([shot.fg(i), shot.composite / 'ratio' / ('%06d.png' % i)])
    before_hashes = {str(p): digest(p) for p in protected}
    for i in range(len(tour)):
        if not lm['found'][i]:
            raise RuntimeError('Missing landmarks at %d' % i)
        fg = read(shot.fg(i))[..., :3][..., ::-1].astype(np.float32)
        for side in data:
            patch, visible, radius = iris_patch(fg, lm['points'][i], topo, side)
            if visible.sum() < 150:
                visible[:] = False
            mx, my = sampling_maps(lm['points'][i], topo, side)
            data[side].append((patch, visible, mx, my))
        if i % 50 == 0:
            print('loaded', i, flush=True)
    yy, xx = np.mgrid[:SIZE, :SIZE]
    holes = [(xx-x)**2 + (yy-y)**2 <= 3**2 for x, y in CENTRES]
    reports, video_rows, invariance = {}, {}, []
    for side, frames in data.items():
        patches = np.stack([r[0] for r in frames])
        valid = np.stack([r[1] for r in frames])
        excluded = np.stack([bright_exclusion(p, v) for p, v in zip(patches, valid)])
        donor_valid = valid & ~excluded
        old_excluded = np.stack([dilate(reflection_candidates(p @ LUMA, v), 5)
                                 for p, v in zip(patches, valid)])
        rows, panels, candidate_rows = [], [], []
        temporal = {name: [None] * len(frames) for name in ('fixed', 'aligned')}
        for i in range(len(frames)):
            # Fixed lower hole in the sequence review, never selected by lowest error.
            review = [tile(patches[i]), tile(patches[i], holes[3]), tile(patches[i], holes[3])]
            for h, hole in enumerate(holes):
                if not valid[i][hole].all() or old_excluded[i][hole].any():
                    continue
                guard = footprint_guard(frames[i][2], frames[i][3], hole)
                results = {name: recover(patches, valid, donor_valid, i, hole, guard, aligned)
                           for name, aligned in [('fixed', False), ('aligned', True)]}
                common = results['fixed'][1] & results['aligned'][1]
                row = {'frame': i, 'centre': CENTRES[h], 'hole_pixels': int(hole.sum()),
                       'guard_pixels': int(guard.sum())}
                for name, (prediction, support, confident, info) in results.items():
                    row[name] = {'all': measure(prediction, patches[i], support),
                                 'paired': measure(prediction, patches[i], common),
                                 'confident': measure(prediction, patches[i], confident), 'fit': info}
                rows.append(row)
                if h == 3:
                    for k, name in enumerate(('fixed', 'aligned'), 1):
                        pred, support, conf, _ = results[name]
                        shown = patches[i].copy()
                        shown[support] = pred[support]
                        review[k] = tile(shown, hole, support)
                        # Use identical pixels for the temporal comparison of the two methods.
                        temporal[name][i] = (pred, common)
                if i in (21, 79, 120) and h == 3:
                    altered = patches.copy()
                    altered[i][guard] = np.array([.99, .001, .7], np.float32)
                    # Recompute all data-dependent masks, not just a frozen selection.
                    altered_dv = valid & ~np.stack([bright_exclusion(p, v)
                                                   for p, v in zip(altered, valid)])
                    for name, aligned in [('fixed', False), ('aligned', True)]:
                        test = recover(altered, valid, altered_dv, i, hole, guard, aligned)
                        ref = results[name]
                        ok = (all(np.array_equal(a, b) for a, b in zip(test[:3], ref[:3]))
                              and test[3] == ref[3])
                        if not ok:
                            raise AssertionError('Hidden RGB entered prediction')
                        invariance.append({'side': side, 'frame': i, 'method': name,
                                           'supported_pixels': int(ref[1].sum()), 'passed': ok})
            # Candidate glints are not rewritten: measure support and uncertainty only.
            glint = reflection_candidates(patches[i] @ LUMA, valid[i])
            if glint.any():
                guard = footprint_guard(frames[i][2], frames[i][3], glint)
                _, support, conf, fit_info = recover(patches, valid, donor_valid, i, glint, guard, True)
                candidate_rows.append({'frame': i, 'candidate_pixels': int(glint.sum()),
                                       'supported_pixels': int(support.sum()),
                                       'confident_pixels': int(conf.sum()), 'fit': fit_info})
            panels.append(np.concatenate(review, 1))
            if i % 25 == 0:
                print(side, i, 'eligible tests', len(rows), flush=True)
        missing = recover(patches, np.zeros_like(valid), np.zeros_like(valid), 21,
                          holes[3], holes[3], True)
        if missing[1].any() or missing[2].any():
            raise AssertionError('Missing evidence produced support')
        reports[side] = {'eligible_patches': len(rows),
                        'summary': {name: {subset: summarise(rows, name, subset)
                                          for subset in ('all', 'paired', 'confident')}
                                    for name in ('fixed', 'aligned')},
                        'temporal_lower_hole': {name: temporal_error(preds, patches)
                                                for name, preds in temporal.items()},
                        'candidate_glints': candidate_rows, 'per_patch': rows}
        video_rows[side] = panels
    checks.update({'withheld_rgb_invariance': invariance, 'missing_support': 'passed'})
    if not invariance:
        raise AssertionError('No real-data invariance tests ran')
    if not any(r['method'] == 'aligned' and r['supported_pixels'] for r in invariance):
        raise AssertionError('Aligned independence checks only tested abstentions')
    seq = out / 'holdout_review'
    seq.mkdir(exist_ok=True)
    for i in range(len(tour)):
        frame = np.concatenate([video_rows[s][i] for s in ('left', 'right')], 0)
        frame = cv2.copyMakeBorder(frame, 48, 0, 0, 0, cv2.BORDER_CONSTANT)
        cv2.putText(frame, '%03d  truth | fixed donors | aligned donors' % i, (8, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
        cv2.putText(frame, 'Hidden lower patch ONLY. Purple = unavailable. NOT reflection removal.',
                    (8, 39), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1)
        write(seq / ('%06d.png' % i), frame)
    subprocess.run([sys.executable, str(Path(__file__).with_name('review_video.py')),
                    '--shot', str(shot.dir), '--name', 'r9_iris/holdout_review', '--fps', 'shot',
                    '--out', 'r9_iris_withheld_comparison.mp4'], check=True)
    cap = cv2.VideoCapture(str(shot.final / 'r9_iris_withheld_comparison.mp4'))
    fps = cap.get(cv2.CAP_PROP_FPS)
    count, tiles, sheets = 0, [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        tiles.append(cv2.resize(frame, (492, 360), interpolation=cv2.INTER_AREA))
        count += 1
        if len(tiles) == 15:
            sheet = np.concatenate([np.concatenate(tiles[j:j+3], 1) for j in range(0, 15, 3)], 0)
            path = out / ('review_%03d.jpg' % (count-15))
            write(path, sheet)
            sheets.append(str(path.relative_to(shot.dir)))
            tiles = []
    cap.release()
    if count != len(tour) or tiles or abs(fps-float(shot.frame_rate)) > .001:
        raise RuntimeError('Incomplete or mistimed review encode')
    after_hashes = {str(p): digest(p) for p in protected}
    if before_hashes != after_hashes:
        raise AssertionError('Production input or baseline changed')
    checks['protected_files_sha256_unchanged'] = len(protected)
    checks['protected_manifest_digest'] = hashlib.sha256(json.dumps(before_hashes, sort_keys=True).encode()).hexdigest()
    report = {'status': 'diagnostic only; no reflection repair enabled', 'frames': len(tour),
              'method': {'alignment': 'masked affine ECC, 60 iterations, Gaussian size 1',
                         'bounds': 'singular values .90..1.10, max corner displacement 4 grid pixels',
                         'donors': '12 shortlisted by context; best 7 fits; at least 3 per pixel; target +/-3 excluded',
                         'confidence': {'max_context_display_mae': MAX_CONTEXT_ERROR, 'max_donor_mad': MAX_DONOR_MAD},
                         'control': 'same conservative masks and native-footprint guard, no residual alignment',
                         'reference': 'https://docs.opencv.org/4.13.0/dc/d6b/group__video__track.html'},
              'eyes': reports, 'checks': checks,
              'review': {'decoded_frames': count, 'fps': fps, 'sheets': sheets,
                         'inspection': 'pending', 'artist_verdict': 'pending'},
              'limitations': ['Detector exclusions are not ground-truth material masks.',
                              'Eligibility uses original RGB; prediction does not use hidden RGB.',
                              'Cached landmarks are held fixed, not retracked after the perturbation.',
                              'Affine correspondence does not model refraction, pupil motion or exposure blur.',
                              'Donor agreement can be confidently wrong; compare hidden-pixel errors.',
                              'No original catchlight removal or lid/source occlusion solution is delivered.',
                              'Hair, lips and all production compositing remain unchanged.']}
    write_report(report, shot)
    print(json.dumps({s: r['summary'] for s, r in reports.items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()
