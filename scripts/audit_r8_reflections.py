"""Isolate added gloss and inspect temporal iris evidence, without altering source pixels.

Requires --shot and the separately computed ratio_r8_no_added_specular sequence.
Temporal lower-radiance observations are candidate donors, NOT recovered diffuse texture:
iris-centre/scale registration does not correct gaze, torsion, pupil dilation or refraction.
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import argparse
import json
import subprocess
import sys
from pathlib import Path
import cv2
import numpy as np
from shot_paths import add_shot_argument, from_args
from color_pipeline import linear_to_srgb

LUMA = np.array([.2126, .7152, .0722], np.float32)


def read(path):
    result = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if result is None:
        raise RuntimeError('Cannot read %s' % path)
    return result


def write(path, pixels):
    if not cv2.imwrite(str(path), pixels):
        raise RuntimeError('Cannot write %s' % path)


def polygon(shape, points):
    result = np.zeros(shape, np.uint8)
    cv2.fillPoly(result, [np.rint(points[:, :2]).astype(np.int32)], 1)
    return result.astype(bool)


def reflection_candidates(lum, valid):
    """Reject bright components reaching the iris/aperture boundary (often sclera)."""
    result = np.zeros_like(valid)
    v = lum[valid]
    if len(v) < 150:
        return result
    bright = valid & (lum > max(3 * np.median(v), np.percentile(v, 90)))
    interior = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(bright.astype(np.uint8))
    for label in range(1, count):
        component = labels == label
        if stats[label, cv2.CC_STAT_AREA] >= 3 and not (component & ~interior).any():
            result |= component
    return result


def recover_patch(patches, valid, excluded, index, hole):
    """Photographed donors ranked using ONLY visible context outside a withheld hole.

    This is a diagnostic reconstruction, not an output-path repair. No target values inside
    the hole take part in ranking. No temporal brightness minimum or colour fitting is used.
    """
    common = valid & valid[index] & ~excluded & ~excluded[index] & ~hole
    counts = common.sum((1, 2))
    delta = np.abs(patches - patches[index]).mean(-1)
    errors = (delta * common).sum((1, 2)) / np.maximum(counts, 1)
    errors[counts < 150] = np.inf
    errors[max(0, index-3):index+4] = np.inf
    order = np.argsort(errors)[:7]
    order = order[np.isfinite(errors[order])]
    if len(order) < 3:
        return patches[index].copy(), np.zeros_like(hole), []
    donors = patches[order].copy()
    usable = valid[order] & ~excluded[order]
    support = usable.sum(0) >= 3
    # Median across available donors; use masked arrays to avoid all-NaN warnings.
    mask = np.broadcast_to(~usable[..., None], donors.shape)
    estimate = np.ma.median(np.ma.array(donors, mask=mask), axis=0).filled(0).astype(np.float32)
    return estimate, support & hole, order.tolist()


def iris_patch(fg, points, topology, side, size=41):
    """Register an iris to [-1.1,1.1] radii; exclude native eyelid boundaries FIRST."""
    ids = topology['iris_' + side]
    centre = points[ids[0], :2]
    radius = float(np.linalg.norm(points[ids[1:], :2] - centre, axis=1).mean())
    axis = np.linspace(-1.1, 1.1, size, dtype=np.float32)
    yy, xx = np.meshgrid(axis, axis, indexing='ij')
    mx, my = centre[0] + xx * radius, centre[1] + yy * radius
    opening = polygon(fg.shape[:2], points[topology['eyelid_' + side]])
    opening = cv2.erode(opening.astype(np.uint8), np.ones((3, 3), np.uint8))
    visible = cv2.remap(opening, mx, my, cv2.INTER_NEAREST) > 0
    visible &= xx * xx + yy * yy < .85 ** 2
    patch = cv2.remap(fg, mx, my, cv2.INTER_LINEAR)
    return patch, visible, radius


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_shot_argument(ap)
    args = ap.parse_args()
    shot = from_args(args)
    result_name = 'ratio_r8_no_added_specular'
    result = shot.composite / result_name
    manifest = json.loads((result / 'composite_manifest.json').read_text())
    if (manifest['plate_specular']['mode'] != 'keep' or manifest['catchlight']['gain'] != 0
            or manifest['specular']['gain'] != 0):
        raise RuntimeError('Expected a plate-preserving, added-specular-only ablation')
    tour = json.loads(shot.log('stress_proxy.json').read_text())['per_frame']
    if len(tour) != manifest['frames']:
        raise RuntimeError('Tour and output length differ')
    lm = np.load(shot.relight / 'face_landmarks.npz')
    topo = json.loads((shot.relight / 'face_landmarks.json').read_text())['topology']
    out = shot.composite / 'r8_audit'
    out.mkdir(exist_ok=True)
    rows, eyes = [], {s: [] for s in ('left', 'right')}
    crops = {'hair': [780, 32, 460, 236]}  # Historical reporter ROI; labelled in report.
    # The head crop above is specific to this test bed. Refuse silent reuse elsewhere.
    if shot.dir.name != 'reporter_a' or shot.dir.parent.parent.name != 'reporter_test':
        raise RuntimeError('This audit uses the reporter crown ROI; generalise before reuse')
    pts = lm['points'][lm['found']]
    for name, indices in [('mouth', topo['lips_outer']),
                          ('eyes', topo['eyelid_left'] + topo['eyelid_right'])]:
        p = pts[:, indices, :2].reshape(-1, 2)
        lo = np.maximum(np.floor(p.min(0) - 40), 0).astype(int)
        hi = np.ceil(p.max(0) + 40).astype(int)
        crops[name] = [*lo.tolist(), *((hi - lo + 1) // 2 * 2).tolist()]
    for rec in tour:
        n, fi = rec['out_frame'], rec['shot_frame']
        if not lm['found'][fi]:
            raise RuntimeError('Missing landmarks: %d' % fi)
        P = lm['points'][fi]
        fg = read(shot.fg(fi))[..., :3][..., ::-1].astype(np.float32)
        alpha = read(shot.alpha(fi))
        if alpha.ndim == 3:
            alpha = alpha[..., 0]
        before = read(shot.composite / 'ratio' / ('%06d.png' % n))
        after = read(result / ('%06d.png' % n))
        if before.shape != after.shape or before.shape[:2] != fg.shape[:2]:
            raise RuntimeError('Audit requires matching native resolution')
        delta = before.astype(np.float32) - after.astype(np.float32)
        L = fg @ LUMA
        mouth = polygon(L.shape, P[topo['lips_inner']])
        lips = polygon(L.shape, P[topo['lips_outer']]) & ~mouth
        opening = np.zeros_like(mouth)
        iris = np.zeros_like(mouth)
        yy, xx = np.mgrid[:L.shape[0], :L.shape[1]]
        for side in eyes:
            opening |= polygon(L.shape, P[topo['eyelid_' + side]])
            ids = topo['iris_' + side]
            c = P[ids[0], :2]
            r = np.linalg.norm(P[ids[1:], :2] - c, axis=1).mean()
            iris |= (xx - c[0]) ** 2 + (yy - c[1]) ** 2 < (r * 1.15) ** 2
            patch, visible, radius = iris_patch(fg, P, topo, side)
            eyes[side].append((patch, visible, radius))
        hair = np.zeros_like(mouth)
        x, y, w, h = crops['hair']
        hair[y:y+h, x:x+w] = True
        hair &= alpha > .99
        masks = {'lips': lips, 'hair': hair, 'iris': iris & opening}
        for name, m in [('teeth', mouth), ('sclera', opening & ~iris)]:
            m = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            masks[name] = m & (L >= np.median(L[m])) if m.any() else m
        row = {'output_frame': n, 'shot_frame': fi, 'regions': {}}
        for name, m in masks.items():
            d = delta[m]
            row['regions'][name] = {'pixels': int(m.sum()),
                'mean_removed_display_levels': float(d.mean()) if d.size else None,
                'p99_abs_display_change': float(np.percentile(np.abs(d), 99)) if d.size else None,
                'max_abs_display_change': float(np.abs(d).max()) if d.size else None}
        rows.append(row)
        if n in (0, 21, 51, 79, 120, 149):
            for name, (x, y, w, h) in crops.items():
                tiles = [a[y:y+h, x:x+w] for a in (before, after)]
                write(out / ('%06d_%s.png' % (n, name)), np.concatenate(tiles, 1))
        if n % 40 == 0:
            print('audited', n, flush=True)

    eye_report = {}
    eye_frames = []
    for side, data in eyes.items():
        patches = np.stack([d[0] for d in data])
        valid = np.stack([d[1] for d in data])
        lum = patches @ LUMA
        # Exclude nearly shut eyes; a few dark pixels under the lid are not a clean iris.
        valid[valid.sum((1, 2)) < 150] = False
        glints = np.zeros_like(valid)
        for i in range(len(data)):
            glints[i] = reflection_candidates(lum[i], valid[i])
        excluded = np.stack([cv2.dilate(g.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
                             for g in glints])
        donor_rows = []
        for i in range(len(data)):
            # Count lower observations from distinct frames outside a +/-3-frame guard.
            candidate = valid & ~excluded & (lum < .5 * lum[i])
            candidate[max(0, i-3):i+4] = False
            support = candidate.sum(0)
            g = glints[i]
            donor_rows.append({'output_frame': tour[i]['out_frame'],
                'glint_grid_pixels': int(g.sum()),
                'fraction_with_three_lower_observations': float((support[g] >= 3).mean()) if g.any() else None})
        samples = [r['fraction_with_three_lower_observations'] for r in donor_rows
                   if r['fraction_with_three_lower_observations'] is not None]
        eye_report[side] = {'median_native_iris_radius_px': float(np.median([d[2] for d in data])),
            'frames_with_sufficient_opening': int(valid.any((1, 2)).sum()),
            'frames_with_glint_candidates': len(samples),
            'mean_glint_fraction_with_three_lower_observations': float(np.mean(samples)) if samples else None,
            'per_frame': donor_rows}
        holdouts = []
        gy, gx = np.mgrid[:lum.shape[1], :lum.shape[2]]
        for i in range(len(data)):
            # Four preselected locations, not selected by reconstruction quality. Artificially
            # withhold a roughly glint-sized disc on known, non-candidate iris pixels.
            for cx, cy in [(14, 20), (26, 20), (20, 14), (20, 26)]:
                hole = (gx-cx)**2 + (gy-cy)**2 <= 3**2
                if not valid[i][hole].all() or excluded[i][hole].any():
                    continue
                prediction, supported, donors = recover_patch(patches, valid, excluded, i, hole)
                if not supported.any():
                    continue
                truth = patches[i][supported]
                error = np.abs(prediction[supported] - truth)
                display_error = np.abs(linear_to_srgb(prediction[supported]) - linear_to_srgb(truth)) * 255
                holdouts.append({'output_frame': tour[i]['out_frame'], 'centre_grid_xy': [cx, cy],
                    'coverage': float(supported.sum() / hole.sum()), 'donor_indices': donors,
                    'relative_rgb_absolute_error': float(error.sum() / max(truth.sum(), 1e-8)),
                    'mean_abs_display_levels': float(display_error.mean())})
        eye_report[side]['withheld_known_pixels'] = {
            'patches': len(holdouts),
            'relative_rgb_absolute_error_p50_p95': np.percentile(
                [r['relative_rgb_absolute_error'] for r in holdouts], [50, 95]).tolist() if holdouts else None,
            'mean_abs_display_levels_p50_p95': np.percentile(
                [r['mean_abs_display_levels'] for r in holdouts], [50, 95]).tolist() if holdouts else None,
            'method': 'Four fixed 3-grid-radius holes per open frame, outside detected glints; '
                      'top 7 donors by visible-context RGB error, excluding +/-3 frames; '
                      'median from at least 3 valid non-glint donors. Hole RGB never enters ranking.',
            'per_patch': holdouts}
        eye_frames.append((patches, valid, glints))
    # Display actual registered samples and an overlay, never a manufactured clean iris.
    eye_dir = out / 'iris_tracking'
    eye_dir.mkdir(exist_ok=True)
    for i in range(len(tour)):
        panels = []
        for patches, valid, glints in eye_frames:
            bgr = (np.clip(linear_to_srgb(patches[i]), 0, 1)[..., ::-1] * 255).astype(np.uint8)
            overlay = bgr.copy()
            overlay[~valid[i]] //= 4
            overlay[glints[i]] = [0, 255, 255]
            panels.extend([cv2.resize(a, (246, 246), interpolation=cv2.INTER_NEAREST)
                           for a in (bgr, overlay)])
        frame = np.concatenate(panels, 1)
        frame = cv2.copyMakeBorder(frame, 32, 0, 0, 0, cv2.BORDER_CONSTANT)
        cv2.putText(frame, '%03d  left: plate | candidates    right: plate | candidates' % i,
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
        write(eye_dir / ('%06d.png' % i), frame)
    summary = {name: {'mean_removed_display_levels': float(np.mean([
        r['regions'][name]['mean_removed_display_levels'] for r in rows
        if r['regions'][name]['pixels']])), 'maximum_display_change': max(
        r['regions'][name]['max_abs_display_change'] or 0 for r in rows)} for name in masks}
    report = {'frames': len(rows), 'ablation': summary, 'iris_temporal_evidence': eye_report,
        'crops_xywh': crops, 'per_frame': rows,
        'limitations': [
            'Ablation compares 8-bit display images; sub-code-value changes are not resolved.',
            'Crown rectangle includes some forehead. Tooth/sclera masks are anatomical candidates.',
            'Iris grid is an interpolated diagnostic, not 41 pixels of captured detail.',
            'Lower observations can be shadow, pupil motion, blur, refraction or misregistration.',
            'Neither donor counts nor bright-pixel masks prove physical diffuse/specular separation.',
            'No source pixels were changed; no temporal repair or new catchlight is enabled.']}
    shot.log('r8_reflection_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'ablation': summary, 'eyes': {s: {k: v for k, v in d.items()
          if k not in ('per_frame', 'withheld_known_pixels')} for s, d in eye_report.items()}}, indent=2), flush=True)

    videos = []
    for name, crop in crops.items():
        filename = 'r8_%s_added_gloss_ablation.mp4' % name
        subprocess.run([sys.executable, str(Path(__file__).with_name('review_video.py')),
            '--shot', str(shot.dir), '--name', result_name, '--compare', 'ratio',
            '--compare-label', 'round 4', '--label', 'added gloss OFF', '--out', filename,
            '--crop', *map(str, crop)], check=True)
        videos.append(filename)
    subprocess.run([sys.executable, str(Path(__file__).with_name('review_video.py')),
        '--shot', str(shot.dir), '--name', 'r8_audit/iris_tracking', '--fps', 'shot',
        '--out', 'r8_iris_tracking_evidence.mp4'], check=True)
    videos.append('r8_iris_tracking_evidence.mp4')
    decoded = []
    for filename in videos:
        cap = cv2.VideoCapture(str(shot.final / filename))
        count, tiles, pages, overviews, overview_tiles = 0, [], [], [], []
        fps = cap.get(cv2.CAP_PROP_FPS)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # Six consecutive native-pixel rows per sheet, preserving all encoded frames.
            cv2.putText(frame, str(count), (3, frame.shape[0]-4), cv2.FONT_HERSHEY_SIMPLEX,
                        .4, (255, 255, 255), 1)
            tiles.append(frame)
            overview_tiles.append(cv2.resize(frame, (492, round(frame.shape[0]*492/frame.shape[1])),
                                             interpolation=cv2.INTER_AREA))
            count += 1
            if len(overview_tiles) == 30:
                overview = np.concatenate([np.concatenate(overview_tiles[j:j+3], 1)
                                           for j in range(0, 30, 3)], 0)
                p = out / ('overview_%s_%03d.jpg' % (Path(filename).stem, count-30))
                write(p, overview)
                overviews.append(str(p.relative_to(shot.dir)))
                overview_tiles = []
            if len(tiles) == 6:
                path = out / ('%s_%03d.png' % (Path(filename).stem, count-6))
                write(path, np.concatenate(tiles, 0))
                pages.append(str(path.relative_to(shot.dir)))
                tiles = []
        cap.release()
        expected_fps = float(shot.frame_rate) if 'tracking' in filename else 24
        if count != len(tour) or abs(fps - expected_fps) > .001:
            raise RuntimeError('Incomplete or mistimed video: %s' % filename)
        if tiles:
            path = out / ('%s_%03d.png' % (Path(filename).stem, count-len(tiles)))
            write(path, np.concatenate(tiles, 0))
            pages.append(str(path.relative_to(shot.dir)))
        decoded.append({'video': filename, 'decoded_frames': count, 'fps': fps, 'sheets': pages,
                        'overview_sheets': overviews, 'overview_pair_width_px': 492})
    shot.log('r8_video_review.json').write_text(json.dumps({'decoded': decoded,
        'inspection_method': 'Full decode and consecutive-frame sheets; not real-time playback.',
        'human_verdict': 'pending'}, indent=2))


if __name__ == '__main__':
    main()
