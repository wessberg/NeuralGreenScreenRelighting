"""Known-truth tests for a shot-independent temporal iris core; requires --shot.

These are synthetic positive/negative controls with ORACLE geometry and masks.
They demonstrate the mechanism, not segmentation or recovery on a real performer.
The supplied shot only locates evidence and native-source comparison artefacts.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from shot_paths import add_shot_argument, from_args
from temporal_iris import (IrisGeometry, footprint_valid, reconstruct,
                           apply_eye_repair, repair_strength)


def write(path, im):
    if not cv2.imwrite(str(path), im):
        raise RuntimeError('Cannot write ' + str(path))


def tile(im, title):
    im = cv2.resize(np.clip(im, 0, 255).astype(np.uint8), (480, 288), interpolation=cv2.INTER_NEAREST)
    im = cv2.copyMakeBorder(im, 30, 0, 0, 0, cv2.BORDER_CONSTANT)
    cv2.putText(im, title, (8, 21), 0, .43, (255, 255, 255), 1, cv2.LINE_AA)
    return im


def scene(geometry, frame, count, locked, colour):
    yy, xx = np.mgrid[:96, :160]
    xy = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)
    uv = geometry.material(xy).reshape(96, 160, 2)
    radius, angle = uv[..., 0], uv[..., 1]
    iris = (radius >= 0) & (radius <= 1)
    texture = 7*np.sin(9*angle+3*radius) + 4*np.sin(19*angle-5*radius) + 5*np.cos(14*radius+2*angle)
    texture += 3*np.sin(33*angle+7*radius)
    clean = np.full((96, 160, 3), [170, 175, 180], np.float32)
    clean[iris] = (np.asarray(colour)[None] + texture[..., None]*np.array([.8, .9, 1.0]))[iris]
    clean[radius < 0] = [9, 10, 11]
    # Known eyelid aperture in native image space; shadows/material boundaries are exact here.
    aperture = np.abs(yy-geometry.centre[1]) < geometry.axes[1]*.83
    clean[~aperture] = [65, 95, 135]
    if locked:
        difference = np.arctan2(np.sin(angle), np.cos(angle))
        reflection = iris & (radius > .22) & (radius < .76) & (np.abs(difference) < .32)
    else:
        phase = frame*2*np.pi/count
        centre = geometry.native(np.array([[.48, phase]], np.float32))[0]
        # A moving rectangle in native coordinates, explicitly a synthetic softbox.
        reflection = (np.abs(xx-centre[0]) <= geometry.axes[0]*.16)
        reflection &= np.abs(yy-centre[1]) <= geometry.axes[1]*.21
        reflection &= iris
    reflection &= aperture
    observed = clean.copy()
    observed[reflection] = .15*clean[reflection]+.85*255
    excluded = cv2.dilate(reflection.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    visible = iris & aperture & ~excluded
    return clean, observed, visible, reflection, xy


def percentile(values):
    return np.percentile(values, [50, 95]).tolist() if values else None


def fallback_unit_checks():
    source = np.random.default_rng(11).uniform(0, 1, (12, 16, 3)).astype(np.float32)
    mask = np.zeros(source.shape[:2], bool)
    mask[4:7, 6:9] = True
    estimate = {'rgb': np.ones((9, 3), np.float32)*.1,
                'supported': np.ones(9, bool), 'agreement': np.ones(9, bool)}
    cases = []
    for detected, matched in [(False, False), (False, True), (True, False)]:
        result, log = apply_eye_repair(source, mask, estimate,
            geometry_reliable=detected, correspondence_reliable=matched)
        assert np.array_equal(result, source) and log['mode'] == 'source_keyed'
        cases.append(log['reason'])
    for key in ('supported', 'agreement', 'rgb'):
        bad = {k: v.copy() for k, v in estimate.items()}
        bad[key][0] = np.nan if key == 'rgb' else False
        result, log = apply_eye_repair(source, mask, bad,
            geometry_reliable=True, correspondence_reliable=True)
        assert np.array_equal(result, source)
        cases.append(key+'_rejected')
    repaired, log = apply_eye_repair(source, mask, estimate,
        geometry_reliable=True, correspondence_reliable=True)
    assert np.array_equal(repaired[~mask], source[~mask])
    assert np.array_equal(repaired[mask], estimate['rgb']) and log['mode'] == 'temporal_repair'
    for reliable in (np.zeros(9, bool), np.ones(9, bool),
                     np.array([1, 1, 0, 1, 0, 1, 1, 1, 0], bool)):
        weights = repair_strength(reliable)
        assert np.all(weights[~reliable] == 0) and np.all((weights >= 0) & (weights <= 1))
        assert not len(weights) > 1 or np.max(np.abs(np.diff(weights))) <= 1/3+1e-6
    return {'rejection_cases': cases, 'successful_repair_only_changes_mask': True,
            'temporal_gate_unreliable_frames_exact_zero': True}


def run_case(out, name, radius, locked, colour):
    n = 48
    geometries = []
    for i in range(n):
        phase = i*2*np.pi/n
        geometries.append(IrisGeometry((80+5*np.sin(phase), 48+3*np.cos(phase)),
            (radius*(1+.08*np.sin(phase)), radius*(.88+.07*np.cos(phase))),
            .15*np.sin(phase), .36+.10*np.sin(phase*2), .16*np.cos(phase)))
    fixed = [replace(g, pupil_ratio=.36, texture_rotation=0) for g in geometries]
    scenes = [scene(g, i, n, locked, colour) for i, g in enumerate(geometries)]
    images = [s[1] for s in scenes]
    masks = [s[2] for s in scenes]
    rows, frames, accurate = [], [], []
    max_roundtrip = 0
    hidden_checks = []
    for i, (truth, observed, visible, reflection, xy) in enumerate(scenes):
        target = reflection.ravel()
        query = xy[target]
        results = [reconstruct(images, model, masks, list(range(n)), i, query, max_mad=2)
                   for model in (fixed, geometries)]
        accurate.append(results[1])
        max_roundtrip = max(max_roundtrip, float(np.max(np.abs(geometries[i].native(
            geometries[i].material(query))-query))) if len(query) else 0)
        if i in (0, 12, 24, 36):
            altered = list(images)
            altered[i] = np.random.default_rng(i).uniform(0, 255, images[i].shape).astype(np.float32)
            again = reconstruct(altered, geometries, masks, list(range(n)), i, query, max_mad=2)
            assert all(np.array_equal(results[1][key], again[key]) for key in results[1])
            hidden_checks.append(i)
        truth_pixels = truth.reshape(-1, 3)[target]
        record = {'frame': i, 'reflection_native_pixels': int(target.sum()), 'arms': {}}
        common = results[0]['supported'] & results[1]['supported']
        views = []
        for arm, result in zip(('fixed_pupil_torsion', 'varying_pupil_torsion'), results):
            good = result['supported']
            error = np.abs(result['rgb']-truth_pixels).mean(1)
            record['arms'][arm] = {'supported': int(good.sum()), 'agreement': int(result['agreement'].sum()),
                'display_mae': float(error[good].mean()) if good.any() else None,
                'matched_display_mae': float(error[common].mean()) if common.any() else None,
                'agreement_display_mae': float(error[result['agreement']].mean()) if result['agreement'].any() else None}
            view = observed.copy().reshape(-1, 3)
            target_ids = np.flatnonzero(target)
            view[target_ids] = [170, 0, 170]
            view[target_ids[good]] = result['rgb'][good]
            views.append(view.reshape(observed.shape))
        rows.append(record)
        header = np.zeros((34, 960, 3), np.uint8)
        cv2.putText(header, '%s  %02d/47  SYNTHETIC / known geometry + masks' % (name, i),
                    (8, 24), 0, .58, (255, 255, 255), 1, cv2.LINE_AA)
        panel = np.concatenate([header,
            np.concatenate([tile(observed, 'Input with synthetic reflection'), tile(truth, 'Known clean truth')], 1),
            np.concatenate([tile(views[0], 'Fixed pupil/torsion / purple = unsupported'),
                            tile(views[1], 'Varying pupil/torsion / purple = unsupported')], 1)], 0)
        frames.append(panel)
    assert max_roundtrip < 1e-4
    summary = {'radius_px': radius, 'locked_to_material': locked,
        'reflection_pixels': sum(r['reflection_native_pixels'] for r in rows), 'arms': {},
        'coordinate_roundtrip_max_px': max_roundtrip, 'target_rgb_independence_frames': hidden_checks}
    for arm in rows[0]['arms']:
        values = [r['arms'][arm] for r in rows]
        summary['arms'][arm] = {'supported': sum(v['supported'] for v in values),
            'agreement': sum(v['agreement'] for v in values),
            'per_frame_display_mae_p50_p95': percentile([v['display_mae'] for v in values if v['display_mae'] is not None]),
            'matched_per_frame_display_mae_p50_p95': percentile([v['matched_display_mae'] for v in values if v['matched_display_mae'] is not None]),
            'agreement_per_frame_display_mae_p50_p95': percentile([v['agreement_display_mae'] for v in values if v['agreement_display_mae'] is not None])}
    return summary, rows, frames, scenes, accurate


def encode_decode(shot, out, name, count, fps_arg='24'):
    subprocess.run([sys.executable, str(Path(__file__).with_name('review_video.py')),
        '--shot', str(shot.dir), '--name', 'r11_atlas/'+name, '--fps', fps_arg,
        '--out', 'r11_'+name+'.mp4'], check=True)
    video = shot.final/('r11_'+name+'.mp4')
    cap = cv2.VideoCapture(str(video))
    decoded, thumbs, sheets = 0, [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        thumb = cv2.resize(frame, (240, 168), interpolation=cv2.INTER_AREA)
        cv2.putText(thumb, str(decoded), (4, 164), 0, .38, (255, 255, 255), 1)
        thumbs.append(thumb)
        decoded += 1
        if len(thumbs) == 24 or decoded == count:
            while len(thumbs) % 6:
                thumbs.append(np.zeros_like(thumbs[0]))
            path = out/('%s_decoded_%03d.jpg' % (name, decoded))
            write(path, np.concatenate([np.concatenate(thumbs[k:k+6], 1) for k in range(0, len(thumbs), 6)], 0))
            sheets.append(str(path.relative_to(shot.dir)))
            thumbs = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    expected_fps = float(shot.frame_rate) if fps_arg == 'shot' else float(fps_arg)
    if decoded != count or abs(fps-expected_fps) > .001:
        raise RuntimeError('Incomplete/mistimed review encode')
    return {'file': str(video.relative_to(shot.dir)), 'decoded_frames': decoded,
            'fps': fps, 'sheets': sheets, 'inspection': 'pending'}


def fallback_check(shot, out, scenes, accurate):
    directory = out/'synthetic_fallback'
    directory.mkdir(exist_ok=True)
    detection = np.ones(len(scenes), bool)
    detection[12:18] = False
    detection[36:42] = False
    complete = np.array([r['agreement'].all() and len(r['agreement']) > 0 for r in accurate])
    strengths = repair_strength(detection & complete)
    assert np.all(strengths[~detection] == 0)
    rows = []
    hard_delta, tapered_delta = [], []
    for i, ((truth, source, visible, mask, xy), estimate) in enumerate(zip(scenes, accurate)):
        repaired, rec = apply_eye_repair(source, mask, estimate,
            geometry_reliable=bool(detection[i]), correspondence_reliable=True, strength=float(strengths[i]))
        hard, _ = apply_eye_repair(source, mask, estimate,
            geometry_reliable=bool(detection[i]), correspondence_reliable=True)
        if not detection[i] or not complete[i]:
            assert np.array_equal(repaired, source)
        assert np.array_equal(repaired[~mask], source[~mask])
        hard_delta.append(float(np.abs(hard-source).mean()))
        tapered_delta.append(float(np.abs(repaired-source).mean()))
        rec['frame'] = i
        rec['detection_reliable'] = bool(detection[i])
        rec['complete_agreement'] = bool(complete[i])
        rows.append(rec)
        panel = np.concatenate([
            np.concatenate([tile(source, 'SYNTHETIC input / frame %02d' % i),
                            tile(repaired, '%s / weight %.2f' % (rec['mode'], rec['strength']))], 1),
            np.concatenate([tile(truth, 'Known clean truth'),
                            tile(hard, 'Untapered switch control')], 1)], 0)
        write(directory/('%06d.png' % i), panel)
    # The taper test is about correction strength, not temporal photorealism.
    report = {'forced_bad_detection_frames': np.flatnonzero(~detection).tolist(),
              'fallback_frames': sum(r['mode'] == 'source_keyed' for r in rows),
              'repaired_frames': sum(r['mode'] == 'temporal_repair' for r in rows),
              'max_adjacent_strength_change': float(np.max(np.abs(np.diff(strengths)))),
              'max_adjacent_mean_correction_change': {'hard': float(np.max(np.abs(np.diff(hard_delta)))),
                  'tapered': float(np.max(np.abs(np.diff(tapered_delta))))},
              'per_frame': rows, 'fallback_exact': True, 'outside_mask_exact': True,
              'video': encode_decode(shot, out, 'synthetic_fallback', len(scenes))}
    # Real source-keyed fallback branch: no new real iris correspondence is certified.
    # Test both rejection gates over every selected frame; no repaired plate is written.
    points_path = shot.relight/'face_landmarks.npz'
    if not points_path.exists():
        report['native_fallback'] = {'status': 'not run: no selected-shot landmark cache'}
        return report
    lm = np.load(points_path)
    topo = json.loads((shot.relight/'face_landmarks.json').read_text())['topology']
    eye_ids = topo['eyelid_left']+topo['eyelid_right']
    native_dir = out/'native_fallback'
    native_dir.mkdir(exist_ok=True)
    hashes, changed, channels = {}, 0, 0
    for i in range(shot.frame_count):
        fg = cv2.imread(str(shot.fg(i)), cv2.IMREAD_UNCHANGED)
        if fg is None or fg.ndim != 3 or fg.shape[2] != 3:
            raise RuntimeError('Expected 3-channel keyed EXR')
        with shot.fg(i).open('rb') as stream:
            hashes[str(shot.fg(i))] = hashlib.file_digest(stream, 'sha256').hexdigest()
        mask = np.zeros(fg.shape[:2], bool)
        # Geometry gate fails before a mask/estimate is used, so absence of detection
        # needs no invented eye location to guarantee preservation of the entire source.
        a, log_a = apply_eye_repair(fg, mask, geometry_reliable=False)
        b, log_b = apply_eye_repair(fg, mask, geometry_reliable=True, correspondence_reliable=False)
        assert np.array_equal(a, fg) and np.array_equal(b, fg)
        changed += log_a['changed_channels']+log_b['changed_channels']
        channels += fg.size*2
        if not lm['found'][i]:
            raise RuntimeError('Native review crop lacks landmarks; fallback is preserved but crop cannot be guessed')
        p = lm['points'][i, eye_ids, :2]
        lo = np.maximum(np.floor(p.min(0)-20), 0).astype(int)
        hi = np.minimum(np.ceil(p.max(0)+20), [fg.shape[1], fg.shape[0]]).astype(int)
        from color_pipeline import linear_to_srgb
        views = [np.clip(linear_to_srgb(v[lo[1]:hi[1], lo[0]:hi[0]]), 0, 1)*255 for v in (fg, b)]
        # Preserve eye-crop aspect; letterbox into tile's 160:96 source canvas.
        panels = []
        for v in views:
            factor = min(160/v.shape[1], 96/v.shape[0])
            small = cv2.resize(v, (max(1, round(v.shape[1]*factor)), max(1, round(v.shape[0]*factor))))
            canvas = np.zeros((96, 160, 3), np.float32)
            top = (96-small.shape[0])//2
            left = (160-small.shape[1])//2
            canvas[top:top+small.shape[0], left:left+small.shape[1]] = small
            panels.append(canvas)
        write(native_dir/('%06d.png' % i), np.concatenate([
            tile(panels[0], 'SOURCE KEYED / %03d' % i), tile(panels[1], 'FALLBACK / identical; no real repair yet')], 1))
    (out/'native_fg_hashes.json').write_text(json.dumps(hashes, indent=2))
    report['native_fallback'] = {'frames': shot.frame_count, 'gates_tested_per_frame': 2,
        'compared_channels': channels, 'changed_channels': changed,
        'note': 'All real frames intentionally fall back: current iris geometry/correspondence is not validated. '
                'This tests preservation, not a working automatic selection of good views.',
        'video': encode_decode(shot, out, 'native_fallback', shot.frame_count, 'shot')}
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_shot_argument(ap)
    shot = from_args(ap.parse_args())
    out = shot.composite/'r11_atlas'
    frames_dir = out/'synthetic_correspondence'
    frames_dir.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    summaries, rows, frame_count = {}, {}, 0
    for name, radius, locked, colour in [('brown_native13_moving', 13, False, [40, 58, 77]),
            ('blue_native30_moving', 30, False, [95, 76, 58]),
            ('brown_native13_locked', 13, True, [40, 58, 77])]:
        summary, records, frames, scenes, accurate = run_case(out, name, radius, locked, colour)
        if name == 'brown_native13_moving':
            fallback_scenes, fallback_results = scenes, accurate
        summaries[name], rows[name] = summary, records
        for frame in frames:
            write(frames_dir/('%06d.png' % frame_count), frame)
            frame_count += 1
        print(name, json.dumps(summary), flush=True)
    # Independent sign checks, with rotations in both directions and nonuniform scales.
    errors = []
    for angle in (-.5, -.2, .2, .5):
        g = IrisGeometry((41, 29), (17, 11), angle, .4, .13)
        uv = np.array([[.2, .3], [.6, 1.5], [.8, -2]], np.float32)
        c, s = np.cos(angle), np.sin(angle)
        r, a = .4+uv[:, 0]*.6, uv[:, 1]+.13
        expected = np.column_stack([41+c*17*r*np.cos(a)-s*11*r*np.sin(a),
                                    29+s*17*r*np.cos(a)+c*11*r*np.sin(a)])
        errors.append(float(np.max(np.abs(g.native(uv)-expected))))
    assert max(errors) < 1e-4
    mask = np.ones((8, 8), bool)
    mask[3, 4] = False
    assert not footprint_valid(mask, np.array([[3.1, 2.9]], np.float32))[0]
    # All donor masks empty must not produce evidence, even with perfect geometry.
    geometry = [IrisGeometry((8, 8), (6, 6), 0, .3)]*10
    empty = reconstruct([np.ones((16, 16, 3))]*10, geometry,
        [np.zeros((16, 16), bool)]*10, list(range(10)), 0, np.array([[12, 8]], np.float32), max_mad=2)
    assert not empty['supported'].any() and not empty['agreement'].any()
    report = {'purpose': 'Synthetic mechanism test ONLY; not a repaired reporter or validated real tracker.',
        'geometry': 'Oracle concentric projected pupil/limbus ellipses, varying dilation and material torsion.',
        'masks': 'Oracle iris/aperture and reflection labels; 1px reflection dilation; all four native donors valid.',
        'reconstruction': 'Every eligible frame sampled per texel; median of >=3 donors separated by >=4 frame IDs; '
                          'target +/-3 frames excluded. No spatial fill, no learned/generative texture.',
        'confidence': 'Median absolute donor deviation <=2 display levels; agreement is not correctness.',
        'summary': summaries, 'per_frame': rows,
        'checks': {'independent_known_rotation_max_px': errors, 'native_footprint_exclusion': True,
                   'missing_evidence_abstention': True, 'fallback': fallback_unit_checks()},
        'real_footage_status': 'No new reflection recovery claimed. Native visibility and material correspondence remain unvalidated.',
        'visual_inspection': 'pending'}
    report['video'] = encode_decode(shot, out, 'synthetic_correspondence', frame_count)
    report['fallback'] = fallback_check(shot, out, fallback_scenes, fallback_results)
    shot.log('r11_atlas_audit.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
