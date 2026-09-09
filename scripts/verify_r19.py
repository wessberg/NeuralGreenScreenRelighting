"""Independently audit round-19 synthesis, sampling, counts and exact fallback."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2
import numpy as np

from audit_r14_neural import load_rows, SIDES, write
from audit_r18_motion import definitions, EVAL, sha, dump
from local_reflection_motion import sequence
from shot_paths import add_shot_argument, from_args
from verify_r14 import search_isolation
from verify_r15 import fallback
from audit_r19_local import preflight
from material_footprints import region_masks
from registered_context import registered_capacity
from verify_r17 import contributors, fits

ROOT = Path(__file__).resolve().parents[1]


def sample(image, xy):
    x, y = map(float, xy)
    ix, iy = math.floor(x), math.floor(y)
    if not (0 <= ix < image.shape[1]-1 and 0 <= iy < image.shape[0]-1):
        return np.zeros(image.shape[2:], np.float64), False
    value = np.zeros(image.shape[2:], np.float64)
    for dx in (0, 1):
        for dy in (0, 1):
            value += image[iy+dy, ix+dx]*((x-ix) if dx else (1-x+ix))*((y-iy) if dy else (1-y+iy))
    return value, True


def check_synthesis(rgb, q, axis, data):
    yy, xx = np.mgrid[:64, :64]
    radius = np.hypot(xx-q[0], yy-q[1])
    weight = np.where(radius <= 8, 1., np.where(radius >= 12, 0.,
                      (1+np.cos(np.pi*(radius-8)/4))/2))
    assert np.max(np.abs(weight-data['local_weight'])) < 1e-7
    def decode_reference(a):
        v = np.asarray(a, np.float64)/255
        return np.where(v <= .04045, v/12.92, ((v+.055)/1.055)**2.4)
    for t in range(24):
        dx, dy = (t//4)*np.asarray(axis)
        translated = cv2.warpAffine(rgb, np.array([[1, 0, dx], [0, 1, dy]], np.float32),
            (64, 64), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
        mixed = decode_reference(translated)*weight[..., None]+decode_reference(rgb)*(1-weight[..., None])
        expected = np.where(mixed <= .0031308, mixed*12.92, 1.055*mixed**(1/2.4)-.055)*255
        assert np.max(np.abs(data['base'][t]-expected)) < .0001
        assert np.array_equal(data['base'][t][radius <= 8], translated[radius <= 8])
        assert np.array_equal(data['base'][t][radius >= 12], rgb[radius >= 12])
        expected = data['base'][t]
        material = np.asarray(q)+[dx, dy]
        reflection = np.asarray(q)+(15-t)*np.asarray(axis)
        assert np.array_equal(data['material'][t], material)
        assert np.array_equal(data['reflection'][t], reflection)
        wx = np.maximum(0, 1+np.cos(np.pi*(xx-reflection[0])/3))/2
        wy = np.maximum(0, 1+np.cos(np.pi*(yy-reflection[1])/3))/2
        expected_add = .6*wx*wy*((np.abs(xx-reflection[0]) < 3) & (np.abs(yy-reflection[1]) < 3))
        assert np.max(np.abs(expected_add-data['addition'][t])) < 1e-7
        assert np.array_equal(data['contaminated'][t][expected_add == 0], expected[expected_add == 0])
        # Independent scalar colour computation only where the addition is nonzero.
        for y, x in zip(*np.where(expected_add > 0)):
            for c in range(3):
                v = float(expected[y, x, c])/255
                linear = v/12.92 if v <= .04045 else ((v+.055)/1.055)**2.4
                summed = linear+expected_add[y, x]
                encoded = (12.92*summed if summed <= .0031308 else 1.055*summed**(1/2.4)-.055)*255
                assert abs(data['contaminated'][t, y, x, c]-min(255, encoded)) < .0001
                assert data['clipped'][t, y, x, c] == (summed > 1)
        x, y = material.astype(int)
        # Complete query footprint tracks exactly the original four contributors.
        qx, qy = map(int, q)
        assert np.array_equal(data['base'][t, y:y+2, x:x+2], rgb[qy:qy+2, qx:qx+2])
    x, y = data['material'][0].astype(int)
    assert not data['addition'][0, y:y+2, x:x+2].any()
    x, y = data['material'][12].astype(int)
    assert (data['addition'][12, y:y+2, x:x+2] > 0).all()


def check_capacity(capacity):
    assert capacity == preflight()
    labels = [json.loads((ROOT/'scripts/data'/n).read_text())
              for n in ('r16_material_labels.json', 'r17_context_labels.json')]
    masks = {(r['frame'], r['side']): region_masks(r, (64,64), 1.5).get('iris', np.zeros((64,64), bool))
             for document in labels for r in document['frames']}
    counts = []
    for case in capacity['cases']:
        target = masks[case['frame'], case['side']]
        q = np.asarray(case['point'])
        hidden = contributors(q)
        for pair in case['pairs']:
            donor = masks[pair['donor'], case['side']]
            ty, tx = np.where(target)
            dy, dx = np.where(donor)
            shift = np.array([round(sum(dx)/len(dx)-sum(tx)/len(tx)),
                              round(sum(dy)/len(dy)-sum(ty)/len(ty))])
            assert np.array_equal(shift, pair['shift'])
            union = contributors(q+shift) | contributors(q+shift+[3,0])
            assert np.array_equal(pair['query_union'], [q+shift, q+shift+[3,0]])
            expected = []
            for y in range(63):
                for x in range(63):
                    native = contributors([x,y])
                    correct = contributors(np.array([x,y])+shift)
                    wrong = contributors(np.array([x+3,y])+shift)
                    if (fits(target, native) and not native & hidden and
                            fits(donor, correct) and fits(donor, wrong) and not (correct | wrong) & union):
                        expected.append([x,y])
            assert expected == pair['context']
            assert len(expected) == pair['context_upper_bound']
            assert pair['capacity_eligible'] == (len(expected) >= 8)
            counts.append(len(expected))
        assert case['eligible_donors'] == sum(p['capacity_eligible'] for p in case['pairs'])
    for shift in ([3,0], [-3,0], [0,3], [0,-3]):
        target = np.zeros((32,32), bool)
        target[8:24,8:24] = True
        donor = np.roll(target, (shift[1], shift[0]), axis=(0,1))
        got = registered_capacity(target, donor, [15.5,15.5])
        assert np.array_equal(got['shift'], shift)
    assert capacity['pairs_with_eight_context_samples'] == sum(n >= 8 for n in counts)
    assert capacity['max_context_upper_bound'] == max(counts)
    assert capacity['cases_with_three_capacity_eligible_donors'] == sum(c['eligible_donors'] >= 3 for c in capacity['cases'])
    return dict(pairs=len(counts), cases=len(capacity['cases']), signed_centroid_controls=4,
                independent_contributors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_shot_argument(ap)
    args = ap.parse_args()
    shot = from_args(args)
    out = shot.composite/'r19_local'
    report = json.loads((shot.logs/'r19_local.json').read_text())
    assert report['protocol_sha256'] == sha(ROOT/'docs/r19-evaluation-protocol.md')
    for name, value in report['labels_sha256'].items():
        assert value == sha(ROOT/'scripts/data'/name)
    assert report['preflight_sha256'] == sha(shot.logs/'r19_preflight.json')
    capacity_checks = check_capacity(json.loads((shot.logs/'r19_preflight.json').read_text()))
    assert report['global_reference_sha256'] == sha(shot.logs/'r18_motion.json')
    prior = json.loads((shot.logs/'r18_motion.json').read_text())
    assert report['global_reference_summary'] == prior['summary']
    for name, value in prior['code_sha256'].items():
        assert sha(ROOT/'scripts'/name) == value
    for case in prior['cases']:
        with np.load(shot.composite/'r18_motion'/case['cache']) as data:
            assert {k: hashlib.sha256(data[k].tobytes()).hexdigest() for k in data.files} == case['arrays_sha256']
    for name, value in report['code_sha256'].items():
        assert sha(ROOT/'scripts'/name) == value
    for prior, field in (('r17_verification.json', 'code_sha256'),
                         ('r16_verification.json', 'code_sha256'),
                         ('r15_verification.json', 'source_hashes')):
        for name, value in json.loads((shot.logs/prior).read_text())[field].items():
            assert sha(ROOT/'scripts'/name) == value
    assert report['models'] == json.loads((shot.logs/'r17_reflection.json').read_text())['models']
    rows = load_rows(shot, [977])
    for side in SIDES:
        assert hashlib.sha256(rows[side][977]['rgb'].tobytes()).hexdigest() == report['source_crops_sha256'][side]
    # Signed directions are verified even though inference freezes +x and +y only.
    for axis in ([1, 0], [-1, 0], [0, 1], [0, -1]):
        rgb = rows['left'][977]['rgb']
        check_synthesis(rgb, [23, 37], axis, sequence(rgb, [23, 37], axis))
    for definition in definitions():
        with np.load(out/f'{definition["name"]}_sequence.npz') as data:
            check_synthesis(rows[definition['side']][977]['rgb'], definition['point'], definition['axis'], data)
    groups = {}
    checked = comparisons = 0
    max_map_error = max_cycle_error = 0.
    assert len(report['cases']) == 32
    assert len({(c['name'], c['condition'], c['anchor']) for c in report['cases']}) == 32
    for case in report['cases']:
        with np.load(out/case['cache']) as data, np.load(out/f'{case["name"]}_sequence.npz') as synth:
            assert {k: hashlib.sha256(data[k].tobytes()).hexdigest() for k in data.files} == case['arrays_sha256']
            assert case['substitutions'] == [dict(color=color, exact_arrays=len(data.files)) for color in ([0, 0, 0], [255, 0, 255])]
            comparisons += 2*len(data.files)
            anchor = case['anchor']
            point = synth['material'][anchor]
            x, y = point.astype(int)
            expected = np.zeros((24, 64, 64), bool)
            expected[anchor, y:y+2, x:x+2] = True
            assert np.array_equal(expected, data['excluded'])
            assert np.array_equal(data['query'], [[anchor, *point]])
            assert np.max(np.abs(data['xy'][anchor, 0]-point)) < 1e-5
            assert data['visible'][anchor, 0]
            for t in range(24):
                _, inbound = sample(synth[case['condition']][t], data['xy'][t, 0])
                assert data['native_footprint'][t, 0] == inbound
                assert data['usable_visibility'][t, 0] == (inbound and data['visible'][t, 0])
            assert sorted((r['frame'], r['arm']) for r in case['records']) == sorted(
                (t, arm) for t in EVAL if t != anchor for arm in ('raft', 'cotracker'))
            for rec in case['records']:
                t, arm = rec['frame'], rec['arm']
                xy = np.asarray(rec['xy'])
                if arm == 'raft':
                    forward, inbound = sample(data[f'{t}_forward'], point)
                    assert inbound
                    mapped = point+forward
                    back, inbound = sample(data[f'{t}_backward'], mapped)
                    residual = np.linalg.norm(mapped+back-point) if inbound else float('inf')
                    max_map_error = max(max_map_error, float(np.max(np.abs(mapped-xy))))
                    if np.isfinite(residual):
                        max_cycle_error = max(max_cycle_error, abs(residual-rec['cycle']))
                    assert rec['accepted'] == bool(inbound and residual <= .5)
                else:
                    assert np.array_equal(xy, data['xy'][t, 0])
                    assert rec['accepted'] == bool(data['usable_visibility'][t, 0])
                epe = math.dist(xy, synth['material'][t])
                refl_epe = math.dist(xy, synth['reflection'][t])
                separated = math.dist(synth['material'][t], synth['reflection'][t]) >= 3
                assert abs(epe-rec['material_epe']) < 2e-5
                assert abs(refl_epe-rec['reflection_epe']) < 2e-5
                wrong = rec['accepted'] and epe > 1
                lock = wrong and separated and refl_epe <= 1
                assert rec['accepted_wrong'] == wrong and rec['reflection_lock'] == lock
                stationary = math.dist(xy, synth['material'][anchor])
                stationary_truth = math.dist(synth['material'][t], synth['material'][anchor])
                assert abs(stationary-rec['stationary_epe']) < 2e-5
                assert stationary_truth == rec['stationary_truth_epe']
                assert rec['stationary_lock'] == bool(wrong and stationary_truth >= 3 and stationary <= 1)
                groups.setdefault(f'{case["condition"]}_{anchor}_{arm}', []).append(rec)
                checked += 1
    assert max_map_error < 1e-5 and max_cycle_error < 1e-5
    for key, records in groups.items():
        got = report['summary'][key]
        accepted = [r for r in records if r['accepted']]
        assert got['attempts'] == 48 and len(records) == 48
        assert got['accepted'] == len(accepted) and got['rejected'] == 48-len(accepted)
        for field in ('accepted_wrong', 'reflection_lock'):
            assert got[field] == sum(r[field] for r in records)
        assert got['separated_attempts'] == sum(r['separation'] >= 3 for r in records)
        assert got['stationary_lock'] == sum(r['stationary_lock'] for r in records)
        assert got['stationary_separated_attempts'] == sum(r['stationary_truth_epe'] >= 3 for r in records)
        assert got['stationary_median_epe'] == np.median([r['stationary_truth_epe'] for r in records])
        for field, values, pct in (
                ('median_material_epe', records, 50), ('p95_material_epe', records, 95),
                ('accepted_median_material_epe', accepted, 50)):
            expected = float(np.percentile([r['material_epe'] for r in values], pct)) if values else None
            assert got[field] == expected
    assert report['production_repairs'] == 0 and not report['scene_reflection_allowed']
    assert report['photometric_scores_computed'] == 0
    video = shot.final/'r19_local.mp4'
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    thumbs = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        index = len(thumbs)
        assert im.shape == (790, 1280, 3)
        if index % 24 in (0, 12, 23):
            write(out/f'decoded_{index:03d}.png', im)
        thumb = cv2.resize(im, (384, 237))
        cv2.putText(thumb, str(index), (4, 14), 0, .45, (255, 255, 255), 1)
        thumbs.append(thumb)
    cap.release()
    assert len(thumbs) == 192 and abs(fps-24) < .001
    for start in range(0, 192, 24):
        sheet = np.concatenate([np.concatenate(thumbs[i:i+4], 1) for i in range(start, start+24, 4)], 0)
        write(out/f'contact_{start:03d}.jpg', sheet)
    checks = dict(synthetic_sequences=8, signed_transform_controls=4, paired_records=checked,
        exact_substitution_runs=64, array_hash_comparisons=comparisons,
        max_independent_mapping_error=max_map_error, max_independent_cycle_error=max_cycle_error,
        fallback=fallback(shot), search_isolation=search_isolation(), capacity=capacity_checks,
        prior_code_and_models_unchanged=True,
        video=dict(frames=len(thumbs), fps=fps, sha256=sha(video), bytes=video.stat().st_size,
                   visual_review='pending separate record', artist_approval=False),
        code_sha256={n: sha(ROOT/'scripts'/n) for n in ('local_reflection_motion.py', 'registered_context.py', 'audit_r19_local.py', 'verify_r19.py')})
    dump(shot.logs/'r19_verification.json', checks)
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
