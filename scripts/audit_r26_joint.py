"""Joint same-eye optical families and withheld source-RGB transport diagnostic."""
import argparse
import json
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2
import numpy as np
from audit_r24_source import ROOT, sha, write, native_hash
from audit_r25_source import camera
from color_pipeline import linear_to_srgb
from coupled_eye_relight import apply_coupled
from joint_eye_geometry import (SHARED, RADII, build_table, joint_profiles, candidate,
    iris_hit, inverse_iris, box_iris, donor_support, sample_rgb)
from partial_eye_geometry import split_arc, fit_family, curve, distances, conservative_open
from shot_paths import add_shot_argument, from_args

DONORS = (30, 60, 90, 120)
TARGET = 105


def evidence(shot):
    paths = [ROOT/'docs/r26-eye-protocol.md', ROOT/'scripts/data/r26_source_boundaries.json',
             ROOT/'scripts/data/r25_source_boundaries.json', shot.logs/'r25_source.json',
             shot.logs/'r26_preflight.json', shot.dir/'shot.json',
             shot.relight/'neural_iris_observations/manifest.json']
    paths += [shot.relight/f'neural_iris_observations/{frame:06d}_{side}.npz'
              for frame in (*DONORS, TARGET) for side in ('left', 'right')]
    record = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    preflight = json.loads((shot.logs/'r26_preflight.json').read_text())
    assert preflight['protocol_sha256'] == sha(paths[0])
    for row in preflight['records']:
        p = shot.relight/f'neural_iris_observations/000105_{row["side"]}.npz'
        assert sha(p) == row['archive_sha256']
    path = shot.logs/'r26_freeze.json'
    if path.exists(): assert json.loads(path.read_text()) == record
    else: write(path, record)
    return record


def observations(shot, side, focal, centre, prior, labels):
    old_labels = json.loads((ROOT/'scripts/data/r25_source_boundaries.json').read_text())['records']
    result = []
    for frame in (*DONORS, TARGET):
        label = next(r for r in (labels if frame == TARGET else old_labels)
                     if r['frame'] == frame and r['side'] == side)
        with np.load(shot.relight/f'neural_iris_observations/{frame:06d}_{side}.npz') as d:
            # RGB shape only. Target radiance is not passed to fitting or mapping.
            shape, origin = d['linear_rgb'].shape[:2], d['origin_xy']
        train, check, _, _ = split_arc(label['limbus'])
        family = (fit_family(train, focal, centre-origin) if frame == TARGET else
                  next(r['family'] for r in prior['records'] if r['frame'] == frame and r['side'] == side))
        result.append(dict(frame=frame, side=side, target=frame == TARGET, shape=shape,
            centre=(centre-origin).tolist(), origin=origin.tolist(), family=family,
            pupil=label['pupil'], label=label, limbus_check=check.tolist()))
    return result


def get_eye(observations, table, row_index, focal):
    row = table[row_index]
    si, oi, ri, pi = row[:4].astype(int)
    obs = observations[oi]
    return candidate(obs['family'][ri], SHARED[si], RADII[pi], focal, obs['centre'])


def mapping(observations, table, profile, focal, uncertainty=2.):
    target = observations[-1]
    _, safe = conservative_open(target['shape'], target['label']['lid'],
                                target['label']['reflection'], uncertainty)
    y, x = np.nonzero(safe)
    xy = np.stack([x, y], -1)
    teye, toffset = get_eye(observations, table, profile['best_rows'][-1], focal)
    valid_target = box_iris(xy, teye, toffset, focal, target['centre'])
    uv, _ = iris_hit(xy, teye, toffset, focal, target['centre'])
    maps, controls, accepted, control_accepted = [], [], [], []
    front = next(r for r in target['family'] if r['yaw_deg'] == r['pitch_deg'] == 0)
    for di, donor in enumerate(observations[:-1]):
        deye, doffset = get_eye(observations, table, profile['best_rows'][di], focal)
        coords, converged, _ = inverse_iris(uv, deye, doffset, focal, donor['centre'])
        _, dsafe = conservative_open(donor['shape'], donor['label']['lid'],
                                     donor['label']['reflection'], uncertainty)
        support = valid_target & converged & donor_support(coords, dsafe, deye, doffset, focal, donor['centre'])
        dfront = next(r for r in donor['family'] if r['yaw_deg'] == r['pitch_deg'] == 0)
        tparams, dparams = np.array(front['parameters']), np.array(dfront['parameters'])
        control = (xy-tparams[:2])*tparams[2]/dparams[2]+dparams[:2]
        csupport = valid_target & donor_support(control, dsafe, deye, doffset, focal, donor['centre'])
        maps.append(coords); controls.append(control); accepted.append(support); control_accepted.append(csupport)
    return dict(xy=xy, target_iris=valid_target, mapped=np.array(maps), control=np.array(controls),
                supported=np.array(accepted), control_supported=np.array(control_accepted))


def witness_spread(observations, table, profiles, focal, xy):
    """Best assignment per shared tuple plus torsion witnesses; never certification."""
    all_maps, all_valid = [], []
    target = observations[-1]
    for pi, profile in enumerate(profiles):
        teye, toffset = get_eye(observations, table, profile['best_rows'][-1], focal)
        uv, thit = iris_hit(xy, teye, toffset, focal, target['centre'])
        for degrees in ((-10, 0, 10) if pi == 0 else (0,)):
            a = np.deg2rad(degrees)
            rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            variant = uv@rot.T
            maps, hits = [], []
            for di, donor in enumerate(observations[:-1]):
                eye, offset = get_eye(observations, table, profile['best_rows'][di], focal)
                m, hit, _ = inverse_iris(variant, eye, offset, focal, donor['centre'])
                maps.append(m); hits.append(hit & thit)
            all_maps.append(maps); all_valid.append(hits)
    maps, valid = np.array(all_maps), np.array(all_valid)
    lo = np.min(np.where(valid[..., None], maps, np.inf), axis=0)
    hi = np.max(np.where(valid[..., None], maps, -np.inf), axis=0)
    counts = valid.sum(0)
    spread = np.where(counts >= 2, np.max(hi-lo, axis=-1), 0.)
    return dict(witness_maps=maps, witness_valid=valid, witness_count=counts,
                witness_diameter_lower_bound_px=spread)


def predict(shot, side, maps):
    """Target RGB is absent. Only supported, same-eye donor pixels are read."""
    predictions = np.zeros((*maps['supported'].shape, 3), float)
    control = predictions.copy()
    paired = maps['supported'] & maps['control_supported']
    for di, frame in enumerate(DONORS):
        with np.load(shot.relight/f'neural_iris_observations/{frame:06d}_{side}.npz') as d:
            rgb = d['linear_rgb']
        mask = paired[di]
        predictions[di, mask] = sample_rgb(rgb, maps['mapped'][di, mask])
        control[di, mask] = sample_rgb(rgb, maps['control'][di, mask])
    return predictions, control, paired


def summarize_errors(pred, control, target, mask):
    if not mask.any(): return dict(samples=0, joint_median=None, control_median=None)
    disp = lambda a: linear_to_srgb(np.maximum(a, 0))*255
    a = np.abs(disp(pred[mask])-disp(target[mask])).mean(-1)
    b = np.abs(disp(control[mask])-disp(target[mask])).mean(-1)
    return dict(samples=int(mask.sum()), joint_median=float(np.median(a)),
        control_median=float(np.median(b)), joint_mean=float(a.mean()), control_mean=float(b.mean()),
        joint_p95=float(np.percentile(a, 95)), control_p95=float(np.percentile(b, 95)))


def score_frozen(shot, side, path):
    # Prediction artifact must exist on disk before target radiance is opened here.
    with np.load(path) as d:
        xy, pred, control, paired = (d[k] for k in ('xy', 'predictions', 'control_predictions', 'paired'))
    with np.load(shot.relight/f'neural_iris_observations/000105_{side}.npz') as d:
        target = d['linear_rgb'][xy[:, 1], xy[:, 0]]
    donors = [summarize_errors(pred[i], control[i], target, paired[i]) for i in range(4)]
    count = paired.sum(0); mask = count >= 3
    mean = pred.sum(0)/np.maximum(count[:, None], 1)
    cmean = control.sum(0)/np.maximum(count[:, None], 1)
    return dict(donors=donors, equal_weight_mean=summarize_errors(mean, cmean, target, mask),
                target_eligible_pixels=len(xy), mean_abstentions=int((~mask).sum()),
                prediction_sha256=sha(path), clean_material_certified=False)


def run_eye(shot, side, obs, focal, out):
    table = build_table(obs, focal)
    profiles = joint_profiles(table, len(obs))
    np.savez_compressed(out/f'family_{side}.npz', table=table, shared=SHARED, pupil_radii=RADII)
    write(out/f'geometry_{side}.json', dict(observations=obs, profiles=profiles))
    sensitivity = {str(t): len(joint_profiles(table, len(obs), t)) for t in (1., 2., 3.)}
    centred = joint_profiles(table, len(obs), allowed=lambda s: np.all(s[2:] == 0))
    baseline_depth = joint_profiles(table, len(obs), allowed=lambda s: s[1] == 3.6)
    report = dict(side=side, tested_rows=len(table), shared_profiles=len(profiles),
        shared_pupil_tolerance_counts=sensitivity, centred_shared_profiles=len(centred),
        baseline_depth_shared_profiles=len(baseline_depth),
        factored_assignments=[dict(shared_index=p['shared_index'], counts=list(map(len, p['choices'])),
            sequence_count=str(np.prod([len(c) for c in p['choices']], dtype=object))) for p in profiles],
        geometry_sha256=sha(out/f'geometry_{side}.json'), family_sha256=sha(out/f'family_{side}.npz'),
        geometry_validated=False, correspondence_reliable=False, production_repairs=0)
    if not profiles: return report
    best = profiles[0]
    report['best_shared'] = SHARED[best['shared_index']].tolist()
    report['best_conditioning_score'] = best['score']
    report['best_rows'] = table[best['best_rows']].tolist()
    tr = table[best['best_rows'][-1]]
    ring = obs[-1]['family'][int(tr[2])]
    contour = curve(ring['parameters'], *np.deg2rad([ring['yaw_deg'], ring['pitch_deg']]), focal, obs[-1]['centre'])
    report['held_limbus_distances_px'] = distances(obs[-1]['limbus_check'], contour).tolist()
    report['held_pupil_distances_px'] = tr[6:8].tolist()
    front = next(r for r in obs[-1]['family'] if r['yaw_deg'] == r['pitch_deg'] == 0)
    report['frontal_held_limbus_distances_px'] = distances(obs[-1]['limbus_check'],
        curve(front['parameters'], 0, 0, focal, obs[-1]['centre'])).tolist()
    maps = mapping(obs, table, best, focal)
    pred, control, paired = predict(shot, side, maps)
    witnesses = witness_spread(obs, table, profiles, focal, maps['xy'])
    path = out/f'prediction_{side}.npz'
    np.savez_compressed(path, **maps, **witnesses, predictions=pred, control_predictions=control, paired=paired)
    report['rgb_check'] = score_frozen(shot, side, path)
    report['support_per_donor'] = maps['supported'].sum(1).tolist()
    report['paired_per_donor'] = paired.sum(1).tolist()
    spread = witnesses['witness_diameter_lower_bound_px']
    tested = paired & (witnesses['witness_count'] >= 2)
    report['witnesses'] = dict(hypotheses=len(profiles)+2, comparable_pairs=int(tested.sum()),
        median_diameter_lower_bound_px=float(np.median(spread[tested])) if tested.any() else None,
        max_diameter_lower_bound_px=float(spread[tested].max()) if tested.any() else None,
        pairs_exceeding_2px=int((tested & (spread > 2)).sum()), certification=False)
    report['lid_sensitivity'] = {}
    for tolerance in (1., 2., 3.):
        m = mapping(obs, table, best, focal, tolerance)
        report['lid_sensitivity'][str(tolerance)] = dict(eligible=len(m['xy']),
            supported=m['supported'].sum(1).tolist(),
            paired=(m['supported'] & m['control_supported']).sum(1).tolist())
    return report


def draw_review(shot, manifest, results, out):
    review = out/'review'; review.mkdir(parents=True, exist_ok=True)
    for frame_row in manifest['frames']:
        frame = frame_row['frame']; panels = []
        for side in ('left', 'right'):
            with np.load(shot.relight/f'neural_iris_observations/{frame:06d}_{side}.npz') as d:
                rgb = d['linear_rgb']
            raw = np.clip(linear_to_srgb(rgb)[..., ::-1]*255, 0, 255).astype('uint8')
            source = cv2.resize(raw, (384, 384), interpolation=cv2.INTER_NEAREST)
            overlay = source.copy(); rec = results[side]
            scale = np.array([384/raw.shape[1], 384/raw.shape[0]])
            toxy = lambda a: np.rint((np.asarray(a)+.5)*scale-.5).astype('int32')
            status = 'NO FIT HERE - original eye and catchlight retained'
            if frame in (*DONORS, TARGET):
                geometry = json.loads((out/f'geometry_{side}.json').read_text())
                obs = next(o for o in geometry['observations'] if o['frame'] == frame)
                cv2.polylines(overlay, [toxy(obs['label']['lid'])], True, (255, 150, 0), 1)
                cv2.polylines(overlay, [toxy(obs['label']['reflection'])], True, (0, 0, 255), 1)
                for p in toxy(obs['label']['limbus']): cv2.circle(overlay, tuple(p), 2, (0, 200, 255), -1)
                status = f'{rec["shared_profiles"]} shared optical tuples - unvalidated'
                if frame == TARGET and rec['shared_profiles']:
                    with np.load(out/f'prediction_{side}.npz') as d:
                        mask = d['paired'].sum(0) >= 3
                        for p in toxy(d['xy'][mask]): cv2.circle(overlay, tuple(p), 1, (0, 255, 0), -1)
                    status = f'WITHHELD RGB: {int(mask.sum())} paired 3+ donor pixels; NOT clean'
            header = np.zeros((96, 768, 3), 'uint8')
            for y, text in [(20, f'{side.upper()} source {frame_row["source_frame"]} | SOURCE / JOINT DIAGNOSTIC'),
                (43, status), (65, 'Observed source RGB transport only; no removal or replacement'),
                (86, 'Green = check locations; original reflections remain in both panels')]:
                cv2.putText(header, text, (8, y), 0, .43, (230, 230, 230), 1)
            panels.append(np.concatenate([header, np.concatenate([source, overlay], 1)], 0))
        board = np.concatenate(panels, 0)
        assert cv2.imwrite(str(review/f'{frame:06d}.png'), board)
    detail = shot.composite/'r26_detail/review'; detail.mkdir(parents=True, exist_ok=True)
    for k, frame in enumerate((30, 60, 90, 105, 120)):
        board = cv2.imread(str(review/f'{frame:06d}.png'))
        header = np.zeros((32, 768, 3), 'uint8')
        cv2.putText(header, 'HELD SOURCE FRAME - 1 second each; not real motion', (8, 22), 0, .5, (255, 255, 255), 1)
        for j in range(24):
            assert cv2.imwrite(str(detail/f'{k*24+j:06d}.png'), np.concatenate([header, board], 0))


def main():
    ap = argparse.ArgumentParser(description=__doc__); add_shot_argument(ap)
    ap.add_argument('--freeze-only', action='store_true'); ap.add_argument('--review-only', action='store_true')
    args = ap.parse_args(); shot = from_args(args)
    frozen = evidence(shot)
    if args.freeze_only: print('R26_FROZEN'); return
    manifest = json.loads((shot.relight/'neural_iris_observations/manifest.json').read_text())
    focal, centre = camera(shot, manifest)
    out = shot.composite/'r26_joint'; out.mkdir(parents=True, exist_ok=True)
    if args.review_only: report = json.loads((shot.logs/'r26_joint.json').read_text())
    else:
        prior = json.loads((shot.logs/'r25_source.json').read_text())
        labels = json.loads((ROOT/'scripts/data/r26_source_boundaries.json').read_text())['records']
        results = {}
        for side in ('left', 'right'):
            obs = observations(shot, side, focal, centre, prior, labels)
            results[side] = run_eye(shot, side, obs, focal, out)
            print('R26_EYE', side, json.dumps(results[side].get('rgb_check')), flush=True)
        report = dict(evidence=frozen, results=results, focal=focal.tolist(), centre=centre.tolist(),
            clean_material_certified=False, production_repairs=0, neural_contours_loaded=False,
            code_sha256={n: sha(ROOT/'scripts'/n) for n in ('audit_r26_joint.py', 'joint_eye_geometry.py',
                'partial_eye_geometry.py', 'connected_eye.py', 'corneal_transport.py', 'coupled_eye_relight.py')})
        write(shot.logs/'r26_joint.json', report)
    draw_review(shot, manifest, report['results'], out)
    print('R26_JOINT_COMPLETE', flush=True)


if __name__ == '__main__': main()
