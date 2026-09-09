"""Frozen reporter local-motion diagnostic and RGB-free real context preflight."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from audit_r14_neural import load_rows, SIDES, tile, write
from audit_r15_dense import plain
from audit_r18_motion import definitions, infer, records, summarize, sha, dump
from dense_iris_flow import DenseEyeFlow
from local_reflection_motion import sequence
from material_footprints import region_masks
from registered_context import registered_capacity
from shot_paths import add_shot_argument, from_args
from temporal_eye_tracker import TemporalEyeTracker

ROOT = Path(__file__).resolve().parents[1]
FILES = ('local_reflection_motion.py', 'registered_context.py', 'audit_r19_local.py',
         'audit_r18_motion.py', 'reflection_motion.py', 'temporal_eye_tracker.py',
         'dense_iris_flow.py', 'material_footprints.py', 'iris_correspondence.py',
         'temporal_iris.py')
LABELS = ('r14_native_labels.json', 'r16_material_labels.json', 'r17_context_labels.json')


def preflight():
    labels = [json.loads((ROOT/'scripts/data'/n).read_text()) for n in LABELS]
    masks = {(v['frame'], v['side']): region_masks(v, (64, 64), 1.5).get('iris', np.zeros((64, 64), bool))
             for document in labels[1:] for v in document['frames']}
    cases = []
    for label in labels[0]['frames']:
        frame, side = label['frame'], label['side']
        for category in ('iris', 'reflection'):
            for index, point in enumerate(label[category]):
                pairs = [dict(donor=d, **registered_capacity(masks[frame, side], masks[d, side], point))
                         for d in (974, 990, 994)]
                cases.append(dict(frame=frame, side=side, category=category, index=index,
                    point=point, pairs=pairs, eligible_donors=sum(p['capacity_eligible'] for p in pairs)))
    return plain(dict(cases=cases, attempted_queries=len(cases),
        pairs_with_eight_context_samples=sum(p['capacity_eligible'] for c in cases for p in c['pairs']),
        cases_with_three_capacity_eligible_donors=sum(c['eligible_donors'] >= 3 for c in cases),
        max_context_upper_bound=max(p['context_upper_bound'] for c in cases for p in c['pairs']),
        photometric_scores_computed=0, control_discrimination_validated=False,
        protocol_sha256=sha(ROOT/'docs/r19-evaluation-protocol.md'),
        labels_sha256={n: sha(ROOT/'scripts/data'/n) for n in LABELS},
        scope='Optimistic registered/+3px capacity only; learned arms still require intersection'))


def render_review(shot, report):
    out = shot.composite/'r19_local'
    lookup = {(c['name'], c['condition'], c['anchor']): c for c in report['cases']}
    for seq_index, definition in enumerate(definitions()):
        name = definition['name']
        with np.load(out/f'{name}_sequence.npz') as synth:
            for frame in range(24):
                rows = []
                for anchor in (0, 12):
                    panels = []
                    for condition in ('base', 'contaminated'):
                        case = lookup[name, condition, anchor]
                        with np.load(out/case['cache']) as data:
                            for arm in ('raft', 'cotracker'):
                                rec = next((r for r in case['records'] if r['frame'] == frame and r['arm'] == arm), None)
                                status = 'FORCED QUERY - NOT SCORED' if frame == anchor else 'outside paired score'
                                predicted, accepted = None, False
                                if rec:
                                    predicted, accepted = rec['xy'], rec['accepted']
                                    status = f'{"accept" if accepted else "reject"}; EPE {rec["material_epe"]:.2f}px'
                                elif arm == 'cotracker':
                                    predicted, accepted = data['xy'][frame, 0], data['usable_visibility'][frame, 0]
                                panel = tile(synth[condition][frame], f'{condition} / {arm} / anchor {anchor}', status)
                                def pos(xy):
                                    return tuple(np.rint(np.asarray(xy)*5+2+[0, 48]).astype(int))
                                cv2.drawMarker(panel, pos(synth['material'][anchor]), (0, 200, 255), cv2.MARKER_SQUARE, 13, 1)
                                cv2.drawMarker(panel, pos(synth['material'][frame]), (255, 255, 0), cv2.MARKER_CROSS, 15, 1)
                                if condition == 'contaminated':
                                    cv2.drawMarker(panel, pos(synth['reflection'][frame]), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 13, 1)
                                if predicted is not None and np.isfinite(predicted).all():
                                    cv2.circle(panel, pos(predicted), 6, (0, 255, 0) if accepted else (0, 0, 255), 1)
                                panels.append(panel)
                    rows.append(np.concatenate(panels, 1))
                board = cv2.copyMakeBorder(np.concatenate(rows, 0), 54, 0, 0, 0, cv2.BORDER_CONSTANT)
                cv2.putText(board, f'SYNTHETIC LOCAL PATCH / STATIC SURROUND | {name} t={frame:02d} | NO REPAIR', (8, 20), 0, .53, (255, 255, 255), 1)
                cv2.putText(board, 'Cyan + material | Yellow square stationary | Magenta x reflection | Green/red circle accepted/rejected', (8, 43), 0, .5, (255, 255, 255), 1)
                write(out/'review'/f'{seq_index*24+frame:06d}.png', board)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_shot_argument(ap)
    ap.add_argument('--preflight-only', action='store_true')
    ap.add_argument('--review-only', action='store_true')
    args = ap.parse_args()
    shot = from_args(args)
    if shot.dir != (ROOT/'music_videos/reporter_test/shots/reporter_a').resolve():
        raise SystemExit('Reporter-only audit; local_reflection_motion.sequence accepts arbitrary crops')
    out = shot.composite/'r19_local'
    out.mkdir(parents=True, exist_ok=True)
    report_path = shot.logs/'r19_local.json'
    if args.review_only:
        render_review(shot, json.loads(report_path.read_text()))
        return
    capacity = preflight()
    capacity_path = shot.logs/'r19_preflight.json'
    if args.preflight_only:
        dump(capacity_path, capacity)
        print(json.dumps({k: v for k, v in capacity.items() if k != 'cases'}, indent=2))
        return
    assert json.loads(capacity_path.read_text()) == capacity, 'Run and freeze preflight first'
    prior_path = shot.logs/'r18_motion.json'
    prior = json.loads(prior_path.read_text())
    for name, value in prior['code_sha256'].items():
        assert sha(ROOT/'scripts'/name) == value
    rows = load_rows(shot, [977])
    source_hashes = {s: hashlib.sha256(rows[s][977]['rgb'].tobytes()).hexdigest() for s in SIDES}
    assert source_hashes == prior['source_crops_sha256']
    code = {n: sha(ROOT/'scripts'/n) for n in FILES}
    protocol = sha(ROOT/'docs/r19-evaluation-protocol.md')
    tracker, flow = TemporalEyeTracker(), DenseEyeFlow()
    assert [tracker.provenance, flow.provenance] == prior['models']
    cases = []
    for definition in definitions():
        synth = sequence(rows[definition['side']][977]['rgb'], definition['point'], definition['axis'])
        np.savez_compressed(out/f'{definition["name"]}_sequence.npz', **synth)
        for condition in ('base', 'contaminated'):
            video = synth[condition]
            for anchor in (0, 12):
                name = f'{definition["name"]}_{condition}_{anchor}'
                data = infer(tracker, flow, video, synth['material'], anchor)
                hashes = {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in data.items()}
                substitutions = []
                for color in ([0, 0, 0], [255, 0, 255]):
                    altered = video.copy()
                    altered[data['excluded']] = color
                    repeated = infer(tracker, flow, altered, synth['material'], anchor)
                    assert {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in repeated.items()} == hashes, name
                    substitutions.append(dict(color=color, exact_arrays=len(hashes)))
                np.savez_compressed(out/f'{name}.npz', **data)
                scored = records(data, synth, anchor)
                for r in scored:
                    r['stationary_epe'] = float(np.linalg.norm(np.asarray(r['xy'])-synth['material'][anchor]))
                    r['stationary_truth_epe'] = float(np.linalg.norm(synth['material'][r['frame']]-synth['material'][anchor]))
                    r['stationary_lock'] = bool(r['accepted_wrong'] and r['stationary_truth_epe'] >= 3 and r['stationary_epe'] <= 1)
                cases.append(dict(**definition, condition=condition, anchor=anchor,
                    cache=f'{name}.npz', arrays_sha256=hashes, substitutions=substitutions,
                    clipped_channels=int(synth['clipped'].sum()) if condition == 'contaminated' else 0,
                    records=scored))
                print('Verified', name, flush=True)
    summary = summarize(cases)
    for key, value in summary.items():
        condition, anchor, arm = key.split('_')
        group = [r for c in cases if c['condition'] == condition and c['anchor'] == int(anchor)
                 for r in c['records'] if r['arm'] == arm]
        value.update(stationary_lock=sum(r['stationary_lock'] for r in group),
            stationary_median_epe=float(np.median([r['stationary_truth_epe'] for r in group])),
            stationary_separated_attempts=sum(r['stationary_truth_epe'] >= 3 for r in group))
    assert code == {n: sha(ROOT/'scripts'/n) for n in FILES}
    assert protocol == sha(ROOT/'docs/r19-evaluation-protocol.md')
    report = dict(protocol_sha256=protocol, code_sha256=code, source_crops_sha256=source_hashes,
        labels_sha256=capacity['labels_sha256'], preflight_sha256=sha(capacity_path),
        global_reference_sha256=sha(prior_path), global_reference_summary=prior['summary'],
        models=[tracker.provenance, flow.provenance], cases=cases, summary=summary,
        production_repairs=0, scene_reflection_allowed=False, photometric_scores_computed=0,
        scope='Synthetic local observed-texture motion; artificial aperture; no real clean-material truth')
    dump(report_path, report)
    render_review(shot, report)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
