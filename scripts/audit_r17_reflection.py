"""Reporter-only single-query context preflight and reflection-material audit."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from audit_r14_neural import load_rows, SIDES, tile, write
from audit_r15_dense import plain
from audit_r16_visibility import sha
from context_capacity import fixed_control_capacity
from dense_iris_flow import DenseEyeFlow, query_flow, sample_native
from material_footprints import region_masks, classify_footprints
from shot_paths import add_shot_argument, from_args
from temporal_eye_tracker import TemporalEyeTracker
from temporal_iris import footprint_valid, apply_eye_repair

ROOT = Path(__file__).resolve().parents[1]
DONORS = (974, 978, 981, 982, 984, 985, 990, 994)
CONTEXT_DONORS = (974, 990, 994)
ARMS = ('raft_cycle', 'raft_proposal', 'cotracker', 'cotracker_proposal')


def dump(path, data):
    path.write_text(json.dumps(plain(data), indent=2, allow_nan=False), encoding='utf-8')


def preflight():
    old = json.loads((ROOT/'scripts/data/r14_native_labels.json').read_text())
    added = json.loads((ROOT/'scripts/data/r17_context_labels.json').read_text())
    frozen = json.loads((ROOT/'scripts/data/r16_material_labels.json').read_text())
    lookup = {(v['frame'],v['side']):v for v in added['frames']+frozen['frames']}
    cases = []
    for label in old['frames']:
        frame, side = label['frame'], label['side']
        target = region_masks(lookup[frame,side], (64,64), 1.5)['iris']
        for category in ('iris', 'reflection'):
            for index, point in enumerate(label[category]):
                pairs = []
                for donor in CONTEXT_DONORS:
                    material = region_masks(lookup[donor,side], (64,64), 1.5)['iris']
                    pairs.append(dict(donor=donor, **fixed_control_capacity(target,material,[point])))
                eligible = sum(p['capacity_eligible'] for p in pairs)
                cases.append(dict(frame=frame, side=side, category=category, index=index,
                    point=point, pairs=pairs, eligible_donors=eligible,
                    status='capacity_only_requires_all_arm_implementation' if eligible>=3 else 'insufficient_context'))
    return plain(dict(protocol_sha256=sha(ROOT/'docs/r17-evaluation-protocol.md'),
        r14_labels_sha256=sha(ROOT/'scripts/data/r14_native_labels.json'),
        r16_labels_sha256=sha(ROOT/'scripts/data/r16_material_labels.json'),
        r17_labels_sha256=sha(ROOT/'scripts/data/r17_context_labels.json'),
        cases=cases, attempted_queries=len(cases),
        cases_with_three_capacity_eligible_donors=sum(c['eligible_donors']>=3 for c in cases),
        photometric_scores_computed=0,
        scope='RGB-free fixed/wrong-map upper bound; adding learned-map intersections only reduces capacity'))


def infer_case(tracker, flow, video, frame, point):
    query = np.array([[frame-950, *point]], np.float32)
    hidden = np.zeros(video.shape[:3], bool)
    x, y = np.floor(point).astype(int)
    hidden[frame-950,y:y+2,x:x+2] = True
    tracks = tracker.track(video, query, excluded=hidden)
    fields = {}
    for donor in DONORS:
        forward, backward = flow.pair(video[frame-950],video[donor-950],
                                     target_excluded=hidden[frame-950])
        fields[f'{donor}_forward'] = forward
        fields[f'{donor}_backward'] = backward
    return dict(**tracks, **fields, excluded=hidden, query=query)


def score_case(data, frame, side, point, labels, proposals, rows):
    lookup = {(l['frame'],l['side']):l for l in labels['frames']}
    records = []
    query = np.asarray([point], np.float32)
    for donor in DONORS:
        masks = region_masks(lookup[donor,side], (64,64), 1.5)
        raft, inbound, cycle = query_flow(data[f'{donor}_forward'],data[f'{donor}_backward'],query)
        cot = data['xy'][donor-950]
        proposal = proposals[f'{donor:06d}_{side}_material']
        rec = dict(donor=donor, closure=lookup[donor,side]['closure'])
        for arm, xy, accepted in (
            ('raft_cycle',raft,inbound & (cycle<=.5)),
            ('raft_proposal',raft,inbound & (cycle<=.5) & footprint_valid(proposal,raft)),
            ('cotracker',cot,data['usable_visibility'][donor-950]),
            ('cotracker_proposal',cot,data['usable_visibility'][donor-950] & footprint_valid(proposal,cot))):
            rec[arm] = dict(xy=xy, accepted=accepted, material=classify_footprints(masks,xy),
                           observed_donor_rgb=sample_native(rows[side][donor]['rgb'],xy))
        rec['raft_roundtrip'] = cycle
        records.append(rec)
    return plain(records)


def summarize(cases):
    result = {}
    for arm in ARMS:
        counts = dict(attempts=0, accepted=0, iris_compatible=0, unknown=0,
                      incompatible=0, classes={}, queries_with_three_iris_donors=0,
                      closure_attempts=0, closure_accepted=0)
        for case in cases:
            iris = 0
            for rec in case['records']:
                p = rec[arm]
                accepted, material = p['accepted'][0], p['material'][0]
                counts['attempts'] += 1
                counts['closure_attempts'] += int(rec['closure'] is True)
                if not accepted:
                    continue
                counts['accepted'] += 1
                counts['closure_accepted'] += int(rec['closure'] is True)
                counts['classes'][material] = counts['classes'].get(material,0)+1
                counts['iris_compatible'] += int(material=='iris')
                counts['unknown'] += int(material=='unknown')
                counts['incompatible'] += int(material not in ('iris','unknown'))
                iris += int(material=='iris')
            counts['queries_with_three_iris_donors'] += int(iris>=3)
        result[arm] = counts
    return result


def render_review(shot, rows, cases, out):
    selected = {(c['side'],c['index']):c for c in cases if c['frame']==977}
    tracks = {(s,k):np.load(out/f'977_{s}_{k}.npz') for s in SIDES for k in (0,1)}
    for t, frame in enumerate(range(950,1000)):
        panels = []
        for side in SIDES:
            rgb = rows[side][frame]['rgb']
            raft, cot = rgb.copy(), rgb.copy()
            for index in (0,1):
                data = tracks[side,index]
                if frame in DONORS:
                    rec = next(r for r in selected[side,index]['records'] if r['donor']==frame)
                    p = rec['raft_cycle']
                    xy, ok = np.asarray(p['xy'][0]), p['accepted'][0]
                    cv2.circle(raft,tuple(np.rint(xy).astype(int)),1,(0,255,0) if ok else (255,70,30),-1)
                xy = data['xy'][t,0]
                ok = data['usable_visibility'][t,0]
                cv2.circle(cot,tuple(np.rint(xy).astype(int)),1,(0,255,0) if ok else (255,70,30),-1)
            fallback, info = apply_eye_repair(rgb,np.ones((64,64),bool),geometry_reliable=False)
            assert np.array_equal(rgb,fallback) and info['changed_channels']==0
            panels.append(np.concatenate([
                tile(rgb,f'{side} SOURCE {frame}','original room reflection'),
                tile(raft,'RAFT: MASKED QUERY 977','green: cycle; NOT clean' if frame in DONORS else 'not evaluated at this frame'),
                tile(cot,'COTRACKER: MASKED 977','FORCED QUERY FRAME' if frame==977 else 'green: visible; NOT clean / identity'),
                tile(fallback,'EXACT SOURCE FALLBACK','NO REMOVAL / REPLACEMENT')],1))
        write(out/'review'/f'{t:06d}.png',np.concatenate(panels,0))
    for data in tracks.values():
        data.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_shot_argument(ap)
    ap.add_argument('--preflight-only',action='store_true')
    ap.add_argument('--review-only',action='store_true')
    args = ap.parse_args()
    shot = from_args(args)
    if shot.dir != (ROOT/'music_videos/reporter_test/shots/reporter_a').resolve():
        raise SystemExit('Reporter-only audit; reusable array APIs accept other prepared footage')
    out = shot.composite/'r17_reflection'
    out.mkdir(parents=True,exist_ok=True)
    capacity = preflight()
    capacity_path = shot.logs/'r17_preflight.json'
    if args.preflight_only:
        dump(capacity_path,capacity)
        print(json.dumps({k:v for k,v in capacity.items() if k!='cases'},indent=2))
        print('Upper bounds:',[p['fixed_control_upper_bound'] for c in capacity['cases'] for p in c['pairs']])
        return
    assert json.loads(capacity_path.read_text()) == capacity, 'Run/freeze preflight first'
    rows = load_rows(shot,range(950,1000))
    report_path = shot.logs/'r17_reflection.json'
    if args.review_only:
        render_review(shot,rows,json.loads(report_path.read_text())['cases'],out)
        return
    # No context scores are evaluated here; preflight cannot authorize repair.
    tracker, flow = TemporalEyeTracker(), DenseEyeFlow()
    labels = json.loads((ROOT/'scripts/data/r16_material_labels.json').read_text())
    queries = json.loads((ROOT/'scripts/data/r14_native_labels.json').read_text())
    cases = []
    with np.load(shot.composite/'r14_neural/proposals.npz') as proposals:
        for label in queries['frames']:
            frame, side = label['frame'],label['side']
            video = np.stack([rows[side][i]['rgb'] for i in range(950,1000)])
            for index, point in enumerate(label['reflection']):
                data = infer_case(tracker,flow,video,frame,point)
                reference = {k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in data.items()}
                invariance = []
                for color in ([0,0,0],[255,0,255]):
                    altered = video.copy()
                    altered[data['excluded']] = color
                    repeated = infer_case(tracker,flow,altered,frame,point)
                    repeated_hashes = {k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in repeated.items()}
                    assert repeated_hashes == reference, (frame,side,index,color)
                    invariance.append(dict(color=color,exact=True,arrays_sha256=repeated_hashes))
                np.savez_compressed(out/f'{frame}_{side}_{index}.npz',**data)
                cases.append(dict(frame=frame,side=side,index=index,point=point,
                    clean_truth_known=False,pupil_crossing='cannot_exclude',
                    hidden_rgb_invariance=invariance,
                    records=score_case(data,frame,side,point,labels,proposals,rows)))
                print('Verified reflection query',frame,side,index,flush=True)
    report = dict(preflight_sha256=sha(capacity_path),models=[tracker.provenance,flow.provenance],
        cases=cases,summary=summarize(cases),
        source_crops_sha256={f'{i}_{s}':hashlib.sha256(rows[s][i]['rgb'].tobytes()).hexdigest() for s in SIDES for i in range(950,1000)},
        proposals_sha256=sha(shot.composite/'r14_neural/proposals.npz'),
        production_repairs=0,scene_reflection_allowed=False,photometric_scores_computed=0,
        scope='Reused sparse reflection anchors; coordinates need not follow underlying iris; unknown clean truth')
    dump(report_path,report)
    render_review(shot,rows,cases,out)
    print(json.dumps(report['summary'],indent=2),flush=True)


if __name__=='__main__':
    main()
