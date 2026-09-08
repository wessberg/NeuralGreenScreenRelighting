"""Reporter-only frozen temporal visibility/material audit; no reflection repair."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from audit_r14_neural import load_rows, SIDES, tile, write
from audit_r15_dense import plain, digest
from dense_iris_flow import sample_native
from material_footprints import region_masks, classify_footprints
from shot_paths import add_shot_argument, from_args
from temporal_eye_tracker import TemporalEyeTracker
from temporal_iris import footprint_valid, apply_eye_repair

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(records):
    out = {}
    for arm in ('raft_cycle','raft_proposal','cotracker','cotracker_proposal'):
        counts = dict(attempts=0,predicted_visible=0,iris_compatible=0,incompatible=0,
                      unknown=0,closure_attempts=0,closure_false_visible=0,classes={})
        for rec in records:
            p = rec[arm]
            accept = np.asarray(p['accepted'],bool)
            classes = np.asarray(p['material'])
            counts['attempts'] += len(accept)
            counts['predicted_visible'] += int(accept.sum())
            counts['iris_compatible'] += int((accept & (classes=='iris')).sum())
            counts['unknown'] += int((accept & (classes=='unknown')).sum())
            counts['incompatible'] += int((accept & ~np.isin(classes,['iris','unknown'])).sum())
            for c in classes[accept]:
                counts['classes'][str(c)] = counts['classes'].get(str(c),0)+1
            if rec['closure'] is True:
                counts['closure_attempts'] += len(accept)
                counts['closure_false_visible'] += int(accept.sum())
        out[arm] = counts
    return out


def evaluate_labels(labels, tracks, baseline, proposals, rows):
    records = []
    for label in labels['frames']:
        i,side = label['frame'],label['side']
        masks = region_masks(label,(64,64),labels['boundary_uncertainty_px'])
        rec = dict(frame=i,side=side,closure=label['closure'],
                   labelled_native_pixels={k:int(v.sum()) for k,v in masks.items()})
        old = baseline[i,side]
        raft_xy = np.asarray(old['mapped'],np.float32)
        cot = tracks[side]
        cot_xy = cot['xy'][i-950]
        cycle = np.array([v is not None and v<=.5 for v in old['roundtrip']])
        proposal = proposals[f'{i:06d}_{side}_material']
        # Saved proposals must reproduce round 15's actual native-footprint test.
        assert np.array_equal(footprint_valid(proposal,raft_xy),old['candidate'])
        for arm,xy,accept in (
            ('raft_cycle',raft_xy,cycle),
            ('raft_proposal',raft_xy,cycle & np.asarray(old['candidate'])),
            ('cotracker',cot_xy,cot['usable_visibility'][i-950]),
            ('cotracker_proposal',cot_xy,cot['usable_visibility'][i-950] & footprint_valid(proposal,cot_xy))):
            rec[arm] = dict(xy=xy,accepted=accept,material=classify_footprints(masks,xy),
                           native_rgb=sample_native(rows[side][i]['rgb'],xy))
        records.append(rec)
    return records


def controlled(tracker,rgb,aspect,direction):
    dx,dy = direction
    # Returns to its initial pose; opaque plate is unrelated observed skin RGB.
    shift = np.array([0,1,2,3,3,2,1,0,-1,-2,-3,-3,-2,-1,0,1,2,3,3,2,1,0,0,0],np.float32)
    offsets = shift[:,None]*np.array([dx,dy],np.float32)
    video = np.stack([cv2.warpAffine(rgb,np.array([[1,0,x],[0,1,y]],np.float32),(64,64),
                        flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_REFLECT_101) for x,y in offsets])
    video[8:16,12:52,12:52] = rgb[10,10]
    points = np.array([[23,37],[27,39],[31,39],[35,37]],np.float32)
    queries = np.column_stack([np.zeros(4),points])
    prediction = tracker.track(video,queries,pixel_aspect=aspect)
    truth = points[None]+offsets[:,None]
    visible = np.ones((24,4),bool)
    visible[8:16] = False
    visible[0] = False  # forced query frame excluded
    error = np.linalg.norm(prediction['xy']-truth,axis=-1)
    out = dict(pixel_aspect=aspect,direction=direction,visible_attempts=int(visible.sum()),
        visible_detected=int((prediction['usable_visibility'] & visible).sum()),
        occluded_attempts=32,occluded_false_visible=int(prediction['usable_visibility'][8:16].sum()),
        recovered_visible=int(prediction['usable_visibility'][16:].sum()),recovery_attempts=32,
        median_visible_endpoint_error=float(np.median(error[visible])),
        p95_visible_endpoint_error=float(np.percentile(error[visible],95)))
    return out,dict(video=video,truth=truth,**prediction)


def render_review(shot,rows,tracks,baseline):
    for k,i in enumerate(range(950,1000)):
        panels=[]
        for side in SIDES:
            rgb=rows[side][i]['rgb']
            old=baseline[i,side]
            overlays=[]
            old_pass=[v is not None and v<=.5 and c for v,c in zip(old['roundtrip'],old['candidate'])]
            for xy,valid in [(old['mapped'],old_pass),(tracks[side]['xy'][k],tracks[side]['usable_visibility'][k])]:
                over=rgb.copy()
                for p,ok in zip(xy,valid):
                    if np.isfinite(p).all():
                        cv2.circle(over,tuple(np.rint(p).astype(int)),1,(0,255,0) if ok else (255,70,30),-1)
                overlays.append(over)
            fallback,info=apply_eye_repair(rgb,np.ones((64,64),bool),geometry_reliable=False)
            assert info['changed_channels']==0 and np.array_equal(rgb,fallback)
            panels.append(np.concatenate([
                tile(rgb,f'{side} SOURCE {i}','original room reflection'),
                tile(overlays[0],'RAFT + IRIS PROPOSAL','green: cycle + candidate; not clean'),
                tile(overlays[1],'COTRACKER VISIBILITY',
                     'QUERY FRAME: forced visible' if i==977 else 'green: model visible; not clean'),
                tile(fallback,'EXACT SOURCE FALLBACK','NO REMOVAL / REPLACEMENT')],1))
        write(shot.composite/'r16_visibility/review'/f'{k:06d}.png',np.concatenate(panels,0))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    add_shot_argument(ap)
    ap.add_argument('--review-only',action='store_true')
    args=ap.parse_args()
    shot=from_args(args)
    if shot.dir != (ROOT/'music_videos/reporter_test/shots/reporter_a').resolve():
        raise SystemExit('Reporter-only audit; use temporal_eye_tracker array API for other shots')
    out=shot.composite/'r16_visibility'
    out.mkdir(parents=True,exist_ok=True)
    rows=load_rows(shot,range(950,1000))
    old=json.loads((shot.logs/'r15_dense.json').read_text())
    baseline={(r['frame'],r['side']):r for r in old['blink']}
    labels=json.loads((ROOT/'scripts/data/r16_material_labels.json').read_text())
    queries=json.loads((ROOT/'scripts/data/r14_native_labels.json').read_text())
    tracks={}
    if args.review_only:
        for side in SIDES:
            with np.load(out/f'{side}_tracks.npz') as data:
                tracks[side]={k:data[k] for k in ('xy','visible','native_footprint','usable_visibility')}
        render_review(shot,rows,tracks,baseline)
        return
    tracker=TemporalEyeTracker()
    invariance=[]
    for side in SIDES:
        video=np.stack([rows[side][i]['rgb'] for i in range(950,1000)])
        points=np.array(next(l['iris'] for l in queries['frames'] if l['frame']==977 and l['side']==side))
        query=np.column_stack([np.full(5,27),points])
        tracks[side]=tracker.track(video,query)
        np.savez_compressed(out/f'{side}_tracks.npz',**tracks[side],queries=query)
        print('unmasked',side,flush=True)
        hidden=np.zeros(video.shape[:3],bool)
        hidden[27,points[:,1],points[:,0]]=True
        reference=tracker.track(video,query,excluded=hidden)
        for color in ([0,0,0],[255,0,255]):
            changed=video.copy()
            changed[hidden]=color
            repeated=tracker.track(changed,query,excluded=hidden)
            assert digest(reference)==digest(repeated),(side,color)
            invariance.append(dict(side=side,color=color,exact=True))
        np.savez_compressed(out/f'{side}_masked_tracks.npz',**reference,excluded=hidden)
        print('hidden RGB verified',side,flush=True)
    with np.load(shot.composite/'r14_neural/proposals.npz') as proposals:
        records=evaluate_labels(labels,tracks,baseline,proposals,rows)
    synthetic=[]
    from anamorphic import SQUEEZE
    for aspect in (1.,SQUEEZE):
        for direction in ([1,0],[0,1]):
            metrics,data=controlled(tracker,rows['left'][977]['rgb'],aspect,direction)
            synthetic.append(metrics)
            np.savez_compressed(out/f'synthetic_{aspect}_{direction[0]}.npz',**data)
            print('controlled',metrics,flush=True)
    report=plain(dict(model=tracker.provenance,
        protocol_sha256=sha(ROOT/'docs/r16-evaluation-protocol.md'),
        annotations_sha256=sha(ROOT/'scripts/data/r16_material_labels.json'),
        reused_queries_sha256=sha(ROOT/'scripts/data/r14_native_labels.json'),
        baseline_sha256=sha(shot.logs/'r15_dense.json'),
        source_crops_sha256={f'{i}_{s}':hashlib.sha256(rows[s][i]['rgb'].tobytes()).hexdigest() for s in SIDES for i in range(950,1000)},
        hidden_rgb_invariance=invariance,records=records,summary=summarize(records),
        synthetic=synthetic,production_repairs=0,scene_reflection_allowed=False,
        label_scope='source-only assistant regions, seen footage; not certified clean radiance or independent human truth'))
    (shot.logs/'r16_visibility.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    render_review(shot,rows,tracks,baseline)
    print(json.dumps(report['summary'],indent=2),flush=True)


if __name__=='__main__':
    main()
