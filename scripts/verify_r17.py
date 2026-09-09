"""Independently verify reflection donor footprints, capacity bounds and fallback."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2
import numpy as np

from audit_r14_neural import load_rows, SIDES, write
from audit_r15_dense import plain
from audit_r16_visibility import sha
from audit_r17_reflection import preflight, score_case, summarize, ARMS
from context_capacity import fixed_control_capacity
from dense_iris_flow import sample_native
from material_footprints import region_masks
from shot_paths import add_shot_argument, from_args
from verify_r14 import search_isolation
from verify_r15 import fallback
from verify_r16 import independent_sample

ROOT = Path(__file__).resolve().parents[1]


def contributors(xy):
    x, y = np.floor(xy).astype(int)
    return {(x,y),(x+1,y),(x,y+1),(x+1,y+1)}


def fits(mask, footprint):
    return all(0<=x<mask.shape[1] and 0<=y<mask.shape[0] and mask[y,x] for x,y in footprint)


def independent_context(target, donor, point):
    hidden = contributors(point)
    union = hidden | contributors(np.asarray(point)+[3,0])
    result = []
    interior, remaining = 0, 0
    for y in range(target.shape[0]-1):
        for x in range(target.shape[1]-1):
            native = contributors([x,y])
            wrong = contributors([x+3,y])
            if not fits(target,native):
                continue
            interior += 1
            if native & hidden:
                continue
            remaining += 1
            if fits(donor,native) and fits(donor,wrong) and not ((native|wrong)&union):
                result.append([x,y])
    return interior, remaining, result


def verify_capacity(capacity):
    assert preflight() == capacity
    new = json.loads((ROOT/'scripts/data/r17_context_labels.json').read_text())
    old = json.loads((ROOT/'scripts/data/r16_material_labels.json').read_text())
    lookup = {(r['frame'],r['side']):region_masks(r,(64,64),1.5) for r in new['frames']+old['frames']}
    counts = []
    for case in capacity['cases']:
        target = lookup[case['frame'],case['side']]['iris']
        for pair in case['pairs']:
            donor = lookup[pair['donor'],case['side']]['iris']
            a,b,xy = independent_context(target,donor,case['point'])
            assert a==pair['target_footprints'] and b==pair['after_single_query_exclusion']
            assert xy==pair['context'] and len(xy)==pair['fixed_control_upper_bound']
            assert pair['capacity_eligible']==(len(xy)>=8)
            counts.append(len(xy))
    # Deliberately corrupt query/wrong-map native contributors; retained context
    # must sample exactly the same values for BOTH controls.
    ramp = np.arange(32*32*3,dtype=np.float32).reshape(32,32,3)
    mask = np.ones((32,32),bool)
    query = np.array([[12.7,15.2]],np.float32)
    rec = fixed_control_capacity(mask,mask,query)
    altered = ramp.copy()
    for point in rec['query_union']:
        for x,y in contributors(point):
            altered[y,x] = -10000
    for delta in ([0,0],[3,0]):
        xy = rec['context']+delta
        assert np.array_equal(sample_native(ramp,xy),sample_native(altered,xy))
    assert not np.array_equal(sample_native(ramp,query),sample_native(altered,query))
    # Adding ANY learned arm's query footprint only removes fixed-bound samples.
    extra = contributors([20.4,20.3])
    subset = [p for p in rec['context'] if not (contributors(p)&extra)]
    assert 0<len(subset)<len(rec['context'])
    return dict(cases=len(capacity['cases']),pairs=len(counts),max_context_upper_bound=max(counts),
        pairs_with_eight_context_samples=sum(n>=8 for n in counts),
        cases_with_three_donors=capacity['cases_with_three_capacity_eligible_donors'],
        donor_query_poison_context_exact=True,all_arm_intersection_only_reduces_capacity=True,
        independent_contributor_enumeration=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_shot_argument(ap)
    args = ap.parse_args()
    shot = from_args(args)
    out = shot.composite/'r17_reflection'
    report = json.loads((shot.logs/'r17_reflection.json').read_text())
    capacity = json.loads((shot.logs/'r17_preflight.json').read_text())
    checks = dict(capacity=verify_capacity(capacity))
    assert sha(shot.logs/'r17_preflight.json')==report['preflight_sha256']
    assert sha(shot.composite/'r14_neural/proposals.npz')==report['proposals_sha256']
    rows = load_rows(shot,range(950,1000))
    labels = json.loads((ROOT/'scripts/data/r16_material_labels.json').read_text())
    masks = {(l['frame'],l['side']):region_masks(l,(64,64),1.5) for l in labels['frames']}
    samples, max_error, comparisons = 0, 0., 0
    with np.load(shot.composite/'r14_neural/proposals.npz') as proposals:
        for case in report['cases']:
            frame,side,index = case['frame'],case['side'],case['index']
            assert not case['clean_truth_known'] and 'truth' not in case
            with np.load(out/f'{frame}_{side}_{index}.npz') as data:
                assert data['xy'].shape==(50,1,2)
                assert data['excluded'].sum()==4
                assert np.array_equal(data['query'],[[frame-950,*case['point']]])
                expected = np.zeros((50,64,64),bool)
                for x,y in contributors(case['point']):
                    expected[frame-950,y,x] = True
                assert np.array_equal(data['excluded'],expected)
                hashes = {k:hashlib.sha256(data[k].tobytes()).hexdigest() for k in data.files}
                assert len(case['hidden_rgb_invariance'])==2
                for repeated in case['hidden_rgb_invariance']:
                    assert repeated['exact'] and repeated['arrays_sha256']==hashes
                    comparisons += len(hashes)
                assert np.max(np.abs(data['xy'][frame-950,0]-case['point']))<1e-5
                assert data['visible'][frame-950,0]
                assert all(r['donor']!=frame for r in case['records'])
                assert score_case(data,frame,side,case['point'],labels,proposals,rows)==case['records']
                for t in range(50):
                    xy = data['xy'][t]
                    valid = fits(np.ones((64,64),bool),contributors(xy[0]))
                    assert data['native_footprint'][t,0]==valid
                    assert data['usable_visibility'][t,0]==(valid and data['visible'][t,0])
                    rgb = rows[side][t+950]['rgb']
                    max_error = max(max_error,float(np.max(np.abs(sample_native(rgb,xy)-independent_sample(rgb,xy)))))
                    samples += 1
                for rec in case['records']:
                    material = masks[rec['donor'],side]
                    rgb = rows[side][rec['donor']]['rgb']
                    for arm in ARMS:
                        p = rec[arm]
                        xy = np.asarray(p['xy'],np.float32)
                        fp = contributors(xy[0])
                        found = [name for name,mask in material.items() if fits(mask,fp)]
                        assert p['material']==(found or ['unknown'])
                        assert np.array_equal(np.asarray(p['observed_donor_rgb'],np.float32),independent_sample(rgb,xy))
                        if arm=='raft_cycle':
                            samples += 1
    assert len(report['cases'])==12 and summarize(report['cases'])==report['summary']
    assert max_error<3e-5
    assert report['production_repairs']==0 and not report['scene_reflection_allowed']
    assert report['photometric_scores_computed']==0
    for side in SIDES:
        for frame in range(950,1000):
            assert report['source_crops_sha256'][f'{frame}_{side}']==hashlib.sha256(rows[side][frame]['rgb'].tobytes()).hexdigest()
    prior16 = json.loads((shot.logs/'r16_verification.json').read_text())
    prior15 = json.loads((shot.logs/'r15_verification.json').read_text())
    for name,value in {**prior16['code_sha256'],**prior15['source_hashes']}.items():
        assert sha(ROOT/'scripts'/name)==value
    prior_report16 = json.loads((shot.logs/'r16_visibility.json').read_text())
    prior_report15 = json.loads((shot.logs/'r15_dense.json').read_text())
    assert report['models']==[prior_report16['model'],prior_report15['model']]
    checks.update(native_coordinates_verified=samples,max_independent_sample_error=max_error,
        exact_substitution_runs=24,exact_array_hash_comparisons=comparisons,
        source_crops_unchanged=100,prior_code_hashes_unchanged=7,
        search_isolation=search_isolation(),fallback=fallback(shot))
    video = shot.final/'r17_reflection.mp4'
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    thumbs = []
    while True:
        ok,im = cap.read()
        if not ok:
            break
        assert im.shape==(736,1280,3)
        frame = 950+len(thumbs)
        if frame in (974,977,981,982,984,985,990,994):
            write(out/f'decoded_{frame}.png',im)
        thumb = cv2.resize(im,(384,221))
        cv2.putText(thumb,str(frame),(3,13),0,.4,(255,255,255),1)
        thumbs.append(thumb)
    cap.release()
    assert len(thumbs)==50 and abs(fps-float(shot.frame_rate))<.001
    for start in (0,25):
        sheet = np.concatenate([np.concatenate(thumbs[i:i+5],1) for i in range(start,start+25,5)],0)
        write(out/f'contact_{start}.jpg',sheet)
    checks['video'] = dict(frames=50,fps=fps,sha256=sha(video),bytes=video.stat().st_size,
                           visual_review='pending separate record',artist_approval=False)
    names = ('context_capacity.py','audit_r17_reflection.py','verify_r17.py')
    checks['code_sha256'] = {n:sha(ROOT/'scripts'/n) for n in names}
    for n in names:
        compile((ROOT/'scripts'/n).read_text(),n,'exec')
    (shot.logs/'r17_verification.json').write_text(json.dumps(plain(checks),indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(plain(checks),indent=2))


if __name__=='__main__':
    main()
