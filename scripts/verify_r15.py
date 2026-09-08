"""Verify native RAFT conventions, recorded donor provenance, exclusions and fallback."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2
import numpy as np
import torch

from anamorphic import SQUEEZE
from audit_r14_neural import load_rows
from audit_r15_dense import plain, summarize
from dense_iris_flow import DenseEyeFlow, native_flow, prepare, sample_native, query_flow
from iris_correspondence import disjoint_footprints
from shot_paths import add_shot_argument, from_args
from temporal_iris import apply_eye_repair
from verify_r14 import search_isolation

ROOT=Path(__file__).resolve().parents[1]


def conventions(shot):
    flow=DenseEyeFlow()
    row=load_rows(shot,[959])['left'][959]
    rgb=row['rgb']
    # Analytic affine radiance verifies axis order and actual bilinear coefficients.
    yy,xx=np.mgrid[:64,:64]
    ramp=np.stack([xx+3*yy,2*xx+yy,xx+yy],-1).astype(np.float32)
    q=np.array([[12.999,15.123],[19.371,42.789]],np.float32)
    expected=np.stack([q[:,0]+3*q[:,1],2*q[:,0]+q[:,1],q[:,0]+q[:,1]],-1)
    sample_error=float(np.max(np.abs(sample_native(ramp,q)-expected)))
    assert sample_error<2e-5
    assert not sample_native(ramp,np.array([[-1,20],[63,30]],np.float32)).any()
    mapping,learned=[],[]
    for aspect in (1.,SQUEEZE):
        prepared=prepare(rgb,pixel_aspect=aspect)
        nh,nw=prepared.shape[-2:]
        for dx,dy in ((3,0),(-3,0),(0,3),(0,-3)):
            known=torch.zeros((1,2,nh,nw))
            known[:,0]=dx*nw/64
            known[:,1]=dy*nh/64
            native=native_flow(known,(64,64))
            err=float(np.max(np.abs(native-[dx,dy])))
            assert err<1e-6
            mapped,inside,cycle=query_flow(native,-native,q)
            assert inside.all() and np.max(cycle)<1e-5
            assert np.max(np.abs(mapped-q-[dx,dy]))<1e-5
            mapping.append(dict(pixel_aspect=aspect,shift=[dx,dy],native_error=err))
            # Known translated observed image: only a coordinate test, not iris repair.
            shifted=cv2.warpAffine(rgb,np.array([[1,0,dx],[0,1,dy]],np.float32),
                                   (64,64),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
            fwd,back=flow.pair(rgb,shifted,pixel_aspect=aspect)
            error=np.linalg.norm(fwd[16:48,16:48]-[dx,dy],axis=-1)
            centre=np.median(fwd[16:48,16:48].reshape(-1,2),axis=0)
            assert float(centre @ [dx,dy])>0, (aspect,dx,dy,centre)
            learned.append(dict(pixel_aspect=aspect,shift=[dx,dy],
                median_native_endpoint_error=float(np.median(error)),
                p95_native_endpoint_error=float(np.percentile(error,95))))
    # Scrubbing before resampling, including anamorphic resampling.
    hidden=np.zeros((64,64),bool)
    hidden[20,30]=True
    altered=rgb.copy()
    altered[hidden]=[255,0,255]
    assert torch.equal(prepare(rgb,hidden,SQUEEZE),prepare(altered,hidden,SQUEEZE))
    # A query value cannot leak into a disjoint photometric context statistic.
    context=np.array([[20.2,20.4],[31.5,32.3],[40.9,41.1]],np.float32)
    query=np.array([[30.9,31.1],[15.3,15.8]],np.float32)
    keep=disjoint_footprints(context,query)
    poisoned=ramp.copy()
    for x,y in np.floor(query).astype(int):
        poisoned[y:y+2,x:x+2]=255
    assert keep.sum()>0 and (~keep).sum()>0
    assert np.array_equal(sample_native(ramp,context[keep]),sample_native(poisoned,context[keep]))
    assert not np.array_equal(sample_native(ramp,query),sample_native(poisoned,query))
    return dict(analytic_sample_max_error=sample_error,native_mapping=mapping,
        learned_known_translation=learned,pre_resize_hidden_rgb_exact=True,
        donor_query_poison_context_exact=True,model=flow.provenance)


def fallback(shot):
    protected=json.loads((shot.composite/'r10_iris/correspondence_detail.json').read_text())['checks']['protected_manifest']
    changed=[p for p,h in protected.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
    assert not changed,changed
    manifest=json.loads((shot.relight/'neural_iris_observations/manifest.json').read_text())
    count=0
    for row in manifest['frames']:
        raw=cv2.imread(str(shot.fg(row['frame'])),cv2.IMREAD_UNCHANGED)[...,::-1]
        for side,eye in row['eyes'].items():
            if 'origin_xy' not in eye:
                continue
            source=np.load(shot.relight/'neural_iris_observations'/f"{row['frame']:06d}_{side}.npz")['linear_rgb']
            h,w=source.shape[:2]
            x,y=eye['origin_xy']
            assert np.array_equal(source,raw[y:y+h,x:x+w])
            result,info=apply_eye_repair(source,np.ones((h,w),bool),geometry_reliable=False)
            assert np.array_equal(result,source) and info['changed_channels']==0
            count+=1
    assert count==300 and len(protected)==304
    return dict(exact_keyed_eye_fallbacks=count,protected_hashes_unchanged=len(protected))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    add_shot_argument(ap)
    args=ap.parse_args()
    shot=from_args(args)
    report=json.loads((shot.logs/'r15_dense.json').read_text())
    checks=dict(conventions=conventions(shot),r14_search_isolation=search_isolation())
    for key,path in [('protocol_sha256',ROOT/'docs/r15-evaluation-protocol.md'),
                     ('annotations_sha256',ROOT/'scripts/data/r14_native_labels.json')]:
        assert report[key]==hashlib.sha256(path.read_bytes()).hexdigest()
    assert report['model']==checks['conventions']['model']
    assert report['attempted_iris_pixels']==30 and report['attempted_reflection_pixels']==12
    assert report['production_repairs']==0 and not report['scene_reflection_allowed']
    assert report['summary']==plain(summarize(report['cases']))
    pairs=0
    for category,cases in [('iris',report['cases']),('reflection',report['reflection_probes'])]:
        assert len(cases)==6
        for case in cases:
            assert all(v['exact'] for v in case['hidden_rgb_invariance'])
            assert len(case['hidden_rgb_invariance'])==2
            i,side=case['frame'],case['side']
            assert case['donor_ids']==[i+d for d in (-12,-8,-4,4,8,12)]
            rows=load_rows(shot,[i]+case['donor_ids'])[side]
            query=np.array(case['points'],np.float32)
            if category=='iris':
                assert case['truth']==[rows[i]['rgb'][y,x].tolist() for x,y in case['points']]
            else:
                assert not case['clean_truth_known'] and 'truth' not in case
            with np.load(shot.composite/'r15_dense'/f'{i}_{side}_{category}_flows.npz') as fields:
                for pair in case['result']['pairs']:
                    donor=pair['donor_frame']
                    qraft,_,cycle=query_flow(fields[f'{donor}_forward'],fields[f'{donor}_backward'],query)
                    for method in ('fixed','raft'):
                        p=pair[method]
                        qxy=np.array(p['mapped'],np.float32)
                        assert np.array_equal(qxy,qraft if method=='raft' else query)
                        assert np.array_equal(np.asarray(p['rgb'],np.float32),sample_native(rows[donor]['rgb'],qxy))
                        fp=p['footprints']
                        union=np.asarray(fp['query_union']).reshape(-1,2)
                        for name in ('context','wrong'):
                            assert disjoint_footprints(np.asarray(fp[name]).reshape(-1,2),union).all()
                        if method=='raft':
                            assert plain(cycle)==p['roundtrip']
                    pairs+=1
    checks.update(recorded_pairs_verified=pairs,hidden_rgb_exact_cases=12,
                  hidden_rgb_substitutions=24,fallback=fallback(shot))
    video=shot.final/'r15_dense.mp4'
    cap=cv2.VideoCapture(str(video))
    fps=cap.get(cv2.CAP_PROP_FPS)
    frames=[]
    while True:
        ok,im=cap.read()
        if not ok:
            break
        thumb=cv2.resize(im,(288,221))
        cv2.putText(thumb,str(950+len(frames)),(3,13),0,.4,(255,255,255),1)
        frames.append(thumb)
    cap.release()
    assert len(frames)==50 and abs(fps-float(shot.frame_rate))<.001
    for start in (0,25):
        sheet=np.concatenate([np.concatenate(frames[i:i+5],1) for i in range(start,start+25,5)],0)
        assert cv2.imwrite(str(shot.composite/'r15_dense'/f'contact_{start}.jpg'),sheet)
    checks['video']=dict(file=str(video.relative_to(shot.dir)),frames=50,fps=fps,
        visual_review='pending',artist_approval=False,sha256=hashlib.sha256(video.read_bytes()).hexdigest())
    checks['source_hashes']={name:hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest()
        for name in ('dense_iris_flow.py','audit_r15_dense.py','verify_r15.py')}
    for name in checks['source_hashes']:
        compile((ROOT/'scripts'/name).read_text(),name,'exec')
    (shot.logs/'r15_verification.json').write_text(json.dumps(plain(checks),indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(plain(checks),indent=2))


if __name__=='__main__':
    main()
