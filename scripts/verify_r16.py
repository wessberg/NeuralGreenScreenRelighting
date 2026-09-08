"""Verify frozen visibility evidence, native sampling, transforms and source fallback."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
import cv2
import numpy as np
import torch

from anamorphic import SQUEEZE
from audit_r14_neural import load_rows, SIDES, write
from audit_r15_dense import plain
from audit_r16_visibility import summarize, evaluate_labels, sha
from dense_iris_flow import sample_native
from material_footprints import region_masks, classify_footprints
from shot_paths import add_shot_argument, from_args
from temporal_eye_tracker import prepare_video, CHECKPOINT_SHA256
from temporal_iris import footprint_valid
from verify_r14 import search_isolation
from verify_r15 import fallback

ROOT=Path(__file__).resolve().parents[1]


def conventions():
    video=np.zeros((4,64,80,3),np.float32)
    q=np.array([[0,20.125,25.875],[1,40.5,37.25]],np.float32)
    cases=[]
    for aspect in (1.,SQUEEZE):
        prepared,query,gain,offset=prepare_video(video,q,pixel_aspect=aspect)
        for delta in ([3,0],[-3,0],[0,3],[0,-3]):
            native=(query[0,:,1:].numpy()+np.array(delta)*gain-offset)/gain
            error=float(np.max(np.abs(native-q[:,1:]-delta)))
            assert error<1e-5
            cases.append(dict(pixel_aspect=aspect,delta=delta,max_native_error=error))
        hidden=np.zeros(video.shape[:3],bool)
        hidden[1,30,40]=True
        poison=video.copy()
        poison[hidden]=[255,0,255]
        a=prepare_video(video,q,hidden,aspect)[0]
        b=prepare_video(poison,q,hidden,aspect)[0]
        assert torch.equal(a,b)
    # Native footprint classification: reject one bad contributor, edges and NaN.
    mask=np.ones((8,9),bool)
    mask[4,4]=False
    xy=np.array([[3.1,3.2],[1.3,1.4],[8,4],[-1,3],[np.nan,3]],np.float32)
    assert footprint_valid(mask,xy).tolist()==[False,True,False,False,False]
    ramp=np.arange(8*9*3,dtype=np.float32).reshape(8,9,3)
    p=np.array([[2.123,4.789]],np.float32)
    x,y=p[0]
    x0,y0=int(x),int(y)
    expected=sum(ramp[y0+j,x0+i]*((x-x0) if i else (1-x+x0))*((y-y0) if j else (1-y+y0))
                 for i in (0,1) for j in (0,1))
    assert np.max(np.abs(sample_native(ramp,p)[0]-expected))<3e-5
    return dict(transforms=cases,scrub_before_resize=True,native_footprint_negatives=True)


def independent_sample(rgb,xy):
    result=[]
    for x,y in xy:
        x0,y0=int(np.floor(x)),int(np.floor(y))
        if min(x0,y0)<0 or x0>=rgb.shape[1]-1 or y0>=rgb.shape[0]-1:
            result.append([0.,0.,0.])
            continue
        fx,fy=float(x)-x0,float(y)-y0
        value=np.zeros(3,np.float64)
        for j,wy in enumerate((1-fy,fy)):
            for i,wx in enumerate((1-fx,fx)):
                value+=rgb[y0+j,x0+i].astype(np.float64)*wx*wy
        result.append(value)
    return np.asarray(result,np.float32)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    add_shot_argument(ap)
    args=ap.parse_args()
    shot=from_args(args)
    report=json.loads((shot.logs/'r16_visibility.json').read_text())
    out=shot.composite/'r16_visibility'
    checks=dict(conventions=conventions())
    for key,path in (
        ('protocol_sha256',ROOT/'docs/r16-evaluation-protocol.md'),
        ('annotations_sha256',ROOT/'scripts/data/r16_material_labels.json'),
        ('reused_queries_sha256',ROOT/'scripts/data/r14_native_labels.json'),
        ('baseline_sha256',shot.logs/'r15_dense.json')):
        assert report[key]==sha(path)
    assert report['model']['sha256']==CHECKPOINT_SHA256==sha(ROOT/'dependencies/cotracker-weights/scaled_offline.pth')
    labels=json.loads((ROOT/'scripts/data/r16_material_labels.json').read_text())
    rows=load_rows(shot,range(950,1000))
    tracks={}
    samples=0
    max_error=0.
    for side in SIDES:
        with np.load(out/f'{side}_tracks.npz') as data:
            tracks[side]={k:data[k] for k in ('xy','visible','native_footprint','usable_visibility')}
            assert data['xy'].shape==(50,5,2)
            assert data['visible'].dtype==bool
            assert np.max(np.abs(data['xy'][27]-data['queries'][:,1:]))<1e-5
            assert data['visible'][27].all()  # forced, excluded from scored frames
        for k,i in enumerate(range(950,1000)):
            rgb=rows[side][i]['rgb']
            assert report['source_crops_sha256'][f'{i}_{side}']==hashlib.sha256(rgb.tobytes()).hexdigest()
            xy=tracks[side]['xy'][k]
            assert np.array_equal(tracks[side]['native_footprint'][k],footprint_valid(np.ones((64,64),bool),xy))
            assert np.array_equal(tracks[side]['usable_visibility'][k],tracks[side]['visible'][k] & tracks[side]['native_footprint'][k])
            err=float(np.max(np.abs(sample_native(rgb,xy)-independent_sample(rgb,xy))))
            max_error=max(max_error,err)
            samples+=len(xy)
    assert max_error<3e-5
    baseline={(r['frame'],r['side']):r for r in json.loads((shot.logs/'r15_dense.json').read_text())['blink']}
    with np.load(shot.composite/'r14_neural/proposals.npz') as proposals:
        assert plain(evaluate_labels(labels,tracks,baseline,proposals,rows))==report['records']
    assert summarize(report['records'])==report['summary']
    assert all(r['frame']!=977 for r in report['records'])
    # Recompute material labels by enumerating each contributor independently.
    coverage=[]
    for label,rec in zip(labels['frames'],report['records']):
        masks=region_masks(label,(64,64),labels['boundary_uncertainty_px'])
        yy,xx=np.mgrid[:63,:63]
        grid=np.column_stack([xx.ravel(),yy.ravel()]).astype(np.float32)
        coverage.append(dict(frame=label['frame'],side=label['side'],
            full_footprints={name:int(footprint_valid(mask,grid).sum()) for name,mask in masks.items()}))
        for arm in ('raft_cycle','raft_proposal','cotracker','cotracker_proposal'):
            classes=[]
            for x,y in rec[arm]['xy']:
                x0,y0=int(np.floor(x)),int(np.floor(y))
                found='unknown'
                if 0<=x0<63 and 0<=y0<63:
                    for name,mask in masks.items():
                        if all(mask[y0+j,x0+i] for j in (0,1) for i in (0,1)):
                            found=name
                classes.append(found)
            assert classes==rec[arm]['material']
    assert len(report['hidden_rgb_invariance'])==4 and all(x['exact'] for x in report['hidden_rgb_invariance'])
    assert report['production_repairs']==0 and not report['scene_reflection_allowed']
    for metric in report['synthetic']:
        data=np.load(out/f"synthetic_{metric['pixel_aspect']}_{metric['direction'][0]}.npz")
        truth=data['truth']
        assert ((truth[8:16]>=12) & (truth[8:16]<51)).all()
        for t in range(8,16):
            assert np.all(data['video'][t,12:52,12:52]==data['video'][t,12,12])
        good=np.ones((24,4),bool)
        good[0]=False
        good[8:16]=False
        err=np.linalg.norm(data['xy']-truth,axis=-1)
        assert metric['median_visible_endpoint_error']==float(np.median(err[good]))
        assert metric['p95_visible_endpoint_error']==float(np.percentile(err[good],95))
        assert metric['visible_detected']==int((data['usable_visibility'] & good).sum())
        assert metric['occluded_false_visible']==int(data['usable_visibility'][8:16].sum())
        assert metric['recovered_visible']==int(data['usable_visibility'][16:].sum())
    checks.update(native_samples=samples,max_independent_sample_error=max_error,
        labelled_attempts=80,material_footprint_coverage=coverage,
        hidden_rgb_substitutions=4,source_crops_unchanged=100,
        r14_search_isolation=search_isolation(),fallback=fallback(shot))
    # Retain evidence that earlier native-query/context and inference code is unchanged.
    prior=json.loads((shot.logs/'r15_verification.json').read_text())
    for name,value in prior['source_hashes'].items():
        assert sha(ROOT/'scripts'/name)==value
    checks['prior_r15_code_hashes_unchanged']=len(prior['source_hashes'])
    video=shot.final/'r16_visibility.mp4'
    cap=cv2.VideoCapture(str(video))
    fps=cap.get(cv2.CAP_PROP_FPS)
    thumbs=[]
    while True:
        ok,frame=cap.read()
        if not ok:
            break
        assert frame.shape==(736,1280,3)
        thumb=cv2.resize(frame,(384,221))
        cv2.putText(thumb,str(950+len(thumbs)),(3,13),0,.4,(255,255,255),1)
        thumbs.append(thumb)
    cap.release()
    assert len(thumbs)==50 and abs(fps-float(shot.frame_rate))<.001
    for start in (0,25):
        sheet=np.concatenate([np.concatenate(thumbs[i:i+5],1) for i in range(start,start+25,5)],0)
        write(out/f'contact_{start}.jpg',sheet)
    checks['video']=dict(frames=len(thumbs),fps=fps,sha256=sha(video),bytes=video.stat().st_size,
                          visual_review='pending separate record',artist_approval=False)
    names=('temporal_eye_tracker.py','material_footprints.py','audit_r16_visibility.py','verify_r16.py')
    checks['code_sha256']={name:sha(ROOT/'scripts'/name) for name in names}
    for name in names:
        compile((ROOT/'scripts'/name).read_text(),name,'exec')
    (shot.logs/'r16_verification.json').write_text(json.dumps(plain(checks),indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in checks.items() if k!='material_footprint_coverage'},indent=2))


if __name__=='__main__':
    main()
