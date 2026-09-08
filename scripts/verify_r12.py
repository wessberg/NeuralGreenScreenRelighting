"""Round 12 sparse semantic review, generic adapter contracts and video decode checks."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
from pathlib import Path

import cv2
import numpy as np
from anamorphic import SQUEEZE
from iris_observations import eye_crop, observe_keyed
from native_iris import detect
from shot_paths import add_shot_argument, from_args


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    add_shot_argument(ap)
    shot=from_args(ap.parse_args())
    out=shot.composite/'r12_native'
    labels=json.loads((Path(__file__).parent/'data/r12_native_labels.json').read_text())
    report=json.loads((out/'observations_detail.json').read_text())
    masks=dict(np.load(out/'observations.npz'))
    annotations=[]
    for r in labels['frames']:
        i,s=r['frame'],r['side']
        ob=report['sides'][s][i]
        mask=masks[s+'_material'][i]
        def count(points):
            return sum(bool(mask[y,x]) for x,y in points)
        item={'frame':i,'side':s,'candidate_ok':ob['candidate_ok'],
              'reject_view':r['reject_view'],'iris_label_count':len(r['iris']),
              'iris_labels_admitted':count(r['iris']),
              'noniris_label_count':len(r['noniris']),'noniris_labels_admitted':count(r['noniris'])}
        for name in ['limbus','pupil']:
            if name in r:
                cx,cy,radius=ob['metrics'][name]
                err=np.abs(np.linalg.norm(np.asarray(r[name])-[cx,cy],axis=1)-radius)
                item[name+'_distance_p50_p95_px']=np.percentile(err,[50,95]).tolist()
        annotations.append(item)
    track=np.load(shot.relight/'face_landmarks.npz')
    meta=json.loads((shot.relight/'face_landmarks.json').read_text())
    rgb=cv2.imread(str(shot.fg(0)),cv2.IMREAD_UNCHANGED)[...,::-1]
    crop,reason=eye_crop(rgb,track['points'][0],meta['topology']['iris_left'],
                         meta['topology']['eyelid_left'],1.)
    assert reason is None
    points=track['points'][0].copy()
    points[:,:2]-=crop['origin']
    checks=[]
    # Geometrically transformed real pixels exercise adapter conventions, NOT model
    # generalization evidence. No simulated clean iris or synthetic repair is rendered.
    for scale,aspect in [(1.,1.),(2.,1.),(1.,SQUEEZE)]:
        width=max(1,round(crop['linear_rgb'].shape[1]*scale/aspect))
        height=max(1,round(crop['linear_rgb'].shape[0]*scale))
        im=cv2.resize(crop['linear_rgb'],(width,height),interpolation=cv2.INTER_NEAREST)
        pts=points.copy()
        pts[:,:2]*=[width/crop['linear_rgb'].shape[1],height/crop['linear_rgb'].shape[0]]
        # Extra native border keeps the dynamically sized ROI in frame.
        im=cv2.copyMakeBorder(im,32,32,32,32,cv2.BORDER_CONSTANT)
        pts[:,:2]+=32
        effective_aspect=(height/crop['linear_rgb'].shape[0])/(width/crop['linear_rgb'].shape[1])
        result=observe_keyed(im,pts,meta['topology'],pixel_aspect=effective_aspect)
        v=result['left']
        assert 'ob' in v and np.array_equal(v['output'],v['crop']['linear_rgb'])
        assert v['fallback']['changed_channels']==0
        checks.append({'scale':scale,'requested_pixel_aspect':aspect,
                       'effective_pixel_aspect':effective_aspect,'roi_shape':list(v['output'].shape),
                       'fallback_exact':True})
    missing=observe_keyed(rgb,track['points'][0],meta['topology'],pixel_aspect=1.,found=False)
    assert all(v['reason']=='missing_face_detection' for v in missing.values())
    # Native exclusion test at the detector boundary with an anisotropic coordinate model.
    im=crop['linear_rgb']*255
    yy,xx=np.mgrid[:im.shape[0],:im.shape[1]]
    hidden=(xx-round(crop['seed'][0]))**2+(yy-round(crop['seed'][1]))**2<9
    baseline=detect(im,crop['seed'],crop['radius'],hidden,pixel_aspect=SQUEEZE,eye_rotation=.2)
    changed=im.copy();changed[hidden]=[255,0,255]
    test=detect(changed,crop['seed'],crop['radius'],hidden,pixel_aspect=SQUEEZE,eye_rotation=.2)
    assert asdict(baseline.geometry)==asdict(test.geometry) and baseline.metrics==test.metrics
    assert np.array_equal(baseline.material,test.material)
    old=json.loads((shot.composite/'r10_iris'/'correspondence_detail.json').read_text())
    manifest=old['checks']['protected_manifest']
    changed_files=[p for p,h in manifest.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
    assert not changed_files,changed_files
    videos=[]
    for name in ['r12_native_source_fallback.mp4','r12_native_masks.mp4','r12_native_holdout.mp4']:
        path=shot.final/name
        if not path.exists():
            continue
        cap=cv2.VideoCapture(str(path));n=0;tiles=[];sheets=[]
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps=cap.get(cv2.CAP_PROP_FPS)
        while True:
            ok,frame=cap.read()
            if not ok:break
            # Preserve aspect and mark ordered frame numbers. Diagnostic masks need
            # native-size inspection too; these sheets establish temporal ordering only.
            thumb=cv2.resize(frame,(160,round(160*height/width)),interpolation=cv2.INTER_AREA)
            cv2.putText(thumb,str(n),(3,14),0,.4,(255,255,255),1)
            tiles.append(thumb);n+=1
            if len(tiles)==100:
                sheet=np.concatenate([np.concatenate(tiles[j:j+10],1) for j in range(0,100,10)],0)
                target=out/(path.stem+'_sheet_%04d.jpg'%n)
                if not cv2.imwrite(str(target),sheet):raise RuntimeError('sheet write')
                sheets.append(str(target.relative_to(shot.dir)));tiles=[]
        cap.release()
        if tiles:
            while len(tiles)%10:tiles.append(np.zeros_like(tiles[0]))
            sheet=np.concatenate([np.concatenate(tiles[j:j+10],1) for j in range(0,len(tiles),10)],0)
            target=out/(path.stem+'_sheet_%04d.jpg'%n)
            if not cv2.imwrite(str(target),sheet):raise RuntimeError('sheet write')
            sheets.append(str(target.relative_to(shot.dir)))
        expected={'r12_native_source_fallback.mp4':shot.frame_count,
                  'r12_native_masks.mp4':len(report['sides']['left']),
                  'r12_native_holdout.mp4':len(json.loads((shot.logs/'r12_correspondence.json').read_text())['cases'])}[name]
        assert n==expected,(name,n,expected)
        videos.append({'file':name,'decoded_frames':n,'fps':fps,'sheets':sheets,
                       'visual_inspection':'pending','artist_approval':'not requested'})
    result={'sparse_annotations':annotations,'annotation_provenance':labels['provenance'],
        'adapter_contract_checks':checks,'missing_detection_fallback':True,
        'native_anisotropic_detector_hidden_rgb_invariant':True,
        'protected_files_unchanged':len(manifest),'videos':videos,
        'generalization_status':'No second real clip with tracking is available in this workspace; transformed crops only test coordinate/fallback contracts.'}
    (shot.logs/'r12_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'annotation_frames':len(annotations),
        'false_positive_rejected_views':sum(r['candidate_ok'] and r['reject_view'] for r in annotations),
        'noniris_sparse_labels_admitted':sum(r['noniris_labels_admitted'] for r in annotations),
        'protected':len(manifest),'videos':[(v['file'],v['decoded_frames']) for v in videos]},indent=2))


if __name__=='__main__':main()
