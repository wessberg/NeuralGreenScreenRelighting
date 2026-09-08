"""Reporter-only frozen EllSeg benchmark against round 12's native crops/labels.

No threshold fitting, fine tuning, donor matching or reflection removal. Reports
raw full-ellipse predictions, including their noniris overlap, without calling
these visible-material masks. Recorded sparse labels are reused evaluation data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from ellseg_iris import EllSeg, summary
from neural_iris_observations import overlay
from shot_paths import add_shot_argument, from_args


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    add_shot_argument(ap)
    args=ap.parse_args();shot=from_args(args)
    cache=shot.composite/'r10_iris'
    if not (cache/'source_landmarks.npz').exists():
        raise SystemExit('This diagnostic requires the reporter round-10 source cache.')
    lm=np.load(cache/'source_landmarks.npz')
    topo=json.loads((cache/'source_tracking.json').read_text())['topology']
    labels_path=Path(__file__).parent/'data/r12_native_labels.json'
    labels=json.loads(labels_path.read_text())
    annotations={(r['frame'],r['side']):r for r in labels['frames']}
    prior=json.loads((shot.composite/'r12_native/observations_detail.json').read_text())
    old_masks=np.load(shot.composite/'r12_native/observations.npz')
    out=shot.composite/'r13_ellseg';review=out/'review'
    review.mkdir(parents=True,exist_ok=True)
    model=EllSeg();rows=[];scores=[];native_masks={s:[] for s in ('left','right')}
    start=time.perf_counter()
    for i in range(len(lm['found'])):
        raw=cv2.imread(str(cache/'source_crops'/('%06d.png'%i)))
        if raw is None:raise RuntimeError('Missing cached source frame')
        tiles=[]
        for side in ('left','right'):
            pts=lm['points'][i,topo['iris_'+side],:2]-lm['origins'][i]
            origin=np.clip(np.rint(pts[0]-32).astype(int),0,np.array(raw.shape[1::-1])-64)
            rgb=raw[origin[1]:origin[1]+64,origin[0]:origin[0]+64,::-1].copy()
            pred=model.predict(rgb)
            native_masks[side].append(pred['labels'])
            row=summary(pred);row.update(frame=i,side=side);rows.append(row)
            if (i,side) in annotations:
                lab=annotations[i,side];pm=pred['labels'];old=old_masks[side+'_material'][i]
                def count(mask,points):return sum(bool(mask[y,x]) for x,y in points)
                score={'frame':i,'side':side,'reject_view':lab['reject_view'],
                    'iris_points':len(lab['iris']),'noniris_points':len(lab['noniris']),
                    'ellseg_iris_admitted':count(pm==1,lab['iris']),
                    'ellseg_noniris_admitted':count(pm==1,lab['noniris']),
                    'native_iris_admitted':count(old,lab['iris']),
                    'native_noniris_admitted':count(old,lab['noniris']),
                    'ellseg_pupil_pixels':int((pm==2).sum()),
                    'ellseg_iris_pixels':int((pm==1).sum())}
                for name in ('pupil','limbus'):
                    if name not in lab:continue
                    boundary=pred['ellipses'][name]['boundary_xy']
                    distance=np.linalg.norm(np.asarray(lab[name])[:,None,:]-boundary[None,:,:],axis=-1).min(1)
                    cx,cy,r=prior['sides'][side][i]['metrics'][name]
                    old_dist=np.abs(np.linalg.norm(np.asarray(lab[name])-[cx,cy],axis=1)-r)
                    score[name+'_ellseg_distance_p50_p95']=np.percentile(distance,[50,95]).tolist()
                    score[name+'_native_distance_p50_p95']=np.percentile(old_dist,[50,95]).tolist()
                scores.append(score)
            old=raw[origin[1]:origin[1]+64,origin[0]:origin[0]+64].copy()
            old[old_masks[side+'_material'][i]] = (old[old_masks[side+'_material'][i]]*.6+[0,90,0]).astype(np.uint8)
            panels=[rgb[...,::-1],old,overlay(rgb,pred)]
            titles=[side+' source '+str(i),'R12 CANDIDATE MATERIAL','ELLSEG FULL ELLIPSES']
            enlarged=[]
            for panel,title in zip(panels,titles):
                panel=cv2.resize(panel,(320,320),interpolation=cv2.INTER_NEAREST)
                panel=cv2.copyMakeBorder(panel,28,0,0,0,cv2.BORDER_CONSTANT)
                cv2.putText(panel,title,(4,19),0,.42,(255,255,255),1)
                enlarged.append(panel)
            tiles.append(np.concatenate(enlarged,1))
        if not cv2.imwrite(str(review/('%06d.png'%i)),np.concatenate(tiles,0)):
            raise RuntimeError('Review frame write failed')
        if i%100==0:print('EllSeg source benchmark',i,flush=True)
    (out/'detail.json').write_text(json.dumps(rows),encoding='utf-8')
    np.savez_compressed(out/'labels.npz',**{s:np.stack(v) for s,v in native_masks.items()})
    totals={k:sum(s[k] for s in scores) for k in ['iris_points','noniris_points',
        'ellseg_iris_admitted','ellseg_noniris_admitted','native_iris_admitted','native_noniris_admitted']}
    result={'provenance':model.provenance,'source_frames':len(lm['found']),'eye_frames':len(rows),
        'elapsed_seconds_including_io':time.perf_counter()-start,
        'annotations_sha256':hashlib.sha256(labels_path.read_bytes()).hexdigest(),
        'annotation_provenance':labels['provenance'],'boundary_uncertainty_px':labels['boundary_uncertainty_native_px'],
        'preprocessing':'native 64x64 reporter diagnostic ROI; width 320 Lanczos; central 240 rows; grayscale z-score; no tuning',
        'totals':totals,'sparse_scores':scores,
        'limits':'Reused sparse assistant labels on one clip; EllSeg is amodal. Noniris overlap is NOT visible-mask accuracy. No removal/correspondence or tracking validation.',
        'production_changes':False}
    (shot.logs/'r13_ellseg.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(totals,indent=2))


if __name__=='__main__':main()
