"""Reporter RITnet baseline on frozen sparse labels and the selected motion segment."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from ritnet_iris import RITnet
from shot_paths import add_shot_argument,from_args


def main():
    ap=argparse.ArgumentParser(description=__doc__);add_shot_argument(ap)
    shot=from_args(ap.parse_args())
    labels=json.loads((Path(__file__).parent/'data/r12_native_labels.json').read_text())
    annotations={(r['frame'],r['side']):r for r in labels['frames']}
    ids=sorted(set(range(shot.source_start_frame,shot.source_start_frame+shot.frame_count)) |
               {r['frame'] for r in labels['frames']})
    cache=shot.composite/'r10_iris';out=shot.composite/'r13_ritnet'
    review=out/'review';review.mkdir(parents=True,exist_ok=True)
    lm=np.load(cache/'source_landmarks.npz')
    topo=json.loads((cache/'source_tracking.json').read_text())['topology']
    model=RITnet();scores=[];masks={};motion=0
    for i in ids:
        raw=cv2.imread(str(cache/'source_crops'/('%06d.png'%i)))
        if raw is None:raise RuntimeError('Missing cached native source')
        tiles=[]
        for side in ('left','right'):
            pts=lm['points'][i,topo['iris_'+side],:2]-lm['origins'][i]
            origin=np.clip(np.rint(pts[0]-32).astype(int),0,np.array(raw.shape[1::-1])-64)
            rgb=raw[origin[1]:origin[1]+64,origin[0]:origin[0]+64,::-1].copy()
            pred=model.predict(rgb);masks['%06d_%s'%(i,side)]=pred
            if (i,side) in annotations:
                lab=annotations[i,side]
                score={k:lab[k] for k in ['frame','side','reject_view']}
                score.update(iris_points=len(lab['iris']),noniris_points=len(lab['noniris']),
                    iris_admitted=sum(pred[y,x]==2 for x,y in lab['iris']),
                    noniris_admitted=sum(pred[y,x]==2 for x,y in lab['noniris']),
                    iris_pixels=int((pred==2).sum()),pupil_pixels=int((pred==3).sum()))
                for name,mask in [('pupil',pred==3),('limbus',pred>=2)]:
                    if name not in lab:continue
                    contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
                    if contours:
                        points=np.concatenate(contours)[:,0,:]
                        dist=np.linalg.norm(np.asarray(lab[name])[:,None,:]-points[None,:,:],axis=-1).min(1)
                        score[name+'_boundary_p50_p95']=np.percentile(dist,[50,95]).tolist()
                    else:score[name+'_boundary_p50_p95']=None
                scores.append({k:int(v) if isinstance(v,np.integer) else v for k,v in score.items()})
            color=np.zeros_like(rgb);color[pred==1]=[0,120,180];color[pred==2]=[0,180,0];color[pred==3]=[180,0,180]
            overlay=cv2.addWeighted(rgb,.7,color,.5,0)
            panels=[]
            for panel,title in [(rgb,side+' SOURCE '+str(i)),(overlay,'RITNET VISIBLE-PART PROPOSAL')]:
                im=cv2.resize(panel[...,::-1],(320,320),interpolation=cv2.INTER_NEAREST)
                im=cv2.copyMakeBorder(im,28,0,0,0,cv2.BORDER_CONSTANT)
                cv2.putText(im,title,(4,19),0,.42,(255,255,255),1);panels.append(im)
            tiles.append(np.concatenate(panels,1))
        board=np.concatenate(tiles,0)
        if (i,'left') in annotations:
            if not cv2.imwrite(str(out/('sparse_%06d.png'%i)),board):raise RuntimeError('write')
        if shot.source_start_frame<=i<shot.source_start_frame+shot.frame_count:
            if not cv2.imwrite(str(review/('%06d.png'%motion)),board):raise RuntimeError('write')
            motion+=1
        if i%25==0:print('RITnet source',i,flush=True)
    np.savez_compressed(out/'labels.npz',**masks)
    result={'provenance':model.provenance,'eye_frames':len(masks),'source_frame_ids':ids,
        'annotation_provenance':labels['provenance'],'scores':scores,
        'totals':{k:sum(r[k] for r in scores) for k in ['iris_points','noniris_points','iris_admitted','noniris_admitted']},
        'limits':'Sparse reused labels, one RGB clip, no threshold tuning. Boundaries are raster contours, unlike EllSeg continuous ellipse curves. No positive repair evidence.',
        'scene_reflection_allowed':False}
    (shot.logs/'r13_ritnet.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result['totals'],indent=2))


if __name__=='__main__':main()
