"""Rejected whole-eye render follow-up: edit only proposed original reflection spots."""
import argparse
import json
from pathlib import Path
from diffusion_eye_trial import rgb8, save_rgb, sha, write
from color_pipeline import srgb_to_linear
from shot_paths import add_shot_argument, from_args
import cv2
import numpy as np


def main():
    ap=argparse.ArgumentParser(description=__doc__); add_shot_argument(ap)
    ap.add_argument('--input',default='r31_diffusion512')
    ap.add_argument('--name',default='r31_reflection_spots')
    args=ap.parse_args(); shot=from_args(args)
    if any(Path(x).name!=x for x in (args.input,args.name)): raise ValueError('Expected trial directory names')
    trial=shot.composite/args.input; root=shot.composite/args.name
    root.mkdir(exist_ok=True); (root/'review').mkdir(exist_ok=True); (root/'review_slow').mkdir(exist_ok=True)
    meta=json.loads((trial/'input.json').read_text()); trackmeta=json.loads((shot.relight/'face_landmarks.json').read_text())
    with np.load(trial/'source.npz') as z: sources,openings=z['linear'],z['masks']
    with np.load(shot.relight/'face_landmarks.npz') as z: points,apertures=z['points'],z['eyelid_aperture']
    h,w=sources.shape[1:3]; yy,xx=np.mgrid[:h,:w]; grids=np.stack([xx,yy],-1)
    lo=np.array(meta['crop_xyxy'][:2]); outputs=[]; masks=[]; records=[]
    for j,row in enumerate(meta['rows']):
        i=row['frame']; raw=sources[j]; display=rgb8(raw).astype(np.float32)/255
        predpath=trial/'inverse/basecolor'/f'{i:06d}.png'
        pred=cv2.imread(str(predpath))[...,::-1].astype(np.float32)/255
        pred=srgb_to_linear(cv2.resize(pred,(w,h),interpolation=cv2.INTER_LINEAR))
        result=raw.copy(); weights=np.zeros((h,w),np.float32); details=[]
        for k,side in enumerate(('left','right')):
            # Conservative open-eye trial. No generated repair through a blink.
            if not np.isfinite(apertures[i,k]) or apertures[i,k]<1.45: continue
            ids=trackmeta['topology'][f'iris_{side}']; iris=points[i,ids,:2]-lo
            if not np.isfinite(iris).all(): continue
            centre=iris[0]; radius=float(np.linalg.norm(iris[1:]-centre,axis=1).mean())
            distance=np.linalg.norm(grids-centre,axis=-1)
            interior=(distance<radius*.78)&(openings[j]>.95)
            core=interior&(display.min(-1)>.65)&(np.ptp(display,axis=-1)<.18)
            n,labels,stats,_=cv2.connectedComponentsWithStats(core.astype(np.uint8),8)
            accepted=np.zeros_like(core)
            for label in range(1,n):
                area=int(stats[label,cv2.CC_STAT_AREA])
                if 2<=area<=max(2,.15*np.pi*radius**2): accepted|=labels==label
            if not accepted.any(): continue
            away=cv2.distanceTransform((~accepted).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
            mask=np.clip((3-away)/1.25,0,1)*interior
            ring=interior&(away>=3)&(away<7)&(~core)
            if ring.sum()<12: continue
            # Match this small generated patch to adjacent source radiance; this
            # is an appearance correction, not certified clean material recovery.
            gain=np.clip(np.median(raw[ring],axis=0)/np.maximum(np.median(pred[ring],axis=0),.002),.2,5)
            candidate=pred*gain
            inside=mask>0
            result[inside]=raw[inside]*(1-mask[inside,None])+candidate[inside]*mask[inside,None]
            weights=np.maximum(weights,mask)
            details.append(dict(side=side,core_pixels=int(accepted.sum()),mask_pixels=int(inside.sum()),
                                ring_pixels=int(ring.sum()),linear_gain=gain.tolist()))
        assert np.isfinite(result).all() and np.array_equal(result[weights==0],raw[weights==0])
        outputs.append(result); masks.append(weights)
        records.append(dict(frame=i,source_frame=row['source_frame'],source_sha256=row['source_sha256'],
                            generated_basecolor_sha256=sha(predpath),mask_pixels=int((weights>0).sum()),eyes=details))
        # Same crop for source/result; proposal overlay remains explicitly visible.
        marked=rgb8(raw).copy(); selected=weights>0
        marked[selected]=np.rint(marked[selected]*.35+np.array([255,0,160])*.65).astype(np.uint8)
        panels=[]
        for title,im in [('SOURCE',rgb8(raw)),('REFLECTION SPOTS ONLY',rgb8(result)),('EDIT MASK',marked)]:
            header=np.zeros((42,384,3),np.uint8)
            cv2.putText(header,title,(8,19),0,.5,(235,235,235),1,cv2.LINE_AA)
            cv2.putText(header,f'Source {row["source_frame"]}',(8,36),0,.4,(235,235,235),1,cv2.LINE_AA)
            face=cv2.resize(im[...,::-1],(384,384),interpolation=cv2.INTER_LINEAR)
            # Fixed source-derived eye band, including all iris landmarks in this frame.
            ids=trackmeta['topology']['iris_left']+trackmeta['topology']['iris_right']
            ir=points[i,ids,:2]-lo; lower=np.floor(ir.min(0)-[25,22]).astype(int)
            upper=np.ceil(ir.max(0)+[25,22]).astype(int); lower=np.maximum(lower,0); upper=np.minimum(upper,[w,h])
            tile=im[lower[1]:upper[1],lower[0]:upper[0]][...,::-1]
            factor=min(384/tile.shape[1],160/tile.shape[0]); dims=(round(tile.shape[1]*factor),round(tile.shape[0]*factor))
            eye=np.zeros((160,384,3),np.uint8); small=cv2.resize(tile,dims,interpolation=cv2.INTER_NEAREST)
            y=(160-dims[1])//2; x=(384-dims[0])//2; eye[y:y+dims[1],x:x+dims[0]]=small
            panels.append(np.concatenate([header,face,eye]))
        board=np.concatenate(panels,1); footer=np.zeros((54,1152,3),np.uint8)
        cv2.putText(footer,'Generated basecolor patches with local colour matching. Automatic spot proposals; source retained elsewhere.',(8,20),0,.46,(235,235,235),1,cv2.LINE_AA)
        cv2.putText(footer,'Removal experiment only. No replacement scene reflections added. Not an accepted repair.',(8,42),0,.46,(235,235,235),1,cv2.LINE_AA)
        assert cv2.imwrite(str(root/'review'/f'{j:06d}.png'),np.concatenate([board,footer]))
        footer[27:]=0
        cv2.putText(footer,'SLOW MOTION: 12 fps. Source 59.94 fps. Removal only; no replacement scene reflections.',(8,42),0,.46,(235,235,235),1,cv2.LINE_AA)
        assert cv2.imwrite(str(root/'review_slow'/f'{j:06d}.png'),np.concatenate([board,footer]))
    np.savez_compressed(root/'patches.npz',linear=np.stack(outputs),masks=np.stack(masks))
    write(shot.logs/f'{args.name}.json',dict(rows=records,input_trial=args.input,code_sha256=sha(__file__),
        source_outside_mask_exact=True,scene_replacement=False,artist_accepted=False,production_enabled=False))
    print('R31_SPOT_TRIAL',[(r['source_frame'],r['mask_pixels']) for r in records],flush=True)


if __name__=='__main__': main()
