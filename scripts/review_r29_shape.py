"""Continuous unchanged source and held, matched-geometry light-shape comparisons."""
import argparse
import json
import cv2
import numpy as np
from audit_r24_source import write
from audit_r27_photometry import read_rgb, FRAMES
from review_r27_photometry import display, text
from shot_paths import add_shot_argument, from_args


def main():
    ap=argparse.ArgumentParser(description=__doc__); add_shot_argument(ap); shot=from_args(ap.parse_args())
    root=shot.composite/'r29_shape'; out=root/'review'; out.mkdir(parents=True,exist_ok=True)
    report=json.loads((shot.logs/'r29_shape.json').read_text())['results']
    manifest=json.loads((shot.relight/'neural_iris_observations/manifest.json').read_text())
    data={}
    for side in ('left','right'):
        with np.load(root/f'predictions_{side}.npz') as z: data[side]={k:z[k] for k in z.files}
    for row in manifest['frames']:
        panels=[]; frame=row['frame']
        for side in ('left','right'):
            raw=display(read_rgb(shot,side,frame))
            src=cv2.resize(raw,(384,384),interpolation=cv2.INTER_NEAREST); overlay=src.copy()
            status='ABSTAIN: no validated continuous geometry, material or scene replacement'
            if frame in FRAMES:
                j=FRAMES.index(frame); z=data[side]; xy=z[f'xy_{j}']; mask=z[f'mask_{j}']; label=z[f'label_{j}']
                scale=np.array([384/raw.shape[1],384/raw.shape[0]])
                for q in np.rint((xy[mask]+.5)*scale-.5).astype(int): cv2.circle(overlay,tuple(q),1,(0,180,0),-1)
                for q in np.rint((xy[mask&label]+.5)*scale-.5).astype(int): cv2.circle(overlay,tuple(q),2,(0,180,255),1)
                status=f'{int(mask.sum())} sampled pixels; {int((mask&label).sum())} labelled reflections; NOT layer confidence'
            header=np.zeros((96,768,3),'uint8')
            text(header,f'{side.upper()} source {row["source_frame"]} | SOURCE / SAMPLING OVERLAY',19)
            text(header,status,42)
            text(header,'Original eyes and catchlights retained through motion and blinks.',64)
            text(header,'Green: sampling support. Orange: labelled reflections. No production repair.',86)
            panels.append(np.concatenate([header,np.concatenate([src,overlay],1)]))
        assert cv2.imwrite(str(out/f'{frame:06d}.png'),np.concatenate(panels))
    held=shot.composite/'r29_layers/review'; held.mkdir(parents=True,exist_ok=True)
    choices={}; clipping={}
    for side in ('left','right'):
        r=report[side]
        c=r['records'][r['selected']]['config'] if r['selected'] is not None else 0
        choices[side]={a:next(x['index'] for x in r['records'] if x['config']==c and x['arm']==a)
                       for a in ('broad','narrow','spatial_backing')}
    for j,frame in enumerate(FRAMES):
        panels=[]
        for side in ('left','right'):
            z=data[side]; xy=z[f'xy_{j}']; mask=np.ones(len(xy),bool) if j<4 else z[f'mask_{j}']
            raw=read_rgb(shot,side,frame); q=xy[mask]
            lo=xy.min(0)-3; hi=xy.max(0)+4
            mid=(lo+hi)//2; radius=int(np.ceil(np.max(hi-lo)/2))
            x0,y0=np.maximum(mid-radius,0); x1,y1=np.minimum(mid+radius,[raw.shape[1],raw.shape[0]])
            layers={a:z[f'layers_{i}_{j}'][:,mask] for a,i in choices[side].items()}
            values=[raw[q[:,1],q[:,0]],layers['broad'].sum(0),layers['narrow'].sum(0),
                    layers['spatial_backing'].sum(0),layers['narrow'][1]]
            titles=['Observed','Broad light sum','Narrow light sum','Spatial backing sum','Narrow reflection']
            images=[]
            for value,title in zip(values,titles):
                im=np.zeros_like(raw); im[q[:,1],q[:,0]]=value
                im=cv2.resize(display(im[y0:y1,x0:x1]),(200,200),interpolation=cv2.INTER_NEAREST)
                text(im,title,18,scale=.4); images.append(im)
            header=np.zeros((90,1000,3),'uint8')
            selected=report[side]['selected']
            choice='UNSELECTED fixed witness 0 / BOX3' if selected is None else 'Geometry / BOX from four-frame reflection-balanced selection'
            text(header,f'{side.upper()} source {frame+700} | {"FITTING" if j<4 else "SEEN DIAGNOSTIC"} | HELD STILL, not motion',18,scale=.48)
            text(header,choice+'; arms share geometry and scored pixels',39)
            text(header,'Unvalidated explanations. Spatial backing can compete with light; a sharp sum is not clean material.',60)
            text(header,f'{len(q)} pixels; black = unscored. Source eyes are never replaced.',80)
            panels.append(np.concatenate([header,np.concatenate(images,1)]))
            clipping[f'{side}_{frame}']=[int(np.any(v>1,axis=-1).sum()) for v in values]
        board=np.concatenate(panels)
        for k in range(24): assert cv2.imwrite(str(held/f'{j*24+k:06d}.png'),board)
    write(shot.logs/'r29_review_build.json',dict(choices=choices,continuous_frames=150,
        held_frames=144,clipped_pixels=clipping,exact_source_retained=True))
    print('R29_REVIEW_COMPLETE',flush=True)


if __name__=='__main__': main()
