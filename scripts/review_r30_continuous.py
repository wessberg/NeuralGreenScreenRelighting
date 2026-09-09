"""Continuous exact-source sampling review and labelled still-held model comparison."""
import argparse
import json
import cv2
import numpy as np
from audit_r24_source import write
from audit_r27_photometry import read_rgb, FRAMES
from review_r27_photometry import display, text
from eye_continuous_fit import ARMS
from shot_paths import add_shot_argument, from_args


def main():
    ap=argparse.ArgumentParser(description=__doc__); add_shot_argument(ap); shot=from_args(ap.parse_args())
    root=shot.composite/'r30_continuous'; out=root/'review'; out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((shot.relight/'neural_iris_observations/manifest.json').read_text())
    report=json.loads((shot.logs/'r30_continuous.json').read_text())
    data={}; masks={}
    for side in ('left','right'):
        with np.load(root/f'predictions_{side}.npz') as z: data[side]=dict(z)
        with np.load(root/f'masks_{side}.npz') as z: masks[side]=dict(z)
    for row in manifest['frames']:
        panels=[]; frame=row['frame']
        for side in ('left','right'):
            raw=display(read_rgb(shot,side,frame)); src=cv2.resize(raw,(384,384),interpolation=cv2.INTER_NEAREST)
            overlay=src.copy(); status='ABSTAIN: no validated continuous geometry or clean material'
            if frame in FRAMES:
                j=FRAMES.index(frame); z=data[side]; xy=z[f'xy_{j}']; mask=masks[side][f'mask_{j}']; label=z[f'label_{j}']
                scale=np.array([384/raw.shape[1],384/raw.shape[0]])
                for q in np.rint((xy[mask]+.5)*scale-.5).astype(int): cv2.circle(overlay,tuple(q),1,(0,180,0),-1)
                for q in np.rint((xy[mask&label]+.5)*scale-.5).astype(int): cv2.circle(overlay,tuple(q),2,(0,180,255),1)
                status=f'{int(mask.sum())} supported samples; {int((mask&label).sum())} labelled reflections; NOT confidence'
            header=np.zeros((96,768,3),'uint8')
            text(header,f'R30 {side.upper()} source {row["source_frame"]} | SOURCE / SAMPLING OVERLAY',19)
            text(header,status,42)
            text(header,'Local shared pose/light fit is diagnostic; original eyes and catchlights remain.',64)
            text(header,'Green: joint sampling. Orange: reused reflection labels. No scene replacement.',86)
            panels.append(np.concatenate([header,np.concatenate([src,overlay],1)]))
        assert cv2.imwrite(str(out/f'{frame:06d}.png'),np.concatenate(panels))
    held=shot.composite/'r30_layers/review'; held.mkdir(parents=True,exist_ok=True)
    clipping={}
    for j,frame in enumerate(FRAMES):
        panels=[]
        for side in ('left','right'):
            z=data[side]; xy=z[f'xy_{j}']; mask=masks[side][f'mask_{j}'].copy()
            if j<4: mask=np.all([z[f'-1_{arm}_valid_{j}'] for arm in ARMS],axis=0)
            raw=read_rgb(shot,side,frame); q=xy[mask]
            lo=xy.min(0)-3; hi=xy.max(0)+4; mid=(lo+hi)//2; radius=int(np.ceil(np.max(hi-lo)/2))
            x0,y0=np.maximum(mid-radius,0); x1,y1=np.minimum(mid+radius,[raw.shape[1],raw.shape[0]])
            values=[raw[q[:,1],q[:,0]]]+[z[f'-1_{arm}_layers_{j}'][:,mask].sum(0) for arm in ARMS]
            values.append(z[f'-1_coupled_layers_{j}'][2,mask])
            titles=['Observed','Fixed seed sum','Light fit sum','Pose + light sum','Spatial backing sum','Coupled reflection']
            images=[]
            for value,title in zip(values,titles):
                im=np.zeros_like(raw); im[q[:,1],q[:,0]]=value
                im=cv2.resize(display(im[y0:y1,x0:x1]),(200,200),interpolation=cv2.INTER_NEAREST)
                text(im,title,18,scale=.4); images.append(im)
            header=np.zeros((90,1200,3),'uint8')
            choice=report['results'][side]['evaluation']['selected']
            text(header,f'{side.upper()} source {frame+700} | {"FITTING" if j<4 else "SEEN DIAGNOSTIC"} | ONE-SECOND STILL HOLD, not motion',18,scale=.48)
            text(header,f'Four-fold selection: {choice or "ABSTAIN"}. All arms share displayed native pixels; local fits are unvalidated.',39)
            text(header,'A sharper sum does not establish clean iris/pupil material. Competing backing and reflected layers remain unresolved.',60)
            text(header,f'{len(q)} pixels; black = unscored. Source eyes are never replaced.',80)
            panels.append(np.concatenate([header,np.concatenate(images,1)]))
            clipping[f'{side}_{frame}']=[int(np.any(v>1,axis=-1).sum()) for v in values]
        board=np.concatenate(panels)
        for k in range(24): assert cv2.imwrite(str(held/f'{j*24+k:06d}.png'),board)
    write(shot.logs/'r30_review_build.json',dict(continuous_frames=150,held_frames=144,
        clipped_pixels=clipping,exact_source_retained=True))
    print('R30_REVIEW_COMPLETE',flush=True)


if __name__=='__main__': main()
