"""Continuous source fallback and explicitly held failed radiance-model panels."""
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
    root=shot.composite/'r28_radiance'; out=root/'review'; out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((shot.relight/'neural_iris_observations/manifest.json').read_text())
    report=json.loads((shot.logs/'r28_radiance.json').read_text())
    data={}
    for side in ('left','right'):
        with np.load(root/f'predictions_{side}.npz') as d: data[side]={k:d[k] for k in d.files}
    for row in manifest['frames']:
        frame=row['frame']; panels=[]
        for side in ('left','right'):
            raw=display(read_rgb(shot,side,frame))
            source=cv2.resize(raw,(384,384),interpolation=cv2.INTER_NEAREST); overlay=source.copy()
            status='ABSTAIN: continuous geometry / material / scene light unvalidated'
            if frame in FRAMES:
                j=FRAMES.index(frame); counts=report['results'][side]['support'][str(j)]
                status=f'Sparse observation: {counts["labelled_candidate"]} labelled reflection candidates'
            if frame==115:
                d=data[side]; mask=d['supported']; xy=d['xy'][mask]
                scale=np.array([384/raw.shape[1],384/raw.shape[0]])
                for point in np.rint((xy+.5)*scale-.5).astype(int):
                    cv2.circle(overlay,tuple(point),1,(0,190,0),-1)
                for point in np.rint((d['xy'][mask&d['labelled_reflection']]+.5)*scale-.5).astype(int):
                    cv2.circle(overlay,tuple(point),2,(0,180,255),1)
                status=f'Seen 815: {len(xy)} scored; {int((mask&d["labelled_reflection"]).sum())} labelled catchlight pixels'
            header=np.zeros((96,768,3),'uint8')
            text(header,f'{side.upper()} source {row["source_frame"]} | SOURCE / SAMPLING OVERLAY',19)
            text(header,status,42); text(header,'Green = sampling coverage only. Orange = labelled reflection checks.',64)
            text(header,'Reflection prediction fails; exact keyed source retained; no scene replacement',86)
            panels.append(np.concatenate([header,np.concatenate([source,overlay],1)]))
        assert cv2.imwrite(str(out/f'{frame:06d}.png'),np.concatenate(panels))
    held=shot.composite/'r28_layers/review'; held.mkdir(parents=True,exist_ok=True)
    choices={}; clipping={}
    for side in ('left','right'):
        s=report['results'][side]['selection']
        choices[side]=[s['selected'],s['control'],s['best_reflected']]
    for k in range(3):
        panels=[]
        for side in ('left','right'):
            d=data[side]; i=choices[side][k]; supported=d['supported']; xy=d['xy'][supported]
            rgb=read_rgb(shot,side,115); material,reflection=d['predictions'][i,:,supported].transpose(1,0,2)
            observed=rgb[xy[:,1],xy[:,0]]; images=[]
            for values,title in zip((observed,material+reflection,material,reflection),
                                    ('Observed source','Predicted sum','Appearance + backing','Reflected radiance')):
                im=np.zeros_like(rgb); im[xy[:,1],xy[:,0]]=values
                im=cv2.resize(display(im),(240,240),interpolation=cv2.INTER_NEAREST)
                text(im,title,19,scale=.42); images.append(im)
            clip=int(np.any(material+reflection>1,axis=-1).sum()); clipping[f'{side}_{k}']=clip
            header=np.zeros((78,960,3),'uint8'); label=('Selected on 805','Appearance-only control','Best reflected model on 805')[k]
            text(header,f'{side.upper()} SEEN source 815 | {label} | model {i}',18,scale=.48)
            text(header,f'HELD FRAME, not motion; black = unscored; display clips {clip}/{len(xy)} predictions',38)
            text(header,'UNVALIDATED layers. Sampling includes catchlights; their predicted appearance fails.',56,color=(100,180,255))
            text(header,'A fitted backing value is not measured pupil material. Source eyes are not replaced.',73)
            panels.append(np.concatenate([header,np.concatenate(images,1)]))
        board=np.concatenate(panels)
        for j in range(24): assert cv2.imwrite(str(held/f'{k*24+j:06d}.png'),board)
    write(shot.logs/'r28_review_build.json',dict(choices=choices,clipped_pixels=clipping,
        continuous_frames=150,held_frames=72,source_not_replaced=True))
    print('R28_REVIEW_FRAMES_COMPLETE',flush=True)


if __name__=='__main__': main()
