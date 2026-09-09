"""Temporal exact-source review and explicit held failed-model comparisons."""
import argparse
import json
import cv2
import numpy as np
from audit_r24_source import write, sha
from audit_r27_photometry import read_rgb, FRAMES
from color_pipeline import linear_to_srgb
from shot_paths import add_shot_argument, from_args


def display(rgb):
    return np.clip(linear_to_srgb(np.maximum(rgb,0))[...,::-1]*255,0,255).astype('uint8')


def text(board,words,y,x=8,scale=.43,color=(230,230,230)):
    cv2.putText(board,words,(x,y),0,scale,color,1,cv2.LINE_AA)


def main():
    ap=argparse.ArgumentParser(description=__doc__); add_shot_argument(ap); shot=from_args(ap.parse_args())
    root=shot.composite/'r27_photometry'; out=root/'review'; out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((shot.relight/'neural_iris_observations/manifest.json').read_text())
    geometry={s:json.loads((root/f'geometry_{s}.json').read_text()) for s in ('left','right')}
    payload={}
    for side in ('left','right'):
        with np.load(root/f'predictions_{side}.npz') as d: payload[side]={k:d[k] for k in d.files}
    for row in manifest['frames']:
        frame=row['frame']; panels=[]
        for side in ('left','right'):
            raw=display(read_rgb(shot,side,frame))
            source=cv2.resize(raw,(384,384),interpolation=cv2.INTER_NEAREST); overlay=source.copy()
            scale=np.array([384/raw.shape[1],384/raw.shape[0]])
            status='ABSTAIN: no continuous inverse fit; exact keyed source retained'
            if frame in FRAMES:
                obs=next(o for o in geometry[side]['observations'] if o['frame']==frame)
                points=np.rint((np.array(obs['label']['limbus'])+.5)*scale-.5).astype(int)
                for point in points: cv2.circle(overlay,tuple(point),2,(0,210,255),-1)
                status='Boundary-conditioned source observation; no clean-material claim'
            if frame==115:
                points=np.rint((payload[side]['xy']+.5)*scale-.5).astype(int)
                for point in points: cv2.circle(overlay,tuple(point),1,(0,255,0),-1)
                status=f'Fresh check: {len(points)} pixels; ZERO labelled catchlight coverage'
            header=np.zeros((96,768,3),'uint8')
            text(header,f'{side.upper()} source {row["source_frame"]} | SOURCE / DIAGNOSTIC OVERLAY',19)
            text(header,status,42); text(header,'Original eye and catchlight retained throughout both panels',64)
            text(header,'Two-layer fit failed validation; no scene-light replacement enabled',86)
            panels.append(np.concatenate([header,np.concatenate([source,overlay],1)]))
        assert cv2.imwrite(str(out/f'{frame:06d}.png'),np.concatenate(panels))
    held=shot.composite/'r27_layers/review'; held.mkdir(parents=True,exist_ok=True)
    choices={}; clipping={}
    for side in ('left','right'):
        selection=json.loads((root/f'selection_{side}.json').read_text())
        records=selection['records']
        choices[side]=[selection['selected']]+[
            min((i for i,r in enumerate(records) if r['reflection_ridge']==ridge),
                key=lambda i:records[i]['development_linear_mse']) for ridge in (1e-4,1e-2,1.)]
    for k in range(4):
        panels=[]
        for side in ('left','right'):
            rgb=read_rgb(shot,side,115); d=payload[side]; index=choices[side][k]
            p=d['xy']; material,reflection=d['predictions'][index]
            observed=rgb[p[:,1],p[:,0]]
            images=[]
            for values,title in zip((observed,material+reflection,material,reflection),
                                    ('Source check','Predicted sum','Appearance layer','Reflection layer')):
                image=np.zeros_like(rgb); image[p[:,1],p[:,0]]=values
                image=cv2.resize(display(image),(240,240),interpolation=cv2.INTER_NEAREST)
                text(image,title,19,scale=.45); images.append(image)
            header=np.zeros((60,960,3),'uint8')
            description='Selected material-only control' if k==0 else f'Two-layer ridge {(1e-4,1e-2,1.)[k-1]:g}; best DEVELOPMENT score'
            clipped=int(np.count_nonzero(np.any(material+reflection>1,axis=1)))
            clipping[f'{side}_{k}']=clipped
            text(header,f'{side.upper()} source 815 | {description} | model {index}',18,scale=.48)
            text(header,f'HELD FRAME, not motion. Black = unscored. Display clips {clipped}/{len(p)} predicted pixels.',38)
            text(header,'UNVALIDATED LAYERS - no removal; zero labelled catchlight pixels in this check',55,color=(100,180,255))
            panels.append(np.concatenate([header,np.concatenate(images,1)]))
        board=np.concatenate(panels)
        for j in range(24): assert cv2.imwrite(str(held/f'{k*24+j:06d}.png'),board)
    write(shot.logs/'r27_review_build.json',dict(held_model_indices=choices,display_clipped_pixels=clipping,
        continuous_frames=len(manifest['frames']),held_frames=96,source_not_replaced=True))
    print('R27_REVIEW_FRAMES_COMPLETE')


if __name__=='__main__': main()
