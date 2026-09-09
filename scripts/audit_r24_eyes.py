"""Independent native BOX integration against Cycles connected-eye renders."""
import argparse
import json
import os
from pathlib import Path
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
import cv2
import numpy as np
from connected_eye import fixture_eye,fixture_textures,trace_connected
from color_pipeline import linear_to_srgb
from audit_r24_source import sha,write
from shot_paths import add_shot_argument,from_args

ROOT=Path(__file__).resolve().parents[1]
LID=np.array([.12,.055,.035]); SCLERA=np.array([.32,.3,.27])


def sample(image,uv):
    xy=np.clip(uv*np.array(image.shape[1::-1])-.5,0,np.array(image.shape[1::-1])-1)
    lo=np.floor(xy).astype(int); hi=np.minimum(lo+1,np.array(image.shape[1::-1])-1); f=xy-lo
    return ((image[lo[...,1],lo[...,0]]*(1-f[...,0,None])+image[lo[...,1],hi[...,0]]*f[...,0,None])*(1-f[...,1,None])+
            (image[hi[...,1],lo[...,0]]*(1-f[...,0,None])+image[hi[...,1],hi[...,0]]*f[...,0,None])*f[...,1,None])


def integrate(i,n=16):
    eye=fixture_eye(i); iris,backing,old,new=fixture_textures()
    yy,xx=np.mgrid[:96,:96]; xy=np.stack([xx,yy],-1)
    sums={k:np.zeros((96,96,3)) for k in ('clean','old','new')}
    counts={k:np.zeros((96,96),int) for k in ('iris','pupil','sclera','lid','cornea','internal_wall')}
    light_only=0; aperture_max=0.; separation=0.
    offsets=(np.arange(n)+.5)/n-.5
    for dy in offsets:
        points=xy[:,:,None]+np.stack([offsets,np.full(n,dy)],-1)
        t=trace_connected(points,eye)
        iris_rgb=sample(iris,(t['iris_xy']/eye.limbus_radius+1)/2)
        pupil_rgb=sample(backing,(t['backing_xy']/eye.limbus_radius+1)/2)
        clean=(iris_rgb*t['iris'][...,None]+pupil_rgb*t['pupil'][...,None]+
               SCLERA*t['internal_wall'][...,None])*(1-t['fresnel'][...,None])
        clean+=SCLERA*t['sclera'][...,None]+LID*t['lid'][...,None]
        # The opaque diagnostic lid is itself luminous: retain its corneal reflection.
        clean+=LID*(t['cornea']&t['light_blocked'])[...,None]*t['fresnel'][...,None]
        sums['clean']+=clean.sum(2)/(n*n)
        for name,tex in [('old',old),('new',new)]:
            rgb=sample(tex,t['environment_uv'])*(t['reflection']*t['fresnel'])[...,None]
            sums[name]+=rgb.sum(2)/(n*n)
        for k in counts: counts[k]+=t[k].sum(2)
        light_only+=int((t['cornea']&t['light_blocked']).sum())
        if t['pupil'].any():
            aperture_max=max(aperture_max,float(np.linalg.norm(t['pupil_aperture_xy'][t['pupil']],axis=-1).max()))
            separation=max(separation,float(np.linalg.norm((t['backing_xy']-t['pupil_aperture_xy'])[t['pupil']],axis=-1).max()))
    classes=[counts[k] for k in ('iris','pupil','sclera','lid','internal_wall')]
    pure=np.logical_or.reduce([a==n*n for a in classes]+[sum(classes)==0])
    masks={k:counts[k]==n*n for k in ('iris','pupil','sclera','lid')}
    masks.update(full=np.ones((96,96),bool),mixed=~pure,cornea=counts['cornea']>0)
    return sums,masks,dict(light_occluded_camera_visible_subrays=light_only,
        maximum_pupil_aperture_radius=aperture_max,maximum_backing_aperture_xy_separation=separation,
        internal_wall_subrays=int(counts['internal_wall'].sum()),quadrature=n)


def metric(a,b,mask):
    if not mask.any(): return dict(pixels=0,linear_mae=None,display_mae=None,linear_max=None)
    error=np.abs(a[mask]-b[mask])
    return dict(pixels=int(mask.sum()),linear_mae=float(error.mean()),linear_max=float(error.max()),
        display_mae=float(np.abs(linear_to_srgb(a[mask])-linear_to_srgb(b[mask])).mean()*255))


def board_tile(rgb,label):
    image=np.clip(linear_to_srgb(rgb)[...,::-1]*255,0,255).astype('uint8')
    image=cv2.resize(image,(288,288),interpolation=cv2.INTER_NEAREST)
    header=np.zeros((48,288,3),np.uint8)
    for j,line in enumerate(label.split('|')): cv2.putText(header,line,(5,18+21*j),0,.44,(255,255,255),1)
    return np.concatenate([header,image],0)


def main():
    ap=argparse.ArgumentParser(description=__doc__); add_shot_argument(ap)
    ap.add_argument('--frames',type=int,nargs='+',default=list(range(24)))
    ap.add_argument('--quadrature',type=int,default=16)
    args=ap.parse_args(); shot=from_args(args)
    manifest=json.loads((shot.logs/'r24_cycles.json').read_text())
    assert manifest['protocol_sha256']==sha(ROOT/'docs/r24-eye-protocol.md')
    assert all(sha(ROOT/'scripts'/k)==v for k,v in manifest['code_sha256'].items() if k!='connected_eye.py')
    # The documented wall-intersection correction only changes the analytic tracer;
    # the rendered fixture parameters and texture arrays remain frozen and replayed by verification.
    out=shot.composite/'r24_eyes'; (out/'review').mkdir(exist_ok=True)
    records=[]
    for i in args.frames:
        sums,masks,info=integrate(i,args.quadrature)
        files=next(r['files'] for r in manifest['records'] if r['frame']==i)
        truth={}
        for k,f in files.items():
            assert sha(f['path'])==f['sha256']
            truth[k]=cv2.imread(f['path'],cv2.IMREAD_UNCHANGED)[...,::-1].astype(float)
        # The new term is computed from scene emitters/geometry, with no target read in integrate().
        coupled=truth['source']-sums['old']+sums['new']
        predictions=dict(clean=sums['clean'],source=sums['clean']+sums['old'],
            target=sums['clean']+sums['new'],coupled=coupled,add_only=truth['source']+sums['new'])
        scores={region:{k:metric(rgb,truth[k] if k in truth else truth['target'],mask)
            for k,rgb in predictions.items()} for region,mask in masks.items()}
        record=dict(frame=i,info=info,scores=scores,
            negative_coupled_channels=int((coupled<0).sum()),geometry_supplied=True,material_supplied=True,
            real_replacement_allowed=False)
        records.append(record)
        np.savez_compressed(out/f'transport_{i:06d}.npz',**sums,**{f'mask_{k}':v for k,v in masks.items()})
        tiles=[board_tile(rgb,label) for rgb,label in [
            (truth['source'],f'{i:02d} Cycles old scene|NONPERFORMER fixture'),
            (truth['clean'],'Cycles black environment|Lid reflection remains'),
            (predictions['clean'],'Analytic clean transport|SUPPLIED material'),
            (truth['target'],'Cycles new scene|Reference only'),
            (coupled,'Source - old + NEW|NEW from scene geometry'),
            (predictions['add_only'],'Add-only control|Original reflection remains')]]
        board=np.concatenate([np.concatenate(tiles[:3],1),np.concatenate(tiles[3:],1)],0)
        for repeat in range(3):
            cv2.imwrite(str(out/'review'/f'{i+repeat*24:06d}.png'),board)
        print('R24_TRANSPORT',i,json.dumps(scores['full']),flush=True)
    report=dict(records=records,code_sha256={n:sha(ROOT/'scripts'/n) for n in ('audit_r24_eyes.py','connected_eye.py')},
        protocol_sha256=sha(ROOT/'docs/r24-eye-protocol.md'),
        correction_sha256=sha(ROOT/'docs/r24-render-orientation-correction.md'),
        quadrature=args.quadrature,review_shape=list(board.shape),
        source_reflection_inferred=False,new_reflection_from_scene=True,geometry_inferred=False,
        production_repairs=0,scope='Forward transport with supplied material/geometry; no inverse acceptance test')
    suffix='' if args.frames==list(range(24)) and args.quadrature==16 else '_probe'
    write(shot.logs/f'r24_transport{suffix}.json',report)
    print('R24_TRANSPORT_COMPLETE',flush=True)


if __name__=='__main__': main()
