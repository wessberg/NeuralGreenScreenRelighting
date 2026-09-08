"""Photographic relighting via temporal geometry, Blender area lights and flow-stable gain.

No generative color or texture synthesis. A smooth RGB gain multiplies the source
plate in linear light; source image pixels are never warped for relighting.
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
from pathlib import Path
import json, argparse
import numpy as np, cv2
from color_pipeline import LUMA709
ROOT=Path(__file__).resolve().parents[1]
W,H=768,432
cv2.setNumThreads(8)

def blur(x,sigma):return cv2.GaussianBlur(x,(0,0),sigma)

def load_alpha(i):
    f=ROOT/f'key/alpha/{i:06d}.exr'
    if f.exists():a=cv2.imread(str(f),-1)
    else:
        rgb=cv2.imread(str(ROOT/f'source/frames/{i:06d}.png'))[...,::-1]/255
        a=1-((rgb[...,1]>.68)&(rgb[...,1]-np.maximum(rgb[...,0],rgb[...,2])>.48)).astype(np.float32)
    return cv2.resize(a,(W,H),interpolation=cv2.INTER_AREA)

def make_geometry(depth,a,calibration,camera):
    # Foreground-only normalized convolution prevents a false normal discontinuity at the matte.
    mask=(a>.6).astype(np.float32)
    d=blur(depth*mask,3)/np.maximum(blur(mask,3),1e-6)
    relative=np.clip((d-calibration['median'])/calibration['spread'],-1.8,1.8)
    Y=-.20*relative
    fullwidth=camera['actor_distance_m']*camera['sensor_width_mm']/camera['lens_mm']
    pix=fullwidth/W
    yy,xx=np.indices((H,W),dtype=np.float32)
    X=(xx+.5-W/2)*pix;Z=camera['height_m']+(H/2-yy-.5)*pix
    gy,gx=np.gradient(Y,pix)
    # A bounded normal field prevents uncertain single-view depth from sculpting faces.
    nx=np.clip(gx,-.90,.90);nz=np.clip(-gy,-.75,.75)
    # Soft convex silhouette contributes geometric rim response, never a generic screen-space halo.
    distance=cv2.distanceTransform((a>.5).astype(np.uint8),cv2.DIST_L2,5)
    dy,dx=np.gradient(blur(distance,3))
    norm=np.sqrt(dx*dx+dy*dy)+1e-6
    edge=np.exp(-distance/14)*mask
    nx=nx*(1-.50*edge)-.55*dx/norm*edge
    nz=nz*(1-.45*edge)+.55*dy/norm*edge
    normals=np.stack([nx,-np.ones_like(nx),nz],-1)
    normals/=np.linalg.norm(normals,axis=-1,keepdims=True)
    positions=np.stack([X,Y,Z],-1)
    return positions,normals,d

def lighting_gain(pos,normals,scene):
    diffuse=np.broadcast_to(np.array(scene['ambient_rgb'],np.float32),(H,W,3)).copy()
    front=np.broadcast_to(np.array([0,-1,0],np.float32),(H,W,3))
    diffuse_ref=diffuse.copy()
    spec=np.zeros_like(diffuse)
    selected={'Cool window key','Camera soft fill','Warm practical soft bounce','Warm practical bulb','Window ambience','Ceiling bounced fill'}
    # Light power/distance and lamp orientation come from the actual saved Blender scene.
    for light in scene['lights']:
        if light['name'] not in selected:continue
        delta=np.asarray(light['position'],np.float32)-pos
        dist2=np.sum(delta*delta,-1)
        ld=delta/np.sqrt(dist2)[...,None]
        cosine=np.sum(normals*ld,-1)
        refcos=np.sum(front*ld,-1)
        if light['type']=='AREA':
            facing=np.maximum(np.sum(-ld*np.asarray(light['direction'],np.float32),-1),0)
            radius=light['size']*.35
            energy=light['power_w']*facing/(np.pi*(dist2+radius*radius))
            wrap=min(.38,radius/3)
        else:
            energy=light['power_w']/(4*np.pi*(dist2+.02));wrap=.10
        rgb=np.array(light['color'],np.float32)
        # Calibrated exposure factor: conservative fill preserves photographed source shading.
        strength=energy*.13
        diffuse+=((np.maximum(cosine+wrap,0)/(1+wrap))*strength)[...,None]*rgb
        diffuse_ref+=((np.maximum(refcos+wrap,0)/(1+wrap))*strength)[...,None]*rgb
        half=ld+front;half/=np.maximum(np.linalg.norm(half,axis=-1,keepdims=True),1e-6)
        # Broad and very restrained dielectric response is exported separately.
        sp=np.maximum(np.sum(normals*half,-1),0)**22
        spec+=(sp*strength*.008)[...,None]*rgb
    # Ratio against an estimated soft neutral source key. Retain baked fine/mid-frequency shading.
    source_dir=np.array([-.28,-.92,.28],np.float32);source_dir/=np.linalg.norm(source_dir)
    source=.66+.34*np.maximum(np.sum(normals*source_dir,-1),0)
    gain=diffuse/source[...,None]
    # Fixed calibration, not per-frame exposure: prevents pumping with head/body movement.
    gain=np.clip(gain,.35,1.50)
    gain=.30+.70*gain
    return gain.astype(np.float32),spec.astype(np.float32)

def main():
    p=argparse.ArgumentParser();p.add_argument('--test',action='store_true');args=p.parse_args()
    scene=json.loads((ROOT/'blender/lighting.json').read_text())
    n=44 if args.test else 300
    vals=[]
    for i in range(0,n,10):
        d=np.load(ROOT/f'relight/depth/{i:06d}.npy');a=load_alpha(i);vals.append(d[a>.99][::16])
    allvals=np.concatenate(vals);q=np.percentile(allvals,[10,50,90])
    cal={'median':float(q[1]),'spread':float(max(q[2]-q[0],1)), 'relief_m':.20}
    (ROOT/'relight/calibration.json').write_text(json.dumps(cal,indent=2))
    out=ROOT/'relight/controls';out.mkdir(exist_ok=True)
    flow=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    prev=None;prevgray=None;stats=[]
    yy,xx=np.indices((H,W),dtype=np.float32)
    for i in range(n):
        a=load_alpha(i);d=np.load(ROOT/f'relight/depth/{i:06d}.npy')
        pos,normals,smoothed=make_geometry(d,a,cal,scene['camera'])
        gain,spec=lighting_gain(pos,normals,scene)
        gain=blur(gain,2.2)
        src=cv2.imread(str(ROOT/f'source/frames/{i:06d}.png'))
        gray=cv2.cvtColor(cv2.resize(src,(W,H),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY)
        raw=gain.copy()
        if prev is not None:
            backward=flow.calc(gray,prevgray,None)
            forward=flow.calc(prevgray,gray,None)
            mx=xx+backward[...,0];my=yy+backward[...,1]
            warped=cv2.remap(prev,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
            fw=cv2.remap(forward,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
            err=np.linalg.norm(backward+fw,axis=-1)
            confidence=np.exp(-err*err/2)*.65*(a>.1)
            difference=np.max(np.abs(raw-warped),axis=-1)
            confidence*=np.exp(-difference*difference/.055)
            gain=raw*(1-confidence[...,None])+warped*confidence[...,None]
        prev=gain.copy();prevgray=gray
        np.savez_compressed(out/f'{i:06d}.npz',gain=gain.astype(np.float16),specular=spec.astype(np.float16))
        if i in [0,15,30,60,150,240,299]:
            cv2.imwrite(str(ROOT/f'relight/normals_{i:06d}.jpg'),((normals[...,::-1]*.5+.5)*255).astype(np.uint8))
            cv2.imwrite(str(ROOT/f'relight/gain_{i:06d}.jpg'),np.clip(gain[...,::-1]/1.5*255,0,255).astype(np.uint8))
        stats.append({'frame':i,'mean_gain':np.mean(gain[a>.99],axis=0).tolist(),'raw_to_stable_mae':float(np.abs(gain-raw)[a>.99].mean())})
        if i%30==0:print(json.dumps(stats[-1]),flush=True)
    (ROOT/'logs/relight_controls.json').write_text(json.dumps(stats,indent=2))

if __name__=='__main__':main()
