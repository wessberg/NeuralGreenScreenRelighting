"""Selective, artist-authorized DiffusionRenderer trial. Research output, not a repair claim."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from shot_paths import ROOT, add_shot_argument, from_args

os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
os.environ['HF_HOME']=str(ROOT/'dependencies/diffusion-runtime/hf-cache')
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import cv2
import numpy as np
from color_pipeline import linear_to_srgb, srgb_to_linear
from setup_diffusion_renderer import sha

PASSES=('basecolor','normal','depth','roughness','metallic')


def write(path, data):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def rgb8(linear):
    return np.rint(np.clip(linear_to_srgb(linear),0,1)*255).astype(np.uint8)


def save_rgb(path, rgb):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    assert cv2.imwrite(str(path),np.asarray(rgb)[...,::-1])


def eye_mask(points, topology, shape, aperture, blinks, found):
    """Supersampled visible-opening proposal, feathered strictly inside its boundary."""
    mask=np.zeros(shape,np.float32)
    if not found or not np.isfinite(points).all(): return mask
    scale=4
    for i,side in enumerate(('left','right')):
        if not np.isfinite(aperture[i]): continue
        # Older tracks have no blendshape estimates (NaN). Use their measured
        # opening alone, and report the missing independent blink signal.
        blink_weight=float(np.clip((.65-blinks[i])/.3,0,1)) if np.isfinite(blinks[i]) else 1.
        openness=float(np.clip((aperture[i]-.35)/.4,0,1)*blink_weight)
        if openness==0: continue
        polygon=points[topology[f'eyelid_{side}'],:2]
        fine=np.zeros((shape[0]*scale,shape[1]*scale),np.uint8)
        cv2.fillPoly(fine,[np.rint((polygon+.5)*scale-.5).astype(np.int32)],1)
        distance=cv2.distanceTransform(fine,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)/scale
        weight=np.clip((distance-.75)/1.25,0,1)*openness
        mask=np.maximum(mask,cv2.resize(weight,(shape[1],shape[0]),interpolation=cv2.INTER_AREA))
    return mask


def prepare(shot, root, start, count, size):
    meta=json.loads((shot.relight/'face_landmarks.json').read_text())
    track=json.loads((shot.relight/'face_track.json').read_text())
    aspect=float(meta['coordinates']['anamorphic_squeeze'])
    indices=list(range(start,start+count)); assert min(indices)>=0 and max(indices)<shot.frame_count
    with np.load(shot.relight/'face_landmarks.npz') as z: lm=dict(z)
    bounds=np.array([track['per_frame'][j]['bbox_px'] for j in indices if track['per_frame'][j]['found']])
    if not len(bounds): raise ValueError('No face track in requested interval')
    lo=bounds[:,:2].min(0); hi=bounds[:,2:].max(0); middle=(lo+hi)/2
    half=max((hi-lo)*[aspect,1])*.56
    lo=np.floor(middle-half/np.array([aspect,1])).astype(int)
    hi=np.ceil(middle+half/np.array([aspect,1])).astype(int)
    source=[]; masks=[]; rows=[]
    topology=meta['topology']; blinkids=[meta['blendshape_order'].index(f'eyeBlink{s}') for s in ('Left','Right')]
    for i in indices:
        p=shot.fg(i); raw=cv2.imread(str(p),cv2.IMREAD_UNCHANGED)
        if raw is None: raise FileNotFoundError(p)
        raw=raw[...,::-1]
        if (lo<0).any() or (hi>raw.shape[1::-1]).any(): raise ValueError('Face crop leaves source frame')
        cropped=raw[lo[1]:hi[1],lo[0]:hi[0]].copy()
        points=lm['points'][i].copy(); points[:,:2]-=lo
        weights=eye_mask(points,topology,cropped.shape[:2],lm['eyelid_aperture'][i],lm['blendshapes'][i,blinkids],lm['found'][i])
        # Physical square crop; this resize desqueezes if the metadata requires it.
        display=cv2.resize(rgb8(cropped),(size,size),interpolation=cv2.INTER_CUBIC)
        save_rgb(root/'input'/f'{i:06d}.png',display)
        source.append(cropped); masks.append(weights)
        rows.append(dict(frame=i,source_frame=shot.source_start_frame+i,source_sha256=sha(p),
            model_input_sha256=sha(root/'input'/f'{i:06d}.png'),
            mask_pixels=int((weights>0).sum()),mask_weight_sum=float(weights.sum()),
            aperture=[float(v) if np.isfinite(v) else None for v in lm['eyelid_aperture'][i]],
            blink=[float(v) if np.isfinite(v) else None for v in lm['blendshapes'][i,blinkids]]))
    source=np.stack(source); masks=np.stack(masks)
    np.savez_compressed(root/'source.npz',linear=source,masks=masks)
    record=dict(rows=rows,crop_xyxy=[*lo.tolist(),*hi.tolist()],pixel_aspect=aspect,
        inference_size=size,frame_rate=str(shot.frame_rate),selected_count=count,
        native_crop_shape=list(source.shape[1:]),input_transfer='display-sRGB',
        source_transfer='scene-linear Rec.709',model_fps_condition=7,
        model_fps_condition_is_not_review_retime=True,
        source_clipped_for_model_channels=int((linear_to_srgb(source)>1).sum()),
        code_sha256=sha(__file__),track_sha256=sha(shot.relight/'face_landmarks.npz'),
        shot_sha256=sha(shot.dir/'shot.json'),generative_region_authorized=True,
        mask_is_unvalidated_landmark_proposal=True)
    write(root/'input.json',record); write(shot.logs/f'{root.name}_input.json',record)
    print('R31_PREPARED',count,source.shape,size,flush=True)


def load_pipeline(kind):
    import torch
    from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
    sys.path.insert(0,str(ROOT/'dependencies/DiffusionRenderer'))
    from src.models.custom_unet_st import UNetCustomSpatioTemporalConditionModel
    from src.models.env_encoder import EnvEncoder
    from src.pipelines.pipeline_rgbx import RGBXVideoDiffusionPipeline
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    from diffusers.models.modeling_outputs import AutoencoderKLOutput
    class LocalRGBXPipeline(RGBXVideoDiffusionPipeline):
        # The released optional encoder is absent from diffusers' component
        # discovery, so its device/offload hooks would otherwise be omitted.
        _optional_components=[*RGBXVideoDiffusionPipeline._optional_components,'env_encoder']
    folder=ROOT/'dependencies/diffusion-weights'/kind
    options=dict(torch_dtype=torch.float16,local_files_only=True,low_cpu_mem_usage=True)
    vae=AutoencoderKLTemporalDecoder.from_pretrained(folder,subfolder='vae',**options)
    unet=UNetCustomSpatioTemporalConditionModel.from_pretrained(folder,subfolder='unet',**options)
    env=EnvEncoder.from_pretrained(folder,subfolder='env_encoder',**options) if kind=='forward' else None
    scheduler=EulerDiscreteScheduler.from_pretrained(folder,subfolder='scheduler',local_files_only=True)
    pipe=LocalRGBXPipeline(vae=vae,unet=unet,scheduler=scheduler,image_encoder=None,
        feature_extractor=None,env_encoder=env,cond_mode='env' if env else 'skip',scale_cond_latents=kind=='forward')
    if kind=='forward':
        pipe.scheduler.register_to_config(timestep_spacing='trailing')
        # Do not silently omit failed LoRA as the demonstration script does.
        pipe.load_lora_weights(str(folder),subfolder='lora',adapter_name='real-lora',
                               weight_name='pytorch_lora_weights.safetensors',local_files_only=True)
    # Frame-independent VAE encoder slicing preserves the same operation. The
    # decoder remains temporal, using a reported 4-frame chunk size.
    original_encode=vae.encode
    def sliced_encode(x,return_dict=True):
        parts=[original_encode(chunk).latent_dist.parameters for chunk in x.split(2)]
        distribution=DiagonalGaussianDistribution(torch.cat(parts,0))
        return AutoencoderKLOutput(latent_dist=distribution) if return_dict else (distribution,)
    vae.encode=sliced_encode
    pipe.unet.enable_forward_chunking(chunk_size=1,dim=0)
    pipe.model_cpu_offload_seq='env_encoder->unet->vae'
    pipe.enable_model_cpu_offload()
    if env is not None: assert hasattr(env,'_hf_hook'), 'Environment encoder must participate in offloading'
    pipe.set_progress_bar_config(disable=False)
    return pipe


def infer(shot,root,kind,steps,seed):
    import torch
    torch.set_num_threads(8)
    meta=json.loads((root/'input.json').read_text()); size=meta['inference_size']; count=len(meta['rows'])
    pipe=load_pipeline(kind)
    timings=[]
    if kind=='inverse':
        source=np.stack([cv2.imread(str(root/'input'/f'{r["frame"]:06d}.png'))[...,::-1] for r in meta['rows']])[None]/255.
        runs=[(p,dict(rgb=source.astype(np.float32),input_context=p),{'rgb':'vae'},1.) for p in PASSES]
    else:
        cond={p:np.stack([cv2.imread(str(root/'inverse'/p/f'{r["frame"]:06d}.png'))[...,::-1]
                         for r in meta['rows']])[None].astype(np.float32)/255 for p in PASSES}
        mapping={p:'vae' for p in PASSES}; mapping.update({p:'env' for p in ('env_ldr','env_log','env_nrm')})
        runs=[]
        with np.load(root/'environments.npz') as envs:
            for name in ('neutral','scene'):
                values=dict(cond)
                for k in ('env_ldr','env_log','env_nrm'):
                    values[k]=np.broadcast_to(envs[f'{name}_{k}'],(1,count,*envs[f'{name}_{k}'].shape)).copy()
                runs.append((name,values,mapping,1.2))
    for label,cond,mapping,guidance in runs:
        output=root/kind/label; stamp=output/'complete.json'
        signature=hashlib.sha256()
        for key,value in sorted(cond.items()):
            signature.update(key.encode())
            signature.update(value.encode() if isinstance(value,str) else np.ascontiguousarray(value).tobytes())
        conditioning_sha256=signature.hexdigest()
        if stamp.exists():
            prior=json.loads(stamp.read_text())
            assert prior['steps']==steps and prior['seed']==seed and prior['size']==size
            assert prior['conditioning_sha256']==conditioning_sha256 and prior['code_sha256']==sha(__file__)
            assert all(sha(output/p)==h for p,h in prior['image_sha256'].items())
            timings.append(prior); continue
        started=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
        print('R31_INFERENCE',kind,label,'frames',count,'size',size,'steps',steps,flush=True)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            frames=pipe(cond,mapping,height=size,width=size,num_frames=count,num_inference_steps=steps,
                min_guidance_scale=guidance,max_guidance_scale=guidance,fps=7,motion_bucket_id=127,
                noise_aug_strength=0,generator=torch.Generator(device='cuda').manual_seed(seed),
                decode_chunk_size=4,**({'cross_attention_kwargs':{'scale':.25}} if kind=='forward' else {})).frames[0]
        assert len(frames)==count
        hashes={}
        for row,frame in zip(meta['rows'],frames):
            rgb=np.asarray(frame); assert rgb.shape==(size,size,3)
            path=output/f'{row["frame"]:06d}.png'; save_rgb(path,rgb); hashes[path.name]=sha(path)
        record=dict(kind=kind,label=label,frames=count,size=size,steps=steps,seed=seed,
            elapsed_seconds=time.perf_counter()-started,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),image_sha256=hashes,
            vae_encode_chunk=2,vae_decode_chunk=4,unet_forward_chunk=1,cpu_offload=True,
            torch=torch.__version__,code_sha256=sha(__file__),conditioning_sha256=conditioning_sha256,
            model_revision=json.loads((ROOT/'dependencies/diffusion-weights'/f'{kind}_pin.json').read_text())['revision'])
        write(stamp,record); timings.append(record)
        print('R31_INFERRED',label,record['elapsed_seconds'],record['peak_cuda_allocated_bytes'],flush=True)
    write(shot.logs/f'{root.name}_{kind}.json',timings)


def environment_vectors(size):
    # Author rendering_utils.envmap_vec's exact sample convention, CPU equivalent.
    theta=np.linspace(1/size,1-1/size,size)*np.pi
    phi=np.linspace(-1+1/size,1-1/size,size)*np.pi
    tt,pp=np.meshgrid(theta,phi,indexing='ij')
    v=np.stack([np.sin(tt)*np.sin(pp),np.cos(tt),-np.sin(tt)*np.cos(pp)],-1)
    return -v[::-1,::-1].copy()


def environments(shot,root):
    size=512; v=environment_vectors(size)
    info=json.loads((shot.environment/'scene_info.json').read_text())
    path=ROOT/'assets/hdri'/info['hdri']['lighting']; hdr=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if hdr is None: raise FileNotFoundError(path)
    if hdr.ndim!=3 or hdr.shape[2] not in (3,4): raise ValueError('Expected RGB or RGBA HDRI')
    # EXR assets may carry alpha; it is not a fourth illumination channel.
    hdr=np.maximum(np.nan_to_num(hdr[...,:3][...,::-1],nan=0.,posinf=65504.,neginf=0.),0)
    hdr*=float(info['hdri']['strength'])
    # Match author's default fixed-camera projected layout directly in latlong;
    # this avoids an extra 512px cubemap resampling and nvdiffrast dependency.
    # Actual Blender-to-model world registration is not certified by this layout.
    rotation=180.+float(info['hdri'].get('rotation_deg',0))
    hdr=np.roll(hdr,int(hdr.shape[1]*rotation/360),axis=1)
    u=np.arctan2(v[...,0],-v[...,2])/(2*np.pi)+.5
    w=np.arccos(np.clip(v[...,1],-1,1))/np.pi
    scene=cv2.remap(hdr,(u*hdr.shape[1]-.5).astype(np.float32),(w*hdr.shape[0]-.5).astype(np.float32),
                    cv2.INTER_LINEAR,borderMode=cv2.BORDER_WRAP)
    maps={}
    for name,values in [('neutral',np.full_like(scene,.1)),('scene',scene)]:
        mapped=values*(1+values/256)/(1+values)
        maps[f'{name}_env_ldr']=np.clip(linear_to_srgb(np.clip(mapped,0,1)),0,1).astype(np.float32)
        maps[f'{name}_env_log']=np.clip(linear_to_srgb(np.log1p(values)/np.log1p(10000)),0,1).astype(np.float32)
        maps[f'{name}_env_nrm']=(v*.5+.5).astype(np.float32)
        save_rgb(root/f'{name}_environment.png',np.rint(maps[f'{name}_env_ldr']*255).astype(np.uint8))
    np.savez_compressed(root/'environments.npz',**maps)
    record=dict(scene=str(path.relative_to(ROOT)),sha256=sha(path),
        strength=info['hdri']['strength'],author_layout_rotation_degrees=rotation,
        neutral_radiance=.1,source='shot environment HDRI',world_registration_validated=False,
        direct_latlong_sampling=True,finite_scene_geometry_reflections=False)
    write(root/'environments.json',record); write(shot.logs/f'{root.name}_environments.json',record)
    print('R31_ENVIRONMENTS',flush=True)


def review(shot,root):
    meta=json.loads((root/'input.json').read_text())
    with np.load(root/'source.npz') as z: source,masks=z['linear'],z['masks']
    output=root/'review'; output.mkdir(exist_ok=True)
    slow_output=root/'review_slow'; slow_output.mkdir(exist_ok=True)
    stats=[]; composites={name:[] for name in ('neutral','scene')}
    support=np.any(masks>0,axis=0); yy,xx=np.nonzero(support)
    eyelo=np.maximum([xx.min()-15,yy.min()-20],0); eyehi=np.minimum([xx.max()+16,yy.max()+21],source.shape[2:0:-1])
    for j,row in enumerate(meta['rows']):
        raw=source[j]; mask=masks[j]; generated={}; blended={}
        for name in ('neutral','scene'):
            p=root/'forward'/name/f'{row["frame"]:06d}.png'
            im=cv2.imread(str(p))
            if im is None: raise FileNotFoundError(p)
            linear=srgb_to_linear(im[...,::-1].astype(np.float32)/255)
            linear=cv2.resize(linear,(raw.shape[1],raw.shape[0]),interpolation=cv2.INTER_LINEAR)
            blend=raw.copy(); inside=mask>0
            blend[inside]=raw[inside]*(1-mask[inside,None])+linear[inside]*mask[inside,None]
            assert np.array_equal(blend[~inside],raw[~inside])
            generated[name]=linear; blended[name]=blend; composites[name].append(blend)
        panels=[]
        for title,im in [('SOURCE',raw),('NEUTRAL: eyes only',blended['neutral']),
                         ('SCENE: eyes only',blended['scene']),('SCENE: full generated crop',generated['scene'])]:
            header=np.zeros((50,320,3),np.uint8)
            cv2.putText(header,title,(6,19),0,.42,(235,235,235),1,cv2.LINE_AA)
            cv2.putText(header,f'Source {row["source_frame"]} | Generative trial',(6,39),0,.36,(235,235,235),1,cv2.LINE_AA)
            face=cv2.resize(rgb8(im)[...,::-1],(320,320),interpolation=cv2.INTER_LINEAR)
            crop=rgb8(im[eyelo[1]:eyehi[1],eyelo[0]:eyehi[0]])[...,::-1]
            scale=min(320/crop.shape[1],180/crop.shape[0]); dim=(round(crop.shape[1]*scale),round(crop.shape[0]*scale))
            eye=np.zeros((180,320,3),np.uint8); tile=cv2.resize(crop,dim,interpolation=cv2.INTER_NEAREST)
            y=(180-dim[1])//2; x=(320-dim[0])//2; eye[y:y+dim[1],x:x+dim[0]]=tile
            panels.append(np.concatenate([header,face,eye]))
        board=np.concatenate(panels,axis=1); footer=np.zeros((50,1280,3),np.uint8)
        cv2.putText(footer,'Selective generation authorized. Unvalidated masks and model output; gaze, blinks, seams and reflections must be judged.',(8,19),0,.48,(230,230,230),1,cv2.LINE_AA)
        cv2.putText(footer,'Neutral = uniform illumination. Scene = shot HDRI; world registration unvalidated. Full generated crop is diagnostic only.',(8,39),0,.48,(230,230,230),1,cv2.LINE_AA)
        assert cv2.imwrite(str(output/f'{j:06d}.png'),np.concatenate([board,footer]))
        slow_footer=footer.copy(); slow_footer[26:]=0
        cv2.putText(slow_footer,f'SLOW MOTION: 12 fps playback; source {float(shot.frame_rate):.2f} fps. No frame interpolation. Scene registration unvalidated.',
                    (8,39),0,.48,(230,230,230),1,cv2.LINE_AA)
        assert cv2.imwrite(str(slow_output/f'{j:06d}.png'),np.concatenate([board,slow_footer]))
        stats.append(dict(frame=row['frame'],mask_pixels=int(inside.sum()),outside_mask_exact=True,
            mean_eye_display_change={name:float(np.abs(rgb8(im).astype(float)-rgb8(raw)).mean(2)[inside].mean()) if inside.any() else 0
                                     for name,im in blended.items()}))
    np.savez_compressed(root/'selective_composites.npz',**{k:np.stack(v) for k,v in composites.items()})
    write(shot.logs/f'{root.name}_review.json',dict(frames=stats,source_fps=str(shot.frame_rate),resolution=[1280,600],
        source_pixels_outside_masks_preserved=True,artist_accepted=False,production_enabled=False))
    print('R31_REVIEW_READY',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__); add_shot_argument(ap)
    ap.add_argument('--stage',choices=['prepare','environments','inverse','forward','review'],required=True)
    ap.add_argument('--start',type=int,default=0); ap.add_argument('--count',type=int,default=24)
    ap.add_argument('--size',type=int,default=384); ap.add_argument('--steps',type=int,default=20)
    ap.add_argument('--seed',type=int,default=31); ap.add_argument('--name',default='r31_diffusion')
    args=ap.parse_args(); shot=from_args(args)
    if Path(args.name).name!=args.name: raise ValueError('Trial name must be a single directory name')
    root=shot.composite/args.name; root.mkdir(parents=True,exist_ok=True)
    if args.stage=='prepare': prepare(shot,root,args.start,args.count,args.size)
    elif args.stage=='environments': environments(shot,root)
    elif args.stage in ('inverse','forward'): infer(shot,root,args.stage,args.steps,args.seed)
    else: review(shot,root)


if __name__=='__main__': main()
