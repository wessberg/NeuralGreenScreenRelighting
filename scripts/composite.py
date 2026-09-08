"""Linear photographic composition, GPU HDR conversion, half EXR and 4K Main10 NVENC.

Encoded with -bf 0. The HDR10 SEI is inserted into the raw elementary stream (scripts/hdr_mux.py),
and ffmpeg must then re-derive presentation timestamps from POC when muxing that raw stream. With
B-frames that derivation produced 3 duplicate PTS and 2-3 frame gaps, so 2 frames fell behind an
edit list. Without reordering, PTS == DTS == frame index and the master is exactly CFR.
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
from pathlib import Path
import argparse, json, subprocess, time
import numpy as np, cv2, torch
import torch.nn.functional as F
from color_pipeline import sdr_preview,REC709_TO_2020,LUMA2020
ROOT=Path(__file__).resolve().parents[1]
FFMPEG=next((ROOT/'dependencies/ffmpeg').rglob('ffmpeg.exe'))
EXR_FLAGS=[cv2.IMWRITE_EXR_TYPE,cv2.IMWRITE_EXR_TYPE_HALF,cv2.IMWRITE_EXR_COMPRESSION,cv2.IMWRITE_EXR_COMPRESSION_ZIP]
cv2.setNumThreads(8);torch.set_num_threads(8)

def load_exr(p):
    v=cv2.imread(str(p),cv2.IMREAD_UNCHANGED)
    if v is None:raise ValueError(f'Cannot read {p}')
    return v

def compose_frame(i,bg,size,relight=True):
    fg=load_exr(ROOT/f'key/fg/{i:06d}.exr')[...,::-1].copy()
    alpha=np.clip(load_exr(ROOT/f'key/alpha/{i:06d}.exr'),0,1)
    control=np.load(ROOT/f'relight/controls/{i:06d}.npz')
    gain=cv2.resize(control['gain'].astype(np.float32),fg.shape[1::-1],interpolation=cv2.INTER_LINEAR)
    spec=cv2.resize(control['specular'].astype(np.float32),fg.shape[1::-1],interpolation=cv2.INTER_LINEAR)
    # Shading-only color operations. Nothing warps or synthesizes photographic texture.
    if relight:
        lit=fg*gain
        # Broad low-energy highlight mostly on skin/wood, diminished on green cloth and dark hair.
        skin=np.clip((fg[...,0]-fg[...,1])/.10+.30,0,1)*np.clip(fg.max(-1)/.25,0,1)
        lit+=spec*skin[...,None]*.45
    else:lit=fg
    # Coverage-weighted resizing prevents RGB from transparent pixels bleeding into the outline.
    premul=lit*alpha[...,None]
    if fg.shape[1::-1]!=size:
        premul=cv2.resize(premul,size,interpolation=cv2.INTER_LANCZOS4)
        alpha=cv2.resize(alpha,size,interpolation=cv2.INTER_LANCZOS4)
        alpha=np.clip(alpha,0,1)
    # Native Cycles proxy shadow/indirect transport is already in the selected background.
    comp=np.maximum(premul,0)+bg*(1-alpha[...,None])
    return comp.astype(np.float32),alpha

def encode_gpu_frame(t):
    # CUDA float scene-linear Rec709 -> 1000-nit display grade, Rec2020, PQ.
    nits=torch.clamp(t,min=0)*200
    peak=nits.amax(dim=-1,keepdim=True)
    shoulder=torch.where(peak<=200,peak,200+800*(-torch.expm1(-torch.clamp(peak-200,min=0)/800)))
    nits=nits*shoulder/torch.clamp(peak,min=1e-8)
    mat=torch.as_tensor(REC709_TO_2020,device='cuda')
    wide=nits@mat.T
    luminance=wide@torch.as_tensor(LUMA2020,device='cuda')
    maxcll=float(luminance.max());fall=float(luminance.mean())
    p=(torch.clamp(wide,min=0)/10000)**(2610/16384)
    pq=((3424/4096+(2413/128)*p)/(1+(2392/128)*p))**(2523/32)
    return pq,maxcll,fall

def main():
    p=argparse.ArgumentParser();p.add_argument('--test',action='store_true');p.add_argument('--final',action='store_true');p.add_argument('--frames',type=int,default=300)
    # Re-encode from the saved scene-linear masters without recompositing, for encoder changes only.
    p.add_argument('--encode-only',action='store_true');args=p.parse_args()
    size=(3840,2160) if args.final else (1920,1080)
    path=ROOT/('blender/background_shadow.exr' if args.final else 'blender/test_background_shadow.exr')
    bg=load_exr(path)[...,:3][...,::-1].copy()
    bg=cv2.resize(bg,size,interpolation=cv2.INTER_LANCZOS4)
    if not args.final:
        ids=[0,15,30] if args.test else [0,60,150,240,299]
        for i in ids:
            comp,a=compose_frame(i,bg,size)
            cv2.imwrite(str(ROOT/f'composite/test_{i:06d}.jpg'),(sdr_preview(comp)[...,::-1]*255).astype(np.uint8))
            plain,_=compose_frame(i,bg,size,False)
            side=np.hstack([sdr_preview(plain),sdr_preview(comp)])
            cv2.imwrite(str(ROOT/f'composite/comparison_{i:06d}.jpg'),(cv2.resize(side,(1920,540))[...,::-1]*255).astype(np.uint8))
            print('COMPOSITE_TEST',i,'scene max',float(comp.max()),'alpha',float(a.mean()),flush=True)
        return
    master=ROOT/'intermediate/linear_4k';master.mkdir(exist_ok=True)
    log=open(ROOT/'logs/nvenc_encode.log','w',encoding='utf-8')
    cmd=[str(FFMPEG),'-hide_banner','-y','-f','rawvideo','-pixel_format','rgb48le','-video_size','3840x2160','-framerate','30','-i','pipe:0',
         '-vf','scale=in_range=full:out_range=limited:out_color_matrix=bt2020,format=p010le',
         '-an','-c:v','hevc_nvenc','-profile:v','main10','-preset','p7','-tune','hq','-rc','vbr','-cq','16','-b:v','0','-multipass','fullres','-spatial-aq','1','-temporal-aq','1','-aq-strength','6','-rc-lookahead','32','-bf','0',
         '-color_range','tv','-colorspace','bt2020nc','-color_primaries','bt2020','-color_trc','smpte2084','-r','30','-f','hevc',str(ROOT/'intermediate/encoded_pq.hevc')]
    (ROOT/'logs/encode_command.json').write_text(json.dumps(cmd,indent=2))
    encoder=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=log,stderr=log)
    stats=[];start=time.time()
    try:
        for i in range(args.frames):
            if args.encode_only:
                comp=load_exr(master/f'{i:06d}.exr')[...,:3][...,::-1].copy()
            else:
                comp,alpha=compose_frame(i,bg,size)
                cv2.imwrite(str(master/f'{i:06d}.exr'),comp[...,::-1],EXR_FLAGS)
            pq,cll,fall=encode_gpu_frame(torch.from_numpy(comp).cuda())
            # 16-bit full-range PQ RGB handed to an explicit limited-range 10-bit YUV conversion.
            raw=torch.clamp(pq*65535+.5,0,65535).to(torch.int32).cpu().numpy().astype('<u2')
            encoder.stdin.write(raw.tobytes())
            stats.append({'frame':i,'max_cll_nits':cll,'frame_average_nits':fall})
            if i in [0,30,60,90,150,210,240,270,299]:
                cv2.imwrite(str(ROOT/f'composite/final_reference_{i:06d}.jpg'),(sdr_preview(comp)[...,::-1]*255).astype(np.uint8))
            if i%10==0:print('FINAL_FRAME',i,'seconds',round(time.time()-start,1),'max_nits',round(cll,1),'fall',round(fall,1),flush=True)
        encoder.stdin.close();rc=encoder.wait()
        if rc:raise RuntimeError(f'NVENC failed: {rc}')
    finally:
        log.close()
    report={'frames':stats,'max_cll_nits':max(x['max_cll_nits'] for x in stats),'max_fall_nits':max(x['frame_average_nits'] for x in stats),'seconds':time.time()-start,'encoder':'hevc_nvenc Main10 P7 CQ16'}
    (ROOT/'logs/hdr_luminance.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
