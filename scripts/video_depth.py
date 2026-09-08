"""Video Depth Anything Small, actual temporal model; CUDA SDPA attention adaptation.

Shot-agnostic: pass --shot to run any shot; omit it for the proof-of-concept example at the
project root. Frame count, frame rate and screen thresholds come from the shot's shot.json.
The temporal model is given the shot's real frame rate, not an assumed 30.
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
from pathlib import Path
import sys, json, argparse, time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'dependencies/VideoDepthAnything'))
sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np, cv2, torch
WORK = None   # working resolution, set per shot from the plate aspect
import torch.nn.functional as F
from video_depth_anything.video_depth import VideoDepthAnything
from video_depth_anything.dinov2_layers.attention import Attention, MemEffAttention
from shot_paths import add_shot_argument, from_args

def sdpa(self,x,attn_bias=None):
    B,N,C=x.shape
    q,k,v=self.qkv(x).reshape(B,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4).unbind(0)
    y=F.scaled_dot_product_attention(q,k,v,dropout_p=0).transpose(1,2).reshape(B,N,C)
    return self.proj_drop(self.proj(y))

def main():
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--test',action='store_true')
    add_shot_argument(p)
    a=p.parse_args()
    shot=from_args(a)
    global WORK
    # Depth must be produced at the SAME working resolution its consumers read, and that is
    # now derived from the plate aspect. A hardcoded 768x432 meant a non-16:9 shot raised a
    # broadcasting error against differently shaped masks, or silently stretched. Found by
    # the gpt-6-astra review.
    WORK = shot.working_resolution
    torch.set_num_threads(8);cv2.setNumThreads(8)
    # Same scaled dot-product calculation, using CUDA kernels supporting RTX 5080.
    Attention.forward=sdpa;MemEffAttention.forward=sdpa
    model=VideoDepthAnything(encoder='vits',features=64,out_channels=[48,96,192,384])
    model.load_state_dict(torch.load(ROOT/'dependencies/VideoDepthAnything/checkpoints/video_depth_anything_vits.pth',map_location='cpu',weights_only=True),strict=True)
    model=model.cuda().eval()
    hint=shot.key_hint
    total=shot.frame_count
    # The temporal window is 32 with 10 overlap, so a test needs more than one window.
    n=min(44,total) if a.test else total
    fps=float(shot.frame_rate)
    print(json.dumps({'shot':shot.name,'frames':n,'of':total,'source_fps':str(shot.frame_rate)}),flush=True)
    frames=[]
    for i in range(n):
        f=cv2.imread(str(shot.frame(i)))
        if f is None:raise SystemExit(f'Missing source frame {shot.frame(i)}')
        # Neutral backing eliminates saturated screen bias in geometric inference.
        rgb=f[...,::-1].astype(np.float32)/255
        bg=(rgb[...,1]>hint['green_min'])&((rgb[...,1]-np.maximum(rgb[...,0],rgb[...,2]))>hint['green_dominance_min'])
        rgb[bg]=.20
        frames.append(cv2.resize((rgb*255).astype(np.uint8),WORK,interpolation=cv2.INTER_AREA))
    start=time.time()
    depths,out_fps=model.infer_video_depth(np.stack(frames),fps,input_size=392,device='cuda',fp32=False)
    shot.depth.mkdir(parents=True,exist_ok=True)
    for i,d in enumerate(depths):np.save(shot.depth_npy(i),d.astype(np.float32))
    shot.logs.mkdir(parents=True,exist_ok=True)
    shot.log('depth.json').write_text(json.dumps({'shot':shot.name,'model':'Video Depth Anything Small','frames':n,'source_fps':str(shot.frame_rate),'input_short_side':392,'output':list(WORK),'seconds':time.time()-start,'temporal_window':32,'overlap':10,'aligned_blend_frames':8,'max_vram_GB':torch.cuda.max_memory_allocated()/1e9},indent=2))
    print(shot.log('depth.json').read_text(),flush=True)

if __name__=='__main__':main()
