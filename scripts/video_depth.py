"""Video Depth Anything Small, actual temporal model; CUDA SDPA attention adaptation."""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
from pathlib import Path
import sys, json, argparse, time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'dependencies/VideoDepthAnything'))
import numpy as np, cv2, torch
import torch.nn.functional as F
from video_depth_anything.video_depth import VideoDepthAnything
from video_depth_anything.dinov2_layers.attention import Attention, MemEffAttention

def sdpa(self,x,attn_bias=None):
    B,N,C=x.shape
    q,k,v=self.qkv(x).reshape(B,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4).unbind(0)
    y=F.scaled_dot_product_attention(q,k,v,dropout_p=0).transpose(1,2).reshape(B,N,C)
    return self.proj_drop(self.proj(y))

def main():
    p=argparse.ArgumentParser();p.add_argument('--test',action='store_true');a=p.parse_args()
    torch.set_num_threads(8);cv2.setNumThreads(8)
    # Same scaled dot-product calculation, using CUDA kernels supporting RTX 5080.
    Attention.forward=sdpa;MemEffAttention.forward=sdpa
    model=VideoDepthAnything(encoder='vits',features=64,out_channels=[48,96,192,384])
    model.load_state_dict(torch.load(ROOT/'dependencies/VideoDepthAnything/checkpoints/video_depth_anything_vits.pth',map_location='cpu',weights_only=True),strict=True)
    model=model.cuda().eval()
    frames=[]
    n=44 if a.test else 300
    for i in range(n):
        f=cv2.imread(str(ROOT/f'source/frames/{i:06d}.png'))
        # Neutral backing eliminates saturated screen bias in geometric inference.
        rgb=f[...,::-1].astype(np.float32)/255
        bg=(rgb[...,1]>.68)&((rgb[...,1]-np.maximum(rgb[...,0],rgb[...,2]))>.48)
        rgb[bg]=.20
        frames.append(cv2.resize((rgb*255).astype(np.uint8),(768,432),interpolation=cv2.INTER_AREA))
    start=time.time()
    depths,fps=model.infer_video_depth(np.stack(frames),30,input_size=392,device='cuda',fp32=False)
    dest=ROOT/'relight/depth';dest.mkdir(exist_ok=True)
    for i,d in enumerate(depths):np.save(dest/f'{i:06d}.npy',d.astype(np.float32))
    (ROOT/'logs/depth.json').write_text(json.dumps({'model':'Video Depth Anything Small','frames':n,'input_short_side':392,'output':[768,432],'seconds':time.time()-start,'temporal_window':32,'overlap':10,'aligned_blend_frames':8,'max_vram_GB':torch.cuda.max_memory_allocated()/1e9},indent=2))
    print((ROOT/'logs/depth.json').read_text(),flush=True)

if __name__=='__main__':main()
