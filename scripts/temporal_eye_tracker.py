"""Frozen CoTracker coordinates/visibility for native eye video; never repair RGB."""
import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

from temporal_iris import footprint_valid

ROOT = Path(__file__).resolve().parents[1]
REVISION = '82e02e8029753ad4ef13cf06be7f4fc5facdda4d'
URL = 'https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth'
CHECKPOINT_SHA256 = '2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834'


def prepare_video(rgb, queries, excluded=None, pixel_aspect=1., canvas=(384,512)):
    a = np.asarray(rgb, np.float32).copy()
    q = np.asarray(queries, np.float32).copy()
    if a.ndim != 4 or a.shape[-1] != 3 or min(a.shape[1:3]) < 2:
        raise ValueError('Expected TxHxWx3 native display RGB')
    t,h,w,_ = a.shape
    if q.ndim != 2 or q.shape[1] != 3 or not len(q) or not np.isfinite(q).all():
        raise ValueError('Expected finite Nx3 t,x,y queries')
    if (q[:,0] != np.floor(q[:,0])).any() or (q[:,0]<0).any() or (q[:,0]>=t).any():
        raise ValueError('Query frame outside video')
    if (q[:,1:]<0).any() or (q[:,1:] > [w-1,h-1]).any():
        raise ValueError('Query outside native image')
    if excluded is not None:
        mask = np.asarray(excluded, bool)
        if mask.shape != a.shape[:3]:
            raise ValueError('Exclusions must be native TxHxW')
        a[mask] = 0
    if not np.isfinite(a).all() or a.min()<0 or a.max()>255:
        raise ValueError('Expected display RGB in 0..255')
    if not np.isfinite(pixel_aspect) or pixel_aspect<=0:
        raise ValueError('Invalid pixel aspect')
    ch,cw = canvas
    scale = min((cw-1)/((w-1)*pixel_aspect), (ch-1)/(h-1))
    nw = max(2, min(cw, round((w-1)*pixel_aspect*scale)+1))
    nh = max(2, min(ch, round((h-1)*scale)+1))
    left,top = (cw-nw)//2,(ch-nh)//2
    gain = np.array([(nw-1)/(w-1),(nh-1)/(h-1)], np.float32)
    offset = np.array([left,top],np.float32)
    v = torch.from_numpy(a).permute(0,3,1,2)
    v = F.interpolate(v,(nh,nw),mode='bilinear',align_corners=True)
    v = F.pad(v,(left,cw-nw-left,top,ch-nh-top),value=0)[None]
    q[:,1:] = q[:,1:]*gain+offset
    return v,torch.from_numpy(q)[None],gain,offset


class TemporalEyeTracker:
    def __init__(self, device=None):
        checkout = ROOT/'dependencies/co-tracker'
        revision = subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip()
        if revision != REVISION:
            raise ValueError('CoTracker checkout differs from evaluated revision')
        dirty = subprocess.check_output(['git','-C',str(checkout),'status','--porcelain','--untracked-files=no'],text=True)
        if dirty.strip():
            raise ValueError('CoTracker source has local changes')
        sys.path.insert(0,str(checkout))
        from cotracker.predictor import CoTrackerPredictor
        checkpoint = ROOT/'dependencies/cotracker-weights/scaled_offline.pth'
        if not checkpoint.exists():
            raise FileNotFoundError(f'Download {URL} to {checkpoint}')
        checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if checkpoint_hash != CHECKPOINT_SHA256:
            raise ValueError('Checkpoint differs from evaluated CoTracker3 release')
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CoTrackerPredictor(checkpoint=None,offline=True,v2=False,window_len=60)
        self.model.model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
        self.model.eval().to(self.device)
        self.provenance = dict(model='CoTracker3 scaled offline',revision=revision,url=URL,
            sha256=checkpoint_hash,device=self.device,
            torch=torch.__version__,dtype='float32',iterations=6,visibility_threshold=.9,
            support_grid=6,query_frame_forced_visible=True,
            input='physical-aspect letterbox; native display RGB 0..255')

    @torch.inference_mode()
    def track(self,rgb,queries,*,excluded=None,pixel_aspect=1.):
        v,q,gain,offset = prepare_video(rgb,queries,excluded,pixel_aspect,tuple(self.model.interp_shape))
        tracks,visible = self.model(v.to(self.device),queries=q.to(self.device),backward_tracking=True)
        xy = (tracks[0].cpu().numpy()-offset)/gain
        valid = np.stack([footprint_valid(np.ones(np.shape(rgb)[1:3],bool),p) for p in xy])
        return dict(xy=xy,visible=visible[0].cpu().numpy(),native_footprint=valid,
                    usable_visibility=visible[0].cpu().numpy() & valid)
