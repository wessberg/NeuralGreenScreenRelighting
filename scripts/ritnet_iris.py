"""Released RITnet visible eye-part proposals; not calibrated repair confidence."""
import hashlib
import importlib.util
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1]


class RITnet:
    def __init__(self,repository=None,device='cuda'):
        repo=Path(repository or ROOT/'dependencies/RITnet')
        spec=importlib.util.spec_from_file_location('_ritnet_original',repo/'densenet.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.model=module.DenseNet2D(dropout=True,prob=.2)
        weights=repo/'best_model.pkl'
        self.model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
        self.device=torch.device(device);self.model.to(self.device).eval()
        self.clahe=cv2.createCLAHE(clipLimit=1.5,tileGridSize=(8,8))
        self.provenance={'repository':'https://github.com/AayushKrChaudhary/RITnet',
            'expected_revision':'6431c57ce7bf0eda935fb6178b926ae9440b50bf',
            'weights_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),
            'architecture_sha256':hashlib.sha256((repo/'densenet.py').read_bytes()).hexdigest(),
            'strict_state_dict':True,'torch':torch.__version__,'device':str(self.device),
            'classes':{'0':'background','1':'sclera','2':'iris','3':'pupil'},
            'preprocessing':'aspect-preserving fit within 400x640, centered zero pad; PIL grayscale; gamma .8; CLAHE 1.5/8x8; (x/255-.5)/.5',
            'limits':'trained OpenEDS near-eye images; RGB performance footage domain and padding adaptation unvalidated'}

    @torch.inference_mode()
    def predict(self,rgb,*,pixel_aspect=1.,excluded=None):
        rgb=np.asarray(rgb,dtype=np.float32).copy()
        if rgb.ndim!=3 or rgb.shape[2]!=3 or not np.isfinite(rgb).all():
            raise ValueError('Expected finite display RGB')
        if not np.isfinite(pixel_aspect) or pixel_aspect<=0:raise ValueError('Invalid pixel aspect')
        if excluded is not None:
            if excluded.shape!=rgb.shape[:2]:raise ValueError('Exclusion pixel space mismatch')
            rgb[np.asarray(excluded,bool)]=0
        h,w=rgb.shape[:2]
        gray=np.asarray(Image.fromarray(np.clip(rgb,0,255).astype(np.uint8)).convert('L'))
        if gray.std()<1e-6:return np.full((h,w),-1,np.int8)
        scale=min(400/(w*pixel_aspect),640/h)
        nw,nh=max(1,round(w*pixel_aspect*scale)),max(1,round(h*scale))
        ox,oy=(400-nw)//2,(640-nh)//2
        canvas=np.zeros((640,400),np.uint8)
        canvas[oy:oy+nh,ox:ox+nw]=cv2.resize(gray,(nw,nh),interpolation=cv2.INTER_LANCZOS4)
        table=255.*np.linspace(0,1,256)**.8
        enhanced=self.clahe.apply(cv2.LUT(canvas,table).astype(np.uint8))
        tensor=torch.from_numpy((enhanced.astype(np.float32)/255.-.5)/.5)[None,None].to(self.device)
        logits=self.model(tensor)
        if not torch.isfinite(logits).all():raise RuntimeError('Nonfinite RITnet output')
        pred=logits.argmax(1)[0].cpu().numpy().astype(np.float32)
        yy,xx=np.mgrid[:h,:w]
        mx=((xx+.5)*nw/w-.5+ox).astype(np.float32)
        my=((yy+.5)*nh/h-.5+oy).astype(np.float32)
        labels=cv2.remap(pred,mx,my,cv2.INTER_NEAREST).astype(np.int8)
        if excluded is not None:labels[np.asarray(excluded,bool)]=-1
        return labels
