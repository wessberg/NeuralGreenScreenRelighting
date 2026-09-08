"""Pretrained EllSeg geometry proposals, never visible-material or repair masks.

Run in dependencies/ellseg-runtime/.venv. The upstream checkpoint learns complete
ellipses, including occluded parts. No output of this adapter authorizes removal.
Input is native display RGB (0..255); geometry is mapped back to native pixels.
"""
import hashlib
from pathlib import Path
import sys
import types

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REVISION = '8f1ea13336fa9c662403bd5823ac13cc6a6dd632'


def prepare(rgb, pixel_aspect=1., excluded=None):
    """Width alignment as upstream, with integer crop/pad and explicit pixel map.

    Exclusions are scrubbed BEFORE grayscale, resize and normalization. Native
    outputs corresponding to cropped-away input are unknown (-1), not background.
    Pixel aspect is incorporated in this single resampling, avoiding double blur.
    """
    rgb = np.asarray(rgb, dtype=np.float32).copy()
    if rgb.ndim != 3 or rgb.shape[2] != 3 or not np.isfinite(rgb).all():
        raise ValueError('Expected finite native display RGB')
    if not np.isfinite(pixel_aspect) or pixel_aspect <= 0:
        raise ValueError('Invalid pixel aspect')
    if excluded is not None:
        if excluded.shape != rgb.shape[:2]:
            raise ValueError('Exclusion pixel space mismatch')
        rgb[np.asarray(excluded, bool)] = 0
    h, w = rgb.shape[:2]
    nh = max(1, int(h * 320 / (w * pixel_aspect)))
    gray = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    scaled = cv2.resize(gray, (320, nh), interpolation=cv2.INTER_LANCZOS4)
    offset = (240 - nh) // 2
    canvas = np.zeros((240, 320), np.uint8)
    lo, hi = max(0, offset), min(240, offset + nh)
    canvas[lo:hi] = scaled[lo-offset:hi-offset]
    std = float(canvas.std())
    # A constant native eye cannot acquire evidence from artificial padding.
    usable = float(gray.std()) > 1e-6 and std > 1e-6
    tensor = (canvas.astype(np.float32) - float(canvas.mean())) / max(std, 1e-6)
    # OpenCV pixel-centre resize convention, followed by integer vertical offset.
    matrix = np.array([[320/w, 0, (320/w-1)/2],
                       [0, nh/h, (nh/h-1)/2+offset], [0, 0, 1]], float)
    return tensor, matrix, usable


class EllSeg:
    def __init__(self, repository=None, weights=None, device='cuda'):
        repository = Path(repository or ROOT/'dependencies/EllSeg').resolve()
        weights = Path(weights or repository/'weights/all.git_ok').resolve()
        sys.path.insert(0, str(repository))
        # Only legacy np.int scalar casts change. Keep the checkout byte-identical,
        # use the authors' architecture and strict-load every pretrained parameter.
        path = repository/'models/RITnet_v3.py'
        source = path.read_text(encoding='utf-8')
        module = types.ModuleType('_ellseg_ritnet_v3_compatible')
        exec(compile(source.replace('np.int(', 'int('), str(path), 'exec'), module.__dict__)
        self.device = torch.device(device)
        self.model = module.DenseNet2D()
        checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.model.to(self.device).eval()
        self.provenance = {
            'repository': 'https://github.com/RSKothari/EllSeg',
            'expected_revision': REVISION,
            'weights': weights.name, 'weights_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
            'architecture_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'compatibility': 'np.int scalar casts -> int in loaded architecture; integer crop/pad; safe weights-only load',
            'torch': torch.__version__, 'numpy': np.__version__, 'opencv': cv2.__version__,
            'device': str(self.device), 'strict_state_dict': True,
            'mask_semantics': 'full inferred iris/pupil, not observed visibility',
        }

    @torch.inference_mode()
    def predict(self, rgb, *, pixel_aspect=1., excluded=None):
        frame, matrix, usable = prepare(rgb, pixel_aspect, excluded)
        h, w = rgb.shape[:2]
        if not usable:
            return {'labels': np.full((h,w), -1, np.int8), 'ellipses': {},
                    'reason': 'constant_input', 'native_to_model': matrix}
        tensor = torch.from_numpy(frame)[None, None].to(self.device)
        x4, x3, x2, x1, x = self.model.enc(tensor)
        reg = self.model.elReg(x, 0)[0]
        logits = self.model.dec(x4, x3, x2, x1, x)
        if not torch.isfinite(logits).all() or not torch.isfinite(reg).all():
            raise RuntimeError('Nonfinite EllSeg inference')
        # Same softargmax temperature and normalized mesh as upstream inference.
        yy, xx = torch.meshgrid(torch.linspace(-1,1,240,device=self.device),
                                torch.linspace(-1,1,320,device=self.device), indexing='ij')
        ellipses = {}
        for name, channel, axes in [('pupil', logits[0,2], reg[7:10]),
                                    ('limbus', -logits[0,0], reg[2:5])]:
            wt = torch.softmax(channel.flatten()*4, dim=0).reshape(240,320)
            params = torch.cat([torch.stack([(wt*xx).sum(), (wt*yy).sum()]), axes]).cpu().numpy()
            # Sample the complete normalized ellipse then apply the exact affine
            # mapping. This also handles pixel aspect without assuming circles.
            t = np.linspace(0, 2*np.pi, 720, endpoint=False)
            c, s = np.cos(params[4]), np.sin(params[4])
            curve = np.column_stack([params[2]*np.cos(t), params[3]*np.sin(t)])
            curve = curve @ np.array([[c,s],[-s,c]]) + params[:2]
            curve = (curve+1)*[160,120]
            inv = np.linalg.inv(matrix)
            native = np.column_stack([curve,np.ones(len(curve))]) @ inv.T
            center = inv @ np.r_[(params[:2]+1)*[160,120],1]
            ellipses[name] = {'boundary_xy':native[:,:2], 'center_xy':center[:2],
                              'normalized_parameters':params}
        labels_model = logits.argmax(1)[0].cpu().numpy().astype(np.float32)
        yy, xx = np.mgrid[:h,:w]
        mx = (matrix[0,0]*xx+matrix[0,2]).astype(np.float32)
        my = (matrix[1,1]*yy+matrix[1,2]).astype(np.float32)
        labels = cv2.remap(labels_model,mx,my,cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT,borderValue=-1).astype(np.int8)
        if excluded is not None:
            labels[np.asarray(excluded,bool)] = -1
        return {'labels':labels, 'ellipses':ellipses,
                'reason':'unvalidated_complete_ellipse', 'native_to_model':matrix}


def summary(prediction):
    return {'reason':prediction['reason'],
            'native_to_model':prediction['native_to_model'].tolist(),
            'ellipses':{n:{k:v.tolist() for k,v in e.items() if k!='boundary_xy'}
                        for n,e in prediction['ellipses'].items()},
            'geometry_validated':False, 'visibility_validated':False,
            'scene_reflection_allowed':False}
