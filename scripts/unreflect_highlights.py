"""Released UnReflectAnything highlight proposals; never synthesizes performer RGB."""
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from dotmap import DotMap
import unreflectanything
from unreflectanything.model_ import UnReflectModel

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_SHA256 = '12c0cf4518a39325b7bfe2bed04a96395189f2643c2f90b7b79360a659dc739b'


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class UnreflectHighlights:
    def __init__(self, device='cuda'):
        self.device = device
        weights = ROOT/'dependencies/unreflect-weights/weights/full_model_weights.pt'
        if file_hash(weights) != CHECKPOINT_SHA256:
            raise ValueError('UnReflectAnything checkpoint differs from recorded author release')
        with torch.serialization.safe_globals([DotMap]):
            checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
        checkpoint.pop('optimizer_state_dict', None)
        config = UnReflectModel._sanitize_embedded_config(checkpoint['config'], False)
        config['MODEL']['RGB_ENCODER']['ENCODER'] = str(ROOT/'dependencies/unreflect-weights/encoder')
        wrapper = unreflectanything.model(weights=checkpoint, config=config, device='cpu', strict=True)
        expected = UnReflectModel._strip_key_prefixes(wrapper._model, checkpoint['model_state_dict'], False)
        actual = wrapper._model.state_dict()
        assert actual.keys() == expected.keys()
        assert all(torch.equal(actual[k], expected[k]) for k in expected)
        self.provenance = dict(model='UnReflectAnything 1.1.1 highlight head only',
            checkpoint_sha256=CHECKPOINT_SHA256, strict_state_entries=len(actual),
            exact_loaded_tensors=True, config=config, torch=torch.__version__,
            device=device, dtype='float32', model_size=896, diffuse_decoder_executed=False,
            token_inpainter_executed=False, source_fallback_required=True)
        self.inner = wrapper._model.to(device).eval()

    @torch.inference_mode()
    def predict(self, rgb):
        a = np.asarray(rgb, np.float32)
        if a.ndim != 3 or a.shape[-1] != 3 or not np.isfinite(a).all() or a.min() < 0 or a.max() > 255:
            raise ValueError('Expected finite native display RGB HxWx3 in 0..255')
        x = torch.from_numpy(a.copy()).permute(2, 0, 1)[None].to(self.device)/255
        prepared = self.inner.dinov3.preprocess_image(x)
        features = self.inner.dinov3(prepared)['selected_hidden_states']
        highlight = self.inner.decoders['highlight'](features)
        native = F.interpolate(highlight, size=a.shape[:2], mode='bilinear', align_corners=False)
        return dict(model_probability=highlight[0, 0].cpu().numpy(),
                    native_probability=native[0, 0].cpu().numpy())
