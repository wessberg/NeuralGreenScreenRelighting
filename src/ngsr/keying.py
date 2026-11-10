"""Photographic-core preservation and per-shot despill, computed on CUDA."""
import numpy as np


def _cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Key postprocessing requires CUDA")
    return torch


def hint_for(rgb, config):
    t = _cuda()
    import torch.nn.functional as F
    image = t.as_tensor(np.ascontiguousarray(rgb), device="cuda")
    r, g, b = image.unbind(-1)
    hint = (~((g > config["green_min"]) &
              (g - t.maximum(r, b) > config["green_dominance_min"]))).float()[None, None]
    hint = -F.max_pool2d(-hint, 3, stride=1, padding=1)
    axis = t.arange(-4, 5, device="cuda", dtype=t.float32)
    kernel = t.exp(-axis.square() / 2)
    kernel /= kernel.sum()
    for weight in (kernel[None, None, None, :], kernel[None, None, :, None]):
        pad = (4, 4, 0, 0) if weight.shape[-1] == 9 else (0, 0, 4, 4)
        hint = F.conv2d(F.pad(hint, pad, mode="reflect"), weight)
    return hint[0, 0].cpu().numpy()


def preserve_core(source_srgb, foreground_linear, alpha, keying, hint):
    t = _cuda()
    import torch.nn.functional as F
    source = t.as_tensor(np.ascontiguousarray(source_srgb), device="cuda")
    fg = t.as_tensor(np.ascontiguousarray(foreground_linear), device="cuda").clamp(min=0)
    a = t.as_tensor(np.ascontiguousarray(alpha), device="cuda").clamp(0, 1)
    support = t.as_tensor(hint > .05, device="cuda").float()[None, None]
    support = F.max_pool2d(support, 17, stride=1, padding=8)[0, 0]
    a = a * support
    core = ((a - .88) / .10).clamp(0, 1)[..., None]
    fg = t.where(fg <= .0031308, fg * 12.92, 1.055 * fg.pow(1/2.4) - .055)
    fg = fg * (1-core) + source * core
    mode = keying.get("despill_mode", "edge_only")
    if mode not in ("off", "edge_only", "global"):
        raise ValueError(f"Unknown despill mode: {mode}")
    if mode != "off":
        ref = keying.get("despill_limit", "max")
        if ref not in ("max", "average"):
            raise ValueError(f"Unknown despill limit: {ref}")
        r, g, b = fg.unbind(-1)
        luma = fg @ t.tensor([.2126, .7152, .0722], device="cuda")
        offset = keying.get("despill_offset_abs", .08) + keying.get("despill_offset_rel", 0) * luma
        limit = t.maximum(r, b) if ref == "max" else (r+b)*.5
        excess = ((g-limit-offset)*keying.get("despill_gain", 2)).clamp(0, keying.get("despill_max", .8))
        weight = excess * (1-core[..., 0]) if mode == "edge_only" else excess
        fg[..., 1] = g - weight*(g-(r+b)*.5)
    fg = fg.clamp(min=0)
    fg = t.where(fg <= .04045, fg / 12.92, ((fg+.055)/1.055).pow(2.4))
    if not t.isfinite(fg).all() or not t.isfinite(a).all():
        raise ValueError("Keyer produced non-finite pixels")
    return fg.cpu().numpy(), a.cpu().numpy()
