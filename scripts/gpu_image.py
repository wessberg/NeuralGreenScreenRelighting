"""GPU image maths for the per-pixel stages, with an honest CPU fallback.

The heavy stages of this pipeline are already on the GPU -- Cycles through OPTIX, and every
neural estimator through torch. The gap was everything in between: the SH irradiance evaluation,
the gain map, the albedo division and the ratio composite are all per-pixel arithmetic over
1920x1080x3, repeated for hundreds of frames, and they were running in numpy on the CPU.

**`cv2.cuda` is not an option here**: the installed OpenCV wheel reports 0 CUDA devices, so every
`cv2.resize`, `cv2.GaussianBlur` and `cv2.inpaint` is CPU-bound no matter what. torch is already a
hard dependency and already has a working sm_120 build, so it is the sane accelerator.

Everything here falls back to CPU rather than failing, because a stage that silently produces
nothing is worse than a slow one -- but `describe()` reports which device was actually used, so a
fallback shows up in the logs instead of hiding.
"""
import os
import numpy as np

_TORCH = None
_DEVICE = None


def _torch():
    global _TORCH, _DEVICE
    if _TORCH is None:
        try:
            import torch
            _TORCH = torch
            forced = os.environ.get('PIPELINE_DEVICE')
            if forced:
                _DEVICE = torch.device(forced)
            else:
                _DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        except Exception:
            _TORCH, _DEVICE = False, 'cpu'
    return _TORCH


def available():
    t = _torch()
    return bool(t) and _DEVICE.type == 'cuda'


def describe():
    t = _torch()
    if not t:
        return {'backend': 'numpy', 'device': 'cpu', 'reason': 'torch unavailable'}
    if _DEVICE.type != 'cuda':
        return {'backend': 'torch', 'device': str(_DEVICE),
                'reason': 'CUDA not available or PIPELINE_DEVICE overrides it'}
    return {'backend': 'torch', 'device': str(_DEVICE),
            'gpu': t.cuda.get_device_name(0),
            'capability': list(t.cuda.get_device_capability(0))}


def to_dev(a):
    """(H,W) or (H,W,C) float array -> NCHW tensor on the device."""
    t = _torch()
    x = np.ascontiguousarray(a, dtype=np.float32)
    if x.ndim == 2:
        x = x[..., None]
    return t.from_numpy(x).permute(2, 0, 1)[None].to(_DEVICE)


def to_cpu(x, squeeze=False):
    a = x[0].permute(1, 2, 0).detach().cpu().numpy()
    return a[..., 0] if (squeeze and a.shape[2] == 1) else a


def gaussian_blur(a, sigma):
    """Separable Gaussian. Matches cv2.GaussianBlur(..., (0,0), sigma) closely enough that a
    stage can be switched between backends without re-tuning its radii."""
    if not available() or sigma <= 0:
        import cv2
        return cv2.GaussianBlur(a, (0, 0), sigma)
    t = _torch()
    # cv2 picks ksize = round(sigma * 4) * 2 + 1 for float images (3x for 8-bit). Truncating at
    # 3 sigma instead left a max error of 0.217 against cv2 on [0,1] data -- far too big for a
    # drop-in replacement, and exactly the kind of quiet mismatch that would show up later as an
    # unexplained difference between two runs of the same stage.
    r = max(int(round(sigma * 4.0)), 1)
    k = t.arange(-r, r + 1, device=_DEVICE, dtype=t.float32)
    k = t.exp(-(k ** 2) / (2 * sigma * sigma))
    k = k / k.sum()
    x = to_dev(a)
    c = x.shape[1]
    kh = k.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
    kv = k.view(1, 1, -1, 1).repeat(c, 1, 1, 1)
    x = t.nn.functional.conv2d(t.nn.functional.pad(x, (r, r, 0, 0), mode='reflect'),
                               kh, groups=c)
    x = t.nn.functional.conv2d(t.nn.functional.pad(x, (0, 0, r, r), mode='reflect'),
                               kv, groups=c)
    return to_cpu(x, squeeze=(np.asarray(a).ndim == 2))


def resize(a, size, mode='bilinear'):
    """size is (width, height), matching cv2's argument order rather than torch's."""
    w, h = size
    if not available():
        import cv2
        interp = {'bilinear': cv2.INTER_LINEAR, 'area': cv2.INTER_AREA,
                  'nearest': cv2.INTER_NEAREST}[mode]
        return cv2.resize(a, (w, h), interpolation=interp)
    t = _torch()
    x = to_dev(a)
    if mode == 'area':
        x = t.nn.functional.interpolate(x, size=(h, w), mode='area')
    else:
        x = t.nn.functional.interpolate(x, size=(h, w), mode=mode,
                                        align_corners=False if mode == 'bilinear' else None)
    return to_cpu(x, squeeze=(np.asarray(a).ndim == 2))


def ratio_composite(E_target, E_source, plate, alpha, anchor, max_stops, floor, blur_sigma):
    """The whole per-frame ratio composite in one device round trip.

    Chaining the individual helpers would move 1920x1080x3 across PCIe six times per frame. This
    does the division, the anchor, the one-sided log clamp, the blur and the plate multiply on
    the device and brings back only the finished frame.
    """
    if not available():
        import cv2
        R = E_target / np.maximum(E_source, 1e-3)
        R = R * anchor
        g = np.log2(np.maximum(R, 1e-8))
        g = np.where(g > 0, max_stops * np.tanh(g / max_stops), g)
        R = np.maximum(np.exp2(g), floor)
        R = cv2.GaussianBlur(R, (0, 0), blur_sigma)
        return plate * R, R
    t = _torch()
    Et, Es = to_dev(E_target), to_dev(E_source)
    R = Et / t.clamp(Es, min=1e-3) * anchor
    g = t.log2(t.clamp(R, min=1e-8))
    # Clamp the TOP only: a shadow going dark is the point of the whole stage.
    g = t.where(g > 0, max_stops * t.tanh(g / max_stops), g)
    R = t.clamp(t.exp2(g), min=floor)
    r = max(int(round(blur_sigma * 4.0)), 1)
    k = t.arange(-r, r + 1, device=_DEVICE, dtype=t.float32)
    k = t.exp(-(k ** 2) / (2 * blur_sigma * blur_sigma))
    k = k / k.sum()
    c = R.shape[1]
    kh = k.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
    kv = k.view(1, 1, -1, 1).repeat(c, 1, 1, 1)
    R = t.nn.functional.conv2d(t.nn.functional.pad(R, (r, r, 0, 0), mode='reflect'), kh, groups=c)
    R = t.nn.functional.conv2d(t.nn.functional.pad(R, (0, 0, r, r), mode='reflect'), kv, groups=c)
    out = to_dev(plate) * R
    return to_cpu(out), to_cpu(R)


def poisson_integrate(div):
    """Solve grad^2 Y = div with Neumann boundaries, via DCT-II. GPU path uses the FFT identity.

    scipy's dctn is CPU-only; torch has no DCT, but a DCT-II is an FFT of the mirror-extended
    signal, which is exact rather than an approximation.
    """
    H, W = div.shape
    if not available():
        from scipy.fft import dctn, idctn
        d = dctn(div, type=2, norm='ortho')
        ii, jj = np.indices((H, W), dtype=np.float64)
        den = 2 * (np.cos(np.pi * ii / H) + np.cos(np.pi * jj / W) - 2)
        den[0, 0] = 1.0
        return idctn(d / den, type=2, norm='ortho')
    t = _torch()

    def dct2(x):
        # DCT-II along the last axis by mirror extension, applied to rows then columns.
        def d1(v):
            n = v.shape[-1]
            ext = t.cat([v, v.flip(-1)], dim=-1)
            f = t.fft.rfft(ext, dim=-1)[..., :n]
            k = t.arange(n, device=v.device, dtype=v.dtype)
            w = t.exp(-1j * np.pi * k / (2 * n))
            out = (f * w).real
            out = out * np.sqrt(2.0 / n)
            out[..., 0] = out[..., 0] / np.sqrt(2.0)
            return out
        return d1(d1(x).transpose(-1, -2)).transpose(-1, -2)

    def idct2(x):
        def i1(v):
            n = v.shape[-1]
            v = v.clone()
            v[..., 0] = v[..., 0] * np.sqrt(2.0)
            v = v * np.sqrt(n / 2.0)
            k = t.arange(n, device=v.device, dtype=v.dtype)
            w = t.exp(1j * np.pi * k / (2 * n))
            f = t.zeros(v.shape[:-1] + (2 * n,), dtype=t.complex64, device=v.device)
            f[..., :n] = v.to(t.complex64) * w
            return t.fft.irfft(f[..., :n + 1], n=2 * n, dim=-1)[..., :n] * (2 * n) / (2 * n)
        return i1(i1(x).transpose(-1, -2)).transpose(-1, -2)

    x = t.from_numpy(np.ascontiguousarray(div, np.float32)).to(_DEVICE)
    d = dct2(x)
    ii = t.arange(H, device=_DEVICE, dtype=t.float32).view(-1, 1)
    jj = t.arange(W, device=_DEVICE, dtype=t.float32).view(1, -1)
    den = 2 * (t.cos(np.pi * ii / H) + t.cos(np.pi * jj / W) - 2)
    den[0, 0] = 1.0
    return idct2(d / den).cpu().numpy()


if __name__ == '__main__':
    import json, time
    print(json.dumps(describe(), indent=2))
    if available():
        a = np.random.rand(1080, 1920, 3).astype(np.float32)
        import cv2
        t0 = time.time()
        for _ in range(10):
            cv2.GaussianBlur(a, (0, 0), 1.5)
        cpu = time.time() - t0
        gaussian_blur(a, 1.5)                       # warm up
        t0 = time.time()
        for _ in range(10):
            gaussian_blur(a, 1.5)
        gpu = time.time() - t0
        print('blur 1920x1080x3 x10:  cpu %.3f s   gpu %.3f s   speedup %.1fx'
              % (cpu, gpu, cpu / max(gpu, 1e-6)))
        err = float(np.abs(gaussian_blur(a, 1.5) - cv2.GaussianBlur(a, (0, 0), 1.5)).max())
        print('max abs difference vs cv2: %.5f' % err)
