"""The eye pass's per-pixel maths on the GPU, batched over eye-frames.

Round 39. The artist, round 38: *"Never run stuff on the CPU when it can be avoided"*, and
this round: *"Port it to the GPU instead of optimizing it on the CPU."* The eye pass
(`eye_soft_removal.py`) ran single-threaded numpy for 21 minutes per 1200 frames with the
RTX 5080 at 1 %. Profiled on 200 frames (181 s): 34 % was converting whole 1920x1080 frames to
sRGB, 31 % the atlas registration search (~1M small remaps), 10 % unpacking whole frames to
read 40-pixel windows, and the eye maths itself about 6 %.

**This module is the per-pixel half; `eye_removal_gpu.py` is the orchestration.** Every
function here takes a BATCH of eye windows of one shape, `(B, h, w)` or `(B, h, w, C)`,
channels last so the code reads line for line against the numpy original it replaces.
Eye windows are 42-43 px, so a clip holds ~2400 of them in a few tens of MB of GPU memory.

**Reproduction, measured primitive by primitive against cv2/numpy (`--selftest`):**

* `remap_linear` / `remap_nearest` are BIT EXACT. cv2 4.13's float remap quantises the map
  to 1/32 px and interpolates with a float weight table, summing
  `((s00*w00 + s01*w01) + s10*w10) + s11*w11`; this does the same, one torch op per step so
  nothing is fused into an FMA.
* `percentile` is bit exact: numpy's weak-scalar `_lerp` in float32.
* `median` is exact (numpy's mean of the two middle values).
* `gaussian_blur` and `distance_chamfer` agree to FLOAT EPSILON (1.8e-7 and ~3e-5): cv2
  accumulates in an order (SIMD FMA; a raster two-pass) a batched kernel does not follow.
* `srgb` uses float64 pow rounded to float32; numpy's `powf` is not correctly rounded and
  disagrees on ~0.01 % of values in the last bit. For the SOURCE frames, which are half
  floats, `rgb8_half` is a 65536-entry table built by numpy itself, so the written source
  PNGs are bit exact.

Everything that is a per-frame SCALAR (a median of three gains, a lid coefficient) stays in
numpy in the orchestration: it is bookkeeping on a few thousand numbers, not pixel maths,
and keeping it in numpy keeps it exact.

    dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/eye_gpu.py --selftest
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
F32, F64 = torch.float32, torch.float64


def dev_scalar(x, dtype=F32):
    """A 0-d DEVICE tensor. Dividing by a Python float makes torch's CUDA kernel multiply by
    its reciprocal, which rounds differently from numpy's true division; a device tensor
    divisor takes the ordinary division path."""
    return torch.tensor(x, dtype=dtype, device=DEV)


def div(a, s):
    """`a / s` with numpy's rounding for a Python-scalar `s` (see `dev_scalar`)."""
    if isinstance(s, torch.Tensor):
        return a / s
    return a / dev_scalar(s, a.dtype)


def as_t(a, dtype=None):
    t = torch.as_tensor(np.ascontiguousarray(a), device=DEV)
    return t if dtype is None else t.to(dtype)


# --------------------------------------------------------------------------- remap
def _taps(src, x, y, border):
    """src (B, H, W, C); x, y int64 (B, Ho, Wo) -> (B, Ho, Wo, C), 0 outside for CONSTANT."""
    B, H, W, C = src.shape
    if border == 'replicate':
        xc, yc = x.clamp(0, W - 1), y.clamp(0, H - 1)
        ok = None
    else:
        ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        xc, yc = x.clamp(0, W - 1), y.clamp(0, H - 1)
    flat = src.reshape(B, H * W, C)
    idx = (yc * W + xc).reshape(B, -1, 1).expand(-1, -1, C)
    out = torch.gather(flat, 1, idx).reshape(*x.shape, C)
    if ok is not None:
        out = torch.where(ok[..., None], out, torch.zeros((), dtype=out.dtype, device=DEV))
    return out


def remap_linear(src, mx, my, border='constant'):
    """cv2.remap(src, mx, my, INTER_LINEAR, borderMode), float32, bit exact.

    `src` (B, H, W) or (B, H, W, C) float32; `mx`, `my` float32 (B, Ho, Wo) or (Ho, Wo).
    """
    squeeze = src.dim() == 3
    if squeeze:
        src = src[..., None]
    B = src.shape[0]
    if mx.dim() == 2:
        mx = mx[None].expand(B, -1, -1)
        my = my[None].expand(B, -1, -1)
    ix = torch.round(mx * 32).to(torch.int64)
    iy = torch.round(my * 32).to(torch.int64)
    x0, y0 = ix >> 5, iy >> 5
    ax = (ix & 31).to(F32) * (1.0 / 32)
    ay = (iy & 31).to(F32) * (1.0 / 32)
    one = torch.ones((), dtype=F32, device=DEV)
    w00 = ((one - ay) * (one - ax))[..., None]
    w01 = ((one - ay) * ax)[..., None]
    w10 = (ay * (one - ax))[..., None]
    w11 = (ay * ax)[..., None]
    s00 = _taps(src, x0, y0, border)
    s01 = _taps(src, x0 + 1, y0, border)
    s10 = _taps(src, x0, y0 + 1, border)
    s11 = _taps(src, x0 + 1, y0 + 1, border)
    out = s00 * w00
    out = out + s01 * w01
    out = out + s10 * w10
    out = out + s11 * w11
    return out[..., 0] if squeeze else out


def remap_separable(src, mx, my, pad=8):
    """`remap_linear` (BORDER_CONSTANT) for maps that are SEPARABLE: mx depends only on the
    output column and my only on the output row -- true of every map onto the canonical
    grid, which is a meshgrid. Bit exact against cv2 (the same 1/32 quantisation, the same
    float32 weight products, the same summation order), without materialising an index
    per sample.

    src (B, H, W, C) float32; mx (B, K, Wo), my (B, K, Ho) float32 -> (B, K, Ho, Wo, C).
    Samples may reach at most `pad` - 2 pixels outside the source.
    """
    B, H, W, C = src.shape
    K, Wo, Ho = mx.shape[1], mx.shape[2], my.shape[2]
    sp = torch.zeros((B, H + 2 * pad, W + 2 * pad, C), dtype=src.dtype, device=DEV)
    sp[:, pad:pad + H, pad:pad + W] = src
    ix = torch.round(mx * 32).to(torch.int64)
    iy = torch.round(my * 32).to(torch.int64)
    x0 = ((ix >> 5) + pad).clamp(0, W + 2 * pad - 2)
    y0 = ((iy >> 5) + pad).clamp(0, H + 2 * pad - 2)
    ax = (ix & 31).to(F32) * (1.0 / 32)
    ay = (iy & 31).to(F32) * (1.0 / 32)
    one = torch.ones((), dtype=F32, device=DEV)
    bi = torch.arange(B, device=DEV).view(B, 1)
    r0 = sp[bi, y0.reshape(B, -1)].view(B, K, Ho, W + 2 * pad, C)
    r1 = sp[bi, (y0 + 1).reshape(B, -1)].view(B, K, Ho, W + 2 * pad, C)

    def cols(r, xi):
        idx = xi.view(B, K, 1, Wo, 1).expand(B, K, Ho, Wo, C)
        return torch.gather(r, 3, idx)

    s00, s01 = cols(r0, x0), cols(r0, x0 + 1)
    s10, s11 = cols(r1, x0), cols(r1, x0 + 1)
    ay_ = ay.view(B, K, Ho, 1)
    ax_ = ax.view(B, K, 1, Wo)
    w00 = ((one - ay_) * (one - ax_))[..., None]
    w01 = ((one - ay_) * ax_)[..., None]
    w10 = (ay_ * (one - ax_))[..., None]
    w11 = (ay_ * ax_)[..., None]
    out = s00 * w00
    out = out + s01 * w01
    out = out + s10 * w10
    out = out + s11 * w11
    return out


def remap_single(src, mx, my):
    """`remap_linear` (BORDER_CONSTANT) of ONE source image at many maps of any shape:
    src (H, W, C), mx/my (...) -> (..., C). No per-map copy of the source."""
    H, W, C = src.shape
    flat = src.reshape(H * W, C)
    ix = torch.round(mx * 32).to(torch.int64)
    iy = torch.round(my * 32).to(torch.int64)
    x0, y0 = ix >> 5, iy >> 5
    ax = ((ix & 31).to(F32) * (1.0 / 32))[..., None]
    ay = ((iy & 31).to(F32) * (1.0 / 32))[..., None]
    one = torch.ones((), dtype=F32, device=DEV)

    def tap(x, y):
        ok = ((x >= 0) & (x < W) & (y >= 0) & (y < H))[..., None]
        v = flat[(y.clamp(0, H - 1) * W + x.clamp(0, W - 1))]
        return torch.where(ok, v, torch.zeros((), dtype=v.dtype, device=DEV))

    out = tap(x0, y0) * ((one - ay) * (one - ax))
    out = out + tap(x0 + 1, y0) * ((one - ay) * ax)
    out = out + tap(x0, y0 + 1) * (ay * (one - ax))
    out = out + tap(x0 + 1, y0 + 1) * (ay * ax)
    return out


_CUBIC_TAB = None


def _cubic_tab():
    """cv2's float bicubic table: 32 fractional positions x 4 coefficients, a = -0.75."""
    global _CUBIC_TAB
    if _CUBIC_TAB is None:
        A = np.float32(-0.75)
        rows = []
        for k in range(32):
            x = np.float32(k / 32.0)
            c0 = ((A * (x + 1) - 5 * A) * (x + 1) + 8 * A) * (x + 1) - 4 * A
            c1 = ((A + 2) * x - (A + 3)) * x * x + 1
            c2 = ((A + 2) * (1 - x) - (A + 3)) * (1 - x) * (1 - x) + 1
            c3 = np.float32(1) - c0 - c1 - c2
            rows.append(np.array([c0, c1, c2, c3], np.float32))
        _CUBIC_TAB = torch.as_tensor(np.stack(rows), device=DEV)
    return _CUBIC_TAB


def remap_cubic(src, mx, my, hw=None):
    """cv2.remap(src, mx, my, INTER_CUBIC, BORDER_REPLICATE) for float32, bit exact.

    src (B, H, W); mx, my (B, ...) float32. `hw` (B, 2) int optionally gives each item's
    TRUE height and width inside a larger canvas, so the replicate border is its own. cv2
    sums the 16 taps row by row where the 4x4 block is inside the image and one tap at a
    time from zero where it touches the border; both are reproduced.
    """
    B, H, W = src.shape
    shape = mx.shape
    mxf, myf = mx.reshape(B, -1), my.reshape(B, -1)
    ix = torch.round(mxf * 32).to(torch.int64)
    iy = torch.round(myf * 32).to(torch.int64)
    x0, y0 = (ix >> 5) - 1, (iy >> 5) - 1
    tab = _cubic_tab()
    cx, cy = tab[ix & 31], tab[iy & 31]               # (B, N, 4)
    if hw is None:
        hh = torch.full((B, 1), H, device=DEV, dtype=torch.int64)
        ww = torch.full((B, 1), W, device=DEV, dtype=torch.int64)
    else:
        hh, ww = hw[:, 0:1].to(torch.int64), hw[:, 1:2].to(torch.int64)
    flat = src.reshape(B, H * W)
    interior = (x0 >= 0) & (x0 < ww - 3) & (y0 >= 0) & (y0 < hh - 3)
    total = None
    seq = None
    for i in range(4):
        yy = torch.minimum(torch.clamp_min(y0 + i, 0), hh - 1)
        row = None
        for j in range(4):
            xx = torch.minimum(torch.clamp_min(x0 + j, 0), ww - 1)
            v = torch.gather(flat, 1, yy * W + xx)
            t = v * (cy[..., i] * cx[..., j])
            row = t if row is None else row + t
            seq = t if seq is None else seq + t
        total = row if total is None else total + row
    return torch.where(interior, total, seq).reshape(shape)


def filter_sep(x, kx, ky, border='reflect101'):
    """cv2.sepFilter2D-style separable filter on (B, h, w) float32 with the given 1-D
    kernels (numpy), rows then columns, BORDER_REPLICATE or BORDER_REFLECT_101."""
    B, h, w = x.shape
    rx, ry = len(kx) // 2, len(ky) // 2

    def idx(n, r):
        i = np.arange(-r, n + r)
        if border == 'replicate':
            i = np.clip(i, 0, n - 1)
        else:
            i = np.abs(i)
            i = np.where(i >= n, 2 * (n - 1) - i, i)
        return torch.as_tensor(i, device=DEV)

    xp = x[:, :, idx(w, rx)]
    row = None
    for k, c in enumerate(kx):
        if c == 0:
            continue
        t = float(np.float32(c)) * xp[:, :, k:k + w]
        row = t if row is None else row + t
    yp = row[:, idx(h, ry), :]
    col = None
    for k, c in enumerate(ky):
        if c == 0:
            continue
        t = float(np.float32(c)) * yp[:, k:k + h, :]
        col = t if col is None else col + t
    return col


def gaussian_k(x, ksize, sigma, border='reflect101'):
    """cv2.GaussianBlur(x, (ksize, ksize), sigma, borderType) on float32 (B, h, w)."""
    import cv2
    k = cv2.getGaussianKernel(ksize, sigma, cv2.CV_32F)[:, 0]
    return filter_sep(x, k, k, border)


def sobel3(x, dx, dy):
    """cv2.Sobel(x, CV_32F, dx, dy, ksize=3), BORDER_REFLECT_101."""
    d, s = np.array([-1.0, 0.0, 1.0]), np.array([1.0, 2.0, 1.0])
    return filter_sep(x, d if dx else s, d if dy else s, 'reflect101')


# --------------------------------------------------------------------------- exact cv2/numpy
# cv2's float separable filters and numpy's float32 matmul run 8-wide SIMD with FMA on this
# machine's build, and plain multiply-add on each row's scalar tail. Reproduced below, so the
# trackers' gradient fields are the CPU's bit for bit (pinned by probe, round 39; checked by
# --selftest). An FMA is a float64 multiply-add rounded once to float32.
def fma32(a, b, c):
    """round32(a * b + c) with the product exact (float64)."""
    return (a.to(F64) * b + c.to(F64)).to(F32)


def _tail_mask(w, vec=8):
    nv = (w // vec) * vec
    return torch.arange(w, device=DEV) >= nv          # True on the scalar tail


def luma_dot_np(s, coeff):
    """numpy's `s @ coeff` for float32 s (B, h, w, 3), coeff (3,) -- the BLAS gemv each
    image row goes through: 8-pixel blocks an FMA chain from x0; then a 4-pixel block and a
    last single pixel an FMA chain from x1; a 2-pixel block plain (x0 + x1) + x2."""
    c0, c1, c2 = (float(np.float32(c)) for c in coeff)
    x0, x1, x2 = s[..., 0], s[..., 1], s[..., 2]
    p0, p1, p2 = x0 * c0, x1 * c1, x2 * c2
    v8 = fma32(x2, c2, fma32(x1, c1, p0))
    v41 = fma32(x2, c2, fma32(x0, c0, p1))
    v2 = (p0 + p1) + p2
    w = s.shape[2]
    nv8 = (w // 8) * 8
    kind = np.zeros(w, np.int64)                      # 0: 8-block, 1: 4/1 chain, 2: pair
    rest = w - nv8
    pos = nv8
    if rest >= 4:
        kind[pos:pos + 4] = 1
        pos += 4
    if w - pos >= 2:
        kind[pos:pos + 2] = 2
        pos += 2
    kind[pos:] = 1
    kt = torch.as_tensor(kind, device=DEV).view(1, 1, -1)
    return torch.where(kt == 0, v8, torch.where(kt == 2, v2, v41))


def _pad_idx(n, r, border):
    i = np.arange(-r, n + r)
    if border == 'replicate':
        i = np.clip(i, 0, n - 1)
    else:
        i = np.abs(i)
        i = np.where(i >= n, 2 * (n - 1) - i, i)
    return torch.as_tensor(i, device=DEV)


def cv_gaussian(x, ksize, sigma, border):
    """cv2.GaussianBlur(x, (k, k), sigma, border) for float32 (B, h, w), k >= 7 (the
    generic row filter and the symmetric column filter), bit exact."""
    import cv2
    k = cv2.getGaussianKernel(ksize, sigma, cv2.CV_32F)[:, 0]
    r = ksize // 2
    B, h, w = x.shape
    tail = _tail_mask(w).view(1, 1, -1)
    xp = x[:, :, _pad_idx(w, r, border)]
    taps = [xp[:, :, j:j + w] for j in range(ksize)]
    kf = [float(np.float32(c)) for c in k]
    vec = taps[0] * kf[0]
    for j in range(1, ksize):
        vec = fma32(taps[j], kf[j], vec)
    sc = taps[0] * kf[0]
    for j in range(1, ksize):
        sc = sc + taps[j] * kf[j]
    row = torch.where(tail, sc, vec)
    yp = row[:, _pad_idx(h, r, border), :]
    ctr = yp[:, r:r + h, :] * kf[r]
    vec, sc = ctr, ctr
    for j in range(1, r + 1):
        pair = yp[:, r + j:r + j + h, :] + yp[:, r - j:r - j + h, :]
        vec = fma32(pair, kf[r + j], vec)
        sc = sc + pair * kf[r + j]
    return torch.where(tail, sc, vec)


def cv_sobel3(x, dx, dy):
    """cv2.Sobel(x, CV_32F, dx, dy, ksize=3) (BORDER_REFLECT_101), bit exact: cv2's small
    symmetric filters sum (a+c)+2b on 8-wide vectors, (a+2b)+c on the scalar tail in pairs
    and (a+c)+2b on a trailing odd pixel; the antisymmetric tap is one subtraction."""
    B, h, w = x.shape
    if dy:                          # smooth along x, derivative along y
        xp = x[:, :, _pad_idx(w, 1, 'reflect101')]
        a, b, c = xp[:, :, :w], xp[:, :, 1:w + 1], xp[:, :, 2:w + 2]
        v1 = (a + c) + 2 * b
        v2 = (a + 2 * b) + c
        nv = (w // 8) * 8
        col = torch.arange(w, device=DEV)
        odd_last = ((w - nv) % 2 == 1) & (col == w - 1)
        row = torch.where((col < nv) | odd_last, v1, v2)
        yp = row[:, _pad_idx(h, 1, 'reflect101'), :]
        return yp[:, 2:h + 2, :] - yp[:, :h, :]
    xp = x[:, :, _pad_idx(w, 1, 'reflect101')]
    row = xp[:, :, 2:w + 2] - xp[:, :, :w]
    yp = row[:, _pad_idx(h, 1, 'reflect101'), :]
    a, b, c = yp[:, :h, :], yp[:, 1:h + 1, :], yp[:, 2:h + 2, :]
    v1 = (a + c) + (b + b)
    v2 = (a + 2 * b) + c
    return torch.where(_tail_mask(w).view(1, 1, -1), v2, v1)


_CUBIC_TAB16 = None


def _cubic_tab16():
    """cv2's BicubicTab_f: for each (fy, fx) the 16 weights vy[i] * vx[j], float32."""
    global _CUBIC_TAB16
    if _CUBIC_TAB16 is None:
        t = _cubic_tab().cpu().numpy()                 # (32, 4)
        tab = (t[:, None, :, None] * t[None, :, None, :]).astype(np.float32)  # fy,fx,i,j
        _CUBIC_TAB16 = torch.as_tensor(tab.reshape(32 * 32, 16), device=DEV)
    return _CUBIC_TAB16


class CubicSampler:
    """cv2.remap INTER_CUBIC / BORDER_REPLICATE of several fields at shared points, bit
    exact, with ONE gather of a precomputed 4x4 neighbourhood per sample.

    fields: list of (B, H, W) float32 canvases; hw (B, 2) each item's true size (its
    replicate border). Bases (the 4x4 block's top-left tap) are clamped to [-3, W-1] --
    beyond that every tap clamps to the same edge pixel, so the value is unchanged.
    """

    def __init__(self, fields, hw):
        self.nf = len(fields)
        B, H, W = fields[0].shape
        self.B, self.H, self.W = B, H, W
        self.hw = hw.to(torch.int64)
        self.P = 3
        by = torch.arange(-3, H, device=DEV)          # base rows
        bx = torch.arange(-3, W, device=DEV)
        self.NY, self.NX = by.numel(), bx.numel()
        hh = self.hw[:, 0].view(B, 1, 1)
        ww = self.hw[:, 1].view(B, 1, 1)
        nb = []
        for f in fields:
            flat = f.reshape(B, H * W)
            taps = []
            for i in range(4):
                yy = torch.minimum(torch.clamp_min(by.view(1, -1, 1) + i, 0), hh - 1)
                for j in range(4):
                    xx = torch.minimum(torch.clamp_min(bx.view(1, 1, -1) + j, 0), ww - 1)
                    idx = (yy * W + xx).expand(B, self.NY, self.NX).reshape(B, -1)
                    taps.append(torch.gather(flat, 1, idx))
            nb.append(torch.stack(taps, -1))           # (B, NY*NX, 16)
        self.nb = torch.cat(nb, -1)                    # (B, NY*NX, 16 * nf)

    def __call__(self, mx, my, items=None):
        """Samples at float32 maps (b, ...) for the items `items` (a slice; all if None)."""
        items = slice(0, self.B) if items is None else items
        nb, hw = self.nb[items], self.hw[items]
        B = nb.shape[0]
        shape = mx.shape
        mxf, myf = mx.reshape(B, -1), my.reshape(B, -1)
        ix = torch.round(mxf * 32).to(torch.int64)
        iy = torch.round(myf * 32).to(torch.int64)
        x0, y0 = (ix >> 5) - 1, (iy >> 5) - 1
        ww, hh = hw[:, 1:2], hw[:, 0:1]
        interior = (x0 >= 0) & (x0 < ww - 3) & (y0 >= 0) & (y0 < hh - 3)
        xb = torch.clamp(x0, -3, self.W - 1) + 3
        yb = torch.clamp(y0, -3, self.H - 1) + 3
        base = yb * self.NX + xb
        vals = nb[torch.arange(B, device=DEV).view(B, 1), base]           # (B, N, 16*nf)
        tab = _cubic_tab()
        cy, cx = tab[iy & 31], tab[ix & 31]                              # (B, N, 4)
        wts = (cy[..., :, None] * cx[..., None, :]).reshape(B, -1, 16)   # cv2's table entries
        outs = []
        for f in range(self.nf):
            v = vals[..., 16 * f:16 * (f + 1)]
            t = v * wts
            rows = [((t[..., 4 * i] + t[..., 4 * i + 1]) + t[..., 4 * i + 2]) + t[..., 4 * i + 3]
                    for i in range(4)]
            total = ((rows[0] + rows[1]) + rows[2]) + rows[3]
            seq = t[..., 0]
            for q in range(1, 16):
                seq = seq + t[..., q]
            outs.append(torch.where(interior, total, seq).reshape(shape))
        return outs


def remap_nearest(src, mx, my):
    """cv2.remap(..., INTER_NEAREST, BORDER_CONSTANT): cvRound of the map, 0 outside."""
    squeeze = src.dim() == 3
    if squeeze:
        src = src[..., None]
    B = src.shape[0]
    if mx.dim() == 2:
        mx = mx[None].expand(B, -1, -1)
        my = my[None].expand(B, -1, -1)
    out = _taps(src, torch.round(mx).to(torch.int64), torch.round(my).to(torch.int64),
                'constant')
    return out[..., 0] if squeeze else out


# --------------------------------------------------------------------------- filters
_KERNELS = {}


def _gauss_kernel(sigma):
    import cv2
    ks = int(round(sigma * 4 * 2 + 1)) | 1
    key = (ks, float(sigma))
    if key not in _KERNELS:
        _KERNELS[key] = torch.as_tensor(cv2.getGaussianKernel(ks, sigma, cv2.CV_32F)[:, 0],
                                        device=DEV)
    return _KERNELS[key]


def _reflect101_index(n, r):
    i = np.arange(-r, n + r)
    i = np.abs(i)
    i = np.where(i >= n, 2 * (n - 1) - i, i)
    return torch.as_tensor(i, device=DEV)


def gaussian_blur(x, sigma):
    """cv2.GaussianBlur(x, (0, 0), sigma) on float32 (B, h, w), BORDER_REFLECT_101.

    Same kernel (cv2's own), same border, symmetric taps summed centre-out. Agrees with cv2
    to float epsilon, not bit for bit: cv2's SIMD rows use FMA.
    """
    k = _gauss_kernel(sigma)
    r = k.numel() // 2
    B, h, w = x.shape
    xi = _reflect101_index(w, r)
    xp = x[:, :, xi]
    row = k[r] * xp[:, :, r:r + w]
    for i in range(1, r + 1):
        row = row + k[r + i] * (xp[:, :, r - i:r - i + w] + xp[:, :, r + i:r + i + w])
    yi = _reflect101_index(h, r)
    yp = row[:, yi, :]
    col = k[r] * yp[:, r:r + h, :]
    for i in range(1, r + 1):
        col = col + k[r + i] * (yp[:, r - i:r - i + h, :] + yp[:, r + i:r + i + h, :])
    return col


def box_blur(x, k=5):
    """cv2.blur(x, (k, k)): normalised box, BORDER_REFLECT_101, float. (B, h, w[, C])."""
    squeeze = x.dim() == 3
    if squeeze:
        x = x[..., None]
    B, h, w, C = x.shape
    r = k // 2
    xi = _reflect101_index(w, r)
    yi = _reflect101_index(h, r)
    xp = x[:, :, xi]
    row = xp[:, :, 0:w]
    for i in range(1, k):
        row = row + xp[:, :, i:i + w]
    yp = row[:, yi]
    col = yp[:, 0:h]
    for i in range(1, k):
        col = col + yp[:, i:i + h]
    out = col * (1.0 / (k * k)) if x.dtype == F64 else col * float(np.float32(1.0 / (k * k)))
    return out[..., 0] if squeeze else out


def shift2d(x, dy, dx, fill):
    """out[y, x] = in[y + dy, x + dx], `fill` outside. x (B, h, w)."""
    B, h, w = x.shape
    out = torch.full_like(x, fill)
    ys = slice(max(0, dy), h + min(0, dy)); yd = slice(max(0, -dy), h + min(0, -dy))
    xs = slice(max(0, dx), w + min(0, dx)); xd = slice(max(0, -dx), w + min(0, -dx))
    out[:, yd, xd] = x[:, ys, xs]
    return out


def dilate(mask, kernel):
    """cv2.dilate(mask, kernel) for a bool batch (B, h, w); anchor at the kernel centre and
    nothing outside the image, as cv2's default border does for dilation."""
    kernel = np.asarray(kernel, bool)
    kh, kw = kernel.shape
    ay, ax = kh // 2, kw // 2
    out = torch.zeros_like(mask)
    for j in range(kh):
        for i in range(kw):
            if kernel[j, i]:
                out |= shift2d(mask, j - ay, i - ax, False)
    return out


def erode(mask, kernel):
    """cv2.erode with the default border (outside counts as set)."""
    kernel = np.asarray(kernel, bool)
    kh, kw = kernel.shape
    ay, ax = kh // 2, kw // 2
    out = torch.ones_like(mask)
    for j in range(kh):
        for i in range(kw):
            if kernel[j, i]:
                out &= shift2d(mask, j - ay, i - ax, True)
    return out


def ellipse_kernel(k):
    import cv2
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)).astype(bool)


_CHAMFER = {3: [(0.955, ((0, 1), (1, 0), (0, -1), (-1, 0))),
                (1.3693, ((1, 1), (1, -1), (-1, 1), (-1, -1)))],
            5: [(1.0, ((0, 1), (1, 0), (0, -1), (-1, 0))),
                (1.4, ((1, 1), (1, -1), (-1, 1), (-1, -1))),
                (2.1969, ((1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2),
                          (-2, -1)))]}
FAR = float(np.finfo(np.float32).max)


def distance_chamfer(mask, mask_size):
    """cv2.distanceTransform(mask, DIST_L2, 3 or 5): chamfer distance to the nearest ZERO.

    cv2 runs two raster passes in float; this relaxes to the same shortest paths in float.
    Equal-cost paths can be summed in a different order, so values agree to float epsilon.
    A mask with no zero comes back FLT_MAX everywhere, as cv2's does.
    """
    big = torch.tensor(3e38, dtype=F32, device=DEV)
    d = torch.where(mask, big, torch.zeros((), dtype=F32, device=DEV))
    steps = [(float(np.float32(c)), offs) for c, offs in _CHAMFER[mask_size]]
    for it in range(512):
        nd = d
        for cost, offs in steps:
            for dy, dx in offs:
                nd = torch.minimum(nd, shift2d(d, dy, dx, 3e38) + cost)
        if it % 4 == 3 and torch.equal(nd, d):
            break
        d = nd
    return torch.where(d >= 1e38, torch.tensor(FAR, dtype=F32, device=DEV), d)


def edt_precise(mask, chunk=48):
    """cv2.distanceTransform(mask, DIST_L2, DIST_MASK_PRECISE): exact Euclidean distance of
    every nonzero pixel to the nearest zero, as sqrt of an exact integer square distance."""
    B, H, W = mask.shape
    out = torch.empty((B, H, W), dtype=F32, device=DEV)
    ys = torch.arange(H, device=DEV, dtype=torch.int64)
    xs = torch.arange(W, device=DEV, dtype=torch.int64)
    INF = 1 << 30
    for s in range(0, B, chunk):
        m = mask[s:s + chunk]
        zero = ~m
        yv = ys.view(1, H, 1).expand_as(m)
        above = torch.where(zero, yv, torch.full_like(yv, -INF)).cummax(1).values
        below = torch.where(zero, yv, torch.full_like(yv, INF)).flip(1).cummin(1).values.flip(1)
        g = torch.minimum(yv - above, below - yv).clamp(max=INF).to(F64)
        g2 = torch.where(g >= INF / 2, torch.full_like(g, 1e18), g * g)
        dx2 = ((xs.view(-1, 1) - xs.view(1, -1)) ** 2).to(F64)         # (W, W')
        d2 = (g2[:, :, None, :] + dx2[None, None]).amin(-1)            # (b, H, W)
        out[s:s + chunk] = torch.sqrt(d2.clamp(max=1e30)).to(F32)
    return out


def resize_area_int(x, f):
    """cv2.resize(x, (w/f, h/f), INTER_AREA) for an integer factor, bit exact for f = 4:
    cv2's fast area path adds each block ROW as ((a0 + a1) + a2) + a3 and the rows into the
    running sum in order, in float32, then multiplies by 1/f^2."""
    B, H, W = x.shape
    h, w = H // f, W // f
    blocks = x[:, :h * f, :w * f].reshape(B, h, f, w, f)
    acc = None
    for j in range(f):
        row = None
        for i in range(f):
            v = blocks[:, :, j, :, i]
            row = v if row is None else row + v
        acc = row if acc is None else acc + row
    return acc * float(np.float32(1.0 / (f * f)))


def connected_components(mask, max_iter=4096):
    """8-connected labels, (B, h, w) int64; 0 is background, a component is labelled by
    1 + the linear index of its first pixel in raster order. cv2 numbers components in
    its own scan order; only WHICH pixels share a label is used downstream."""
    B, h, w = mask.shape
    big = h * w + 10
    lin = torch.arange(1, h * w + 1, device=DEV, dtype=torch.int64).view(1, h, w).expand(B, -1, -1)
    lab = torch.where(mask, lin, torch.full_like(lin, big))
    for it in range(max_iter):
        nb = lab
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy or dx:
                    nb = torch.minimum(nb, shift2d(lab, dy, dx, big))
        nb = torch.where(mask, nb, torch.full_like(nb, big))
        # pointer jumping: follow each label to its own label's label
        flat = nb.reshape(B, -1)
        idx = (flat.clamp(max=h * w) - 1).clamp(min=0)
        jumped = torch.gather(flat, 1, idx)
        jumped = torch.where(flat >= big, flat, torch.minimum(flat, jumped)).view(B, h, w)
        if torch.equal(jumped, lab):
            break
        lab = jumped
    return torch.where(mask, lab, torch.zeros_like(lab))


# --------------------------------------------------------------------------- statistics
def median(x, mask=None, dim=-1):
    """numpy's median over `dim` of the selected entries: the mean of the two middle values
    in the array's own dtype. Entries outside `mask` are ignored. Returns (value, count)."""
    if mask is not None:
        big = torch.tensor(float('inf'), dtype=x.dtype, device=DEV)
        x = torch.where(mask, x, big)
        n = mask.sum(dim)
    else:
        n = torch.full(x.shape[:dim % x.dim()] + x.shape[dim % x.dim() + 1:], x.shape[dim],
                       device=DEV, dtype=torch.int64)
    s = x.sort(dim=dim).values
    lo = ((n - 1).clamp(min=0) // 2).unsqueeze(dim)
    hi = (n // 2).clamp(max=x.shape[dim] - 1).unsqueeze(dim)
    a = torch.gather(s, dim, lo).squeeze(dim)
    b = torch.gather(s, dim, hi).squeeze(dim)
    return (a + b) / dev_scalar(2.0, x.dtype), n


def nanmedian0(x):
    """np.nanmedian(x, axis=0) for float64: mean of the two middle non-NaN values; NaN
    where a column has none."""
    s = x.sort(dim=0).values                      # NaN sorts last
    n = (~torch.isnan(x)).sum(0)
    lo = ((n - 1).clamp(min=0) // 2)[None]
    hi = (n // 2).clamp(max=x.shape[0] - 1)[None]
    a = torch.gather(s, 0, lo)[0]
    b = torch.gather(s, 0, hi)[0]
    out = (a + b) / dev_scalar(2.0, x.dtype)
    return torch.where(n > 0, out, torch.full_like(out, float('nan')))


def percentile(x, mask, q):
    """np.percentile(x[mask], q) with q a Python number, float32, per batch item (last dim
    flattened). numpy casts the interpolation weight weakly, so the lerp is in float32."""
    B = x.shape[0]
    flat = x.reshape(B, -1)
    m = mask.reshape(B, -1)
    big = torch.tensor(float('inf'), dtype=x.dtype, device=DEV)
    s = torch.where(m, flat, big).sort(dim=1).values
    n = m.sum(1)
    out = torch.full((B,), float('nan'), dtype=x.dtype, device=DEV)
    qf = q / 100
    ns = n.cpu().numpy()
    lo_l, hi_l, g_l = [], [], []
    for c in ns:
        vi = int(c) * qf + (1.0 + qf * (1 - 1.0 - 1.0)) - 1
        lo = int(np.floor(vi)) if c > 0 else 0
        lo_l.append(min(max(lo, 0), max(int(c) - 1, 0)))
        hi_l.append(min(lo + 1, max(int(c) - 1, 0)))
        g_l.append(vi - lo)
    lo_t = torch.as_tensor(lo_l, device=DEV).view(B, 1)
    hi_t = torch.as_tensor(hi_l, device=DEV).view(B, 1)
    a = torch.gather(s, 1, lo_t)[:, 0]
    b = torch.gather(s, 1, hi_t)[:, 0]
    d = b - a
    g = torch.as_tensor(np.float32(g_l), device=DEV)
    one_m = torch.as_tensor(np.float32([1 - v for v in g_l]), device=DEV)
    lerp = torch.where(torch.as_tensor(np.array(g_l) >= 0.5, device=DEV), b - d * one_m,
                       a + d * g)
    return torch.where(n > 0, lerp, out), n


# --------------------------------------------------------------------------- colour
_POW_EXP = float(np.float32(1 / 2.4))


def srgb(x):
    """color_pipeline.linear_to_srgb for float32 (or float64) tensors."""
    x = torch.clamp_min(x, 0)
    if x.dtype == F32:
        p = torch.pow(x.to(F64), _POW_EXP).to(F32)
    else:
        p = torch.pow(x, 1 / 2.4)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * p - 0.055)


def rgb8(x):
    """eye_reflection_repair.rgb8: rint(clip(srgb(float32), 0, 1) * 255) -> uint8."""
    return torch.round(torch.clamp(srgb(x.to(F32)), 0, 1) * 255).to(torch.uint8)


_LUT8 = None


def rgb8_half(x16):
    """rgb8 of a HALF-float image, bit exact: numpy's own rgb8 over all 65536 halves."""
    global _LUT8
    if _LUT8 is None:
        import eye_reflection_repair as r32
        allh = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float32)
        with np.errstate(invalid='ignore'):
            lut = r32.rgb8(np.nan_to_num(allh, nan=0.0))
        _LUT8 = torch.as_tensor(lut, device=DEV)
    idx = x16.contiguous().view(torch.int16).to(torch.int64) & 0xFFFF
    return _LUT8[idx]


_SRGB_LUT = None


def srgb_half(x16):
    """color_pipeline.linear_to_srgb of a HALF-float tensor, float32, bit exact: numpy's own
    result for each of the 65536 halves."""
    global _SRGB_LUT
    if _SRGB_LUT is None:
        from color_pipeline import linear_to_srgb
        allh = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float32)
        with np.errstate(invalid='ignore', over='ignore'):
            lut = linear_to_srgb(np.nan_to_num(allh, nan=0.0, posinf=0.0, neginf=0.0))
        _SRGB_LUT = torch.as_tensor(lut.astype(np.float32), device=DEV)
    idx = x16.contiguous().view(torch.int16).to(torch.int64) & 0xFFFF
    return _SRGB_LUT[idx]


def interp(x, xp, fp):
    """np.interp(x, xp, fp) for float64: x (B, ...), xp (K,) shared, fp (B, K)."""
    B = x.shape[0]
    K = xp.numel()
    flat = x.reshape(B, -1)
    j = torch.searchsorted(xp.contiguous(), flat.contiguous(), right=True) - 1
    j = j.clamp(0, K - 2)
    x0 = xp[j]
    x1 = xp[j + 1]
    y0 = torch.gather(fp, 1, j)
    y1 = torch.gather(fp, 1, j + 1)
    slope = (y1 - y0) / (x1 - x0)
    res = slope * (flat - x0) + y0
    res = torch.where(flat == x0, y0, res)
    res = torch.where(flat < xp[0], fp[:, :1].expand_as(res), res)
    res = torch.where(flat >= xp[-1], fp[:, -1:].expand_as(res), res)
    return res.view(x.shape)


# --------------------------------------------------------------------------- self test
def selftest():
    import cv2
    from color_pipeline import linear_to_srgb
    rng = np.random.default_rng(7)
    report = {}
    B, h, w = 6, 43, 42
    src = rng.random((B, h, w, 3), dtype=np.float32)
    mx = rng.uniform(-3, 45, (B, h, w)).astype(np.float32)
    my = rng.uniform(-3, 45, (B, h, w)).astype(np.float32)
    for border, flag in (('constant', cv2.BORDER_CONSTANT), ('replicate', cv2.BORDER_REPLICATE)):
        ref = np.stack([cv2.remap(src[b], mx[b], my[b], cv2.INTER_LINEAR, borderMode=flag)
                        for b in range(B)])
        got = remap_linear(as_t(src), as_t(mx), as_t(my), border).cpu().numpy()
        report['remap_linear_' + border + '_mismatch'] = int((ref != got).sum())
    # separable maps onto a 121-cell grid, as the atlas registration uses
    t = np.linspace(-1.25, 1.25, 121).astype(np.float32)
    grid = rng.random((3, 121, 121, 2), dtype=np.float32)
    sx = rng.uniform(0.95, 1.05, (3, 5)).astype(np.float32)
    tx = rng.uniform(-0.1, 0.1, (3, 5)).astype(np.float32)
    ty = rng.uniform(-0.1, 0.1, (3, 5)).astype(np.float32)
    step = np.float32(2.5 / 120)
    mxs = ((t[None, None, :] * sx[..., None] + tx[..., None]) / step + np.float32(60)).astype(np.float32)
    mys = ((t[None, None, :] * sx[..., None] + ty[..., None]) / step + np.float32(60)).astype(np.float32)
    bad = 0
    got = remap_separable(as_t(grid), as_t(mxs), as_t(mys)).cpu().numpy()
    for b in range(3):
        for k in range(5):
            fx = np.broadcast_to(mxs[b, k][None, :], (121, 121)).astype(np.float32)
            fy = np.broadcast_to(mys[b, k][:, None], (121, 121)).astype(np.float32)
            ref = cv2.remap(grid[b], np.ascontiguousarray(fx), np.ascontiguousarray(fy),
                            cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            bad += int((ref != got[b, k]).sum())
    report['remap_separable_mismatch'] = bad
    img2 = rng.random((B, 50, 57), dtype=np.float32)
    mx2 = rng.uniform(-3, 59, (B, 30, 40)).astype(np.float32)
    my2 = rng.uniform(-3, 52, (B, 30, 40)).astype(np.float32)
    ref = np.stack([cv2.remap(img2[b], mx2[b], my2[b], cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE) for b in range(B)])
    got = remap_cubic(as_t(img2), as_t(mx2), as_t(my2)).cpu().numpy()
    report['remap_cubic_replicate_mismatch'] = int((ref != got).sum())
    canvas = np.zeros((B, 60, 64), np.float32)
    canvas[:, :50, :57] = img2
    got = remap_cubic(as_t(canvas), as_t(mx2), as_t(my2),
                      hw=as_t(np.array([[50, 57]] * B))).cpu().numpy()
    report['remap_cubic_canvas_mismatch'] = int((ref != got).sum())
    ref = np.stack([cv2.GaussianBlur(img2[b], (7, 7), 1.0, borderType=cv2.BORDER_REPLICATE)
                    for b in range(B)])
    got = gaussian_k(as_t(img2), 7, 1.0, 'replicate').cpu().numpy()
    report['gauss7_replicate_maxdiff'] = float(np.abs(ref - got).max())
    for wdt in (57, 61, 64, 70, 99):
        im = rng.random((B, 41, wdt), dtype=np.float32)
        ref = np.stack([cv2.GaussianBlur(im[b], (7, 7), 1.0, borderType=cv2.BORDER_REPLICATE)
                        for b in range(B)])
        got = cv_gaussian(as_t(im), 7, 1.0, 'replicate').cpu().numpy()
        report['cv_gauss7_w%d_mismatch' % wdt] = int((ref != got).sum())
        # eyelid_fit's specular map: a 0/1 mask, (0, 0) kernel at sigma 0.8, default border
        mk = (im > 0.7).astype(np.float32)
        ref = np.stack([cv2.GaussianBlur(mk[b], (0, 0), 0.8) for b in range(B)])
        got = cv_gaussian(as_t(mk), 7, 0.8, 'reflect101').cpu().numpy()
        report['cv_gauss_s08_w%d_mismatch' % wdt] = int((ref != got).sum())
        for dx, dy in ((1, 0), (0, 1)):
            ref = np.stack([cv2.Sobel(im[b], cv2.CV_32F, dx, dy, ksize=3) for b in range(B)])
            got = cv_sobel3(as_t(im), dx, dy).cpu().numpy()
            report['cv_sobel_%d%d_w%d_mismatch' % (dx, dy, wdt)] = int((ref != got).sum())
        s3 = rng.random((B, 41, wdt, 3), dtype=np.float32)
        L = np.float32([0.2126, 0.7152, 0.0722])
        ref = s3 @ L
        got = luma_dot_np(as_t(s3), L).cpu().numpy()
        report['luma_dot_w%d_mismatch' % wdt] = int((ref != got).sum())
    samp = CubicSampler([as_t(canvas), as_t(canvas * 2)], as_t(np.array([[50, 57]] * B)))
    g1, g2 = samp(as_t(mx2), as_t(my2))
    ref = np.stack([cv2.remap(img2[b], mx2[b], my2[b], cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE) for b in range(B)])
    ref2 = np.stack([cv2.remap(img2[b] * 2, mx2[b], my2[b], cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE) for b in range(B)])
    report['cubic_sampler_mismatch'] = int((ref != g1.cpu().numpy()).sum()
                                           + (ref2 != g2.cpu().numpy()).sum())
    ref = np.stack([cv2.Sobel(img2[b], cv2.CV_32F, 0, 1, ksize=3) for b in range(B)])
    got = sobel3(as_t(img2), 0, 1).cpu().numpy()
    report['sobel_dy_maxdiff'] = float(np.abs(ref - got).max())
    ref = np.stack([cv2.Sobel(img2[b], cv2.CV_32F, 1, 0, ksize=3) for b in range(B)])
    got = sobel3(as_t(img2), 1, 0).cpu().numpy()
    report['sobel_dx_maxdiff'] = float(np.abs(ref - got).max())
    ref = np.stack([cv2.remap(src[b, ..., 0], mx[b], my[b], cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT) for b in range(B)])
    got = remap_nearest(as_t(src[..., 0]), as_t(mx), as_t(my)).cpu().numpy()
    report['remap_nearest_mismatch'] = int((ref != got).sum())
    img = rng.random((B, h, w), dtype=np.float32)
    for sigma in (0.7, 1.0, 1.5):
        ref = np.stack([cv2.GaussianBlur(img[b], (0, 0), sigma) for b in range(B)])
        got = gaussian_blur(as_t(img), sigma).cpu().numpy()
        report['blur_%.1f_maxdiff' % sigma] = float(np.abs(ref - got).max())
    ref = np.stack([cv2.blur(src[b], (5, 5)) for b in range(B)])
    got = box_blur(as_t(src), 5).cpu().numpy()
    report['box5_maxdiff'] = float(np.abs(ref - got).max())
    m = rng.random((B, h, w)) > 0.1
    for ms in (3, 5):
        ref = np.stack([cv2.distanceTransform(m[b].astype(np.uint8), cv2.DIST_L2, ms)
                        for b in range(B)])
        got = distance_chamfer(as_t(m), ms).cpu().numpy()
        report['dt%d_maxdiff' % ms] = float(np.abs(ref - got).max())
    fine = rng.random((B, 4 * h, 4 * w)) > 0.02
    ref = np.stack([cv2.distanceTransform(fine[b].astype(np.uint8), cv2.DIST_L2,
                                          cv2.DIST_MASK_PRECISE) for b in range(B)])
    got = edt_precise(as_t(fine)).cpu().numpy()
    report['edt_precise_maxdiff'] = float(np.abs(ref - got).max())
    ref = np.stack([cv2.resize(ref[b] / 4, (w, h), interpolation=cv2.INTER_AREA)
                    for b in range(B)])
    got = resize_area_int(as_t((got / 4).astype(np.float32)), 4).cpu().numpy()
    report['resize_area_maxdiff'] = float(np.abs(ref - got).max())
    for k in (3, 5, 9):
        kern = ellipse_kernel(k)
        ref = np.stack([cv2.dilate(m[b].astype(np.uint8), kern.astype(np.uint8)) > 0
                        for b in range(B)])
        got = dilate(as_t(m), kern).cpu().numpy()
        report['dilate_ellipse%d_mismatch' % k] = int((ref != got).sum())
    ref = np.stack([cv2.erode(m[b].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
                    for b in range(B)])
    report['erode3_mismatch'] = int((ref != erode(as_t(m), np.ones((3, 3))).cpu().numpy()).sum())
    cm = rng.random((B, h, w)) > 0.55
    got = connected_components(as_t(cm)).cpu().numpy()
    bad = 0
    for b in range(B):
        n, lab = cv2.connectedComponents(cm[b].astype(np.uint8), connectivity=8)
        # same partition: a bijection between labels
        pairs = set(zip(lab[cm[b]].tolist(), got[b][cm[b]].tolist()))
        bad += (len(pairs) != n - 1) or (len({p[0] for p in pairs}) != len({p[1] for p in pairs}))
    report['ccl_partition_mismatch_items'] = int(bad)
    vals = (rng.random((B, 500)) ** 2).astype(np.float32)
    msk = rng.random((B, 500)) > 0.3
    ref = np.array([np.median(vals[b][msk[b]]) for b in range(B)], np.float32)
    got = median(as_t(vals), as_t(msk))[0].cpu().numpy()
    report['median_f32_mismatch'] = int((ref != got).sum())
    ref = np.array([np.percentile(vals[b][msk[b]], 90) for b in range(B)], np.float32)
    got = percentile(as_t(vals), as_t(msk), 90)[0].cpu().numpy()
    report['percentile90_mismatch'] = int((ref != got).sum())
    x = (rng.random(1_000_000) ** 3).astype(np.float32)
    ref = linear_to_srgb(x)
    got = srgb(as_t(x)).cpu().numpy()
    report['srgb_lastbit_mismatch_frac'] = float((ref != got).mean())
    import eye_reflection_repair as r32
    x16 = (rng.random((64, 64, 3)) * 1.3).astype(np.float16)
    ref = r32.rgb8(x16.astype(np.float32))
    got = rgb8_half(as_t(x16)).cpu().numpy()
    report['rgb8_half_mismatch'] = int((ref != got).sum())
    got2 = rgb8(as_t(x16.astype(np.float32))).cpu().numpy()
    report['rgb8_float_mismatch'] = int((ref != got2).sum())
    xp = np.linspace(0, 1, 17)
    fp = rng.random((B, 17))
    xs = rng.uniform(-0.2, 1.2, (B, 100))
    ref = np.stack([np.interp(xs[b], xp, fp[b]) for b in range(B)])
    got = interp(as_t(xs), as_t(xp), as_t(fp)).cpu().numpy()
    report['interp_maxdiff'] = float(np.abs(ref - got).max())
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        import json
        print(json.dumps(selftest(), indent=1))
