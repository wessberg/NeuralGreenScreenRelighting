"""The limbus and eyelid fits on the GPU, batched over eyes: `limbus_fit.py --device cuda`
and `eyelid_fit.py --device cuda`.

Round 39, after the eye pass: the two trackers were the rest of the CPU time per clip
(limbus 3 min, eyelid 67 s on the 1200-frame reporter clip, single-threaded numpy). Same
construction, same outputs; this module holds only the per-pixel and per-sample maths.

**Both write the CPU's track exactly** (reporter_full: every row of limbus_track.json,
eyelid_track_w.json and the --update eyelid_track_wt.json identical). Every primitive the
samples go through is cv2's or numpy's own arithmetic, pinned by probe and held by
`eye_gpu.py --selftest`: the bicubic remap (1/32 fixed-point map, a = -0.75 table, 16 taps
row by row inside the image and one at a time at its border), the Gaussian and Sobel
fields (AVX2 FMA chains, plain scalar tail), the luma dot (OpenBLAS gemv order), numpy's
float32 pairwise mean and percentile. The small robust fits (circle, centre, parabola;
~60 points an eye) run afterwards in the CPU scripts' own numpy functions: batched on the
GPU they agreed to 1e-15 except where an IRLS step is rank-deficient, and there the CPU's
answer is a ratio of rounding residues no reimplementation reproduces.
"""
import numpy as np
import torch

import eye_gpu as G
from eye_gpu import DEV, F32, F64, as_t
import eye_reflection_repair as r32

LUMA = (0.2126, 0.7152, 0.0722)


def display_luma(patch16):
    """limbus_fit.display_luma: clip(srgb(linear), 0, 1) @ LUMA, float32 (B, h, w), bit
    exact for the HALF-float frames every consumer reads (sRGB by numpy's own table, the dot
    in numpy's BLAS order)."""
    s = torch.clamp(G.srgb_half(patch16.half()), 0, 1)
    return G.luma_dot_np(s, LUMA)


def gradient_field(luma):
    """limbus_fit.gradient_field: 7x7 sigma-1 blur (BORDER_REPLICATE), Sobel / 8 -- cv2's
    arithmetic, bit exact."""
    blur = G.cv_gaussian(luma, 7, 1.0, 'replicate')
    return (G.cv_sobel3(blur, 1, 0) / G.dev_scalar(8.0),
            G.cv_sobel3(blur, 0, 1) / G.dev_scalar(8.0), blur)


def pairwise_mean(v, n):
    """numpy's float32 mean over the last axis of the first `n` entries (per leading item),
    `v` (B, ..., K) with the kept entries compacted to the front; n (B,) int, 8 <= n <= 128."""
    B = v.shape[0]
    K = v.shape[-1]
    nb = n.view(B, *([1] * (v.dim() - 2)))
    full = nb - nb % 8
    pad = (-K) % 8
    if pad:
        v = torch.cat([v, torch.zeros(*v.shape[:-1], pad, dtype=v.dtype, device=DEV)], -1)
        K = v.shape[-1]
    blocks = v.reshape(*v.shape[:-1], K // 8, 8)
    r = blocks[..., 0, :].clone()                      # the 8 accumulators, lanes last
    lane = torch.arange(8, device=DEV)
    for blk in range(1, K // 8):
        add = (blk * 8 + lane) < full[..., None]
        r = torch.where(add, r + blocks[..., blk, :], r)
    res = ((r[..., 0] + r[..., 1]) + (r[..., 2] + r[..., 3])) + \
          ((r[..., 4] + r[..., 5]) + (r[..., 6] + r[..., 7]))
    base = full
    for j in range(7):
        idx = (base + j).clamp(max=K - 1)
        val = torch.gather(v, v.dim() - 1, idx.expand(*v.shape[:-1]).unsqueeze(-1))[..., 0]
        res = torch.where((base + j) < nb, res + val, res)
    return res / nb.to(F32)


def compact(mask):
    """Order-preserving indices of the True entries first: (B, K) -> (B, K) int64, count."""
    B, K = mask.shape
    key = torch.where(mask, torch.arange(K, device=DEV).view(1, K),
                      torch.arange(K, device=DEV).view(1, K) + K)
    return key.sort(1).indices, mask.sum(1)


def median_f32(x, m, dim=-1):
    return G.median(x, m, dim=dim)[0]


class _Win:
    """What eye_removal_gpu.lid_distance needs of an eye: polygon and window origin."""
    __slots__ = ('poly', 'x0', 'y0')

    def __init__(self, poly, x0, y0):
        self.poly, self.x0, self.y0 = poly, x0, y0


class _Grp:
    def __init__(self, eyes, h, w):
        self.eyes, self.B, self.h, self.w = eyes, len(eyes), h, w


def _polar_maps(c0x, c0y, x0, y0, rad, cos, sin, off_x=None, off_y=None):
    """Float32 sample points ((c0 + off) + rad * cos) - x0, the order limbus_fit writes
    them in. c0x/x0 (B,), rad (B, R), cos/sin (B, A); off_x/off_y (O,) optional."""
    B = c0x.shape[0]
    f32 = lambda t: t.to(F32)
    rc = f32(rad)[:, :, None] * f32(cos)[:, None, :]          # (B, R, A)
    rs = f32(rad)[:, :, None] * f32(sin)[:, None, :]
    if off_x is None:
        mx = (f32(c0x).view(B, 1, 1) + rc) - f32(x0).view(B, 1, 1)
        my = (f32(c0y).view(B, 1, 1) + rs) - f32(y0).view(B, 1, 1)
    else:
        O = off_x.numel()
        cx = f32(c0x).view(B, 1, 1, 1, 1) + off_x.view(1, 1, O, 1, 1)
        cy = f32(c0y).view(B, 1, 1, 1, 1) + off_y.view(1, O, 1, 1, 1)
        mx = (cx + rc[:, None, None]) - f32(x0).view(B, 1, 1, 1, 1)
        my = (cy + rs[:, None, None]) - f32(y0).view(B, 1, 1, 1, 1)
        mx, my = torch.broadcast_tensors(mx, my)
    return mx.contiguous(), my.contiguous()


def limbus_fields(patch, meta):
    """The per-window fields of one shape group: gradient (gx, gy) and lid distance.
    Computed per shape so every border rule is the window's own."""
    from eye_removal_gpu import lid_distance
    B, h, w = patch.shape[:3]
    luma = display_luma(patch)
    gx, gy, _ = gradient_field(luma)
    lid = lid_distance(_Grp([_Win(m['poly'], m['x0'], m['y0']) for m in meta], h, w))
    return gx, gy, lid


def edge_support(shot, items, points, topology, margin=10, io_workers=12):
    """Daugman's operator -- the limbus fit's own objective -- at GIVEN centres: the mean
    radial derivative of display luma around the circle of the given radius, over the
    angles clear of the eyelid polygon by limbus_fit.LID_MIN (its `valid_angles` gate).

    `items`: list of (frame, side, radius, [centre, ...]). Returns one list of scores per
    item, NaN where fewer than limbus_fit.MIN_ANGLES angles are clear or the window leaves
    the frame. The window is centred between the item's centres, so every circle is scored
    on the same fields."""
    import cv2
    from concurrent.futures import ThreadPoolExecutor
    import limbus_fit as lf
    th = np.arange(lf.ANGLES, dtype=np.float32) * (2 * np.pi / lf.ANGLES)
    cos_t, sin_t = as_t(np.cos(th)), as_t(np.sin(th))
    out = [[float('nan')] * len(it[3]) for it in items]

    def read(i):
        raw = cv2.imread(str(shot.fg(i)), cv2.IMREAD_UNCHANGED)
        return None if raw is None else raw[..., ::-1]

    frames = sorted({it[0] for it in items})
    groups = {}
    with ThreadPoolExecutor(max_workers=io_workers) as pool:
        imgs = dict(zip(frames, pool.map(read, frames)))
    for n, (i, side, radius, centres) in enumerate(items):
        img = imgs.get(i)
        if img is None:
            continue
        H, W = img.shape[:2]
        c = np.asarray(centres, float)
        mid = c.mean(0)
        reach = radius * 1.7 + margin + float(np.hypot(*(c - mid).T).max())
        x0 = int(np.floor(mid[0] - reach)); x1 = int(np.ceil(mid[0] + reach)) + 1
        y0 = int(np.floor(mid[1] - reach)); y1 = int(np.ceil(mid[1] + reach)) + 1
        if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
            continue
        poly = np.asarray(points[i])[topology['eyelid_' + side], :2]
        groups.setdefault((y1 - y0, x1 - x0), []).append(dict(
            n=n, x0=x0, y0=y0, poly=poly, radius=radius, centres=c,
            patch=torch.as_tensor(np.ascontiguousarray(img[y0:y1, x0:x1]), device=DEV).half()))
    for (h, w), lst in groups.items():
        B = len(lst)
        gx, gy, lid = limbus_fields(torch.stack([m['patch'] for m in lst]), lst)
        samp = G.CubicSampler([gx, gy, lid], as_t(np.array([[h, w]] * B)))
        K = max(len(m['centres']) for m in lst)
        for k in range(K):
            cx = as_t(np.float32([m['centres'][min(k, len(m['centres']) - 1)][0] for m in lst]))
            cy = as_t(np.float32([m['centres'][min(k, len(m['centres']) - 1)][1] for m in lst]))
            rad = as_t(np.float32([[m['radius']] for m in lst]))
            x0 = as_t(np.float32([m['x0'] for m in lst]))
            y0 = as_t(np.float32([m['y0'] for m in lst]))
            mx, my = _polar_maps(cx, cy, x0, y0, rad, cos_t.view(1, -1).expand(B, -1),
                                 sin_t.view(1, -1).expand(B, -1))
            sgx, sgy, slid = samp(mx, my)
            keep = slid[:, 0] > lf.LID_MIN
            d = sgx[:, 0] * cos_t + sgy[:, 0] * sin_t
            cnt = keep.sum(1)
            score = (torch.where(keep, d, torch.zeros_like(d)).sum(1).to(F64)
                     / cnt.clamp(min=1).to(F64))
            score = torch.where(cnt >= lf.MIN_ANGLES, score, torch.full_like(score, float('nan')))
            for m, s in zip(lst, score.cpu().numpy()):
                if k < len(m['centres']):
                    out[m['n']][k] = float(s)
    return out


def limbus_group(fields, meta, th_cos, th_sin, radius_given, model='circle', min_arc=0.12,
                 min_angles=14, max_se=0.60, lid_min=1.25, chunk=6):
    """limbus_fit.fit_eye for a batch of eyes of ANY window shapes.

    `fields` = (gx, gy, lid, hw): canvases (B, Hm, Wm) holding each eye's fields in its
    top-left corner and hw (B, 2) its true size, so the replicate border of every sample is
    its own window's. `meta` list of dicts with landmark (np.float32 (2,)), x0, y0, r0
    (python float), radius, rec (the record so far). `radius_given` True for the second
    (shot radius) pass. Fills each rec.
    """
    gx, gy, lid, hw = fields
    B = gx.shape[0]
    A = th_cos.numel()
    cosb = th_cos.view(1, A).expand(B, A)
    sinb = th_sin.view(1, A).expand(B, A)
    c0x = as_t(np.array([m['landmark'][0] for m in meta], np.float32))
    c0y = as_t(np.array([m['landmark'][1] for m in meta], np.float32))
    x0 = as_t(np.array([m['x0'] for m in meta], np.float32))
    y0 = as_t(np.array([m['y0'] for m in meta], np.float32))
    r0 = as_t(np.array([m['r0'] for m in meta], np.float32)).view(B, 1)

    samp_lid = G.CubicSampler([lid], hw)
    samp_g = G.CubicSampler([gx, gy], hw)

    def valid(cx, cy, rr):
        mx, my = _polar_maps(cx, cy, x0, y0, rr, cosb, sinb)
        return samp_lid(mx, my)[0][:, 0, :] > lid_min

    keep = valid(c0x, c0y, r0)
    alive = torch.ones(B, dtype=torch.bool, device=DEV)
    kc = keep.sum(1)
    arc = (kc.to(F64) / A).cpu().numpy()
    bad = ((kc < min_angles) | (kc.to(F64) / A < min_arc)).cpu().numpy()
    for n, m in enumerate(meta):
        if bad[n]:
            m['rec'].update(reason='limbus_occluded', open_arc=float(arc[n]))
    alive &= ~torch.as_tensor(bad, device=DEV)

    def daugman(cx, cy, radii_np, keepm, span, step):
        offs = np.arange(-span, span + 1e-9, step, dtype=np.float32)
        O = len(offs)
        offs_t = as_t(offs)
        R = max(len(r) for r in radii_np)
        rad = np.full((B, R), np.nan, np.float32)
        for n, r in enumerate(radii_np):
            rad[n, :len(r)] = r
        rad_t = as_t(rad)
        order, cnt = compact(keepm)
        Amax = int(cnt.max().item()) if B else 0
        Amax = max(Amax, 8)
        ci = torch.gather(cosb, 1, order[:, :Amax])
        si = torch.gather(sinb, 1, order[:, :Amax])
        best_s = torch.full((B,), -float('inf'), dtype=F32, device=DEV)
        best_j = torch.zeros((B, 3), dtype=torch.int64, device=DEV)
        scores = []
        per = max(1, int(4e6 // (O * O * R * Amax)))
        for s0 in range(0, B, per):
            sl = slice(s0, min(B, s0 + per))
            radc = torch.nan_to_num(rad_t[sl], nan=0.0)
            mx, my = _polar_maps(cx[sl], cy[sl], x0[sl], y0[sl], radc, ci[sl], si[sl],
                                 offs_t, offs_t)
            sgx, sgy = samp_g(mx, my, items=sl)
            d = sgx * ci[sl].view(-1, 1, 1, 1, Amax) + sgy * si[sl].view(-1, 1, 1, 1, Amax)
            sc = pairwise_mean(d, torch.clamp(cnt[sl], min=8))
            sc = torch.where(torch.isnan(rad_t[sl]).view(-1, 1, 1, R), torch.full_like(sc, -float('inf')), sc)
            scores.append(sc)
        sc = torch.cat(scores)                                      # (B, O, O, R)
        flat = sc.reshape(B, -1)
        mx_ = flat.max(1)
        j = mx_.indices
        jy, rest = j // (O * R), j % (O * R)
        jx, jr = rest // R, rest % R
        return offs, jy, jx, jr, mx_.values, rad

    # pass A: global search
    if radius_given is None:
        radii1 = [np.arange(0.70 * m['r0'], 1.30 * m['r0'] + 1e-9, 0.125, dtype=np.float32)
                  for m in meta]
    else:
        radii1 = [np.float32([m['radius']]) for m in meta]
    offs, jy, jx, jr, sc1, rad1 = daugman(c0x, c0y, radii1, keep, 3.0, 0.25)
    jy, jx, jr = jy.cpu().numpy(), jx.cpu().numpy(), jr.cpu().numpy()
    cx1 = [float(np.float32(m['landmark'][0]) + offs[jx[n]]) for n, m in enumerate(meta)]
    cy1 = [float(np.float32(m['landmark'][1]) + offs[jy[n]]) for n, m in enumerate(meta)]
    rr1 = [float(rad1[n, jr[n]]) for n in range(B)]
    cxt = as_t(np.float32(cx1)); cyt = as_t(np.float32(cy1))
    keep2 = valid(cxt, cyt, as_t(np.float32(rr1)).view(B, 1))
    kc2 = keep2.sum(1)
    arc2 = (kc2.to(F64) / A).cpu().numpy()
    bad2 = ((kc2 < min_angles) | (kc2.to(F64) / A < min_arc)).cpu().numpy()
    al = alive.cpu().numpy()
    for n, m in enumerate(meta):
        if al[n] and bad2[n]:
            m['rec'].update(reason='limbus_occluded', open_arc=float(arc2[n]))
    alive &= ~torch.as_tensor(bad2, device=DEV)
    # pass B: fine
    if radius_given is None:
        radii2 = [np.arange(rr1[n] - 0.75, rr1[n] + 0.75 + 1e-9, 0.0625, dtype=np.float32)
                  for n in range(B)]
    else:
        radii2 = [np.float32([m['radius']]) for m in meta]
    offs2, jy, jx, jr, sc2, rad2 = daugman(cxt, cyt, radii2, keep2, 0.75, 0.0625)
    jy, jx, jr = jy.cpu().numpy(), jx.cpu().numpy(), jr.cpu().numpy()
    sc2 = sc2.cpu().numpy()
    cx2 = [float(np.float32(cx1[n]) + offs2[jx[n]]) for n in range(B)]
    cy2 = [float(np.float32(cy1[n]) + offs2[jy[n]]) for n in range(B)]
    rr2 = [float(rad2[n, jr[n]]) for n in range(B)]
    # ray edges
    ss_l = [np.arange(0.72 * rr2[n], 1.32 * rr2[n] + 1e-9, 0.1, dtype=np.float32) for n in range(B)]
    S = max(len(s) for s in ss_l)
    ssn = np.full((B, S), np.nan, np.float32)
    for n, s in enumerate(ss_l):
        ssn[n, :len(s)] = s
    ss_t = as_t(ssn)
    order, cnt = compact(keep2)
    Amax = max(int(cnt.max().item()), 1)
    ci = torch.gather(cosb, 1, order[:, :Amax])
    si = torch.gather(sinb, 1, order[:, :Amax])
    cx2t = as_t(np.float32(cx2)); cy2t = as_t(np.float32(cy2))
    mx, my = _polar_maps(cx2t, cy2t, x0, y0, torch.nan_to_num(ss_t), ci, si)
    sgx, sgy = samp_g(mx, my)
    d = sgx * ci[:, None, :] + sgy * si[:, None, :]                   # (B, S, A)
    lens = as_t(np.array([len(s) for s in ss_l]))
    valid_s = torch.arange(S, device=DEV).view(1, S, 1) < lens.view(B, 1, 1)
    dm = torch.where(valid_s, d, torch.full_like(d, -float('inf')))
    k = dm.argmax(1)                                                   # (B, A)
    peak = torch.gather(d, 1, k[:, None, :])[:, 0]
    inner = torch.minimum(torch.clamp_min(k, 1), (lens - 2).view(B, 1))
    ga = lambda t: torch.gather(d, 1, t[:, None, :])[:, 0]
    a_, b_, c_ = ga(inner - 1), ga(inner), ga(inner + 1)
    denom = (a_ - 2 * b_) + c_
    shift = torch.where(denom.abs() > 1e-9,
                        0.5 * (a_ - c_) / torch.where(denom == 0, torch.ones_like(denom), denom),
                        torch.zeros_like(denom))
    shift = torch.clamp(shift, -1, 1)
    r_edge = torch.gather(torch.nan_to_num(ss_t), 1, inner) + shift * float(np.float32(0.1))
    ok = (k > 0) & (k < (lens - 1).view(B, 1)) & (peak > 0)
    ang_valid = torch.arange(Amax, device=DEV).view(1, Amax) < cnt.view(B, 1)
    ok &= ang_valid
    pmed = median_f32(peak, ok, dim=1)
    anyok = ok.any(1)
    strong = torch.where(anyok.view(B, 1), ok & (peak > 0.25 * pmed.view(B, 1)), ok)
    nstrong = strong.sum(1).cpu().numpy()
    px = (cx2t.view(B, 1) + r_edge * ci)
    py = (cy2t.view(B, 1) + r_edge * si)
    # The robust fit itself runs in numpy -- limbus_fit's own function on the (bit-exact)
    # float32 edge points: ~60 points an eye, milliseconds over the clip, and it makes the
    # track the CPU's exactly. Batched on the GPU it agreed to 1e-15 except where an IRLS
    # step is rank-deficient, and there the CPU's answer is a ratio of rounding residues.
    import limbus_fit as lf
    px_np, py_np = px.cpu().numpy(), py.cpu().numpy()
    strong_np = strong.cpu().numpy()
    al = alive.cpu().numpy()
    for n, m in enumerate(meta):
        rec = m['rec']
        if not al[n]:
            continue
        if nstrong[n] < min_angles:
            rec.update(reason='too_few_edges', edges=int(nstrong[n]), open_arc=float(arc2[n]))
            continue
        sel = strong_np[n]
        pts = np.column_stack([px_np[n][sel], py_np[n][sel]])
        if radius_given is not None:
            fit = lf.fit_centre(pts, (cx2[n], cy2[n]), float(m['radius']))
            if fit is None:
                rec['reason'] = 'centre_fit_failed'
                continue
            rec.update(centre_se_px=round(fit['centre_se_px'], 4), inliers=fit['inliers'])
            if fit['centre_se_px'] > max_se:
                rec.update(reason='centre_underdetermined', open_arc=float(arc2[n]),
                           rms=float(fit['rms']))
                continue
            rec.update(fitted=True, centre=list(fit['centre']),
                       axes=[fit['radius'], fit['radius']])
        else:
            fit = lf.fit_circle(pts, (cx2[n], cy2[n]), rr2[n])
            if fit is None:
                rec['reason'] = 'circle_fit_failed'
                continue
            rec.update(fitted=True, centre=list(fit['centre']),
                       axes=[fit['radius'], fit['radius']])
        rec.update(daugman=[round(cx2[n], 4), round(cy2[n], 4), round(rr2[n], 4),
                            round(float(sc2[n]), 6)],
                   edges=int(nstrong[n]), open_arc=float(arc2[n]), rms=float(fit['rms']),
                   edge_points=[[round(float(a), 3), round(float(b), 3)] for a, b in pts])

# =========================================================================== eyelid
def lid_fields(patch16, meta):
    """eyelid_fit.fit_eye's fields for one window shape: blurred luma, gy, and the
    specular map (for the upper lid), plus `sclera` per eye. Bit exact."""
    B, h, w = patch16.shape[:3]
    luma = display_luma(patch16)
    blur = G.cv_gaussian(luma, 7, 1.0, 'replicate')
    gy = G.cv_sobel3(blur, 0, 1) / G.dev_scalar(8.0)
    lx = as_t(np.array([m['local'][0] for m in meta], np.float64)).view(B, 1, 1)
    ly = as_t(np.array([m['local'][1] for m in meta], np.float64)).view(B, 1, 1)
    rad = as_t(np.array([max(m['radius'], 1e-6) for m in meta], np.float64)).view(B, 1, 1)
    yy = torch.arange(h, device=DEV, dtype=F64).view(1, h, 1)
    xx = torch.arange(w, device=DEV, dtype=F64).view(1, 1, w)
    rho = torch.hypot(xx - lx, yy - ly) / rad
    ring = (rho > 1.15) & (rho < 1.55)
    pct, n = G.percentile(luma, ring, 90)
    sclera = [float(p) if c > 30 else 0.5 for p, c in zip(pct.cpu().numpy(), n.cpu().numpy())]
    thr = as_t(np.float32([max(0.42, s + 0.06) for s in sclera])).view(B, 1, 1)
    spec = G.cv_gaussian(((luma > thr) & (rho < 1.0)).to(F32), 7, 0.8, 'reflect101')
    return blur, gy, spec


def lid_boundary(samp, meta, sign, with_spec, min_step=0.030, column_step=0.5,
                 sample_step=0.1, inner=0.15, outer=1.80):
    """eyelid_fit.lid_boundary for a batch: per eye (xs, out_y, out_s) as padded tensors.

    The walk is the CPU's: the FIRST sample under -min_step outward, then down while the
    gradient keeps falling, then a parabolic refinement -- found for every column at once."""
    B = len(meta)
    xs_l = [np.arange(m['x_lo'], m['x_hi'] + 1e-9, column_step, dtype=np.float32) for m in meta]
    ss_l = [np.arange(inner * m['radius'], outer * m['radius'], sample_step, dtype=np.float32)
            for m in meta]
    C = max(1, max(len(x) for x in xs_l))
    S = max(len(s) for s in ss_l)
    xs = np.full((B, C), np.nan, np.float32)
    ss = np.full((B, S), np.nan, np.float32)
    for n in range(B):
        xs[n, :len(xs_l[n])] = xs_l[n]
        ss[n, :len(ss_l[n])] = ss_l[n]
    xs_t = as_t(xs)
    ss_t = as_t(ss)
    ncol = as_t(np.array([len(x) for x in xs_l]))
    nss = as_t(np.array([len(s) for s in ss_l]))
    cy = as_t(np.float32([m['local'][1] for m in meta])).view(B, 1, 1)
    sg = float(np.float32(sign))
    yy = cy + sg * torch.nan_to_num(ss_t).view(B, S, 1)            # float32, as numpy
    xx = torch.nan_to_num(xs_t).view(B, 1, C)
    yy, xx = torch.broadcast_tensors(yy, xx)
    if with_spec:
        gys, specs = samp(xx.contiguous(), yy.contiguous())
    else:
        gys = samp(xx.contiguous(), yy.contiguous())[0]
    grad = gys * sg
    rows = torch.arange(S, device=DEV).view(1, S, 1)
    vrow = rows < nss.view(B, 1, 1)
    if with_spec:
        hit_spec = (specs > 0.05) & vrow          # padded rows must not spread into real ones
        block = hit_spec.clone()
        for sh in range(1, 6):
            block[:, sh:] |= hit_spec[:, :-sh]
            block[:, :-sh] |= hit_spec[:, sh:]
        # the CPU's shifts stay inside each eye's own ss length; padded rows carry nothing
        grad = torch.where(block & vrow, torch.zeros_like(grad), grad)
    hit = (grad < -min_step) & vrow
    anyhit = hit.any(1)
    first = torch.where(hit, rows.expand_as(hit), torch.full_like(hit, S, dtype=torch.int64)).min(1).values
    L = nss.view(B, 1)
    nxt = torch.cat([grad[:, 1:], torch.full_like(grad[:, :1], float('inf'))], 1)
    stop = ((rows + 1) >= (L.view(B, 1, 1) - 1)) | ~(nxt < grad)
    cand = stop & (rows >= first.view(B, 1, C))
    kk = torch.where(cand, rows.expand_as(cand), torch.full_like(cand, S, dtype=torch.int64)).min(1).values
    good = anyhit & (kk >= 1) & (kk < L - 1)
    kc = kk.clamp(1, S - 2)
    g = lambda t: torch.gather(grad, 1, t.view(B, 1, C))[:, 0]
    a_, b_, c_ = g(kc - 1), g(kc), g(kc + 1)
    denom = (a_ - 2 * b_) + c_
    shift = torch.where(denom.abs() < 1e-9, torch.zeros_like(denom),
                        torch.clamp((0.5 * (a_ - c_)) / torch.where(denom == 0, torch.ones_like(denom), denom), -1, 1))
    ssk = torch.gather(torch.nan_to_num(ss_t), 1, kc)
    step32 = (shift.to(F64) * sample_step).to(F32)                  # python float, cast weakly
    yv = cy.view(B, 1) + sg * (ssk - step32)
    colv = torch.arange(C, device=DEV).view(1, C) < ncol.view(B, 1)
    good &= colv
    out_y = torch.where(good, yv, torch.full_like(yv, float('nan')))
    out_s = torch.where(good, -b_, torch.zeros_like(b_))
    return xs_t, out_y, out_s, colv

