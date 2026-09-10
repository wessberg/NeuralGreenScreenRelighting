"""The limbus and eyelid fits on the GPU, batched over eyes: `limbus_fit.py --device cuda`
and `eyelid_fit.py --device cuda`.

Round 39, after the eye pass: the two trackers were the rest of the CPU time per clip
(limbus 3 min, eyelid 67 s on the 1200-frame reporter clip, single-threaded numpy). Same
construction, same outputs; this module holds only the per-pixel and per-sample maths.

Reproduction, per primitive (`eye_gpu.py --selftest`): the bicubic remap every sample goes
through is BIT EXACT against cv2 (fixed-point 1/32 map, cv2's a = -0.75 table, 16 taps
summed row by row inside the image and one at a time at its border); the Gaussian and
Sobel gradient field agree to 5e-7. numpy's float32 mean over angles is its 8-accumulator
pairwise sum, reproduced. The small robust fits are float64 on both sides. End to end,
`validate_track_gpu.py` compares the written tracks field by field.
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


def fit_centre(pts, mask, c0, radius, iterations=8, tukey=2.5):
    """limbus_fit.fit_centre, batched: pts (B, N, 2) float64, mask (B, N), c0 (B, 2),
    radius (B,). Returns dict of tensors and ok (B,)."""
    B, N, _ = pts.shape
    c = c0.clone()
    done = torch.zeros(B, dtype=torch.bool, device=DEV)
    ok = torch.ones(B, dtype=torch.bool, device=DEV)
    R = radius.view(B, 1)
    for _ in range(iterations):
        d = pts - c[:, None, :]
        rho = torch.clamp_min(torch.hypot(d[..., 0], d[..., 1]), 1e-9)
        u = d / rho[..., None]
        resid = rho - R
        med, _ = G.median(resid, mask, dim=1)
        mad, _ = G.median((resid - med[:, None]).abs(), mask, dim=1)
        scale = torch.clamp_min(1.4826 * mad, 0.05)
        t = torch.clamp(resid / (tukey * scale[:, None]), -1, 1)
        w = torch.where(mask, (1 - t ** 2) ** 2, torch.zeros_like(t))
        J = -u
        M = (J[..., :, None] * (w[..., None, None] * J[..., None, :])).sum(1)
        rhs = -(J * (w * resid)[..., None]).sum(1)
        det = M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]
        # LAPACK meets an exactly zero pivot when fewer points carry weight than there are
        # parameters; numpy raises there and the CPU fit stops. A GPU determinant of the same
        # rank-deficient matrix comes out tiny but not zero, so the count is the test.
        good = (det != 0) & ((w > 0).sum(1) >= 2)
        Mi = torch.where(good.view(B, 1, 1), M, torch.eye(2, dtype=F64, device=DEV).expand(B, 2, 2))
        step = torch.linalg.solve(Mi, rhs)
        upd = ~done & good
        c = torch.where(upd[:, None], c + step, c)
        ok &= torch.isfinite(c).all(1)
        done |= ~good | (step.abs().max(1).values < 1e-4)
    d = pts - c[:, None, :]
    resid = torch.hypot(d[..., 0], d[..., 1]) - R
    med, _ = G.median(resid, mask, dim=1)
    mad, _ = G.median((resid - med[:, None]).abs(), mask, dim=1)
    scale = torch.clamp_min(1.4826 * mad, 0.05)
    w = torch.where(mask, (1 - torch.clamp(resid / (tukey * scale[:, None]), -1, 1) ** 2) ** 2,
                    torch.zeros_like(resid))
    ok &= w.sum(1) > 1e-9
    rms = torch.sqrt((w * resid ** 2).sum(1) / torch.clamp_min(w.sum(1), 1e-300))
    u = d / torch.clamp_min(torch.hypot(d[..., 0], d[..., 1]), 1e-9)[..., None]
    info = (u[..., :, None] * (w[..., None, None] * u[..., None, :])).sum(1)
    eig = torch.linalg.eigvalsh(info)
    se = rms / torch.sqrt(torch.clamp_min(eig[:, 0], 1e-9))
    inl = ((resid.abs() < 2 * scale[:, None]) & mask).sum(1)
    return dict(centre=c, rms=rms, se=se, inliers=inl), ok


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
    pts = torch.stack([px, py], -1).to(F64)
    c0 = torch.stack([as_t(np.array(cx2, np.float64)), as_t(np.array(cy2, np.float64))], 1)
    al = alive.cpu().numpy()
    if radius_given is not None or model == 'circle':
        if radius_given is not None:
            fit, fok = fit_centre(pts, strong, c0, as_t(np.array([m['radius'] for m in meta], np.float64)))
        else:
            fit, fok = fit_circle(pts, strong, c0, as_t(np.array(rr2, np.float64)))
    fit = {k_: v.cpu().numpy() for k_, v in fit.items()}
    fok = fok.cpu().numpy()
    pts_np = pts.cpu().numpy()
    strong_np = strong.cpu().numpy()
    arc2n = arc2
    for n, m in enumerate(meta):
        rec = m['rec']
        if not al[n]:
            continue
        if nstrong[n] < min_angles:
            rec.update(reason='too_few_edges', edges=int(nstrong[n]), open_arc=float(arc2n[n]))
            continue
        if radius_given is not None:
            if not fok[n]:
                rec['reason'] = 'centre_fit_failed'
                continue
            rec.update(centre_se_px=round(float(fit['se'][n]), 4), inliers=int(fit['inliers'][n]))
            if fit['se'][n] > max_se:
                rec.update(reason='centre_underdetermined', open_arc=float(arc2n[n]),
                           rms=float(fit['rms'][n]))
                continue
            rec.update(fitted=True, centre=[float(fit['centre'][n, 0]), float(fit['centre'][n, 1])],
                       axes=[float(m['radius']), float(m['radius'])])
        else:
            if not fok[n]:
                rec['reason'] = 'circle_fit_failed'
                continue
            rec.update(fitted=True, centre=[float(fit['centre'][n, 0]), float(fit['centre'][n, 1])],
                       axes=[float(fit['radius'][n]), float(fit['radius'][n])])
        p = pts_np[n][strong_np[n]]
        rec.update(daugman=[round(cx2[n], 4), round(cy2[n], 4), round(rr2[n], 4),
                            round(float(sc2[n]), 6)],
                   edges=int(nstrong[n]), open_arc=float(arc2n[n]), rms=float(fit['rms'][n]),
                   edge_points=[[round(float(a), 3), round(float(b), 3)] for a, b in p])


def fit_circle(pts, mask, c0, r0, iterations=6, tukey=2.5):
    """limbus_fit.fit_circle, batched (radius free)."""
    B, N, _ = pts.shape
    c = c0.clone()
    r = r0.clone()
    done = torch.zeros(B, dtype=torch.bool, device=DEV)
    ok = torch.ones(B, dtype=torch.bool, device=DEV)
    w = mask.to(F64)
    for _ in range(iterations):
        d = pts - c[:, None, :]
        rho = torch.hypot(d[..., 0], d[..., 1])
        rho = torch.where(rho < 1e-9, torch.full_like(rho, 1e-9), rho)
        u = d / rho[..., None]
        resid = rho - r[:, None]
        med, _ = G.median(resid, mask, dim=1)
        mad, _ = G.median((resid - med[:, None]).abs(), mask, dim=1)
        scale = torch.clamp_min(1.4826 * mad, 0.05)
        t = torch.clamp(resid / (tukey * scale[:, None]), -1, 1)
        wn = torch.where(mask, (1 - t ** 2) ** 2, torch.zeros_like(t))
        w = torch.where(done[:, None], w, wn)
        J = torch.stack([-u[..., 0], -u[..., 1], -torch.ones_like(rho)], -1)
        M = (J[..., :, None] * (wn[..., None, None] * J[..., None, :])).sum(1)
        rhs = -(J * (wn * resid)[..., None]).sum(1)
        # one weighted point gives LAPACK an exactly zero pivot (numpy raises, the CPU fit
        # stops); two leave a small but non-zero one and numpy carries on -- measured.
        good = (torch.linalg.det(M) != 0) & ((wn > 0).sum(1) >= 2)
        Mi = torch.where(good.view(B, 1, 1), M, torch.eye(3, dtype=F64, device=DEV).expand(B, 3, 3))
        step = torch.linalg.solve(Mi, rhs)
        upd = ~done & good
        c = torch.where(upd[:, None], c + step[:, :2], c)
        r = torch.where(upd, r + step[:, 2], r)
        ok &= torch.isfinite(c).all(1) & torch.isfinite(r) & (r > 1)
        done |= ~good | (step.abs().max(1).values < 1e-4) | ~ok
    d = pts - c[:, None, :]
    resid = torch.hypot(d[..., 0], d[..., 1]) - r[:, None]
    ok &= w.sum(1) > 1e-9
    rms = torch.sqrt((w * resid ** 2).sum(1) / torch.clamp_min(w.sum(1), 1e-300))
    return dict(centre=c, radius=r, rms=rms), ok
