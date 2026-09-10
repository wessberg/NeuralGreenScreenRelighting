"""Fit the eye opening to the IMAGE, because the mediapipe eyelid contour is not on it.

Round 35 fixed the limbus and the artist's verdict was immediate and exactly right: *"It
really is just the eyelid you need to mask out. You track the image perfectly, but don't
fully exclude the lid from it."*

The mediapipe eyelid polygon is wrong the same way the four-point iris ellipse was, and by
more. Measured against the image's own sclera-to-lash edge:

| polygon vs the true opening | left | right |
|---|---:|---:|
| sits INSIDE the opening by (median) | **8.34 px** | −1.01 px |
| p25 | 15.98 px | 0.20 px |

So it crosses the middle of the iris on the left eye with open eye and the softbox above
it, and is roughly correct on the right. **No fixed margin can repair that** -- a margin
big enough for the left eye paints canonical iris onto the right eye's lashes on every
half-closed frame, which is what round 35's first attempt did and what the artist caught.

The construction is the one that worked for the limbus:

* **Find the boundary in the image.** Down each column the profile reads skin, then the
  lash as a dark minimum, then the eye. The opening begins at the strongest DARK-TO-BRIGHT
  step below that minimum -- sclera where sclera is visible, iris where it is not. The
  lower lid is the mirror of it.
* **Fit a parabola**, robustly, weighted by how strong each column's step is. A lid margin
  is very close to parabolic, so the columns where the answer is unambiguous -- the ones
  with sclera under the lash -- carry the columns over the iris where it is not. This is
  the standard eyelid model in iris recognition and it is used here for that reason.
* **Abstain rather than guess.** Too few confident columns, or a parabola that disagrees
  with the columns it was fitted to, and the frame keeps the tracked polygon and says so.

Unlike the limbus radius this cannot be a shot constant: a lid moves, that is what a lid
does. What is held constant is the CURVATURE -- one `a` per eye for the shot, since the
lid's shape is anatomy while its height is motion -- which removes most of the per-frame
freedom without adding lag.

Measurement only. Writes no pixels and enables no repair.

    dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/eyelid_fit.py \
        --shot music_videos/reporter_test/shots/reporter_a
"""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shot_paths import add_shot_argument, from_args
import eye_reflection_repair as r32
import limbus_fit as lf

COLUMN_STEP = 0.5        # native px between sampled columns
SAMPLE_STEP = 0.1        # native px along a column
MIN_STEP = 0.030         # minimum dark-to-bright gradient to trust a column
MIN_COLUMNS = 10
MAX_RESIDUAL = 1.5
LID_BAND = (-0.30, 1.70)   # limbus radii from the eye centre a lid edge may occupy       # native px; a parabola that misses its own columns abstains

# Round 40, `--lower-iris-edge`: where the lower lid covers her IRIS the column walk never
# sees it -- iris into lid skin is a DARK-to-bright step, and the walk only takes
# bright-to-dark ones (sclera into the lid margin). Under the iris the lower lid was an
# extrapolation of the sclera columns, and at source 933-955 right it sat 3-5 px below her
# lid: 76 % of every skin pixel the fill painted, clip-wide, was there (rounds/round40.md).
IRIS_EDGE_RHO_LO = 0.35    # limbus radii; nearer the centre a column is still in the pupil
IRIS_EDGE_INSET = 1.5      # native px inside the limbus a lid edge must be to be told from it
IRIS_EDGE_HOLD = 10        # samples (1 px) the column must stay at lid level past the edge
IRIS_EDGE_CONTRAST = 0.12  # display luma her lid skin must stand above her iris, or no rule


def iris_edge_threshold(luma, rho, below, spec_level):
    """Halfway between HER iris and HER lower-lid skin, in display luma, or None.

    The iris level is the median of the disc annulus 0.45-0.85 radii with the specular out;
    the skin level the 75th percentile of the band 1.3-1.65 radii below the centre (lashes
    are the dark quarter). Relative to her own two levels, so a light iris raises it -- and
    where the two do not separate (a light iris against darker skin) there is no rule and
    the walk is the one it always was."""
    sel = (rho > 0.45) & (rho < 0.85) & ~(luma > spec_level)
    band = (rho > 1.3) & (rho < 1.65) & below
    if sel.sum() <= 20 or band.sum() <= 20:
        return None
    iris = np.median(luma[sel])
    skin = np.percentile(luma[band], 75)
    if not (skin - iris >= IRIS_EDGE_CONTRAST):
        return None
    return np.float32(0.5 * (iris + skin))


def iris_edge_first(lum, grad, xx, yy, centre, radius, thr, block=None):
    """Per column, the first sample index where the walk leaves her iris for lid skin while
    still inside the limbus, and the position of that edge; -1 where there is none.

    An up-crossing of `thr` between two samples both at least IRIS_EDGE_RHO_LO radii out and
    IRIS_EDGE_INSET px inside the limbus, that stays above `thr` for IRIS_EDGE_HOLD samples
    and is a real step (gradient over MIN_STEP). Inside the limbus the only thing brighter
    than her iris is a lid (or the specular, which is blocked). `lum` and `grad` (S, C)
    float32 at float32 points xx, yy. Returns (k (C,) int64, y (C,) float64 at the
    crossing, strength (C,) float32)."""
    S, C = lum.shape
    dx = xx.astype(np.float64) - float(centre[0])
    dy = yy.astype(np.float64) - float(centre[1])
    d2 = dx * dx + dy * dy
    lo = IRIS_EDGE_RHO_LO * float(radius)
    hi = float(radius) - IRIS_EDGE_INSET
    inside = (d2 >= lo * lo) & (d2 <= hi * hi)
    bright = lum > thr
    hold = np.ones((S, C), bool)
    for h in range(IRIS_EDGE_HOLD):
        sh = np.ones((S, C), bool)
        sh[:S - h] = bright[h:]
        hold &= sh
    cross = np.zeros((S, C), bool)
    cross[1:] = ~bright[:-1] & bright[1:] & inside[:-1] & inside[1:] & hold[1:]
    if block is not None:
        cross[1:] &= ~block[:-1] & ~block[1:]
    strength = np.zeros((S, C), np.float32)
    strength[1:] = np.maximum(grad[:-1], grad[1:])
    cross &= strength > MIN_STEP
    anyc = cross.any(0)
    k = np.where(anyc, cross.argmax(0), -1)
    kc = np.clip(k, 1, S - 1)
    cols = np.arange(C)
    l0 = lum[kc - 1, cols].astype(np.float64)
    l1 = lum[kc, cols].astype(np.float64)
    y0 = yy[kc - 1, cols].astype(np.float64)
    y1 = yy[kc, cols].astype(np.float64)
    t = (np.float64(thr) - l0) / np.where(l1 > l0, l1 - l0, 1.0)
    y = y0 + t * (y1 - y0)
    return k, y, strength[kc, cols]


def _profile(field, xs, ys):
    return lf._sample(field, xs, ys)


def _spec_block(spec, xx, yy):
    """The specular and a 5-sample collar along each column, not eligible as a lid edge."""
    hit_spec = _profile(spec, xx, yy) > 0.05
    block = hit_spec.copy()
    for shift in (1, 2, 3, 4, 5):
        block[shift:] |= hit_spec[:-shift]
        block[:-shift] |= hit_spec[shift:]
    return block


def iris_edge_samples(luma, gy, spec, centre, radius, x_lo, x_hi, thr, inner=0.15,
                      outer=1.80):
    """`--lower-iris-edge`: where the LOWER lid covers her iris, per column -- the same
    columns and samples as the lower walk, the crossing found by `iris_edge_first`, the
    specular blocked. Returns (xs, ys, strength) with NaN where a column has none. Kept
    apart from the walk's own samples: the lower lid parabola is unchanged by it."""
    cx, cy = centre
    xs = np.arange(x_lo, x_hi + 1e-9, COLUMN_STEP, dtype=np.float32)
    if not len(xs):
        return xs, np.zeros(0, np.float32), np.zeros(0, np.float32)
    ss = np.arange(inner * radius, outer * radius, SAMPLE_STEP, dtype=np.float32)
    yy = np.broadcast_to(cy + 1.0 * ss[:, None], (len(ss), len(xs)))
    xx = np.broadcast_to(xs[None, :], (len(ss), len(xs)))
    grad = _profile(gy, xx, yy) * 1.0
    block = _spec_block(spec, xx, yy)
    grad = np.where(block, 0.0, grad)
    lum = _profile(luma, xx, yy)
    k, y, s = iris_edge_first(lum, grad, xx, yy, centre, radius, thr, block)
    out_y = np.full(len(xs), np.nan, np.float32)
    out_s = np.zeros(len(xs), np.float32)
    got = k >= 0
    out_y[got] = y[got]
    out_s[got] = s[got]
    return xs, out_y, out_s


def lid_boundary(luma, gy, centre, radius, sign, x_lo, x_hi, spec=None,
                 inner=0.15, outer=1.80):
    """Boundary of the eye opening along each column, for one lid.

    `sign` is -1 for the upper lid (smaller y) and +1 for the lower. Returns the boundary
    y per column, the step strength, and the column x.

    Walking OUTWARD from the eye centre a column reads eye, then the lash as a fall in
    brightness. The boundary is the strongest **bright-to-dark** step on that walk: over
    the sclera it is large and unambiguous, over the iris it is small, which is why the
    columns vote into one parabola rather than each being trusted alone.

    Two bounds keep it honest, and both were bugs first. The columns stop short of the
    eye CORNERS -- past them there is no eye boundary to find, only skin, and those
    columns dragged the parabola into a diagonal sweep across the eye. And the walk stops
    at `outer` radii, short of the brow and the lid crease, which are darker steps than
    the lash and won every column they were allowed to reach.
    """
    cx, cy = centre
    xs = np.arange(x_lo, x_hi + 1e-9, COLUMN_STEP, dtype=np.float32)
    if not len(xs):
        return xs, np.zeros(0, np.float32), np.zeros(0, np.float32)
    ss = np.arange(inner * radius, outer * radius, SAMPLE_STEP, dtype=np.float32)
    yy = np.broadcast_to(cy + sign * ss[:, None], (len(ss), len(xs)))
    xx = np.broadcast_to(xs[None, :], (len(ss), len(xs)))
    # d(luma)/d(outward): negative is the bright-to-dark step into the lash.
    grad = _profile(gy, xx, yy) * sign
    if spec is not None:
        # **The corneal highlight is the brightest thing in the eye, so the first big fall
        # on the way out is its own upper edge, not the lash.** Fitted without this the
        # upper lid landed just above the softbox, across the middle of the iris -- the
        # very error being repaired, reproduced by the repair's own tracker. The specular
        # and a two-pixel collar are simply not eligible to be a lid edge.
        grad = np.where(_spec_block(spec, xx, yy), 0.0, grad)
    out_y = np.full(len(xs), np.nan, np.float32)
    out_s = np.zeros(len(xs), np.float32)
    # **The FIRST strong step outward, not the strongest.** Above the eye the lid crease
    # and the brow are darker steps than the lash and win a global argmin every time --
    # which put the fitted upper lid out in her eyebrow. The eye ends at the first fall
    # in brightness on the way out; everything past that is lid, whatever its contrast.
    hit = grad < -MIN_STEP
    for j in range(len(xs)):
        idx = np.flatnonzero(hit[:, j])
        if not len(idx):
            continue
        run = idx[0]
        # walk to the bottom of that step
        kk = run
        while kk + 1 < len(ss) - 1 and grad[kk + 1, j] < grad[kk, j]:
            kk += 1
        if kk < 1 or kk >= len(ss) - 1:
            continue
        strength = float(-grad[kk, j])
        a, b, c = grad[kk - 1, j], grad[kk, j], grad[kk + 1, j]
        denom = a - 2 * b + c
        shift = 0.0 if abs(denom) < 1e-9 else float(np.clip(0.5 * (a - c) / denom, -1, 1))
        out_y[j] = cy + sign * (ss[kk] - shift * SAMPLE_STEP)
        out_s[j] = strength
    return xs, out_y, out_s


def fit_parabola(xs, ys, w, centre, curvature=None, tukey=2.0, iterations=5):
    """Robust y = a u^2 + b u + c in u = x - cx, weighted by step strength."""
    ok = np.isfinite(ys) & (w > 0)
    if ok.sum() < MIN_COLUMNS:
        return None
    u = (xs[ok] - centre[0]).astype(np.float64)
    y = ys[ok].astype(np.float64)
    weight = w[ok].astype(np.float64)
    if curvature is None:
        basis = np.column_stack([u ** 2, u, np.ones(len(u))])
        target = y
    else:
        basis = np.column_stack([u, np.ones(len(u))])
        target = y - curvature * u ** 2
    beta = None
    for _ in range(iterations):
        try:
            beta = np.linalg.solve(basis.T @ (weight[:, None] * basis),
                                   basis.T @ (weight * target))
        except np.linalg.LinAlgError:
            return None
        resid = target - basis @ beta
        scale = max(1.4826 * np.median(np.abs(resid - np.median(resid))), 0.25)
        weight = w[ok] * (1 - np.clip(resid / (tukey * scale), -1, 1) ** 2) ** 2
        if weight.sum() <= 1e-9:
            return None
    resid = target - basis @ beta
    coef = ((float(beta[0]), float(beta[1]), float(beta[2])) if curvature is None
            else (float(curvature), float(beta[0]), float(beta[1])))
    return dict(coef=coef, columns=int(ok.sum()),
                rms=float(np.sqrt(np.average(resid ** 2, weights=weight))),
                inliers=int((np.abs(resid) < 2 * scale).sum()))


def evaluate(coef, x, centre):
    u = np.asarray(x, np.float64) - centre[0]
    return coef[0] * u ** 2 + coef[1] * u + coef[2]


def opening_mask(shape, offset, coef_upper, coef_lower, centre, feather=0.8):
    """Soft matte of the eye opening: below the upper parabola and above the lower one."""
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    xx = xx + offset[0]
    yy = yy + offset[1]
    m = np.ones(shape, np.float32)
    for coef, sign in ((coef_upper, 1.0), (coef_lower, -1.0)):
        if coef is None:
            continue
        edge = evaluate(coef, xx, centre)
        m = np.minimum(m, np.clip(sign * (yy - edge) / max(feather, 1e-6) + 0.5, 0.0, 1.0))
    return m.astype(np.float32)


def fit_eye(values, centre, radius, corners, margin=14, lower_iris_edge=False,
            lower_spec=False):
    """Both lids for one eye in one frame, or the reason it abstains.

    `lower_spec` blocks the specular for the LOWER walk too (round 40: at a sideways glance
    the softbox sits beside the pupil and its lower edge was taken for the lower lid at
    961-964 and 1050-1054 left, collapsing the opening on an open eye). `lower_iris_edge`
    adds the lid over her iris (`iris_edge_first`)."""
    rec = dict(fitted=False, reason=None)
    h, w = values.shape[:2]
    reach = radius * 2.9 + margin
    x0 = int(np.floor(centre[0] - reach)); x1 = int(np.ceil(centre[0] + reach)) + 1
    y0 = int(np.floor(centre[1] - reach)); y1 = int(np.ceil(centre[1] + reach)) + 1
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        rec['reason'] = 'lid_crop_leaves_frame'
        return rec
    window = (slice(y0, y1), slice(x0, x1))
    luma = lf.display_luma(values[window])
    k = int(2 * np.ceil(3 * lf.SIGMA) + 1)
    blur = cv2.GaussianBlur(luma, (k, k), lf.SIGMA, borderType=cv2.BORDER_REPLICATE)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    local = (centre[0] - x0, centre[1] - y0)
    yy_, xx_ = np.mgrid[:luma.shape[0], :luma.shape[1]]
    rho = np.hypot(xx_ - local[0], yy_ - local[1]) / max(radius, 1e-6)
    ring = (rho > 1.15) & (rho < 1.55)
    sclera = float(np.percentile(luma[ring], 90)) if ring.sum() > 30 else 0.5
    spec = cv2.GaussianBlur(
        ((luma > max(0.42, sclera + 0.06)) & (rho < 1.0)).astype(np.float32),
        (0, 0), 0.8)
    # The eye corners are the one thing the mediapipe contour does get right -- they are
    # sharp, unoccluded and far from the lash. Its UPPER BRANCH is what is wrong, so the
    # corners set where to look and the image sets what is found there.
    inset = 0.16 * (corners[1] - corners[0])
    x_lo = max(1.0, corners[0] + inset - x0)
    x_hi = min(float(luma.shape[1] - 2), corners[1] - inset - x0)
    out = {}
    for name, sign in (('upper', -1.0), ('lower', 1.0)):
        xs, ys, ss = lid_boundary(blur, gy, local, radius, sign, x_lo, x_hi,
                                  spec=spec if (sign < 0 or lower_spec) else None)
        out[name] = dict(xs=xs, ys=ys, ss=ss)
    if lower_iris_edge:
        iris_thr = iris_edge_threshold(luma, rho, yy_ > local[1], max(0.42, sclera + 0.06))
        rec['iris_edge_thr'] = None if iris_thr is None else float(iris_thr)
        if iris_thr is not None:
            xs, ys, ss = iris_edge_samples(blur, gy, spec, local, radius, x_lo, x_hi, iris_thr)
            out['lower_iris'] = dict(xs=xs, ys=ys, ss=ss)
    rec.update(window=(x0, y0, x1, y1), local_centre=local, samples=out)
    return rec


def raw_fits_gpu(shot, data, track, topology, points, only=None, margin=14, io_workers=12,
                 chunk=384, lower_iris_edge=False, lower_spec=False):
    """Every eye's `fit_eye` record on the GPU (track_gpu.py): the fields, the column
    samples and the walk batched; the same records, bit for bit, as fit_eye on the CPU
    (the parabolas are fitted afterwards by this file's own numpy code). `only`: a set of
    (frame, side) to fit instead of every tracked eye."""
    import torch
    from concurrent.futures import ThreadPoolExecutor
    import eye_gpu as G
    import track_gpu as T
    frames = [int(row['frame']) for row in data['rows']]
    src = {int(row['frame']): row['source_frame'] for row in data['rows']}
    todo = []
    for i in frames:
        for side in r32.SIDES:
            got = track.get(i, side)
            if got is not None and (only is None or (i, side) in only):
                todo.append((i, side, got))
    need = {}
    for i, side, got in todo:
        need.setdefault(i, []).append((side, got))

    def read(i):
        img = cv2.imread(str(shot.fg(i)), cv2.IMREAD_UNCHANGED)
        return None if img is None else img[..., ::-1]

    out = {}
    metas = []
    with ThreadPoolExecutor(max_workers=io_workers) as pool:
        futures = {i: pool.submit(read, i) for i in need}
        for i in sorted(need):
            img = futures.pop(i).result()
            if img is None:
                continue
            h, w = img.shape[:2]
            for side, got in need[i]:
                centre, radius = got[0], got[1]
                poly = np.asarray(points[i])[topology['eyelid_' + side], :2]
                corners = (float(poly[:, 0].min()), float(poly[:, 0].max()))
                rec = dict(fitted=False, reason=None)
                reach = radius * 2.9 + margin
                x0 = int(np.floor(centre[0] - reach)); x1 = int(np.ceil(centre[0] + reach)) + 1
                y0 = int(np.floor(centre[1] - reach)); y1 = int(np.ceil(centre[1] + reach)) + 1
                out[(i, side)] = (rec, centre, radius, src[i])
                if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
                    rec['reason'] = 'lid_crop_leaves_frame'
                    continue
                local = (centre[0] - x0, centre[1] - y0)
                inset = 0.16 * (corners[1] - corners[0])
                ww = x1 - x0
                x_lo = max(1.0, corners[0] + inset - x0)
                x_hi = min(float(ww - 2), corners[1] - inset - x0)
                metas.append(dict(rec=rec, local=local, radius=radius, x_lo=x_lo, x_hi=x_hi,
                                  window=(x0, y0, x1, y1), shape=(y1 - y0, ww),
                                  patch=torch.as_tensor(np.ascontiguousarray(
                                      img[y0:y1, x0:x1]), device=G.DEV).half()))
    by_shape = {}
    for m in metas:
        by_shape.setdefault(m['shape'], []).append(m)
    fields = []
    for shape, lst in sorted(by_shape.items()):
        got = T.lid_fields(torch.stack([m['patch'] for m in lst]), lst,
                           iris_edge=lower_iris_edge)
        blur, gy, spec = got[:3]
        for n, m in enumerate(lst):
            m['gy'], m['spec'] = gy[n], spec[n]
            if lower_iris_edge:
                m['blur'], m['iris_thr'] = blur[n], got[3][n]
            del m['patch']
        fields += lst
    for s0 in range(0, len(fields), chunk):
        part = fields[s0:s0 + chunk]
        Hm = max(m['shape'][0] for m in part)
        Wm = max(m['shape'][1] for m in part)
        cg = torch.zeros((len(part), Hm, Wm), dtype=torch.float32, device=G.DEV)
        cs = torch.zeros_like(cg)
        cl = torch.zeros_like(cg) if lower_iris_edge else None
        hw = torch.zeros((len(part), 2), dtype=torch.int64, device=G.DEV)
        for n, m in enumerate(part):
            h, w = m['shape']
            cg[n, :h, :w], cs[n, :h, :w] = m['gy'], m['spec']
            if cl is not None:
                cl[n, :h, :w] = m['blur']
            hw[n, 0], hw[n, 1] = h, w
        samp = G.CubicSampler([cg, cs] + ([cl] if cl is not None else []), hw)
        samples = {}
        for name, sign in (('upper', -1.0), ('lower', 1.0)):
            xs, ys, ss, colv = T.lid_boundary(samp, part, sign,
                                              with_spec=(sign < 0 or lower_spec))
            samples[name] = (xs.cpu().numpy(), ys.cpu().numpy(), ss.cpu().numpy(),
                             colv.sum(1).cpu().numpy())
        if lower_iris_edge:
            # an eye without a threshold samples at 0: every sample bright, no crossing
            thr = [np.float32(0.0) if m['iris_thr'] is None else m['iris_thr'] for m in part]
            xs, ys, ss, colv = T.iris_edge_samples(samp, part, thr)
            samples['lower_iris'] = (xs.cpu().numpy(), ys.cpu().numpy(), ss.cpu().numpy(),
                                     colv.sum(1).cpu().numpy())
        for n, m in enumerate(part):
            smp = {}
            for name in samples:
                if name == 'lower_iris' and m['iris_thr'] is None:
                    continue
                xs, ys, ss, nc = samples[name]
                k = int(nc[n])
                smp[name] = dict(xs=xs[n, :k].copy(), ys=ys[n, :k].copy(), ss=ss[n, :k].copy())
            m['rec'].update(window=m['window'], local_centre=m['local'], samples=smp)
            if lower_iris_edge:
                m['rec']['iris_edge_thr'] = (None if m['iris_thr'] is None
                                             else float(m['iris_thr']))
        del samp, cg, cs, cl
    for m in fields:
        m.pop('gy', None); m.pop('spec', None); m.pop('blur', None)
    return out


def fit_lower_iris(rec, curv_lower):
    """`lower_iris`: the lower lid where it covers her iris, fitted through the iris-edge
    samples ONLY (`iris_edge_samples`) at the lower lid's curvature. None where fewer than
    MIN_COLUMNS columns (5 px) saw it or the fit misses its own columns."""
    d = (rec.get('samples') or {}).get('lower_iris')
    if d is None or curv_lower is None:
        return None
    f = fit_parabola(d['xs'], d['ys'], d['ss'], rec['local_centre'], curvature=curv_lower)
    if f is None or f['rms'] > MAX_RESIDUAL:
        return None
    a, b, c = f['coef']
    # The columns the fit is measured over: its inliers. Past them it is an extrapolation
    # of a few pixels of lid, and a tilted parabola climbs into the disc there (round 40:
    # 21-22, 386, 836-837 left), so the consumer holds it to this span.
    ok = np.isfinite(d['ys']) & (d['ss'] > 0)
    u = (d['xs'][ok] - rec['local_centre'][0]).astype(np.float64)
    resid = d['ys'][ok].astype(np.float64) - (a * u * u + b * u + c)
    scale = max(1.4826 * np.median(np.abs(resid - np.median(resid))), 0.25)
    inl = np.abs(resid) < 2 * scale
    return dict(coef=[a, b, float(c + rec['window'][1])], columns=f['columns'],
                rms=round(f['rms'], 4),
                u_span=[round(float(u[inl].min()), 3), round(float(u[inl].max()), 3)])


def fit_lids(rec, radius, curv_upper, curv_lower):
    """Both parabolas for one eye's samples at the given curvatures, as main() makes them."""
    entry = {}
    reason = None
    if 'lower_iris' in (rec.get('samples') or {}):
        entry['lower_iris'] = fit_lower_iris(rec, curv_lower)
    for name, curv in (('upper', curv_upper), ('lower', curv_lower)):
        d = rec['samples'][name]
        f = fit_parabola(d['xs'], d['ys'], d['ss'], rec['local_centre'], curvature=curv)
        height = None
        if f is not None:
            height = (f['coef'][2] - rec['local_centre'][1]) / radius
        if (f is None or f['rms'] > MAX_RESIDUAL or height is None
                or not (LID_BAND[0] <= (height if name == 'lower' else -height)
                        <= LID_BAND[1])):
            entry[name] = None
            reason = '%s_lid_unfitted' % name
            continue
        a, b, c = f['coef']
        x0, y0 = rec['window'][0], rec['window'][1]
        entry[name] = dict(coef=[a, b, float(c + y0)], columns=f['columns'],
                           rms=round(f['rms'], 4))
    return entry, reason


def update_track(args, shot, track, data, topology, points):
    """--update: refit only the eyes whose limbus centre moved (see the flag's help)."""
    base = json.loads((shot.relight / args.update).read_text(encoding='utf-8'))
    base_rows = {int(r['frame']): r for r in base['rows']}
    # an update refits with the construction the base track was built with
    walk = dict(lower_iris_edge=bool(base.get('lower_iris_edge', False)),
                lower_spec=bool(base.get('lower_spec', False)))
    for k, v in walk.items():
        if getattr(args, k) and not v:
            raise SystemExit('--%s: the base track %s was fitted without it; refit it instead '
                             'of updating' % (k.replace('_', '-'), args.update))

    def moved(i, be):
        got = track.get(i, be['side'])
        same = (got is not None and be.get('centre') is not None
                and np.allclose([round(float(c), 4) for c in got[0]], be['centre']))
        return got is not None and not same

    fits = None
    if args.device == 'cuda':
        fits = raw_fits_gpu(shot, data, track, topology, points,
                            only={(int(row['frame']), be['side']) for row in data['rows']
                                  for be in base_rows[int(row['frame'])]['eyes']
                                  if moved(int(row['frame']), be)}, **walk)
    rows, refit = [], []
    cache = {}
    for row in data['rows']:
        i = int(row['frame'])
        brow = base_rows[i]
        eyes = []
        for be in brow['eyes']:
            side = be['side']
            got = track.get(i, side)
            if not moved(i, be):
                eyes.append(be)
                continue
            centre, radius = got[0], got[1]
            if fits is not None:
                rec = fits[(i, side)][0]
            else:
                if i not in cache:
                    img = cv2.imread(str(shot.fg(i)), cv2.IMREAD_UNCHANGED)
                    cache.clear()
                    cache[i] = img[..., ::-1].astype(np.float32)
                poly = np.asarray(points[i])[topology['eyelid_' + side], :2]
                corners = (float(poly[:, 0].min()), float(poly[:, 0].max()))
                rec = fit_eye(cache[i], centre, radius, corners, **walk)
            if 'samples' not in rec:
                eyes.append(dict(side=side, fitted=False, reason=rec['reason'], refit=True))
                refit.append((i, side, 'abstained'))
                continue
            curv = be.get('curvature') or {}
            cu = curv.get('upper', (base.get('curvature') or {}).get('%s_upper' % side))
            cl = curv.get('lower', (base.get('curvature') or {}).get('%s_lower' % side))
            lids, reason = fit_lids(rec, radius, cu, cl)
            entry = dict(side=side, fitted=reason is None, reason=reason,
                         centre=[round(float(c), 4) for c in centre],
                         radius=round(float(radius), 4), refit=True)
            if 'curvature' in be:
                entry['curvature'] = be['curvature']
            entry.update(lids)
            eyes.append(entry)
            refit.append((i, side, 'fitted' if reason is None else reason))
        rows.append(dict(frame=i, source_frame=row['source_frame'], eyes=eyes))
    payload = dict(base, rows=rows, limbus_track=args.track,
                   updated_from=args.update,
                   updated_from_sha256=r32.sha(shot.relight / args.update),
                   refit_eyes=len(refit), code_sha256=r32.sha(Path(__file__)))
    payload['fitted'] = {s: sum(1 for row in rows for e in row['eyes']
                                if e['side'] == s and e.get('fitted')) for s in r32.SIDES}
    r32.write_json(shot.relight / args.out, payload)
    print('refit %d eyes whose limbus centre moved; copied the rest' % len(refit))
    print(refit)
    print('wrote %s' % (shot.relight / args.out))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_shot_argument(ap)
    ap.add_argument('--track', default='limbus_track.json')
    ap.add_argument('--out', default='eyelid_track.json')
    ap.add_argument('--free-curvature', action='store_true',
                    help='fit the parabola curvature per frame instead of holding one '
                         'per eye for the shot. A lid moves; its shape does not.')
    ap.add_argument('--curvature-window', type=float, default=0.0,
                    help="SECONDS of footage each frame's curvature is pooled from, instead "
                         "of one per eye for the whole shot. 0 is the shot constant. Over "
                         "the 20 s reporter clip the free upper-lid curvature runs from "
                         "~0.003 to ~0.02 in 100-frame stretches as her head turns and her "
                         "gaze moves, so the shot median (0.011) is 2-3x the 0.004-0.005 of "
                         "the stretch round 37 was accepted on. The window is shifted, not "
                         "cut, at the ends of the shot, so a shot shorter than the window "
                         "gets its own median back (of the anatomically possible fits).")
    ap.add_argument('--device', choices=('cuda', 'cpu'), default='cuda',
                    help="'cuda' computes every eye's column samples on the GPU "
                         "(track_gpu.py; identical records); 'cpu' is the numpy reference")
    ap.add_argument('--update', default=None,
                    help="an existing eyelid track (relight/<file>) to UPDATE rather than "
                         "refit: only the eyes whose limbus centre in --track differs from the "
                         "centre that track was fitted around are refitted, with the curvature "
                         "it recorded for them; every other eye is copied unchanged. Round 39: "
                         "a full refit after limbus_trust.py moved 88 centres changed the "
                         "window-pooled curvature, and so the lids, on 1333 eyes.")
    ap.add_argument('--lower-iris-edge', action='store_true',
                    help="round 40: the lower lid ALSO where it covers her iris -- a column "
                         "that leaves her iris for lid-skin level while still inside the "
                         "limbus (iris_edge_first). Without it the lower lid under the iris "
                         "is an extrapolation of the sclera columns.")
    ap.add_argument('--lower-spec', action='store_true',
                    help="round 40: block the corneal specular for the LOWER walk as the "
                         "upper one always has; at a sideways glance the softbox's lower edge "
                         "was taken for the lower lid.")
    args = ap.parse_args()
    shot = from_args(args)
    track = lf.Track.load(shot.relight / args.track)
    data = json.loads((shot.relight / args.track).read_text(encoding='utf-8'))
    topology = json.loads((shot.relight / 'face_landmarks.json').read_text())['topology']
    with np.load(shot.relight / 'face_landmarks.npz') as z:
        points = z['points']
    if args.update:
        update_track(args, shot, track, data, topology, points)
        return

    raw_fits = {}
    walk = dict(lower_iris_edge=args.lower_iris_edge, lower_spec=args.lower_spec)
    if args.device == 'cuda':
        raw_fits = raw_fits_gpu(shot, data, track, topology, points, **walk)
    for row in ([] if args.device == 'cuda' else data['rows']):
        i = row['frame']
        img = cv2.imread(str(shot.fg(i)), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        values = img[..., ::-1].astype(np.float32)
        for side in r32.SIDES:
            got = track.get(i, side)
            if got is None:
                continue
            centre, radius = got[0], got[1]
            poly = np.asarray(points[i])[topology['eyelid_' + side], :2]
            corners = (float(poly[:, 0].min()), float(poly[:, 0].max()))
            raw_fits[(i, side)] = (fit_eye(values, centre, radius, corners, **walk), centre,
                                   radius, row['source_frame'])

    # One curvature per eye for the shot, from the frames that fit it freely and well.
    curvature = {}
    votes = {}
    if not args.free_curvature:
        for side in r32.SIDES:
            for name in ('upper', 'lower'):
                got = {}
                for (i, s), (rec, centre, radius, _) in raw_fits.items():
                    if s != side or 'samples' not in rec:
                        continue
                    d = rec['samples'][name]
                    f = fit_parabola(d['xs'], d['ys'], d['ss'], rec['local_centre'])
                    if f and f['rms'] <= 1.2 and f['columns'] >= 24:
                        got[i] = f['coef'][0]
                votes[(side, name)] = got
                curvature[(side, name)] = (float(np.median(list(got.values())))
                                           if len(got) >= 10 else None)
    # --- optionally, the curvature of the frames AROUND each frame --------------------
    # Only anatomically possible shapes vote: an upper margin arches UP (a > 0 with y
    # down) and a lower one down. Measured on the reporter clip, source 800-899 left: half
    # the free upper-lid fits that pass the quality gate curve the wrong way (p25 -0.014)
    # -- they are on something else -- and a windowed median of them came out NEGATIVE,
    # which would have bent every lid in the stretch inside out. The shot constant hid that
    # only because the other 1100 frames outvoted it.
    local = {}
    if args.curvature_window > 0 and not args.free_curvature:
        frames_all = sorted(int(row['frame']) for row in data['rows'])
        first, last = frames_all[0], frames_all[-1]
        half = max(1, int(round(0.5 * args.curvature_window * float(shot.frame_rate))))
        for key, got in votes.items():
            sign = 1.0 if key[1] == 'upper' else -1.0
            ok = sorted((j, a) for j, a in got.items() if a * sign > 0)
            js = np.array([j for j, _ in ok], int)
            vals = np.array([a for _, a in ok], float)
            for i in frames_all:
                h = half
                while True:
                    lo = max(first, min(i - h, last - 2 * h))
                    hi = min(last, lo + 2 * h)
                    sel = vals[(js >= lo) & (js <= hi)]
                    if sel.size >= 10 or (lo <= first and hi >= last):
                        break
                    h *= 2
                local[(key, i)] = float(np.median(sel)) if sel.size >= 10 else curvature[key]

    rows, stats = [], {s: {'upper': [], 'lower': []} for s in r32.SIDES}
    for row in data['rows']:
        i = row['frame']
        eyes = []
        for side in r32.SIDES:
            key = (i, side)
            if key not in raw_fits:
                eyes.append(dict(side=side, fitted=False, reason='no_limbus_fit'))
                continue
            rec, centre, radius, _ = raw_fits[key]
            if 'samples' not in rec:
                eyes.append(dict(side=side, fitted=False, reason=rec['reason']))
                continue
            entry = dict(side=side, fitted=True, reason=None,
                         centre=[round(float(c), 4) for c in centre],
                         radius=round(float(radius), 4))
            if local:
                entry['curvature'] = {name: local.get(((side, name), i))
                                      for name in ('upper', 'lower')}
            if args.lower_iris_edge:
                entry['lower_iris'] = fit_lower_iris(
                    rec, local.get(((side, 'lower'), i), curvature.get((side, 'lower'))))
            for name in ('upper', 'lower'):
                d = rec['samples'][name]
                f = fit_parabola(d['xs'], d['ys'], d['ss'], rec['local_centre'],
                                 curvature=local.get(((side, name), i),
                                                     curvature.get((side, name))))
                # A lid cannot be in her eyebrow. The fitted edge above the iris centre
                # has to sit in a band a lid can actually occupy; on a few near-blink
                # frames the first strong step outward is the brow and the parabola
                # follows it there. Those abstain and keep the tracked polygon.
                height = None
                if f is not None:
                    height = (f['coef'][2] - rec['local_centre'][1]) / radius
                if (f is None or f['rms'] > MAX_RESIDUAL or height is None
                        or not (LID_BAND[0] <= (height if name == 'lower' else -height)
                                <= LID_BAND[1])):
                    entry[name] = None
                    entry['fitted'] = False
                    entry['reason'] = '%s_lid_unfitted' % name
                    continue
                # Back to native image coordinates: the parabola was fitted in the crop.
                a, b, c = f['coef']
                x0, y0 = rec['window'][0], rec['window'][1]
                entry[name] = dict(coef=[a, b, float(c + y0)],
                                   columns=f['columns'], rms=round(f['rms'], 4))
                stats[side][name].append(f['rms'])
            eyes.append(entry)
        rows.append(dict(frame=i, source_frame=row['source_frame'], eyes=eyes))

    fitted = {s: sum(1 for row in rows for e in row['eyes']
                     if e['side'] == s and e.get('fitted')) for s in r32.SIDES}
    payload = dict(rows=rows, frames=len(rows), fitted=fitted,
                   curvature={'%s_%s' % k: v for k, v in curvature.items()},
                   free_curvature=bool(args.free_curvature),
                   curvature_window_s=args.curvature_window,
                   curvature_local_range=({'%s_%s' % key: [
                       round(min(v for (k, _), v in local.items() if k == key and v is not None), 5),
                       round(max(v for (k, _), v in local.items() if k == key and v is not None), 5)]
                       for key in votes if any(k == key and v is not None
                                               for (k, _), v in local.items())}
                       if local else None),
                   min_step=MIN_STEP, column_step=COLUMN_STEP,
                   lower_iris_edge=bool(args.lower_iris_edge),
                   lower_spec=bool(args.lower_spec),
                   method='per-column dark-to-bright step below the lash minimum, robust '
                          'parabola per lid, curvature held per eye for the shot',
                   limbus_track=args.track,
                   production_enabled=False, artist_accepted=False,
                   code_sha256=r32.sha(Path(__file__)))
    r32.write_json(shot.relight / args.out, payload)
    print('fitted %s of %d frames' % (fitted, len(rows)))
    for side in r32.SIDES:
        for name in ('upper', 'lower'):
            v = stats[side][name]
            k = curvature.get((side, name))
            print('  %-5s %-5s lid: column residual median %.3f px  p95 %.3f  n=%d  '
                  'curvature %s' % (side, name, np.median(v) if v else float('nan'),
                                    np.percentile(v, 95) if v else float('nan'), len(v),
                                    'free' if k is None else '%.5f' % k))
    print('wrote %s' % (shot.relight / args.out))


if __name__ == '__main__':
    main()
