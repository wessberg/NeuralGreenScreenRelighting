"""Fit the limbus to the IMAGE EDGE, once per frame per eye, and smooth the trajectory.

Round 34 measured the blocker: the mediapipe four-point iris ellipse sits at **0.936 of
the observed iris radius on the left eye**, with 1.66 px of spread and 2.49 px p95
frame-to-frame change *against the photograph* (`audit_r34_track.py`). On a 12.9 px iris
that is a 13 percent systematic size error and plus/minus 1.5 px of wander. Every fill
this project has built since round 32 is anchored on that ellipse, so every fill inherits
it -- which is the artist's "double iris" and his "delayed, slow to react".

Round 33 reported this landmark's jitter as 0.353 px rms and concluded it was solid. That
measured SMOOTHNESS against its own trajectory, not ACCURACY against the image. **The
control has to be the photograph.**

What this module does, and why each piece is the way it is:

* **Daugman's integro-differential operator** finds the circle globally, by maximising the
  mean radial derivative around the contour. It is not a per-angle argmax, so a single
  lash, a bright sclera vessel or the lid margin cannot pull it: they are one angle out of
  a hundred in a mean. Round 34's audit *was* a per-angle argmax with a median over 18
  angles, which is why its own spread (0.127 of a radius) is partly its own noise.
* **Then per-angle sub-pixel edges and a robust circle fit**, for the last fraction of a
  pixel and -- more importantly -- for a residual per angle, so the fit can report where
  it disagrees with the image rather than only that it converged.
* **A CIRCLE, not the landmark's axis-aligned ellipse.** The iris is a rigid disc; at this
  gaze it projects to something within a few percent of a circle. Mediapipe's ellipse has
  ax/ay = 1.16 on this plate -- 12.81 by 11.08 px -- which is not foreshortening, it is
  the vertical iris landmarks being pulled in by the lids. `--model ellipse` fits the
  extra parameter so the choice is measured rather than assumed.
* **The radius is one number for the shot, not a per-frame free parameter.** Her iris does
  not change size; only her distance to the camera does, and that is slow. Fitting it per
  frame hands a rigid quantity to the noise. This is the same construction that made round
  34's fill stable -- build the invariant ONCE -- applied to the geometry that fill sits
  on. `--radius-model per-frame` keeps the old behaviour for comparison.
* **No temporal smoothing, and that is a measurement, not an omission.** The fitted
  centre's residual against a local quadratic through its own neighbours is **0.048 px
  median, 0.229 p95 on the left eye** -- the trajectory is already smooth to a twentieth
  of a pixel, because a fit over ~60 sub-pixel edge samples has a standard error far
  below the motion. There is nothing left to filter, and filtering would only cost: at
  the fast head move around source 757 a plus/minus 3-frame local-linear smoother moved
  the centre **3.4 px**, which is lag, which is precisely the artist's "delayed, slow to
  react". `--smooth-half` is kept and DEFAULTS TO 0; when it is used it is a centred
  local-linear regression, which at least has no group delay for constant velocity.
  **The landmark was never jittery either** (0.049 px against the same control) -- round
  33 measured that correctly and drew the wrong conclusion from it. Its error is a
  SYSTEMATIC 2.5 px offset and a 4 percent size error, and no temporal filter addresses
  either.

Everything here is measurement of the plate. It writes no pixels and enables no repair.

    dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/limbus_fit.py \
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
from color_pipeline import linear_to_srgb
from shot_paths import add_shot_argument, from_args
import eye_reflection_repair as r32

# Rec.709 luma. Round 34's audit used the max channel; luma is the full-resolution
# component of 4:2:0 footage, and the limbus edge is a luminance edge. The audit keeps
# the max channel deliberately, so its estimator shares nothing with this one.
LUMA = np.float32([0.2126, 0.7152, 0.0722])

# Derivative scale, native pixels. The limbus transition on this plate spans about two
# pixels; sigma 1.0 matches it. On a circle of radius 13 px a Gaussian of this width
# biases the apparent edge inward by sigma^2 / 2r = 0.038 px, which is below anything
# measurable here.
SIGMA = 1.0
ANGLES = 128
LID_MIN = 1.25          # native px inside the eyelid polygon before an angle is trusted
MIN_ARC = 0.12          # fraction of the circle that must survive gating
MIN_ANGLES = 14
MAX_SE = 0.60           # native px; standard error of the fitted centre on its worst axis


def display_luma(values):
    """Display-referred luma from scene-linear keyed RGB.

    The edge is found in the domain a viewer sees, not in linear light: linear crushes
    the iris into a few thousandths and puts the gradient peak where the eye does not
    see one.
    """
    lin = np.asarray(values, np.float32)
    return (np.clip(linear_to_srgb(lin), 0, 1) @ LUMA).astype(np.float32)


def gradient_field(luma, sigma=SIGMA):
    """Gaussian-derivative gradient, sampled later by bicubic remap.

    Bilinear sampling of the IMAGE and differencing gives a piecewise-constant derivative
    that snaps the peak to pixel boundaries; a smooth gradient field sampled bicubically
    does not, and that is the whole sub-pixel budget on a 13 px iris.
    """
    k = int(2 * np.ceil(3 * sigma) + 1)
    blur = cv2.GaussianBlur(luma, (k, k), sigma, borderType=cv2.BORDER_REPLICATE)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return gx, gy


def _sample(field, xs, ys):
    """Bicubic sample of a scalar field, chunked past OpenCV's SHRT_MAX map width."""
    xs = np.ascontiguousarray(xs, np.float32)
    ys = np.ascontiguousarray(ys, np.float32)
    if xs.ndim == 2 and xs.shape[1] < 32000:
        return cv2.remap(field, xs, ys, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    flat_x, flat_y = xs.reshape(-1), ys.reshape(-1)
    width = 16384
    pad = (-len(flat_x)) % width
    gx = np.pad(flat_x, (0, pad), mode='edge').reshape(-1, width)
    gy = np.pad(flat_y, (0, pad), mode='edge').reshape(-1, width)
    out = cv2.remap(field, gx, gy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return out.reshape(-1)[:len(flat_x)].reshape(xs.shape)


def radial_derivative(gx, gy, xs, ys, cos, sin):
    """Outward radial derivative of the blurred luma at the given sample points."""
    return _sample(gx, xs, ys) * cos + _sample(gy, xs, ys) * sin


def valid_angles(lid, x0, y0, centre, radius, cos, sin, lid_min=LID_MIN):
    """Angles whose limbus point is clear of the eyelid polygon by `lid_min` pixels.

    The gate is on the LIDS, which know nothing about the edge, so it cannot select the
    angles that happen to agree with the current estimate. That circularity is the easy
    mistake here and it would manufacture any accuracy figure asked of it.
    """
    xs = (centre[0] + radius * cos - x0)[None, :]
    ys = (centre[1] + radius * sin - y0)[None, :]
    return (_sample(lid, xs, ys)[0] > lid_min)


def daugman(gx, gy, x0, y0, centre, radii, cos, sin, keep, span=3.0, step=0.25):
    """Global circle search: maximise the mean radial derivative around the contour.

    Daugman's integro-differential operator, restricted to the unoccluded arcs. It is a
    MEAN over angles, so no single strong edge -- a lash, a vessel, the lid margin -- can
    move it, which is the failure mode of every per-angle argmax this project has used on
    this eye.
    """
    offsets = np.arange(-span, span + 1e-9, step, dtype=np.float32)
    n_off, n_r, n_a = len(offsets), len(radii), int(keep.sum())
    c, s = cos[keep], sin[keep]
    # (dy, dx, r, angle) sample coordinates, one remap for the whole search volume.
    cx = (centre[0] + offsets[None, :, None, None] + radii[None, None, :, None] * c) - x0
    cy = (centre[1] + offsets[:, None, None, None] + radii[None, None, :, None] * s) - y0
    cx = np.broadcast_to(cx, (n_off, n_off, n_r, n_a)).reshape(1, -1)
    cy = np.broadcast_to(cy, (n_off, n_off, n_r, n_a)).reshape(1, -1)
    d = (_sample(gx, cx, cy) * np.tile(c, n_off * n_off * n_r)
         + _sample(gy, cx, cy) * np.tile(s, n_off * n_off * n_r))
    score = d.reshape(n_off, n_off, n_r, n_a).mean(-1)
    j = np.unravel_index(int(np.argmax(score)), score.shape)
    return (float(centre[0] + offsets[j[1]]), float(centre[1] + offsets[j[0]]),
            float(radii[j[2]]), float(score[j]))


def ray_edges(gx, gy, x0, y0, centre, radius, cos, sin, keep, lo=0.72, hi=1.32, step=0.1):
    """Sub-pixel edge radius per angle, by parabolic refinement of the derivative peak."""
    ss = np.arange(lo * radius, hi * radius + 1e-9, step, dtype=np.float32)
    c, s = cos[keep], sin[keep]
    xs = (centre[0] + ss[:, None] * c[None, :]) - x0
    ys = (centre[1] + ss[:, None] * s[None, :]) - y0
    d = radial_derivative(gx, gy, xs, ys, c[None, :], s[None, :])
    k = np.argmax(d, axis=0)
    peak = d[k, np.arange(d.shape[1])]
    inner = np.clip(k, 1, len(ss) - 2)
    a, b, cc = d[inner - 1, np.arange(d.shape[1])], d[inner, np.arange(d.shape[1])], d[inner + 1, np.arange(d.shape[1])]
    denom = a - 2 * b + cc
    shift = np.where(np.abs(denom) > 1e-9, 0.5 * (a - cc) / np.where(denom == 0, 1, denom), 0.0)
    shift = np.clip(shift, -1, 1)
    r_edge = ss[inner] + shift * step
    # A peak on the first or last sample is not a peak; the edge is outside the band.
    ok = (k > 0) & (k < len(ss) - 1) & (peak > 0)
    return r_edge, peak, ok


def fit_circle(pts, c0, r0, iterations=6, tukey=2.5):
    """Robust geometric circle fit, IRLS with a Tukey biweight on the radial residual."""
    p = np.asarray(pts, np.float64)
    c = np.array(c0, np.float64)
    r = float(r0)
    w = np.ones(len(p))
    for _ in range(iterations):
        d = p - c
        rho = np.hypot(d[:, 0], d[:, 1])
        rho = np.where(rho < 1e-9, 1e-9, rho)
        u = d / rho[:, None]
        resid = rho - r
        scale = max(1.4826 * np.median(np.abs(resid - np.median(resid))), 0.05)
        t = np.clip(resid / (tukey * scale), -1, 1)
        w = (1 - t ** 2) ** 2
        # Gauss-Newton on (cx, cy, r): d(rho - r)/d(cx,cy,r) = (-ux, -uy, -1)
        J = np.column_stack([-u[:, 0], -u[:, 1], -np.ones(len(p))])
        W = w[:, None]
        try:
            step = np.linalg.solve(J.T @ (W * J), -(J.T @ (w * resid)))
        except np.linalg.LinAlgError:
            break
        c = c + step[:2]
        r = r + step[2]
        if not np.isfinite([c[0], c[1], r]).all() or r <= 1:
            return None
        if np.abs(step).max() < 1e-4:
            break
    d = p - c
    resid = np.hypot(d[:, 0], d[:, 1]) - r
    if w.sum() <= 1e-9:
        return None
    return dict(centre=(float(c[0]), float(c[1])), radius=float(r), weights=w,
                resid=resid, rms=float(np.sqrt(np.average(resid ** 2, weights=w))))


def fit_centre(pts, c0, radius, iterations=8, tukey=2.5):
    """Robust centre fit with the radius held at the shot constant.

    This is the fit that matters. With the radius free, a frame whose lids leave only a
    short arc trades radius against centre and collapses -- on this plate the free radius
    ran down to 8.8 px on a 12.3 px iris, p5 over the shot. Holding the radius removes the
    degeneracy entirely, and it is the physically honest constraint: her iris is a rigid
    disc and only her distance to the camera can change its projected size.
    """
    p = np.asarray(pts, np.float64)
    c = np.array(c0, np.float64)
    for _ in range(iterations):
        d = p - c
        rho = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)
        u = d / rho[:, None]
        resid = rho - radius
        scale = max(1.4826 * np.median(np.abs(resid - np.median(resid))), 0.05)
        t = np.clip(resid / (tukey * scale), -1, 1)
        w = (1 - t ** 2) ** 2
        J = -u
        try:
            step = np.linalg.solve(J.T @ (w[:, None] * J), -(J.T @ (w * resid)))
        except np.linalg.LinAlgError:
            break
        c = c + step
        if not np.isfinite(c).all():
            return None
        if np.abs(step).max() < 1e-4:
            break
    d = p - c
    resid = np.hypot(d[:, 0], d[:, 1]) - radius
    scale = max(1.4826 * np.median(np.abs(resid - np.median(resid))), 0.05)
    w = (1 - np.clip(resid / (tukey * scale), -1, 1) ** 2) ** 2
    if w.sum() <= 1e-9:
        return None
    rms = float(np.sqrt(np.average(resid ** 2, weights=w)))
    # Standard error of the centre along its worst-determined axis. On a narrowed eye
    # the visible limbus is two short arcs near the nasal and temporal corners, and the
    # information they carry about the VERTICAL centre is the small eigenvalue of
    # sum(u u^T). Gating on this rather than on a raw arc fraction is the difference
    # between abstaining on 37 frames and abstaining only where the eye is really shut.
    d = p - c
    u = d / np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)[:, None]
    info = u.T @ (w[:, None] * u)
    eig = np.linalg.eigvalsh(info)
    se = float(rms / np.sqrt(max(eig[0], 1e-9)))
    return dict(centre=(float(c[0]), float(c[1])), radius=float(radius), weights=w,
                resid=resid, rms=rms, centre_se_px=se,
                inliers=int((np.abs(resid) < 2 * scale).sum()))


def fit_ellipse(pts, c0, ax0, ay0, iterations=8, tukey=2.5):
    """Robust axis-aligned ellipse fit, for measuring whether the extra parameter earns
    its place. Same IRLS, four parameters, residual in the same radial pixels."""
    p = np.asarray(pts, np.float64)
    c = np.array(c0, np.float64)
    a, b = float(ax0), float(ay0)
    for _ in range(iterations):
        dx, dy = p[:, 0] - c[0], p[:, 1] - c[1]
        rho = np.hypot(dx, dy)
        rho = np.where(rho < 1e-9, 1e-9, rho)
        ct, st = dx / rho, dy / rho
        # Radius of the ellipse along this direction.
        q = np.sqrt((ct / a) ** 2 + (st / b) ** 2)
        r_e = 1.0 / np.where(q < 1e-9, 1e-9, q)
        resid = rho - r_e
        scale = max(1.4826 * np.median(np.abs(resid - np.median(resid))), 0.05)
        t = np.clip(resid / (tukey * scale), -1, 1)
        w = (1 - t ** 2) ** 2
        dr_da = (r_e ** 3) * (ct ** 2) / a ** 3
        dr_db = (r_e ** 3) * (st ** 2) / b ** 3
        J = np.column_stack([-ct, -st, -dr_da, -dr_db])
        try:
            step = np.linalg.solve(J.T @ (w[:, None] * J), -(J.T @ (w * resid)))
        except np.linalg.LinAlgError:
            break
        c = c + step[:2]
        a, b = a + step[2], b + step[3]
        if not np.isfinite([c[0], c[1], a, b]).all() or min(a, b) <= 1:
            return None
        if np.abs(step).max() < 1e-4:
            break
    dx, dy = p[:, 0] - c[0], p[:, 1] - c[1]
    rho = np.hypot(dx, dy)
    ct, st = dx / np.maximum(rho, 1e-9), dy / np.maximum(rho, 1e-9)
    r_e = 1.0 / np.sqrt((ct / a) ** 2 + (st / b) ** 2)
    resid = rho - r_e
    return dict(centre=(float(c[0]), float(c[1])), ax=float(a), ay=float(b),
                resid=resid, rms=float(np.sqrt(np.mean(resid ** 2))))


def fit_eye(values, points, topology, side, model='circle', radius=None, margin=10,
            min_arc=MIN_ARC, min_angles=MIN_ANGLES, max_se=MAX_SE):
    """Fit one eye in one frame. Returns a record, always, including why it abstained."""
    landmark, lax, lay = r32.limbus_ellipse(points, topology['iris_' + side])
    rec = dict(side=side, landmark_centre=[float(landmark[0]), float(landmark[1])],
               landmark_axes=[float(lax), float(lay)], fitted=False, reason=None)
    if not np.isfinite([lax, lay]).all() or min(lax, lay) < 4:
        rec['reason'] = 'degenerate_landmark'
        return rec
    r0 = float(radius) if radius else float(0.5 * (lax + lay))
    reach = r0 * 1.7 + margin
    h, w = values.shape[:2]
    x0 = int(np.floor(landmark[0] - reach)); x1 = int(np.ceil(landmark[0] + reach)) + 1
    y0 = int(np.floor(landmark[1] - reach)); y1 = int(np.ceil(landmark[1] + reach)) + 1
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        rec['reason'] = 'eye_crop_leaves_frame'
        return rec
    window = (slice(y0, y1), slice(x0, x1))
    luma = display_luma(values[window])
    gx, gy = gradient_field(luma)
    lid = r32.lid_distance(points, topology['eyelid_' + side], window)
    th = np.arange(ANGLES, dtype=np.float32) * (2 * np.pi / ANGLES)
    cos, sin = np.cos(th), np.sin(th)

    keep = valid_angles(lid, x0, y0, landmark, r0, cos, sin)
    if keep.sum() < min_angles or keep.mean() < min_arc:
        rec.update(reason='limbus_occluded', open_arc=float(keep.mean()))
        return rec
    if radius:
        radii = np.float32([radius])
    else:
        radii = np.arange(0.70 * r0, 1.30 * r0 + 1e-9, 0.125, dtype=np.float32)
    cx, cy, rr, score = daugman(gx, gy, x0, y0, landmark, radii, cos, sin, keep)
    # Re-gate on the new circle, then a finer global pass around it.
    keep = valid_angles(lid, x0, y0, (cx, cy), rr, cos, sin)
    if keep.sum() < min_angles or keep.mean() < min_arc:
        rec.update(reason='limbus_occluded', open_arc=float(keep.mean()))
        return rec
    fine_r = np.float32([radius]) if radius else np.arange(rr - 0.75, rr + 0.75 + 1e-9, 0.0625,
                                                          dtype=np.float32)
    cx, cy, rr, score = daugman(gx, gy, x0, y0, (cx, cy), fine_r, cos, sin, keep,
                                span=0.75, step=0.0625)

    r_edge, peak, ok = ray_edges(gx, gy, x0, y0, (cx, cy), rr, cos, sin, keep)
    strong = ok & (peak > 0.25 * np.median(peak[ok]) if ok.any() else ok)
    idx = np.where(keep)[0][strong]
    if strong.sum() < min_angles:
        rec.update(reason='too_few_edges', edges=int(strong.sum()), open_arc=float(keep.mean()))
        return rec
    pts = np.column_stack([cx + r_edge[strong] * cos[idx], cy + r_edge[strong] * sin[idx]])
    if model == 'ellipse':
        fit = fit_ellipse(pts, (cx, cy), rr, rr)
        if fit is None:
            rec['reason'] = 'ellipse_fit_failed'
            return rec
        rec.update(fitted=True, centre=list(fit['centre']), axes=[fit['ax'], fit['ay']])
    elif radius:                         # the shot radius is fixed; only the centre moves
        fit = fit_centre(pts, (cx, cy), float(radius))
        if fit is None:
            rec['reason'] = 'centre_fit_failed'
            return rec
        rec.update(centre_se_px=round(fit['centre_se_px'], 4), inliers=fit['inliers'])
        if fit['centre_se_px'] > max_se:
            rec.update(reason='centre_underdetermined', open_arc=float(keep.mean()),
                       rms=float(fit['rms']))
            return rec
        rec.update(fitted=True, centre=list(fit['centre']),
                   axes=[fit['radius'], fit['radius']])
    else:
        fit = fit_circle(pts, (cx, cy), rr)
        if fit is None:
            rec['reason'] = 'circle_fit_failed'
            return rec
        rec.update(fitted=True, centre=list(fit['centre']),
                   axes=[fit['radius'], fit['radius']])
    rec.update(daugman=[round(cx, 4), round(cy, 4), round(rr, 4), round(score, 6)],
               edges=int(strong.sum()), open_arc=float(keep.mean()), rms=float(fit['rms']),
               edge_points=[[round(float(a), 3), round(float(b), 3)] for a, b in pts])
    return rec


def local_linear(t, y, valid, half=3, tukey=2.5):
    """Zero-lag robust smoothing: a local LINEAR fit centred on each frame.

    Centred, so a constant velocity passes through untouched -- there is no group delay
    to add the "delayed, slow to react" the artist saw. A causal filter cannot say that.
    """
    t = np.asarray(t, float); y = np.asarray(y, float); valid = np.asarray(valid, bool)
    out = np.array(y, float)
    if half <= 0:
        return out
    for i in range(len(t)):
        if not valid[i]:
            continue
        sel = valid & (np.abs(t - t[i]) <= half)
        if sel.sum() < 3:
            continue
        dt = t[sel] - t[i]
        v = y[sel]
        w = np.ones(sel.sum())
        for _ in range(3):
            A = np.column_stack([np.ones(len(dt)), dt])
            try:
                beta = np.linalg.solve(A.T @ (w[:, None] * A), A.T @ (w * v))
            except np.linalg.LinAlgError:
                beta = None
                break
            resid = v - A @ beta
            scale = max(1.4826 * np.median(np.abs(resid - np.median(resid))), 0.02)
            u = np.clip(resid / (tukey * scale), -1, 1)
            w = (1 - u ** 2) ** 2
        if beta is not None:
            out[i] = beta[0]
    return out



def bridge_gaps(rows, side, max_gap):
    """Fill short runs of failed frames by interpolating the CENTRE between good fits.

    A frame where the lids leave only one short arc has a degenerate centre -- the
    information matrix collapses and the standard error runs to thousands of pixels --
    so it abstains, and the fill abstains with it and the softbox pops back for a single
    frame. The artist sees exactly that at source 716.

    Interpolation is the right answer there and it is not a fudge: the fitted trajectory
    is smooth to **0.048 px against a local quadratic through its own neighbours**, so a
    one- or two-frame gap between two good fits is known far better by its neighbours
    than by any fit this frame can support. Long gaps are blinks and stay abstained --
    there is no eye to interpolate across.
    """
    idx = [(j, next((e for e in row['eyes'] if e['side'] == side), None))
           for j, row in enumerate(rows)]
    good = [(rows[j]['frame'], e) for j, e in idx if e is not None and e.get('fitted')]
    if len(good) < 2:
        return 0
    frames = np.array([g[0] for g in good], float)
    cx = np.array([g[1]['centre'][0] for g in good])
    cy = np.array([g[1]['centre'][1] for g in good])
    radius = float(np.median([g[1]['axes'][0] for g in good]))
    bridged = 0
    for j, e in idx:
        if e is None or e.get('fitted'):
            continue
        f = rows[j]['frame']
        before = frames[frames < f]
        after = frames[frames > f]
        if not len(before) or not len(after):
            continue
        gap = int(after[0] - before[-1]) - 1
        if gap > max_gap:
            continue
        e.update(fitted=True, bridged=True, gap_frames=gap,
                 centre=[float(np.interp(f, frames, cx)), float(np.interp(f, frames, cy))],
                 axes=[radius, radius], reason='bridged_from_neighbours')
        bridged += 1
    return bridged


class Track:
    """A fitted limbus track, keyed by (frame, side), with the landmark as fallback."""

    def __init__(self, data):
        self.meta = {k: v for k, v in data.items() if k != 'rows'}
        self.rows = {}
        for row in data['rows']:
            for eye in row['eyes']:
                if eye.get('fitted'):
                    self.rows[(int(row['frame']), eye['side'])] = (
                        tuple(eye['smoothed_centre']), float(eye['smoothed_axes'][0]),
                        float(eye['smoothed_axes'][1]))

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding='utf-8')))

    def get(self, frame, side):
        """(centre, ax, ay) for this eye, or None so the caller keeps the landmark."""
        return self.rows.get((int(frame), side))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_shot_argument(ap)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--count', type=int, default=150)
    ap.add_argument('--model', choices=('circle', 'ellipse'), default='circle')
    ap.add_argument('--radius-model', choices=('shot', 'per-frame'), default='shot',
                    help="'shot' fits one radius for the whole sequence (her iris does "
                         "not change size); 'per-frame' leaves it free, for comparison")
    ap.add_argument('--calib-arc', type=float, default=0.42,
                    help='minimum unoccluded fraction of the limbus for a frame to vote '
                         'on the shot radius')
    ap.add_argument('--calib-rms', type=float, default=0.35,
                    help='maximum edge residual, native px, for a calibration frame')
    ap.add_argument('--trust-se', type=float, default=0.35,
                    help='centre standard error, native px, at which a frame is half '
                         'trusted against its neighbours. The fit reports its own '
                         'uncertainty along the worst-determined axis, which on this '
                         'plate is VERTICAL and is large exactly where the lid hides the '
                         'top of the iris; 0 disables the shrinkage.')
    ap.add_argument('--trust-half', type=int, default=2,
                    help='half-width, in frames, of the zero-lag local-linear predictor '
                         'the uncertain frames lean on. It is fitted over CONFIDENT '
                         'frames only.')
    ap.add_argument('--bridge-gap', type=int, default=3,
                    help='interpolate the centre across failed runs of at most this many '
                         'frames. A single dropped frame puts the softbox back for one '
                         'frame and the eye pops; the trajectory is smooth to 0.05 px, '
                         'so its neighbours know that frame better than it knows itself. '
                         '0 disables it.')
    ap.add_argument('--smooth-half', type=int, default=0,
                    help='half-width in frames of the centred local-linear smoother. 0 '
                         'is the default and the measured right answer: the per-frame '
                         'fit is already smooth to 0.05 px, and smoothing only adds lag')
    ap.add_argument('--out', default='limbus_track.json')
    args = ap.parse_args()
    shot = from_args(args)

    meta = json.loads((shot.relight / 'face_landmarks.json').read_text())
    topology = meta['topology']
    with np.load(shot.relight / 'face_landmarks.npz') as z:
        landmarks = {k: z[k] for k in ('points', 'found', 'eyelid_aperture')}
    indices = list(range(args.start, args.start + args.count))
    if min(indices) < 0 or max(indices) >= shot.frame_count:
        raise ValueError('Requested interval leaves the shot')

    values = {}
    for i in indices:
        raw = cv2.imread(str(shot.fg(i)), cv2.IMREAD_UNCHANGED)
        if raw is None or raw.dtype != np.float32 or raw.ndim != 3:
            raise ValueError('Expected scene-linear keyed RGB EXR at %s' % shot.fg(i))
        values[i] = raw[..., ::-1].astype(np.float32)

    def pass_over(radius_by_side):
        rows = []
        for i in indices:
            eyes = []
            if landmarks['found'][i] and np.isfinite(landmarks['points'][i]).all():
                for side in r32.SIDES:
                    eyes.append(fit_eye(values[i], landmarks['points'][i], topology, side,
                                        model=args.model,
                                        radius=(radius_by_side or {}).get(side)))
            rows.append(dict(frame=i, source_frame=shot.source_start_frame + i, eyes=eyes))
        return rows

    rows = pass_over(None)
    free = {}
    for side in r32.SIDES:
        r = [e['axes'][0] for row in rows for e in row['eyes']
             if e['side'] == side and e.get('fitted')]
        # The shot radius is calibrated only on frames where the lids leave a real arc
        # and the edge agrees with a circle. A half-closed frame's free radius is the
        # degeneracy this constant exists to remove; it must not vote on the constant.
        good = [e['axes'][0] for row in rows for e in row['eyes']
                if e['side'] == side and e.get('fitted')
                and e['open_arc'] >= args.calib_arc and e['rms'] <= args.calib_rms]
        free[side] = dict(n=len(r), median=float(np.median(r)) if r else None,
                          p5=float(np.percentile(r, 5)) if r else None,
                          p95=float(np.percentile(r, 95)) if r else None,
                          sd=float(np.std(r)) if r else None,
                          calibration_frames=len(good),
                          calibration_median=float(np.median(good)) if good else None,
                          calibration_sd=float(np.std(good)) if good else None,
                          calibration_p5=float(np.percentile(good, 5)) if good else None,
                          calibration_p95=float(np.percentile(good, 95)) if good else None)
    fixed = None
    if args.radius_model == 'shot' and args.model == 'circle':
        fixed = {s: free[s]['calibration_median'] for s in r32.SIDES
                 if free[s]['calibration_median']}
        if len(fixed) != len(r32.SIDES):
            raise ValueError('No frame passed radius calibration for %s'
                             % sorted(set(r32.SIDES) - set(fixed)))
        rows = pass_over(fixed)

    bridged = {}
    for side in r32.SIDES:
        bridged[side] = bridge_gaps(rows, side, args.bridge_gap)

    # Zero-lag smoothing of the centre; the radius is already a shot constant unless the
    # caller asked for it per frame, in which case it is smoothed the same way.
    out_rows = []
    for side in r32.SIDES:
        t, cx, cy, ax, ay, ok = [], [], [], [], [], []
        for row in rows:
            e = next((e for e in row['eyes'] if e['side'] == side), None)
            t.append(row['frame'])
            good = bool(e and e.get('fitted'))
            ok.append(good)
            cx.append(e['centre'][0] if good else np.nan)
            cy.append(e['centre'][1] if good else np.nan)
            ax.append(e['axes'][0] if good else np.nan)
            ay.append(e['axes'][1] if good else np.nan)
        ok = np.array(ok)
        sx = local_linear(t, np.nan_to_num(cx), ok, args.smooth_half)
        sy = local_linear(t, np.nan_to_num(cy), ok, args.smooth_half)
        sa = local_linear(t, np.nan_to_num(ax), ok, args.smooth_half)
        sb = local_linear(t, np.nan_to_num(ay), ok, args.smooth_half)
        # **The centre's error is VERTICAL, and it is worst exactly where the lid hides
        # the top of the iris.** Measured against the independent estimator in
        # `audit_r35_limbus.py`: |offset_x| is 0.05 px median and |offset_y| 0.42-0.50,
        # with excursions to -4.94 px, and every one of those sits on a frame with a
        # short arc. The geometry says why -- with the radius held at the shot constant,
        # a visible arc that is only the sides carries almost no information about the
        # centre's HEIGHT, so the circle slides down and snaps back when the lid opens.
        # That is the artist's *"the iris didn't follow along at the exact right time"*
        # and *"jump into position ... as it corrects the mask offset suddenly"*.
        #
        # `centre_se_px` already measures precisely that: it is the standard error along
        # the worst-determined axis, the small eigenvalue of sum(u u^T). So the fix is
        # not to smooth the track -- round 35 measured that as 3.4 px of LAG at a fast
        # head move and set `--smooth-half 0` for good reason -- but to shrink each
        # frame's centre toward its neighbours IN PROPORTION TO ITS OWN UNCERTAINTY.
        # A frame the fit resolved (se ~ 0.1 px) is untouched; a frame where the lid left
        # only two side arcs (se > 1 px) leans on the frames either side of it, which is
        # what a human tracker would do and what `--bridge-gap` already does for a frame
        # that failed outright.
        #
        # The predictor is the same zero-lag local linear one, run over the CONFIDENT
        # frames only, so an uncertain frame cannot pull its own reference.
        se = np.array([ (next((e for e in row['eyes'] if e['side'] == side), None) or {})
                        .get('centre_se_px', np.nan) for row in rows ], float)
        conf = ok & np.isfinite(se) & (se <= args.trust_se)
        if conf.sum() >= 5 and args.trust_se > 0:
            px = local_linear(t, np.nan_to_num(cx), conf, max(args.trust_half, 1))
            py = local_linear(t, np.nan_to_num(cy), conf, max(args.trust_half, 1))
            lam = np.where(np.isfinite(se), se ** 2 / (se ** 2 + args.trust_se ** 2), 0.0)
            lam = np.clip(lam, 0.0, 1.0) * ok
            sx = (1 - lam) * sx + lam * px
            sy = (1 - lam) * sy + lam * py
            moved = np.hypot(sx - np.nan_to_num(cx), sy - np.nan_to_num(cy))[ok]
            print('LIMBUS_TRUST %s confident %d of %d, shrunk %d frames, moved median '
                  '%.3f px p95 %.3f max %.3f' % (side, int(conf.sum()), int(ok.sum()),
                  int((lam > 0.05).sum()), float(np.median(moved)),
                  float(np.percentile(moved, 95)), float(moved.max())), flush=True)
        for j, row in enumerate(rows):
            e = next((e for e in row['eyes'] if e['side'] == side), None)
            if e is None or not ok[j]:
                continue
            e['smoothed_centre'] = [round(float(sx[j]), 4), round(float(sy[j]), 4)]
            e['smoothed_axes'] = [round(float(sa[j]), 4), round(float(sb[j]), 4)]
            e['smoothing_shift_px'] = round(float(np.hypot(sx[j] - e['centre'][0],
                                                           sy[j] - e['centre'][1])), 4)

    out_rows = rows

    fitted = {s: sum(1 for row in rows for e in row['eyes']
                     if e['side'] == s and e.get('fitted')) for s in r32.SIDES}
    payload = dict(rows=out_rows, frames=len(indices), model=args.model,
                   bridged=bridged, bridge_gap=args.bridge_gap,
                   radius_model=args.radius_model, fitted=fitted,
                   free_radius=free, shot_radius=fixed,
                   smooth_half_frames=args.smooth_half, sigma_px=SIGMA,
                   trust_se_px=args.trust_se, trust_half_frames=args.trust_half,
                   angles=ANGLES, lid_min_px=LID_MIN,
                   method='Daugman integro-differential global search, then per-angle '
                          'sub-pixel edges and a robust geometric fit; zero-lag centred '
                          'local-linear temporal smoothing',
                   production_enabled=False, artist_accepted=False,
                   code_sha256=r32.sha(Path(__file__)),
                   track_sha256=r32.sha(shot.relight / 'face_landmarks.npz'))
    r32.write_json(shot.relight / args.out, payload)
    print('fitted %s of %d frames (bridged %s)' % (fitted, len(indices), bridged))
    for side in r32.SIDES:
        f = free[side]
        print('  %-5s free radius: median %.3f px  p5 %.3f  p95 %.3f  sd %.3f'
              % (side, f['median'], f['p5'], f['p95'], f['sd']))
        print('        calibration (%d frames): median %.3f px  p5 %.3f  p95 %.3f  sd %.3f'
              % (f['calibration_frames'], f['calibration_median'], f['calibration_p5'],
                 f['calibration_p95'], f['calibration_sd']))
        rms = [e['rms'] for row in rows for e in row['eyes']
               if e['side'] == side and e.get('fitted')]
        sh = [e['smoothing_shift_px'] for row in rows for e in row['eyes']
              if e['side'] == side and 'smoothing_shift_px' in e]
        print('        edge residual rms: median %.3f px  p95 %.3f  max %.3f'
              % (np.median(rms), np.percentile(rms, 95), max(rms)))
        print('        smoothing moved the centre: median %.3f px  p95 %.3f  max %.3f'
              % (np.median(sh), np.percentile(sh, 95), max(sh)))
    print('wrote %s' % (shot.relight / args.out))


if __name__ == '__main__':
    main()
