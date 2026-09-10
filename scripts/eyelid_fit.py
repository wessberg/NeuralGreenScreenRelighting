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


def _profile(field, xs, ys):
    return lf._sample(field, xs, ys)


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
        hit_spec = _profile(spec, xx, yy) > 0.05
        block = hit_spec.copy()
        for shift in (1, 2, 3, 4, 5):
            block[shift:] |= hit_spec[:-shift]
            block[:-shift] |= hit_spec[shift:]
        grad = np.where(block, 0.0, grad)
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


def fit_eye(values, centre, radius, corners, margin=14):
    """Both lids for one eye in one frame, or the reason it abstains."""
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
                                  spec=spec if sign < 0 else None)
        out[name] = dict(xs=xs, ys=ys, ss=ss)
    rec.update(window=(x0, y0, x1, y1), local_centre=local, samples=out)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_shot_argument(ap)
    ap.add_argument('--track', default='limbus_track.json')
    ap.add_argument('--out', default='eyelid_track.json')
    ap.add_argument('--free-curvature', action='store_true',
                    help='fit the parabola curvature per frame instead of holding one '
                         'per eye for the shot. A lid moves; its shape does not.')
    args = ap.parse_args()
    shot = from_args(args)
    track = lf.Track.load(shot.relight / args.track)
    data = json.loads((shot.relight / args.track).read_text(encoding='utf-8'))
    topology = json.loads((shot.relight / 'face_landmarks.json').read_text())['topology']
    with np.load(shot.relight / 'face_landmarks.npz') as z:
        points = z['points']

    raw_fits = {}
    for row in data['rows']:
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
            raw_fits[(i, side)] = (fit_eye(values, centre, radius, corners), centre,
                                   radius, row['source_frame'])

    # One curvature per eye for the shot, from the frames that fit it freely and well.
    curvature = {}
    if not args.free_curvature:
        for side in r32.SIDES:
            for name in ('upper', 'lower'):
                got = []
                for (i, s), (rec, centre, radius, _) in raw_fits.items():
                    if s != side or 'samples' not in rec:
                        continue
                    d = rec['samples'][name]
                    f = fit_parabola(d['xs'], d['ys'], d['ss'], rec['local_centre'])
                    if f and f['rms'] <= 1.2 and f['columns'] >= 24:
                        got.append(f['coef'][0])
                curvature[(side, name)] = float(np.median(got)) if len(got) >= 10 else None

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
            for name in ('upper', 'lower'):
                d = rec['samples'][name]
                f = fit_parabola(d['xs'], d['ys'], d['ss'], rec['local_centre'],
                                 curvature=curvature.get((side, name)))
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
                   min_step=MIN_STEP, column_step=COLUMN_STEP,
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
