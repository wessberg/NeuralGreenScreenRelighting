"""Is the anchor on her iris? Measured by an estimator that shares nothing with the fit.

The rule this project keeps relearning: when the artist sees an artefact and the numbers
do not, suspect a metric that shares the bug's assumption. `limbus_fit.py` finds the
limbus as a **peak of the Gaussian derivative of Rec.709 luma**, sampled bicubically from
a gradient field, and fits a circle to it. If this audit did the same it would only
confirm that the fit converged.

So this estimator is different in every part:

* **the feature** is the MAX channel, round 34's audit domain, not luma;
* **the statistic** is a LEVEL crossing -- the radius where the profile passes halfway
  between the iris plateau and the sclera plateau -- not a derivative peak, so its noise
  and its bias are unrelated;
* **the sampling** is bilinear on the image itself, not bicubic on a smoothed derivative;
* **the model** is a first harmonic of r(theta), fitted independently per frame.

That last part is what makes the number readable. If a candidate circle is centred on the
iris, the crossing radius is the same at every angle. If it is displaced by (a, b), the
crossing radius reads

    r(theta) = R + a cos(theta) + b sin(theta) + O(offset^2 / R)

so the first harmonic IS the displacement, in native pixels, and R over the candidate
radius is the size error. Both are reported for the fitted track and for the mediapipe
landmark on **exactly the same frames and the same angles**, which is the paired control.

`--synthetic` runs both estimators on a rendered disc of known centre and radius, so each
one's own bias is bounded before either is used to judge the other.

    dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/audit_r35_limbus.py \
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

ANGLES = 96
INNER = (0.55, 0.80)     # iris plateau, clear of the pupil and of the limbal ring
OUTER = (1.25, 1.55)     # sclera plateau, clear of the transition
BAND = (0.80, 1.30)      # where the crossing is looked for
STEP = 0.05              # native px between profile samples
MIN_CONTRAST = 0.06      # display levels; below this the angle carries no limbus


def max_channel(values):
    return np.clip(linear_to_srgb(np.asarray(values, np.float32)), 0, 1).max(-1)


def crossings(image, centre, radius, cos, sin, keep):
    """Level-crossing radius per kept angle, in units of `radius`.

    No derivative anywhere: the iris and sclera plateaus are medians of the profile, and
    the edge is where it passes halfway between them. The outermost crossing is taken, so
    a dark lash inside the iris cannot be mistaken for the limbus.
    """
    c, s = cos[keep], sin[keep]
    lo = np.arange(BAND[0] * radius, BAND[1] * radius + 1e-9, STEP, dtype=np.float32)
    xs = centre[0] + lo[:, None] * c[None, :]
    ys = centre[1] + lo[:, None] * s[None, :]
    prof = cv2.remap(image, xs.astype(np.float32), ys.astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def plateau(rng):
        rr = np.arange(rng[0] * radius, rng[1] * radius + 1e-9, STEP, dtype=np.float32)
        px = centre[0] + rr[:, None] * c[None, :]
        py = centre[1] + rr[:, None] * s[None, :]
        return np.median(cv2.remap(image, px.astype(np.float32), py.astype(np.float32),
                                   cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE), axis=0)

    iris, sclera = plateau(INNER), plateau(OUTER)
    contrast = sclera - iris
    level = 0.5 * (iris + sclera)
    above = prof > level[None, :]
    out = np.full(prof.shape[1], np.nan)
    for j in range(prof.shape[1]):
        if contrast[j] < MIN_CONTRAST:
            continue
        idx = np.where(~above[:, j])[0]
        if not len(idx) or idx[-1] >= len(lo) - 1:
            continue
        k = idx[-1]                       # last sample below the level: the outermost edge
        a, b = prof[k, j], prof[k + 1, j]
        f = 0.0 if b == a else (level[j] - a) / (b - a)
        out[j] = (lo[k] + f * STEP) / radius
    return out, contrast


def harmonic(r, cos, sin, keep, tukey=2.5, iterations=4):
    """Robust fit of r(theta) = R + a cos + b sin. (a, b) is the displacement, in radii."""
    ok = np.isfinite(r)
    if ok.sum() < 12:
        return None
    c, s = cos[keep][ok], sin[keep][ok]
    y = r[ok]
    A = np.column_stack([np.ones(len(y)), c, s])
    w = np.ones(len(y))
    beta = None
    for _ in range(iterations):
        try:
            beta = np.linalg.solve(A.T @ (w[:, None] * A), A.T @ (w * y))
        except np.linalg.LinAlgError:
            return None
        resid = y - A @ beta
        scale = max(1.4826 * np.median(np.abs(resid - np.median(resid))), 1e-3)
        w = (1 - np.clip(resid / (tukey * scale), -1, 1) ** 2) ** 2
    resid = y - A @ beta
    return dict(R=float(beta[0]), a=float(beta[1]), b=float(beta[2]), n=int(ok.sum()),
                rms=float(np.sqrt(np.mean(resid ** 2))))


def measure(image, lid, x0, y0, centre, radius, cos, sin, lid_min=1.25):
    """One candidate circle, judged against the photograph. Two passes: the harmonic is
    only first order in the displacement, so the second pass is taken about the corrected
    centre, where it is not."""
    from limbus_fit import valid_angles
    keep = valid_angles(lid, x0, y0, centre, radius, cos, sin, lid_min)
    if keep.sum() < 14:
        return None
    r, _ = crossings(image, centre, radius, cos, sin, keep)
    first = harmonic(r, cos, sin, keep)
    if first is None:
        return None
    moved = (centre[0] + first['a'] * radius, centre[1] + first['b'] * radius)
    keep2 = valid_angles(lid, x0, y0, moved, radius, cos, sin, lid_min)
    if keep2.sum() < 14:
        return dict(first, offset_px=float(np.hypot(first['a'], first['b']) * radius),
                    size_ratio=first['R'], refined=False, centre=list(centre))
    r2, _ = crossings(image, moved, radius, cos, sin, keep2)
    second = harmonic(r2, cos, sin, keep2)
    if second is None:
        return dict(first, offset_px=float(np.hypot(first['a'], first['b']) * radius),
                    size_ratio=first['R'], refined=False, centre=list(centre))
    truth = (moved[0] + second['a'] * radius, moved[1] + second['b'] * radius)
    return dict(R=second['R'] * radius, n=second['n'], rms=float(second['rms'] * radius),
                offset_px=float(np.hypot(truth[0] - centre[0], truth[1] - centre[1])),
                offset_xy=[float(truth[0] - centre[0]), float(truth[1] - centre[1])],
                size_ratio=float(second['R']), refined=True,
                observed_centre=[float(truth[0]), float(truth[1])],
                observed_radius=float(second['R'] * radius))


def synthetic_control(radius=12.3, offset=(0.37, -0.21), blur=1.1, noise=0.004, seed=35):
    """Both estimators on a disc whose centre and radius are known exactly.

    Bounds each estimator's own bias before either is used to judge the other. The disc
    is 8x supersampled, blurred to the plate's own edge width, quantised to 8 bit and
    given read noise, because a synthetic control that skips quantisation is measuring a
    different problem from the one the plate poses.
    """
    import limbus_fit as lf
    rng = np.random.default_rng(seed)
    size, sup = 64, 8
    cy = cx = size / 2
    tx, ty = cx + offset[0], cy + offset[1]
    yy, xx = np.mgrid[:size * sup, :size * sup]
    xx = (xx + 0.5) / sup - 0.5
    yy = (yy + 0.5) / sup - 0.5
    disc = (np.hypot(xx - tx, yy - ty) <= radius).astype(np.float32)
    img = cv2.resize(disc, (size, size), interpolation=cv2.INTER_AREA)
    img = 0.72 - 0.42 * img                       # sclera 0.72, iris 0.30, display levels
    k = int(2 * np.ceil(3 * blur) + 1)
    img = cv2.GaussianBlur(img, (k, k), blur, borderType=cv2.BORDER_REPLICATE)
    img = img + rng.normal(0, noise, img.shape).astype(np.float32)
    img = np.rint(np.clip(img, 0, 1) * 255).astype(np.uint8).astype(np.float32) / 255.0

    th = np.arange(ANGLES, dtype=np.float32) * (2 * np.pi / ANGLES)
    cos, sin = np.cos(th), np.sin(th)
    keep = np.ones(ANGLES, bool)
    lid = np.full(img.shape, 99.0, np.float32)

    gx, gy = lf.gradient_field(img)
    th_f = np.arange(lf.ANGLES, dtype=np.float32) * (2 * np.pi / lf.ANGLES)
    cf, sf = np.cos(th_f), np.sin(th_f)
    keep_f = np.ones(lf.ANGLES, bool)
    dx, dy, dr, _ = lf.daugman(gx, gy, 0, 0, (cx, cy),
                               np.arange(0.8 * radius, 1.2 * radius, 0.125, np.float32),
                               cf, sf, keep_f)
    re, pk, okk = lf.ray_edges(gx, gy, 0, 0, (dx, dy), dr, cf, sf, keep_f)
    strong = okk & (pk > 0.25 * np.median(pk[okk]))
    idx = np.where(keep_f)[0][strong]
    pts = np.column_stack([dx + re[strong] * cf[idx], dy + re[strong] * sf[idx]])
    free = lf.fit_circle(pts, (dx, dy), dr)
    pinned = lf.fit_centre(pts, (dx, dy), radius)

    audit = measure(img, lid, 0, 0, (cx, cy), radius, cos, sin, lid_min=0.0)
    return dict(truth=dict(centre=[tx, ty], radius=radius),
                fitter_free=dict(centre=list(free['centre']), radius=free['radius'],
                                 centre_error_px=float(np.hypot(free['centre'][0] - tx,
                                                                free['centre'][1] - ty)),
                                 radius_error_px=float(free['radius'] - radius)),
                fitter_pinned=dict(centre=list(pinned['centre']),
                                   centre_error_px=float(np.hypot(pinned['centre'][0] - tx,
                                                                  pinned['centre'][1] - ty))),
                audit=dict(observed_centre=audit['observed_centre'],
                           observed_radius=audit['observed_radius'],
                           centre_error_px=float(np.hypot(audit['observed_centre'][0] - tx,
                                                          audit['observed_centre'][1] - ty)),
                           radius_error_px=float(audit['observed_radius'] - radius)))


def report(label, rows, key):
    v = np.array([r[key] for r in rows if r.get(key) is not None], float)
    if not len(v):
        return '%-28s no samples' % label
    return ('%-28s median %7.3f  p95 %7.3f  max %7.3f  n=%d'
            % (label, np.median(v), np.percentile(v, 95), v.max(), len(v)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_shot_argument(ap)
    ap.add_argument('--track', default='limbus_track.json')
    ap.add_argument('--out', default='r35_limbus_audit.json')
    args = ap.parse_args()
    shot = from_args(args)
    data = json.loads((shot.relight / args.track).read_text(encoding='utf-8'))
    meta = json.loads((shot.relight / 'face_landmarks.json').read_text())
    topology = meta['topology']
    with np.load(shot.relight / 'face_landmarks.npz') as z:
        points = z['points']

    th = np.arange(ANGLES, dtype=np.float32) * (2 * np.pi / ANGLES)
    cos, sin = np.cos(th), np.sin(th)
    out = {s: {'fit': [], 'landmark': []} for s in r32.SIDES}
    for row in data['rows']:
        i = row['frame']
        raw = cv2.imread(str(shot.fg(i)), cv2.IMREAD_UNCHANGED)
        image = max_channel(raw[..., ::-1].astype(np.float32))
        for eye in row['eyes']:
            if not eye.get('fitted'):
                continue
            side = eye['side']
            lc = eye['landmark_centre']
            lax, lay = eye['landmark_axes']
            fc = eye.get('smoothed_centre', eye['centre'])
            fr = eye.get('smoothed_axes', eye['axes'])[0]
            reach = max(fr, lax, lay) * 1.7 + 10
            x0 = int(np.floor(min(lc[0], fc[0]) - reach))
            y0 = int(np.floor(min(lc[1], fc[1]) - reach))
            x1 = int(np.ceil(max(lc[0], fc[0]) + reach)) + 1
            y1 = int(np.ceil(max(lc[1], fc[1]) + reach)) + 1
            if x0 < 0 or y0 < 0 or x1 > image.shape[1] or y1 > image.shape[0]:
                continue
            window = (slice(y0, y1), slice(x0, x1))
            lid = r32.lid_distance(points[i], topology['eyelid_' + side], window)
            crop = image[window]
            # Both candidates are judged on the same crop with the same estimator. The
            # landmark's ellipse is judged at its own mean radius, which is the fairest
            # single number for a circle-based statistic; its size error is reported
            # separately against each semi-axis.
            for name, centre, radius in (('fit', fc, fr),
                                         ('landmark', lc, 0.5 * (lax + lay))):
                local = (centre[0] - x0, centre[1] - y0)
                m = measure(crop, lid, 0, 0, local, radius, cos, sin)
                if m is None:
                    continue
                if m.get('observed_centre'):
                    m['observed_centre'] = [m['observed_centre'][0] + x0,
                                            m['observed_centre'][1] + y0]
                m.update(frame=i, source_frame=row['source_frame'], model_radius=radius)
                if name == 'landmark':
                    m['landmark_axes'] = [lax, lay]
                out[side][name].append(m)

    lines = []
    for side in r32.SIDES:
        lines.append('%s eye' % side)
        for name in ('landmark', 'fit'):
            rows = out[side][name]
            lines.append('  %-9s %s' % (name, report('offset from her iris, px', rows,
                                                     'offset_px')))
            lines.append('  %-9s %s' % ('', report('radius ratio (obs/model)', rows,
                                                   'size_ratio')))
            lines.append('  %-9s %s' % ('', report('crossing residual, px', rows, 'rms')))
        paired = {r['frame']: r for r in out[side]['fit']}
        both = [(paired[r['frame']]['offset_px'], r['offset_px'])
                for r in out[side]['landmark'] if r['frame'] in paired]
        if both:
            f = np.array([b[0] for b in both]); l = np.array([b[1] for b in both])
            lines.append('  paired on %d frames: fit %.3f px vs landmark %.3f px median '
                         '(fit better on %d)' % (len(both), np.median(f), np.median(l),
                                                 int((f < l).sum())))
        # Frame-to-frame change of the OBSERVED iris centre, per model. This is the
        # artist's "delayed, slow to react": a good anchor tracks her eye, so its error
        # should not move; a biased one has an error that breathes with the lids.
        for name in ('landmark', 'fit'):
            rows = sorted((r for r in out[side][name] if r.get('offset_xy')),
                          key=lambda r: r['frame'])
            if len(rows) < 3:
                continue
            fr = np.array([r['frame'] for r in rows])
            off = np.array([r['offset_xy'] for r in rows], float)
            adj = np.diff(fr) == 1
            if adj.sum() < 3:
                continue
            step = np.hypot(*np.diff(off, axis=0).T)[adj]
            lines.append('  %-9s error step between adjacent frames: median %.3f  '
                         'p95 %.3f  max %.3f' % (name, np.median(step),
                                                 np.percentile(step, 95), step.max()))
    control = synthetic_control()
    lines.append('synthetic control (known disc, 8-bit, blurred, noisy):')
    lines.append('  fitter free   centre err %.3f px  radius err %+.3f px'
                 % (control['fitter_free']['centre_error_px'],
                    control['fitter_free']['radius_error_px']))
    lines.append('  fitter pinned centre err %.3f px'
                 % control['fitter_pinned']['centre_error_px'])
    lines.append('  audit         centre err %.3f px  radius err %+.3f px'
                 % (control['audit']['centre_error_px'],
                    control['audit']['radius_error_px']))
    print('\n'.join(lines))

    r32.write_json(shot.logs / args.out, dict(
        rows=out, synthetic_control=control, summary=lines,
        estimator='max-channel level crossing at the iris/sclera half level, bilinear, '
                  'first harmonic of r(theta) about the candidate centre, two passes',
        independence='shares no feature, statistic, sampling or model with limbus_fit.py; '
                     'the eyelid gate is shared and is not part of either estimator',
        track=args.track, track_sha256=r32.sha(shot.relight / args.track),
        production_enabled=False, artist_accepted=False,
        code_sha256=r32.sha(Path(__file__))))


if __name__ == '__main__':
    main()
