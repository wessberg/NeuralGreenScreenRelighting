"""Remove the capture room corneal highlight using the eye own observed anatomy.

**REJECTED BY THE ARTIST, ROUND 32. Not a production path. Read rounds/round32.md
before extending or reusing any of this.** It removes the highlight and beats round
31 flat-gain fill by 27-42 percent on 12,082 withheld pixels, and it still flickers:
the filled colour traverses R 36->132 and B 2->66 across the shot, with an
adjacent-frame p95 of 12-16 and a max of 52-59 display levels.

The reason is structural, and it is the finding worth carrying forward. EVERY term of
the fill is refit from scratch on each frame -- the radial profile is a median over a
few dozen of that frame pixels, the angular gain is fitted on that frame, the donors
are that frame pixels. Nothing carries across frames. So although nothing here fits an
explicit per-frame gain, which is what round 31 did wrong, it reproduces round 31
symptom by a different mechanism. **No per-frame-refit fill can be temporally stable
at a 24-pixel iris.** A next attempt needs the eye estimated ONCE over the sequence in
a stabilized frame, with only a smooth, explicitly constrained per-frame illumination
term.

What the method does, for the record:

1. Limbus ellipse from the four mediapipe cardinal iris landmarks. This is the only
   stable anchor the plate carries. Centring the polar frame on a fitted dark core was
   tried and is wrong -- that core is the highlight own undershoot rim, so centring on
   it puts the highlight at small radius where it dominates its own profile bins and
   hides itself.
2. Appearance model = per-channel median radial profile out to rho 0.88, times a
   smooth angular shading gain fitted on PER-ANGLE MEDIANS. Two things here are load
   bearing and were bugs first. The model must stop INSIDE the limbus: the sclera is
   bright and sits left and right of the iris but not above and below it, so it is not
   radially symmetric about the iris centre, and including it made 106-197 of about
   460 visible pixels look anomalous while the highlight did not stand out. And the
   gain must be fitted robustly: an ordinary least squares fit on the ratio lets the
   highlight dozen pixels at ratio near ten drag the whole model upward.
3. Footprint = sign-aware bounded growth on the residual. The highlight is a positive
   core; the coding undershoot beside it is a negative rim; iris texture is both signs
   at random, so growing on the absolute residual cannot tell them apart. Growth is
   bounded, not a flood fill, and confined near the seed. The size guard is ABSOLUTE,
   because a fraction-of-visible-area cap abstains on half-shut eyes whose footprints
   are perfectly normal in size, and that produces removal on one frame and not the
   next.
4. **The dark rim is NOT repaired.** It is coding undershoot clipped at legal black by
   the tagged limited-range expansion -- green and blue exactly 0, red 8-26 -- but it
   is also the only dark material the plate carries around the pupil, and the artist
   reads it as the pupil. Replacing it made the repaired eye uniformly brown where the
   source is black, which he rejected on sight.
5. Completion = the model plus an angular interpolation of the RESIDUAL from the same
   frame at the same normalized radius. Measured contribution: essentially none.
   9.4311 display levels against 9.7077 for the radial profile alone.

Two claims an earlier pass of this round made and had to retract, recorded so they are
not inherited: that the plate carries no resolvable pupil edge (an artefact of
profiling about the limbus centre -- but once the highlight AND its rim are excluded a
pupil does not localize either, scattering over the whole +/-4 px search range on 6-10
pixels of evidence), and that centring on the fitted dark core is a temporal
constraint (it is not; see 1).

Everything outside the reported edit mask is bit-exact source. Frames whose eye is
closed or whose track is invalid abstain and keep exact source; that is recorded.

    dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/eye_reflection_repair.py \
        --shot music_videos/reporter_test/shots/reporter_a

This writes no production default.
"""
import argparse
import hashlib
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

SIDES = ('left', 'right')
# Radial binning of the eye. 0 is the iris centre, 1.0 the limbus. The model stops
# just INSIDE the limbus. Reaching past it was the round's main bug: the sclera is
# bright and sits left and right of the iris but not above and below it, so it is not
# radially symmetric about the iris centre. The limbal ring just inside it is also
# left out: it is a real radial feature, but its radius varies with angle, so it
# smears across the bins and inflates the spread. Including the sclera made the profile and
# the angular gain fit iris, sclera and lid together, the residual was then large
# everywhere, and 106-197 of about 460 visible pixels tripped the 4-sigma level while
# the highlight itself did not stand out. No threshold could have fixed that.
RHO_MAX = 0.88
RHO_STEP = 0.055
# Search half-width, in native pixels, for the radial centre around the limbus landmark.
CENTRE_SEARCH = 3.5
CENTRE_STEP = 0.25


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def rgb8(linear):
    return np.rint(np.clip(linear_to_srgb(np.asarray(linear, np.float32)), 0, 1) * 255).astype(np.uint8)


def limbus_ellipse(points, iris_ids):
    """Axis-aligned limbus from the four mediapipe cardinal iris landmarks.

    The landmark circle was checked against the plate at 24x before it was trusted; it
    tracks the visible limbus, unlike EllSeg's ellipse on this footage, whose softargmax
    pupil is centred on the highlight itself.
    """
    p = np.asarray(points)[iris_ids, :2]
    centre = p[0]
    offsets = p[1:] - centre
    ax = float((abs(offsets[0, 0]) + abs(offsets[2, 0])) / 2)
    ay = float((abs(offsets[1, 1]) + abs(offsets[3, 1])) / 2)
    return centre, ax, ay


def lid_distance(points, lid_ids, window, scale=4):
    """Distance in native pixels from inside the eyelid polygon to its boundary.

    Rasterized inside the eye's own window. A full-frame 4x supersample is a 33-megapixel
    distance transform per eye per frame, which alone made a 150-frame run take hours.
    """
    ys, xs = window
    height, width = ys.stop - ys.start, xs.stop - xs.start
    fine = np.zeros((height * scale, width * scale), np.uint8)
    polygon = np.asarray(points)[lid_ids, :2] - [xs.start, ys.start]
    cv2.fillPoly(fine, [np.rint((polygon + .5) * scale - .5).astype(np.int32)], 1)
    distance = cv2.distanceTransform(fine, cv2.DIST_L2, cv2.DIST_MASK_PRECISE) / scale
    return cv2.resize(distance, (width, height), interpolation=cv2.INTER_AREA)


def radial_profile(values, rho, edges):
    """Per-channel median radiance as a function of normalized radius.

    Medians, not means, so a surviving fragment of the reflection cannot drag the
    profile it is about to be measured against.
    """
    centres = (edges[:-1] + edges[1:]) / 2
    table = np.full((len(centres), values.shape[-1]), np.nan, np.float64)
    for k in range(len(centres)):
        band = (rho >= edges[k]) & (rho < edges[k + 1])
        if int(band.sum()) >= 4:
            table[k] = np.median(values[band], axis=0)
    good = np.isfinite(table[:, 0])
    if good.sum() < 4:
        return None, centres
    filled = np.stack([np.interp(centres, centres[good], table[good, c])
                       for c in range(values.shape[-1])], axis=-1)
    return filled, centres


def angular_basis(theta):
    """Smooth shading around the eye: constant plus the first two harmonics."""
    return np.stack([np.ones_like(theta), np.cos(theta), np.sin(theta),
                     np.cos(2 * theta), np.sin(2 * theta)], axis=-1)


def angular_gain(luma, luma_profile, theta, usable, bins=16, ridge=0.05):
    """Multiplicative shading across the eye, fitted robustly on the unmasked pixels.

    A radial profile alone cannot describe this eye: the lash line and lid shadow darken
    the top of the iris, and that is not radially symmetric about the iris centre.

    The fit is on per-angle MEDIANS, not on the pixels directly. An ordinary least
    squares fit was tried first and fails badly: the highlight is a dozen pixels whose
    ratio to the profile is nearly ten, so it drags the whole shading model upward, the
    model then over-predicts the rest of the iris, and the resulting negative residual
    across the upper iris -- not the highlight -- was what tripped the detector.
    A shadow scales every channel, so one luminance gain is fitted and applied to all.
    """
    reference = luma_profile[usable]
    valid = reference > 1e-4
    if valid.sum() < 40:
        return np.ones_like(theta)
    ratio = luma[usable][valid] / reference[valid]
    angle = theta[usable][valid]
    index = ((angle + np.pi) / (2 * np.pi) * bins).astype(int) % bins
    nodes, medians, weights = [], [], []
    for k in range(bins):
        block = index == k
        if int(block.sum()) >= 3:
            nodes.append(-np.pi + (k + 0.5) * 2 * np.pi / bins)
            medians.append(float(np.median(ratio[block])))
            weights.append(float(block.sum()))
    if len(nodes) < 5:
        return np.ones_like(theta)
    a = angular_basis(np.array(nodes))
    w = np.sqrt(np.array(weights))[:, None]
    prior = np.zeros(a.shape[1])
    prior[0] = 1.0
    penalty = ridge * np.eye(a.shape[1]) * float(np.sum(weights))
    aw = a * w
    solution = np.linalg.solve(aw.T @ aw + penalty,
                               aw.T @ (np.array(medians) * w[:, 0]) + penalty @ prior)
    return np.clip(angular_basis(theta) @ solution, 0.35, 2.5)


def evaluate_profile(profile, centres, rho):
    return np.stack([np.interp(rho, centres, profile[:, c]) for c in range(profile.shape[-1])], axis=-1)


EDGES = np.arange(0, RHO_MAX + 1e-9, RHO_STEP)


def fit_dark_core(luma, xx, yy, limbus, ax, ay, usable):
    """Centre the polar frame on the eye's own dark core, not on the limbus landmark.

    The pupil in this plate is small and sits OFF the limbus centre. Profiling about the
    limbus therefore smears it into a smooth ramp -- which is what made an earlier pass
    of this round conclude, wrongly, that the plate carries no pupil at all, and then
    fill the removed highlight with uniform brown while the source plainly shows black
    around it. Measured on source 701 left, the inner profile about the limbus centre is
    .0382 / .0363 / .0410 linear -- flat -- and about the fitted centre
    .0065 / .0185 / .0382.

    The score is an outer annulus median minus an inner disc median, over non-highlight
    pixels. It is a DEPTH, not a radius, so it cannot run to a bound the way an earlier
    radius search did.
    """
    best = None
    grid = np.arange(-CENTRE_SEARCH, CENTRE_SEARCH + 1e-9, CENTRE_STEP)
    for dy in grid:
        for dx in grid:
            distance = np.hypot((xx - (limbus[0] + dx)) / ax, (yy - (limbus[1] + dy)) / ay)
            core = usable & (distance < 0.22)
            ring = usable & (distance > 0.45) & (distance < 0.75)
            if core.sum() < 5 or ring.sum() < 40:
                continue
            depth = float(np.median(luma[ring]) - np.median(luma[core]))
            if best is None or depth > best[0]:
                best = (depth, np.array([dx, dy]), int(core.sum()))
    if best is None:
        return None
    return dict(depth=best[0], offset=best[1], core_pixels=best[2])


def pupil_radius(profile, centres):
    """Normalized radius of the steepest rise of the radial profile.

    Reported so the fill level in the dark core is auditable. The fill does not depend
    on it: the measured profile carries the darkening whether or not this number is a
    correctly located pupil edge.
    """
    luma = profile.max(axis=-1)
    inner = centres < 0.75
    if inner.sum() < 4:
        return None
    gradient = np.gradient(luma, centres)
    k = int(np.argmax(np.where(inner, gradient, -np.inf)))
    if not np.isfinite(gradient[k]) or gradient[k] <= 0:
        return None
    if 0 < k < len(centres) - 1:
        a, b, c = gradient[k - 1], gradient[k], gradient[k + 1]
        denom = a - 2 * b + c
        shift = 0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0
        return float(np.clip(centres[k] + np.clip(shift, -1, 1) * RHO_STEP, 0.08, 0.75))
    return float(centres[k])


def detect_anomaly(values, rho, theta, inside, iterations=4, sigmas=4.0, floor=0.012,
                   grow=1, reach=7.0):
    """Pixels the eye's own anatomy cannot explain, in a compact blob at the highlight.

    The model is a radial profile times a smooth angular shading gain -- the eye's own
    structure, measured from its own unmasked pixels. Both signs of residual are taken:
    the bright core is obvious, and the dark rim hugging it is equally inconsistent
    with the anatomy. On this plate that rim reaches zero in green and blue after the
    tagged limited-range expansion, so leaving it would ring a removed highlight in
    black. `reach` keeps the footprint compact: a corneal image of a softbox is a few
    pixels across, and without it the component leaks along the lash line.
    """
    anomaly = np.zeros(inside.shape, bool)
    profile = centres = base = None
    threshold = low = None
    for _ in range(iterations):
        usable = inside & ~anomaly
        if usable.sum() < 40:
            return None, None, 'too_few_usable_pixels'
        profile, centres = radial_profile(values[usable], rho[usable], EDGES)
        if profile is None:
            return None, None, 'no_radial_profile'
        radial = evaluate_profile(profile, centres, rho)
        gain = angular_gain(values.max(axis=-1), radial.max(axis=-1), theta, usable)
        base = radial * gain[..., None]
        signed = (values - base).max(axis=-1)
        if threshold is None:
            # Frozen at the first pass, over ALL visible pixels. Recomputing it every
            # iteration is a runaway: excluding the largest residuals shrinks the
            # spread, which lowers the threshold, which grows the mask.
            sample = signed[inside]
            spread = float(np.median(np.abs(sample - np.median(sample))) * 1.4826)
            threshold = max(sigmas * spread, floor)
            low = max(2.5 * spread, floor / 2)
        deviation = np.abs(values - base).max(axis=-1)
        peak = np.where(inside, signed, -np.inf)
        if not np.isfinite(peak).any() or peak.max() <= threshold:
            return None, None, 'no_reflection_found'
        seed = np.unravel_index(int(np.argmax(peak)), peak.shape)
        rows, cols = np.mgrid[:inside.shape[0], :inside.shape[1]]
        near = np.hypot(cols - seed[1], rows - seed[0]) <= reach
        # Hysteresis. A single symmetric threshold set by the bright side is far too
        # high for the dark rim: the rim is crushed to zero, so its deviation is capped
        # by the local base level itself -- about .05 linear against a .09-.15 threshold
        # -- and a black smear survived every removal. The high level seeds, the low
        # level grows, and `reach` plus connectivity keep the result compact.
        # The two parts of the corruption have OPPOSITE signs, and using |residual|
        # cannot tell either from iris texture, which is both signs at random.
        # The highlight is a positive core; the coding undershoot beside it is a
        # negative rim touching that core. Growing on the signed residual keeps the
        # footprint on the highlight instead of walking out into real material.
        kernel = np.ones((3, 3), np.uint8)
        strong = inside & near & (signed > threshold)
        if not strong.any():
            return None, None, 'seed_outside_candidates'
        core = strong.copy()
        for _ in range(2):
            grown = (cv2.dilate(core.astype(np.uint8), kernel) > 0) & inside & near & (signed > low)
            if grown.sum() == core.sum():
                break
            core = grown
        # The dark rim beside the highlight is NOT repaired. It is part corruption --
        # the tagged limited-range expansion crushes it to zero in green and blue -- but
        # it is also the only dark material the plate shows around the pupil, and the
        # artist reads it as the pupil. Replacing it with the model's level at that
        # radius made the repaired eye uniformly brown where the source is black, which
        # he rejected on sight. Removing only the white leaves that black observed.
        anomaly = core
    if grow:
        kernel = np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)
        anomaly = (cv2.dilate(anomaly.astype(np.uint8), kernel) > 0) & inside
    return anomaly, dict(base=base, profile=profile, centres=centres,
                         threshold=threshold, low=low), None


def angular_fill(residual, rho, theta, mask, donor, rho_tolerance=0.075, neighbours=3):
    """Complete the residual at each masked pixel from the same eye, same radius.

    An iris is fibrous along radius, so the honest donors for a masked pixel are the
    unmasked pixels of the SAME frame at the SAME normalized radius. Inside the pupil
    the annulus is uniform and this degenerates to the pupil's own level, which is why
    a footprint spanning the pupil boundary cannot come out as one flat material.
    """
    count = int(mask.sum())
    filled = np.zeros((count, residual.shape[-1]), np.float64)
    supported = np.zeros(count, bool)
    donor_rho, donor_theta, donor_res = rho[donor], theta[donor], residual[donor]
    for j, (r0, t0) in enumerate(zip(rho[mask], theta[mask])):
        band = np.abs(donor_rho - r0) < rho_tolerance
        if band.sum() < 2:
            band = np.abs(donor_rho - r0) < rho_tolerance * 2
        if band.sum() < 2:
            continue
        delta = np.angle(np.exp(1j * (donor_theta[band] - t0)))
        available = donor_res[band]
        total = np.zeros(residual.shape[-1])
        weight = 0.0
        for sign in (1, -1):
            side = np.where(np.sign(delta) == sign)[0]
            if not len(side):
                continue
            order = side[np.argsort(np.abs(delta[side]))][:neighbours]
            w = 1.0 / max(float(np.abs(delta[order]).min()), 1e-3)
            total += w * available[order].mean(axis=0)
            weight += w
        if weight <= 0:
            continue
        filled[j] = total / weight
        supported[j] = True
    return filled, supported


def parabola_opening(coef_upper, coef_lower, centre, window, x_range, feather=0.8,
                     inset=0.0):
    """Soft matte of the eye opening from two fitted lid parabolas.

    Evaluated here rather than imported from `eyelid_fit` only to keep this module free
    of a circular import; the arithmetic is three lines and is the same three lines.
    The parabolas run forever sideways, so the opening is closed at the eye corners --
    which is the one part of the mediapipe contour that is right, because a corner is
    sharp, unoccluded and nowhere near a lash.
    """
    ys, xs = window
    yy, xx = np.mgrid[ys.start:ys.stop, xs.start:xs.stop]
    u = xx - centre[0]
    m = np.ones(yy.shape, np.float32)
    for coef, sign in ((coef_upper, 1.0), (coef_lower, -1.0)):
        if coef is None:
            continue
        # `inset` moves each edge INTO the eye, in native pixels.
        edge = coef[0] * u ** 2 + coef[1] * u + coef[2] + sign * inset
        m = np.minimum(m, np.clip(sign * (yy - edge) / max(feather, 1e-6) + 0.5, 0.0, 1.0))
    inside_x = np.clip((xx - (x_range[0] + inset)) / max(feather, 1e-6) + 0.5, 0.0, 1.0)
    inside_x = np.minimum(inside_x, np.clip((x_range[1] - inset - xx) / max(feather, 1e-6)
                                            + 0.5, 0.0, 1.0))
    return (m * inside_x).astype(np.float32)


def eye_frame(values, points, topology, side, margin=6, limbus_override=None,
              eyelid_override=None, lid_inset=0.0):
    """Local crop, geometry and visibility for one eye, or a reason it abstains.

    Everything downstream works inside this crop: the centre search evaluates hundreds
    of candidate origins and a full-frame grid per candidate would cost minutes a frame.

    `limbus_override` is `(centre, ax, ay)` from `limbus_fit.py`, fitted to the image edge
    rather than to the four mediapipe iris landmarks. Round 35 measured the landmark at
    **3.08 px from her own iris centre on the left eye** (median, by an independent
    level-crossing estimator) against 0.42 px for the fit, and the landmark's error moves
    up to 10 px between adjacent frames while the fit's moves 2.7. Every fill since round
    32 is anchored here, so this is the single place that has to change.
    """
    if limbus_override is not None:
        limbus, ax, ay = limbus_override
        limbus = np.asarray(limbus, float)
    else:
        limbus, ax, ay = limbus_ellipse(points, topology['iris_' + side])
    record = dict(side=side, limbus_centre=[float(limbus[0]), float(limbus[1])],
                  limbus_axes=[round(ax, 3), round(ay, 3)],
                  limbus_source='image_fit' if limbus_override is not None else 'landmark')
    if not np.isfinite([ax, ay]).all() or min(ax, ay) < 4:
        record.update(repaired=False, reason='degenerate_limbus')
        return None, record
    height, width = values.shape[:2]
    reach = max(ax, ay) * RHO_MAX + CENTRE_SEARCH + margin
    x0 = int(np.floor(limbus[0] - reach)); x1 = int(np.ceil(limbus[0] + reach)) + 1
    y0 = int(np.floor(limbus[1] - reach)); y1 = int(np.ceil(limbus[1] + reach)) + 1
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        record.update(repaired=False, reason='eye_crop_leaves_frame')
        return None, record
    window = (slice(y0, y1), slice(x0, x1))
    lid = lid_distance(points, topology['eyelid_' + side], window)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    visible = lid > 0.85
    if eyelid_override is not None:
        # The mediapipe eyelid contour is the second bad anchor round 35 found: measured
        # against the image's own sclera-to-lash edge it sits 8.34 px INSIDE the true
        # opening on the left eye and 1.01 px outside it on the right, so it crosses the
        # middle of the iris with the softbox above it on one eye and is right on the
        # other. `eyelid_fit.py` fits both lids to the image; this is where that lands.
        poly = np.asarray(points)[topology['eyelid_' + side], :2]
        opening = parabola_opening(eyelid_override.get('upper'),
                                   eyelid_override.get('lower'),
                                   eyelid_override['centre'], window,
                                   (float(poly[:, 0].min()), float(poly[:, 0].max())),
                                   inset=lid_inset)
        # Pull the opening in by `lid_inset`. The lid fit's own column residual is
        # 0.43 px median / 0.59 p95, so a pixel is about 1.7 sigma of it, and the two
        # ways of being wrong are not symmetric: too far in leaves a sliver of her own
        # eye at the lash, too far out repaints the lash itself and the junction reads
        # as a bright band.
        visible = opening > 0.5
        record['lid_inset_px'] = float(lid_inset)
        record['lid_source'] = 'image_fit'
    else:
        opening = None
        record['lid_source'] = 'landmark'
    record['visible_pixels'] = int((visible & (np.hypot((xx - limbus[0]) / ax,
                                                        (yy - limbus[1]) / ay) < RHO_MAX)).sum())
    if record['visible_pixels'] < 90:
        record.update(repaired=False, reason='eye_closed_or_occluded')
        return None, record
    # `opening` is the SOFT matte `visible` thresholds; kept so a caller can anti-alias
    # the lid edge instead of toggling whole pixels as the lid moves sub-pixel.
    return dict(limbus=limbus, ax=ax, ay=ay, xx=xx, yy=yy, visible=visible,
                opening=opening, window=window), record


def solve_eye(patch, geom, offset=None):
    """Build the eye polar frame on the limbus.

    Centring it on the fitted dark core was tried and is wrong: that core was measured
    to be the highlight's own undershoot rim, and centring on it put the highlight at
    small radius where it dominated its own profile bins and hid itself, so the mask
    came out shifted off the highlight and left a bright speck behind. With the
    highlight and rim excluded, no pupil localizes at all -- across six eyes the fitted
    offset scatters over the whole +/-4 px search range on 6-10 pixels of evidence and
    the profile shows no step. The limbus is the only stable anchor this plate carries.
    """
    limbus, ax, ay, xx, yy = geom['limbus'], geom['ax'], geom['ay'], geom['xx'], geom['yy']
    fit = None
    if offset is None:
        offset = np.zeros(2)
    centre = np.asarray(limbus, float) + np.asarray(offset, float)
    rho = np.hypot((xx - centre[0]) / ax, (yy - centre[1]) / ay)
    theta = np.arctan2((yy - centre[1]) / ay, (xx - centre[0]) / ax)
    inside = geom['visible'] & (rho < RHO_MAX)
    if inside.sum() < 60:
        return None
    return dict(centre=centre, offset=np.asarray(offset, float), fit=fit,
                rho=rho, theta=theta, inside=inside)


def repair_eye(patch, geom, state):
    rho, theta, inside = state['rho'], state['theta'], state['inside']
    anomaly, model, reason = detect_anomaly(patch, rho, theta, inside)
    if anomaly is None:
        return None, dict(repaired=False, reason=reason)
    base = model['base']
    residual = patch - base
    rp = pupil_radius(model['profile'], model['centres'])
    # The guard is ABSOLUTE, because a corneal image of a softbox is a few pixels
    # across whatever the lid is doing. A fraction-of-visible-area cap was tried first
    # and abstained on 27 eyes whose footprints were 27-101 pixels -- the same size as
    # the accepted ones -- purely because the eye was half shut and the denominator
    # small. That produced removal on one frame and not the next, which is flicker.
    # The loose ratio bound stays only to catch a model that has failed outright.
    limit = 3.4 * geom['ax'] ** 2 / 4 * np.pi / 4
    if anomaly.sum() > limit or anomaly.sum() > 0.55 * inside.sum():
        return None, dict(repaired=False, reason='footprint_implausibly_large',
                          anomaly_pixels=int(anomaly.sum()), visible_pixels=int(inside.sum()),
                          footprint_limit=round(float(limit), 1))
    # Donors keep a two-pixel collar off the footprint: the pixels immediately outside
    # it carry the sub-threshold tail of the same highlight, and feeding those back in
    # is what made the first repair look blotchy.
    collar = cv2.dilate(anomaly.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    donor = inside & ~collar
    if donor.sum() < 60:
        return None, dict(repaired=False, reason='insufficient_donor_annulus')

    filled, supported = angular_fill(residual, rho, theta, anomaly, donor)
    dropped = int((~supported).sum())
    if dropped:
        holes = np.zeros(anomaly.shape, bool)
        holes[anomaly] = ~supported
        anomaly = anomaly & ~holes
        filled = filled[supported]
        if anomaly.sum() < 3:
            return None, dict(repaired=False, reason='no_supported_pixels')

    completed = patch.copy()
    completed[anomaly] = np.maximum(base[anomaly] + filled, 0.0)
    weight = anomaly.astype(np.float32)
    out = patch * (1 - weight[..., None]) + completed * weight[..., None]

    edited = weight > 0
    # Reported as inner / outer of the profile's steepest step, NOT as pupil / iris:
    # see solve_eye. The split exists so the fill level in the dark centre is auditable.
    pupil = edited & (rho < rp) if rp is not None else np.zeros_like(edited)
    iris = edited & ~pupil
    record = dict(
        repaired=True, reason=None,
        radial_frame='dark-core centre, offset from the limbus landmark',
        core_offset=[round(float(v), 3) for v in state['offset']],
        profile_step_radius_norm=None if rp is None else round(rp, 4),
        profile_step_radius_px=None if rp is None else round(rp * geom['ax'], 3),
        anomaly_pixels=int(anomaly.sum()), dropped_unsupported_pixels=dropped,
        footprint_inner_pixels=int(pupil.sum()), footprint_outer_pixels=int(iris.sum()),
        peak_source_display=round(float(linear_to_srgb(patch[edited].max(axis=-1).max())), 4),
        peak_repaired_display=round(float(linear_to_srgb(out[edited].max(axis=-1).max())), 4),
        source_zero_channels_in_footprint=int((patch[edited] <= 0).sum()),
        fill_inner_linear=None if not pupil.any() else [round(float(v), 6) for v in out[pupil].mean(axis=0)],
        fill_outer_linear=None if not iris.any() else [round(float(v), 6) for v in out[iris].mean(axis=0)],
        donor_pixels=int(donor.sum()),
        anomaly_threshold_linear=round(float(model['threshold']), 6),
        anomaly_grow_threshold_linear=round(float(model['low']), 6))
    return (out, weight), record


def build_review_tile(source, repaired, weight, lo, hi, zoom, source_frame, eyes):
    band = (slice(lo[1], hi[1]), slice(lo[0], hi[0]))
    marked = rgb8(source)[band].copy()
    selected = weight[band] > 0
    marked[selected] = np.rint(marked[selected] * .3 + np.array([255, 40, 170]) * .7).astype(np.uint8)
    detail = ' '.join('%s:%s' % (e['side'][0].upper(),
                                 e.get('anomaly_pixels') if e.get('repaired') else e.get('reason'))
                      for e in eyes) if eyes else 'no track'
    panels = []
    for title, image in (('SOURCE', rgb8(source)[band]),
                         ('REPAIR - eye anatomy, no generated material', rgb8(repaired)[band]),
                         ('EDIT MASK', marked)):
        view = cv2.resize(image[..., ::-1], None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
        header = np.zeros((44, view.shape[1], 3), np.uint8)
        cv2.putText(header, title, (10, 20), 0, .55, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(header, 'source %d   %s' % (source_frame, detail), (10, 37), 0, .42,
                    (200, 200, 200), 1, cv2.LINE_AA)
        panels.append(np.concatenate([header, view]))
    return np.concatenate(panels, axis=1)


def contact_sheet(tiles, columns=5):
    """Every frame on one page, at eye scale.

    Rounds are judged on four cherry-picked frames far too easily. The whole interval
    goes on a sheet so a leak on one frame in fifty cannot pass unseen.
    """
    height = max(t.shape[0] for t in tiles)
    width = max(t.shape[1] for t in tiles)
    padded = []
    for tile in tiles:
        canvas = np.zeros((height, width, 3), np.uint8)
        canvas[:tile.shape[0], :tile.shape[1]] = tile
        padded.append(canvas)
    while len(padded) % columns:
        padded.append(np.zeros((height, width, 3), np.uint8))
    rows = [np.concatenate(padded[k:k + columns], axis=1) for k in range(0, len(padded), columns)]
    return np.concatenate(rows, axis=0)


def eye_tile(source, repaired, weight, points, topology, zoom, source_frame, eyes, half=17):
    """Four panels for one frame: source and repair of each eye, at native pixels."""
    panels = []
    for side in SIDES:
        centre = np.asarray(points)[topology['iris_' + side][0], :2]
        x0, y0 = int(round(centre[0])) - half, int(round(centre[1])) - half
        x0 = max(0, min(x0, source.shape[1] - 2 * half))
        y0 = max(0, min(y0, source.shape[0] - 2 * half))
        window = (slice(y0, y0 + 2 * half), slice(x0, x0 + 2 * half))
        touched = weight[window] > 0
        for label, image in (('src', rgb8(source)[window]), ('rep', rgb8(repaired)[window])):
            view = image[..., ::-1].copy()
            if label == 'rep' and touched.any():
                edge = (cv2.dilate(touched.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & ~touched
                view[edge] = np.rint(view[edge] * .55 + np.array([255, 40, 170]) * .45).astype(np.uint8)
            panels.append(cv2.resize(view, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST))
    strip = np.concatenate(panels, axis=1)
    header = np.zeros((24, strip.shape[1], 3), np.uint8)
    note = ' | '.join('%s %s' % (e['side'][:1].upper(),
                                 ('%dpx' % e['anomaly_pixels']) if e.get('repaired') else e.get('reason', '?'))
                      for e in eyes) if eyes else 'no track'
    cv2.putText(header, 'src %d   L src/rep   R src/rep   %s' % (source_frame, note),
                (6, 16), 0, .38, (240, 240, 240), 1, cv2.LINE_AA)
    return np.concatenate([header, strip])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_shot_argument(ap)
    ap.add_argument('--name', default='r32_repair')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--count', type=int, default=24)
    ap.add_argument('--zoom', type=int, default=8)
    ap.add_argument('--sheet-zoom', type=int, default=4)
    ap.add_argument('--sheet-frames', type=int, default=25)
    ap.add_argument('--offset-window', type=int, default=7,
                    help='half-width, in frames, of the rolling median on the core offset')
    args = ap.parse_args()
    shot = from_args(args)
    if Path(args.name).name != args.name:
        raise ValueError('Expected a directory name, not a path')
    root = shot.composite / args.name
    for sub in ('source', 'repair', 'review', 'linear', 'mask', 'sheets'):
        (root / sub).mkdir(parents=True, exist_ok=True)

    meta = json.loads((shot.relight / 'face_landmarks.json').read_text())
    topology = meta['topology']
    with np.load(shot.relight / 'face_landmarks.npz') as z:
        landmarks = {k: z[k] for k in ('points', 'found', 'eyelid_aperture')}
    indices = list(range(args.start, args.start + args.count))
    if min(indices) < 0 or max(indices) >= shot.frame_count:
        raise ValueError('Requested interval leaves the shot')

    eye_ids = topology['iris_left'] + topology['iris_right']
    tracked = np.array([landmarks['points'][i][eye_ids, :2] for i in indices if landmarks['found'][i]])
    if not len(tracked):
        raise ValueError('No face track in the requested interval')
    lo = np.floor(tracked.reshape(-1, 2).min(0) - [30, 26]).astype(int)
    hi = np.ceil(tracked.reshape(-1, 2).max(0) + [30, 26]).astype(int)

    # --- pass 1: per-frame radial centre, pooled to one constant offset per eye ------
    sources = {}
    per_frame = {s: [] for s in SIDES}
    for i in indices:
        path = shot.fg(i)
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None or raw.dtype != np.float32 or raw.ndim != 3:
            raise ValueError('Expected scene-linear keyed RGB EXR at %s' % path)
        sources[i] = raw[..., ::-1].astype(np.float32)
        if not (landmarks['found'][i] and np.isfinite(landmarks['points'][i]).all()):
            continue
        for side in SIDES:
            geom, record = eye_frame(sources[i], landmarks['points'][i], topology, side)
            if geom is None:
                continue
            state = solve_eye(sources[i][geom['window']], geom)
            if state is None or state['fit'] is None:
                continue
            per_frame[side].append((i, state['offset'], state['fit']['depth']))
        print('R32_DIAGNOSTIC frame', i, flush=True)

    pooled = {}
    for side in SIDES:
        rows = per_frame[side]
        if len(rows) < 3:
            pooled[side] = None
            continue
        # One constant offset for the whole shot: the pupil does not wander relative to
        # the limbus, so a per-frame offset would only add temporal noise.
        frames = np.array([r[0] for r in rows])
        offsets = np.array([r[1] for r in rows], float)
        depths = np.array([r[2] for r in rows], float)
        # A rolling median along the sequence. The pupil does not jump relative to the
        # limbus between adjacent frames, so a per-frame offset is largely measurement
        # noise; a single constant offset would instead ignore real slow gaze change.
        smoothed = {}
        for frame in frames:
            window = np.abs(frames - frame) <= args.offset_window
            smoothed[int(frame)] = np.median(offsets[window], axis=0)
        centre = np.median(offsets, axis=0)
        pooled[side] = dict(
            per_frame=smoothed,
            median_offset=[round(float(v), 4) for v in centre],
            offset_mad=[round(float(v), 4) for v in np.median(np.abs(offsets - centre), axis=0)],
            core_depth_median=round(float(np.median(depths)), 5),
            core_depth_range=[round(float(depths.min()), 5), round(float(depths.max()), 5)],
            frames=len(rows), smoothing_window_frames=args.offset_window)
        print('R32_CORE', side, {k: v for k, v in pooled[side].items() if k != 'per_frame'},
              flush=True)

    # --- pass 2: repair with the pooled geometry ------------------------------------
    rows = []
    sheet_tiles = []
    for i in indices:
        values = sources[i]
        out = values.copy()
        combined = np.zeros(values.shape[:2], np.float32)
        eyes = []
        if landmarks['found'][i] and np.isfinite(landmarks['points'][i]).all():
            for side in SIDES:
                geom, record = eye_frame(out, landmarks['points'][i], topology, side)
                if geom is None:
                    eyes.append(record)
                    continue
                patch = out[geom['window']]
                state = solve_eye(patch, geom, offset=np.zeros(2))
                if state is None:
                    record.update(repaired=False, reason='no_eye_frame')
                    eyes.append(record)
                    continue
                result, detail = repair_eye(patch, geom, state)
                record.update(detail)
                if result is not None:
                    repaired_patch, weight = result
                    out[geom['window']] = repaired_patch
                    combined[geom['window']] = np.maximum(combined[geom['window']], weight)
                eyes.append(record)
        else:
            eyes = [dict(side=s, repaired=False, reason='no_face_track') for s in SIDES]

        untouched = combined == 0
        assert np.isfinite(out).all() and (out >= 0).all()
        assert np.array_equal(out[untouched], values[untouched]), 'source changed outside the mask'

        cv2.imwrite(str(root / 'linear' / ('%06d.exr' % i)), out[..., ::-1].astype(np.float32),
                    [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
        cv2.imwrite(str(root / 'source' / ('%06d.png' % i)), rgb8(values)[..., ::-1])
        cv2.imwrite(str(root / 'repair' / ('%06d.png' % i)), rgb8(out)[..., ::-1])
        cv2.imwrite(str(root / 'mask' / ('%06d.png' % i)),
                    np.rint(np.clip(combined, 0, 1) * 255).astype(np.uint8))
        cv2.imwrite(str(root / 'review' / ('%06d.png' % i)),
                    build_review_tile(values, out, combined, lo, hi, args.zoom,
                                      shot.source_start_frame + i, eyes))
        sheet_tiles.append(eye_tile(values, out, combined, landmarks['points'][i], topology,
                                    args.sheet_zoom, shot.source_start_frame + i, eyes)
                           if landmarks['found'][i] and np.isfinite(landmarks['points'][i]).all()
                           else np.zeros((24 + 34 * args.sheet_zoom, 4 * 34 * args.sheet_zoom, 3), np.uint8))
        rows.append(dict(frame=i, source_frame=shot.source_start_frame + i,
                         source_sha256=sha(shot.fg(i)),
                         changed_pixels=int((combined > 0).sum()),
                         aperture=[float(v) if np.isfinite(v) else None
                                   for v in landmarks['eyelid_aperture'][i]],
                         eyes=eyes))
        print('R32_REPAIR frame', i, 'changed', int((combined > 0).sum()), flush=True)

    sheets = []
    for k in range(0, len(sheet_tiles), args.sheet_frames):
        path = root / 'sheets' / ('contact_%03d.jpg' % indices[k])
        assert cv2.imwrite(str(path), contact_sheet(sheet_tiles[k:k + args.sheet_frames]),
                           [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        sheets.append(path.name)
        print('R32_SHEET', path.name, flush=True)

    repaired = [r for r in rows if r['changed_pixels']]
    record = dict(
        rows=rows, crop_xyxy=[*lo.tolist(), *hi.tolist()],
        dark_core_geometry={s: (None if pooled[s] is None else
                                {k: v for k, v in pooled[s].items() if k != 'per_frame'})
                            for s in SIDES},
        polar_frame='limbus landmark',
        dark_rim_repaired=False,
        dark_rim_note=('The undershoot beside the highlight is left as observed source. '
                       'It is the only dark material the plate carries around the pupil.'),
        contact_sheets=sheets,
        frames=len(rows), frames_repaired=len(repaired),
        frames_exact_source=len(rows) - len(repaired),
        method='pooled radial centre, per-frame radial profile, same-radius angular residual completion',
        material_source='the same eye in the same frame; no generated material',
        generated_material_used=False, scene_reflection_added=False,
        world_registration_validated=False,
        production_enabled=False, artist_accepted=False,
        source_outside_mask_exact=True,
        code_sha256=sha(__file__), track_sha256=sha(shot.relight / 'face_landmarks.npz'),
        shot_sha256=sha(shot.dir / 'shot.json'))
    write_json(root / 'repair.json', record)
    write_json(shot.log(args.name + '.json'), record)
    print('R32_REPAIR_DONE', len(rows), 'frames,', len(repaired), 'repaired', flush=True)


if __name__ == '__main__':
    main()
