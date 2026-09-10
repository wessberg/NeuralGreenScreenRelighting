"""Estimate the eye ONCE over the sequence in a registered frame, then relight it.

Round 32 removed the highlight and was rejected: not temporally stable in shape or in
colour, and the pupil sometimes read brown. Its structural cause is that every term of
its fill is refit per frame from a few dozen pixels. This module answers that with the
opposite construction:

* **One appearance atlas per eye for the whole shot.** Built from observed clean pixels
  only, pooled across all frames in a sub-pixel registered eye frame. Nothing about the
  atlas is refit per frame, so the filled colour cannot walk frame to frame.
* **One smooth per-frame illumination term.** Three scalars per frame per eye, robustly
  fitted then explicitly temporally smoothed, with the adjacent-frame step reported.
* **A mask that comes from the atlas, not from a per-frame model.** The residual is
  measured against the same fixed reference in every frame, so the footprint stops
  breathing.

Why this is possible at all, which round 33 measured first
(`eye_sequence_occupancy.py`, `logs/r33_occupancy.json`): the highlight SWEEPS across
the eye's own frame over the shot -- rho .383->.749 left with 50.6 px of arc travel,
.284->.741 right with 22.0 px -- so 506 of 507 ever-corrupt cells left, and 550 of 621
right, are observed CLEAN in other frames. There is real observed material behind the
highlight. That is not what the round 8-17 donor searches asked: they looked for
correspondence between individual pixels, and this pools in the eye's own coordinates.

Why the registration is load bearing, and not a refinement: the limbus landmark jitters
between frames, and the pupil is about 8 native pixels across. Pooling on the landmark
alone smears the pupil edge into a smooth ramp -- the pooled radial profile runs 55, 55,
56, 59 ... 98, 108 display with no step, while single frames show 35-70 inside an iris
at 90-130. An unregistered atlas would BE the uniformly brown pupil the artist rejected.
Registration is translation and isotropic scale only. Torsion is not fitted: rounds 12
and 14 both measured that fitting it makes withheld-pixel error worse.

The dark undershoot beside the highlight is coding, not material -- green and blue
exactly 0 under the tagged limited-range expansion -- but the artist reads it as the
pupil, so it stays excluded from the atlas AND from the edit mask by default.

Writes no production default.

    dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/eye_sequence_atlas.py \
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
from limbus_fit import Track as LimbusTrack

# The atlas grid, in the eye's normalized coordinates: 1.0 is the limbus. It reaches
# a little past RHO_MAX so the registration has edge context, but only rho < RHO_MAX is
# ever used as material -- the sclera is not radially symmetric about the iris centre.
ATLAS_REACH = 1.0
ATLAS_N = 97                      # about 0.27 native pixels per atlas cell on this plate
ATLAS_STEP = 2 * ATLAS_REACH / (ATLAS_N - 1)


def configure(reach, cells=None):
    """Resize the canonical eye's own coordinate grid.

    The grid stopped at the limbus (reach 1.0) and the atlas was only ever SAMPLED out to
    RHO_MAX = 0.88, a constant round 32 chose for its radial profile model. Round 35
    measured the consequence: of the specular pixels still standing after removal, **92.5
    percent sit at rho > 0.95 with zero removal weight**, median rho 1.022 -- on the
    limbal ring, where the canonical eye simply has no material to put in their place.
    Reaching past the limbus is what lets that ring be repaired at all.
    """
    global ATLAS_REACH, ATLAS_N, ATLAS_STEP
    ATLAS_REACH = float(reach)
    if cells:
        ATLAS_N = int(cells)
    ATLAS_STEP = 2 * ATLAS_REACH / (ATLAS_N - 1)
    return dict(reach=ATLAS_REACH, cells=ATLAS_N, step=ATLAS_STEP)
# Registration search, in normalized units, and the HARD bound on total shift.
# The limbus landmark's own jitter was measured on this track before the search was
# sized: 0.353 px rms and 0.578 px p95 left, 0.256 / 0.521 right, against a local
# quadratic trajectory fit. So the correction wanted is well under one native pixel,
# and a wide search does not find it -- it drifts. An earlier unbounded version walked
# to 1.9 px rms and 3.0 px max while mean NCC FELL from .921 to .617 across iterations,
# which is atlas drift: the reference is rebuilt from the frames it just moved.
# SHIFT_BOUND is therefore about 1 native pixel, absolute from the landmark, and the
# mean shift is re-zeroed after every rebuild so the atlas cannot walk.
COARSE_SHIFT, COARSE_STEP = 0.078, 0.013
FINE_SHIFT, FINE_STEP = 0.0195, 0.0039
SHIFT_BOUND = 0.078
# Measured axis jitter is 0.361 / 0.443 px on a mean semi-axis of 12.888 / 12.540, so
# about 3 percent. The ladder matches that; a wider one lets scale absorb shift.
SCALES = (0.97, 0.985, 1.0, 1.015, 1.03)


def atlas_grid():
    t = np.linspace(-ATLAS_REACH, ATLAS_REACH, ATLAS_N).astype(np.float32)
    return np.meshgrid(t, t)


def sample_eye(patch, geom, valid, u, v):
    """Resample a native eye patch onto normalized eye coordinates (u, v).

    Returns the resampled radiance and the fraction of the source footprint that was
    valid, so a cell straddling the lid edge or the footprint can be dropped rather
    than silently averaging corrupt pixels in.
    """
    y0, x0 = geom['window'][0].start, geom['window'][1].start
    mx = (geom['limbus'][0] + u * geom['ax'] - x0).astype(np.float32)
    my = (geom['limbus'][1] + v * geom['ay'] - y0).astype(np.float32)
    rgb = cv2.remap(patch, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    w = cv2.remap(valid.astype(np.float32), mx, my, cv2.INTER_LINEAR,
                  borderMode=cv2.BORDER_CONSTANT)
    return rgb, w


def register(frame_luma, frame_w, ref_luma, ref_w, core, shift, step, scales,
             centre=(0.0, 0.0)):
    """Bounded translation/scale search maximizing normalized cross-correlation.

    NCC on log radiance, over cells valid in both, so a per-frame exposure difference
    cannot masquerade as a shift. Exhaustive rather than gradient based: the iris is a
    few pixels of texture and a local descent on this objective finds the highlight.

    Scored on the IRIS ONLY (`core`, rho < RHO_MAX). Including the sclera and lid lets
    the bright, high-contrast lid edge dominate the correlation, and the lid moves for
    reasons that have nothing to do with where the iris is.

    Returns None when no candidate has enough overlap, so the caller can drop the frame
    rather than silently accept a zero shift -- an earlier version reported ncc -2.0
    sentinels as if they were registrations.
    """
    offs = np.arange(-shift, shift + 1e-9, step, dtype=np.float32)
    best = None
    ref = np.log(np.maximum(ref_luma, 1e-4))
    u, v = atlas_grid()
    for s in scales:
        for du in offs:
            for dv in offs:
                tu, tv = centre[0] + float(du), centre[1] + float(dv)
                if max(abs(tu), abs(tv)) > SHIFT_BOUND + 1e-9:
                    continue
                mx = ((u * s + tu) / ATLAS_STEP + (ATLAS_N - 1) / 2).astype(np.float32)
                my = ((v * s + tv) / ATLAS_STEP + (ATLAS_N - 1) / 2).astype(np.float32)
                a = cv2.remap(frame_luma, mx, my, cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT)
                aw = cv2.remap(frame_w, mx, my, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT)
                m = (aw > 0.98) & (ref_w > 0.98) & core
                if m.sum() < 400:
                    continue
                x = np.log(np.maximum(a[m], 1e-4))
                y = ref[m]
                x = x - x.mean()
                y = y - y.mean()
                d = np.sqrt((x * x).sum() * (y * y).sum())
                if d <= 0:
                    continue
                c = float((x * y).sum() / d)
                if best is None or c > best[0]:
                    best = (c, tu, tv, float(s))
    return best


def gain_basis(u, v):
    """The per-frame illumination basis: constant plus a linear ramp in eye coordinates.

    Three terms per channel and no more. A single scalar per channel was tried first and
    is not enough: the upper lid casts a shadow across the top of the iris, that shadow
    MOVES as the lid moves, and a scalar cannot represent it -- so the pooled atlas
    quietly averaged different shadow states together and then mispredicted every frame
    whose lid sat differently. A linear ramp is the smallest term that tracks it.

    It stays deliberately small. Round 32's failure was a model with enough freedom to
    refit the eye every frame; anything higher order here would start doing that again.
    """
    return np.stack([np.ones_like(u), u, v], axis=-1).astype(np.float64)


def robust_gain(frame, ref, mask, basis, floor=1e-4, iterations=4):
    """Per-channel log-domain illumination field, fitted by iteratively reweighted LS.

    Robust, not ordinary least squares. Round 32's angular gain was an OLS fit on a
    ratio and a dozen highlight pixels at ratio near ten dragged the whole model upward;
    the same trap applies at sequence scale, and the residual here still contains
    whatever the footprint detector missed.
    """
    if mask.sum() < 60:
        return None
    A = basis[mask]
    coef = np.zeros((3, A.shape[1]))
    for c in range(3):
        a, b = frame[..., c][mask], ref[..., c][mask]
        ok = (b > floor) & (a > floor)
        if ok.sum() < 60:
            return None
        y = np.log(a[ok]) - np.log(b[ok])
        X = A[ok]
        w = np.ones_like(y)
        for _ in range(iterations):
            sw = np.sqrt(w)[:, None]
            sol, *_ = np.linalg.lstsq(X * sw, y * np.sqrt(w), rcond=None)
            r = y - X @ sol
            s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
            w = 1.0 / (1.0 + (r / (2.0 * s)) ** 2)
        coef[c] = sol
    return coef


def apply_gain(coef, basis):
    """Evaluate the fitted illumination field, clamped to a sane multiplicative range."""
    return np.exp(np.clip(np.einsum('...k,ck->...c', basis, coef), -2.5, 2.5))


def smooth_series(values, window):
    """Rolling median then a rolling mean of the same width, over frame index.

    The median kills a single-frame outlier, the mean removes the median's own steps.
    Both are explicit and their effect on the adjacent-frame step is reported.
    """
    n = len(values)
    out = np.array(values, float)
    for _ in range(2):
        acc = np.empty_like(out)
        for i in range(n):
            lo, hi = max(0, i - window), min(n, i + window + 1)
            acc[i] = np.median(out[lo:hi], axis=0)
        out = acc
    acc = np.empty_like(out)
    for i in range(n):
        lo, hi = max(0, i - window), min(n, i + window + 1)
        acc[i] = out[lo:hi].mean(axis=0)
    return acc


def sharpness(atlas, mask):
    """Mean gradient magnitude of atlas log-luma inside the mask, per normalized unit.

    The point of registering is a SHARPER pooled eye -- an unregistered pool smears the
    pupil edge, which is what produced a uniformly brown pupil. A gradient norm is the
    direct measure of that, and it is reported for the unregistered seed as the control.
    """
    luma = np.log(np.maximum(atlas.max(-1), 1e-4)).astype(np.float32)
    gy, gx = np.gradient(luma, ATLAS_STEP)
    inner = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    if not inner.any():
        return None
    return float(np.hypot(gx, gy)[inner].mean())


MIN_CLEAN_OBSERVATIONS = 3
# How far out the atlas may be USED as material. Round 32's model had to stop at
# RHO_MAX because it was a radial profile times an angular gain, and the sclera is not
# radially symmetric about the iris centre -- including it made 106-197 of about 460
# pixels look anomalous. That reasoning does not carry to an atlas: this is a full 2D
# appearance map in eye coordinates and assumes no radial symmetry at all, so it can
# hold the limbal ring and the start of the sclera honestly. Stopping it at RHO_MAX
# anyway left the outer edge of the highlight unpredictable, and the repair left a light
# speck there. Registration and the illumination fit still use RHO_MAX: those DO want
# stable iris material and no moving lid.
FILL_REACH = 0.98


def fill_atlas_holes(atlas, support, core, iterations=24):
    """Interpolate the atlas across cells with too few clean observations.

    Done ONCE for the whole shot, on the pooled atlas, so it cannot introduce anything
    that varies frame to frame -- which is the property this whole construction exists
    to protect. The holes are where the highlight dwells longest, and they are small:
    the alternative is abstaining exactly where the repair is needed.
    """
    known = (support & core).astype(np.uint8)
    out = atlas.copy()
    out[~(support & core)] = 0
    for _ in range(iterations):
        holes = core & ~(known > 0)
        if not holes.any():
            break
        blur = cv2.blur(out, (5, 5))
        wsum = cv2.blur(known.astype(np.float32), (5, 5))
        grow = (cv2.dilate(known, np.ones((3, 3), np.uint8)) > 0) & holes & (wsum > 1e-3)
        out[grow] = (blur[grow] / wsum[grow][..., None])
        known[grow] = 1
    return out, (known > 0) & core


def build(frames, iterations=5, gain_window=5, do_register=True):
    """Alternate: register to the atlas, refit per-frame gain, rebuild the atlas.

    The atlas is a per-cell MEDIAN over frames of gain-normalized radiance, so one frame
    whose footprint detection missed cannot pull a cell.

    Two guards against the atlas drift an earlier version showed (mean NCC falling .921
    to .617 while total shift grew to 3.0 px): every shift is bounded absolutely from the
    landmark inside `register`, and the mean shift is re-zeroed after each rebuild, so
    the whole atlas cannot translate away from the landmark it is anchored to.
    """
    u, v = atlas_grid()
    rho = np.hypot(u, v)
    core = rho < r32.RHO_MAX
    basis = gain_basis(u, v)
    history = []
    # Seed: the unregistered landmark pool. Deliberately the blurred starting point, so
    # any reported sharpening is the registration's own and not the pooling's.
    wstack = np.stack([(f['w'] > 0.98).astype(np.float32) for f in frames])
    atlas = np.divide((np.stack([f['rgb'] for f in frames]) * wstack[..., None]).sum(0),
                      np.maximum(wstack.sum(0), 1e-6)[..., None])
    aw = wstack.sum(0) >= MIN_CLEAN_OBSERVATIONS
    seed_profile = radial_profile(atlas, rho, aw & core)
    seed_sharpness = sharpness(atlas, aw & core)
    for f in frames:
        f['registered'] = not do_register

    for it in range(iterations):
        ref_luma = atlas.max(-1).astype(np.float32)
        ref_w = aw.astype(np.float32)
        first = it == 0
        shift, step = (COARSE_SHIFT, COARSE_STEP) if first else (FINE_SHIFT, FINE_STEP)
        unregistered = 0
        if do_register:
            for f in frames:
                # Scale is searched over the full ladder in the first two passes and
                # then held: once the atlas is sharp, a free scale trades against shift.
                scales = SCALES if it <= 1 else (f['scale'],)
                best = register(f['luma'], f['w'], ref_luma, ref_w, core, shift, step,
                                scales, centre=(0.0, 0.0) if first else (f['du'], f['dv']))
                if best is None:
                    # Not enough overlap with the atlas at any candidate -- a nearly
                    # closed eye, typically. Keep the landmark pose and record it.
                    f['registered'] = False
                    unregistered += 1
                    continue
                f.update(ncc=best[0], du=best[1], dv=best[2], scale=best[3],
                         registered=True)
            # Re-zero the mean shift so the pool stays anchored on the landmark.
            reg = [f for f in frames if f['registered']]
            if reg:
                mu = np.median([[f['du'], f['dv']] for f in reg], axis=0)
                for f in frames:
                    f['du'] = float(np.clip(f['du'] - mu[0], -SHIFT_BOUND, SHIFT_BOUND))
                    f['dv'] = float(np.clip(f['dv'] - mu[1], -SHIFT_BOUND, SHIFT_BOUND))

        for f in frames:
            mx = ((u * f['scale'] + f['du']) / ATLAS_STEP + (ATLAS_N - 1) / 2).astype(np.float32)
            my = ((v * f['scale'] + f['dv']) / ATLAS_STEP + (ATLAS_N - 1) / 2).astype(np.float32)
            f['rgb_w'] = cv2.remap(f['rgb'], mx, my, cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT)
            f['w_w'] = cv2.remap(f['w'], mx, my, cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT)
            f['valid'] = f['w_w'] > 0.98
        raw = []
        for f in frames:
            g = robust_gain(f['rgb_w'], atlas, f['valid'] & aw & core, basis)
            raw.append(np.zeros((3, basis.shape[-1])) if g is None else g)
            f['gain_raw'] = raw[-1]
        coefs = smooth_series(raw, gain_window)
        stack, weights = [], []
        for f, k in zip(frames, coefs):
            f['gain_coef'] = k
            f['gain'] = apply_gain(k, basis)
            stack.append(f['rgb_w'] / np.maximum(f['gain'], 1e-3))
            weights.append(f['valid'].astype(np.float32))
        W = np.stack(weights)
        masked = np.where(W[..., None] > 0, np.stack(stack), np.nan)
        with np.errstate(invalid='ignore'):
            atlas = np.nan_to_num(np.nanmedian(masked, axis=0))
        aw = W.sum(0) >= MIN_CLEAN_OBSERVATIONS
        nccs = [f['ncc'] for f in frames if f['registered']] or [float('nan')]
        history.append(dict(
            iteration=it, unregistered_frames=unregistered,
            mean_ncc=round(float(np.mean(nccs)), 5), min_ncc=round(float(np.min(nccs)), 5),
            atlas_cells=int((aw & core).sum()),
            sharpness=round(sharpness(atlas, aw & core), 5),
            shift_rms_norm=round(float(np.sqrt(np.mean([f['du'] ** 2 + f['dv'] ** 2
                                                       for f in frames]))), 5)))
        print('R33_ATLAS iter', history[-1], flush=True)
    return atlas, aw, history, seed_profile, rho, seed_sharpness


def radial_profile(atlas, rho, mask, bins=18):
    edges = np.linspace(0, r32.RHO_MAX, bins + 1)
    out = []
    for k in range(bins):
        m = mask & (rho >= edges[k]) & (rho < edges[k + 1])
        out.append(None if m.sum() < 8 else
                   [round(float(v), 6) for v in np.median(atlas[m], axis=0)])
    return out


def display_profile(profile):
    return [None if p is None else
            [int(round(float(linear_to_srgb(np.clip(c, 0, 1)) * 255))) for c in p]
            for p in profile]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_shot_argument(ap)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--count', type=int, default=150)
    ap.add_argument('--name', default='r33_atlas')
    ap.add_argument('--limbus-track', default='limbus_track.json',
                    help="relight/<file> from limbus_fit.py: the limbus fitted to the "
                         "IMAGE EDGE. This is the anchor everything here sits on. The "
                         "mediapipe four-point ellipse it replaces sits 3.08 px from her "
                         "own iris centre on the left eye (median, independent estimator) "
                         "and its error jumps up to 10 px between adjacent frames -- the "
                         "artist's 'double iris' and his 'delayed, slow to react'. Pass "
                         "'' to go back to the landmark for comparison.")
    ap.add_argument('--iterations', type=int, default=5)
    ap.add_argument('--gain-window', type=int, default=5)
    ap.add_argument('--no-register', action='store_true',
                    help='ablation: pool on the raw landmark pose, no sub-pixel search')
    args = ap.parse_args()
    shot = from_args(args)
    track = (LimbusTrack.load(shot.relight / args.limbus_track)
             if getattr(args, 'limbus_track', '') else None)
    root = shot.composite / args.name
    root.mkdir(parents=True, exist_ok=True)

    meta = json.loads((shot.relight / 'face_landmarks.json').read_text())
    topology = meta['topology']
    with np.load(shot.relight / 'face_landmarks.npz') as z:
        landmarks = {k: z[k] for k in ('points', 'found', 'eyelid_aperture')}
    indices = list(range(args.start, args.start + args.count))
    if min(indices) < 0 or max(indices) >= shot.frame_count:
        raise ValueError('Requested interval leaves the shot')

    u, v = atlas_grid()
    collected = {s: [] for s in r32.SIDES}
    skipped = {s: [] for s in r32.SIDES}
    for i in indices:
        raw = cv2.imread(str(shot.fg(i)), cv2.IMREAD_UNCHANGED)
        if raw is None or raw.dtype != np.float32 or raw.ndim != 3:
            raise ValueError('Expected scene-linear keyed RGB EXR at %s' % shot.fg(i))
        values = raw[..., ::-1].astype(np.float32)
        if not (landmarks['found'][i] and np.isfinite(landmarks['points'][i]).all()):
            for s in r32.SIDES:
                skipped[s].append(dict(frame=i, reason='no_face_track'))
            continue
        for side in r32.SIDES:
            override = track.get(i, side) if track is not None else None
            if track is not None and override is None:
                skipped[side].append(dict(frame=i, reason='no_limbus_fit'))
                continue
            geom, record = r32.eye_frame(values, landmarks['points'][i], topology, side,
                                         limbus_override=override)
            if geom is None:
                skipped[side].append(dict(frame=i, reason=record.get('reason')))
                continue
            patch = values[geom['window']]
            state = r32.solve_eye(patch, geom)
            if state is None:
                skipped[side].append(dict(frame=i, reason='no_eye_frame'))
                continue
            anomaly, model, reason = r32.detect_anomaly(patch, state['rho'], state['theta'],
                                                        state['inside'])
            if anomaly is None:
                skipped[side].append(dict(frame=i, reason=reason))
                continue
            # Exclude the footprint AND its collar: the collar carries the highlight's
            # sub-threshold tail, and the undershoot rim beside it is clipped coding,
            # not material. Neither belongs in an appearance atlas.
            collar = cv2.dilate(anomaly.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
            valid = state['inside'] & ~collar
            rgb, w = sample_eye(patch, geom, valid, u, v)
            collected[side].append(dict(frame=i, rgb=rgb, w=w,
                                        luma=rgb.max(-1).astype(np.float32),
                                        du=0.0, dv=0.0, scale=1.0, ncc=0.0,
                                        anomaly_pixels=int(anomaly.sum()),
                                        ax=geom['ax'], ay=geom['ay']))
        print('R33_COLLECT frame', i, flush=True)

    report = dict(atlas_n=ATLAS_N, atlas_reach=ATLAS_REACH, rho_max=r32.RHO_MAX,
                  frames=len(indices), sides={},
                  registration='translation and isotropic scale; torsion NOT fitted',
                  generated_material_used=False, production_enabled=False,
                  artist_accepted=False,
                  code_sha256=r32.sha(__file__))
    for side in r32.SIDES:
        frames = collected[side]
        if len(frames) < 20:
            report['sides'][side] = dict(frames=len(frames), reason='too_few_frames')
            continue
        atlas, aw, history, seed_profile, rho, seed_sharp = build(
            frames, args.iterations, args.gain_window, do_register=not args.no_register)
        core = rho < r32.RHO_MAX
        prof = radial_profile(atlas, rho, aw & core)
        # Temporal stability of the illumination term, measured where it is applied:
        # the mean absolute change of the FIELD over the iris between adjacent frames,
        # smoothed against raw. Round 32's lesson is that a median hides this, so p95
        # and max are reported alongside.
        gm = aw & core
        basis = gain_basis(*atlas_grid())
        fields = np.stack([apply_gain(f['gain_coef'], basis)[gm] for f in frames])
        rawf = np.stack([apply_gain(f['gain_raw'], basis)[gm] for f in frames])
        step = np.abs(np.diff(fields, axis=0)).mean(axis=(1, 2))
        rstep = np.abs(np.diff(rawf, axis=0)).mean(axis=(1, 2))
        gmed = np.median(fields, axis=1)
        # Held-frame check: predict a frame's own CLEAN pixels from the atlas times its
        # smoothed gain, with the frame excluded from neither registration nor atlas.
        # This is a seen-data consistency number, not a generalization score, and it is
        # labelled that way in the round notes.
        errs = []
        for f in frames:
            m = f['valid'] & aw & core
            if m.sum() < 40:
                continue
            pred = atlas * f['gain']
            d = np.abs(linear_to_srgb(np.clip(pred[m], 0, 1))
                       - linear_to_srgb(np.clip(f['rgb_w'][m], 0, 1))) * 255
            errs.append(float(np.median(d)))
        report['sides'][side] = dict(
            frames=len(frames), skipped=len(skipped[side]),
            iterations=history,
            atlas_cells=int((aw & core).sum()),
            registered_frames=int(sum(1 for f in frames if f['registered'])),
            sharpness=dict(unregistered_seed=None if seed_sharp is None else round(seed_sharp, 5),
                           atlas=round(sharpness(atlas, aw & core), 5)),
            registration_shift_px=dict(
                rms=round(float(np.sqrt(np.mean([(f['du'] * f['ax']) ** 2
                                                 + (f['dv'] * f['ay']) ** 2 for f in frames]))), 4),
                max=round(float(np.max([np.hypot(f['du'] * f['ax'], f['dv'] * f['ay'])
                                        for f in frames])), 4)),
            scale_range=[round(float(min(f['scale'] for f in frames)), 3),
                         round(float(max(f['scale'] for f in frames)), 3)],
            ncc=dict(mean=round(float(np.mean([f['ncc'] for f in frames])), 5),
                     p5=round(float(np.percentile([f['ncc'] for f in frames], 5)), 5),
                     min=round(float(np.min([f['ncc'] for f in frames])), 5)),
            gain=dict(
                basis='constant + linear ramp in eye coordinates, per channel',
                field_median_over_frames=[round(float(x), 5) for x in np.median(gmed, axis=0)],
                field_range=[[round(float(x), 5) for x in gmed.min(0)],
                             [round(float(x), 5) for x in gmed.max(0)]],
                adjacent_step_smoothed=dict(median=round(float(np.median(step)), 5),
                                            p95=round(float(np.percentile(step, 95)), 5),
                                            max=round(float(step.max()), 5)),
                adjacent_step_raw=dict(median=round(float(np.median(rstep)), 5),
                                       p95=round(float(np.percentile(rstep, 95)), 5),
                                       max=round(float(rstep.max()), 5))),
            atlas_radial_profile_linear=prof,
            atlas_radial_profile_display=display_profile(prof),
            seed_radial_profile_display=display_profile(seed_profile),
            seen_frame_clean_median_display_error=dict(
                median=round(float(np.median(errs)), 4),
                p95=round(float(np.percentile(errs, 95)), 4),
                max=round(float(np.max(errs)), 4)))
        print('R33_ATLAS_SIDE', side, json.dumps(
            {k: report['sides'][side][k] for k in
             ('atlas_cells', 'registration_shift_px', 'ncc', 'sharpness', 'scale_range',
              'seen_frame_clean_median_display_error')}), flush=True)
        np.savez_compressed(root / ('atlas_%s.npz' % side), atlas=atlas.astype(np.float32),
                            weight=aw, rho=rho.astype(np.float32),
                            frames=np.array([f['frame'] for f in frames]),
                            gain_coef=np.stack([f['gain_coef'] for f in frames]).astype(np.float32),
                            du=np.array([f['du'] for f in frames], np.float32),
                            dv=np.array([f['dv'] for f in frames], np.float32),
                            scale=np.array([f['scale'] for f in frames], np.float32))
        # Atlas preview at 6x nearest neighbour, with the unregistered seed beside it.
        seed_w = np.stack([(f['w'] > 0.98).astype(np.float32) for f in frames])
        seed = np.divide(np.stack([f['rgb'] for f in frames]).sum(0),
                         np.maximum(seed_w.sum(0), 1)[..., None])
        pair = np.concatenate([r32.rgb8(seed), np.zeros((ATLAS_N, 4, 3), np.uint8),
                               r32.rgb8(atlas)], axis=1)
        big = cv2.resize(pair, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(root / ('atlas_%s.png' % side)), big[..., ::-1])

    r32.write_json(root / 'atlas.json', report)
    r32.write_json(shot.log(args.name + '.json'), report)
    print('R33_ATLAS_DONE', flush=True)


if __name__ == '__main__':
    main()
