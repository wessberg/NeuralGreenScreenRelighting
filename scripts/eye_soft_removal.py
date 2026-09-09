"""Remove the capture-room corneal highlight with a CONTINUOUS weight, not a mask.

Round 33's atlas was measurably the best observed-pixel estimate this project has had and
the artist still rejected the frames. Looking at them at 8x, three things are wrong, and
only one of them is about the estimate:

1. **The highlight is only partly removed on most frames.** Round 33 reported it -- a
   light residue on 112 of 140 left eyes -- but the reason it is *visible* is that how
   much survives changes frame to frame, so the eye alternates between a dark blob and a
   surviving bright speck. That reads as exactly the shape-and-colour flicker the artist
   rejected in round 32.
2. **Its detection threshold is refit per frame.** `eye_atlas_repair.detect` takes the
   MAD of this frame's own residual and multiplies by four. Round 33 fixed the fill's
   per-frame refit and left the *mask's* per-frame refit in place, so the footprint still
   breathes -- which is what its own IoU p5 of 0.19 / 0.23 was measuring.
3. **The result has no specular at all.** Peak display inside the footprint falls from
   .83 to .42, and nothing puts anything back, so a lit eye becomes a flat brown disc.
   That is not a temporal-stability problem and no amount of tuning removal fixes it. The
   other half is `render_eye_catchlight.py`, which puts a mirror where the cornea is and
   lets Cycles find the shot's own environment; `--catchlight <dir>` composites it here,
   through the tracked eyelid matte the rendered cornea cannot know about.

This module answers 1 and 2 with one construction. There is no mask. The weight is a
continuous ramp in the excess over the sequence prediction,

    w = clip((excess - k_lo * sigma) / ((k_hi - k_lo) * sigma), 0, 1)

with **one sigma per eye for the whole shot**, taken as the median over frames of each
frame's robust residual scale. Consequences, all of which are the point:

* A pixel just under the old threshold is now removed *partly* rather than not at all, so
  the sub-threshold tail of the highlight cannot survive as a speck. Round 33 needed
  three iterative mask-growth passes to chase that tail and still left residue.
* Nothing is binary, so there is no footprint to breathe, no IoU to collapse, and no
  morphological dilation/vote pair that can erode or jump onto the eyelid.
* The threshold no longer moves with this frame's noise.

Everything round 33 earned is kept unchanged: the sequence-wide registered atlas and its
smooth illumination field (`eye_sequence_atlas.py`), the registration bound, the
abstention when registration collapses at a blink, and the requirement that a repair
darken the eye rather than brighten it.

**Grain.** The fill is a pooled median of ~24 observations, so it carries none of the
plate's own sensor and compression noise, while every pixel around it does. That is why a
correctly-levelled fill still reads as a smooth smear against a grainy iris. `--grain`
matches the plate's own measured high-frequency scale in the changed region. It adds no
structure -- the noise is white, its amplitude is measured from this eye's own clean
pixels, and `--grain 0` turns it off.

Everything outside the reported weight is bit-exact source, and a frame whose eye is
closed, untracked or unpredicted abstains and keeps exact source.

Writes no production default.

    dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/eye_soft_removal.py \
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
from color_pipeline import linear_to_srgb, srgb_to_linear
from shot_paths import add_shot_argument, from_args
import eye_reflection_repair as r32
import eye_sequence_atlas as atl
from eye_atlas_repair import predict


def residual_scale(signed, inside):
    """Robust scale of the eye's residual against the atlas, for ONE frame.

    Used only to be pooled over the shot. The highlight is a small minority of the iris
    so a MAD is not dragged by it, but it is not immune either, which is exactly why the
    per-frame value must not set a per-frame threshold.
    """
    sample = signed[inside]
    if sample.size < 40:
        return None
    return float(np.median(np.abs(sample - np.median(sample))) * 1.4826)


def soft_weight(signed, under, inside, support, sigma, k_lo, k_hi, reach, blur=0.7,
                rim_reach=4.0, rim=True):
    """Continuous removal weight from the residual against the fixed sequence prediction.

    Spatially gated to the neighbourhood of the eye's own brightest excess, the same
    `reach` round 32 and 33 used, so a bright limbal-ring pixel on the far side of the
    iris cannot start a second removal. The gate is a hard radius on a soft field, not a
    threshold on the field itself: inside it the weight is still continuous.

    **Both signs of the residual are removed, and that reverses a standing rule.** Rounds
    28-33 all left the black crescent beside the highlight alone, on the grounds recorded
    in AGENTS.md: it is coding undershoot, not material -- green and blue exactly 0 and
    red 8-26 under the tagged limited-range expansion -- and the artist reads it as the
    pupil. The first half of that is exactly why it now goes. **That black is the codec's
    ringing AROUND the highlight; it exists only because the highlight does.** Delete the
    bright core and keep the ringing and what is left is a hard black crescent with a
    smooth fill inside it, which is the shape the artist rejected in rounds 32, 33 and
    the first pass of 34 and described as "weird brown stuff where the softbox reflection
    used to be". Removing one lobe of a two-lobed artefact was never going to look right.

    The negative lobe is removed under two constraints that keep it from eating real eye:
    it must lie within `rim_reach` pixels of the confident positive core, and it is
    scored on the same sequence sigma. Genuine dark iris texture elsewhere in the eye is
    not near the highlight and is not touched.
    """
    peak = np.where(inside, signed, -np.inf)
    if not np.isfinite(peak).any():
        return None, None
    top = float(peak.max())
    if top <= k_lo * sigma:
        return None, None
    seed = np.unravel_index(int(np.argmax(peak)), peak.shape)
    rows, cols = np.mgrid[:inside.shape[0], :inside.shape[1]]
    near = np.hypot(cols - seed[1], rows - seed[0]) <= reach
    span = max((k_hi - k_lo) * sigma, 1e-6)
    w = np.clip((signed - k_lo * sigma) / span, 0.0, 1.0).astype(np.float32)
    w *= (inside & near).astype(np.float32)
    if rim:
        core = (w > 0.5).astype(np.uint8)
        k = int(2 * np.ceil(rim_reach) + 1)
        collar = (cv2.dilate(core, np.ones((k, k), np.uint8)) > 0) & inside & near
        wn = np.clip((under - k_lo * sigma) / span, 0.0, 1.0).astype(np.float32)
        wn *= collar.astype(np.float32)
        w = np.maximum(w, wn)
    if blur > 0:
        # A sub-pixel blur, so the ramp has no single-pixel staircase of its own. It is
        # applied to the WEIGHT, never to the radiance: the plate's own texture outside
        # the changed region must stay bit exact.
        #
        # The blur may only SOFTEN THE EDGE, never erode the confident core. Blurring a
        # blob a few pixels across pulls its own maximum down, and the first version of
        # this reached full weight on a median of FOUR pixels per eye -- so the middle of
        # a highlight sitting 80 sigma above the prediction was still only ~90 % removed,
        # and the leftover 10 % of an 80-sigma excess is not small. Taking the maximum of
        # the blurred and unblurred fields keeps the core at 1 and the edge soft. It is
        # the same rule round 33 had to learn about its temporal vote: a smoothing step
        # over a mask may add, never subtract.
        w = np.maximum(cv2.GaussianBlur(w, (0, 0), blur), w)
        w *= (inside & near).astype(np.float32)
    # Where the atlas predicts nothing there is nothing to replace with, so the weight
    # tapers to zero rather than being cut, which would put a hard edge back. A distance
    # transform INSIDE the support, not a blur of it: a blur would leak non-zero weight
    # onto unsupported pixels, which is precisely where there is nothing to fill with.
    s = np.clip(cv2.distanceTransform(support.astype(np.uint8), cv2.DIST_L2, 3) / 2.0,
                0.0, 1.0)
    w *= s.astype(np.float32)
    # Quantize to the 8 bits the weight is WRITTEN with. The written weight is the
    # round's audit trail -- "everything outside it is bit-exact source" is only a
    # guarantee if the file says what was actually applied -- and a float weight of
    # 1/1000 rounds to zero on disk while still changing the pixel. Rounding here instead
    # makes the two agree by construction rather than by tolerance.
    w = np.rint(np.clip(w, 0.0, 1.0) * 255.0).astype(np.float32) / 255.0
    return w, dict(peak_excess=top, seed=[int(seed[0]), int(seed[1])])


def _peak(values, mask):
    """Peak display level over a mask, or None when the mask is empty.

    `--fill-iris` replaces the whole disc, so 'the iris outside the treated region' can
    legitimately be empty; reporting None beats crashing on an empty reduction.
    """
    if not np.any(mask):
        return None
    return round(float(linear_to_srgb(np.clip(values[mask].max(), 0, 1))), 4)


def local_gain(patch, base, ring, lo=0.5, hi=2.0):
    """One robust scale per channel, from the observed iris ring around the footprint.

    This replaces the per-frame illumination RAMP for the purpose of levelling the fill,
    and it exists because the ramp was measured to be the round's remaining flicker.

    The ramp (`eye_sequence_atlas.gain_basis`) is a constant plus a linear term in eye
    coordinates, fitted per frame in the log domain and clamped at exp(+/-2.5) -- a
    12-fold multiplicative range. It is fitted on the CLEAN iris and then evaluated
    INSIDE the footprint, where it has no evidence, so it extrapolates. Measured on the
    shipped runs: the filled region's level divided by the observed iris level right
    beside it traverses 3.1-5.0x on the left eye and 6.5-10.4x on the right across the
    shot, running from 0.21 to 2.19 -- the fill is sometimes BRIGHTER than the iris
    around it, which is not a thing a pupil does. The paired control, the untouched ring
    against its own source, traverses 1.000x, so that is the fill and not the eye.

    A local ratio cannot do that. It is one number per channel taken from ~100 observed
    pixels immediately around the footprint, so it tracks the moving lid shadow for the
    same reason the ramp was introduced -- the evidence is right there -- while being
    unable to extrapolate anywhere. The bound is 0.5-2.0, not 12x.
    """
    if ring.sum() < 40:
        return None
    k = np.empty(3, np.float32)
    for c in range(3):
        a, b = patch[..., c][ring], base[..., c][ring]
        ok = (a > 1e-4) & (b > 1e-4)
        if ok.sum() < 40:
            return None
        k[c] = float(np.clip(np.median(a[ok] / b[ok]), lo, hi))
    return k


def hf_scale(values, mask):
    """Scale of the plate's own high-frequency content, per channel, over `mask`.

    Measured in DISPLAY space, not linear, because that is where the noise was made: it
    is sensor read noise and h264 quantization, both of which act on the gamma-encoded
    signal. Measured in linear it comes out as one absolute number that is far too large
    for the dark pupil and too small for the bright limbus -- which, applied, speckles
    the pupil.

    A blur residual rather than a plain high-pass: it is insensitive to the smooth
    illumination ramp across the iris, which is signal, not noise.
    """
    if mask.sum() < 30:
        return None
    disp = linear_to_srgb(np.clip(values, 0.0, 1.0))
    hp = disp - cv2.GaussianBlur(disp, (0, 0), 1.0)
    return np.array([float(np.median(np.abs(hp[..., c][mask])) * 1.4826)
                     for c in range(values.shape[-1])], np.float32)


def add_grain(blended, weight, scale, rng):
    """Match the plate's measured HF scale inside the changed region.

    The fill is a pooled median of about two dozen observations and therefore carries
    about 1/sqrt(24) of the plate's noise. Matching it back is not invention: the
    amplitude is measured from this eye's own clean pixels in this frame, the field is
    white, and it is applied only where the weight is non-zero, scaled by it.

    Two things this gets right that the first version did not, both visible on the
    contact sheets as red and magenta speckle in the pupil and neither visible in any
    log:

    * **One shared noise field for all three channels**, scaled per channel. Independent
      per-channel noise is chroma noise, and this plate is 4:2:0 -- its chroma is stored
      at half resolution and carries almost no high-frequency content, so independent
      draws put colour where the source has none.
    * **Applied in display space.** A single absolute linear amplitude is enormous
      relative to a pupil sitting at 0.01-0.03 linear and small relative to the limbus.
    """
    if scale is None:
        return blended
    disp = linear_to_srgb(np.clip(blended, 0.0, 1.0))
    field = rng.standard_normal(blended.shape[:2]).astype(np.float32)
    disp = disp + field[..., None] * scale[None, None, :] * weight[..., None]
    return srgb_to_linear(np.clip(disp, 0.0, 1.0))


def lid_matte(points, topology, side, window, geom, feather_px=1.0, iris_feather=1.5):
    """A soft matte of the OPEN CORNEA: inside the eyelid polygon AND inside the limbus.

    Two gates, and both were learned from frames rather than from logs.

    The **lid** is the piece whose absence got stage 11d rejected. The rendered corneal
    reflection knows nothing about eyelids -- the proxy has none -- so a cornea that is
    half closed still returns a full disc, and adding it unmasked paints a silver arc
    across the upper lid.

    The **limbus** gate is this round's own: the renderer already truncates the corneal
    cap at the limbus, but the cap is aimed down the MEASURED gaze, whose two estimators
    disagree by 29.8 degrees on this plate, and a cap tilted by a fraction of that puts
    its edge a few pixels past the iris. On the first pass that dropped a white speck on
    the sclera, outside the eye's own dark iris, in plain sight on the contact sheet.
    Beyond the limbus there is sclera, not cornea, so there is nothing there to reflect
    a room in.

    Both feathered by about a pixel: a hard matte edge on a 25-pixel eye is a step.
    """
    d = r32.lid_distance(points, topology['eyelid_' + side], window)
    m = np.clip(d / max(feather_px, 1e-6), 0.0, 1.0).astype(np.float32)
    rho = np.hypot((geom['xx'] - geom['limbus'][0]) / geom['ax'],
                   (geom['yy'] - geom['limbus'][1]) / geom['ay'])
    # rho is in limbus radii; convert the feather from pixels to that unit.
    edge = iris_feather / max(min(geom['ax'], geom['ay']), 1e-6)
    m *= np.clip((1.0 - rho) / max(edge, 1e-6), 0.0, 1.0).astype(np.float32)
    return m


def eye_tile(source, images, weight, points, topology, zoom, source_frame, eyes, half=17):
    """Both eyes, source on top and every result below it, at eye scale.

    The r29 contact-sheet standard, extended to more than one result row because this
    round has two: the removal, and the removal plus the scene's own reflection. The
    artist has to be able to see them against the same source pixels on the same sheet.
    """
    cols = []
    for side, rec in zip(r32.SIDES, eyes):
        centre, _, _ = r32.limbus_ellipse(points, topology['iris_' + side])
        cx, cy = int(round(centre[0])), int(round(centre[1]))
        y0, y1 = max(0, cy - half), min(source.shape[0], cy + half)
        x0, x1 = max(0, cx - half), min(source.shape[1], cx + half)
        s = r32.rgb8(source)[y0:y1, x0:x1, ::-1].copy()
        w = weight[y0:y1, x0:x1]
        edge = cv2.dilate((w > 0.02).astype(np.uint8), np.ones((3, 3), np.uint8))
        edge = (edge > 0) & ~(w > 0.02)
        s[edge] = (0, 200, 255)
        stack = [s] + [r32.rgb8(im)[y0:y1, x0:x1, ::-1] for im in images]
        pair = np.concatenate(stack, axis=0)
        cols.append(cv2.resize(pair, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST))
    h = max(c.shape[0] for c in cols)
    cols = [np.pad(c, ((0, h - c.shape[0]), (0, 4), (0, 0))) for c in cols]
    tile = np.concatenate(cols, axis=1)
    label = np.zeros((22, tile.shape[1], 3), np.uint8)
    text = '%d %s' % (source_frame, ''.join('.' if e.get('repaired') else 'x' for e in eyes))
    cv2.putText(label, text, (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1,
                cv2.LINE_AA)
    return np.concatenate([label, tile], axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_shot_argument(ap)
    ap.add_argument('--name', default='r34_removal')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--count', type=int, default=150)
    ap.add_argument('--iterations', type=int, default=5)
    ap.add_argument('--gain-window', type=int, default=5)
    ap.add_argument('--no-union-footprint', dest='union_footprint', action='store_false',
                    help="stop unioning the tracked footprint into the removal weight. "
                         "On by default: without it the weight can collapse on frames "
                         "where the atlas is itself contaminated, and the softbox "
                         "survives on exactly those frames.")
    ap.add_argument('--fill-iris', type=float, default=0.0,
                    help="replace the WHOLE iris disc out to this radius in limbus radii, "
                         "not just the highlight footprint. 0 is off. This is round 31's "
                         "'render a new iris' with the one lesson every round since has "
                         "taught: the new iris is ONE image in the eye's registered frame, "
                         "rendered into all 150 frames, so it cannot flicker -- round 31 "
                         "generated per frame and wandered. Bounded by the tracked eyelid "
                         "and by atlas support; everything outside stays bit-exact source.")
    ap.add_argument('--pupil-target', type=float, default=0.0,
                    help="darken the canonical iris's centre so its pupil matches the "
                         "plate. The value is the radius in limbus radii to darken to; 0 "
                         "is off. The LEVEL is not a dial: it is measured from her own "
                         "frames as the darkest clean pupil actually observed, so this "
                         "deepens the pupil to something photographed rather than to "
                         "something chosen.")
    ap.add_argument('--atlas-mode', choices=('holes', 'replace'), default='holes',
                    help="how --atlas-frames material is used. 'holes' (default) builds "
                         "the atlas from HER OWN observed clean pixels exactly as round 33 "
                         "did, and takes the model's material ONLY in cells no frame ever "
                         "observed clean -- round 33's occupancy measured that at 1 cell "
                         "left and 71 right out of 507 / 621. So almost every filled pixel "
                         "is her real iris, photographed a few frames earlier or later, "
                         "and the generated fraction is reported rather than assumed. "
                         "'replace' pools the model's frames throughout.")
    ap.add_argument('--gain-model', choices=('local', 'ramp'), default='local',
                    help="how the fill is levelled into this frame. 'local' takes one "
                         "robust ratio per channel from the observed iris ring around "
                         "the footprint; 'ramp' is round 33's per-frame log-linear field, "
                         "which was measured to swing the fill/iris ratio by up to 10x "
                         "across the shot.")
    ap.add_argument('--k-lo', type=float, default=2.0,
                    help='excess, in sequence sigmas, where removal starts')
    ap.add_argument('--k-hi', type=float, default=4.0,
                    help='excess, in sequence sigmas, where removal is complete')
    ap.add_argument('--reach', type=float, default=8.0,
                    help='native-pixel radius around the brightest excess in the eye')
    ap.add_argument('--keep-rim', action='store_true',
                    help="leave the highlight's black undershoot rim in place, as rounds "
                         "28-33 did. Off by default: the rim is the codec's ringing "
                         "around the highlight, so removing the bright lobe and keeping "
                         "the dark one leaves a hard black crescent round a smooth fill.")
    ap.add_argument('--rim-reach', type=float, default=4.0,
                    help='pixels from the confident core within which the negative lobe '
                         'may be removed')
    ap.add_argument('--grain', type=float, default=1.0,
                    help='fraction of the measured plate HF scale to restore; 0 is off')
    ap.add_argument('--min-ncc', type=float, default=0.5)
    ap.add_argument('--atlas-frames', default=None,
                    help="build the sequence atlas from THESE frames instead of the "
                         "source. Point it at a generative inpainter's output "
                         "(composite/r34_inpaint/inpaint) and the model supplies the "
                         "material while the atlas supplies the temporal stability: the "
                         "fill becomes ONE image for the whole shot, pooled by a per-cell "
                         "median in the registered eye frame, so it CANNOT change between "
                         "frames. Taking a generator's per-frame output directly is what "
                         "made the colour inside the pupil crawl -- that is round 32's "
                         "per-frame refit again, with a neural network doing the refitting.")
    ap.add_argument('--catchlight', default=None,
                    help='directory of rendered corneal-reflection EXRs to ADD after removal')
    ap.add_argument('--catchlight-gain', default='auto',
                    help="scale from the environment's radiance units into the plate's "
                         "exposure. 'auto' measures it from the relight controls as "
                         "1 / median(E_target / E_source) at the eyes, which is the same "
                         "ratio composite_ratio.py multiplies the plate by; a number "
                         "overrides it.")
    ap.add_argument('--sheet-zoom', type=int, default=4)
    ap.add_argument('--sheet-frames', type=int, default=25)
    ap.add_argument('--seed', type=int, default=20340909)
    args = ap.parse_args()
    shot = from_args(args)
    if Path(args.name).name != args.name:
        raise ValueError('Expected a directory name, not a path')
    root = shot.composite / args.name
    subs = ['source', 'removal', 'weight', 'sheets'] + (['replace'] if args.catchlight else [])
    for sub in subs:
        (root / sub).mkdir(parents=True, exist_ok=True)

    meta = json.loads((shot.relight / 'face_landmarks.json').read_text())
    topology = meta['topology']
    with np.load(shot.relight / 'face_landmarks.npz') as z:
        landmarks = {k: z[k] for k in ('points', 'found', 'eyelid_aperture')}
    indices = list(range(args.start, args.start + args.count))
    if min(indices) < 0 or max(indices) >= shot.frame_count:
        raise ValueError('Requested interval leaves the shot')

    u, v = atl.atlas_grid()
    atlas_dir = Path(args.atlas_frames) if args.atlas_frames else None
    if atlas_dir is not None and not atlas_dir.is_dir():
        raise ValueError('No atlas frame directory at %s' % atlas_dir)
    sources, pooled, geoms = {}, {}, {}
    collected = {s: [] for s in r32.SIDES}
    collected_alt = {s: [] for s in r32.SIDES}
    for i in indices:
        raw = cv2.imread(str(shot.fg(i)), cv2.IMREAD_UNCHANGED)
        if raw is None or raw.dtype != np.float32 or raw.ndim != 3:
            raise ValueError('Expected scene-linear keyed RGB EXR at %s' % shot.fg(i))
        sources[i] = raw[..., ::-1].astype(np.float32)
        if atlas_dir is not None:
            alt = cv2.imread(str(atlas_dir / ('%06d.png' % i)))
            if alt is None:
                raise ValueError('No atlas frame %06d.png in %s' % (i, atlas_dir))
            pooled[i] = srgb_to_linear(alt[..., ::-1].astype(np.float32) / 255.0)
        else:
            pooled[i] = sources[i]
        if not (landmarks['found'][i] and np.isfinite(landmarks['points'][i]).all()):
            continue
        for side in r32.SIDES:
            geom, record = r32.eye_frame(sources[i], landmarks['points'][i], topology, side)
            if geom is None:
                continue
            patch = sources[i][geom['window']]
            state = r32.solve_eye(patch, geom)
            if state is None:
                continue
            # The atlas must be pooled from clean pixels only, so the footprint still has
            # to be found once, per frame, to be EXCLUDED. That is round 32's detector
            # used exactly as round 33 used it. What is new is downstream: nothing about
            # the removal weight is a threshold on this frame's own noise.
            anomaly, model, reason = r32.detect_anomaly(patch, state['rho'], state['theta'],
                                                        state['inside'])
            if anomaly is None:
                continue
            collar = cv2.dilate(anomaly.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            # When the atlas is pooled from an inpainted sequence the footprint is
            # already filled there, so excluding it would throw away exactly the
            # material this is being run for. The per-cell MEDIAN over ~140 frames is
            # what makes that safe: a frame where the model failed and left the
            # highlight is outvoted rather than averaged in.
            holes_mode = atlas_dir is not None and args.atlas_mode == 'holes'
            usable = (state['inside'] & ~collar
                      if (atlas_dir is None or holes_mode) else state['inside'])
            base_patch = sources[i][geom['window']] if (atlas_dir is None or holes_mode)                 else pooled[i][geom['window']]
            rgb, w = atl.sample_eye(base_patch, geom, usable, u, v)
            geoms[(i, side)] = geom
            collected[side].append(dict(frame=i, rgb=rgb, w=w,
                                        luma=rgb.max(-1).astype(np.float32),
                                        du=0.0, dv=0.0, scale=1.0, ncc=0.0,
                                        ax=geom['ax'], ay=geom['ay']))
            if holes_mode:
                arg, aww = atl.sample_eye(pooled[i][geom['window']], geom,
                                          state['inside'], u, v)
                collected_alt[side].append(dict(frame=i, rgb=arg, w=aww,
                                                luma=arg.max(-1).astype(np.float32),
                                                du=0.0, dv=0.0, scale=1.0, ncc=0.0,
                                                ax=geom['ax'], ay=geom['ay']))
        print('R34_COLLECT frame', i, flush=True)

    fitted = {}
    for side in r32.SIDES:
        frames = collected[side]
        if len(frames) < 20:
            fitted[side] = None
            continue
        atlas, aw, history, _, rho, seed_sharp = atl.build(frames, args.iterations,
                                                          args.gain_window)
        generated_cells = 0
        if atlas_dir is not None and args.atlas_mode == 'holes':
            # A second atlas from the model's frames, used ONLY where her own eye was
            # never observed clean. Everything else stays photographed.
            alt = collected_alt[side]
            if len(alt) >= 20:
                m_atlas, m_aw, _, _, _, _ = atl.build(alt, args.iterations,
                                                      args.gain_window)
                holes = (~aw) & m_aw & (rho < atl.FILL_REACH)
                generated_cells = int(holes.sum())
                atlas = np.where(holes[..., None], m_atlas, atlas)
                aw = aw | holes
                print('R34_HOLES', side, generated_cells, 'cells from the model of',
                      int((rho < atl.FILL_REACH).sum()), flush=True)
        core = rho < atl.FILL_REACH
        holes_before = int((core & ~aw).sum())
        atlas, aw = atl.fill_atlas_holes(atlas, aw, core)
        pupil_before = pupil_after = None
        if args.pupil_target > 0:
            # Measure the darkest pupil her plate actually shows, from clean pixels only,
            # then scale the canonical iris's centre down to it. A pooled median over the
            # shot sits well above the darkest observation because the highlight's own
            # bleed lifts most frames; the artist reads that as 'the pupil is not black
            # enough', and he is right -- the plate contains darker pupil than the pool.
            obs = []
            for fr in frames:
                lum = fr['rgb'].max(-1)
                sel = (fr['w'] > 0.98) & (np.hypot(*atl.atlas_grid()) < args.pupil_target)
                if sel.sum() >= 8:
                    obs.append(float(np.percentile(lum[sel], 5)))
            if obs:
                target = float(np.percentile(obs, 5))
                sel = rho < args.pupil_target
                cur = float(np.median(atlas[sel].max(-1))) if sel.any() else 0.0
                if cur > 1e-5 and target < cur:
                    ramp = np.clip((args.pupil_target - rho) /
                                   max(args.pupil_target * 0.45, 1e-6), 0.0, 1.0)
                    k = target / cur
                    atlas = atlas * (1.0 - ramp[..., None] * (1.0 - k))
                    pupil_before, pupil_after = round(cur, 5), round(target, 5)
                    print('R34_PUPIL', side, 'atlas centre', pupil_before, '->',
                          pupil_after, flush=True)
        fitted[side] = dict(atlas=atlas, aw=aw, history=history,
                            atlas_holes_interpolated=holes_before - int((core & ~aw).sum()),
                            atlas_holes_remaining=int((core & ~aw).sum()),
                            seed_sharpness=seed_sharp,
                            generated_cells=generated_cells,
                            pupil_before=pupil_before, pupil_after=pupil_after,
                            sharpness=atl.sharpness(atlas, aw & (rho < r32.RHO_MAX)),
                            by_frame={f['frame']: f for f in frames})
        print('R34_ATLAS_READY', side, len(frames), 'frames', flush=True)

    # --- pass A: predict once per frame, and pool ONE residual scale per eye -----------
    detail = {s: {} for s in r32.SIDES}
    scales = {s: [] for s in r32.SIDES}
    for side in r32.SIDES:
        fit = fitted[side]
        if fit is None:
            continue
        for i, f in fit['by_frame'].items():
            geom = geoms[(i, side)]
            patch = sources[i][geom['window']]
            state = r32.solve_eye(patch, geom)
            if args.gain_model == 'local':
                # Un-gained atlas first, then ONE robust ratio per channel from the
                # observed iris ring around the footprint. See local_gain().
                base0, support, (p, q) = predict(geom, fit['atlas'], fit['aw'],
                                                 f['du'], f['dv'], f['scale'],
                                                 np.zeros((3, 3)))
                anomaly, _, _ = r32.detect_anomaly(patch, state['rho'], state['theta'],
                                                   state['inside'])
                foot = (anomaly if anomaly is not None
                        else np.zeros(state['inside'].shape, bool))
                near_foot = cv2.dilate(foot.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
                ring = state['inside'] & support & ~near_foot & (state['rho'] < 0.85)
                k = local_gain(patch, base0, ring)
                if k is None:
                    # Not enough observed ring -- fall back to round 33's fitted field
                    # rather than levelling the fill on nothing.
                    base, support, (p, q) = predict(geom, fit['atlas'], fit['aw'],
                                                    f['du'], f['dv'], f['scale'],
                                                    f['gain_coef'])
                    gain_used = 'ramp_fallback'
                else:
                    base = np.maximum(base0 * k[None, None, :], 0.0)
                    gain_used = 'local'
            else:
                base, support, (p, q) = predict(geom, fit['atlas'], fit['aw'],
                                                f['du'], f['dv'], f['scale'],
                                                f['gain_coef'])
                anomaly, _, _ = r32.detect_anomaly(patch, state['rho'], state['theta'],
                                                   state['inside'])
                foot = (anomaly if anomaly is not None
                        else np.zeros(state['inside'].shape, bool))
                k, gain_used = None, 'ramp'
            signed = (patch - base).max(axis=-1)
            # The other lobe of the same artefact: where the plate is DARKER than the
            # prediction. Max over channels because the undershoot clips green and blue
            # at legal black while red survives at 8-26, so a luma-only test understates
            # it by exactly the channels that hit the floor.
            under = (base - patch).max(axis=-1)
            s = residual_scale(signed, state['inside'])
            if s is not None:
                scales[side].append(s)
            detail[side][i] = dict(base=base, support=support, inside=state['inside'],
                                   signed=signed, under=under, geom=geom, frame_scale=s,
                                   footprint=foot, gain_used=gain_used,
                                   local_gain=None if k is None else [round(float(x), 4) for x in k])
    # ONE sigma per eye for the whole shot. Round 33 recomputed this per frame and its
    # footprint breathed with it.
    sigma = {}
    for side in r32.SIDES:
        sigma[side] = float(np.median(scales[side])) if scales[side] else None
        print('R34_SIGMA', side, sigma[side],
              'p5', None if not scales[side] else round(float(np.percentile(scales[side], 5)), 6),
              'p95', None if not scales[side] else round(float(np.percentile(scales[side], 95)), 6),
              flush=True)

    catch, catch_gain, catch_gain_source = None, 1.0, None
    if args.catchlight:
        catch = Path(args.catchlight)
        if not catch.is_dir():
            raise ValueError('No catchlight render directory at %s' % catch)
        if str(args.catchlight_gain) == 'auto':
            # The corneal reflection is rendered in the ENVIRONMENT's radiance units and
            # added to a plate that is in its own exposure. The ratio composite already
            # carries the conversion between them -- it multiplies the plate by
            # R = E_target / E_source -- so the inverse of R at the eyes is the scale, and
            # it is measured rather than dialled.
            g = []
            for i in indices[::5]:
                cp = shot.relight / 'controls' / ('%06d.npz' % i)
                if not cp.exists():
                    continue
                with np.load(cp) as z:
                    gm = z['gain'].astype(np.float32)
                gh, gw = gm.shape[:2]
                for side in r32.SIDES:
                    d = detail[side].get(i) if fitted[side] is not None else None
                    if d is None:
                        continue
                    cx, cy = d['geom']['limbus']
                    gx = int(round(float(cx) * gw / sources[i].shape[1]))
                    gy = int(round(float(cy) * gh / sources[i].shape[0]))
                    sub = gm[max(0, gy - 3):gy + 4, max(0, gx - 3):gx + 4]
                    if sub.size:
                        g.append(float(np.median(sub.max(-1))))
            if not g:
                raise ValueError('--catchlight-gain auto needs relight/controls/*.npz')
            catch_gain = 1.0 / float(np.median(g))
            catch_gain_source = dict(measured='1 / median(E_target/E_source) at the eyes',
                                     samples=len(g),
                                     median_relight_gain=round(float(np.median(g)), 4))
        else:
            catch_gain = float(args.catchlight_gain)
            catch_gain_source = dict(measured=None, given=catch_gain)
        print('R34_CATCH_GAIN', round(catch_gain, 5), catch_gain_source, flush=True)

    rng = np.random.default_rng(args.seed)
    rows, sheet_tiles = [], []
    for i in indices:
        values = sources[i]
        out = values.copy()
        combined = np.zeros(values.shape[:2], np.float32)
        eyes = []
        for side in r32.SIDES:
            fit = fitted[side]
            d = detail[side].get(i) if fit is not None else None
            if d is None or sigma[side] is None:
                eyes.append(dict(side=side, repaired=False, reason='no_eye_frame'))
                continue
            geom = d['geom']
            f = fit['by_frame'][i]
            record = dict(side=side, limbus_centre=[round(float(c), 3) for c in geom['limbus']],
                          limbus_axes=[round(geom['ax'], 3), round(geom['ay'], 3)],
                          registration_shift_px=[round(float(f['du'] * geom['ax']), 4),
                                                 round(float(f['dv'] * geom['ay']), 4)],
                          registration_ncc=round(float(f['ncc']), 5),
                          frame_residual_scale=None if d['frame_scale'] is None
                          else round(d['frame_scale'], 6))
            if not f['registered'] or f['ncc'] < args.min_ncc:
                record.update(repaired=False, reason='registration_below_threshold')
                eyes.append(record)
                continue
            if args.fill_iris > 0:
                # The whole iris, as one image. No detection at all: there is nothing to
                # detect when every pixel of the disc is being replaced.
                #
                # The disc MUST be measured in the atlas's own registered coordinates,
                # not from the raw landmark. The content is sampled through
                # ((p - du) / scale), so a disc cut at the landmark disagrees with the
                # material inside it by exactly the registration offset -- and on frames
                # where the landmark ellipse is biased (a lid covering part of the iris
                # skews the four-point fit) the whole painted iris lands off her real
                # one and you see BOTH limbal edges. That is the double iris.
                geom_ = d['geom']
                p_ = (geom_['xx'] - geom_['limbus'][0]) / geom_['ax']
                q_ = (geom_['yy'] - geom_['limbus'][1]) / geom_['ay']
                au = (p_ - f['du']) / f['scale']
                av = (q_ - f['dv']) / f['scale']
                rho = np.hypot(au, av)
                edge = 1.5 / max(min(geom_['ax'], geom_['ay']), 1e-6)
                w = np.clip((args.fill_iris - rho) / max(edge, 1e-6), 0.0, 1.0).astype(np.float32)
                w *= (d['inside'] & d['support']).astype(np.float32)
                sup = np.clip(cv2.distanceTransform(d['support'].astype(np.uint8),
                                                    cv2.DIST_L2, 3) / 2.0, 0.0, 1.0)
                w = (w * sup).astype(np.float32)
                w = np.rint(np.clip(w, 0, 1) * 255.0).astype(np.float32) / 255.0
                info = dict(peak_excess=float(np.where(d['inside'], d['signed'], -np.inf).max()),
                            seed=[0, 0])
            else:
                w, info = soft_weight(d['signed'], d['under'], d['inside'], d['support'],
                                      sigma[side], args.k_lo, args.k_hi, args.reach,
                                      rim_reach=args.rim_reach, rim=not args.keep_rim)
            # UNION with the tracked footprint, never intersection. The excess-based
            # weight can collapse on a frame where the atlas itself still holds
            # highlight-contaminated material at those cells: there is then little
            # "excess" to find, and the artist sees the softbox survive on exactly those
            # frames. Measured on the frames he flagged, weighted_pixels fell from ~40 to
            # 15 and full-weight pixels from ~30 to 7 while the highlight was plainly
            # still there. The tracker knows where it is regardless of what the atlas
            # says, so it may only ADD -- the same rule round 33 learned about its
            # temporal vote.
            if (w is not None and args.union_footprint and args.fill_iris <= 0
                    and d.get('footprint') is not None):
                foot = d['footprint'] & d['inside'] & d['support']
                if foot.any():
                    fw = np.clip(cv2.GaussianBlur(foot.astype(np.float32), (0, 0), 0.7),
                                 0.0, 1.0)
                    fw = np.maximum(fw, foot.astype(np.float32))
                    fw = np.rint(fw * 255.0).astype(np.float32) / 255.0
                    w = np.maximum(w, fw)
            if w is None or w.max() <= 0:
                record.update(repaired=False, reason='no_reflection_found')
                eyes.append(record)
                continue
            patch = out[geom['window']].copy()
            heavy = w > 0.5
            if heavy.sum() < 3:
                record.update(repaired=False, reason='no_confident_pixels',
                              weighted_pixels=round(float(w.sum()), 3))
                eyes.append(record)
                continue
            # A removal must darken the HIGHLIGHT. If the prediction is brighter than
            # the source over the positive core, the atlas is in the wrong place and
            # blending would ADD light to the eye -- round 33's bright-blob-at-a-blink
            # bug. Scored on the positive core only: filling the undershoot rim raises
            # it on purpose, and testing the two lobes together would make the gate
            # reject every correct repair.
            bright = (d['signed'] > args.k_hi * sigma[side]) & (w > 0.5)
            if bright.sum() < 3:
                bright = heavy
            if float(d['base'][bright].max()) >= float(patch[bright].max()):
                record.update(repaired=False, reason='prediction_not_darker_than_source',
                              weighted_pixels=round(float(w.sum()), 3))
                eyes.append(record)
                continue
            blended = patch * (1 - w[..., None]) + d['base'] * w[..., None]
            grain = None
            if args.grain > 0:
                clean = d['inside'] & (w <= 0.0)
                grain = hf_scale(patch, clean)
                if grain is not None:
                    grain = grain * float(args.grain)
                    blended = add_grain(blended, w, grain, rng)
            blended = np.maximum(blended, 0.0)
            out[geom['window']] = np.where((w > 0)[..., None], blended, patch)
            reg = geom['window']
            combined[reg] = np.maximum(combined[reg], w)
            after = blended.max(axis=-1) - d['base'].max(axis=-1)
            # The footprint's own neighbourhood, defined the way round 33 defined it --
            # a 3-pixel collar on the CONFIDENT region -- so the residue numbers of the
            # two rounds describe the same piece of eye. Using the whole soft support
            # (w > 0.02) instead counts ordinary iris texture several pixels away and
            # reports a residue that has nothing to do with the highlight.
            near = cv2.dilate((w > 0.5).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
            near &= d['inside']
            # Round 33's own residue criterion, recomputed here so the two rounds can be
            # compared on identical terms: how many pixels near the footprint are still
            # more than FOUR TIMES THIS FRAME'S OWN robust spread above the prediction.
            # That per-frame spread is the thing this round stopped using as a threshold,
            # but it is still the number round 33 reported, and a new method that quietly
            # changed the yardstick would not be answering the complaint.
            r33_threshold = (max(4.0 * d['frame_scale'], 0.012)
                             if d['frame_scale'] is not None else None)
            record.update(
                repaired=True, reason=None,
                sequence_residual_scale=round(sigma[side], 6),
                r33_threshold_linear=(None if r33_threshold is None
                                      else round(float(r33_threshold), 6)),
                residual_pixels_above_r33_threshold=(
                    None if r33_threshold is None
                    else int((d['inside'] & near & (after > r33_threshold)).sum())),
                peak_excess_sigmas=round(float(info['peak_excess'] / sigma[side]), 3),
                weighted_pixels=round(float(w.sum()), 3),
                pixels_touched=int((w > 0).sum()),
                pixels_full_weight=int((w > 0.99).sum()),
                residual_pixels_above_2sigma=int((near & (after > 2 * sigma[side])).sum()),
                residual_peak_display=_peak(blended, d['inside'] & near),
                surround_peak_display=_peak(blended, d['inside'] & ~near),
                peak_source_display=round(float(linear_to_srgb(np.clip(
                    patch[bright].max(), 0, 1))), 4),
                peak_removed_display=round(float(linear_to_srgb(np.clip(
                    blended[bright].max(), 0, 1))), 4),
                rim_pixels_removed=int(((d['under'] > args.k_lo * sigma[side]) & (w > 0.5)).sum()),
                floor_pixels_removed=int((((patch <= 1e-6).sum(axis=-1) >= 2) & (w > 0.5)).sum()),
                grain_scale=None if grain is None else [round(float(g), 6) for g in grain],
                fill_display=[int(round(float(linear_to_srgb(np.clip(c, 0, 1)) * 255)))
                              for c in d['base'][heavy].mean(axis=0)])
            eyes.append(record)

        untouched = combined == 0
        assert np.isfinite(out).all() and (out >= 0).all()
        assert np.array_equal(out[untouched], values[untouched]), 'source changed outside weight'
        images = [out]
        replaced = None
        if catch is not None:
            raw_add = cv2.imread(str(catch / ('%06d.exr' % i)), cv2.IMREAD_UNCHANGED)
            if raw_add is None:
                raise ValueError('No catchlight frame %06d.exr' % i)
            raw_add = raw_add[..., :3][..., ::-1].astype(np.float32) * catch_gain
            # The rendered cornea has no eyelids. Gate it with the tracked lid polygon --
            # and ONLY inside the eyes' own windows, so nothing the render leaves
            # elsewhere in a 1920x1080 frame can reach the picture.
            matte = np.zeros(values.shape[:2], np.float32)
            if landmarks['found'][i] and np.isfinite(landmarks['points'][i]).all():
                for side, rec in zip(r32.SIDES, eyes):
                    d = detail[side].get(i) if fitted[side] is not None else None
                    if d is None:
                        continue
                    win = d['geom']['window']
                    m = lid_matte(landmarks['points'][i], topology, side, win,
                                  d['geom'])
                    matte[win] = np.maximum(matte[win], m)
                    e_before = float(raw_add[win].sum())
                    e_after = float((raw_add[win] * m[..., None]).sum())
                    rec['catchlight_energy_rendered'] = round(e_before, 4)
                    rec['catchlight_energy_after_lid'] = round(e_after, 4)
                    rec['catchlight_lid_kept'] = round(e_after / max(e_before, 1e-9), 4)
                    rec['catchlight_peak_added'] = round(
                        float((raw_add[win] * m[..., None]).max()), 4)
            add = raw_add * matte[..., None]
            replaced = np.maximum(out + add, 0.0)
            images.append(replaced)
        cv2.imwrite(str(root / 'source' / ('%06d.png' % i)), r32.rgb8(values)[..., ::-1])
        cv2.imwrite(str(root / 'removal' / ('%06d.png' % i)), r32.rgb8(out)[..., ::-1])
        weight8 = np.rint(np.clip(combined, 0, 1) * 255).astype(np.uint8)
        assert np.array_equal((weight8 > 0), (combined > 0)),             'the written weight and the applied weight disagree about which pixels changed'
        cv2.imwrite(str(root / 'weight' / ('%06d.png' % i)), weight8)
        if replaced is not None:
            cv2.imwrite(str(root / 'replace' / ('%06d.png' % i)), r32.rgb8(replaced)[..., ::-1])
        if landmarks['found'][i] and np.isfinite(landmarks['points'][i]).all():
            sheet_tiles.append(eye_tile(values, images, combined, landmarks['points'][i],
                                        topology, args.sheet_zoom,
                                        shot.source_start_frame + i, eyes))
        else:
            sheet_tiles.append(np.zeros((22 + 34 * (1 + len(images)) * args.sheet_zoom,
                                         2 * 34 * args.sheet_zoom + 8, 3), np.uint8))
        rows.append(dict(frame=i, source_frame=shot.source_start_frame + i,
                         source_sha256=r32.sha(shot.fg(i)),
                         changed_pixels=int((combined > 0).sum()),
                         weighted_pixels=round(float(combined.sum()), 3), eyes=eyes))
        print('R34_REMOVE frame', i, 'changed', int((combined > 0).sum()), flush=True)

    sheets = []
    for k in range(0, len(sheet_tiles), args.sheet_frames):
        path = root / 'sheets' / ('contact_%03d.jpg' % indices[k])
        assert cv2.imwrite(str(path), r32.contact_sheet(sheet_tiles[k:k + args.sheet_frames]),
                           [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        sheets.append(path.name)
        print('R34_SHEET', path.name, flush=True)

    repaired = [r for r in rows if r['changed_pixels']]
    record = dict(
        rows=rows, contact_sheets=sheets,
        frames=len(rows), frames_repaired=len(repaired),
        frames_exact_source=len(rows) - len(repaired),
        total_changed_pixels=int(sum(r['changed_pixels'] for r in rows)),
        sequence_residual_scale={s: sigma[s] for s in r32.SIDES},
        weight_ramp_sigmas=[args.k_lo, args.k_hi], reach_px=args.reach,
        gain_model=args.gain_model, union_footprint=bool(args.union_footprint),
        fill_iris=args.fill_iris, pupil_target=args.pupil_target,
        pupil_level={sd: (None if fitted[sd] is None else
                          [fitted[sd].get('pupil_before'), fitted[sd].get('pupil_after')])
                     for sd in r32.SIDES},
        grain_fraction=args.grain,
        atlas={s: (None if fitted[s] is None else dict(
            frames=len(fitted[s]['by_frame']), iterations=fitted[s]['history'],
            sharpness=round(fitted[s]['sharpness'], 5),
            unregistered_seed_sharpness=round(fitted[s]['seed_sharpness'], 5),
            holes_interpolated=fitted[s]['atlas_holes_interpolated'],
            holes_remaining=fitted[s]['atlas_holes_remaining']))
            for s in r32.SIDES},
        method='round 33 sequence atlas, with a CONTINUOUS removal weight on the excess '
               'over it and ONE residual scale per eye for the whole shot; no mask',
        atlas_frames=None if atlas_dir is None else str(atlas_dir),
        atlas_mode=args.atlas_mode,
        generated_cells={sd: (None if fitted[sd] is None else fitted[sd].get('generated_cells'))
                         for sd in r32.SIDES},
        material_source=('observed clean pixels of the same eye, pooled over the shot'
                         if atlas_dir is None else
                         'a generative video inpainter, pooled into ONE sequence-wide '
                         'atlas so the fill is identical in every frame'),
        generated_material_used=atlas_dir is not None,
        scene_reflection_added=bool(args.catchlight),
        catchlight_render=None if catch is None else str(catch),
        catchlight_gain=None if catch is None else round(catch_gain, 5),
        catchlight_gain_source=catch_gain_source,
        catchlight_matte='tracked eyelid polygon AND the limbus, both feathered ~1 px',
        dark_rim_repaired=not args.keep_rim,
        rim_reach_px=args.rim_reach,
        production_enabled=False, artist_accepted=False,
        source_outside_weight_exact=True,
        code_sha256=r32.sha(__file__), atlas_code_sha256=r32.sha(atl.__file__),
        track_sha256=r32.sha(shot.relight / 'face_landmarks.npz'))
    r32.write_json(root / 'removal.json', record)
    r32.write_json(shot.log(args.name + '.json'), record)
    print('R34_REMOVE_DONE', len(rows), 'frames,', len(repaired), 'repaired', flush=True)


if __name__ == '__main__':
    main()
