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
from limbus_fit import Track as LimbusTrack
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


def lid_inset_for(cfg, args):
    """How far to pull the fitted opening in, from the fit's OWN residual on that frame.

    A fixed inset treats a lid fit that agreed with its columns to a tenth of a pixel the
    same as one that missed them by a pixel and a half. The fit reports which it was, so
    the inset follows it: where the fit is confident the fill reaches the lash line, and
    where it is not the fill gives ground rather than repainting her lashes.
    """
    if cfg is None:
        return 0.0
    return float(args.lid_inset + args.lid_inset_rms * cfg.get('rms', 0.0))


def signed_lid_distance(points, topology, side, window):
    """Distance to the eyelid polygon, POSITIVE inside and NEGATIVE outside.

    `r32.lid_distance` is zero everywhere outside the polygon, which is fine for a matte
    and useless for asking "how far outside is this". Round 35 needs the second question
    because the polygon itself is wrong: it crosses the middle of the iris, three to five
    pixels below the actual lash line, with open eye and the softbox above it.
    """
    inside = r32.lid_distance(points, topology['eyelid_' + side], window)
    ys, xs = window
    poly = np.asarray(points)[topology['eyelid_' + side], :2]
    filled = np.zeros((ys.stop - ys.start, xs.stop - xs.start), np.uint8)
    cv2.fillPoly(filled, [np.rint((poly - [xs.start, ys.start]) * 4).astype(np.int32)],
                 1, cv2.LINE_8, 2)
    outside = cv2.distanceTransform((1 - filled).astype(np.uint8), cv2.DIST_L2, 3)
    return np.where(filled > 0, inside, -outside).astype(np.float32)


def soft_weight(signed, under, inside, support, sigma, k_lo, k_hi, reach, blur=0.7,
                rim_reach=4.0, rim=True, rim_region=None):
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
        # A CIRCULAR structuring element. A square one puts square corners on the collar,
        # and on a half-closed eye -- where the lid shadow makes the whole lower iris
        # darker than the canonical atlas, so `under` is large everywhere -- those corners
        # were plainly visible in the weight map as a rectangle painted across the lashes.
        # The undershoot ring is a ring; dilate it with a disc.
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        # The negative lobe stays INSIDE the eyelids even when the positive one is allowed
        # past them. A highlight can sit above a polygon that is drawn too low; "darker
        # than her iris" above the lash line is just lash.
        region = inside if rim_region is None else (inside & rim_region)
        collar = (cv2.dilate(core, kern) > 0) & region & near
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


def lash_ceiling(patch, geom, state, footprint=None, clamp_to_footprint=False,
                 guard=0.0, lash_run=3, smooth=9, pupil_rho=0.42, spec_level=0.45,
                 spec_reach=1.05, ring_px=3.0, ring_level=0.10):
    """The highest row the paint may touch, per column, read off HER OWN image.

    **The artist's rule, and it is a statement about physics rather than a tolerance:**
    *"the softbox reflection will never be present in the eyelashes or along the eyelid
    itself ... we should prevent us from drawing on her eyelid and lashes."* A corneal
    specular is on the cornea; the lash and the lid margin are opaque things in front of
    it. So nothing -- neither the iris fill nor the highlight removal -- has any business
    above the lash, and everything up there stays the photograph.

    **Why this is not another lid anchor.** `eyelid_fit.py` is fitted to the image and
    audited to 0.43 px, and it is still confidently wrong on some frames: measured on
    `r35_final`, source 727 left puts the fitted margin about seven pixels above the lash,
    up on the LID CREASE, and source 723 left abstains altogether. On both, round 35
    painted straight through her lashes -- which is what the artist is looking at. A bound
    that is sometimes wrong in the permissive direction cannot be the only thing standing
    between the fill and her lashes.

    This is a GUARD, not an anchor. It can only ever remove paint, never add any, so the
    worst case is that it gives up a pixel of iris; it cannot reintroduce a wrong-anchor
    artefact the way a new tracker could. It is combined with the fitted lid by taking
    whichever is more conservative.

    **The rule, per column: the lash is the DARKEST row above the eye's interior.** That
    is not a guess, it is what her columns say. Reading one down from the crop top at
    source 727 left, four pixels right of the iris centre:

        skin  .73 .73 .73 .71 .67 .61 .64 .64 .63    lid, then the crease
        fall  .55 .42                                the lid margin
        LASH  .24 .23 .19 .13 .13                    the darkest run in the column
        iris  .15 .18 .19 .21 .23                    rising monotonically below it

    The shadowed upper iris (.15-.23) and the lash (.13) are close enough in LEVEL that no
    threshold separates them -- which is why the first version of this guard, anchored on
    finding skin above, latched onto the corneal specular instead: at .78 it is brighter
    than her eyelid. What does separate them is that the lash is a MINIMUM and the iris
    below it only rises. So the ceiling is the argmin of the column, and the paint may
    start one pixel below it.

    Three things are excluded from the search or it finds them instead of the lash, and
    all three cost a pass: the corneal specular (brighter than skin), its coding
    undershoot (which reaches legal black, .00-.04, far darker than any lash), and the
    pupil (also darker than the lash). The first two are excluded together, by level, in
    the SOURCE; the pupil by radius.

    The per-column minima are median-filtered along x -- not fitted to a parabola. A fit
    is what `eyelid_fit.py` does and its failure mode is being confidently wrong on the
    crease; a median filter can only ever move the ceiling to a value one of its own
    neighbours actually measured.
    """
    lum = np.clip(linear_to_srgb(np.clip(patch, 0, 1)), 0, 1).max(-1)
    rho = state['rho']
    h, w = lum.shape
    centre_row = float(geom['limbus'][1] - geom['window'][0].start)
    # The specular and its undershoot ring, located in the SOURCE by level exactly as
    # `audit_r35_defects.py` locates them, so this shares nothing with the repair's own
    # footprint detector.
    spec = (lum > spec_level) & (rho < spec_reach)
    k = int(2 * np.ceil(ring_px) + 1)
    near = cv2.dilate(spec.astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    # `ring_level` is LEGAL BLACK, not merely dark, and the difference cost a pass. The
    # coding undershoot beside the highlight clips green and blue at exactly 0 with red
    # surviving at 8-26, so it sits at 0.02-0.09 display. Excluding everything under 0.22
    # within reach of the specular instead excluded HER LASH -- which is dark and, on a
    # 24-pixel iris, three pixels from the highlight -- so the walk stepped straight over
    # the thing it exists to find and put the ceiling down in the lower iris. That was
    # visible on the review sheet as the ceiling collapsing beside the highlight and in
    # no number anywhere.
    blocked = spec | (near & (lum < ring_level)) | (rho < pupil_rho)
    rows = np.arange(h)[:, None]
    # Search from the crop top to a little past the iris centre: on a half-closed eye the
    # lid margin is BELOW the centre, and stopping at the centre would miss the lash on
    # exactly the frames where the fill is most likely to run onto it.
    floor_row = int(np.ceil(centre_row + 0.45 * geom['ay']))
    band = rows <= floor_row
    search = np.where(band & ~blocked, lum, np.inf)
    # Only the columns that actually cross the iris get a ceiling. Outside them the fill
    # is already bounded by the limbus, and the nasal corner carries her tear line and
    # the canthal shadow -- both darker than any lash, and both pulling the argmin down
    # into the eye, which showed up on the review sheet as the ceiling giving away iris.
    p = (geom['xx'][0, :] - geom['limbus'][0]) / geom['ax']
    ceiling = np.full(w, np.inf, np.float32)
    for x in range(w):
        if abs(float(p[x])) > 1.0:
            continue
        rowsx = np.nonzero(np.isfinite(search[:, x]))[0]
        if rowsx.size < 6:
            continue
        # **Walk DOWN from the lid skin at the top of the crop.** Two earlier versions
        # searched from the eye outward -- the column's global minimum, then the first
        # minimum walking up -- and both kept landing on the pupil's own edge or on the
        # canthal shadow, several pixels below the lash, which the review sheet showed as
        # the ceiling stepping down across the middle of the iris. Nothing inside the eye
        # is as bright as her eyelid, so the skin above is the one unambiguous anchor;
        # the reason the FIRST version of this failed on it is that the corneal specular
        # is brighter still, and that is now masked out before the walk starts.
        skin_ref = float(np.median(lum[rowsx[:3], x]))
        if skin_ref <= 0.15:
            continue
        entry, floor_v = skin_ref * 0.60, skin_ref * 0.45
        got = None
        for n, t in enumerate(rowsx):
            if lum[int(t), x] >= entry:
                continue
            run = rowsx[n:n + 9]
            vals = lum[run, x]
            # A lid CREASE is a dark line with skin under it; a lash is far darker than
            # any crease and the eye is what lies below it. Requiring the run to reach
            # `floor_v` is what tells them apart, and it is why the fitted lid margin --
            # which takes an edge and cannot ask this question -- lands on the crease on
            # frames like source 727 left.
            if float(vals.min()) > floor_v:
                continue
            j = int(np.argmin(vals))
            # **The ceiling is the lash's OWN inner edge, not the darkest row plus a
            # fixed margin.** A fixed +2 px offset is what produced the artist's white
            # "tetris block": at source 721 left the lash minimum is at row 14 and the
            # highlight starts at row 17, so +2 landed inside the specular and sheared off
            # its upper-left corner at 0.45-0.62 display. Walking down from the minimum
            # while the column is still lash-dark ends exactly where the eye begins --
            # row 16 on that column, leaving the whole highlight reachable -- and on a
            # column with a thick lash it reaches further, which a fixed offset cannot.
            # The walk is capped so it cannot run on into a dark upper iris.
            k = j
            while (k + 1 < len(run) and k - j < lash_run
                   and float(vals[k + 1]) < floor_v):
                k += 1
            got = int(run[k])
            break
        if got is None:
            continue
        ceiling[x] = float(got)
    if not np.isfinite(ceiling).any():
        return None
    # Median-filtered along x over the columns that reported, then held flat outward.
    known = np.nonzero(np.isfinite(ceiling))[0]
    vals = ceiling[known]
    out = np.full(w, np.inf, np.float32)
    pad = smooth // 2
    for n, x in enumerate(known):
        win = vals[max(0, n - pad):n + pad + 1]
        out[x] = float(np.median(win))
    if known.size:
        out[:known[0]] = out[known[0]]
        out[known[-1] + 1:] = out[known[-1]]
        # Any column between two reporting columns that abstained takes the more
        # conservative of its neighbours rather than no ceiling at all.
        last = known[0]
        for x in range(known[0], known[-1] + 1):
            if np.isfinite(out[x]):
                last = x
            else:
                nxt = known[known > x]
                out[x] = max(out[last], out[nxt[0]]) if nxt.size else out[last]
    out = out + guard
    # **The ceiling may never cut below the specular, and this is the artist's own rule
    # read backwards.** He is right that a corneal reflection is never on a lash -- which
    # also means that wherever the reflection IS, that is cornea, and the repair must be
    # allowed to reach it. Without this the guard clipped the top of the softbox on the
    # frames where it sits high under a half-closed lid and left a light speck standing
    # at the top of the pupil, which is visible on the contact sheet at source 723 and
    # 736 left and is exactly the artefact the round exists to remove.
    #
    # The clamp is one pixel ABOVE the specular's own top row so its coding ringing goes
    # with it, which round 34 established has to happen: deleting the bright lobe and
    # keeping the black one leaves a hard crescent.
    # The clamp takes the specular as a COMPACT BLOB, not as "bright pixels", and it
    # lifts the ceiling by ONE value across the blob's whole column span. Both halves of
    # that cost a pass and are visible only in pixels:
    #
    # * `lum > spec_level` inside the limbus is not the specular. Her LID SKIN sits at
    #   0.71-0.77 and overlaps the iris disc whenever the lid is low, so at source 727
    #   left the clamp took the eyelid for the reflection, lifted the ceiling above the
    #   lash and painted canonical iris at 0.40 over lash pixels at 0.13 -- the exact
    #   thing the guard exists to stop. A corneal specular is a few pixels across; a lid
    #   is not, so the discriminator is SIZE, not level.
    # * A PER-COLUMN lift punches one-column holes through the fill wherever a column of
    #   the blob is a row shorter than its neighbours -- measured as `w = 255 255 0 255`
    #   across three adjacent columns, which reads as a comb. One value for the blob
    #   cannot do that.
    # **The ceiling may not cut below the specular, and the specular is supplied by the
    # repair's OWN tracked footprint rather than re-detected here.** Wherever the
    # reflection is, that is cornea -- the artist's rule read backwards -- so the repair
    # must be allowed to reach it. Without this the guard clips the top of a softbox
    # sitting high under a half-closed lid: measured on the specular core (rho < 0.6),
    # source 736 left goes from clean at 0.26 display to a **0.59 remnant**, and 845, 846
    # and 848 left keep 5-11 bright pixels each.
    #
    # **Re-detecting the specular inside this function was tried twice and both versions
    # painted her eyelid.** By level (`lum > 0.45` inside the limbus) it took her LID SKIN
    # at 0.71-0.77, lifted the ceiling above the lash and painted canonical iris at 0.40
    # over lash pixels at 0.13 -- source 727 left. Narrowed to compact blobs it then took
    # her WET LID MARGIN highlight, which is also compact, and reached seven rows into the
    # lid -- source 723 left. The footprint has neither failure mode because it is the
    # thing the round-32 detector already decided to remove: lifting the ceiling over it
    # only permits what the repair was going to do anyway.
    #
    # The lift is ONE value across the footprint's column span. Applied per column it
    # punched one-column holes through the fill -- measured as `w = 255 255 255 0 255 255`
    # across six adjacent columns, which reads as a comb.
    # **Growing the clamp to the footprint's whole BRIGHT COMPONENT was tried and it
    # broke the guard outright: eyes painting above the lash went from 0 to 164.** On a
    # half-closed eye the specular touches the bright sclera, so the component runs out
    # of the cornea, along the sclera and up under the lid, and the clamp then lifts the
    # ceiling across that entire span. The footprint alone has no such path -- it is a
    # bounded blob the round-32 detector fitted to the highlight -- so the clamp uses it
    # and nothing more. The cost is a few fringe pixels of specular left standing on
    # squinting frames, recorded in round36.md; the alternative was her eyelid.
    # **The clamp is OFF, and `footprint` is accepted only so this stays measurable.**
    # Even bounded to the round-32 footprint it puts paint back on her lashes: measured
    # over all 300 eyes, 56 eyes painted above the lash line, median 24 px and up to 80,
    # concentrated on exactly the half-closed frames (715-718, 826, 836-838) where the
    # lid fit is least reliable and the specular sits closest to the lash. Grown to the
    # footprint's bright component it was 164 eyes.
    #
    # The artist's rule is not a tolerance -- *"we do not draw our own pixels ON or above
    # that eyelid arc"* -- so the guard wins and the residue is reported instead: a few
    # fringe pixels of specular survive on squinting frames (source 843, 845, 846, 848
    # left, 5-11 px each at 0.66-0.81 display). That is round 37's problem, and the way
    # in is a lash line that is right on a squint, not a licence to paint over it.
    if footprint is not None and clamp_to_footprint and np.any(footprint):
        cols = np.nonzero(footprint.any(axis=0))[0]
        rows_f = np.nonzero(footprint.any(axis=1))[0]
        if cols.size and rows_f.size:
            out[cols.min():cols.max() + 1] = np.minimum(
                out[cols.min():cols.max() + 1], float(rows_f.min()) - 1.0)
    # A COMB is the failure mode of a per-column bound, and it is what a one-row
    # difference between neighbouring columns looks like once the fill starts at the
    # ceiling: measured at source 727 left as `w = 255 255 255 0 255 255` across six
    # adjacent columns -- single columns of untouched source standing inside the fill.
    # A morphological close along x removes a one-column spike without letting the
    # ceiling rise anywhere it was not already.
    if out.size >= 3:
        pad = np.pad(out, 1, mode='edge')
        closed = np.array([max(pad[i:i + 3]) for i in range(out.size)], np.float32)
        pad2 = np.pad(closed, 1, mode='edge')
        out = np.array([min(pad2[i:i + 3]) for i in range(out.size)], np.float32)
    return out.astype(np.float32)


def lid_depth(cfg, geom):
    """Native pixels BELOW the fitted upper lid margin, per pixel of the eye patch.

    The lid shadow is cast by the lid, so its natural coordinate is depth under the lid
    edge -- not eye-radial distance, which is what every gain model in rounds 32-35 used.
    A lid moves and an iris does not, so a shadow written in iris coordinates has to be
    refit every frame (round 33's ramp: it flickered) or be constant (round 34's scalar:
    it cannot bend). Written in lid coordinates it is neither -- ONE shape for the shot,
    riding on a margin that round 35 measures to 0.43 px.
    """
    if not cfg or not cfg.get('upper'):
        return None
    a, b, c = cfg['upper']
    u = geom['xx'] - cfg['centre'][0]
    return (geom['yy'] - (a * u ** 2 + b * u + c)).astype(np.float32)


def fit_lid_shadow(samples, lo=-0.6, hi=0.6, tau_lo=1.0, tau_hi=10.0, min_bin=120):
    """One bounded, monotone lid-shadow correction per eye, from pooled observations.

    `samples` is a list of (depth, ratio) arrays -- her own observed iris divided by the
    canonical iris painted at the same place, with THIS FRAME's scalar gain already
    divided out. So a ratio of 1 means the scalar was enough and anything else is the
    shape the scalar cannot carry.

    **Why this exists.** Measured on `r35_final`, pooled over 150 frames with the
    specular and its coding ring excluded, her iris and the repair diverge in a band
    three to six pixels below the lash and agree everywhere below it:

        depth   3     4     5     6     7
        left    .81   .86   .90   .91   .92    source
                .66   .65   .77   .89   .93    R35 result
        right   .80   .85   .91   .95   .95    source
                .86   .93   .96   .98   .99    R35 result

    -- a step of up to 0.20 display on the left and 0.09 the other way on the right,
    switching on exactly where the fill's weight does. That is the artist's "not
    seamless around the upper eyelid", and no eyelid mask removes it: moving the
    boundary moves the step, it does not close it.

    **The form is `1 + A * exp(-d / tau)`.** Bounded by construction (it can only
    approach 1 + A at the lash and 1 in the open iris), monotone in depth, two
    parameters fitted on thousands of pooled samples -- so it cannot do what round 33's
    unconstrained per-frame linear ramp did, which was extrapolate 12-fold into a region
    it had no evidence for. A is SIGNED: the measurement above says the canonical iris
    is too dark near the lash on one eye and too bright on the other, so a model that
    could only darken would be fitting half the defect.

    The fit is on per-depth MEDIANS, not on the samples, so a lash pixel that survives
    the level guard is outvoted rather than averaged in, and bins are weighted by count.
    """
    if not samples:
        return None
    d = np.concatenate([s[0] for s in samples])
    r = np.concatenate([s[1] for s in samples])
    ok = np.isfinite(d) & np.isfinite(r) & (r > 0.2) & (r < 5.0)
    d, r = d[ok], r[ok]
    if d.size < 2000:
        return None
    bins, meds, counts = [], [], []
    for b in range(0, 24):
        sel = (d >= b) & (d < b + 1)
        n = int(sel.sum())
        if n >= min_bin:
            bins.append(b + 0.5)
            meds.append(float(np.median(r[sel])))
            counts.append(n)
    if len(bins) < 6:
        return None
    bins = np.asarray(bins); meds = np.asarray(meds); counts = np.asarray(counts, float)
    w = counts / counts.sum()
    best = None
    for tau in np.linspace(tau_lo, tau_hi, 46):
        basis = np.exp(-bins / tau)
        # Least squares in A alone at this tau, weighted by how many pixels voted.
        denom = float(np.sum(w * basis * basis))
        if denom < 1e-9:
            continue
        A = float(np.sum(w * basis * (meds - 1.0)) / denom)
        A = float(np.clip(A, lo, hi))
        resid = float(np.sum(w * (meds - (1.0 + A * basis)) ** 2))
        if best is None or resid < best[2]:
            best = (A, float(tau), resid)
    if best is None:
        return None
    A, tau, resid = best
    flat = float(np.sum(w * (meds - 1.0) ** 2))
    return dict(amplitude=round(A, 5), tau_px=round(tau, 4),
                weighted_rms=round(float(np.sqrt(resid)), 5),
                weighted_rms_if_flat=round(float(np.sqrt(flat)), 5),
                samples=int(d.size), bins=len(bins),
                observed=[[round(float(b), 1), int(c), round(float(m), 4)]
                          for b, c, m in zip(bins, counts, meds)])


def lid_shadow_field(shadow, depth, strength=1.0):
    """`1 + A exp(-d/tau)` evaluated on this frame's depth map, held flat above the lid.

    Above the margin (d < 0) the correction is HELD at its d = 0 value rather than
    allowed to keep growing: the exponential runs away upward and there is no evidence
    up there anyway -- it is lash. The fill is bounded by the lid, so this only matters
    for the few pixels the removal is allowed past it.
    """
    if shadow is None or depth is None:
        return None
    d = np.maximum(depth, 0.0)
    return (1.0 + strength * shadow['amplitude']
            * np.exp(-d / max(shadow['tau_px'], 1e-3))).astype(np.float32)


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
    ap.add_argument('--min-ncc-frozen', type=float, default=0.2,
                    help="the NCC floor that applies when --freeze-registration is on. "
                         "The 0.5 gate exists because the render used to SAMPLE THE ATLAS "
                         "through this frame's own fitted offset, so a bad fit meant bad "
                         "material. With the offset frozen to a shot constant the NCC is "
                         "a report on the atlas build, not on the render, and vetoing on "
                         "it drops a single frame out of a repaired run -- the softbox "
                         "reappears for one frame and the eye pops. Source 720 right, "
                         "NCC 0.458 between two frames at 0.898 and 0.942, was exactly "
                         "that. The floor stays, low, to catch a frame that is broken.")
    ap.add_argument('--atlas-reach', type=float, default=1.25,
                    help="how far past the limbus the canonical eye's own grid extends, "
                         "in limbus radii. The grid used to stop AT the limbus and was "
                         "only sampled to 0.88, so the limbal ring had no canonical "
                         "material and a specular lapping onto it could not be repaired "
                         "-- 92.5 percent of the softbox pixels still standing after "
                         "removal were there, at median rho 1.022 with zero weight.")
    ap.add_argument('--atlas-cells', type=int, default=121,
                    help='cells across the canonical eye, kept in step with --atlas-reach '
                         'so the cell stays about 0.27 native pixels on this plate')
    ap.add_argument('--sample-reach', type=float, default=1.10,
                    help='radius, in limbus radii, out to which each frame is sampled '
                         "INTO the canonical eye. Round 32's RHO_MAX of 0.88 was a "
                         "constant for its radial profile model, not a statement about "
                         "how far her iris goes.")
    ap.add_argument('--lid-inset-rms', type=float, default=2.0,
                    help="extra lid inset per native pixel of the lid fit's own column "
                         "residual on that frame, so an uncertain fit gives more ground")
    ap.add_argument('--atlas-lid-inset', type=float, default=3.0,
                    help="native pixels to pull the eye opening in before POOLING the "
                         "canonical iris. This is separate from --lid-inset, which "
                         "bounds where the canonical iris is PAINTED, and it is the "
                         "reason the two exist. "
                         "The canonical iris is a per-cell median over ~140 registered "
                         "frames. Sampled through the same mask it is painted through, "
                         "every cell near the lid is pooled from whatever the lid fit "
                         "let in on the frames that reached it -- and where the fit runs "
                         "a pixel high, that is lash and lid skin. The median does not "
                         "save it: if the fit is high on most contributing frames, most "
                         "votes are lash. The result is a warm arc following the top of "
                         "the mask in the rendered eye that is nowhere in her source "
                         "photograph, which is exactly what the artist saw at source "
                         "726-728 left. Cells left empty by this inset are interpolated "
                         "from clean iris by fill_atlas_holes, which is what that stage "
                         "is for.")
    ap.add_argument('--atlas-cap', type=float, default=0.0,
                    help="percentile of her OWN observed clean iris radiance that the "
                         "canonical iris may not exceed. 0 disables it. One number per "
                         "eye, applied once to the canonical image, so it cannot "
                         "flicker; it exists because the atlas rim pools lid margin and "
                         "lash specular on the few frames those cells are visible, and "
                         "the fitted eyelid is what finally renders them.")
    ap.add_argument('--lash-bright', type=float, default=0.45,
                    help='source display luma above which the LASH GUARD does not apply. '
                         'Her lashes run 0.13-0.35 and her lid-shadowed iris 0.15-0.30, '
                         'while a corneal specular is 0.45-0.90, so this exempts the '
                         'reflection from the guard without exempting any lash. It is '
                         'needed because lash and shadowed iris merge into one dark run '
                         'on this plate, so the photometric lash line can land inside the '
                         'highlight -- source 740 left.')
    ap.add_argument('--lash-ring', type=float, default=2.0,
                    help="native px around an exempted specular within which its own "
                         "legal-black coding undershoot is exempted too. Removing the "
                         "bright lobe and leaving the dark one puts a black crescent "
                         "round the fill, which is the artefact rounds 32-34 were "
                         "rejected for.")
    ap.add_argument('--no-lash-guard', dest='lash_guard', action='store_false',
                    help='disable the image-derived LASH CEILING. It is on by default: '
                         "the artist's rule is that a corneal specular is never on a lash "
                         'or a lid, so nothing may be painted there, and the fitted lid '
                         'margin alone does not enforce it -- it abstains on some frames '
                         'and lands on the lid crease on others. See lash_ceiling().')
    ap.add_argument('--lid-shadow', type=float, default=1.0,
                    help='strength of the fitted LID SHADOW, one shape per eye for the '
                         'shot, as a multiple of the fitted amplitude. 0 disables it and '
                         "restores round 34's scalar-only fill. The canonical iris is "
                         'painted with one scalar gain per frame, so it cannot carry the '
                         "shadow her lid casts on her own iris; round 35 measured the "
                         'resulting step at up to 0.20 display three to six pixels below '
                         'the lash, which is the artist\'s "not seamless around the upper '
                         'eyelid". See fit_lid_shadow().')
    ap.add_argument('--shadow-inset', type=float, default=1.0,
                    help='native px inside the fitted opening that the lid-shadow '
                         'evidence is pooled from. Smaller than --atlas-lid-inset on '
                         'purpose: the shape has to be measured where the shadow IS, and '
                         'the per-depth median over thousands of samples is what rejects '
                         'a lash rather than a wide margin.')
    ap.add_argument('--shadow-reference', type=float, default=8.0,
                    help='depth below the lid margin, native px, beyond which the iris '
                         'is taken to be out of lid shadow. Each frame is normalised by '
                         'its own level there before pooling, so what is fitted is only '
                         'the part the per-frame scalar cannot represent.')
    ap.add_argument('--lid-inset', type=float, default=2.0,
                    help="native pixels to pull the FITTED eye opening in before filling. "
                         "The upper-lid fit's own column residual is 0.43 px median and "
                         "0.59 p95, so one pixel is about 1.7x its p95: where the fit "
                         "lands high the fill would otherwise paint canonical iris onto "
                         "the lashes, which reads as a bright band exactly where the lid "
                         "meets the iris. Erring inward leaves a sliver of her own eye; "
                         "erring outward repaints her lash line. "
                         "2.0 is where the two failures cross, measured on all 271 "
                         "treated eyes: at 1.0 px, 140 eyes carry a brightened lash and "
                         "4 carry a single surviving softbox pixel; at 2.0, 113 and 4; "
                         "at 3.0 the lash count falls to 81 but the softbox comes back "
                         "on 7 eyes at up to 4 px and ten eyes stop being treated at all."
                         " Past 2 px the fill is buying lash safety with softbox.")
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
    ap.add_argument('--eyelid-track', default='eyelid_track.json',
                    help="relight/<file> from eyelid_fit.py: both lid margins fitted to "
                         "the IMAGE. The mediapipe contour it replaces sits 8.34 px "
                         "inside the true opening on the left eye -- it crosses the "
                         "middle of the iris with the softbox above it -- and 1.01 px "
                         "outside it on the right, so nothing but a fit can serve both. "
                         "Pass '' to go back to the polygon for comparison.")
    ap.add_argument('--lid-extend', type=float, default=0.0,
                    help="native pixels the highlight REMOVAL may reach outside the "
                         "mediapipe eyelid polygon. **DEFAULTS TO 0, and that is a "
                         "measurement, not caution.** The polygon does cross the middle "
                         "of the iris with the softbox above it, so bounding removal by "
                         "it leaves the highlight standing -- but its error is not a "
                         "margin that can be added back: measured against the image's "
                         "own sclera-to-lash edge, the polygon sits 8.34 px INSIDE the "
                         "true opening on the left eye (p25 15.98) and 1.01 px OUTSIDE "
                         "it on the right. No single extension is safe for both, and a "
                         "generous one paints the canonical iris onto her lashes on "
                         "every half-closed frame. The eyelid contour has to be FITTED "
                         "to the image, exactly as limbus_fit.py fits the limbus. The "
                         "iris FILL is never extended -- only the removal.")
    ap.add_argument('--no-freeze-registration', dest='freeze_registration',
                    action='store_false',
                    help="refit the atlas registration per frame when rendering it back. "
                         "OFF by default: it moves the whole painted iris up to 1.62 px "
                         "between adjacent frames, which is the jitter the artist sees.")
    ap.add_argument('--limbus-track', default='limbus_track.json',
                    help="relight/<file> from limbus_fit.py: the limbus fitted to the "
                         "IMAGE EDGE. This is the anchor everything here sits on. The "
                         "mediapipe four-point ellipse it replaces sits 3.08 px from her "
                         "own iris centre on the left eye (median, independent estimator) "
                         "and its error jumps up to 10 px between adjacent frames -- the "
                         "artist's 'double iris' and his 'delayed, slow to react'. Pass "
                         "'' to go back to the landmark for comparison.")
    ap.add_argument('--sheet-zoom', type=int, default=4)
    ap.add_argument('--sheet-frames', type=int, default=25)
    ap.add_argument('--seed', type=int, default=20340909)
    args = ap.parse_args()
    shot = from_args(args)
    track = (LimbusTrack.load(shot.relight / args.limbus_track)
             if getattr(args, 'limbus_track', '') else None)
    lids = {}
    if getattr(args, 'eyelid_track', ''):
        for row in json.loads((shot.relight / args.eyelid_track)
                              .read_text(encoding='utf-8'))['rows']:
            for e in row['eyes']:
                if e.get('fitted'):
                    lids[(int(row['frame']), e['side'])] = dict(
                        centre=e['centre'],
                        upper=(e['upper'] or {}).get('coef'),
                        lower=(e['lower'] or {}).get('coef'),
                        rms=max((e['upper'] or {}).get('rms', 0.0),
                                (e['lower'] or {}).get('rms', 0.0)))
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

    atlas_grid_config = atl.configure(args.atlas_reach, args.atlas_cells)
    print('R35_ATLAS_GRID', atlas_grid_config, flush=True)
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
            # The anchor. When a fitted track is supplied and has no fit for this eye on
            # this frame, the eye ABSTAINS rather than falling back on the landmark: a
            # wrong anchor is the bug this replaces, and silently reinstating it on the
            # hardest frames is how it would come back.
            override = track.get(i, side) if track is not None else None
            if track is not None and override is None:
                continue
            geom, record = r32.eye_frame(sources[i], landmarks['points'][i], topology,
                                         side, limbus_override=override,
                                         eyelid_override=lids.get((i, side)),
                                         lid_inset=lid_inset_for(lids.get((i, side)),
                                                                 args))
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
            # The canonical eye is sampled out to --sample-reach, not to the round-32
            # profile radius; `state['inside']` stops at RHO_MAX = 0.88.
            strict = geom['visible'] & (state['rho'] < args.sample_reach)
            if args.atlas_lid_inset > 0 and lids.get((i, side)) is not None:
                poly = np.asarray(landmarks['points'][i])[topology['eyelid_' + side], :2]
                lid_cfg = lids[(i, side)]
                strict = strict & (r32.parabola_opening(
                    lid_cfg.get('upper'), lid_cfg.get('lower'), lid_cfg['centre'],
                    geom['window'], (float(poly[:, 0].min()), float(poly[:, 0].max())),
                    inset=args.lid_inset + args.atlas_lid_inset) > 0.5)
            usable = (strict & ~collar
                      if (atlas_dir is None or holes_mode) else strict)
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
        # **The registration is frozen after the atlas is built, and this is the last
        # per-frame refit in the eye path.** Building the atlas WANTS a per-frame
        # alignment: it is what makes the pooled median sharp. Rendering it back does
        # not. With round 34's landmark anchor the per-frame offset was doing real work
        # -- the anchor was 3.08 px off her iris and moving -- but the round 35 anchor is
        # 0.42 px, so what is left in the offset is mostly NCC chasing the lid shadow and
        # the highlight, and it TRANSLATES THE WHOLE PAINTED IRIS by a median 0.21 px and
        # up to 1.62 px between adjacent frames. That is the artist's "our iris copy
        # sometimes shifts a little bit around in-between frames", measured. One offset
        # per eye for the shot removes it by construction, exactly as one radius per eye
        # and one atlas per eye did.
        registration = None
        if args.freeze_registration:
            registration = dict(du=float(np.median([f['du'] for f in frames])),
                                dv=float(np.median([f['dv'] for f in frames])),
                                scale=float(np.median([f['scale'] for f in frames])),
                                du_p95=float(np.percentile([f['du'] for f in frames], 95)),
                                dv_p95=float(np.percentile([f['dv'] for f in frames], 95)))
            for f in frames:
                f['du'], f['dv'], f['scale'] = (registration['du'], registration['dv'],
                                                registration['scale'])
            print('R35_FREEZE', side, {k: round(v, 5) for k, v in registration.items()},
                  flush=True)
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
        core = rho < min(args.sample_reach, atl.ATLAS_REACH - 0.02)
        holes_before = int((core & ~aw).sum())
        atlas, aw = atl.fill_atlas_holes(atlas, aw, core)
        # **The canonical iris may not be brighter than her own iris.** Cells near the
        # atlas rim are observed on few frames and only when the lid is high, so what
        # gets pooled there is partly lid margin and lash specular. With the eyelid
        # bounded by the mediapipe polygon those cells were never rendered; with the
        # fitted lid they are, and the artist saw the result at once -- source 727 left,
        # a pixel at 0.298 display coming out at 0.976, "the point where the eyelid meets
        # the iris becomes really bright".
        #
        # The cap is ONE number per eye, measured from her own well-supported observed
        # iris, applied to the canonical image once. It cannot flicker, because nothing
        # about it is per frame. A per-pixel clamp against each frame's own source would
        # fix the same artefact and reintroduce exactly the per-frame dependence that
        # rounds 31-34 were rejected for.
        atlas_cap = None
        if args.atlas_cap > 0:
            obs = np.concatenate([fr['rgb'].max(-1)[(fr['w'] > 0.98) & core]
                                  for fr in frames if ((fr['w'] > 0.98) & core).any()])
            if obs.size >= 200:
                atlas_cap = float(np.percentile(obs, args.atlas_cap))
                luma = atlas.max(-1)
                over = luma > atlas_cap
                if over.any():
                    scale = np.ones_like(luma)
                    scale[over] = atlas_cap / np.maximum(luma[over], 1e-6)
                    atlas = atlas * scale[..., None]
                print('R35_ATLAS_CAP', side, round(atlas_cap, 5), 'clamped',
                      int(over.sum()), 'of', int(core.sum()), 'cells', flush=True)
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
                            frozen_registration=registration, atlas_cap=atlas_cap,
                            atlas_holes_interpolated=holes_before - int((core & ~aw).sum()),
                            atlas_holes_remaining=int((core & ~aw).sum()),
                            seed_sharpness=seed_sharp,
                            generated_cells=generated_cells,
                            pupil_before=pupil_before, pupil_after=pupil_after,
                            sharpness=atl.sharpness(atlas, aw & (rho < r32.RHO_MAX)),
                            by_frame={f['frame']: f for f in frames})
        print('R34_ATLAS_READY', side, len(frames), 'frames', flush=True)

    # --- pass A0: pool the LID SHADOW, once per eye for the shot -----------------------
    # Round 34 painted the canonical iris with one scalar per frame and round 35 measured
    # what that costs: a step of up to 0.20 display three to six pixels below the lash,
    # on essentially every treated frame, because a scalar cannot carry a gradient. The
    # shape is pooled here from every frame at once -- exactly as the canonical iris
    # itself is -- so nothing about it can move between frames. See fit_lid_shadow().
    prepass = {s: {} for s in r32.SIDES}
    shadow_samples = {s: [] for s in r32.SIDES}
    for side in r32.SIDES:
        fit = fitted[side]
        if fit is None:
            continue
        for i, f in fit['by_frame'].items():
            geom = geoms[(i, side)]
            patch = sources[i][geom['window']]
            state = r32.solve_eye(patch, geom)
            base0, support, _ = predict(geom, fit['atlas'], fit['aw'],
                                        f['du'], f['dv'], f['scale'], np.zeros((3, 3)))
            anomaly, _, _ = r32.detect_anomaly(patch, state['rho'], state['theta'],
                                               state['inside'])
            foot = (anomaly if anomaly is not None
                    else np.zeros(state['inside'].shape, bool))
            near_foot = cv2.dilate(foot.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            ring = state['inside'] & support & ~near_foot & (state['rho'] < 0.85)
            k0 = local_gain(patch, base0, ring)
            prepass[side][i] = dict(base0=base0, support=support, state=state, foot=foot,
                                    near_foot=near_foot, ring=ring, k0=k0)
            if k0 is None or args.lid_shadow <= 0:
                continue
            depth = lid_depth(lids.get((i, side)), geom)
            if depth is None:
                continue
            # The evidence: HER iris over the canonical one, on pixels that are neither
            # the specular, nor its coding ring, nor a lash.
            #
            # `near_foot` takes out both lobes of the artefact, which matters more here
            # than anywhere else: the softbox sits high on the iris on the frames where
            # the lid is low, which is precisely where the small depths are sampled, so
            # leaving it in would fit the highlight and call it a shadow.
            #
            # The lash guard is on LEVEL and it is deliberately loose -- a lash is far
            # darker than any shadowed iris -- because the per-bin median over thousands
            # of samples is what actually rejects contamination. A tight guard here would
            # instead select the brightest pixels at each depth and bias the shape.
            lum_s = np.clip(linear_to_srgb(np.clip(patch, 0, 1)), 0, 1).max(-1)
            lum_b = np.clip(linear_to_srgb(np.clip(base0 * k0[None, None, :], 0, 1)),
                            0, 1).max(-1)
            clean = (support & ~near_foot & (state['rho'] < 0.95)
                     & (lum_s > 0.12) & (lum_b > 0.05) & (depth > 0.0))
            if lids.get((i, side)) is not None:
                lid_cfg = lids[(i, side)]
                poly = np.asarray(landmarks['points'][i])[topology['eyelid_' + side], :2]
                clean &= r32.parabola_opening(
                    lid_cfg.get('upper'), lid_cfg.get('lower'), lid_cfg['centre'],
                    geom['window'], (float(poly[:, 0].min()), float(poly[:, 0].max())),
                    inset=args.shadow_inset) > 0.5
            if clean.sum() < 60:
                continue
            # Normalise by THIS FRAME's own level before pooling. What is left is only
            # the part the scalar gain cannot represent, which is the thing being fitted;
            # without it the pool would be dominated by exposure differences between
            # frames and the shape would come out of the wrong signal.
            ratio = lum_s[clean] / np.maximum(lum_b[clean], 1e-4)
            deep = clean & (depth > args.shadow_reference)
            if deep.sum() < 40:
                continue
            level = float(np.median(lum_s[deep] / np.maximum(lum_b[deep], 1e-4)))
            if not (0.2 < level < 5.0):
                continue
            shadow_samples[side].append((depth[clean].astype(np.float32),
                                         (ratio / level).astype(np.float32)))
    shadow = {}
    for side in r32.SIDES:
        shadow[side] = (fit_lid_shadow(shadow_samples[side])
                        if args.lid_shadow > 0 else None)
        print('R36_LID_SHADOW', side,
              None if shadow[side] is None else
              {k: v for k, v in shadow[side].items() if k != 'observed'}, flush=True)
        if shadow[side]:
            print('R36_LID_SHADOW_BINS', side, shadow[side]['observed'], flush=True)

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
            pre = prepass[side][i]
            state = pre['state']
            if args.gain_model == 'local':
                # Un-gained atlas first, then the shot's ONE lid-shadow shape, then ONE
                # robust ratio per channel from the observed iris ring around the
                # footprint. The order matters: the scalar is fitted AFTER the shape, so
                # it is levelling a prediction that already bends the right way and the
                # ring's own ratio is flat. Fitting it first and bending afterwards would
                # move the level of the whole fill every time the shape changed.
                base0, support = pre['base0'], pre['support']
                foot, near_foot, ring = pre['foot'], pre['near_foot'], pre['ring']
                shadow_field = lid_shadow_field(
                    shadow[side], lid_depth(lids.get((i, side)), geom), args.lid_shadow)
                if shadow_field is not None:
                    base0 = np.maximum(base0 * shadow_field[..., None], 0.0)
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
                shadow_field = None
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
                                   visible=geom['visible'], state=state,
                                   lid_signed=signed_lid_distance(
                                       landmarks['points'][i], topology, side,
                                       geom['window']),
                                   signed=signed, under=under, geom=geom, frame_scale=s,
                                   footprint=foot, gain_used=gain_used,
                                   lid_shadow_range=(
                                       None if shadow_field is None else
                                       [round(float(shadow_field.min()), 4),
                                        round(float(shadow_field.max()), 4)]),
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

    # --- pass A2: the lash ceiling for every frame, then ONE temporal median ----------
    # **A per-frame boundary moves, and the artist saw it move.** His words on the first
    # version: *"I do see some temporal instability in the colors in the region that
    # changes from frame to frame"* and *"the retina/iris/pupil jump into position a few
    # times as it corrects the mask offset suddenly"*. Measured, the painted disc's own
    # centre stepped up to 0.35 iris radii between adjacent frames and its top edge up to
    # 0.52 -- five or six pixels -- because the ceiling is read from the image on every
    # frame and its per-column argmin flips between adjacent rows.
    #
    # This is the lesson rounds 32-35 each had to learn again: anything refit per frame
    # moves. The ceiling cannot simply be frozen, because a lid MOVES -- that is what a
    # lid does -- so it is stabilised instead, in the eye's own coordinates so her head
    # motion is not mistaken for lid motion, by a THREE-FRAME MEDIAN.
    #
    # A median of three is the right filter and a mean is not: on smooth motion the
    # median of (previous, current, next) IS the current value, so it adds no lag on a
    # blink, while a single frame whose argmin flipped a row is outvoted. That is the same
    # rule round 33 learned about its temporal vote and round 35 about smoothing the
    # limbus: filter the spikes, never the motion.
    ceilings = {s: {} for s in r32.SIDES}
    if args.lash_guard:
        grid_u = np.linspace(-1.25, 1.25, 101)
        curves, geoms_by = {s: {} for s in r32.SIDES}, {s: {} for s in r32.SIDES}
        for side in r32.SIDES:
            for i, d in detail[side].items():
                geom = d['geom']
                c = lash_ceiling(sources[i][geom['window']], geom, d['state'],
                                 footprint=d.get('footprint'))
                if c is None:
                    continue
                ys, xs = geom['window']
                u = ((np.arange(xs.start, xs.stop) - geom['limbus'][0])
                     / geom['ax']).astype(np.float64)
                v = ((c + ys.start - geom['limbus'][1]) / geom['ay']).astype(np.float64)
                ok = np.isfinite(v)
                if ok.sum() < 8:
                    continue
                curve = np.interp(grid_u, u[ok], v[ok], left=np.nan, right=np.nan)
                curve[(grid_u < u[ok].min()) | (grid_u > u[ok].max())] = np.nan
                curves[side][i] = curve
                geoms_by[side][i] = geom
            for i, curve in curves[side].items():
                stack = [curves[side][j] for j in (i - 1, i, i + 1) if j in curves[side]]
                med = (np.nanmedian(np.vstack(stack), axis=0) if len(stack) > 1
                       else curve)
                med = np.where(np.isfinite(med), med, curve)
                geom = geoms_by[side][i]
                ys, xs = geom['window']
                u = ((np.arange(xs.start, xs.stop) - geom['limbus'][0])
                     / geom['ax']).astype(np.float64)
                good = np.isfinite(med)
                if good.sum() < 4:
                    continue
                back = np.interp(u, grid_u[good], med[good])
                ceilings[side][i] = (back * geom['ay'] + geom['limbus'][1]
                                     - ys.start).astype(np.float32)
            steps = []
            ks = sorted(curves[side])
            for x, y in zip(ks, ks[1:]):
                if y == x + 1:
                    dd = np.abs(curves[side][y] - curves[side][x])
                    if np.isfinite(dd).any():
                        steps.append(float(np.nanmax(dd)))
            print('R36_CEILING', side, 'frames', len(curves[side]),
                  'raw frame-to-frame max (iris radii) p95',
                  None if not steps else round(float(np.percentile(steps, 95)), 4),
                  'max', None if not steps else round(max(steps), 4), flush=True)

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
            ncc_floor = (args.min_ncc_frozen if args.freeze_registration
                         else args.min_ncc)
            # **A frame that failed to register is not vetoed once the registration is
            # FROZEN, and this is the artist's "glimpses of the softbox in the last
            # seconds".** `atl.build` returns None from `register()` when it finds no
            # candidate with enough overlap and then leaves `ncc` at its initialiser,
            # which is 0.0 -- not a measured correlation, a placeholder. Source 843, 846
            # and 848 left all carry exactly 0.0 while their neighbours sit at 0.985,
            # 0.988 and 0.975, so the eye was declined outright, the whole softbox stayed
            # in the photograph, and it reads as a glint appearing for a frame.
            #
            # The veto made sense when the offset was per frame: an unregistered frame
            # then had no offset worth using. Frozen, every frame is placed by the SAME
            # median offset, so an unregistered frame is placed exactly as well as its
            # neighbours and there is nothing to distrust. What still guards these frames
            # is everything that reads the image: the limbus and eyelid fits abstain, the
            # prediction must be darker than the source, the support mask must reach, and
            # the lash ceiling bounds the paint.
            registered_veto = (not f['registered']) and not args.freeze_registration
            if registered_veto or (f['registered'] and f['ncc'] < ncc_floor):
                record.update(repaired=False, reason='registration_below_threshold',
                              registration_frozen=bool(args.freeze_registration))
                eyes.append(record)
                continue
            ceil_map = ceilings[side].get(i)
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
                # `inside` stops at RHO_MAX = 0.88, which is the radius the round-32
                # PROFILE model needed and has nothing to do with how far the iris goes.
                # Multiplying the fill by it silently capped --fill-iris 0.95 at 0.88 and
                # left an unfilled ring 1.5 px wide all the way round -- and the softbox
                # lives in that ring on the frames where it sits high on the iris. The
                # fill is bounded by its own radius and by the lids, which is what it
                # meant to say.
                w *= (d['visible'] & d['support']).astype(np.float32)
                sup = np.clip(cv2.distanceTransform(d['support'].astype(np.uint8),
                                                    cv2.DIST_L2, 3) / 2.0, 0.0, 1.0)
                w = (w * sup).astype(np.float32)
                # **The iris fill is bounded by the lids; the highlight REMOVAL is not.**
                # The mediapipe eyelid contour is the next bad anchor after the iris
                # landmark: at source 770 it crosses the middle of the iris, three to
                # five pixels below the actual lash line, with open eye and the softbox
                # itself sitting above it. Bounding the fill by it left the highlight
                # standing on exactly the frames the artist flagged.
                #
                # Widening the lid mask by guesswork would paint canonical iris onto her
                # lashes. Physics gives a bound that needs no lid fit at all: **a lash is
                # DARKER than her own pooled iris and a specular highlight is brighter**,
                # so the removal may extend over the whole limbus disc while the fill
                # stays inside the lids, and the two share the same canonical material.
                # The rim collar goes with it, as round 34 established -- the black beside
                # the highlight is that highlight's own coding ringing.
                # How far past the polygon the removal may reach is a MEASURED shot
                # constant, not a dial: the corneal specular's outermost pixel sits a
                # median 3.2 px (left) / 4.4 px (right) INSIDE the polygon, and where the
                # polygon errs it is the specular that ends up outside. Four pixels covers
                # that error and stops short of the lash.
                # **The removal follows the specular BLOB, not a lid.** The fill is
                # bounded by the fitted eyelid, which is right: painting iris past a lid
                # can only damage a lash. Removal is a different question. Where the
                # fitted lid crosses the softbox -- and it does, on 30 eyes, by one to
                # three pixels -- bounding removal by it leaves a speck of the capture
                # room standing at half a stop above her iris, which is the artist's
                # "glimpses of the softbox suddenly appearing for a few frames".
                #
                # So removal may leave the opening, but only along the source's own
                # specular, and only on the component of it that reaches inside the
                # opening. A lash is not bright, so this cannot repaint one; round 35's
                # first attempt let removal have the whole disc and did exactly that.
                # `patch` is not bound yet in this loop -- it is set below, and the name
                # is still holding a different eye from the previous pass. Read the
                # window explicitly.
                here = sources[i][geom_['window']]
                lum = np.clip(linear_to_srgb(np.clip(here, 0, 1)), 0, 1).max(-1)
                ring_out = (rho > 1.15) & (rho < 1.55)
                sclera = (float(np.percentile(lum[ring_out], 90))
                          if ring_out.sum() > 40 else 0.5)
                hot = (lum > max(0.45, sclera + 0.06)) & (rho < 1.05)
                blob = np.zeros(hot.shape, bool)
                if hot.any():
                    n, labels = cv2.connectedComponents(hot.astype(np.uint8))
                    inside_opening = d['visible'] & d['support']
                    keep_ids = set(np.unique(labels[hot & inside_opening])) - {0}
                    if keep_ids:
                        blob = np.isin(labels, list(keep_ids))
                        blob = cv2.dilate(blob.astype(np.uint8),
                                          cv2.getStructuringElement(
                                              cv2.MORPH_ELLIPSE, (5, 5))) > 0
                        blob &= (rho < 1.05)
                wide = ((rho < args.fill_iris) & d['support']
                        & (d['lid_signed'] > -args.lid_extend)) | (blob & d['support'])
                wide &= d['support']
                ws, wi = soft_weight(d['signed'], d['under'], wide, d['support'],
                                     sigma[side], args.k_lo, args.k_hi, args.reach,
                                     rim_reach=args.rim_reach, rim=not args.keep_rim,
                                     rim_region=d['visible'])
                above_lid = 0
                if ws is not None:
                    above_lid = int(((ws > 0) & ~(d['visible'] & d['support'])).sum())
                    w = np.maximum(w, ws)
                w = np.rint(np.clip(w, 0, 1) * 255.0).astype(np.float32) / 255.0
                info = dict(peak_excess=float(np.where(wide, d['signed'], -np.inf).max()),
                            seed=(wi or {}).get('seed', [0, 0]),
                            highlight_pixels_above_lid=above_lid)
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
            # **The lash ceiling has the last word on where paint may land.** The
            # artist's rule: *"the softbox reflection will never be present in the
            # eyelashes or along the eyelid itself ... we should prevent us from drawing
            # on her eyelid and lashes"*, and *"we do not draw our own pixels ON or above
            # that eyelid arc"*.
            #
            # It is applied HERE, after the fill, after the removal's excursion past the
            # fitted lid and after the union with the tracked footprint, because each of
            # those can put weight on a lash and the guard has to outrank all three.
            # Round 35 painted through her lashes wherever the fitted lid abstained
            # (source 723 left) or landed on the lid crease seven pixels high (source 727
            # left), and both of those frames are in the set the artist flagged. See
            # lash_ceiling().
            #
            # **The footprint is exempt from the guard, PER PIXEL.** The artist's
            # rejection of the first version was a white "tetris block" in the corner of
            # her eye, and it is the guard eating the specular: at source 721 left the
            # lash minimum is at patch row 14 and **the highlight starts at row 15-17**,
            # so the +2 px guard offset lands inside it and shears off its upper-left
            # corner at 0.45-0.62 display. R35 removed those pixels and R36 left them.
            #
            # **Exempting the footprint from the guard was tried and it is NOT safe.**
            # The argument for it -- the footprint is where the plate is brighter than the
            # canonical iris, and a lash is darker -- is false once the lid shadow is
            # modelled: the canonical iris near the lash is then DARKER than a lash, so
            # the detector marks lash pixels as excess. Measured, it changed 1333 pixels
            # above the lash line and **70 % of them had a source luma below 0.35, which
            # is lash**. The offset below is the fix instead.
            lash_px = 0
            if w is not None and ceil_map is not None:
                rows_here = np.arange(w.shape[0], dtype=np.float32)[:, None]
                allow = np.clip(rows_here - ceil_map[None, :], 0.0, 1.0)
                # **A lash is never BRIGHT, and that is the one discriminator on this
                # plate that cannot fail.** The photometric lash line can and does land
                # inside the highlight: at source 740 left the lash and the lid-shadowed
                # upper iris form ONE continuous dark run from row 11 to row 16, so the
                # walk's "first dark run below skin" ends mid-iris and the ceiling came
                # out at row 19 with the specular at rows 18-22 -- it blocked the
                # brightest part of the reflection and left the white blob the artist
                # rejected.
                #
                # So the guard is not applied to pixels the SOURCE shows as bright. Her
                # lashes run 0.13-0.35 display and her lid-shadowed iris 0.15-0.30; a
                # corneal specular is 0.45-0.90. Exempting above `lash_bright` therefore
                # cannot repaint a lash by construction, and it is a statement about her
                # photograph rather than about any tracker.
                # The exemption carries the highlight's OWN COAT OF BLACK with it. Round
                # 34 established that the undershoot beside the specular is that
                # specular's coding ringing and has to go WITH it -- delete the bright
                # lobe and keep the dark one and what is left is a hard black crescent,
                # which is what the artist rejected in rounds 32, 33 and 34. Exempting
                # only the bright lobe reproduced it exactly: source 736-745 left came
                # back with a dark ring round the top of the pupil.
                #
                # The ring is admitted by ADJACENCY and by LEGAL BLACK, not by level
                # alone: only pixels within `ring_px` of the exempted core and below 0.22
                # display, which is where the undershoot clips (green and blue at exactly
                # 0, red 8-26). Her lashes sit at 0.13-0.35, so a few lash pixels within
                # two of the specular can be caught -- that is a handful of pixels where
                # the reflection touches the lash, against a black crescent on every
                # treated frame.
                here_src = np.clip(linear_to_srgb(np.clip(sources[i][geom['window']],
                                                          0, 1)), 0, 1).max(-1)
                bright = np.clip((here_src - args.lash_bright) / 0.06, 0.0, 1.0)
                core = (bright > 0.5).astype(np.uint8)
                if core.any():
                    k = int(2 * np.ceil(args.lash_ring) + 1)
                    near_core = cv2.dilate(core, cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (k, k))) > 0
                    bright = np.maximum(bright, (near_core & (here_src < 0.22)
                                                 ).astype(np.float32))
                allow = np.maximum(allow, bright.astype(np.float32))
                lash_px = int(((w > 0) & (allow < 1.0)).sum())
                w = np.rint(np.clip(w * allow, 0, 1) * 255.0).astype(np.float32) / 255.0
            if info is not None:
                info['lash_guard_pixels'] = lash_px
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
                highlight_pixels_above_lid=info.get('highlight_pixels_above_lid'),
                lash_guard_pixels=info.get('lash_guard_pixels'),
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
        lid_extend_px=args.lid_extend,
        lid_inset_px=args.lid_inset,
        atlas_lid_inset_px=args.atlas_lid_inset,
        lash_guard=bool(args.lash_guard), lash_bright=args.lash_bright,
        lash_guard_rule='no paint on or above the lash: per-column ceiling read from the '
                        'SOURCE by walking down from lid skin to the first dark run that '
                        'reaches legal-lash level, masked against the specular, its '
                        'legal-black undershoot and the pupil. It can only ever REMOVE '
                        'paint, so it cannot introduce a wrong-anchor artefact.',
        lid_shadow_strength=args.lid_shadow,
        lid_shadow_inset_px=args.shadow_inset,
        lid_shadow_reference_px=args.shadow_reference,
        lid_shadow_model='1 + A * exp(-depth_below_fitted_upper_lid / tau); ONE per eye '
                         'for the shot, fitted on pooled observed-over-canonical ratios '
                         'with each frame normalised by its own level beyond the '
                         'reference depth, so only the part the per-frame scalar cannot '
                         'represent is fitted',
        lid_shadow=shadow,
        atlas_grid=atlas_grid_config, sample_reach=args.sample_reach,
        atlas_cap_percentile=args.atlas_cap,
        atlas_cap_level={k: (v.get('atlas_cap') if v else None)
                         for k, v in fitted.items()},
        ncc_floor=args.min_ncc_frozen if args.freeze_registration else args.min_ncc,
        eyelid_track=args.eyelid_track or None,
        eyelid_track_sha256=(r32.sha(shot.relight / args.eyelid_track)
                             if args.eyelid_track else None),
        eyelid_source='image_fit' if lids else 'mediapipe_polygon',
        eyelid_fitted_eyes=len(lids),
        freeze_registration=bool(args.freeze_registration),
        frozen_registration={k: (v['frozen_registration'] if v else None)
                             for k, v in fitted.items()},
        limbus_track=args.limbus_track or None,
        limbus_track_sha256=(r32.sha(shot.relight / args.limbus_track)
                             if args.limbus_track else None),
        limbus_source='image_fit' if track is not None else 'mediapipe_landmark',
        code_sha256=r32.sha(__file__), atlas_code_sha256=r32.sha(atl.__file__),
        limbus_code_sha256=r32.sha(Path(__file__).parent / 'limbus_fit.py'),
        track_sha256=r32.sha(shot.relight / 'face_landmarks.npz'))
    r32.write_json(root / 'removal.json', record)
    r32.write_json(shot.log(args.name + '.json'), record)
    print('R34_REMOVE_DONE', len(rows), 'frames,', len(repaired), 'repaired', flush=True)


if __name__ == '__main__':
    main()
