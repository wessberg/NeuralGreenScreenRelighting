"""EXPERIMENTAL, REJECTED: split the plate into estimated diffuse and specular layers.

Round-5 playback exposed damaged teeth, sclera and unstable hair patches. The model below
does not establish that its subtraction is specular, or that photographed detail survives.
The composite defaults to keeping the original plate and disabling replacement catchlights.
Historical rationale below is not a validation claim; see the round6 rejection addendum.

    dependencies/CorridorKey/.venv/Scripts/python.exe scripts/separate_specular.py --shot <dir>

This exists because of a ceiling the artist named on sight, watching frame 21 of the reporter
test: *"I can kind of see the shine in her hair from the original green screen room where they
recorded her."* He was right, and it is architectural rather than a setting.

The delivery composite is ``out = plate x R`` with a smooth ``R``. A smooth multiplier can
**recolour** a specular highlight but it cannot move, reshape or delete one. So every highlight
in the plate keeps reporting where the ORIGINAL room's lights were, no matter where the Blender
key is: the sheen band across the crown of the head, the glints on the lower lip, and -- the one
that gives it away fastest -- the corneal catchlight, which on this plate is a plainly
rectangular softbox reflection sitting at 11 o'clock on both irises.

``make_albedo.py`` already does this for the diffuse half: divide out ``E_source`` or the plate
is lit twice. Nothing did it for the specular half. This is that stage.

    plate = D + S            D = diffuse (obeys the ratio),  S = specular (does not)
    out   = D x R + S_new    S_new is synthesised from the ACTUAL scene lights

WHY A HIGHLIGHT CANNOT JUST BE MULTIPLIED
-----------------------------------------
Diffuse radiance is ``albedo x E(n)`` -- change the light and it scales. Specular radiance is
``rho(n, l, v)``, a lobe whose POSITION on the surface depends on the light direction. Relighting
moves it. ``plate x R`` cannot, because R has no way to know that the pixel it is scaling is a
mirror image of a lamp rather than a piece of skin. Removing S first is the only way the new
lighting gets to decide where the shine goes.

THE MODEL, AND WHY IT IS TRACTABLE HERE
---------------------------------------
The dichromatic reflection model: for a dielectric -- skin, hair, lips, cornea, cotton, wood --

    I = d * L + s * G

where ``G`` is the ILLUMINANT colour and ``L`` the surface's own diffuse chromaticity. Metals
break this; nothing on a performer is a metal except jewellery, which is masked by the neutral
guard below. The consequence that makes separation possible is that the specular term is the
same colour for every material in the frame, so it is the achromatic direction once the image is
divided by the illuminant.

Two things must be measured from the plate rather than assumed, and both are: the illuminant
colour ``G`` (below) and, per pixel, the local material's own diffuse chromaticity (further
below). Neither is a constant anyone types in.

**NO ON-SET REFERENCE.** House rule, and this stage keeps it: the illuminant is measured from the
performer's own corneal glints -- the eye is the plate's own chrome ball -- and cross-checked
against the brightest near-neutral pixels in frame. On the reporter plate the two agree to 0.3%:

    corneal glints      R:G:B = 1.000 : 0.955 : 0.922   (88 samples over the sequence)
    neutral wardrobe    R:G:B = 1.000 : 0.957 : 0.927   (903 px)

They share no assumption and no pixels, so the agreement is a real cross-validation rather than
one estimator confirming itself. Round 4's despill bug is the standing warning about that: three
measurements agreed the plate was clean and all three were computing the buggy gate.

HOW MUCH IS SPECULAR, WITHOUT A SEGMENTATION MODEL
--------------------------------------------------
Every pixel is fitted against two directions in RGB: the LOCAL MATERIAL colour and the
ILLUMINANT colour.

    I = a * b_hat + s * G_hat,        a >= 0, s >= 0

``b_hat`` is the unit chromaticity of a local morphological opening of the plate -- the colour of
the nearest genuinely diffuse pixels of whatever material this is, so it needs no skin mask, no
hair mask and no assumption about skin tone, hair colour or wardrobe. The two-by-two normal
equations are closed form, so this is one vectorised pass. ``a`` absorbs "this patch of the
material got more light"; ``s`` absorbs "there is a mirror image of a lamp here". A rise in only
the red channel -- blood, lipstick, a blush -- has nowhere to go but ``a`` and claims no specular
at all.

The opening is alpha-aware. A plain opening takes a local minimum, so at the silhouette it reads
the black outside the matte and reports a near-zero base, which throws b_hat off and paints an
orange fringe down every edge. Outside-matte pixels are pushed up before the erosion instead.

THE FLOOR, AND WHY IT IS NOT A FUDGE FACTOR
-------------------------------------------
Skin defeats the naive version of this, and the reason is physical rather than numerical: **skin
gets MORE saturated as it gets darker**, because in shadow the surviving light is the part that
went deep enough to be reddened by blood. Measured on the reporter plate over 58 819 cheek
pixels of one material, ``min/sum`` runs 0.060 at p2 to 0.216 at p95 -- a factor of 3.5 that has
nothing to do with specular. A dichromatic separator reads that drift as specular and takes it,
which desaturates the whole face to grey-green. That was visible before it was measured.

So the estimate carries a floor proportional to the LOCAL DIFFUSE LEVEL:

    s <- max(0, s - tau * D_local)

Below ``tau`` of local diffuse, a near-illuminant excess is not distinguishable from skin's own
subsurface behaviour, so this stage does not claim it. Above it -- a catchlight at 8-10x the
local level, a lip gloss band, a hair sheen band, a nose-tip highlight -- the evidence is
unambiguous. On the reporter plate the specular layer runs 8% of subject energy ungated and 4.6%
at the default floor, and the 3.4 points in between are exactly the ambiguous ground.

**This is a real ceiling and it is stated rather than hidden.** A single unpolarised image cannot
fully separate diffuse from specular on skin; cross-polarised capture can, and the artist is not
going to shoot cross-polarised. What this stage removes is the specular that is strong enough to
tell the viewer where the original room's lights were -- which is the part he can see and name.
The residual low-level sheen stays in the diffuse layer and is relit as if it were diffuse.

WHAT THIS STAGE DOES NOT DO
---------------------------
It does not invent anything. Every output pixel of D is an input pixel minus a scalar times a
measured colour, so constraint 15 holds by construction -- there is no generative model in the
path and identity cannot drift. It also cannot recover texture under a CLIPPED highlight: where
a specular blew past the sensor's range the diffuse underneath was never recorded, and
subtraction leaves a flat patch. The stage measures and reports clipping rather than filling it.
On the reporter plate the corneal glints peak at 0.59-0.63 linear against a 1.0 ceiling, so
nothing clips and the iris texture under the catchlight is genuinely recoverable.

**On the artist's own footage this stage MUST run after the V-Log/V-Gamut decode (constraint 18,
still red).** The dichromatic model is linear in radiance. Fed log-encoded pixels every quantity
above -- min/sum, the hue bins, the subtraction itself -- is computed in the wrong space and
will still produce a plausible-looking picture.

Outputs ``key/specular/*.exr`` only. The diffuse layer is ``fg - specular``, derived on demand by
``load_layers()`` below, because storing both would double the disk for a sequence whose two
halves must agree exactly anyway.
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import sys, json, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np, cv2
from shot_paths import add_shot_argument, from_args

LUMA = np.array([.2126, .7152, .0722], np.float32)
# Human corneal radius of curvature. An anatomical constant, not a per-shot tuning value; the
# per-shot scale comes from the measured iris radius in pixels.
CORNEA_RADIUS_MM = 7.8
IRIS_RADIUS_MM = 5.85


def specular_dir(shot):
    return shot.key / 'specular'


def load_layers(shot, i):
    """(diffuse, specular) in linear RGB, or (plate, zeros) if the stage has not been run.

    Consumers call this instead of reading ``key/fg`` directly, so a shot without a separation
    keeps working unchanged and one with a separation gets the split for free.
    """
    fg = cv2.imread(str(shot.fg(i)), -1)[..., ::-1].astype(np.float32)
    p = specular_dir(shot) / ('%06d.exr' % i)
    if not p.exists():
        return fg, np.zeros_like(fg)
    spec = cv2.imread(str(p), -1)[..., ::-1].astype(np.float32)
    if spec.shape != fg.shape:
        spec = cv2.resize(spec, (fg.shape[1], fg.shape[0]), interpolation=cv2.INTER_LINEAR)
    spec = np.clip(spec, 0.0, None)
    return np.clip(fg - spec, 0.0, None), spec


# --------------------------------------------------------------------------------------------
# Illuminant


def glint_samples(shot, frames, lmk):
    """Specular excess inside each iris disc: the plate photographing its own light source.

    A corneal reflection is a first-surface mirror image of the source, so its colour IS the
    illuminant's, contaminated only by the iris diffuse underneath -- which a morphological
    opening removes, because the opening's structuring element is larger than the glint and so
    reports the local diffuse level around it.
    """
    pts, found, ap = lmk['points'], lmk['found'], lmk['eyelid_aperture']
    out, clipped, peaks = [], 0, []
    for i in frames:
        if not found[i]:
            continue
        # Both gates: no detection is not the same as a shut eye, and eyeBlink saturates late.
        if not np.all(np.nan_to_num(ap[i]) > 0.9):
            continue
        fg = cv2.imread(str(shot.fg(i)), -1)[..., ::-1].astype(np.float32)
        L = fg @ LUMA
        for c0 in (468, 473):
            c = pts[i, c0, :2]
            rad = float(np.linalg.norm(pts[i, c0 + 1:c0 + 5, :2] - c, axis=1).mean())
            if not np.isfinite(rad) or rad < 2:
                continue
            x0, y0 = int(c[0] - rad * 1.6), int(c[1] - rad * 1.6)
            s = int(rad * 3.2)
            if x0 < 0 or y0 < 0 or y0 + s > fg.shape[0] or x0 + s > fg.shape[1]:
                continue
            pat, pl = fg[y0:y0 + s, x0:x0 + s], L[y0:y0 + s, x0:x0 + s]
            yy, xx = np.mgrid[0:s, 0:s]
            disc = ((xx + x0 - c[0]) ** 2 + (yy + y0 - c[1]) ** 2) <= (rad * 1.1) ** 2
            if disc.sum() < 30:
                continue
            k = max(int(round(rad * 0.7)) | 1, 5)
            base = cv2.morphologyEx(pat, cv2.MORPH_OPEN,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
            exc = pat - base
            m = disc & (pl > np.percentile(pl[disc], 97)) & ((exc @ LUMA) > 0.02)
            if m.sum() < 3:
                continue
            out.append(exc[m].sum(0))
            peaks.append(float(pat[m].max()))
            clipped += int((pat[m].max(-1) >= 0.995).any())
    return np.array(out, np.float32), peaks, clipped


def neutral_samples(shot, frames):
    """Brightest low-saturation pixels in the matte. An independent second opinion on G.

    Independent because it shares no pixels and no mechanism with the glints: this is a diffuse
    reflection off a near-white surface, that is a mirror reflection off a wet one.
    """
    acc = []
    for i in frames:
        fg = cv2.imread(str(shot.fg(i)), -1)[..., ::-1].astype(np.float32)
        a = cv2.imread(str(shot.alpha(i)), -1)
        a = a[..., 0] if a.ndim == 3 else a
        core = a > 0.99
        if core.sum() < 500:
            continue
        L = fg @ LUMA
        sat = 1.0 - fg.min(-1) / np.maximum(fg.max(-1), 1e-6)
        m = core & (sat < 0.13) & (L > np.percentile(L[core], 80))
        if m.sum() > 50:
            acc.append(fg[m].mean(0) * m.sum())
            acc[-1] = np.append(acc[-1], m.sum())
    if not acc:
        return None, 0
    A = np.array(acc)
    return A[:, :3].sum(0) / A[:, 3].sum(), int(A[:, 3].sum())


def estimate_illuminant(shot, frames, lmk, args):
    """G, with the estimator that produced it named in the result. Never silently defaulted."""
    res = {'candidates': {}}
    G = None
    if lmk is not None:
        gl, peaks, clipped = glint_samples(shot, frames, lmk)
        if len(gl) >= args.min_glints:
            g = gl.sum(0)
            g = g / max(float(g.max()), 1e-9)
            res['candidates']['corneal_glints'] = {
                'rgb': g.round(4).tolist(), 'samples': int(len(gl)),
                'peak_linear_mean': float(np.mean(peaks)) if peaks else None,
                'clipped_samples': int(clipped),
                'clipping_note': 'a clipped glint has no diffuse underneath to recover; '
                                 'subtraction leaves a flat patch there and this stage reports '
                                 'it rather than filling it'}
            G = g
        else:
            res['candidates']['corneal_glints'] = {
                'samples': int(len(gl)), 'rejected': 'fewer than --min-glints usable samples'}
    neu, npx = neutral_samples(shot, frames)
    if neu is not None:
        n = neu / max(float(neu.max()), 1e-9)
        res['candidates']['neutral_pixels'] = {'rgb': n.round(4).tolist(), 'pixels': npx}
        if G is None:
            G = n
    if G is None:
        G = np.ones(3, np.float32)
        res['source'] = 'FALLBACK to white -- no usable glints and no neutral pixels'
        res['warning'] = ('The illuminant was NOT measured on this shot. Every removed highlight '
                          'therefore carries a colour cast. Treat the separation as provisional.')
    else:
        res['source'] = ('corneal_glints' if 'rgb' in res['candidates'].get('corneal_glints', {})
                         else 'neutral_pixels')
    a = res['candidates'].get('corneal_glints', {}).get('rgb')
    b = res['candidates'].get('neutral_pixels', {}).get('rgb')
    if a and b:
        d = np.abs(np.array(a) - np.array(b)).max()
        res['cross_check'] = {
            'max_channel_disagreement': float(d),
            'verdict': 'AGREE' if d < 0.03 else 'DISAGREE -- do not trust either without looking',
            'why_it_matters': 'the two estimators share no pixels and no mechanism, so agreement '
                              'is real cross-validation rather than one metric confirming itself'}
    res['rgb'] = np.asarray(G, np.float32).round(5).tolist()
    return np.asarray(G, np.float32), res


# --------------------------------------------------------------------------------------------
# Separation


def local_base(fg, alpha, r):
    """Local diffuse reference: an ALPHA-AWARE morphological opening.

    A plain opening erodes first, and erosion takes the local minimum, so within r of the
    silhouette it reads the black outside the matte and reports a base of nearly zero. b_hat then
    comes from noise and the diffuse layer grows an orange fringe down every edge. Pushing the
    outside up to the frame maximum before the erosion makes the outside lose every min-test, so
    the base at the edge is decided by the subject's own pixels.
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))
    out = np.where((alpha > 0.5)[..., None], fg, float(fg.max()) + 1.0).astype(np.float32)
    out = cv2.dilate(cv2.erode(out, k), k)
    return cv2.GaussianBlur(np.minimum(out, fg.max()), (0, 0), max(r / 6.0, 0.8))


def separate(fg, alpha, G, r, args):
    """(specular RGB, diagnostics). See the module docstring for the model.

    Returns the specular layer only; the diffuse layer is fg - specular, so the two cannot drift
    apart and only one of them has to be stored.
    """
    Gh = (G / max(float(np.linalg.norm(G)), 1e-9)).astype(np.float32)
    base = local_base(fg, alpha, r)
    bh = base / np.maximum(np.linalg.norm(base, axis=-1, keepdims=True), 1e-8)
    # cos between the material direction and the illuminant direction. At 1 the two are parallel
    # and the system is singular -- a neutral material, where "white surface" and "grey surface
    # in a highlight" are the same measurement and no method can tell them apart.
    c = np.clip((bh * Gh).sum(-1), -1.0, 1.0)
    ib, ig = (fg * bh).sum(-1), (fg * Gh).sum(-1)
    det = np.maximum(1.0 - c * c, 1e-6)
    a = (ib - c * ig) / det
    s = (ig - c * ib) / det
    # a < 0 is unphysical (it would mean negative material reflectance); the least-squares answer
    # under the non-negativity constraint is then the pure-specular projection.
    s = np.clip(np.where(a < 0, ig, s), 0.0, None)
    unsep = c > args.cos_max
    s = np.where(unsep, 0.0, s)
    # The floor. D_local is the local diffuse LEVEL, from the same opening that gave its colour.
    D_local = np.maximum(cv2.GaussianBlur(base, (0, 0), max(r / 2.0, 1.0)) @ LUMA, 1e-5)
    s_raw = s.copy()
    s = np.maximum(s - args.floor_fraction * D_local, 0.0)
    # Never take more than a channel has: the diffuse residue must stay non-negative.
    s = np.minimum(s, (fg / np.maximum(Gh, 1e-6)).min(-1))
    s = s * np.clip(alpha, 0.0, 1.0)
    return np.clip(s[..., None] * Gh, 0.0, fg), s_raw, unsep, D_local


def opening_radius(shot, args):
    """Radius of the local-material window, in pixels, scaled to the FACE not to the frame.

    Everything downstream of this has to work on a 1080p bust, on a 6K anamorphic macro
    close-up where an iris is 200 px across, and on a wide shot where the head is small. A
    constant pixel radius is wrong in at least two of those. The interocular distance is the
    most stable facial scale available and face_track.py already measures it per frame.
    """
    if args.radius:
        return max(int(args.radius) | 1, 3), 'explicit --radius'
    p = shot.relight / 'face_track.json'
    if p.exists():
        ipd = (json.loads(p.read_text()).get('ipd_px_stats') or {}).get('median')
        if ipd:
            return max(int(round(args.radius_ipd * ipd)) | 1, 3), 'ipd %.1f px x %.3f' % (
                ipd, args.radius_ipd)
    # No face: fall back to the matte's own height, which at least scales with the subject.
    a = cv2.imread(str(shot.alpha(0)), -1)
    a = a[..., 0] if a.ndim == 3 else a
    ys = np.nonzero((a > 0.5).any(1))[0]
    h = float(ys.max() - ys.min() + 1) if len(ys) else a.shape[0]
    return max(int(round(0.02 * h)) | 1, 3), 'no face track; 2%% of matte height %.0f px' % h



def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_shot_argument(ap)
    ap.add_argument('--frames', type=int, default=0)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--sample-stride', type=int, default=5,
                    help='frame stride for the illuminant fit, which is pooled over the shot so '
                         'this only trades fit time against sample count')
    ap.add_argument('--radius', type=int, default=0,
                    help='local-material window in pixels; 0 derives it from the face scale, '
                         'which is what makes this work on a macro close-up and a wide shot alike')
    ap.add_argument('--radius-ipd', type=float, default=0.11,
                    help='window radius as a fraction of the interocular distance')
    ap.add_argument('--floor-fraction', type=float, default=0.25,
                    help='specular below this fraction of the LOCAL DIFFUSE LEVEL is not claimed. '
                         'Not a fudge: below it a near-illuminant excess is indistinguishable '
                         'from skin getting more saturated as it darkens, which is a real '
                         'subsurface effect and spans a factor of 3.5 across one cheek')
    ap.add_argument('--cos-max', type=float, default=0.9995,
                    help='above this the material colour and the illuminant colour are parallel '
                         'and the split is singular -- a white shirt, grey foam, bare metal. '
                         'Those pixels are declared UNSEPARABLE and reported, not zeroed quietly')
    ap.add_argument('--min-glints', type=int, default=8)
    ap.add_argument('--dry-run', action='store_true', help='fit and report, write nothing')
    args = ap.parse_args()
    shot = from_args(args)

    n = args.frames or shot.frame_count
    frames = list(range(args.start, args.start + n))
    fit_frames = frames[::max(1, args.sample_stride)]

    lp = shot.relight / 'face_landmarks.npz'
    lmk = np.load(lp) if lp.exists() else None
    if lmk is None:
        print('WARNING: no relight/face_landmarks.npz -- the illuminant falls back to the '
              'neutral-pixel estimator and loses its cross-check. Run scripts/face_track.py.',
              file=sys.stderr)

    G, illum = estimate_illuminant(shot, fit_frames, lmk, args)
    print(json.dumps(illum, indent=2), flush=True)
    r, r_why = opening_radius(shot, args)
    print('local-material window: %d px (%s)' % (r, r_why), flush=True)

    out = specular_dir(shot)
    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)
    Gh = G / max(float(np.linalg.norm(G)), 1e-9)
    stats = []
    for i in frames:
        fg = cv2.imread(str(shot.fg(i)), -1)[..., ::-1].astype(np.float32)
        a = cv2.imread(str(shot.alpha(i)), -1)
        a = a[..., 0] if a.ndim == 3 else a
        spec, s_raw, unsep, D_local = separate(fg, a, G, r, args)
        if not args.dry_run:
            # float32 in, HALF on the way out. Handing cv2.imwrite a float16 ARRAY does not
            # work: the EXR encoder rejects it with "Unsupported depth image for selected
            # encoder is fallbacked to CV_8U" and then asserts on CV_8U, which reads like a
            # codec problem rather than a dtype one. IMWRITE_EXR_TYPE_HALF does the conversion.
            cv2.imwrite(str(out / ('%06d.exr' % i)),
                        np.ascontiguousarray(spec[..., ::-1]).astype(np.float32),
                        [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF,
                         cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_ZIP])
        core = a > 0.99
        if core.any():
            sl, fl = spec[core] @ LUMA, fg[core] @ LUMA
            # s actually kept, recovered from the written layer, versus s the fit proposed. The
            # gap is the ambiguous ground between "clearly a highlight" and "skin being skin",
            # and reporting it is the honest form of the ceiling.
            s_kept = spec @ Gh / max(float(Gh @ Gh), 1e-9)
            gated = float(np.clip(s_raw - s_kept, 0, None)[core].sum()
                          / max(float(s_raw[core].sum()), 1e-9))
            stats.append({'frame': i,
                          'specular_fraction_of_energy': float(sl.sum() / max(fl.sum(), 1e-9)),
                          'pixels_over_5pct': float((sl > 0.05 * np.maximum(fl, 1e-6)).mean()),
                          'unseparable_fraction': float(unsep[core].mean()),
                          'discarded_by_floor_fraction': gated,
                          'max_specular_luma': float(sl.max()),
                          'diffuse_negative_pixels': int(((fg - spec)[core] < -1e-6).sum())})
        if i % 25 == 0 and stats:
            print(json.dumps(stats[-1]), flush=True)

    sf = np.array([s['specular_fraction_of_energy'] for s in stats])
    summary = {
        'shot': shot.name, 'frames': len(stats), 'output': str(out),
        'model': 'I = a*b_hat + s*G_hat, a,s >= 0. b_hat is the LOCAL material chromaticity from '
                 'an alpha-aware opening; G_hat is the illuminant, measured from the plate.',
        'illuminant': illum,
        'window': {'radius_px': r, 'derivation': r_why,
                   'why_face_scaled': 'a constant pixel radius is wrong on at least two of: a '
                                      '1080p bust, a 6K anamorphic macro close-up, a wide shot'},
        'floor': {'fraction_of_local_diffuse': args.floor_fraction,
                  'why': 'skin gets MORE saturated as it darkens (subsurface reddening), which '
                         'mimics the dichromatic specular signature. Measured over 58819 cheek '
                         'pixels of one material, min/sum spans 0.060 (p2) to 0.216 (p95). '
                         'Ungated, the separator reads that drift as specular and desaturates '
                         'the whole face to grey-green.',
                  'consequence': 'the low-level sheen below the floor stays in the diffuse layer '
                                 'and is relit as if it were diffuse. That is a real ceiling on '
                                 'constraint 5, not a tuning choice.'},
        'result': {'mean_specular_fraction_of_energy': float(sf.mean()) if len(sf) else 0.0,
                   'max_specular_fraction_of_energy': float(sf.max()) if len(sf) else 0.0,
                   'mean_unseparable_fraction': float(np.mean(
                       [s['unseparable_fraction'] for s in stats])) if stats else 0.0,
                   'mean_discarded_by_floor': float(np.mean(
                       [s['discarded_by_floor_fraction'] for s in stats])) if stats else 0.0},
        'identity': 'D = plate - s*G_hat. Every diffuse pixel is a plate pixel minus a scalar '
                    'times a measured colour; no generative model is in the path (constraint 15).',
        'limits': ['cannot recover texture under a CLIPPED highlight -- none was recorded there',
                   'near-neutral materials (a white shirt, grey foam, metal) are unseparable by '
                   'ANY dichromatic method; see mean_unseparable_fraction',
                   'the sub-floor sheen on skin is not separable from a single unpolarised image',
                   'on V-Log footage this MUST run after the decode (constraint 18)'],
        'per_frame': stats}
    if not args.dry_run:
        shot.log('specular_separation.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ('frames', 'window', 'result')}, indent=2))


if __name__ == '__main__':
    main()
