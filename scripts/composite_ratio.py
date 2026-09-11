"""Relight the real photograph by a rendered ratio. The plate's own pixels are never replaced.

    dependencies/CorridorKey/.venv/Scripts/python.exe scripts/composite_ratio.py --shot <dir>

    out = plate x R,     R = E_target(rendered) / E_source(assumed)

This is the delivery architecture, and it exists because of a specific failure the artist named
precisely: rendering the proxy and looking at it produces "a Vision Pro avatar", "a Sims
character", "a plastic Barbie doll". He was right, and the reason is structural rather than a
matter of quality settings.

**Rendering a reconstruction can only ever show the reconstruction.** The proxy has no pores, no
individual hair strands, no fabric weave, no corneal bulge, no eyelash. His own frame-by-frame
verdicts map exactly onto how much of her the light revealed: the backlit and heavily shadowed
frames read as acceptable, the broadly and evenly lit ones read as a video-game character,
because a broad light exposes precisely the detail a smooth shell does not have.

The ratio composite never renders her. It renders the *irradiance field* on a neutral proxy under
the new lighting, divides it by the plate's own assumed lighting, and multiplies the photograph
by the result. Every pore, hair, eyelash and specular in the output is a photographed pixel. What
the renderer contributes is only the CHANGE in light -- which is smooth almost everywhere, and
sharp exactly where it should be, at a shadow edge. Those are the two things geometry is good at
and a per-pixel method cannot do at all.

The ratio is bounded rather than free:

* the **upper** end is clamped in log space, because a division can spike where the proxy and the
  matte disagree at the silhouette;
* the **lower** end is deliberately allowed to go dark, because a shadow going dark is the entire
  point. The old gain map's 1.5-stop tanh capped the whole map at 2.83:1 total range, which is
  why it could never produce a terminator.

**Specular is added, never multiplied.** Leaving Principled's specular on in the ratio pass was a
real bug: `Combined = diffuse + subsurface + glossy`, and a glossy highlight divided by a
diffuse-only `E_source` is not a lighting ratio. Measured on the tour, E_target ran 0.073 at p90
and 19.17 at p99 -- a 455x spike centred on the top of her head -- which the upper clamp then
flattened into a blown patch with a hard edge and no falloff. Specular now comes from the
additive G term instead.

G is computed here rather than rendered, for three reasons: it needs no second render pass; it
uses the **neural normals**, which carry more facial structure than the smooth proxy shell, so
highlights land on the nose tip, cheekbone and lower lip where they belong; and the dominant
light direction is estimated from the rendered irradiance field itself, so it works for an
environment as well as for a single bulb. It is gated by the diffuse ratio, so a point in shadow
gets no highlight -- specular without occlusion is what makes CG skin look like wet plastic.

What this inherits from the proxy, honestly: the *shape* of the shadows, and therefore the proxy's
smoothness at their edges. A nose shadow will land in the right place with roughly the right
outline, and hair will shadow as a mass rather than as strands. It does not inherit the proxy's
appearance, which is the entire difference.

REJECTED ROUND 5 EXPERIMENTS (historical rationale, not validated behaviour)
------------------------------------------------------------------------
The following experiments damaged teeth, sclera, hair and eyelids. Defaults keep the plate
and disable catchlight replacement. The apparent lip improvement below did not validate the
separation: see rounds/round6.md's rejection addendum and rounds/round7.md.

ROUND 5, PART 1: THE RATIO MULTIPLIES THE DIFFUSE LAYER, NOT THE PLATE
----------------------------------------------------------------------
``plate = D + S``. The ratio is a lighting scale and only D obeys it -- a smooth R can recolour a
highlight but cannot move or delete one, so every specular in the plate kept reporting where the
green-screen room's lamps were. ``separate_specular.py`` splits the plate; this multiplies D and
adds a NEW specular from the actual scene lights:

    out = D x R + S_new

Where no separation exists the loader returns (plate, zeros) and the behaviour is exactly
round 4's, so this cannot silently change a shot that has not been separated.

ROUND 5, PART 2: E_source AND E_target MUST SHARE A BANDWIDTH
--------------------------------------------------------------
``R = E_target / E_source`` divided a SMOOTH numerator by a SHARP one. E_target is rendered on the
proxy, whose mouth is a smooth swelling; E_source is evaluated on the neural normals, which
resolve the lip roll, the philtrum, the nostril rims and the teeth. Measured over the face as the
relative RMS of a log high-pass:

    scale     E_target   E_source   E_source has this much more
      2 px     0.0098     0.0367      3.75x
      4 px     0.0306     0.0832      2.72x
      8 px     0.0837     0.1512      1.81x
     16 px     0.1860     0.2280      1.23x
     32 px     0.3318     0.3074      0.93x   <- they cross here

Below the crossover R carries 1/E_source, so it BRIGHTENS whatever the plate's own key had
shaded and never re-darkens it: the plate's fine shading is removed and nothing puts it back.
On frame 21 that lifted the lips by 21% relative to the skin around them, compressing the
photographed lip-to-skin contrast from 0.695 to 0.883 -- which is exactly the "lips are somehow
lit up more than the skin" the artist reported, and the same mechanism is the strongest candidate
for "the skin has no pores".

The rule this suggests -- you may only divide out lighting at the resolution at which you can put
it back -- is implemented as ``--source-detail-sigma`` and is **OFF BY DEFAULT, because it was
tested and it did not fix the symptom.** Measured on frame 21 against the photographed
lip-to-skin contrast of 0.8189, with the panel registered to correlation 1.0000:

    band-limit off, plate specular kept       0.8764   +7.0%
    band-limit 8 px, plate specular kept      0.9133   +11.5%   <- WORSE
    band-limit off, plate specular REMOVED    0.8259   +0.9%    <- REJECTED; damages anatomy

So the bandwidth gap above is real and measured, but it is not what lifted the lips. What lifted
them was the plate's own lip gloss: a photographed specular band from the green-screen room's
softbox, which ``plate x R`` could only recolour, so under a warm key it became a bright orange
wet band sitting on top of lips that had otherwise been relit correctly. Removing it lands the
lip-to-skin contrast within 0.9% of the photograph.

The flag stays, off, with its measurement recorded, for the same reason ``--extend-edges`` did:
an unproven idea left on by default is one that ships silently. Whether the bandwidth gap
explains the artist's separate "the skin has no pores" note is still open -- that needs the
proxy's own normals, which nothing currently writes.

ROUND 5, PART 3: THE CATCHLIGHT IS RENDERED, NOT PAINTED
---------------------------------------------------------
The plate's own corneal reflection leaves with the specular layer, and a new one is added from
``stress_proxy.py --catchlight``: a dielectric mirror at each measured corneal position, so what
lands in the eye is whatever the scene actually contains. Presence, size, shape, colour and
occlusion are all decided by the scene rather than by a rule, which is the whole reason not to
paint a dot: a bulb 0.5 m away gives a bright disc, a window a soft rectangle, an overcast sky a
broad wash, an unlit corner nothing at all.

KEEP OR REPLACE IS COMPUTED, NOT SET. A photographed catchlight is better than any synthetic one
when it happens to fit -- it has the right shape, the right softness and real lash occlusion. It
fits only when the new key is close to the original, and the glint moves by HALF the angle the
light moves, so the tolerance looks generous and is not. Both angles are measured: the plate's
from its own catchlights (``eye_geometry.py``: 54.6 deg off the camera axis on this shot) and the
scene's from the L1 moment of the rendered irradiance field. Below ``--catchlight-keep-deg`` the
plate's own reflection is put back instead; above it the rendered one is used. Relighting into a
different environment is exactly the case where the answer is "replace", so replace is what
normally happens -- but the decision is a measured angle, not a preference.

The added pass is masked to the corneal cap, because the mirror is a sphere of the corneal radius
and only the cap within the limbus is really cornea. Beyond it the model is wrong and the sclera
is doing something else.
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import sys, json, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np, cv2
from shot_paths import add_shot_argument, from_args
from color_pipeline import linear_to_srgb, srgb_to_linear
from eyelid_fit import evaluate as lid_evaluate
from eye_reflection_repair import lid_distance as landmark_lid_distance
import gpu_image
from ratio_edges import extend_irradiance
from separate_specular import load_layers, specular_dir

LUMA = np.array([.2126, .7152, .0722], np.float32)
# Must match scripts/stress_proxy.py --ratio and scripts/make_albedo.py.
RATIO_ALBEDO = 0.6
SRC_AMBIENT, SRC_KEY = 0.55, 0.45
L_SRC = np.array([-0.28, -0.92, 0.28], np.float32)
L_SRC /= np.linalg.norm(L_SRC)


def source_irradiance(normals):
    """E_source: the plate's own lighting. ASSUMED, and the weakest link in any ratio method.

    Identical to relight_sh.py and make_albedo.py on purpose -- three different models of the
    original key would put three different errors into three stages of the same pipeline.
    """
    return SRC_AMBIENT + SRC_KEY * np.maximum(np.sum(normals * L_SRC, -1), 0)


def load_source_light(shot):
    """The MEASURED E_source for this shot, or None if it has not been estimated yet.

    scripts/estimate_source_light.py fits SH-L2 lighting to the plate's own skin radiance against
    the neural normals. It replaces a hand-written frontal-key guess that differed from the
    measurement by 24% relative RMS over the visible normals, mostly because the real plate is
    about 2.3x more directionally lit than the guess allowed.
    """
    p = shot.log('source_light.json')
    if not p.exists():
        return None
    return np.array(json.loads(p.read_text())['sh_coefficients_rgb'], np.float32)


def source_irradiance_measured(normals, C):
    """E_source(n) from the fitted SH coefficients."""
    from sh_project import sh_basis
    Y = sh_basis(normals[..., 0], normals[..., 1], normals[..., 2])
    return np.maximum(np.einsum('hwk,kc->hwc', Y, C), 1e-4)


def ldir_colour(E, mask):
    """Colour of the dominant light, normalised to unit luminance, so G carries its hue."""
    if not mask.any():
        return np.ones(3, np.float32)
    c = E[mask].mean(0)
    return (c / max(float(c @ LUMA), 1e-6)).astype(np.float32)


def dominant_light(E, normals, mask):
    """Estimate the dominant light direction from the irradiance field and the normals.

    The first-order spherical-harmonic moment of E over the surface, sum(E * n), points at the
    dominant source. Estimating it from the render rather than reading it out of the rig log
    means this works unchanged for an environment, several sources, or bounce -- and it cannot
    drift out of sync with whatever actually lit the frame.
    """
    if not mask.any():
        return np.array([0.0, -1.0, 0.0], np.float32), 0.0
    lum = (E[mask] @ LUMA).astype(np.float32)
    n = normals[mask]
    # The sample is the camera-VISIBLE surface, not a uniform sphere, so its mean normal already
    # points at the camera. Taking sum(E * n) raw therefore reports a frontal light even under
    # perfectly flat ambient -- the review measured concentration 0.678 for a constant field,
    # which invented a frontal highlight out of nothing. Subtracting the mean-weighted normal
    # removes the sampling bias and leaves only the part of E that actually VARIES with
    # orientation, which is what a directional source produces.
    mean_n = n.mean(0)
    v = ((lum[:, None] * (n - mean_n))).sum(0)
    mag = float(np.linalg.norm(v))
    if mag < 1e-8:
        return np.array([0.0, -1.0, 0.0], np.float32), 0.0
    d = v / mag
    # Directionality: the anisotropic energy as a fraction of the total. A flat ambient now gives
    # ~0 because the bias is gone; a single hard source stays high.
    conc = float(mag / max(lum.sum(), 1e-8))
    return d.astype(np.float32), min(conc, 1.0)


def specular_term(normals, light_dir, view_dir, roughness, ndl_gate):
    """A GGX lobe NORMALISED TO PEAK 1, times Schlick Fresnel and N.L. Dimensionless, in [0,1].

    The un-normalised GGX D is a probability density per steradian, not an image quantity: it
    peaks at 1/(pi*a^2), which for skin roughness reaches ~39. Added straight to a plate whose
    mean is 0.089 it produced a washed-out slab across half the frame -- the artist spotted it
    immediately as "vertical stripes". Dividing by that peak makes the term a dimensionless
    fraction, so the CALLER supplies the physical magnitude by multiplying it with the actual
    irradiance.
    """
    h = light_dir + view_dir
    h = h / max(float(np.linalg.norm(h)), 1e-8)
    ndh = np.clip(np.sum(normals * h, -1), 0.0, 1.0)
    ndv = np.clip(np.sum(normals * view_dir, -1), 0.0, 1.0)
    a = np.maximum(roughness, 1e-3) ** 2       # roughness is a per-pixel field, not a scalar
    a2 = a * a
    denom = ndh * ndh * (a2 - 1.0) + 1.0
    D = a2 / np.maximum(np.pi * denom * denom, 1e-8)
    D = D * (np.pi * a2)                       # peak of D is 1/(pi*a2); normalise it away
    F = 0.04 + 0.96 * (1.0 - ndv) ** 5
    return np.clip(D, 0.0, 1.0) * F * ndl_gate


def detail_sigma(E_t, E_s, mask, scales=(2, 4, 8, 16, 32, 64)):
    """Scale at which the rendered irradiance stops resolving what the neural normals resolve.

    Both fields are compared as the relative RMS of a log-domain high-pass, over the matte core.
    The crossover is where the proxy stops being the less detailed of the two; below it, dividing
    by E_source removes shading the renderer has no way to restore. Measuring it per shot rather
    than fixing a pixel count matters because it moves with the mesh stride, the depth smoothing
    and how large the subject is in frame.
    """
    if not mask.any():
        return 0.0, []
    lt = np.log2(np.maximum(E_t, 1e-6))
    ls = np.log2(np.maximum(E_s, 1e-6))
    table, cross = [], 0.0
    prev = None
    for sc in scales:
        a = float((lt - cv2.GaussianBlur(lt, (0, 0), sc))[mask].std())
        b = float((ls - cv2.GaussianBlur(ls, (0, 0), sc))[mask].std())
        ratio = b / max(a, 1e-9)
        table.append({'scale_px': sc, 'e_target_rms': round(a, 4),
                      'e_source_rms': round(b, 4), 'source_over_target': round(ratio, 3)})
        if prev is not None and prev[1] > 1.0 >= ratio:
            # Log-linear interpolation between the bracketing scales.
            s0, r0 = prev
            t = (r0 - 1.0) / max(r0 - ratio, 1e-9)
            cross = float(np.exp(np.log(s0) + t * (np.log(sc) - np.log(s0))))
        prev = (sc, ratio)
    if cross == 0.0 and table and table[-1]['source_over_target'] > 1.0:
        cross = float(scales[-1])
    return cross, table


# ---- round 41: the eye pass under the ratio, and the tour reflected in her corneas ---------
# `plate = D + S` was the round 5 lesson: the ratio can recolour a highlight but cannot move or
# delete it. Rounds 33-40 built the removal that deletes the capture room's softbox from her
# corneas (accepted: r40_floorB). This is the other half. The eye pass's frames are substituted
# for the plate where it changed a pixel (it is exact source everywhere else), the ratio
# multiplies THAT, and the tour's own reflection -- rendered by stress_proxy.py --corneal in
# a corneal cap that sees the same bulb and fill the ratio pass was lit by -- is ADDED through
# the tracked eyelids and the limbus, in the same exposure units as the ratio's anchor.

def load_eye_tracks(shot, limbus_name, eyelid_name):
    """{(frame, side): (centre, radius, lid record or None)} from the fitted tracks."""
    limbus = json.loads((shot.relight / limbus_name).read_text(encoding='utf-8'))
    lids = json.loads((shot.relight / eyelid_name).read_text(encoding='utf-8'))
    lid_by = {(int(r['frame']), e['side']): e for r in lids['rows'] for e in r['eyes']
              if e.get('fitted')}
    out = {}
    for r in limbus['rows']:
        for e in r['eyes']:
            if e.get('fitted'):
                key = (int(r['frame']), e['side'])
                out[key] = (tuple(e['smoothed_centre']), float(e['smoothed_axes'][0]),
                            lid_by.get(key))
    return out


def corneal_matte(shape, centre, radius, lid, feather=1.0, iris_feather=1.5, reach=2.9,
                  landmarks=None, limbus=True):
    """Soft matte of the OPEN CORNEA for one eye: inside the fitted lids AND inside the limbus.

    The lid matte is what got stage 11d rejected when it was missing (silver arcs on the upper
    lid); the limbus gate is round 34's (a cap aimed down an imperfect gaze puts its edge a
    pixel or two past the iris, onto sclera). Both feathered about a pixel: a hard edge on a
    25-pixel eye is a step. Returns (window slices, matte, source) or None if the eye's opening is unknown -- then it
    gets nothing, never a glint on a closed lid. Where round 40's `lower_iris` fit exists (the
    lower lid over her iris, which the lower walk cannot see), it is the lower bound; it is at
    or above `lower` by construction.

    Without a lid fit (the eyelid fit abstains around a blink and on a few open eyes -- 42 of
    300 on the tour), the eye pass falls back to the mediapipe eyelid polygon, and so does
    this: `landmarks` is (points, lid_ids) for the frame. Round 35 measured that polygon
    8.34 px INSIDE the true opening on the left eye, so a glint near a lid can be clipped
    there; it never paints one on a closed lid.
    """
    have_fit = lid is not None and lid.get('upper') is not None and lid.get('lower') is not None
    if not have_fit and landmarks is None:
        return None
    h, w = shape
    cx, cy = centre
    r = radius * reach + 14
    x0, x1 = max(0, int(np.floor(cx - r))), min(w, int(np.ceil(cx + r)) + 1)
    y0, y1 = max(0, int(np.floor(cy - r))), min(h, int(np.ceil(cy + r)) + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    xx = xx.astype(np.float64)
    yy = yy.astype(np.float64)
    m = np.ones(xx.shape, np.float32)
    if have_fit:
        c = lid['centre']
        edges = [(lid['upper']['coef'], 1.0), (lid['lower']['coef'], -1.0)]
        if lid.get('lower_iris'):
            edges.append((lid['lower_iris']['coef'], -1.0))
        for coef, sign in edges:
            edge = lid_evaluate(coef, xx, c)
            m = np.minimum(m, np.clip(sign * (yy - edge) / max(feather, 1e-6) + 0.5, 0.0, 1.0)
                           .astype(np.float32))
        source = 'image_fit'
    else:
        points, lid_ids = landmarks
        d = landmark_lid_distance(points, lid_ids, (slice(y0, y1), slice(x0, x1)))
        m = np.clip(d / max(feather, 1e-6), 0.0, 1.0).astype(np.float32)
        source = 'mediapipe_polygon'
    if limbus:
        rho = np.hypot(xx - cx, yy - cy) / max(radius, 1e-6)
        m *= np.clip((1.0 - rho) / max(iris_feather / max(radius, 1e-6), 1e-6), 0.0, 1.0
                     ).astype(np.float32)
    # Round 42 (`limbus=False`): the pass carries a scleral globe as well as the cap, so the
    # matte is the fitted lids only and the fill's sheen continues past the limbus.
    return (slice(y0, y1), slice(x0, x1)), m, source


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_shot_argument(ap)
    ap.add_argument('--render-dir', default='',
                    help='EXR irradiance renders; default composite/ratio_render')
    ap.add_argument('--name', default='ratio')
    ap.add_argument('--max-stops', type=float, default=4.0,
                    help='upper clamp on the ratio, in stops. The lower end is NOT clamped: a '
                         'shadow going dark is the point')
    ap.add_argument('--floor', type=float, default=0.004,
                    help='smallest ratio, so a fully shadowed pixel keeps a trace of its own '
                         'texture instead of becoming pure black')
    ap.add_argument('--anchor-frame', type=int, default=0)
    ap.add_argument('--exposure', type=float, default=1.0)
    ap.add_argument('--assumed-source', action='store_true',
                    help='fall back to the old hand-written frontal key instead of the measured '
                         'E_source, for comparison only')
    ap.add_argument('--specular', type=float, default=1.0,
                    help='gain on the additive specular term; 0 disables it')
    ap.add_argument('--plate-specular', choices=('remove', 'keep'), default='keep',
                    help="'remove' multiplies the DIFFUSE layer from separate_specular.py, so "
                         "the green-screen room's own highlights leave with it. 'keep' is the "
                         'default round-4 behaviour. Removal is experimental and damages teeth, '
                         'sclera and hair (round6 rejection addendum). With no separation both '
                         'behave identically')
    ap.add_argument('--source-detail-sigma', type=float, default=-1.0,
                    help='EXPERIMENT, off by default: low-pass on E_source before the division, '
                         'in plate px. 0 measures the crossover from this shot; negative '
                         'disables it. Tested against the lip contrast it made things WORSE '
                         '(+11.5%% vs +7.0%%), so it is not the fix and does not ship on')
    ap.add_argument('--catchlight', type=float, default=0.0,
                    help='EXPERIMENT: gain on the rendered corneal reflection; off by default. '
                         'The cap lacks eyelid occlusion and replacement depends on an unsafe split')
    ap.add_argument('--catchlight-dir', default='',
                    help='EXR catchlight renders; default composite/catchlight_render')
    ap.add_argument('--catchlight-cap', type=float, default=0.8,
                    help='corneal cap radius as a fraction of the iris radius. Below 1.0 because '
                         'the proxy has no eye sockets, so near the limbus the sphere reflects a '
                         'room that a real brow and lid would hide')
    ap.add_argument('--catchlight-baseline-pct', type=float, default=70.0,
                    help='percentile of the corneal reflection treated as the broad ambient '
                         'wash and NOT added. It is already in the diffuse layer: '
                         'separate_specular.py only removes what stands above the local diffuse '
                         'level, so the plate keeps its dim corneal reflection of the original '
                         'room and adding a full new one on top double-counts it')
    ap.add_argument('--catchlight-keep-deg', type=float, default=15.0,
                    help="if the scene's dominant light is within this angle of the key measured "
                         "from the PLATE's own catchlights, the photographed reflection is put "
                         'back instead of the rendered one -- it has the right shape, softness '
                         'and lash occlusion for free. The glint moves by HALF the angle the '
                         'light moves, which makes this tolerance tighter than it looks')
    ap.add_argument('--roughness-skin', type=float, default=0.30)
    ap.add_argument('--roughness-other', type=float, default=0.80,
                    help='wool, hair and foam are rough; only skin is tight-specular')
    ap.add_argument('--specular-other', type=float, default=0.15,
                    help='specular weight for non-skin, as a fraction of skin. Specular does not '
                         'scale with albedo, so a dark jacket needs this or it reads as wet')
    ap.add_argument('--resolution', default='',
                    help='WxH output; defaults to the shot delivery resolution, then the plate')
    ap.add_argument('--only-output', default='', help='comma-separated output frames, plus exposure anchor')
    # Round 4 experiment, OPT-IN on purpose. It replaces the inpainted rim with a nearest
    # extension of the reliable interior. It does suppress the dotted silhouette, but radius 8
    # leaves dots and radius 24 flattens real rim light, so it is not the fix -- the fix is the
    # mesh. Left on by default it would have shipped silently in the next production run.
    ap.add_argument('--eye-removal', default='',
                    help='round 41: composite/<run> holding removal/ and weight/ PNGs from the eye '
                         'pass (eye_removal_gpu.py; exact source outside the weight). Its pixels '
                         'replace the plate wherever the weight is non-zero, BEFORE the ratio')
    ap.add_argument('--corneal-dir', default='',
                    help='round 41: composite/<dir> of stress_proxy.py --corneal EXRs -- the tour '
                         'reflected in her corneas. ADDED after the ratio through the tracked lids '
                         'and limbus, scaled by the same anchor the plate is multiplied by')
    ap.add_argument('--limbus-track', default='limbus_track_tv_import.json')
    ap.add_argument('--eyelid-track', default='eyelid_track_wtv40_import.json')
    ap.add_argument('--corneal-blur', type=float, default=0.0,
                    help='Gaussian sigma in plate pixels applied to the corneal pass before it is '
                         'added: the render is sharper than the photograph (measure_glint_psf.py)')
    ap.add_argument('--corneal-gain', type=float, default=1.0)
    ap.add_argument('--corneal-feather', type=float, default=1.0)
    ap.add_argument('--corneal-matte', choices=('cornea', 'lids'), default='cornea',
                    help='round 42: `lids` mattes the corneal pass by the fitted lids ONLY (no '
                         'limbus gate), for a pass rendered with stress_proxy.py --corneal '
                         '--globe, whose wet sclera reflects too. `cornea` is round 41')
    ap.add_argument('--extend-edges', action='store_true',
                    help='experimental: nearest-extend irradiance from the eroded reliable interior')
    ap.add_argument('--edge-radius', type=int, default=8,
                    help='with --extend-edges: unreliable proxy rim in plate pixels')
    args = ap.parse_args()
    shot = from_args(args)
    if args.plate_specular == 'remove' or args.catchlight > 0:
        print('EXPERIMENTAL round-5 path: rejected for damaged teeth/sclera, unstable hair '
              'and catchlights over eyelids. See rounds/round6.md rejection addendum.',
              file=sys.stderr, flush=True)

    # stress_proxy.py --ratio --name X writes to composite/X, so that is where this looks by
    # default. They previously disagreed and the documented command found nothing.
    rdir = Path(args.render_dir) if args.render_dir else (shot.composite / 'ratio_render')
    renders = sorted(rdir.glob('*.exr'))
    if not renders:
        raise SystemExit('No irradiance renders in %s. Run stress_proxy.py --ratio first.' % rdir)
    tour = json.loads(shot.log('stress_proxy.json').read_text())
    pf = {r['out_frame']: r for r in tour['per_frame']}

    out_dir = shot.composite / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    # Output at the shot's DELIVERY resolution, not the plate's. For the artist's own footage
    # that is 3840x1600; for this 1080p stock plate the delivery spec is 16:9 and upscaling it
    # would invent nothing, so the plate size is kept unless asked otherwise.
    out_res = None
    if args.resolution:
        out_res = tuple(int(x) for x in args.resolution.lower().split('x'))
    elif tuple(shot.resolution) != (3840, 2160):
        out_res = tuple(shot.resolution)

    # The anchor frame is chosen, not just the first one on disk: an atypically dark or bright
    # opening frame would otherwise set the exposure for the whole sequence. --anchor-frame was
    # parsed and never read until the review pointed it out.
    src_C = None if args.assumed_source else load_source_light(shot)
    print('E_source: %s' % ('MEASURED from the plate (logs/source_light.json)' if src_C is not None
                            else 'assumed frontal key -- run estimate_source_light.py'), flush=True)

    anchor_idx = min(max(args.anchor_frame, 0), len(renders) - 1)
    order = [renders[anchor_idx]] + [r for i, r in enumerate(renders) if i != anchor_idx]
    if args.only_output:
        selected = {int(x) for x in args.only_output.split(',')}
        order = [order[0]] + [r for r in order[1:] if int(r.stem) in selected]

    # Corneal geometry and the plate's own measured key. Both optional: without them the
    # catchlight stage simply does not run, and the composite is unchanged.
    eye_geo, plate_key = {}, None
    egp = shot.relight / 'eye_geometry.json'
    if egp.exists():
        _eg = json.loads(egp.read_text())
        eye_geo = {r['frame']: r for r in _eg['per_frame']}
        pk = _eg.get('plate_key_light') or {}
        if pk.get('mean_direction_world'):
            plate_key = np.array(pk['mean_direction_world'], np.float32)
    cdir = Path(args.catchlight_dir) if args.catchlight_dir else (shot.composite /
                                                                  'catchlight_render')

    eye_tracks, removal_dir, corneal_dir = None, None, None
    if args.eye_removal:
        removal_dir = Path(args.eye_removal)
        if not removal_dir.is_absolute():
            removal_dir = shot.composite / args.eye_removal
        if not (removal_dir / 'removal').is_dir() or not (removal_dir / 'weight').is_dir():
            raise SystemExit('No removal/ and weight/ under %s' % removal_dir)
    if args.corneal_dir:
        corneal_dir = Path(args.corneal_dir)
        if not corneal_dir.is_absolute():
            corneal_dir = shot.composite / args.corneal_dir
        if not corneal_dir.is_dir():
            raise SystemExit('No corneal pass at %s. Run stress_proxy.py --corneal.' % corneal_dir)
        eye_tracks = load_eye_tracks(shot, args.limbus_track, args.eyelid_track)
        lm_points, lm_found, lm_topology = None, None, None
        if (shot.relight / 'face_landmarks.npz').exists():
            with np.load(shot.relight / 'face_landmarks.npz') as z:
                lm_points, lm_found = z['points'], z['found']
            lm_topology = json.loads((shot.relight / 'face_landmarks.json').read_text())['topology']
        print('corneal pass: %s through %s + %s, blur sigma %.2f px, gain %.2f' % (
            corneal_dir.name, args.limbus_track, args.eyelid_track, args.corneal_blur,
            args.corneal_gain), flush=True)

    anchor, stats = None, []
    sigma = None if args.source_detail_sigma >= 0 else 0.0
    sigma_table = []
    for r in order:
        n = int(r.stem)
        fi = pf[n]['shot_frame'] if n in pf else n

        img = cv2.imread(str(r), -1)
        if img is None:
            raise SystemExit('cannot read %s' % r)
        E_t = img[..., 2::-1].astype(np.float32) / RATIO_ALBEDO      # BGRA -> RGB, undo albedo
        cov = img[..., 3].astype(np.float32) if img.shape[2] == 4 else np.ones(img.shape[:2], np.float32)

        # D and S. Without a separation on disk this returns (plate, zeros), so an unseparated
        # shot composites exactly as it did in round 4.
        if args.plate_specular == 'remove':
            D_plate, S_plate = load_layers(shot, fi)
            fg = D_plate
        else:
            # Read the photograph directly: an old, corrupt or overshooting experimental
            # cache must never affect keep mode, even through a subtract/add round trip.
            fg = cv2.imread(str(shot.fg(fi)), -1)[..., ::-1].astype(np.float32)
            S_plate = np.zeros_like(fg)
        removal_px = 0
        if removal_dir is not None:
            # The eye pass wrote 8-bit sRGB of the SAME linear fg (its removal.json records the
            # fg's sha256 per frame); where its weight is zero the pixel is the untouched fg.
            rem = cv2.imread(str(removal_dir / 'removal' / ('%06d.png' % fi)), -1)
            wgt = cv2.imread(str(removal_dir / 'weight' / ('%06d.png' % fi)), -1)
            if rem is None or wgt is None:
                raise SystemExit('No eye removal frame %06d under %s' % (fi, removal_dir))
            if wgt.ndim == 3:
                wgt = wgt[..., 0]
            changed = wgt > 0
            removal_px = int(changed.sum())
            if removal_px:
                fg = fg.copy()
                fg[changed] = srgb_to_linear(rem[..., ::-1].astype(np.float32) / 255.0)[changed]
        alpha = cv2.imread(str(shot.alpha(fi)), -1)
        if alpha.ndim == 3:
            alpha = alpha[..., 0]
        alpha = np.clip(alpha, 0, 1)
        h, w = alpha.shape
        if E_t.shape[:2] != (h, w):
            E_t = cv2.resize(E_t, (w, h), interpolation=cv2.INTER_LINEAR)
            cov = cv2.resize(cov, (w, h), interpolation=cv2.INTER_LINEAR)

        nrm = np.load(shot.relight / 'normals' / ('%06d.npz' % fi))['normal'].astype(np.float32)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-6)
        if src_C is not None:
            E_s = cv2.resize(source_irradiance_measured(nrm, src_C), (w, h),
                             interpolation=cv2.INTER_LINEAR)
            E_s = (E_s @ LUMA)[..., None]
        else:
            E_s = cv2.resize(source_irradiance(nrm), (w, h),
                             interpolation=cv2.INTER_LINEAR)[..., None]
        # Band-limit E_source to what the proxy can put back. See the module docstring: below the
        # crossover the division can remove fine shading the numerator cannot restore. This
        # was tested and did NOT resolve the lip complaint; the experiment remains off.
        if sigma is None:
            core0 = (alpha > 0.99) & (cov > 0.5)
            measured, sigma_table = detail_sigma(E_t @ LUMA, E_s[..., 0], core0)
            sigma = (measured if args.source_detail_sigma == 0 else args.source_detail_sigma)
            print('E_source band-limit: sigma %.1f px (%s)' % (
                sigma, 'MEASURED crossover' if args.source_detail_sigma == 0
                else 'from --source-detail-sigma'), flush=True)
        if sigma > 0:
            # In log space, so the low-pass is on the ratio rather than on the level.
            lo = np.log2(np.maximum(E_s, 1e-6))
            E_s = np.exp2(cv2.GaussianBlur(lo[..., 0], (0, 0), sigma))[..., None]
        # Specular needs normals at plate resolution, and a per-material roughness.
        nrm_full = cv2.resize(nrm, (w, h), interpolation=cv2.INTER_LINEAR)
        nrm_full /= np.maximum(np.linalg.norm(nrm_full, axis=-1, keepdims=True), 1e-6)
        skin_p = shot.relight / 'skin_mask' / ('%06d.png' % fi)
        if skin_p.exists():
            raw = cv2.imread(str(skin_p), cv2.IMREAD_UNCHANGED)
            # Test the dtype BEFORE casting. Casting first made the check always false, so the
            # 16-bit mask was used at 0..65535 instead of 0..1: roughness reached -32766 and the
            # specular weight 55704. Found by the gpt-6-astra review.
            scale = {np.dtype(np.uint16): 65535.0, np.dtype(np.uint8): 255.0}.get(raw.dtype, 1.0)
            sk = raw.astype(np.float32) / scale
            if sk.ndim == 3:
                sk = sk[..., 0]
            skin = np.clip(cv2.resize(sk, (w, h), interpolation=cv2.INTER_LINEAR), 0.0, 1.0)
        else:
            skin = np.zeros((h, w), np.float32)

        # Where the proxy did not cover a matte pixel -- a hair wisp outside the shell, a
        # silhouette disagreement -- the division is meaningless. Fill from nearby valid ratio
        # rather than letting an edge artefact multiply the plate.
        #
        # Inpainting at full 1920x1080 dominated the frame time and cv2.inpaint has no CUDA path
        # in this wheel. The gap is a thin rim around the silhouette and the fill only has to be
        # plausible, so it is solved at quarter resolution and upsampled: visually identical,
        # about 16x less work.
        valid = (cov > 0.5)
        if args.extend_edges:
            E_t, valid = extend_irradiance(E_t, cov, args.edge_radius)
        elif not valid.all():
            q = (w // 4, h // 4)
            scale = max(float(np.max(E_t[valid])), 1e-6)
            small = cv2.resize(np.clip(E_t / scale, 0, 1), q, interpolation=cv2.INTER_AREA)
            smask = (cv2.resize((~valid).astype(np.uint8) * 255, q,
                                interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8) * 255
            filled = cv2.inpaint((small * 255).astype(np.uint8), smask, 3, cv2.INPAINT_TELEA)
            fill = cv2.resize(filled.astype(np.float32) / 255.0 * scale, (w, h),
                              interpolation=cv2.INTER_LINEAR)
            E_t = np.where(valid[..., None], E_t, fill)

        if anchor is None:
            R0 = E_t / np.maximum(E_s, 1e-3)
            m = alpha > 0.99
            ref = float(np.median((R0[m] @ LUMA))) if m.any() else 1.0
            anchor = 1.0 / max(ref, 1e-6)

        # Division, anchor, one-sided log clamp, blur and plate multiply, fused into a single
        # device round trip. Chaining them would move 1920x1080x3 across PCIe six times a frame.
        lit, R = gpu_image.ratio_composite(E_t, np.broadcast_to(E_s, E_t.shape).copy(), fg, alpha,
                                           anchor * args.exposure, args.max_stops, args.floor, 1.5)

        spec_energy = 0.0
        # The scene's dominant light direction is needed by the catchlight decision too, so it
        # is estimated whether or not the additive specular term is enabled.
        core = (alpha > 0.99) & valid
        ldir_c, conc_c = dominant_light(E_t, nrm_full, core)
        conc_ok = conc_c > 1e-4
        if args.specular > 0:
            ldir, conc = ldir_c, conc_c
            # The plate camera looks along +Y, so the view direction from any surface point back
            # to the camera is -Y. Exact enough at this field of view, and it avoids needing a
            # per-pixel world position.
            vdir = np.array([0.0, -1.0, 0.0], np.float32)
            ndl = np.clip(np.sum(nrm_full * ldir, -1), 0.0, 1.0)
            rough = args.roughness_other + (args.roughness_skin - args.roughness_other) * skin
            # Specular does NOT scale with albedo, so without a per-material weight a dark wool
            # jacket receives the same highlight as skin and reads as wet plastic -- which is
            # exactly how it looked. Wool is rough and its specular is weak and broad; skin is
            # the only strongly specular surface here.
            spec_w = args.specular_other + (1.0 - args.specular_other) * skin
            spec = specular_term(nrm_full, ldir, vdir, rough, ndl)
            # G = normalised lobe x THE ACTUAL IRRADIANCE x gain. E_t already carries both the
            # magnitude and the colour of the light AND the renderer's occlusion, so a point in
            # shadow gets no highlight for free -- no separate gate, and no unit mismatch. The
            # first version normalised the colour to unit luminance and so discarded the
            # magnitude entirely, which is why it blew out.
            G = (spec * spec_w * conc * args.specular)[..., None] * E_t * (anchor * args.exposure)
            # A highlight brighter than a few times the local diffuse level is a bug, not a look.
            G = np.minimum(G, 3.0 * np.maximum(lit, 1e-4))
            lit = lit + G
            spec_energy = float(np.mean(G[core] @ LUMA) / max(float(np.mean(lit[core] @ LUMA)), 1e-9)) if core.any() else 0.0
        # ---- corneal reflection -------------------------------------------------------------
        cl_log = None
        if args.catchlight > 0 and eye_geo:
            rec = eye_geo.get(fi) or {}
            eyes = {k: v for k, v in (rec.get('eyes') or {}).items() if v.get('visible')}
            if eyes:
                # Corneal cap mask. The mirror is a sphere of the corneal radius, but only the
                # cap inside the limbus is cornea; beyond it the sphere is not the real surface.
                cap = np.zeros((h, w), np.float32)
                yy, xx = np.mgrid[0:h, 0:w]
                yy = yy.astype(np.float32); xx = xx.astype(np.float32)
                for d in eyes.values():
                    cx, cy = d['iris_centre_px']
                    # The cap stops at the LIMBUS, which is the iris radius -- not at the
                    # sphere's silhouette. The cornea is a 7.8 mm sphere but only the cap within
                    # 5.85 mm of the axis is cornea, i.e. 48.6 deg, and Fresnel there is still a
                    # few per cent. Masking out to the silhouette instead put the GRAZING rim,
                    # where a dielectric reflects ~100%, right on the iris edge and drew a bright
                    # ring around each eye.
                    # The cap stops well inside the limbus, and the reason is a PROXY limitation
                    # rather than optics: make_proxy_mesh.py builds a displaced silhouette shell
                    # with NO EYE SOCKETS, so the corneal sphere sees the whole room from the
                    # sides, where a real brow, lid and lash occlude nearly all of it. A sphere
                    # also compresses an entire hemisphere into its last few rim pixels, so
                    # those directions arrive concentrated as well as unoccluded. Left at the
                    # limbus it drew a bright arc around each iris. Near the axis, where the
                    # geometry is trustworthy, the reflection is right.
                    rr = max(d['iris_radius_px_desqueezed'], 1.0) * args.catchlight_cap
                    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
                    cap = np.maximum(cap, np.clip((rr - dist) / max(rr * 0.45, 1.0), 0.0, 1.0))
                # Keep or replace, from the two measured angles.
                keep = False
                ang = None
                if plate_key is not None and conc_ok:
                    ang = float(np.degrees(np.arccos(np.clip(
                        float(plate_key @ ldir_c), -1.0, 1.0))))
                    keep = ang <= args.catchlight_keep_deg
                if keep:
                    add = S_plate * cap[..., None] * args.catchlight
                    src = 'plate (kept: scene key %.1f deg from the measured original)' % ang
                else:
                    cp = cdir / ('%06d.exr' % n)
                    if cp.exists():
                        cimg = cv2.imread(str(cp), -1)
                        cr = cimg[..., 2::-1].astype(np.float32)
                        if cr.shape[:2] != (h, w):
                            cr = cv2.resize(cr, (w, h), interpolation=cv2.INTER_LINEAR)
                        # ONLY THE COMPACT PART, and this is not a cosmetic choice -- it is what
                        # makes the addition balance the removal. separate_specular.py takes only
                        # what stands above the LOCAL DIFFUSE BASELINE, so the plate's broad,
                        # dim corneal reflection of the room was never removed; it is still in D
                        # and is relit with it. Adding the full mirror wash on top double-counts
                        # it, and it looked exactly like that: a pale disc over the whole iris,
                        # a contact lens rather than a catchlight. Subtracting the render's own
                        # local baseline leaves the reflections of actual SOURCES, which is what
                        # a catchlight is. A 4 cm bulb at 0.5 m subtends 4.6 deg and reflects in
                        # a 7.8 mm cornea as a spot well under a pixel wide -- small and bright
                        # is correct, and a disc filling the iris never was.
                        # Per eye, because the two corneas see different parts of the room. An
                        # opening will not do it: the wash covers the whole corneal disc, so any
                        # structuring element small enough to preserve the glint also fits
                        # inside the wash and preserves that too -- which is what happened.
                        crL = cr @ LUMA
                        keep = np.zeros_like(crL)
                        for d in eyes.values():
                            cx, cy = d['iris_centre_px']
                            rr = max(d['iris_radius_px_desqueezed'], 1.0)
                            disc = ((xx - cx) ** 2 + (yy - cy) ** 2) <= rr * rr
                            if disc.sum() < 8:
                                continue
                            b = float(np.percentile(crL[disc], args.catchlight_baseline_pct))
                            keep[disc] = np.clip(crL[disc] - b, 0.0, None)
                        cr = cr * (keep / np.maximum(crL, 1e-9))[..., None]
                        # Same anchor and exposure as the ratio: the render is in scene-linear
                        # units and the plate has been scaled to sit against them.
                        add = cr * cap[..., None] * (anchor * args.exposure) * args.catchlight
                        src = ('rendered' if ang is None else
                               'rendered (scene key %.1f deg from the measured original)' % ang)
                    else:
                        add, src = None, 'MISSING %s -- run stress_proxy.py --catchlight' % cp
                if add is not None:
                    lit = lit + add
                    cl_log = {'source': src, 'eyes': sorted(eyes),
                              'angle_to_plate_key_deg': ang,
                              'peak_added': float((add @ LUMA).max())}
                else:
                    cl_log = {'source': src, 'eyes': sorted(eyes)}

        corneal_log = None
        if corneal_dir is not None:
            cp = corneal_dir / ('%06d.exr' % n)
            if not cp.exists():
                raise SystemExit('No corneal frame %s' % cp)
            cimg = cv2.imread(str(cp), -1)
            cr = cimg[..., 2::-1].astype(np.float32)
            if cr.shape[:2] != (h, w):
                raise SystemExit('corneal pass %s is %s, plate is %s' % (cp, cr.shape[:2], (h, w)))
            if args.corneal_blur > 0:
                cr = cv2.GaussianBlur(cr, (0, 0), args.corneal_blur)
            corneal_log = {'eyes': {}, 'scale': float(anchor * args.exposure * args.corneal_gain)}
            for side in ('left', 'right'):
                tr = eye_tracks.get((fi, side))
                if tr is None:
                    corneal_log['eyes'][side] = {'added': False, 'why': 'no limbus fit'}
                    continue
                centre, radius, lid_rec = tr
                lm = None
                if (lm_points is not None and fi < len(lm_found) and lm_found[fi]
                        and np.isfinite(lm_points[fi]).all()):
                    lm = (lm_points[fi], lm_topology['eyelid_' + side])
                mt = corneal_matte((h, w), centre, radius, lid_rec, args.corneal_feather,
                                   landmarks=lm, limbus=(args.corneal_matte == 'cornea'))
                if mt is None:
                    corneal_log['eyes'][side] = {'added': False, 'why': 'no lid fit or polygon'}
                    continue
                win, m, lid_source = mt
                add = cr[win] * m[..., None] * (anchor * args.exposure * args.corneal_gain)
                e_all = float((cr[win] @ LUMA).sum())
                e_kept = float(((cr[win] @ LUMA) * m).sum())
                lit[win] = lit[win] + add
                corneal_log['eyes'][side] = {
                    'added': True, 'lid_source': lid_source, 'energy_rendered': e_all, 'lid_limbus_kept': e_kept / max(e_all, 1e-9),
                    'peak_added_linear': float((add @ LUMA).max()),
                    'peak_added_display': float(linear_to_srgb(np.clip((add @ LUMA).max(), 0, 1)))}
        out_rgb = np.clip(lit * alpha[..., None], 0, None)
        if out_res is not None and (out_res[0], out_res[1]) != (w, h):
            # Downscale AFTER compositing, so the matte's fractional alpha and the ratio's shadow
            # edges are both supersampled rather than decided at output resolution.
            out_rgb = cv2.resize(out_rgb, out_res, interpolation=cv2.INTER_AREA)
        rgb = np.clip(linear_to_srgb(out_rgb), 0, 1)
        cv2.imwrite(str(out_dir / ('%06d.png' % n)), (rgb[..., ::-1] * 255).astype(np.uint8))

        m = alpha > 0.99
        if m.any():
            rl = R[m] @ LUMA
            stats.append({'frame': n, 'shot_frame': fi, 'catchlight': cl_log,
                          'eye_removal_px': removal_px, 'corneal': corneal_log,
                          'station': pf[n]['station'] if n in pf else None,
                          'ratio_luma_median': float(np.median(rl)),
                          'ratio_luma_p5': float(np.percentile(rl, 5)),
                          'ratio_luma_p95': float(np.percentile(rl, 95)),
                          'dynamic_range_stops': float(np.log2(
                              max(np.percentile(rl, 95), 1e-6) / max(np.percentile(rl, 5), 1e-6)))})
        if n % 40 == 0 and stats:
            print(json.dumps(stats[-1]), flush=True)

    stats.sort(key=lambda x: x['frame'])
    dr = np.array([s['dynamic_range_stops'] for s in stats])
    dev = gpu_image.describe()
    summary = {
        'shot': shot.name, 'frames': len(stats), 'output': str(out_dir),
        'eye_removal': (str(removal_dir) if removal_dir is not None else None),
        'corneal': ({'dir': str(corneal_dir), 'limbus_track': args.limbus_track,
                     'eyelid_track': args.eyelid_track, 'blur_sigma_px': args.corneal_blur,
                     'gain': args.corneal_gain, 'feather_px': args.corneal_feather,
                     'matte_mode': args.corneal_matte,
                     'matte': 'tracked upper/lower(/lower_iris) parabolas AND the limbus disc, '
                              'feathered; without a lid fit the mediapipe eyelid polygon, as '
                              'the eye pass does; with neither, nothing',
                     'eyes_on_polygon': int(sum(1 for st in stats for e in ((st.get('corneal') or {}).get('eyes') or {}).values() if e.get('lid_source') == 'mediapipe_polygon')),
                     'scale': 'anchor x exposure x gain -- the same anchor the plate is '
                              'multiplied by, so render and plate share units',
                     'eyes_added': int(sum(1 for st in stats for e in ((st.get('corneal') or {}).get('eyes') or {}).values() if e.get('added'))),
                     'peak_added_display_max': max([e['peak_added_display'] for st in stats for e in ((st.get('corneal') or {}).get('eyes') or {}).values() if e.get('added')] or [0.0])}
                    if corneal_dir is not None else None),
        'formula': 'out = plate x R,  R = E_target(rendered) / E_source(assumed)',
        'why': 'the plate keeps every pore, hair strand and eye specular; the renderer supplies '
               'only the change in light. Rendering the proxy instead shows the proxy.',
        'E_source': {'model': ('MEASURED by estimate_source_light.py' if src_C is not None
                              else 'soft frontal studio key (ASSUMED, not measured)'),
                     'direction': L_SRC.tolist(), 'ambient': SRC_AMBIENT, 'key': SRC_KEY,
                     'shared_with': 'relight_sh.py and make_albedo.py, deliberately identical'},
        'clamp': {'upper_stops': args.max_stops, 'lower': 'unclamped down to floor %.4f' % args.floor,
                  'contrast_with_gain_map': 'relight_sh.py clamped BOTH ends at 1.5 stops, '
                                            'capping the whole map at 2.83:1, which is why it '
                                            'could never produce a terminator'},
        'exposure_anchor': anchor, 'compute': dev,
        'plate_specular': {
            'mode': args.plate_specular,
            'layer': str(specular_dir(shot)),
            'present': specular_dir(shot).exists(),
            'why': 'a smooth R can recolour a highlight but cannot move or delete one, so with '
                   'the plate specular left in, every shot keeps reporting where the ORIGINAL '
                   "room's lamps were -- the crown sheen, the lip gloss, and the corneal "
                   'catchlight most of all'},
        'source_band_limit': {
            'sigma_px': sigma,
            'source': ('MEASURED crossover' if args.source_detail_sigma == 0
                       else ('disabled -- round 4 behaviour' if args.source_detail_sigma < 0
                             else 'explicit --source-detail-sigma')),
            'bandwidth_table': sigma_table,
            'rule': 'you may only divide out lighting at the resolution at which you can put it '
                    'back. Below the crossover E_target cannot restore what E_source removes, so '
                    'R carries 1/E_source and brightens whatever the original key had shaded'},
        'output_resolution': list(out_res) if out_res else 'plate resolution',
        'specular': {'gain': args.specular, 'mean_fraction_of_image': None,
                     'model': 'GGX + Schlick Fresnel on the NEURAL normals, added not multiplied',
                     'light_direction': 'estimated per frame from the L1 moment of the rendered '
                                        'irradiance field, so it also works for environments',
                     'magnitude': 'normalised GGX lobe (peak 1) x the rendered irradiance, '
                                  'so it carries the real light level, colour AND occlusion',
                     'clamped_to': '3x the local diffuse level',
                     'roughness': {'skin': args.roughness_skin, 'other': args.roughness_other},
                     'weight': {'skin': 1.0, 'other': args.specular_other}},
        'catchlight': {
            'gain': args.catchlight,
            'render_dir': str(cdir), 'render_present': cdir.exists(),
            'eye_geometry': str(egp), 'eye_geometry_present': egp.exists(),
            'plate_key_direction_world': (None if plate_key is None
                                          else plate_key.round(4).tolist()),
            'keep_threshold_deg': args.catchlight_keep_deg,
            'model': 'a dielectric mirror (IOR 1.376) at the measured corneal position, rendered '
                     'with the plate camera, masked to the corneal cap, reduced to its compact '
                     'part and added',
            'compact_only': 'the render minus its own local baseline. separate_specular.py only '
                            'removes what stands above the local diffuse level, so the broad dim '
                            'corneal reflection of the original room is STILL IN D. Adding the '
                            'full mirror wash on top double-counts it and reads as a contact '
                            'lens over the whole iris',
            'why_not_painted': 'presence, size, shape, colour and occlusion all come from the '
                               'scene. A distance rule would get sunlight backwards: a mirror '
                               'shows RADIANCE, which does not fall off with distance, so the '
                               'sun makes a tiny brilliant spark rather than nothing',
            'keep_or_replace': 'computed from two MEASURED angles -- the plate key from its own '
                               'catchlights, the scene key from the L1 moment of the rendered '
                               'irradiance. Not an artist setting'},
        'result': {'mean_dynamic_range_stops': float(dr.mean()),
                   'max_dynamic_range_stops': float(dr.max())},
        'per_frame': stats}
    (out_dir / 'composite_manifest.json').write_text(json.dumps(summary, indent=2))
    if not args.only_output:
        shot.log('composite_ratio.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ('frames', 'exposure_anchor', 'result')}, indent=2))


if __name__ == '__main__':
    main()
