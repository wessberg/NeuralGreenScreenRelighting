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
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import sys, json, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np, cv2
from shot_paths import add_shot_argument, from_args
from color_pipeline import linear_to_srgb
import gpu_image
from ratio_edges import extend_irradiance

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
    ap.add_argument('--extend-edges', action='store_true',
                    help='experimental: nearest-extend irradiance from the eroded reliable interior')
    ap.add_argument('--edge-radius', type=int, default=8,
                    help='with --extend-edges: unreliable proxy rim in plate pixels')
    args = ap.parse_args()
    shot = from_args(args)

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

    anchor, stats = None, []
    for r in order:
        n = int(r.stem)
        fi = pf[n]['shot_frame'] if n in pf else n

        img = cv2.imread(str(r), -1)
        if img is None:
            raise SystemExit('cannot read %s' % r)
        E_t = img[..., 2::-1].astype(np.float32) / RATIO_ALBEDO      # BGRA -> RGB, undo albedo
        cov = img[..., 3].astype(np.float32) if img.shape[2] == 4 else np.ones(img.shape[:2], np.float32)

        fg = cv2.imread(str(shot.fg(fi)), -1)[..., ::-1].astype(np.float32)
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
        if args.specular > 0:
            core = (alpha > 0.99) & valid
            ldir, conc = dominant_light(E_t, nrm_full, core)
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
            stats.append({'frame': n, 'shot_frame': fi,
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
        'result': {'mean_dynamic_range_stops': float(dr.mean()),
                   'max_dynamic_range_stops': float(dr.max())},
        'per_frame': stats}
    (out_dir / 'composite_manifest.json').write_text(json.dumps(summary, indent=2))
    if not args.only_output:
        shot.log('composite_ratio.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ('frames', 'exposure_anchor', 'result')}, indent=2))


if __name__ == '__main__':
    main()
