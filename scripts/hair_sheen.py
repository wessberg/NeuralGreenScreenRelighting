"""Round 45: take the CAPTURE ROOM's overhead band off her hair, and hand the composite what it
needs to put its own light up there instead.

    dependencies/CorridorKey/.venv/Scripts/python.exe scripts/hair_sheen.py --shot <dir> \\
        --name r45_sheen --length 150

The artist, after accepting round 44's skin work: *"we still have the shine in the very top of
her hair. Very clearly from a light hanging above her head."* And, on what the fix has to be:
*"if we move one of our own lights up there, in principle it should be able to recreate a
similar shine ... So we still need to capture the nuanced hair details and so we can re-light
the hair and preserve the specular response."* That is exactly what this stage is -- the room's
overhead band, removed from the photograph with her strand structure kept, together with the
strand TANGENT and strand TEXTURE that let the composite light the same hair with the tour's own
sources -- an anisotropic fibre highlight where a photographed one used to be.

WHAT THE REMOVAL IS, and the two things that had to be measured to get there.

The first build identified the band by level against a LOCAL dark-hair baseline and subtracted
the excess per pixel. The artist saw both of its faults in one still. A window narrower than the
band lets the band set its own baseline, so the broad arc over the top of her head reported no
excess and kept shining (*"it still shines along the arc/edge of her head"*); and a per-pixel
subtraction of everything above a baseline flattens the strand structure, which is the "nuanced
hair detail" that has to survive. A chroma gate tight enough to protect her parting punched a
HOLE in the weight there and left it standing bright over a darkened crown (*"a gap like she is
bald"*).

The second build fitted a physical fibre lobe -- one distant source, five parameters for the
whole clip -- and MEASURED that it is not good enough to remove with: over 60 000 crown samples
from five stations, the best direction correlates with the band's envelope at only **0.45**, so
a single Kajiya-Kay lobe explains about a fifth of its variance. That fit is still run and still
reported (`room_light_fit`), because what it DOES establish is useful and was otherwise unknown:
the room's source sits **above and in front of her**, and the composite's replacement can be
checked against it. But it is a diagnostic, not the subtraction.

So the removal is empirical and the model is kept for reading:

  lum      = env * tex             exactly, with `env` the mask-normalised low-pass of the hair
                                   luminance at `--envelope-sigma` FACE RADII and `tex` the
                                   strand modulation that is left
  b        = the REGION-WIDE dark-hair level (a low percentile over the whole region, which is
             the statistic round 44 SS5 measured its 23x against -- never a local window, which
             the band would set itself)
  S        = max(env - b, 0) * fade(env / b) * tex

Subtracting S leaves `(b + whatever the fade kept) * tex`: the band goes, the strand structure
stays, scaled onto the dark hair. `fade` is a smoothstep between `--level-lo` and `--level-hi`
on the ENVELOPE's own ratio, so a region only half a stop above the dark hair is barely touched
while the band, at 23x, goes entirely.

TEMPORAL. `env` is filtered in time before it is used: a symmetric median over `--window`
SECONDS in a head-stabilised frame (a similarity transform fitted to all mediapipe landmarks,
the same registration `audit_r44_temporal.py` uses), at quarter resolution because the envelope
is smooth by construction. The capture room's light does not move, so its band is nearly static
in head coordinates and the filter costs nothing real; what it removes is the per-frame jitter
that a per-frame estimate would otherwise inject, which is round 32's lesson applied here.

WHAT IS WRITTEN, per frame, at plate resolution:

  sheen/  16-bit linear RGB   the room's highlight to SUBTRACT from the plate before the ratio
  field/  16-bit RGB          R = strand tangent angle / 180 deg, G = strand texture / 4,
                              B = structure-tensor coherence
  weight/ 8-bit               where this stage acts: the hair region, feathered on its ARBITRARY
                              edges (hairline, face oval, radius) and carried by the matte's own
                              alpha at the silhouette -- the brightest sheen is ON the fractional
                              arc, and feathering there is what left it shining in the first pass

WHAT THIS CANNOT DO, stated once so no round re-discovers it: individual flyaway strands at the
silhouette are sub-pixel and live in the matte's alpha ramp; hair-on-hair self-shadowing is not
modelled, because the proxy shell has no strand geometry and the plate's strand texture is only
a stand-in for it; and the hair does not change SHAPE with the light. What it does do is make
the band respond -- move and change level with the new source -- and keep her photographed
strand texture while doing it.

The replacement belongs to the composite (`composite_ratio.py --hair-sheen`), because only
there are the per-source irradiance passes, the exposure anchor and the plate's albedo scale.
Removing without replacing leaves a black mass, which is a worse artefact than the one being
named; the two halves are one change and must be judged together.
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import sys, json, time, hashlib, argparse
from fractions import Fraction
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np, cv2
from shot_paths import add_shot_argument, from_args
from light_geometry import position_map, camera_world
import gpu_image
import hair_region as hr

LUMA = np.array([.2126, .7152, .0722], np.float32)
TEX_SCALE = 4.0          # strand texture is stored as texture / TEX_SCALE in a 16-bit channel


def smoothstep(x, a, b):
    t = np.clip((x - a) / max(b - a, 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def read_skin(shot, fi, shape):
    raw = cv2.imread(str(shot.relight / 'skin_mask' / ('%06d.png' % fi)), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return np.zeros(shape, np.float32)
    scale = {np.dtype(np.uint16): 65535.0, np.dtype(np.uint8): 255.0}.get(raw.dtype, 1.0)
    sk = (raw[..., 0] if raw.ndim == 3 else raw).astype(np.float32) / scale
    if sk.shape != shape:
        sk = cv2.resize(sk, shape[::-1], interpolation=cv2.INTER_LINEAR)
    return np.clip(sk, 0.0, 1.0)


def source_irradiance_measured(normals, C):
    """E_source(n) from this shot's fitted SH coefficients -- the call composite_ratio makes."""
    from sh_project import sh_basis
    Y = sh_basis(normals[..., 0], normals[..., 1], normals[..., 2])
    return np.maximum(np.einsum('hwk,kc->hwc', Y, C), 1e-4)


def unit_from_angles(theta, phi):
    """World unit vector from elevation `theta` (0 = horizon, +pi/2 = straight up) and azimuth."""
    ct = np.cos(theta)
    return np.array([ct * np.sin(phi), ct * np.cos(phi), np.sin(theta)], np.float64)


def kk_lobe(T, L, V, p, f0):
    """sin(T, H)^p with a Schlick Fresnel and the Blinn-Phong normalisation. numpy, (N,) out.

    The normalisation constant is a SURFACE lobe's, borrowed for a fibre lobe that is really
    one-dimensional; it is an approximation, and the magnitude it implies is not asserted --
    the fitted `A` absorbs it, and the composite reports the fitted peak beside the model's.
    """
    H = L + V
    H = H / np.maximum(np.linalg.norm(H, axis=-1, keepdims=True), 1e-6)
    th = np.clip(np.sum(T * H, -1), -1.0, 1.0)
    sin_th = np.sqrt(np.maximum(1.0 - th * th, 0.0))
    vdh = np.clip(np.sum(V * H, -1), 0.0, 1.0)
    F = f0 + (1.0 - f0) * (1.0 - vdh) ** 5
    return F * (p + 2.0) / (8.0 * np.pi) * np.power(np.maximum(sin_th, 1e-6), p)


def fit_room_light(samples, f0, p_bounds=(8.0, 400.0), seed_grid=12):
    """Solve (omega, A, p, rho) once for the clip. Returns the fit and how well it describes her.

    Robust (`soft_l1`): the region contains materials the model does not describe -- the scalp at
    the parting, an earring, a stray hand -- and a handful of outliers must not steer a five-
    parameter solution. The direction is seeded from a coarse grid first, because the lobe is
    badly non-convex in omega and a gradient step from one seed lands in that seed's own minimum.
    """
    from scipy.optimize import least_squares
    y, E, T, V = samples['lum'], samples['E'], samples['T'], samples['V']

    def spec_of(par):
        theta, phi, logA, logp, _rho = par
        L = np.broadcast_to(unit_from_angles(theta, phi), T.shape)
        return np.exp(logA) * kk_lobe(T, L, V, np.exp(logp), f0)

    def model(par):
        return par[4] * E + spec_of(par)

    rho0 = float(np.percentile(y, 20) / max(np.median(E), 1e-6))
    best = None
    for theta in np.linspace(np.radians(5.0), np.radians(88.0), max(seed_grid // 2, 2)):
        for phi in np.linspace(-np.pi, np.pi, seed_grid, endpoint=False):
            par = np.array([theta, phi, np.log(max(np.percentile(y, 99), 1e-4)), np.log(60.0), rho0])
            c = float(np.mean((model(par) - y) ** 2))
            if best is None or c < best[0]:
                best = (c, par)
    lo = np.array([np.radians(-20.0), -2.0 * np.pi, np.log(1e-6), np.log(p_bounds[0]), 0.0])
    hi = np.array([np.radians(89.0), 2.0 * np.pi, np.log(1e3), np.log(p_bounds[1]), 10.0])
    sol = least_squares(lambda par: model(par) - y, np.clip(best[1], lo, hi), bounds=(lo, hi),
                        loss='soft_l1',
                        f_scale=float(np.percentile(np.abs(y - np.median(y)), 60) + 1e-5),
                        max_nfev=400)
    theta, phi, logA, logp, rho = sol.x
    pred, spec = model(sol.x), spec_of(sol.x)
    ss_res = float(np.sum((pred - y) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    top = y >= np.percentile(y, 90)
    return {'direction_world': unit_from_angles(theta, phi).tolist(),
            'elevation_deg': float(np.degrees(theta)), 'azimuth_deg': float(np.degrees(phi) % 360.0),
            'amplitude': float(np.exp(logA)), 'exponent': float(np.exp(logp)),
            'lobe_half_width_deg': float(np.degrees(np.arcsin(min(np.sqrt(2 * np.log(2) / np.exp(logp)), 1.0)))),
            'hair_albedo': float(rho), 'f0': f0, 'n_samples': int(len(y)),
            'r2': 1.0 - ss_res / max(ss_tot, 1e-12),
            'rel_rms': float(np.sqrt(np.mean((pred - y) ** 2)) / max(float(np.mean(y)), 1e-9)),
            'spec_share_p50': float(np.median(spec / np.maximum(pred, 1e-9))),
            'spec_share_top_decile': float(np.mean((spec / np.maximum(pred, 1e-9))[top])),
            'success': bool(sol.success), 'cost': float(sol.cost),
            'axes': '+X right, +Y away from camera, +Z up (Blender world, stress_proxy.py)'}


def frame_fields(shot, f, topo, pts_all, args, src_C, cam_pos):
    """Everything the model needs at one frame, at plate resolution."""
    p = pts_all[f]
    c, rad, chin = hr.face_frame(p, topo)
    alpha = cv2.imread(str(shot.alpha(f)), -1)
    alpha = np.clip(alpha[..., 0] if alpha.ndim == 3 else alpha, 0.0, 1.0)
    h, w = alpha.shape
    skin = read_skin(shot, f, (h, w))
    hair, cuts = hr.hair_mask(alpha, skin, p, topo, radius=args.hair_radius,
                             skin_grow=args.skin_grow, parts=True)
    fg = cv2.imread(str(shot.fg(f)), -1)[..., ::-1].astype(np.float32)
    lum = fg @ LUMA
    nrm = np.load(shot.relight / 'normals' / ('%06d.npz' % f))['normal'].astype(np.float32)
    if nrm.shape[:2] != (h, w):
        nrm = cv2.resize(nrm, (w, h), interpolation=cv2.INTER_LINEAR)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-6)
    E = source_irradiance_measured(nrm, src_C) @ LUMA
    t2, coh = hr.structure_tangent(lum, hair)
    T = hr.tangent_world(t2, nrm)
    # The view direction comes from the proxy's world surface, the same map the bulb tour's own
    # geometry uses, so the removal and the replacement share one camera model rather than two.
    P = position_map(shot, f, (h, w))
    V = cam_pos[None, None].astype(np.float32) - P
    V = V / np.maximum(np.linalg.norm(V, axis=-1, keepdims=True), 1e-6)
    return {'pts': p, 'rad': rad, 'alpha': alpha, 'hair': hair, 'cuts': cuts, 'fg': fg,
            'lum': lum, 'nrm': nrm, 'E': E, 't2': t2, 'coh': coh, 'T': T, 'V': V, 'shape': (h, w)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_shot_argument(ap)
    ap.add_argument('--name', default='r45_sheen')
    ap.add_argument('--start', type=int, default=0, help='first frame index')
    ap.add_argument('--length', type=int, default=0, help='0 = every frame with a landmark')
    ap.add_argument('--envelope-sigma', type=float, default=0.022,
                    help='the ONE scale that separates the band from the strands, in FACE RADII. '
                         'Below it is strand texture and is kept; above it is the band, which is '
                         'what the removal acts on')
    ap.add_argument('--baseline-pct', type=float, default=20.0,
                    help='the region-wide dark-hair percentile the band is measured against')
    ap.add_argument('--level-lo', type=float, default=1.5,
                    help='envelope over the dark hair below which nothing is removed')
    ap.add_argument('--level-hi', type=float, default=4.0,
                    help='envelope over the dark hair at and above which the whole excess goes')
    ap.add_argument('--window', type=float, default=0.5,
                    help='temporal median of the envelope, in SECONDS, symmetric, head-stabilised. '
                         '0 disables it')
    ap.add_argument('--chroma-full', type=float, default=20.0,
                    help='degrees from the measured plate illuminant: full removal at or below. '
                         'The sheen and her dark hair both sit at 8-9 deg, her skin at 30')
    ap.add_argument('--chroma-none', type=float, default=45.0,
                    help='degrees: no removal at or above. Wide on purpose -- a tight gate punched '
                         'a hole in the weight over her parting, which is scalp at skin chroma, and '
                         'left it bright over a darkened crown ("a gap like she is bald")')
    ap.add_argument('--feather', type=float, default=0.012, help='edge feather, in FACE RADII')
    ap.add_argument('--hair-radius', type=float, default=1.55, help='region cut, in face radii')
    ap.add_argument('--skin-grow', type=float, default=0.025, help='skin dilation, in face radii')
    ap.add_argument('--gain', type=float, default=1.0, help='scale on the sheen written out')
    ap.add_argument('--fit-frames', type=int, default=8,
                    help='frames sampled for the DIAGNOSTIC room-light fit; 0 skips it')
    ap.add_argument('--fit-samples', type=int, default=60000, help='hair pixels pooled into the fit')
    ap.add_argument('--fit-coherence', type=float, default=0.45,
                    help='only fit where the strand direction is trustworthy')
    ap.add_argument('--f0', type=float, default=0.0465, help='cuticle reflectance, IOR 1.55')
    args = ap.parse_args()
    shot = from_args(args)

    topo = json.loads((shot.relight / 'face_landmarks.json').read_text())['topology']
    with np.load(shot.relight / 'face_landmarks.npz') as z:
        pts_all, found = z['points'], z['found']
    sep = json.loads((shot.log('specular_separation.json')).read_text())
    g_rgb = np.array(sep['illuminant']['rgb'], np.float32)
    g_hat = (g_rgb / max(float(g_rgb @ LUMA), 1e-6)).astype(np.float32)   # unit LUMINANCE
    g_unit = (g_rgb / np.linalg.norm(g_rgb)).astype(np.float32)
    slp = shot.log('source_light.json')
    if not slp.exists():
        raise SystemExit('No logs/source_light.json. Run estimate_source_light.py first -- this '
                         "stage will not invent a second model of the plate's own key.")
    src_C = np.array(json.loads(slp.read_text())['sh_coefficients_rgb'], np.float32)
    cam_pos = camera_world(shot)

    frames = [f for f in range(max(args.start, 0), len(found)) if found[f] and shot.fg(f).exists()]
    if args.length > 0:
        frames = frames[:args.length]
    if not frames:
        raise SystemExit('no frames with landmarks and a keyed foreground from %d' % args.start)
    fps = float(Fraction(str(shot.frame_rate)))
    win = int(round(args.window * fps)) if args.window > 0 else 0
    if win > 0:
        win += (win + 1) % 2                                     # odd, so the window is symmetric
    half = win // 2

    root = shot.composite / args.name
    for sub in ('sheen', 'field', 'weight'):
        (root / sub).mkdir(parents=True, exist_ok=True)
    print('hair sheen: %d frames, temporal window %d frames (%.2f s) -> %s (%s)'
          % (len(frames), win, win / fps if fps else 0.0, root, json.dumps(gpu_image.describe())),
          flush=True)

    # ---- pass 1: the band's envelope per frame, at quarter resolution -----------------------
    # The envelope is a low-pass by construction, so quarter resolution loses nothing and makes a
    # symmetric temporal median over the whole clip cost a few hundred MB instead of gigabytes.
    Q = 4                 # the envelope is a low-pass, so quarter resolution loses nothing
    QT = 2                # the tangent field is finer -- half resolution
    env_q, tan_q, base_l, meta = {}, {}, {}, {}
    t0 = time.time()
    for f in frames:
        d = frame_fields(shot, f, topo, pts_all, args, src_C, cam_pos)
        h, w = d['shape']
        hair, lum, rad = d['hair'], d['lum'], d['rad']
        mm = hair.astype(np.float32)
        sig = max(args.envelope_sigma * rad, 0.6)
        env = (gpu_image.gaussian_blur(lum * mm, sig)
               / np.maximum(gpu_image.gaussian_blur(mm, sig), 1e-4)).astype(np.float32)
        env_q[f] = cv2.resize(env, (w // Q, h // Q), interpolation=cv2.INTER_AREA)
        # The tangent gets the same treatment as the envelope, for the same reason: it is
        # estimated per frame and the highlight rides it, so its jitter becomes the highlight's.
        # A direction has no head or tail, so it is carried as the DOUBLE-ANGLE vector (the only
        # representation in which averaging two tangents 179 degrees apart gives the right answer
        # instead of their perpendicular), weighted by the tensor's own coherence.
        a2 = 2.0 * np.arctan2(d['t2'][..., 1], d['t2'][..., 0])
        cvec = np.stack([np.cos(a2), np.sin(a2)], -1) * d['coh'][..., None]
        tan_q[f] = cv2.resize(np.concatenate([cvec, d['coh'][..., None]], -1),
                              (w // QT, h // QT), interpolation=cv2.INTER_AREA)
        base_l[f] = max(float(np.percentile(lum[hair], args.baseline_pct)) if hair.any() else 1e-5, 1e-5)
        meta[f] = {'rad': rad, 'pts': d['pts'][:, :2].astype(np.float32), 'shape': (h, w)}
        if len(env_q) % 50 == 0:
            print('  envelope %d/%d  %.1f s' % (len(env_q), len(frames), time.time() - t0), flush=True)

    # ---- the DIAGNOSTIC room-light fit -----------------------------------------------------
    fit = None
    if args.fit_frames > 0:
        fit_idx = [frames[i] for i in np.linspace(0, len(frames) - 1,
                                                  min(args.fit_frames, len(frames))).astype(int)]
        take_n = max(args.fit_samples // max(len(fit_idx), 1), 1)
        rng = np.random.default_rng(0)
        cols = {'lum': [], 'E': [], 'T': [], 'V': []}
        for f in fit_idx:
            d = frame_fields(shot, f, topo, pts_all, args, src_C, cam_pos)
            m = d['hair'] & (d['alpha'] > 0.95) & (d['coh'] > args.fit_coherence)
            idx = np.flatnonzero(m.ravel())
            if idx.size == 0:
                continue
            take = rng.choice(idx, size=min(take_n, idx.size), replace=False)
            h, w = d['shape']
            # The fit is scored against the ENVELOPE, not the raw pixel: a smooth lobe cannot
            # predict strand-level structure, so scoring it against one only measures strands.
            env = cv2.resize(env_q[f], (w, h), interpolation=cv2.INTER_LINEAR)
            cols['lum'].append(env.ravel()[take])
            cols['E'].append(d['E'].ravel()[take])
            cols['T'].append(d['T'].reshape(-1, 3)[take])
            cols['V'].append(d['V'].reshape(-1, 3)[take])
        if cols['lum']:
            samples = {k: np.concatenate(v, 0).astype(np.float64) for k, v in cols.items()}
            print('room-light diagnostic: fitting on %d envelope samples from %d frames...'
                  % (len(samples['lum']), len(fit_idx)), flush=True)
            fit = fit_room_light(samples, args.f0)
            fit['fit_frames'] = fit_idx
            fit['role'] = ('DIAGNOSTIC ONLY -- it says where the room light was, not what is '
                           'subtracted. A single fibre lobe explains only part of the band')
            print(json.dumps({k: v for k, v in fit.items() if k != 'fit_frames'}, indent=2), flush=True)

    # ---- pass 2: temporal median of the envelope, then write --------------------------------
    per, prev, t0 = [], None, time.time()
    for i, f in enumerate(frames):
        d = frame_fields(shot, f, topo, pts_all, args, src_C, cam_pos)
        h, w = d['shape']
        hair, fg, lum, rad, alpha = d['hair'], d['fg'], d['lum'], d['rad'], d['alpha']
        hq, wq = h // Q, w // Q

        # Symmetric median in the head's own frame. Neighbours are warped onto THIS frame by a
        # similarity fit to all landmarks; a one-sided window would lag the band across a head
        # turn, which is round 37's rule for the eye's edge shading applied to the same problem.
        stack, used, warps = [env_q[f]], [f], []
        for j in range(1, half + 1):
            for k in (i - j, i + j):
                if not (0 <= k < len(frames)):
                    continue
                g = frames[k]
                if meta[g]['shape'] != (h, w):
                    continue
                M, _ = cv2.estimateAffinePartial2D(meta[g]['pts'], meta[f]['pts'], method=cv2.LMEDS)
                if M is None:
                    continue
                Mq = M.copy()
                Mq[:, 2] /= Q                        # the same similarity at quarter resolution
                stack.append(cv2.warpAffine(env_q[g], Mq, (wq, hq), flags=cv2.INTER_LINEAR))
                used.append(g)
                warps.append((g, M))
        env = cv2.resize(np.median(np.stack(stack, 0), 0).astype(np.float32), (w, h),
                         interpolation=cv2.INTER_LINEAR)

        # The tangent, over the same window. Warping a DIRECTION field is not warping an image:
        # the similarity rotates the picture, so each double-angle vector has to be rotated by
        # TWICE that angle as well, or a filtered head turn quietly combs the strands the wrong
        # way. Summed rather than medianed, because the sum of coherence-weighted unit vectors is
        # the circular mean and its length is the agreement -- which is exactly the confidence
        # the composite then weights the lobe by.
        tsum = tan_q[f].copy()
        for g, M in warps:
            tq = tan_q[g]
            hq2, wq2 = tq.shape[:2]
            Mt = M.copy()
            Mt[:, 2] /= QT
            wq = cv2.warpAffine(tq, Mt, (wq2, hq2), flags=cv2.INTER_LINEAR)
            phi = 2.0 * np.arctan2(M[1, 0], M[0, 0])
            cs, sn = np.cos(phi), np.sin(phi)
            tsum[..., 0] += wq[..., 0] * cs - wq[..., 1] * sn
            tsum[..., 1] += wq[..., 0] * sn + wq[..., 1] * cs
            tsum[..., 2] += wq[..., 2]
        tf = cv2.resize(tsum, (w, h), interpolation=cv2.INTER_LINEAR)
        ang2 = np.arctan2(tf[..., 1], tf[..., 0])
        t2f = np.stack([np.cos(0.5 * ang2), np.sin(0.5 * ang2)], -1).astype(np.float32)
        coh_f = np.clip(np.hypot(tf[..., 0], tf[..., 1]) / np.maximum(tf[..., 2], 1e-6),
                        0.0, 1.0) * np.clip(tf[..., 2] / max(len(warps) + 1, 1), 0.0, 1.0)
        d['t2'], d['coh'] = t2f, coh_f.astype(np.float32)

        b = float(np.median([base_l[g] for g in used]))
        # lum = env * tex exactly, so removing (env - b) * tex leaves b * tex: the band goes and
        # the strand structure stays, scaled onto the dark hair.
        tex = np.clip(lum / np.maximum(env, 1e-5), 0.0, TEX_SCALE)
        fade = smoothstep(env / b, args.level_lo, args.level_hi)
        S_lum = np.maximum(env - b, 0.0) * fade * tex

        u = fg / np.maximum(np.linalg.norm(fg, axis=-1, keepdims=True), 1e-9)
        ang = np.degrees(np.arccos(np.clip(u @ g_unit, -1.0, 1.0)))
        w_chroma = 1.0 - smoothstep(ang, args.chroma_full, args.chroma_none)
        # Feather only the ARBITRARY edges and let the matte carry the silhouette: the brightest
        # sheen sits on the fractional arc, and feathering there is what left it shining.
        wt = gpu_image.gaussian_blur(d['cuts'].astype(np.float32), max(args.feather * rad, 0.5))
        weight = (np.clip(wt, 0.0, 1.0) * alpha * hair * w_chroma).astype(np.float32)

        # Taken along the PIXEL's OWN chroma, not the illuminant's. Round 44 SS5 measured the band
        # and the dark hair beside it at the same 8-9 deg from the illuminant, so on hair the two
        # are interchangeable -- but on her PARTING, which is scalp at 30 deg, subtracting a neutral
        # left a red streak down the middle of the crown. A per-pixel chroma cannot shift a hue by
        # construction. Clamped to the photograph so nothing can go negative.
        frac = np.clip(S_lum / np.maximum(lum, 1e-6), 0.0, 1.0) * weight * args.gain
        S = np.minimum(fg * frac[..., None], fg).astype(np.float32)

        angle = (np.degrees(np.arctan2(d['t2'][..., 1], d['t2'][..., 0])) % 180.0).astype(np.float32)
        # The strand texture handed on is the one this removal divided out, so the composite's new
        # highlight can carry exactly the structure the old one did.
        cv2.imwrite(str(root / 'sheen' / ('%06d.png' % f)),
                    (np.clip(S, 0.0, 1.0)[..., ::-1] * 65535.0 + 0.5).astype(np.uint16))
        field = np.stack([angle / 180.0, tex / TEX_SCALE, np.clip(d['coh'], 0.0, 1.0)], -1)
        cv2.imwrite(str(root / 'field' / ('%06d.png' % f)),
                    (np.clip(field, 0, 1)[..., ::-1] * 65535.0 + 0.5).astype(np.uint16))
        cv2.imwrite(str(root / 'weight' / ('%06d.png' % f)),
                    (np.clip(weight, 0, 1) * 255.0 + 0.5).astype(np.uint8))

        m = hair & (weight > 0.5)
        crown = hr.crown_mask(hair) & (weight > 0.5)
        rec = {'frame': f, 'face_radius_px': rad, 'hair_px': int(hair.sum()), 'acting_px': int(m.sum()),
               'dark_hair_level': b, 'window_frames': len(stack),
               'removed_energy_frac_hair': float(S[m].sum() / max(fg[m].sum(), 1e-9)) if m.any() else 0.0,
               'removed_energy_frac_crown': float(S[crown].sum() / max(fg[crown].sum(), 1e-9)) if crown.any() else 0.0,
               'clipped_frac': float((S[m] >= fg[m] - 1e-6).mean()) if m.any() else 0.0,
               'coherence_p50': float(np.median(d['coh'][hair])) if hair.any() else 0.0,
               # What the ROOM's band implies about the hair's specular reflectance: a highlight
               # is f_r x pi x E, and E_room on these pixels is already measured, so S / (pi E) is
               # f_r at the geometry the band was made at -- which is at or near the lobe's peak,
               # since that is what a band IS. The composite's --hair-specular default is this
               # number over the model's own peak, so the replacement is calibrated to the thing
               # it replaces rather than to a chosen gain.
               'fr_peak_measured': float(np.percentile(S_lum[m] / np.maximum(np.pi * d['E'][m], 1e-6), 99)) if m.any() else 0.0,
               'fg_sha256': hashlib.sha256(shot.fg(f).read_bytes()).hexdigest()}
        # Temporal, with the plate as its own control: the step of the DE-SHEENED luminance beside
        # the step of the plate's over the same pixels, head-registered. A removal that adds
        # flicker shows up as the first number standing above the second.
        out_lum = np.maximum((fg - S) @ LUMA, 1e-5)
        if prev is not None and prev['shape'] == (h, w):
            M, _ = cv2.estimateAffinePartial2D(meta[f]['pts'], prev['pts'], method=cv2.LMEDS)
            if M is not None:
                both = prev['mask'] & (cv2.warpAffine(m.astype(np.float32), M, (w, h),
                                                      flags=cv2.INTER_NEAREST) > 0.5)
                if both.any():
                    for key, cur, ref in (('step', out_lum, prev['out']),
                                          ('plate_step', np.maximum(lum, 1e-5), prev['lum'])):
                        cw = cv2.warpAffine(np.log2(cur).astype(np.float32), M, (w, h),
                                            flags=cv2.INTER_LINEAR)
                        dd = np.abs(cw - ref)[both]
                        rec['%s_p50_stops' % key] = float(np.median(dd))
                        rec['%s_p95_stops' % key] = float(np.percentile(dd, 95))
                        rec['%s_max_stops' % key] = float(dd.max())
        prev = {'out': np.log2(out_lum), 'lum': np.log2(np.maximum(lum, 1e-5)), 'mask': m,
                'pts': meta[f]['pts'], 'shape': (h, w)}
        per.append(rec)
        if len(per) % 25 == 0:
            print('  write %d/%d  %.1f s' % (len(per), len(frames), time.time() - t0), flush=True)

    def agg(key):
        v = [r[key] for r in per if key in r]
        return {'p50': float(np.median(v)), 'p95': float(np.percentile(v, 95)),
                'max': float(np.max(v))} if v else None

    man = {'name': args.name, 'frames': frames, 'tex_scale': TEX_SCALE,
           'illuminant_rgb_unit_luma': g_hat.tolist(), 'room_light_fit': fit,
           'temporal_window_frames': win, 'fps': fps,
           'settings': vars(args) | {'shot': str(shot.dir)}, 'device': gpu_image.describe(),
           'summary': {k: agg(k) for k in ('removed_energy_frac_hair', 'removed_energy_frac_crown',
                                           'acting_px', 'clipped_frac', 'coherence_p50',
                                           'fr_peak_measured',
                                           'step_p95_stops', 'step_max_stops',
                                           'plate_step_p95_stops', 'plate_step_max_stops')},
           'per_frame': per,
           'why': "the capture room's overhead band, taken off the photograph as an ENVELOPE over "
                  'the region-wide dark hair with the strand texture kept and the envelope filtered '
                  'in time; removal only, the replacement is composite_ratio.py --hair-sheen'}
    (root / 'manifest.json').write_text(json.dumps(man, indent=2) + '\n')
    (shot.logs / ('%s.json' % args.name)).write_text(json.dumps(
        {k: v for k, v in man.items() if k != 'per_frame'}, indent=2) + '\n')
    print(json.dumps(man['summary'], indent=2))
    print('%d frames in %.1f s -> %s' % (len(per), time.time() - t0, root))


if __name__ == '__main__':
    main()
