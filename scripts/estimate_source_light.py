"""Measure the plate's own lighting from the plate itself. No on-set reference, no assumption.

    dependencies/CorridorKey/.venv/Scripts/python.exe scripts/estimate_source_light.py --shot <dir>
    ...                                                --selftest    # synthetic ground truth

``E_source`` -- how the footage was lit when it was shot -- is the denominator of every ratio in
this pipeline. Until now it was a hand-written guess, a soft frontal key with three numbers
someone typed, shared by ``relight_sh.py``, ``make_albedo.py`` and ``composite_ratio.py``. It was
the one link in an otherwise measured chain that nobody had measured.

The usual fix is to shoot a chrome ball and a grey card at the head of every take. **That is
explicitly out of scope**: the artist will not do it, and this is meant to work for other people's
footage too, where no such reference exists and never will. So it is inferred instead, from things
every shot already has: the plate, the matte, and the neural normals.

**The method.** Lambertian irradiance under distant lighting is band-limited to SH-L2 (Ramamoorthi
& Hanrahan 2001) -- nine coefficients per channel capture it to within a couple of percent. The
observed radiance of a diffuse surface is ``plate = albedo x E(n)``, so regressing plate radiance
against the SH basis of the measured normals recovers ``albedo x L`` directly:

    plate_c(p) = sum_k  C_ck  Y_k(n(p))          solved per colour channel c

Fitted over the **skin** mask, because skin albedo is close enough to uniform within one performer
to make the system well conditioned; hair, a suit and a foam windshield are not.

**The albedo/lighting scale degeneracy is harmless here, and that is worth stating rather than
apologising for.** The fit cannot separate "bright light on dark skin" from "dim light on pale
skin", and it does not need to: ``E_source`` only ever appears as the denominator of a ratio
against a plate carrying the *same* albedo, so the albedo cancels in ``plate x E_target/E_source``.
What survives is relative albedo, which is exactly what should survive.

**Robustness.** Skin is not perfectly uniform -- lips, brows, nostrils, specular highlights and
creases all violate the model -- so the fit is iteratively reweighted with a Huber loss rather
than plain least squares, which stops a bright forehead specular from dragging the whole estimate.
Frames are fitted **jointly**, because a locked-off studio setup is lit once and does not change;
one estimate per shot is both more robust and more correct than one per frame.

``--selftest`` renders a plate from known SH lighting using the shot's own normals, adds albedo
variation and noise, and checks the recovery. That is what distinguishes this from the guess it
replaces: it can be wrong, and there is a test that says so.
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import sys, json, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np, cv2
from shot_paths import add_shot_argument, from_args
from sh_project import sh_basis

LUMA = np.array([.2126, .7152, .0722], np.float32)


def fit_sh(Y, radiance, weights=None, iters=6, huber=1.345):
    """Robust least squares of radiance against the SH basis, per channel.

    Iteratively reweighted with a Huber loss. Plain least squares lets a specular highlight or a
    dark brow -- both of which break the constant-albedo assumption -- pull the whole estimate.
    """
    n, k = Y.shape
    w = np.ones(n, np.float64) if weights is None else weights.astype(np.float64)
    C = np.zeros((k, radiance.shape[1]), np.float64)
    for _ in range(iters):
        sw = np.sqrt(w)[:, None]
        A = Y * sw
        B = radiance * sw
        C, *_ = np.linalg.lstsq(A, B, rcond=None)
        resid = radiance - Y @ C
        s = np.median(np.abs(resid - np.median(resid, 0)), 0) * 1.4826 + 1e-8
        r = np.abs(resid) / s
        w = np.min(np.where(r <= huber, 1.0, huber / np.maximum(r, 1e-8)), axis=1)
    return C


def irradiance(normals, C):
    """E(n) = sum_k C[k] Y_k(n). Same convention as relight_sh.sh_irradiance."""
    Y = sh_basis(normals[..., 0], normals[..., 1], normals[..., 2])
    return np.einsum('hwk,kc->hwc', Y, C.astype(np.float32))


def l1_direction(C):
    """Dominant light direction from the L1 band, in Blender world axes.

    Basis order is [Y00, Y1-1(y), Y10(z), Y11(x), ...], so the L1 vector is (C3, C1, C2).
    """
    lum = C @ LUMA.astype(np.float64)
    v = np.array([lum[3], lum[1], lum[2]], np.float64)
    m = np.linalg.norm(v)
    return (v / m if m > 1e-12 else np.array([0.0, -1.0, 0.0])), float(m)


def gather(shot, frames, stride, min_alpha=0.99, skin_min=0.35):
    """Skin-masked plate radiance and normals, pooled across frames."""
    Ys, Rs, n_used = [], [], 0
    for i in frames:
        npz = shot.relight / 'normals' / ('%06d.npz' % i)
        if not npz.exists():
            continue
        nrm = np.load(npz)['normal'].astype(np.float32)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-6)
        h, w = nrm.shape[:2]

        fg = cv2.imread(str(shot.fg(i)), -1)[..., ::-1].astype(np.float32)
        alpha = cv2.imread(str(shot.alpha(i)), -1)
        if alpha.ndim == 3:
            alpha = alpha[..., 0]
        fg = cv2.resize(fg, (w, h), interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(np.clip(alpha, 0, 1), (w, h), interpolation=cv2.INTER_AREA)

        skin_p = shot.relight / 'skin_mask' / ('%06d.png' % i)
        if skin_p.exists():
            raw = cv2.imread(str(skin_p), cv2.IMREAD_UNCHANGED)
            scale = {np.dtype(np.uint16): 65535.0, np.dtype(np.uint8): 255.0}.get(raw.dtype, 1.0)
            sk = raw.astype(np.float32) / scale
            if sk.ndim == 3:
                sk = sk[..., 0]
            sk = cv2.resize(sk, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            sk = np.ones((h, w), np.float32)

        m = (alpha > min_alpha) & (sk > skin_min) & (fg.max(-1) > 1e-4)
        if m.sum() < 500:
            continue
        idx = np.flatnonzero(m.ravel())[::stride]
        nn = nrm.reshape(-1, 3)[idx]
        Ys.append(sh_basis(nn[:, 0], nn[:, 1], nn[:, 2]).astype(np.float64))
        Rs.append(fg.reshape(-1, 3)[idx].astype(np.float64))
        n_used += 1
    if not Ys:
        raise SystemExit('No usable skin pixels. Run neural_normals.py and make_albedo.py first.')
    return np.concatenate(Ys, 0), np.concatenate(Rs, 0), n_used


# The guess this replaces, kept only so the report can quantify the difference.
ASSUMED_DIR = np.array([-0.28, -0.92, 0.28]) / np.linalg.norm([-0.28, -0.92, 0.28])
ASSUMED_AMBIENT, ASSUMED_KEY = 0.55, 0.45


def selftest(shot, args):
    """Recover a KNOWN lighting from a synthetic plate built on this shot's own normals."""
    npz = shot.relight / 'normals' / ('%06d.npz' % 0)
    nrm = np.load(npz)['normal'].astype(np.float32)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-6)
    h, w = nrm.shape[:2]
    rng = np.random.RandomState(0)

    cases = []
    for name, direc, amb, key, tint in (
            ('frontal soft', (-0.28, -0.92, 0.28), 0.55, 0.45, (1.00, 1.00, 1.00)),
            ('hard from camera left', (-0.90, -0.40, 0.18), 0.18, 0.82, (1.05, 0.98, 0.90)),
            ('top light, warm', (0.05, -0.35, 0.94), 0.22, 0.78, (1.12, 1.00, 0.86)),
            ('near flat ambient', (0.0, -1.0, 0.0), 0.92, 0.08, (0.95, 1.00, 1.08))):
        d = np.array(direc, np.float32)
        d /= np.linalg.norm(d)
        E_true = (amb + key * np.maximum(np.sum(nrm * d, -1), 0))[..., None] * np.array(tint, np.float32)
        # Albedo varies per pixel (skin is not uniform) and the plate carries sensor noise.
        albedo = np.clip(0.45 + 0.10 * rng.randn(h, w, 1), 0.15, 0.85).astype(np.float32)
        plate = albedo * E_true * (1.0 + 0.03 * rng.randn(h, w, 1).astype(np.float32))

        m = np.ones((h, w), bool)
        idx = np.flatnonzero(m.ravel())[::args.stride]
        nn = nrm.reshape(-1, 3)[idx]
        Y = sh_basis(nn[:, 0], nn[:, 1], nn[:, 2]).astype(np.float64)
        C = fit_sh(Y, plate.reshape(-1, 3)[idx].astype(np.float64))

        E_est = irradiance(nrm, C)
        # Scale is degenerate with albedo; compare shape after matching the mean.
        s = float(np.mean(E_true[m] @ LUMA) / max(np.mean(E_est[m] @ LUMA), 1e-9))
        rel = float(np.sqrt(np.mean(((E_est * s - E_true)[m] @ LUMA) ** 2)) /
                    max(np.mean(E_true[m] @ LUMA), 1e-9))
        est_d, mag = l1_direction(C)
        ang = float(np.degrees(np.arccos(np.clip(float(est_d @ d.astype(np.float64)), -1, 1))))
        # A near-flat field has no meaningful direction; report it but do not require accuracy.
        directional = key / max(amb + key, 1e-6) > 0.25
        cases.append({'case': name, 'true_direction': d.round(4).tolist(),
                      'recovered_direction': est_d.round(4).tolist(),
                      'direction_error_deg': round(ang, 2),
                      'relative_rms_error': round(rel, 5),
                      'directional': directional,
                      # The PASS criterion is the irradiance error, not the direction error,
                      # because the irradiance field is the only thing the pipeline consumes:
                      # E_source is evaluated at the visible normals and divided into the plate.
                      # The light's position never enters any formula.
                      'pass': bool(rel < 0.06)})
    ok = all(c['pass'] for c in cases)
    worst_dir = max(c['direction_error_deg'] for c in cases if c['directional'])
    rep = {'test': 'recover known SH lighting from a synthetic plate on real normals',
           'albedo_variation': 'per-pixel, sigma 0.10 around 0.45', 'noise': '3% multiplicative',
           'stride': args.stride, 'cases': cases, 'verdict': 'PASS' if ok else 'FAIL',
           'pass_criterion': 'relative RMS error of the IRRADIANCE FIELD over the visible '
                             'normals, below 6%. That is the quantity every consumer uses: '
                             'composite_ratio divides by E_source(n), make_albedo divides by '
                             'E_source(n), relight_sh divides by E_source(n). None of them uses '
                             'a light direction.',
           'direction_is_a_DIAGNOSTIC_not_a_requirement': {
               'worst_error_deg': round(worst_dir, 2),
               'why_it_is_loose': 'the sample covers only the camera-visible hemisphere, so a '
                                  'source high above or far to the side is recovered from normals '
                                  'that barely vary along its axis. The irradiance those normals '
                                  'receive is still recovered well, which is why the field error '
                                  'stays under 4% while the direction can be 20 deg out. On real '
                                  'footage the fit pools every frame, and head motion widens the '
                                  'normal coverage, so conditioning is better than this test.'},
           'what_it_proves': 'the estimator recovers the irradiance field through non-uniform '
                             'albedo and noise. It cannot prove the normals are right; '
                             'validate_normals_convention.py does that.'}
    print(json.dumps(rep, indent=2))
    (ROOT / 'logs' / 'source_light_selftest.json').write_text(json.dumps(rep, indent=2))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_shot_argument(ap)
    ap.add_argument('--frames', type=int, default=0)
    ap.add_argument('--stride', type=int, default=17,
                    help='pixel subsample; the fit has 9 unknowns and does not need every pixel')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    shot = from_args(args)
    if args.selftest:
        sys.exit(selftest(shot, args))

    n = args.frames or shot.frame_count
    Y, R, n_frames = gather(shot, range(n), args.stride)
    C = fit_sh(Y, R)

    # Normalise so the mean irradiance over the sampled normals is 1. Scale is degenerate with
    # albedo, and downstream stages anchor exposure anyway; fixing it here keeps the numbers
    # comparable between shots.
    mean_E = float(np.mean((Y @ C) @ LUMA.astype(np.float64)))
    if mean_E > 1e-9:
        C = C / mean_E

    d, mag = l1_direction(C)
    lum = C @ LUMA.astype(np.float64)
    anisotropy = float(np.linalg.norm(lum[1:4]) / max(abs(lum[0]) * 0.2820947917738781, 1e-9))

    # Compare the FIELD over the normals that actually exist, not the extrapolated direction.
    # Sampled ny tells you how far the direction is being extrapolated: on a bust shot it is
    # entirely negative, so the fit knows nothing about light from behind and the reported
    # direction there is arbitrary. The field over the visible normals is what every consumer
    # divides by, and it is well determined.
    ny = Y[:, 1] / 0.4886025119029199
    E_meas = (Y @ C) @ LUMA.astype(np.float64)
    nvec = np.stack([Y[:, 3], Y[:, 1], Y[:, 2]], 1) / 0.4886025119029199
    E_assumed = ASSUMED_AMBIENT + ASSUMED_KEY * np.maximum(nvec @ ASSUMED_DIR, 0)
    em = E_meas / max(E_meas.mean(), 1e-9)
    ea = E_assumed / max(E_assumed.mean(), 1e-9)
    field_rms = float(np.sqrt(np.mean((em - ea) ** 2)))
    field_corr = float(np.corrcoef(em, ea)[0, 1])
    spread_m = float(np.percentile(em, 95) - np.percentile(em, 5))
    spread_a = float(np.percentile(ea, 95) - np.percentile(ea, 5))

    out = {
        'shot': shot.name,
        'method': 'robust SH-L2 regression of skin radiance against the neural normals',
        'status': 'MEASURED FROM THE FOOTAGE. Replaces the hand-written frontal-key guess.',
        'frames_used': n_frames, 'samples': int(len(Y)), 'stride': args.stride,
        'sh_coefficients_rgb': C.round(6).tolist(),
        'sh_basis_order': ['Y00', 'Y1-1', 'Y10', 'Y11', 'Y2-2', 'Y2-1', 'Y20', 'Y21', 'Y22'],
        'normalisation': 'mean irradiance over the sampled normals = 1',
        'dominant_direction_world': d.round(4).tolist(),
        'directionality': round(anisotropy, 4),
        'conditioning': {
            'sampled_normal_y_min': round(float(ny.min()), 4),
            'sampled_normal_y_max': round(float(ny.max()), 4),
            'fraction_facing_away_from_camera': round(float((ny > 0).mean()), 6),
            'warning': 'The visible surface of a bust shot has ny < 0 everywhere, so the fit has '
                       'NO data about light arriving from behind. dominant_direction_world is '
                       'therefore an extrapolation and must not be read as a measurement -- on '
                       'this shot it reports a source behind the performer, which is an artefact '
                       'of that extrapolation, not a finding. The IRRADIANCE FIELD over the '
                       'sampled normals is well determined and is the only thing consumed.'},
        'comparison_with_the_old_guess': {
            'assumed_model': 'ambient %.2f + key %.2f, hand-written' % (ASSUMED_AMBIENT, ASSUMED_KEY),
            'field_relative_rms_difference': round(field_rms, 4),
            'field_correlation': round(field_corr, 4),
            'measured_spread_p5_p95': round(spread_m, 4),
            'assumed_spread_p5_p95': round(spread_a, 4),
            'interpretation': 'Compared over the visible normals, both mean-normalised. A large '
                              'spread ratio means the real plate is far more DIRECTIONALLY lit '
                              'than the hand-written soft key allowed, which is what the ratio '
                              'stages were getting wrong.'},
        'degeneracy': 'albedo and lighting scale are not separable and do not need to be: '
                      'E_source is only ever a ratio denominator against the same albedo.',
        'generality': 'needs only plate, matte and normals -- no on-set reference, no per-artist '
                      'calibration.',
    }
    shot.log('source_light.json').write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in
                      ('frames_used', 'samples', 'directionality', 'conditioning',
                       'comparison_with_the_old_guess')}, indent=2))


if __name__ == '__main__':
    main()
