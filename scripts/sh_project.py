"""Project baked light probes to second-order spherical harmonics.

    dependencies/CorridorKey/.venv/Scripts/python.exe scripts/sh_project.py --shot <dir>

Turns each 128x64 probe render into 9 coefficients per channel -- 27 floats -- that evaluate
to true incident irradiance for any surface normal, including global illumination, coloured
bounce, area-light solid angle and static occlusion. A Lambertian BRDF is a low-pass filter
retaining ~99% of its energy in bands 0-2 (Ramamoorthi & Hanrahan 2001), so SH L2 is the
correct band-limited representation of diffuse irradiance; the flat ambient_rgb constant that
relight_controls.py uses today is band 0 alone.

SH cannot represent a hard shadow edge or the tight falloff of a very near source. Those stay
analytic (or come from ray-traced proxy passes); this supplies the environment and indirect
term that the analytic model has no way to compute at all.

The axis convention is validated, not assumed: the shot's key light sweeps through a known
azimuth, and the direction recovered from the L1 band is compared against it. A silent sign
error here produces lighting that is subtly wrong in a way no single frame reveals.
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import sys, json, argparse, math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np, cv2
from shot_paths import add_shot_argument, from_args

# Lambertian convolution constants: A_0 = pi, A_1 = 2pi/3, A_2 = pi/4.
A_L = np.array([3.141592653589793,
                2.0943951023931953, 2.0943951023931953, 2.0943951023931953,
                0.7853981633974483, 0.7853981633974483, 0.7853981633974483,
                0.7853981633974483, 0.7853981633974483])


def sh_basis(x, y, z):
    """Real SH, l <= 2, in the order used throughout this file."""
    return np.stack([
        np.full_like(x, 0.2820947917738781),          # Y00
        0.4886025119029199 * y,                       # Y1-1
        0.4886025119029199 * z,                       # Y10
        0.4886025119029199 * x,                       # Y11
        1.0925484305920792 * x * y,                   # Y2-2
        1.0925484305920792 * y * z,                   # Y2-1
        0.3153915652525200 * (3 * z * z - 1),         # Y20
        1.0925484305920792 * x * z,                   # Y21
        0.5462742152960396 * (x * x - y * y),         # Y22
    ], -1)


def directions(h, w):
    """Equirect pixel centres -> unit directions in Blender world axes (+Z up).

    theta runs 0 at +Z down to pi at -Z; phi runs -pi..pi, for an equirectangular Cycles camera
    at object rotation (90, 0, -90).

    VERIFIED, not assumed. scripts/validate_sh_convention.py bakes a single emissive panel at
    six known world directions in a black world and checks the recovered L1 direction points
    back at it. The obvious pairing (x = sin*sin, y = -sin*cos) fails: it maps +X to -Y and +Y
    to -X, a reflection across y = -x. That reflection leaves the XY diagonal invariant, so a
    diagonal-only spot check passes while every axis is 90 degrees wrong."""
    th = (np.arange(h) + 0.5) * np.pi / h
    ph = (np.arange(w) + 0.5) * 2 * np.pi / w - np.pi
    th, ph = np.meshgrid(th, ph, indexing='ij')
    d = np.stack([np.sin(th) * np.cos(ph), -np.sin(th) * np.sin(ph), np.cos(th)], -1)
    solid = (np.sin(th) * (np.pi / h) * (2 * np.pi / w))
    return d, solid


def project(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit('Cannot read probe %s' % path)
    L = img[..., :3][..., ::-1].astype(np.float64)     # BGR -> RGB radiance
    h, w = L.shape[:2]
    d, solid = directions(h, w)
    Y = sh_basis(d[..., 0], d[..., 1], d[..., 2])
    return np.einsum('hwk,hwc,hw->kc', Y, L, solid)    # (9,3) radiance coefficients


def dominant_direction(coeffs):
    """Direction of the L1 band, i.e. where the light predominantly comes from."""
    v = np.array([coeffs[3].sum(), coeffs[1].sum(), coeffs[2].sum()])   # x, y, z
    n = np.linalg.norm(v)
    return (v / n if n > 1e-12 else v), float(n)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_shot_argument(ap)
    ap.add_argument('--env', default=None, help='environment dir (defaults to the shot\'s)')
    args = ap.parse_args()
    shot = from_args(args)
    env = Path(args.env) if args.env else shot.environment
    probe_dir = env / 'probes'
    meta = json.loads((probe_dir / 'probes.json').read_text())
    frames = meta.get('baked_frames')
    names = [p['name'] for p in meta['probes']]

    out = {'scene': env.name, 'shot': shot.name,
           'technique': 'SH-L2 probe irradiance (environment + indirect), Lambertian-convolved',
           'sh_order': 2, 'coefficient_order': ['Y00', 'Y1-1', 'Y10', 'Y11', 'Y2-2', 'Y2-1', 'Y20', 'Y21', 'Y22'],
           'axes': 'Blender world, +Z up; probe camera rotation (90, 0, -90)',
           'lambertian_constants': A_L.tolist(),
           'note': 'Coefficients are ALREADY Lambertian-convolved: irradiance E(n) = sum_k C[k] * Y_k(n). Divide by pi for a unit-albedo diffuse reflectance.',
           'probe_positions': {p['name']: p['position'] for p in meta['probes']},
           'frames': {}}

    seq = frames if frames else [None]
    for f in seq:
        entry = {}
        for nm in names:
            fn = probe_dir / (('%s.exr' % nm) if f is None else ('%s_%06d.exr' % (nm, f)))
            if not fn.exists():
                continue
            C = project(fn) * A_L[:, None]             # Lambertian-convolved irradiance coeffs
            d, mag = dominant_direction(C)
            entry[nm] = {'sh': C.round(6).tolist(),
                         'dominant_direction': d.round(4).tolist(),
                         'directional_strength': round(mag, 4),
                         'ambient_rgb': (C[0] * 0.2820947917738781).round(5).tolist()}
        if entry:
            out['frames'][str(f if f is not None else 'static')] = entry

    dest = env / 'lighting_sh.json'
    dest.write_text(json.dumps(out, indent=2))

    # --- validation: does the recovered direction track the animated key sweep? ---
    anim_path = env / 'light_animation.json'
    report = {'probes': names, 'frames': len(out['frames'])}
    if anim_path.exists() and frames:
        anim = json.loads(anim_path.read_text())
        total = float(anim.get('sweep_degrees', 0.0))
        if total:
            n_f = anim['frames']
            rec, exp = [], []
            for f in frames:
                e = out['frames'].get(str(f), {}).get('head')
                if not e:
                    continue
                dx, dy, _ = e['dominant_direction']
                rec.append(math.degrees(math.atan2(dx, -dy)))
                exp.append(-total / 2 + total * (f / max(n_f - 1, 1)))
            if len(rec) > 3:
                rec_u = np.unwrap(np.radians(rec)); exp_u = np.unwrap(np.radians(exp))
                corr = float(np.corrcoef(rec_u, exp_u)[0, 1])
                err = np.degrees(np.abs(((rec_u - exp_u) + np.pi) % (2 * np.pi) - np.pi))
                report['sweep_validation'] = {
                    'expected_azimuth_range_deg': [round(min(exp), 1), round(max(exp), 1)],
                    'recovered_azimuth_range_deg': [round(min(rec), 1), round(max(rec), 1)],
                    'correlation': round(corr, 4),
                    'median_abs_error_deg': round(float(np.median(err)), 2),
                    'verdict': ('axis convention CONSISTENT' if corr > 0.9
                                else 'MISMATCH - check the probe camera rotation and the direction mapping')}
    out['validation'] = report
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
