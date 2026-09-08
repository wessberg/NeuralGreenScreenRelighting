"""Relight the plate from baked SH light probes.

    dependencies/CorridorKey/.venv/Scripts/python.exe scripts/relight_sh.py --shot <dir>

Replaces the constant ambient_rgb and analytic light list of relight_controls.py with real
measured irradiance: three SH-L2 probes per frame, baked in Cycles at head, chest and waist,
blended by each pixel's world height. That captures direction, colour, falloff, global
illumination and coloured bounce, none of which the analytic model could compute.

The gain formulation drops every magic constant:

    gain(n) = E_target(n) / E_source(n)  x  exposure_anchor
    g       = log2(gain)
    g       = s * tanh(g / s)                        s = 1.5 stops
    gain    = 2**g

tanh in log space replaces clip(0.35, 1.5): it is monotone, symmetric in stops, has unit slope
at g = 0 so small corrections pass through untouched, and saturates smoothly instead of leaving
a hard edge inside the gain map. The old `0.30 + 0.70*gain` damping is gone -- it compressed
every correction by 30% and was the main reason the shot only moved 6.1%.

Normals come from ``scripts/neural_normals.py`` (Sapiens2) when that stage has been run, and
fall back to the depth gradient otherwise. Round 1 predicted this would be the fix for the key
not visibly sweeping across the face. Measured, it is not: the two normal fields drive the
shading centroid across 50% and 60% of the face width respectively, so the depth gradient was
never the thing throwing direction away. Neural normals are still the better input -- they are
coherent rather than noisy, they extinguish correctly when lit from behind, and they carry real
facial structure -- but the ceiling on directional modelling is the ratio formulation itself,
not its normals. See rounds/round2.md.

E_source, the plate's own lighting, is the weak link in any ratio method and is ASSUMED here:
a soft frontal studio key, which is what the plate looks like. Measuring it properly means
shooting a chrome ball and an 18% grey ball at the head of the take. The exposure anchor is a
stated rule, not a knob: the alpha-weighted mean gain over the skin mask on the anchor frame
is forced to 1.0, and the value is logged.
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import sys, json, argparse, math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np, cv2
from shot_paths import add_shot_argument, from_args
from sh_project import sh_basis

W = H = None          # set per shot from the plate aspect; see Shot.working_resolution
LUMA = np.array([.2126, .7152, .0722], np.float32)


def blur(x, s):
    return cv2.GaussianBlur(x, (0, 0), s)


def normals_from_depth(depth, alpha, cal, fx_small, cam_h, subject_distance):
    """Depth-gradient normals, restricted to the foreground.

    Foreground-only normalised convolution (blur(d*mask)/blur(mask)) avoids a false normal
    discontinuity at the matte edge -- that part of the original implementation was right and
    is kept. Unlike the original, X and Z are back-projected with PER-PIXEL depth rather than
    a single scalar, so the geometry is a real pinhole reconstruction.
    """
    mask = (alpha > .6).astype(np.float32)
    d = blur(depth * mask, 3) / np.maximum(blur(mask, 3), 1e-6)
    # Affine-invariant disparity -> metric depth. Scale/shift come from the calibration.
    Zc = 1.0 / np.maximum(cal['a'] * d + cal['b'], 1e-4)
    Zc = np.clip(Zc, subject_distance - 1.0, subject_distance + 1.0)
    yy, xx = np.indices((H, W), dtype=np.float32)
    X = (xx + .5 - W / 2) * Zc / fx_small
    Zw = cam_h - (yy + .5 - H / 2) * Zc / fx_small
    Yw = Zc - subject_distance                      # +Y is away from camera
    gy, gx = np.gradient(Yw, edge_order=1)
    px = np.maximum(np.gradient(X, axis=1), 1e-6)
    pz = np.minimum(np.gradient(Zw, axis=0), -1e-6)
    # For a surface Y = f(X, Z) the tangents are (1, fX, 0) and (0, fZ, 1), so the normal is
    # (fX, -1, fZ) -- facing -Y, toward the camera. fX = (dY/dcol)/(dX/dcol), likewise for Z.
    nx = np.clip(gx / px, -3.0, 3.0)
    nz = np.clip(gy / pz, -3.0, 3.0)
    n = np.stack([nx, -np.ones_like(nx), nz], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return n.astype(np.float32), np.stack([X, Yw, Zw], -1).astype(np.float32)


def load_neural_normals(shot, i):
    """Sapiens2 normals for frame i, already in Blender world axes, or None if not baked.

    Returned at this module's working resolution and re-normalised: the npz is float16, and
    half precision is not accurate enough to leave a unit vector unit after interpolation.
    """
    p = shot.relight / 'normals' / ('%06d.npz' % i)
    if not p.exists():
        return None
    n = np.load(p)['normal'].astype(np.float32)
    if n.shape[:2] != (H, W):
        n = cv2.resize(n, (W, H), interpolation=cv2.INTER_LINEAR)
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)


def sh_irradiance(normals, coeffs):
    """E(n) = sum_k C[k] Y_k(n); coefficients are already Lambertian-convolved."""
    Y = sh_basis(normals[..., 0], normals[..., 1], normals[..., 2])      # (H,W,9)
    return np.einsum('hwk,kc->hwc', Y, coeffs)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_shot_argument(ap)
    ap.add_argument('--anchor-frame', type=int, default=0)
    ap.add_argument('--stops', type=float, default=1.5, help='tanh saturation, in stops')
    ap.add_argument('--frames', type=int, default=0)
    ap.add_argument('--normals', choices=('auto', 'neural', 'depth'), default='auto',
                    help="'auto' uses Sapiens2 normals when relight/normals/ is populated")
    args = ap.parse_args()
    shot = from_args(args)
    global W, H
    W, H = shot.working_resolution
    env = shot.environment
    lighting = json.loads((env / 'lighting_sh.json').read_text())
    probe_pos = lighting['probe_positions']
    order = sorted(probe_pos, key=lambda k: -probe_pos[k][2])            # high to low
    heights = [probe_pos[k][2] for k in order]

    cfg = shot.config
    cam = cfg.get('camera') or {}
    lens, sensor = float(cam.get('lens_mm', 50)), float(cam.get('sensor_width_mm', 36))
    dist = float(cam.get('actor_distance_m', 1.30))
    cam_h = float(cam.get('height_m', 1.55))
    src_w = cfg.get('source_resolution', [1920, 1080])[0]
    fx_small = (src_w * lens / sensor) * (W / src_w)

    n = args.frames or shot.frame_count

    # Depth calibration: fit 1/Z = a*d + b so the face centroid sits at the assumed distance
    # and the torso has a plausible depth spread. Fitted ONCE per shot and held, which is what
    # stops the body's apparent relief breathing frame to frame.
    track = json.loads((shot.relight / 'face_track.json').read_text())
    rows = [r for r in track['per_frame'] if r.get('found')]
    mid = rows[len(rows) // 2]
    d0 = np.load(shot.depth_npy(len(rows) // 2))
    a0 = np.clip(cv2.resize(cv2.imread(str(shot.alpha(len(rows) // 2)), -1), (W, H)), 0, 1)
    fu, fv = mid['face_centroid_px']
    fx_px, fy_px = int(fu * W / src_w), int(fv * H / cfg.get('source_resolution', [1920, 1080])[1])
    d_face = float(np.median(d0[max(fy_px - 6, 0):fy_px + 7, max(fx_px - 6, 0):fx_px + 7]))
    body = d0[a0 > .99]
    d_lo, d_hi = np.percentile(body, [10, 90]) if body.size else (d_face - 1, d_face + 1)
    TORSO_SPREAD_M = 0.26                       # 10-90 depth spread of a clothed adult torso
    a_fit = (1.0 / (dist - TORSO_SPREAD_M / 2) - 1.0 / (dist + TORSO_SPREAD_M / 2)) / max(d_hi - d_lo, 1e-6)
    b_fit = 1.0 / dist - a_fit * d_face
    cal = {'a': float(a_fit), 'b': float(b_fit), 'd_face': d_face,
           'torso_spread_m': TORSO_SPREAD_M, 'method': 'two constraints, two unknowns: face anchor + torso depth spread'}

    # E_source: assumed soft frontal studio key. Documented as an assumption, not a measurement.
    l_src = np.array([-0.28, -0.92, 0.28], np.float32); l_src /= np.linalg.norm(l_src)
    SRC_AMBIENT, SRC_KEY = 0.55, 0.45

    out = shot.controls
    out.mkdir(parents=True, exist_ok=True)
    stats = []
    anchor = None
    normal_source = 'depth_gradient'
    for i in range(n):
        alpha = np.clip(cv2.resize(cv2.imread(str(shot.alpha(i)), -1), (W, H), interpolation=cv2.INTER_AREA), 0, 1)
        depth = np.load(shot.depth_npy(i))
        nrm_d, pos = normals_from_depth(depth, alpha, cal, fx_small, cam_h, dist)
        # pos is still needed whichever source wins: the probe height blend reads world Z from
        # the depth back-projection, which the neural normals do not provide.
        nrm = nrm_d
        if args.normals != 'depth':
            nn = load_neural_normals(shot, i)
            if nn is not None:
                nrm = nn
                normal_source = 'sapiens2'
            elif args.normals == 'neural':
                raise SystemExit(
                    'No %s. Run scripts/neural_normals.py --shot %s first, or pass '
                    '--normals depth.' % (shot.relight / 'normals', shot.name))

        key = str(i) if str(i) in lighting['frames'] else 'static'
        fr = lighting['frames'].get(key) or lighting['frames'][next(iter(lighting['frames']))]
        # Piecewise-linear height blend, clamped outside the probe span. A naive accumulate
        # double-counts the lowest probe, so each pixel is assigned to exactly one segment.
        z = pos[..., 2]
        Es = [sh_irradiance(nrm, np.array(fr[k]['sh'], np.float32)) for k in order]
        E = Es[-1].copy()                                   # below the lowest probe
        for k in range(len(order) - 1):
            hi, lo = heights[k], heights[k + 1]
            t = np.clip((z - lo) / max(hi - lo, 1e-6), 0, 1)[..., None]
            inside = ((z >= lo) & (z <= hi))[..., None]
            E = np.where(inside, Es[k] * t + Es[k + 1] * (1 - t), E)
        E = np.where((z > heights[0])[..., None], Es[0], E)  # above the highest probe
        E = np.maximum(E, 1e-6)

        E_src = SRC_AMBIENT + SRC_KEY * np.maximum(np.sum(nrm * l_src, -1), 0)
        gain = E / np.maximum(E_src, 1e-6)[..., None]

        if anchor is None:
            i_anchor = min(args.anchor_frame, n - 1)
            m = alpha > .99
            anchor = 1.0 / max(float(np.median(gain[m] @ LUMA)), 1e-6) if m.any() else 1.0
        gain = gain * anchor

        # tanh in stops: monotone, unit slope at zero, smooth saturation.
        g = np.log2(np.maximum(gain, 1e-6))
        g = args.stops * np.tanh(g / args.stops)
        gain = np.exp2(g).astype(np.float32)
        gain = blur(gain, 2.2)

        np.savez_compressed(out / ('%06d.npz' % i), gain=gain.astype(np.float16),
                            specular=np.zeros_like(gain, np.float16))
        m = alpha > .99
        if m.any():
            mg = gain[m].mean(0)
            stats.append({'frame': i, 'mean_gain': mg.tolist(),
                          'luma': float(mg @ LUMA), 'warm_rb': float(mg[0] / max(mg[2], 1e-6))})
        if i % 25 == 0:
            print(json.dumps(stats[-1]), flush=True)

    g = np.array([s['mean_gain'] for s in stats])
    lum = g @ LUMA
    rb = g[:, 0] / np.maximum(g[:, 2], 1e-6)
    summary = {'shot': shot.name, 'frames': len(stats), 'environment': env.name,
               'depth_calibration': cal, 'exposure_anchor': anchor,
               'tanh_stops': args.stops, 'normal_source': normal_source,
               'E_source': {'model': 'soft frontal studio key (ASSUMED, not measured)',
                            'direction': l_src.tolist(), 'ambient': SRC_AMBIENT, 'key': SRC_KEY,
                            'how_to_measure': 'chrome ball + 18% grey ball on a slate at the head of the take'},
               'result': {'luma_min': float(lum.min()), 'luma_max': float(lum.max()),
                          'luma_swing_pct': float(100 * np.ptp(lum) / lum.mean()),
                          'warm_rb_min': float(rb.min()), 'warm_rb_max': float(rb.max()),
                          'warm_rb_swing_pct': float(100 * np.ptp(rb) / rb.mean())},
               'per_frame': stats}
    shot.log('relight_sh.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ('frames', 'exposure_anchor', 'result')}, indent=2))


if __name__ == '__main__':
    main()
