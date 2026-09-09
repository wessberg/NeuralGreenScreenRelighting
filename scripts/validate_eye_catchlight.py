"""Does the rendered corneal reflection show the environment the geometry says it should?

    dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/validate_eye_catchlight.py \
        --shot music_videos/reporter_test/shots/reporter_a --render composite/r34_catchlight

`validate_catchlight.py` already established that the glint MOVES the way the light moves,
by rendering one source at four positions. That is the reflection geometry. It says nothing
about the panorama, and an equirectangular convention error is exactly the kind of mistake
this project keeps writing down: it produces a plausible catchlight in the wrong place, in
one frame, ungradeably.

So this recomputes the whole reflection independently -- in numpy, from the camera
intrinsics, the solved corneal sphere and the panorama file -- and compares it to what
Cycles rendered. Two implementations that share no code agreeing on where the restaurant's
lamps land in a 25-pixel cornea is the check. Reported per eye:

* the offset, in native pixels, between the rendered brightest pixel and the predicted one;
* the normalized cross-correlation of the two images over the corneal disc;
* the same for three DELIBERATELY WRONG conventions -- panorama flipped in u, flipped in v,
  and the reflection replaced by the surface normal. A test that only ever sees the right
  answer cannot fail, and a mirrored equirect leaves a symmetric scene nearly invariant.

Blender's `direction_to_equirectangular` is the convention under test:

    u = (atan2(d.y, d.x) - pi) / (-2 pi)        v = (atan2(d.z, hypot(d.x, d.y)) + pi/2) / pi

with v = 0 at the BOTTOM of the image, which is why the sampled row is (1 - v) * height.
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
from shot_paths import add_shot_argument, from_args
from corneal_anchor import smoothed_iris_centres, anchor_centre

ROOT = Path(__file__).resolve().parents[1]
IOR = 1.376
F0 = ((IOR - 1.0) / (IOR + 1.0)) ** 2


def equirect_uv(d, flip_u=False, flip_v=False):
    """Blender's equirectangular direction -> (u, v), v = 0 at the image BOTTOM."""
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    u = (np.arctan2(y, x) - np.pi) / (-2.0 * np.pi)
    v = (np.arctan2(z, np.hypot(x, y)) + np.pi / 2.0) / np.pi
    u = np.mod(u, 1.0)
    if flip_u:
        u = 1.0 - u
    if flip_v:
        v = 1.0 - v
    return u, v


def sample_env(env, d, rotation_deg=0.0, flip_u=False, flip_v=False):
    if rotation_deg:
        a = np.radians(-rotation_deg)
        ca, sa = np.cos(a), np.sin(a)
        d = np.stack([ca * d[..., 0] - sa * d[..., 1],
                      sa * d[..., 0] + ca * d[..., 1], d[..., 2]], axis=-1)
    u, v = equirect_uv(d, flip_u, flip_v)
    h, w = env.shape[:2]
    mx = (u * w - 0.5).astype(np.float32)
    my = ((1.0 - v) * h - 0.5).astype(np.float32)
    return cv2.remap(env, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


_PREFILTER_CACHE = {}


def prefilter_env(env, sigma_px):
    """Blur a panorama, wrapping in longitude.

    `cv2.GaussianBlur` rejects BORDER_WRAP outright, so the wrap is done by padding the
    image with its own opposite edge and cropping back. Getting this wrong shows up only
    as a seam at the panorama's back, which no corneal patch may ever look at -- so it
    would pass silently.
    """
    if sigma_px <= 0.5:
        return env
    key = round(sigma_px, 2)
    if key in _PREFILTER_CACHE:
        return _PREFILTER_CACHE[key]
    pad = int(np.ceil(3 * sigma_px))
    wide = np.concatenate([env[:, -pad:], env, env[:, :pad]], axis=1)
    blurred = cv2.GaussianBlur(wide, (0, 0), sigma_px, borderType=cv2.BORDER_REPLICATE)
    _PREFILTER_CACHE[key] = blurred[:, pad:pad + env.shape[1]]
    return _PREFILTER_CACHE[key]


def ncc(a, b, mask):
    if mask.sum() < 20:
        return float('nan')
    x = a[mask].astype(np.float64)
    y = b[mask].astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 1e-12 else float('nan')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_shot_argument(ap)
    ap.add_argument('--render', default='composite/r34_catchlight',
                    help='directory of rendered corneal-reflection EXRs, relative to the shot')
    ap.add_argument('--frames', default='20,21,22')
    ap.add_argument('--half', type=int, default=20, help='patch half-width in native pixels')
    ap.add_argument('--out', default='r34_catchlight_validation.json')
    ap.add_argument('--proxy-anchor', action='store_true',
                    help='match a render made with --proxy-anchor')
    ap.add_argument('--raw-anchor', action='store_true',
                    help='match a render made with --raw-anchor')
    args = ap.parse_args()
    shot = from_args(args)
    cfg = shot.config
    cam = cfg.get('camera') or {}
    lens = float(cam.get('lens_mm', 50.0))
    sensor = float(cam.get('sensor_width_mm', 36.0))
    dist = float(cam.get('actor_distance_m', 1.30))
    cam_h = float(cam.get('height_m', 1.55))
    W, H = cfg.get('source_resolution', [1920, 1080])
    fx = W * lens / sensor
    cx, cy = W / 2.0, H / 2.0
    eye = np.array([0.0, -dist, cam_h])

    env_dir = ROOT / (cfg.get('environment') or {}).get('dir', 'scene_library/night_restaurant')
    info = json.loads((env_dir / 'scene_info.json').read_text())
    rotation = float(info['hdri'].get('rotation_deg', 0.0))
    strength = float((cfg.get('environment') or {}).get('hdri_strength',
                                                        info['hdri'].get('strength', 1.0)))
    env_path = ROOT / 'assets' / 'hdri' / info['hdri']['lighting']
    env = cv2.imread(str(env_path), cv2.IMREAD_UNCHANGED)
    if env is None:
        raise SystemExit('Could not read %s' % env_path)
    env = env[..., :3][..., ::-1].astype(np.float32) * strength

    geo = {r['frame']: r for r in json.loads((shot.relight / 'eye_geometry.json').read_text())['per_frame']}
    # The sphere has to be placed exactly where the renderer placed it, or this compares
    # two different corneas and blames the panorama. Both call scripts/corneal_anchor.py.
    smooth = None if (args.raw_anchor or args.proxy_anchor) else smoothed_iris_centres(geo)
    render_dir = shot.dir / args.render
    rows = []
    for fi in [int(x) for x in args.frames.split(',')]:
        rendered = cv2.imread(str(render_dir / ('%06d.exr' % fi)), cv2.IMREAD_UNCHANGED)
        if rendered is None:
            raise SystemExit('No render at %s' % (render_dir / ('%06d.exr' % fi)))
        rendered = rendered[..., :3][..., ::-1].astype(np.float32)
        rec = geo.get(fi) or {}
        for side, e in (rec.get('eyes') or {}).items():
            if not e.get('visible'):
                continue
            centre = np.asarray(e['cornea_centre_world'], float)
            radius = float(e['cornea_radius_m'])
            px, py = e['iris_centre_px']
            if not args.proxy_anchor:
                ax, ay = px, py
                if smooth is not None:
                    sm = smooth.get(side, {}).get(fi)
                    if sm is not None:
                        ax, ay = float(sm[0]), float(sm[1])
                centre, _ = anchor_centre(centre, radius,
                                          e.get('gaze_world') or [0.0, -1.0, 0.0],
                                          (ax, ay), eye, fx, W, H)
            x0, y0 = int(round(px)) - args.half, int(round(py)) - args.half
            xs, ys = np.meshgrid(np.arange(x0, x0 + 2 * args.half),
                                 np.arange(y0, y0 + 2 * args.half))
            # Camera looks along +Y with +Z up, so image x is world +X and image y is -Z.
            d = np.stack([(xs + 0.5 - cx) / fx, np.ones_like(xs, float),
                          -(ys + 0.5 - cy) / fx], axis=-1)
            d /= np.linalg.norm(d, axis=-1, keepdims=True)
            oc = eye - centre
            b = 2.0 * np.sum(d * oc, axis=-1)
            c = float(oc @ oc) - radius * radius
            disc = b * b - 4.0 * c
            hit = disc > 0
            t = np.where(hit, (-b - np.sqrt(np.maximum(disc, 0.0))) / 2.0, 0.0)
            p = eye + d * t[..., None]
            n = (p - centre) / radius
            v = -d
            ndv = np.sum(n * v, axis=-1)
            # A cap, exactly as the renderer truncates it: sin(theta) = iris / cornea.
            gaze = np.asarray(e.get('gaze_world') or [0, -1, 0], float)
            gaze /= max(np.linalg.norm(gaze), 1e-9)
            cap = np.sum(n * gaze, axis=-1) >= np.cos(np.arcsin(min(0.999, 5.85 / 7.8)))
            disc_mask = hit & (t > 0) & (ndv > 0) & cap
            r = 2.0 * ndv[..., None] * n - v
            fres = F0 + (1.0 - F0) * np.clip(1.0 - ndv, 0.0, 1.0) ** 5
            variants = {
                'blender_convention': dict(flip_u=False, flip_v=False, use_normal=False),
                'wrong_flip_u': dict(flip_u=True, flip_v=False, use_normal=False),
                'wrong_flip_v': dict(flip_u=False, flip_v=True, use_normal=False),
                'wrong_normal_not_reflection': dict(flip_u=False, flip_v=False, use_normal=True),
            }
            got = rendered[y0:y0 + 2 * args.half, x0:x0 + 2 * args.half].max(axis=-1)
            entry = dict(frame=fi, side=side, disc_pixels=int(disc_mask.sum()),
                         rendered_peak=round(float(got[disc_mask].max()), 5) if disc_mask.any() else None)
            # One output pixel of a 25-pixel convex mirror collects a WIDE cone of
            # reflected directions -- the whole cornea folds about 100 degrees into 25
            # pixels. A point sample of the panorama therefore misses small lamps that
            # Cycles' many samples find, which shows up as a large peak-magnitude gap and
            # nothing else. Measure that footprint from the reflection field itself and
            # prefilter the panorama by it, so the two are compared at one bandwidth.
            step = np.degrees(np.arccos(np.clip(
                np.sum(r[:, 1:] * r[:, :-1], axis=-1), -1.0, 1.0)))
            valid_step = step[disc_mask[:, 1:] & disc_mask[:, :-1]]
            deg_per_px = float(np.median(valid_step)) if valid_step.size else 0.0
            sigma_px = max(deg_per_px / 360.0 * env.shape[1] / 2.0, 0.0)
            env_filtered = prefilter_env(env, sigma_px)
            entry['reflection_deg_per_pixel'] = round(deg_per_px, 3)
            entry['prefilter_sigma_env_px'] = round(sigma_px, 2)
            variants['blender_convention_prefiltered'] = dict(flip_u=False, flip_v=False,
                                                              use_normal=False, prefilter=True)
            for label, kw in variants.items():
                kw = dict(kw)
                use_env = env_filtered if kw.pop('prefilter', False) else env
                direction = n if kw.pop('use_normal') else r
                env_rgb = sample_env(use_env, direction, rotation, **kw)
                pred = (env_rgb * fres[..., None]).max(axis=-1) * disc_mask
                if not disc_mask.any():
                    continue
                pk_r = np.unravel_index(int(np.argmax(np.where(disc_mask, got, -np.inf))), got.shape)
                pk_p = np.unravel_index(int(np.argmax(np.where(disc_mask, pred, -np.inf))), pred.shape)
                entry[label] = dict(
                    ncc=round(ncc(got, pred, disc_mask), 4),
                    log_ncc=round(ncc(np.log(np.maximum(got, 1e-5)),
                                      np.log(np.maximum(pred, 1e-5)), disc_mask), 4),
                    peak_offset_px=round(float(np.hypot(pk_r[0] - pk_p[0], pk_r[1] - pk_p[1])), 2),
                    predicted_peak=round(float(pred[disc_mask].max()), 5),
                    energy_ratio=round(float(pred[disc_mask].sum()
                                             / max(got[disc_mask].sum(), 1e-9)), 4))
            rows.append(entry)
            print('R34_VALIDATE', fi, side, json.dumps({k: v for k, v in entry.items()
                                                        if k != 'frame'}), flush=True)

    ok = [r for r in rows if 'blender_convention' in r]
    summary = dict(
        render=str(args.render), frames=args.frames, eyes=len(ok),
        environment=str(env_dir.relative_to(ROOT)), hdri=info['hdri']['lighting'],
        hdri_rotation_deg=rotation, hdri_strength=strength,
        anchor=('proxy apex' if args.proxy_anchor
                else ('tracked iris landmark (raw)' if args.raw_anchor
                      else 'tracked iris landmark smoothed by a local quadratic')),
        fresnel_f0=round(F0, 5), ior=IOR,
        by_variant={k: dict(
            ncc_median=round(float(np.median([r[k]['ncc'] for r in ok])), 4),
            log_ncc_median=round(float(np.median([r[k]['log_ncc'] for r in ok])), 4),
            peak_offset_px_median=round(float(np.median([r[k]['peak_offset_px'] for r in ok])), 2),
            peak_offset_px_max=round(float(np.max([r[k]['peak_offset_px'] for r in ok])), 2),
            energy_ratio_median=round(float(np.median([r[k]['energy_ratio'] for r in ok])), 4))
            for k in ('blender_convention', 'blender_convention_prefiltered', 'wrong_flip_u',
                      'wrong_flip_v', 'wrong_normal_not_reflection')},
        rows=rows,
        note='Cycles and this numpy model share no code. Agreement on where a 25-pixel '
             'cornea puts the restaurant lamps is the check; the three wrong variants are '
             'there so the test can fail.')
    best = max(summary['by_variant'], key=lambda k: summary['by_variant'][k]['log_ncc_median'])
    summary['best_variant'] = best
    summary['passes'] = bool(best.startswith('blender_convention'))
    out = shot.log(args.out)
    out.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary['by_variant'], indent=2))
    print('R34_VALIDATE_BEST', best, 'passes', summary['passes'], '->', out, flush=True)


if __name__ == '__main__':
    main()
