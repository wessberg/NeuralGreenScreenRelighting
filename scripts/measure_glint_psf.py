"""How sharp is a catchlight in the PHOTOGRAPH, and how sharp is the rendered one?

    dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/measure_glint_psf.py \
        --shot music_videos/reporter_test/shots/reporter_a --render composite/r41_corneal

The corneal pass is a Cycles render: a bulb that images as a spot 0.14-0.8 px across, spread
only by the pixel filter (Blackman-Harris, 1.5 px). The plate's own softbox glint has been
through the taking lens, the sensor, 4:2:0 chroma, and a codec. Adding a razor-sharp spot to
a photograph whose sharpest feature is softer than that is the "computer-ey" tell at the one
place the eye goes first, so the pass has to be softened to the plate -- by a MEASURED amount.

Both are measured the same way, on the glint itself:

* **plate**: on every open eye with a softbox glint (source frames from the eye pass, which
  carry the untouched photograph), a rectangle blurred by a Gaussian -- `b + A * erf-box(W, H,
  x0, y0, sigma)` -- is fitted to a crop around the glint. `sigma` is the plate's point-spread
  at the eye, in native pixels; `W`, `H` the softbox image.
* **render**: on the tour's far frontal frames the bulb subtends a 0.14 px spot, so the rendered
  glint IS the pixel filter: a Gaussian is fitted directly. On the close frontal frames the spot
  is 0.66 px and a disc (*) Gaussian with the disc fixed from the geometry must give the same
  sigma -- the check that nothing but the filter is spreading it.

The number to use is `extra_sigma = sqrt(sigma_plate^2 - sigma_render^2)`, the Gaussian that
takes the render to the plate; composite_ratio.py --corneal-blur takes it. Medians, p95 and
counts are reported, never a single frame.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shot_paths import add_shot_argument, from_args
from color_pipeline import srgb_to_linear
from eyelid_fit import evaluate as lid_evaluate

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def erf_box(xx, yy, x0, y0, W, H, sigma):
    s = max(sigma, 0.05) * np.sqrt(2.0)
    fx = 0.5 * (erf((xx - x0 + W / 2) / s) - erf((xx - x0 - W / 2) / s))
    fy = 0.5 * (erf((yy - y0 + H / 2) / s) - erf((yy - y0 - H / 2) / s))
    return fx * fy


def fit_box(crop, x0, y0, fixed_wh=None):
    h, w = crop.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float64)
    b0 = float(np.percentile(crop, 30))
    A0 = float(crop.max() - b0)

    def model(p):
        if fixed_wh is None:
            b, A, cx, cy, W, H, sg = p
        else:
            b, A, cx, cy, sg = p
            W, H = fixed_wh
        return b + A * erf_box(xx, yy, cx, cy, abs(W), abs(H), abs(sg))

    if fixed_wh is None:
        p0 = [b0, A0, x0, y0, 3.0, 3.0, 0.8]
        lo = [-1, 0, 0, 0, 0.2, 0.2, 0.1]
        hi = [2, 50, w, h, w, h, 4.0]
    else:
        p0 = [b0, A0, x0, y0, 0.8]
        lo = [-1, 0, 0, 0, 0.05]
        hi = [2, 1e4, w, h, 4.0]
    r = least_squares(lambda p: (model(p) - crop).ravel(), p0, bounds=(lo, hi))
    res = float(np.sqrt(np.mean(r.fun ** 2)))
    p = r.x
    if fixed_wh is None:
        return dict(b=p[0], A=p[1], x=p[2], y=p[3], W=abs(p[4]), H=abs(p[5]), sigma=abs(p[6]),
                    rms=res, ok=bool(r.success))
    return dict(b=p[0], A=p[1], x=p[2], y=p[3], W=fixed_wh[0], H=fixed_wh[1], sigma=abs(p[4]),
                rms=res, ok=bool(r.success))


def plate_glints(shot, source_dir, limbus, lids, half=8):
    out = []
    for r in limbus['rows']:
        fi = int(r['frame'])
        img = None
        for e in r['eyes']:
            if not e.get('fitted'):
                continue
            side = e['side']
            lid = lids.get((fi, side))
            if lid is None or lid.get('upper') is None or lid.get('lower') is None:
                continue
            cx, cy = e['smoothed_centre']
            rad = float(e['smoothed_axes'][0])
            opening = float(lid_evaluate(lid['lower']['coef'], cx, lid['centre'])
                            - lid_evaluate(lid['upper']['coef'], cx, lid['centre']))
            if opening < 1.4 * rad:
                continue
            if img is None:
                img = cv2.imread(str(source_dir / ('%06d.png' % fi)), -1)[..., ::-1]
                img = srgb_to_linear(img.astype(np.float32) / 255.0) @ LUMA
            x0, x1 = int(cx - rad - 2), int(cx + rad + 3)
            y0, y1 = int(cy - rad - 2), int(cy + rad + 3)
            crop = img[y0:y1, x0:x1]
            yy, xx = np.mgrid[y0:y1, x0:x1]
            disc = np.hypot(xx - cx, yy - cy) <= rad * 0.95
            L = np.where(disc, crop, 0.0)
            if L.max() < srgb_to_linear(0.5):
                continue
            m = (L >= 0.6 * L.max()).astype(np.uint8)
            n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
            k = max(range(1, n), key=lambda k: float(L[lab == k].sum()))
            wgt = np.where(lab == k, L, 0.0)
            gx, gy = float((xx * wgt).sum() / wgt.sum()), float((yy * wgt).sum() / wgt.sum())
            # The whole glint must be inside the limbus with a margin, or the fit sees the lid.
            if np.hypot(gx - cx, gy - cy) > rad - 4.5:
                continue
            X0, Y0 = int(round(gx)) - half, int(round(gy)) - half
            c2 = img[Y0:Y0 + 2 * half + 1, X0:X0 + 2 * half + 1]
            if c2.shape != (2 * half + 1, 2 * half + 1):
                continue
            f = fit_box(c2.astype(np.float64), gx - X0, gy - Y0)
            f.update(frame=fi, side=side)
            out.append(f)
    return out


def render_spots(shot, render_dir, half=6):
    man = json.loads((render_dir / 'render_manifest.json').read_text(encoding='utf-8'))
    cp = man['corneal_pass']['per_frame']
    R_C = 0.0078
    out = []
    for rec in cp:
        n = rec['out_frame']
        img = cv2.imread(str(render_dir / ('%06d.exr' % n)), -1)
        if img is None:
            continue
        L = img[..., 2::-1].astype(np.float64) @ LUMA.astype(np.float64)
        S = np.asarray(rec['bulb_world'])
        for side, e in rec['eyes'].items():
            if not e.get('rendered'):
                continue
            C = np.asarray(e['centre_world'])
            d = float(np.linalg.norm(S - C))
            # Convex mirror: image radius = (R/2) * r_bulb / d, in metres, then to pixels via
            # the iris scale (iris radius in px over 5.85 mm).
            r_img_m = (R_C / 2.0) * rec['bulb_radius_m'] / d
            px_per_m = float(e['iris_radius_px']) / 0.00585
            r_img_px = r_img_m * px_per_m
            cx, cy = e['anchor_px']
            rad = float(e['iris_radius_px'])
            x0, x1 = int(cx - rad - 2), int(cx + rad + 3)
            y0, y1 = int(cy - rad - 2), int(cy + rad + 3)
            crop = L[y0:y1, x0:x1]
            yy, xx = np.mgrid[y0:y1, x0:x1]
            disc = np.hypot(xx - cx, yy - cy) <= rad
            Lc = np.where(disc, crop, 0.0)
            if Lc.max() < 0.02:
                continue
            iy, ix = np.unravel_index(np.argmax(Lc), Lc.shape)
            gx, gy = ix + x0, iy + y0
            if np.hypot(gx - cx, gy - cy) > rad - half:
                continue
            X0, Y0 = gx - half, gy - half
            c2 = L[Y0:Y0 + 2 * half + 1, X0:X0 + 2 * half + 1]
            # The bulb image is a DISC; a square of the same area is the erf-box equivalent.
            side_px = max(r_img_px * np.sqrt(np.pi), 0.02)
            f = fit_box(c2, half, half, fixed_wh=(side_px, side_px))
            f.update(frame=n, side=side, bulb_distance_m=d, spot_radius_px=r_img_px,
                     peak=float(c2.max()))
            out.append(f)
    return out


def stats(v):
    v = np.asarray(v, float)
    return {'n': int(v.size), 'median': float(np.median(v)), 'p5': float(np.percentile(v, 5)),
            'p95': float(np.percentile(v, 95))} if v.size else {'n': 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_shot_argument(ap)
    ap.add_argument('--source-run', default='r40_floorB_import',
                    help='composite/<run>/source: the untouched photograph, 8-bit sRGB')
    ap.add_argument('--render', default='composite/r41_corneal')
    ap.add_argument('--limbus-track', default='limbus_track_tv_import.json')
    ap.add_argument('--eyelid-track', default='eyelid_track_wtv40_import.json')
    ap.add_argument('--out', default='r41_glint_psf.json')
    args = ap.parse_args()
    shot = from_args(args)
    limbus = json.loads((shot.relight / args.limbus_track).read_text(encoding='utf-8'))
    lids_raw = json.loads((shot.relight / args.eyelid_track).read_text(encoding='utf-8'))
    lids = {(int(r['frame']), e['side']): e for r in lids_raw['rows'] for e in r['eyes']
            if e.get('fitted')}
    plate = plate_glints(shot, shot.composite / args.source_run / 'source', limbus, lids)
    good = [f for f in plate if f['ok'] and f['rms'] < 0.35 * f['A']]
    ps = stats([f['sigma'] for f in good])
    print('PLATE  glints fitted %d (kept %d)  sigma median %.3f  p5 %.3f  p95 %.3f px | box W %.2f x H %.2f px'
          % (len(plate), len(good), ps['median'], ps['p5'], ps['p95'],
             np.median([f['W'] for f in good]), np.median([f['H'] for f in good])), flush=True)
    rd = Path(args.render)
    rd = rd if rd.is_absolute() else shot.dir / rd
    spots = render_spots(shot, rd)
    far = [f for f in spots if f['spot_radius_px'] < 0.25]
    near = [f for f in spots if f['spot_radius_px'] > 0.5]
    fs, ns = stats([f['sigma'] for f in far]), stats([f['sigma'] for f in near])
    print('RENDER far spots (r < 0.25 px) n %d  sigma median %.3f p5 %.3f p95 %.3f'
          % (fs['n'], fs['median'], fs['p5'], fs['p95']), flush=True)
    if ns['n']:
        print('RENDER near spots (r > 0.5 px, disc fixed) n %d  sigma median %.3f p5 %.3f p95 %.3f'
              % (ns['n'], ns['median'], ns['p5'], ns['p95']), flush=True)
    sig_p, sig_r = ps['median'], fs['median']
    extra = float(np.sqrt(max(sig_p ** 2 - sig_r ** 2, 0.0)))
    print('EXTRA blur to take the render to the plate: sigma %.3f px' % extra, flush=True)
    rep = {'plate': {'stats_sigma_px': ps, 'box_px': {'W': float(np.median([f['W'] for f in good])),
                                                       'H': float(np.median([f['H'] for f in good]))},
                     'per_eye': [{k: (round(v, 4) if isinstance(v, float) else v)
                                  for k, v in f.items()} for f in good]},
           'render': {'far_sigma_px': fs, 'near_sigma_px': ns,
                      'per_eye': [{k: (round(v, 4) if isinstance(v, float) else v)
                                   for k, v in f.items()} for f in spots]},
           'extra_sigma_px': extra,
           'model': 'b + A * erf-box(W, H, sigma); plate on the softbox glint of open eyes, '
                    'render on the tour bulb with its disc size fixed from the geometry'}
    p = shot.log(args.out)
    p.write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print('->', p)


if __name__ == '__main__':
    main()
