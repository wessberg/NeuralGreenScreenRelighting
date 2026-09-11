"""Is the corneal sphere in the right place, and does the tour's bulb reflect where it should?

    dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/validate_tour_catchlight.py \
        --shot music_videos/reporter_test/shots/reporter_a --plate-glint
    ... --render composite/r41_corneal          # check a rendered pass

Two checks, both independent of Cycles, both in numpy through `corneal_anchor.py` (shared
with the renderer so the two cannot place different spheres).

**--plate-glint: the photograph grades the sphere.** The capture room's softbox is static. Its
glint is measured here on every OPEN eye (the luminance-weighted centroid of the brightest blob
inside the limbus on the untouched source frame; `eye_geometry.py`'s own glint centres proved
too noisy for this at 3-5 px, and at a blink the visible part of the glint moves with the lid).
For a candidate sphere placement (a gaze model), the glint of ONE fixed distant source is
predicted on every eye-frame; the source direction is fitted (two angles, global search) to
minimise the pixel residual, and what remains is placement error -- the same softbox cannot
move. Reported rms / p95 / max in native pixels, per eye, per model, and the frame-to-frame
step of the residual, which is what would flicker. Round 41: geometry gaze 1.31 px rms,
camera-axis gaze 1.87 with a 1.3 px lateral bias of opposite sign on each eye.

**--render: the tour grades the render.** Per rendered eye, the bulb's world position from the
pass manifest is reflected in the SAME sphere the renderer placed (recomputed here from the
track and eye_geometry, and cross-checked against the manifest's stored centre), projected to a
pixel, and compared with the rendered peak. Three DELIBERATELY WRONG conventions are scored
alongside -- bulb mirrored in X, bulb mirrored in Z about the eye, and the surface normal used
instead of the reflection -- because a test that only sees the right answer cannot fail. Also:
the glint's chromaticity against the bulb's colour (Fresnel is achromatic), and that stations
with the bulb behind her put nothing in the eye.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shot_paths import add_shot_argument, from_args
from corneal_anchor import (smoothed_iris_centres, anchor_limbus, reflect_point, project)
from eyelid_fit import evaluate as lid_evaluate

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
IRIS_R = 0.00585


def load_track(shot, name):
    data = json.loads((shot.relight / name).read_text(encoding='utf-8'))
    raw = {}
    for row in data['rows']:
        eyes = {}
        for e in row['eyes']:
            if e.get('fitted'):
                eyes[e['side']] = {'visible': True, 'iris_centre_px': list(e['smoothed_centre']),
                                   'iris_radius_px': float(e['smoothed_axes'][0])}
        raw[int(row['frame'])] = {'eyes': eyes}
    return raw


def camera_of(shot):
    cfg = shot.config
    cam = cfg.get('camera') or {}
    lens = float(cam.get('lens_mm', 50.0))
    sensor = float(cam.get('sensor_width_mm', 36.0))
    W, H = cfg.get('source_resolution', [1920, 1080])
    return dict(pos=np.array([0.0, -float(cam.get('actor_distance_m', 1.3)),
                              float(cam.get('height_m', 1.55))]),
                fx=W * lens / sensor, W=W, H=H)


def sphere_for(eye_rec, track_rec, px, gaze_model, cam):
    """The sphere the renderer places, from the same inputs."""
    depth_pt = np.asarray(eye_rec['cornea_apex_world'], float)
    R = float(eye_rec['cornea_radius_m'])
    if gaze_model == 'camera':
        # Toward the camera from the eye, not the camera axis: the eye is off-centre in frame.
        gaze = cam['pos'] - depth_pt
    else:
        gaze = np.asarray(eye_rec.get('gaze_world') or [0.0, -1.0, 0.0], float)
    C, iris_w = anchor_limbus(depth_pt, R, IRIS_R, gaze, px, cam['pos'], cam['fx'],
                              cam['W'], cam['H'])
    return C, R, gaze / max(np.linalg.norm(gaze), 1e-9)


def eye_frames(shot, eye_geo, track, smooth, gaze_model, cam):
    """(frame, side, sphere centre, R, anchor px) for every eye with a fit and geometry."""
    out = []
    for fi in sorted(track):
        rec = eye_geo.get(fi) or {}
        for side, tr in (track[fi].get('eyes') or {}).items():
            e = ((rec.get('eyes') or {}).get(side)) or {}
            if not e.get('visible'):
                continue
            px = tr['iris_centre_px']
            sm = (smooth or {}).get(side, {}).get(fi)
            if sm is not None:
                px = (float(sm[0]), float(sm[1]))
            C, R, g = sphere_for(e, tr, px, gaze_model, cam)
            out.append((fi, side, C, R, np.asarray(px, float), e))
    return out


def fit_direction(entries, cam):
    """One distant source direction that best explains the observed glints: a global grid
    over the front hemisphere, then three refinements. Two angles, no initial guess."""
    obs = np.array([e['obs'] for e in entries])

    def resid(l):
        pred = np.array([project(reflect_point(e['C'], e['R'], cam['pos'], l, True)[0],
                                 cam['pos'], cam['fx'], cam['W'], cam['H']) for e in entries])
        return pred - obs

    def to_dir(th, ph):
        return np.array([np.cos(ph) * np.sin(th), -np.cos(ph) * np.cos(th), np.sin(ph)])

    best = None
    for th in np.radians(np.arange(-80, 81, 4)):
        for ph in np.radians(np.arange(-60, 61, 4)):
            r = resid(to_dir(th, ph))
            sc = float(np.sqrt((r ** 2).sum(1).mean()))
            if best is None or sc < best[0]:
                best = (sc, th, ph)
    sc, th, ph = best
    for step in (1.0, 0.25, 0.06):
        bt, bp = th, ph
        for dt in np.radians(np.arange(-3, 4) * step):
            for dp in np.radians(np.arange(-3, 4) * step):
                r = resid(to_dir(th + dt, ph + dp))
                s2 = float(np.sqrt((r ** 2).sum(1).mean()))
                if s2 < sc:
                    sc, bt, bp = s2, th + dt, ph + dp
        th, ph = bt, bp
    l = to_dir(th, ph)
    return l, resid(l)


def measure_glint(img_luma, cx, cy, r):
    """Luminance-weighted centroid of the brightest blob inside the limbus, or None."""
    x0, x1 = int(cx - r - 2), int(cx + r + 3)
    y0, y1 = int(cy - r - 2), int(cy + r + 3)
    crop = img_luma[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    disc = np.hypot(xx - cx, yy - cy) <= r * 0.95
    L = np.where(disc, crop, 0.0)
    if L.max() < 0.5:
        return None
    m = (L >= 0.6 * L.max()).astype(np.uint8)
    n, lab, _, _ = cv2.connectedComponentsWithStats(m, 8)
    if n < 2:
        return None
    k = max(range(1, n), key=lambda k: float(L[lab == k].sum()))
    w = np.where(lab == k, L, 0.0)
    return np.array([(xx * w).sum() / w.sum(), (yy * w).sum() / w.sum()])


def open_eyes(shot, eyelid_name):
    lids = json.loads((shot.relight / eyelid_name).read_text(encoding='utf-8'))
    return {(int(r['frame']), e['side']): e for r in lids['rows'] for e in r['eyes']
            if e.get('fitted') and e.get('upper') and e.get('lower')}


def plate_glint_check(shot, eye_geo, track, cam, models, smooth_flag, source_dir, lids):
    smooth = smoothed_iris_centres(track) if smooth_flag else None
    # Observed glints, once: display luma of the untouched source frame.
    obs = {}
    cache = {}
    for fi, side, C, R, px, e in eye_frames(shot, eye_geo, track, smooth, 'camera', cam):
        lid = lids.get((fi, side))
        if lid is None:
            continue
        rad = track[fi]['eyes'][side]['iris_radius_px']
        opening = float(lid_evaluate(lid['lower']['coef'], px[0], lid['centre'])
                        - lid_evaluate(lid['upper']['coef'], px[0], lid['centre']))
        if opening < 1.4 * rad:
            continue
        if fi not in cache:
            im = cv2.imread(str(source_dir / ('%06d.png' % fi)), -1)
            cache[fi] = (im[..., ::-1].astype(np.float32) / 255.0) @ LUMA
        g = measure_glint(cache[fi], px[0], px[1], rad)
        if g is not None:
            obs[(fi, side)] = g
    report = {'open_eyes_with_glint': len(obs)}
    for model in models:
        entries = []
        for fi, side, C, R, px, e in eye_frames(shot, eye_geo, track, smooth, model, cam):
            g = obs.get((fi, side))
            if g is None:
                continue
            entries.append({'frame': fi, 'side': side, 'C': C, 'R': R, 'obs': g, 'anchor': px})
        l, rv = fit_direction(entries, cam)
        r = np.linalg.norm(rv, axis=1)
        per_side = {}
        step = []
        for side in ('left', 'right'):
            idx = [i for i, e in enumerate(entries) if e['side'] == side]
            if not idx:
                continue
            rs = rv[idx]
            for j in range(len(idx) - 1):
                if entries[idx[j + 1]]['frame'] == entries[idx[j]]['frame'] + 1:
                    step.append(float(np.linalg.norm(rs[j + 1] - rs[j])))
            nn = np.linalg.norm(rs, axis=1)
            per_side[side] = {'n': len(idx), 'mean_xy_px': rs.mean(0).round(3).tolist(),
                              'std_xy_px': rs.std(0).round(3).tolist(),
                              'rms_px': float(np.sqrt(np.mean(nn ** 2))),
                              'p95_px': float(np.percentile(nn, 95)), 'max_px': float(nn.max())}
        step = np.array(step) if step else np.zeros(1)
        report[model] = {'fitted_direction_world': l.round(4).tolist(),
                         'angle_from_camera_axis_deg': float(np.degrees(np.arccos(
                             np.clip(l @ np.array([0.0, -1.0, 0.0]), -1, 1)))),
                         'n': len(entries), 'rms_px': float(np.sqrt(np.mean(r ** 2))),
                         'p95_px': float(np.percentile(r, 95)), 'max_px': float(r.max()),
                         'per_side': per_side,
                         'residual_step_px': {'p95': float(np.percentile(step, 95)),
                                              'max': float(step.max())}}
        print('PLATE_GLINT %-9s n %3d  rms %.2f  p95 %.2f  max %.2f px   step p95 %.2f max %.2f'
              '   bias L %s R %s   dir %s (%.1f deg)'
              % (model, len(entries), report[model]['rms_px'], report[model]['p95_px'],
                 report[model]['max_px'], report[model]['residual_step_px']['p95'],
                 report[model]['residual_step_px']['max'],
                 per_side.get('left', {}).get('mean_xy_px'),
                 per_side.get('right', {}).get('mean_xy_px'),
                 report[model]['fitted_direction_world'],
                 report[model]['angle_from_camera_axis_deg']), flush=True)
    return report


def peak_of(rgb, cx, cy, r):
    """Sub-pixel centroid of the brightest pixels inside the iris disc, and the peak RGB."""
    h, w = rgb.shape[:2]
    x0, x1 = max(0, int(cx - r - 2)), min(w, int(cx + r + 3))
    y0, y1 = max(0, int(cy - r - 2)), min(h, int(cy + r + 3))
    crop = rgb[y0:y1, x0:x1]
    L = crop @ LUMA
    yy, xx = np.mgrid[y0:y1, x0:x1]
    disc = np.hypot(xx - cx, yy - cy) <= r + 1.0
    L = np.where(disc, L, 0.0)
    if L.max() <= 0:
        return None
    top = L >= 0.5 * L.max()
    wsum = float(L[top].sum())
    return {'centroid': np.array([float((xx * L)[top].sum() / wsum),
                                  float((yy * L)[top].sum() / wsum)]),
            'peak': float(L.max()), 'peak_rgb': crop.reshape(-1, 3)[np.argmax(L)].tolist(),
            'energy': float(L.sum())}


def local_maxima_outside(rgb, cx, cy, r, lo=1.07, hi=2.0, floor=0.03):
    """Every local maximum (3x3, above `floor`) OUTSIDE the corneo-scleral junction and inside
    `hi` limbus radii, as sub-pixel centroids of its 3x3 neighbourhood, with its level. The
    globe shows the bulb AND the fill panel's broad lobe at once, and at some stations the
    lobe is the brighter, so the test is 'is there a peak where the model says', per
    convention, not 'is the brightest peak there'."""
    x0, x1 = int(cx - hi * r - 3), int(cx + hi * r + 4)
    y0, y1 = int(cy - hi * r - 3), int(cy + hi * r + 4)
    L = rgb[y0:y1, x0:x1] @ LUMA
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rho = np.hypot(xx - cx, yy - cy) / r
    mx = cv2.dilate(L, np.ones((3, 3), np.uint8))
    ys, xs = np.where((L == mx) & (L > floor) & (rho > lo) & (rho < hi))
    out = []
    for y, x in zip(ys, xs):
        w = L[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2]
        wy = yy[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2]
        wx = xx[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2]
        out.append((np.array([(wx * w).sum() / w.sum(), (wy * w).sum() / w.sum()]),
                    float(L[y, x])))
    return out


def globe_check(shot, eye_geo, track, cam, render_dir, gaze_model, smooth_flag):
    """Round 42: the bulb reflected in the scleral GLOBE, by the same Alhazen solve, against the
    rendered peak outside the junction -- scored only on eyes where the prediction itself lands
    outside the junction (elsewhere the globe is under the cornea and nothing of it renders).
    Three wrong conventions alongside, as for the cap."""
    man = json.loads((render_dir / 'render_manifest.json').read_text(encoding='utf-8'))
    cp = man.get('corneal_pass') or {}
    g = cp.get('globe')
    if not g:
        print('GLOBE: pass has no globe', flush=True)
        return None
    back, Rg = g['back_mm'] / 1000.0, g['radius_mm'] / 1000.0
    junction_r = g['junction_mm'] / 5.85            # in iris radii
    sock = cp.get('socket')

    def socket_blocks(side, fwd, point, source):
        """Does the head socket (as the renderer built it) hide `source` from `point`? The
        cup is centred on the globe and the ray starts on its surface, so this is the
        renderer's own rule applied from the surface point: a few degrees of parallax at
        24 mm, none of the geometry."""
        if sock is None:
            return False
        fwd = np.asarray(fwd, float)
        fwd = fwd / np.linalg.norm(fwd)
        up = np.array([0.0, 0.0, 1.0])
        yv = up - (up @ fwd) * fwd
        yv /= np.linalg.norm(yv)
        xv = np.cross(yv, fwd)                       # Blender to_track_quat('Z', 'Y'): X = Y x Z
        d = np.asarray(source, float) - np.asarray(point, float)
        d /= np.linalg.norm(d)
        theta = np.degrees(np.arccos(np.clip(d @ fwd, -1, 1)))
        temporal_x = 1.0 if side == 'left' else -1.0
        phi = np.arctan2(d @ yv, (d @ xv) * temporal_x)
        cs, sn = np.cos(phi) ** 2, np.sin(phi) ** 2
        th_h = sock['temporal_deg'] if np.cos(phi) >= 0 else sock['nasal_deg']
        th_v = sock['up_deg'] if np.sin(phi) >= 0 else sock['down_deg']
        return bool(theta > th_h * cs + th_v * sn)
    smooth = smoothed_iris_centres(track) if smooth_flag else None
    spheres = {(fi, side): (C, R, px, e) for fi, side, C, R, px, e in
               eye_frames(shot, eye_geo, track, smooth, gaze_model, cam)}
    rows = []
    for rec in cp.get('per_frame') or []:
        n, fi = rec['out_frame'], rec['shot_frame']
        p = render_dir / ('%06d.exr' % n)
        if not p.exists():
            continue
        rgb = None
        S = np.asarray(rec['bulb_world'], float)
        for side, e in rec['eyes'].items():
            if not e.get('rendered') or not e.get('globe') or (fi, side) not in spheres:
                continue
            C, R, px, geo = spheres[(fi, side)]
            gz = np.asarray(e['gaze_world'], float)
            Cg = C - back * gz
            drift = float(np.linalg.norm(Cg - np.asarray(e['globe']['centre_world'])))
            r_px = float(e['iris_radius_px'])
            q = project(reflect_point(Cg, Rg, cam['pos'], S)[0], cam['pos'], cam['fx'],
                        cam['W'], cam['H'])
            rho_pred = float(np.hypot(q[0] - px[0], q[1] - px[1]) / r_px)
            p_ref = reflect_point(Cg, Rg, cam['pos'], S)[0]
            blocked = (socket_blocks(side, e['socket']['head_forward_world'], p_ref, S)
                       if e.get('socket') else False)
            row = {'out_frame': n, 'shot_frame': fi, 'side': side, 'globe_drift_m': drift,
                   'pred_rho': rho_pred, 'socket_blocks': blocked,
                   'scored': rho_pred > junction_r and rho_pred < 2.0}
            if row['scored']:
                if rgb is None:
                    rgb = cv2.imread(str(p), -1)[..., 2::-1].astype(np.float32)
                peaks = local_maxima_outside(rgb, px[0], px[1], r_px, lo=junction_r)
                row['rendered_peak'] = max([pk[1] for pk in peaks] or [None]) if peaks else None
                if peaks:
                    eye_z = Cg[2]
                    conv = {
                        'reflection': lambda: reflect_point(Cg, Rg, cam['pos'], S)[0],
                        'bulb_x_mirrored': lambda: reflect_point(Cg, Rg, cam['pos'],
                                                                 S * [-1, 1, 1])[0],
                        'bulb_z_mirrored': lambda: reflect_point(
                            Cg, Rg, cam['pos'], np.array([S[0], S[1], 2 * eye_z - S[2]]))[0],
                        'normal_not_reflection': lambda: Cg + Rg * (S - Cg) / np.linalg.norm(S - Cg),
                        'cap_not_globe': lambda: reflect_point(C, R, cam['pos'], S)[0],
                    }
                    for k, f in conv.items():
                        v = project(f(), cam['pos'], cam['fx'], cam['W'], cam['H'])
                        dists = [float(np.linalg.norm(v - c)) for c, _ in peaks]
                        row['offset_px_' + k] = min(dists)          # nearest rendered peak
                        row['peak_at_' + k] = float(peaks[int(np.argmin(dists))][1])
            rows.append(row)
    pf = {r['out_frame']: r for r in man['per_frame']}
    for r in rows:
        r['station'] = pf.get(r['out_frame'], {}).get('station')
    cand = [r for r in rows if r['scored']]
    open_ = [r for r in cand if not r['socket_blocks']]
    blocked = [r for r in cand if r['socket_blocks']]
    scored = [r for r in open_ if r.get('rendered_peak')]
    summary = {'eyes_with_globe': len(rows), 'eyes_prediction_outside_junction': len(cand),
               'eyes_socket_open': len(open_), 'eyes_socket_blocked': len(blocked),
               'eyes_open_with_no_peak': len(open_) - len(scored),
               'eyes_scored': len(scored),
               # a blocked eye must NOT show a peak where the unblocked model says
               'blocked_eyes_with_peak_within_1.5px': sum(
                   1 for r in blocked if r.get('offset_px_reflection', 99) < 1.5),
               'junction_iris_radii': junction_r,
               'globe_drift_max_m': max([r['globe_drift_m'] for r in rows] or [0.0]),
               'stations_scored': sorted({r['station'] for r in scored})}
    for k in ('reflection', 'bulb_x_mirrored', 'bulb_z_mirrored', 'normal_not_reflection',
              'cap_not_globe'):
        v = np.array([r['offset_px_' + k] for r in scored if 'offset_px_' + k in r])
        if v.size:
            summary['offset_px_' + k] = {'median': float(np.median(v)),
                                         'p95': float(np.percentile(v, 95)), 'max': float(v.max())}
    for k, v in summary.items():
        print('GLOBE', k, v, flush=True)
    return {'summary': summary, 'rows': rows}


def render_check(shot, eye_geo, track, cam, render_dir, gaze_model, smooth_flag):
    man = json.loads((render_dir / 'render_manifest.json').read_text(encoding='utf-8'))
    cp = man.get('corneal_pass') or {}
    smooth = smoothed_iris_centres(track) if smooth_flag else None
    spheres = {(fi, side): (C, R, px) for fi, side, C, R, px, _ in
               eye_frames(shot, eye_geo, track, smooth, gaze_model, cam)}
    rows = []
    for rec in cp.get('per_frame') or []:
        n, fi = rec['out_frame'], rec['shot_frame']
        p = render_dir / ('%06d.exr' % n)
        if not p.exists():
            continue
        img = cv2.imread(str(p), -1)
        rgb = img[..., 2::-1].astype(np.float32)
        S = np.asarray(rec['bulb_world'], float)
        for side, e in rec['eyes'].items():
            if not e.get('rendered'):
                continue
            key = (fi, side)
            if key not in spheres:
                continue
            C, R, px = spheres[key]
            drift = float(np.linalg.norm(C - np.asarray(e['centre_world'])))
            r_px = float(e['iris_radius_px'])
            pk = peak_of(rgb, px[0], px[1], r_px)
            eye_z = C[2]
            conventions = {
                'reflection': lambda: reflect_point(C, R, cam['pos'], S)[0],
                'bulb_x_mirrored': lambda: reflect_point(C, R, cam['pos'], S * [-1, 1, 1])[0],
                'bulb_z_mirrored': lambda: reflect_point(C, R, cam['pos'],
                                                         np.array([S[0], S[1], 2 * eye_z - S[2]]))[0],
                'normal_not_reflection': lambda: C + R * (S - C) / np.linalg.norm(S - C),
            }
            preds = {k: project(f(), cam['pos'], cam['fx'], cam['W'], cam['H'])
                     for k, f in conventions.items()}
            # Does the bulb face the cap at all? The cap sees ~97 deg around its axis.
            to_bulb = S - C
            to_bulb /= np.linalg.norm(to_bulb)
            n_ref = reflect_point(C, R, cam['pos'], S)[1]
            axis = cam['pos'] - C
            axis /= np.linalg.norm(axis)
            # The cap's half-angle: round 41's 48.6 deg (sin = 5.85/7.8), or the extended cap
            # of a --globe pass, whose manifest records cap_cos.
            cap_deg = float(np.degrees(np.arccos(cp['globe']['cap_cos']))) if cp.get('globe') else 48.6
            in_cap = float(np.degrees(np.arccos(np.clip(n_ref @ axis, -1, 1)))) < cap_deg
            row = {'out_frame': n, 'shot_frame': fi, 'side': side, 'station': None,
                   'bulb_in_cap': in_cap, 'sphere_drift_m': drift,
                   'rendered_peak': None if pk is None else pk['peak'],
                   'rendered_energy': None if pk is None else pk['energy']}
            if pk is not None:
                for k, v in preds.items():
                    row['offset_px_' + k] = float(np.linalg.norm(v - pk['centroid']))
                # Chromaticity: rendered peak vs the bulb colour. Fresnel is achromatic, so
                # the rendered pixel's r:g:b must be the bulb's, up to the fill's dim wash.
                b = np.asarray(rec['bulb_rgb'], float)
                pr = np.asarray(pk['peak_rgb'], float)
                if b.sum() > 0 and pr.sum() > 0:
                    row['chroma_err'] = float(np.abs(b / b.sum() - pr / pr.sum()).max())
            rows.append(row)
    pf = {r['out_frame']: r for r in man['per_frame']}
    for r in rows:
        r['station'] = pf.get(r['out_frame'], {}).get('station')
    front = [r for r in rows if r['bulb_in_cap'] and r['rendered_peak']]
    behind = [r for r in rows if not r['bulb_in_cap']]
    summary = {'eyes_rendered': len(rows), 'eyes_with_bulb_in_cap': len(front),
               'eyes_with_bulb_behind': len(behind),
               'sphere_drift_max_m': max([r['sphere_drift_m'] for r in rows] or [0.0])}
    for k in ('reflection', 'bulb_x_mirrored', 'bulb_z_mirrored', 'normal_not_reflection'):
        v = np.array([r['offset_px_' + k] for r in front if 'offset_px_' + k in r])
        if v.size:
            summary['offset_px_' + k] = {'median': float(np.median(v)),
                                         'p95': float(np.percentile(v, 95)), 'max': float(v.max())}
    ce = np.array([r['chroma_err'] for r in front if 'chroma_err' in r])
    if ce.size:
        summary['chroma_err'] = {'median': float(np.median(ce)), 'p95': float(np.percentile(ce, 95)),
                                 'max': float(ce.max())}
    be = np.array([r['rendered_energy'] or 0.0 for r in behind])
    fe = np.array([r['rendered_energy'] or 0.0 for r in front])
    summary['energy_behind_over_front_median'] = (float(np.median(be) / max(np.median(fe), 1e-9))
                                                  if be.size and fe.size else None)
    for k, v in summary.items():
        print('RENDER', k, v, flush=True)
    return {'summary': summary, 'rows': rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_shot_argument(ap)
    ap.add_argument('--limbus-track', default='limbus_track_tv_import.json')
    ap.add_argument('--eyelid-track', default='eyelid_track_wtv40_import.json')
    ap.add_argument('--source-run', default='r40_floorB_import',
                    help='composite/<run>/source: the untouched photograph, 8-bit sRGB')
    ap.add_argument('--plate-glint', action='store_true')
    ap.add_argument('--models', default='camera,geometry')
    ap.add_argument('--render', default='', help='composite/<dir> of a --corneal pass')
    ap.add_argument('--gaze', default='camera', help='gaze model the render used')
    ap.add_argument('--globe', action='store_true',
                    help='round 42: also score the bulb reflected in the scleral globe of a '
                         '--corneal --globe pass, outside the junction, with wrong-convention '
                         'controls')
    ap.add_argument('--raw-anchor', action='store_true')
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    shot = from_args(args)
    eye_geo_meta = json.loads((shot.relight / 'eye_geometry.json').read_text(encoding='utf-8'))
    eye_geo = {r['frame']: r for r in eye_geo_meta['per_frame']}
    track = load_track(shot, args.limbus_track)
    cam = camera_of(shot)
    out = {'shot': shot.name, 'limbus_track': args.limbus_track, 'smoothed': not args.raw_anchor}
    if args.plate_glint:
        out['plate_glint'] = plate_glint_check(
            shot, eye_geo, track, cam, args.models.split(','), not args.raw_anchor,
            shot.composite / args.source_run / 'source', open_eyes(shot, args.eyelid_track))
    if args.render:
        rd = Path(args.render)
        if not rd.is_absolute():
            rd = shot.dir / rd
        out['render'] = render_check(shot, eye_geo, track, cam, rd, args.gaze, not args.raw_anchor)
        if args.globe:
            out['globe'] = globe_check(shot, eye_geo, track, cam, rd, args.gaze,
                                       not args.raw_anchor)
    if args.out:
        p = Path(args.out)
        p = p if p.is_absolute() else shot.log(args.out)
        p.write_text(json.dumps(out, indent=1), encoding='utf-8')
        print('->', p)


if __name__ == '__main__':
    main()
