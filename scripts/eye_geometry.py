"""Corneal geometry per eye per frame, and the plate's own key light solved from the catchlights.

    dependencies/CorridorKey/.venv/Scripts/python.exe scripts/eye_geometry.py --shot <dir>

Two jobs, both of which exist because a catchlight is a MIRROR IMAGE OF THE SOURCE and therefore
carries more information than any other feature on a performer.

**1. Where the cornea is, so a new catchlight can be rendered into it.** Writes a corneal sphere
per eye per frame -- world-space apex, centre of curvature, radius, gaze axis, visible iris
radius, and a visibility gate. ``stress_proxy.py --catchlight`` puts a mirror there and lets
Cycles decide what the eye reflects. Nothing about that needs a threshold or an artist decision:
a bulb 0.5 m in front reflects as a bright disc, a window as a soft rectangle, an overcast sky as
a broad wash, a candle across the room as a dim speck, and an unlit corner as nothing at all.

A note on the intuition, because it is very nearly right and the exception matters: a mirror
shows the source's RADIANCE, and radiance does not fall off with distance -- so a distant lamp
still makes a catchlight, just a small one. What distance changes is the SOLID ANGLE, hence the
size. Sunlight is the extreme case: a half-degree source of enormous radiance, so it makes a
tiny brilliant spark rather than nothing, and outdoors most of what reads as "the catchlight" is
the sky, which is huge and soft. Rendering the reflection handles all of these identically;
a distance threshold would get the sun and the overcast sky exactly backwards.

**2. Where the PLATE's key light was.** The glint sits where the corneal normal equals the
half-vector between the light and the camera, so a measured glint position inverts to a light
direction -- and two eyes give two rays that intersect at the light POSITION if it was near.
Measured on the reporter plate over 264 eye-frames: the original key sat **54.6 deg off the
camera axis**, up and camera-left, with an 18.5 deg spread between samples.

The POSITION does not come out. Two corneas 63 mm apart cannot triangulate a source a metre or
more away -- the rays are near parallel and the least-squares point collapses onto the corneas
themselves, with a small residual that makes the nonsense look converged. That is gated
explicitly rather than reported, and the direction is the usable output.

Even so, the direction matters beyond the eyes, because ``estimate_source_light.py`` fits SH to
the visible surface only, where every normal has ny < 0, and its own log warns that the resulting
dominant direction is an extrapolation which must not be read as a measurement. The catchlights
are the one place in the frame where the plate photographed its light sources directly.

WHY THE ANGLE TOLERANCE IS TIGHTER THAN IT LOOKS
------------------------------------------------
The glint moves by HALF the angle the light moves, because it tracks the half-vector. That
halving is the whole reason a photographed catchlight is ever reusable. It is not enough here:
keeping the plate's own catchlight is only defensible while the new key is within roughly 15 deg
of the measured original, and relighting into a different environment is precisely the case where
it is not. So the composite REPLACES by default and keeps only when the measured angle
difference says keeping is free. That decision is computed from this file and the rendered
lighting, not set by hand.

Shape matters as well as angle. This plate's glint is a visibly rectangular softbox; a bare
practical in the target scene should read as a small round disc. Keeping a rectangle under a
bulb is wrong even when the angle happens to agree, which is the other reason replace is the
default.

WHAT IS ASSUMED, AND WHAT IS MEASURED
-------------------------------------
Assumed: the corneal radius of curvature, 7.8 mm, and the visible iris radius, 5.85 mm. These are
human anatomical constants with about 5% population spread, not per-shot tuning -- everything
that scales with the individual (pixels per mm, gaze, aperture) is measured from the plate.

Measured per frame: the iris centre and rim, which give the pixels-per-mm scale directly.

GAZE IS RESOLUTION-DEPENDENT, AND THAT IS MEASURED RATHER THAN ASSUMED
----------------------------------------------------------------------
A circle seen off-axis images as an ellipse whose minor/major ratio is cos(angle), so in
principle the iris outline reads gaze off the plate with no gaze model at all. In practice it
needs a big iris. On this 1080p bust the iris radius is 12.92 +- 0.41 px, and the ellipse
estimator returned a gaze of **29.3 deg off axis with only +-2 deg of scatter** for a reporter
looking straight down the lens -- a stable, confident, wrong answer, because one pixel of
landmark asymmetry on a 13 px circle is already 23 deg of apparent rotation. It poisoned the
light solve downstream (per-eye light directions scattered by 40 deg) before it was caught.

So gaze comes from whichever estimator the frame can actually support, and the choice is
recorded per eye in ``gaze_source``:

* ``ellipse`` -- only when the iris is at least ``--ellipse-min-radius`` px (default 40). This is
  the artist's macro close-up case, where it is the best estimator available and needs no head
  model at all.
* ``headpose_iris`` -- mediapipe's facial transformation matrix gives the head's forward axis
  robustly at any resolution, and the iris centre's displacement from the eye-corner midpoint,
  divided by the eyeball radius, adds the eye's own rotation within the head. This is the
  default for normal framing, and it handles a performer turned to profile, because the head
  axis is measured rather than assumed frontal.
* ``camera_axis`` -- last resort when there is no head pose: the eye is assumed to look at the
  lens.

All available estimates and their disagreement are written out, so a later stage can see when
they diverge instead of inheriting one silently.

**ANAMORPHIC.** On the artist's 1.6x squeezed plate an iris is an ellipse before the eye has
rotated at all, and reading gaze off that ellipse without desqueezing reports a permanent
sideways stare. x is desqueezed here before any circular fit; ``scripts/anamorphic.py`` holds
the factor.

Limits worth knowing at macro scale: the cornea is a prolate ellipsoid rather than a sphere, and
flattens toward the limbus, so the sphere model degrades at large glint offsets; and there are
dimmer secondary reflections off the lens surfaces (Purkinje III/IV) that this does not model.
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
# Anatomical constants. ~5% population spread; nothing here is tuned per shot.
CORNEA_RADIUS_MM = 7.8
IRIS_RADIUS_MM = 5.85
EYEBALL_RADIUS_MM = 12.0
# mediapipe refined-landmark indices. 468-472 is the SUBJECT's right iris, 473-477 the left.
IRIS = {'right': 468, 'left': 473}
# Inner and outer canthus per eye. Their midpoint is where the optical axis pierces the opening
# when the eye is centred in the head, so the iris centre's displacement from it is eye rotation.
CANTHI = {'right': (133, 33), 'left': (362, 263)}


def squeeze_of(shot):
    a = (shot.config.get('camera') or {}).get('anamorphic')
    if isinstance(a, dict):
        return float(a.get('squeeze', 1.6))
    return 1.6 if a else 1.0


def camera_model(shot):
    """fx in PLATE pixels, plus the distance and height the proxy was built against.

    Deliberately the same numbers make_proxy_mesh.py used, read from the same place: if the two
    disagreed the corneal sphere would float off the rendered head and the error would look like
    a tracking failure rather than a units mismatch.
    """
    cfg = shot.config
    cam = cfg.get('camera') or {}
    src_w, src_h = cfg.get('source_resolution', [1920, 1080])
    lens = float(cam.get('lens_mm', 50))
    sensor = float(cam.get('sensor_width_mm', 36))
    return {'fx': src_w * lens / sensor, 'src_w': src_w, 'src_h': src_h,
            'dist': float(cam.get('actor_distance_m', 1.30)),
            'cam_h': float(cam.get('height_m', 1.55)),
            'lens_mm': lens, 'sensor_width_mm': sensor}


def project(verts, cm, squeeze):
    """World (Blender axes) -> plate pixels. The inverse of make_proxy_mesh.build's mapping."""
    Zc = np.maximum(verts[:, 1] + cm['dist'], 1e-4)
    px = verts[:, 0] * cm['fx'] / Zc + cm['src_w'] / 2 - 0.5
    py = (cm['cam_h'] - verts[:, 2]) * cm['fx'] / Zc + cm['src_h'] / 2 - 0.5
    return np.stack([px * squeeze, py], 1), Zc


def unproject(px, py, Zc, cm, squeeze):
    """Plate pixels + camera-space depth -> world (Blender axes)."""
    x = (px / squeeze - cm['src_w'] / 2 + 0.5) * Zc / cm['fx']
    z = cm['cam_h'] - (py - cm['src_h'] / 2 + 0.5) * Zc / cm['fx']
    return np.array([x, Zc - cm['dist'], z], np.float64)


def iris_ellipse(pts, base, squeeze):
    """(centre_px, radius_px, gaze_unit_in_camera_axes) from the four rim points.

    The iris is a circle on the eyeball. Seen off-axis it images as an ellipse whose minor axis
    shrinks as cos(theta) and whose MAJOR axis is the axis the eye rotated about, so the ellipse
    alone gives the gaze direction with no gaze model and no head-pose dependency. Desqueezed
    first, or an anamorphic plate reports a permanent sideways stare.
    """
    c = pts[base, :2].astype(np.float64).copy()
    rim = pts[base + 1:base + 5, :2].astype(np.float64).copy()
    c[0] /= squeeze
    rim[:, 0] /= squeeze
    d = rim - c
    r = np.linalg.norm(d, axis=1)
    a_i = int(np.argmax(r))
    major, minor = float(r[a_i]), float(r[(a_i + 1) % 4])
    if major < 1e-6:
        return None
    # The two rim pairs are orthogonal on the iris, so the smaller is the foreshortened one.
    minor = float(min(r[(a_i + 1) % 4], r[(a_i + 3) % 4]))
    cos_t = float(np.clip(minor / major, 0.05, 1.0))
    axis = d[a_i] / major                       # rotation axis, in image coordinates
    # Gaze tilts away from the camera axis, perpendicular to the rotation axis. Camera looks
    # along +Y in world; -Y is "toward the camera". Sign of the tilt is ambiguous from one
    # ellipse alone (a circle tilted left and right image identically); it is resolved below
    # against the iris centre's offset within the eyelid opening.
    perp = np.array([-axis[1], axis[0]])
    sin_t = float(np.sqrt(max(1.0 - cos_t * cos_t, 0.0)))
    return c, major, cos_t, axis, perp, sin_t


def headpose_forward(M):
    """Head forward axis in WORLD axes from mediapipe's facial transformation matrix.

    mediapipe works in an OpenGL-ish camera frame (+x right, +y up, +z toward the viewer); this
    project uses +x right, +y away from camera, +z up. So (x, y, z) -> (x, -z, y). A head facing
    the lens has a mediapipe forward of +z, which maps to (0, -1, 0) -- toward the camera, as it
    should be. Verified by that identity rather than assumed.
    """
    if M is None or not np.all(np.isfinite(M)):
        return None
    f = np.asarray(M, float)[:3, 2]
    w = np.array([f[0], -f[2], f[1]], float)
    n = np.linalg.norm(w)
    return w / n if n > 1e-9 else None


def gaze_estimates(pts, side, base, c_dsq, r_major, cos_t, axis, perp, sin_t,
                   M, squeeze, args):
    """Every gaze estimate this frame supports, plus which one to use and why."""
    est = {}
    # 1. Ellipse. Magnitude from the aspect ratio; the sign of the tilt is ambiguous from one
    #    ellipse (a circle tilted toward and away images identically), so it is resolved by the
    #    iris centre's displacement within the eye opening.
    i_in, i_out = CANTHI[side]
    m = pts[[i_in, i_out]][:, :2].astype(np.float64).mean(0)
    m[0] /= squeeze
    disp = c_dsq - m
    g_img = perp * sin_t
    if float(g_img @ disp) < 0:
        g_img = -g_img
    ell = np.array([g_img[0], -cos_t, -g_img[1]], float)
    ell /= max(np.linalg.norm(ell), 1e-9)
    est['ellipse'] = ell

    # 2. Head pose + the eye's own rotation inside it. sin(rotation) = displacement / eyeball
    #    radius, which needs no per-shot constant because mm_per_px comes from the same iris.
    fwd = headpose_forward(M)
    if fwd is not None:
        mm = IRIS_RADIUS_MM / max(r_major, 1e-6)
        d_mm = disp * mm
        sr = np.clip(np.linalg.norm(d_mm) / EYEBALL_RADIUS_MM, 0.0, np.sin(np.radians(45)))
        if np.linalg.norm(d_mm) > 1e-9:
            u = d_mm / np.linalg.norm(d_mm)
            # Displacement is in image axes (+x right, +y down) -> world (+x right, +z up).
            t = np.array([u[0], 0.0, -u[1]], float)
            t -= fwd * (t @ fwd)
            if np.linalg.norm(t) > 1e-9:
                t /= np.linalg.norm(t)
                g = fwd * np.sqrt(max(1 - sr * sr, 0.0)) + t * sr
                est['headpose_iris'] = g / max(np.linalg.norm(g), 1e-9)
            else:
                est['headpose_iris'] = fwd
        else:
            est['headpose_iris'] = fwd

    est['camera_axis'] = np.array([0.0, -1.0, 0.0])

    if r_major >= args.ellipse_min_radius:
        src = 'ellipse'
    elif 'headpose_iris' in est:
        src = 'headpose_iris'
    else:
        src = 'camera_axis'
    keys = list(est)
    dis = 0.0
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            dis = max(dis, float(np.degrees(np.arccos(
                np.clip(est[keys[a]] @ est[keys[b]], -1, 1)))))
    return est[src], src, {k: np.round(v, 4).tolist() for k, v in est.items()}, dis


def eye_records(shot, lmk, cm, squeeze, args):
    pts, found, ap = lmk['points'], lmk['found'], lmk['eyelid_aperture']
    blend = lmk['blendshapes']
    n = len(found)
    proxy_dir = shot.relight / 'proxy'
    rows = []
    for i in range(n):
        rec = {'frame': int(i), 'found': bool(found[i])}
        if not found[i]:
            rows.append(rec)
            continue
        mp = proxy_dir / ('%06d.npz' % i)
        verts = nf = None
        if mp.exists():
            z = np.load(mp)
            verts = z['verts'].astype(np.float64)[:int(z['n_front'])]
            nf, Zc_v = project(verts, cm, squeeze)
        fg = cv2.imread(str(shot.fg(i)), -1)
        fg = fg[..., ::-1].astype(np.float32) if fg is not None else None
        for side, base in IRIS.items():
            e = iris_ellipse(pts[i], base, squeeze)
            if e is None:
                continue
            c_dsq, r_major, cos_t, axis, perp, sin_t = e
            c_px = np.array([c_dsq[0] * squeeze, c_dsq[1]])       # back to plate pixels
            mm_per_px = IRIS_RADIUS_MM / max(r_major, 1e-6)
            d = {'side': side,
                 'iris_centre_px': [float(c_px[0]), float(c_px[1])],
                 'iris_radius_px_desqueezed': float(r_major),
                 'mm_per_px': float(mm_per_px),
                 'ellipse_cos_to_camera': float(cos_t),
                 'ellipse_off_axis_deg': float(np.degrees(np.arccos(cos_t))),
                 'eyelid_aperture_iris_radii': float(np.nan_to_num(
                     ap[i][0 if side == 'left' else 1], nan=-1.0)),
                 'blink': float(np.nan_to_num(
                     blend[i][0 if side == 'left' else 1], nan=-1.0))}
            # Visible: lids open enough to show cornea, and the eye not turned so far that the
            # cornea is edge-on. Both gates, because eyeBlink saturates late.
            d['visible'] = bool(d['eyelid_aperture_iris_radii'] > args.min_aperture
                                and d['blink'] < args.max_blink)
            # Surface point: the front proxy vertex whose projection is nearest the iris centre.
            # Using the MESH rather than re-deriving depth guarantees the corneal sphere sits on
            # the surface Cycles will actually render, not 3 mm inside or outside it.
            if verts is not None and len(verts):
                k = int(np.argmin(((nf - c_px) ** 2).sum(1)))
                apex = verts[k].copy()
                d['proxy_hit_px_error'] = float(np.linalg.norm(nf[k] - c_px))
            else:
                apex = unproject(c_px[0], c_px[1], cm['dist'], cm, squeeze)
                d['proxy_hit_px_error'] = None
            # Gaze axis in world, from whichever estimator this frame's resolution supports.
            gaze, gsrc, gall, gdis = gaze_estimates(
                pts[i], side, base, c_dsq, r_major, cos_t, axis, perp, sin_t,
                lmk['xform'][i], squeeze, args)
            d['gaze_world'] = np.round(gaze, 5).tolist()
            d['gaze_source'] = gsrc
            d['gaze_off_axis_deg'] = float(np.degrees(np.arccos(
                np.clip(-gaze[1], -1, 1))))
            if d['gaze_off_axis_deg'] > args.max_gaze_off_axis_deg:
                # The cornea is far enough round that the sphere model and the glint
                # measurement both degrade, and the far eye is usually partly occluded.
                d['visible'] = False
            d['gaze_candidates'] = gall
            d['gaze_estimator_disagreement_deg'] = round(gdis, 2)
            d['cornea_apex_world'] = apex.round(6).tolist()
            d['cornea_centre_world'] = (apex - gaze * (CORNEA_RADIUS_MM / 1000.0)).round(6).tolist()
            d['cornea_radius_m'] = CORNEA_RADIUS_MM / 1000.0
            d['iris_radius_m'] = float(IRIS_RADIUS_MM / 1000.0)
            # The plate's OWN glint, measured, so the composite can decide keep vs replace and
            # so the light solve below has something to invert.
            if fg is not None and d['visible']:
                g = measure_glint(fg, c_px, r_major * squeeze, mm_per_px)
                if g:
                    d['plate_glint'] = g
            rec.setdefault('eyes', {})[side] = d
        rows.append(rec)
    return rows


def measure_glint(fg, c_px, rad_px, mm_per_px):
    """Position, size and peak of the plate's own catchlight inside the iris disc."""
    L = fg @ LUMA
    x0, y0 = int(c_px[0] - rad_px * 1.6), int(c_px[1] - rad_px * 1.6)
    s = int(rad_px * 3.2)
    if x0 < 0 or y0 < 0 or y0 + s > fg.shape[0] or x0 + s > fg.shape[1] or s < 6:
        return None
    pat, pl = fg[y0:y0 + s, x0:x0 + s], L[y0:y0 + s, x0:x0 + s]
    yy, xx = np.mgrid[0:s, 0:s]
    disc = ((xx + x0 - c_px[0]) ** 2 + (yy + y0 - c_px[1]) ** 2) <= (rad_px * 1.1) ** 2
    if disc.sum() < 30:
        return None
    v = pl[disc]
    m = disc & (pl >= np.percentile(v, 97)) & (pl > 3.0 * np.median(v))
    if m.sum() < 3:
        return None
    wy, wx = np.nonzero(m)
    ww = pl[m]
    gx, gy = x0 + float((wx * ww).sum() / ww.sum()), y0 + float((wy * ww).sum() / ww.sum())
    return {'centre_px': [gx, gy],
            'offset_px': [gx - float(c_px[0]), gy - float(c_px[1])],
            'offset_mm': [(gx - float(c_px[0])) * mm_per_px, (gy - float(c_px[1])) * mm_per_px],
            'area_px': int(m.sum()),
            'peak_linear': float(pat[m].max()),
            'clipped': bool(pat[m].max() >= 0.995),
            'contrast_over_iris_median': float(pat[m].max(-1).max() / max(np.median(v), 1e-6))}


def solve_plate_light(rows, cm, squeeze, args_min_reach=10.0):
    """Invert the measured glints into a light direction per eye, then intersect the two.

    A glint at world point p on a corneal sphere of centre C has surface normal h = (p-C)/R, and
    h is by definition the half-vector between the light and the view. So l = 2(h.v)h - v.
    Distant source: both eyes report the same l. Near source: they differ, and the two rays
    p + t*l intersect at the lamp.
    """
    dirs, origins, per_eye = [], [], []
    for r in rows:
        for side, d in (r.get('eyes') or {}).items():
            g = d.get('plate_glint')
            if not g or not d.get('visible'):
                continue
            C = np.array(d['cornea_centre_world'], float)
            R = d['cornea_radius_m']
            # Back-project the glint pixel onto the corneal sphere: the near intersection of the
            # camera ray with the sphere.
            ray = unproject(g['centre_px'][0], g['centre_px'][1], 1.0, cm, squeeze)
            o = np.array([0.0, -cm['dist'], cm['cam_h']])       # camera centre in world
            u = ray - o
            u /= max(np.linalg.norm(u), 1e-12)
            oc = o - C
            b = 2 * u @ oc
            c2 = oc @ oc - R * R
            disc = b * b - 4 * c2
            if disc <= 0:
                continue
            t = (-b - np.sqrt(disc)) / 2.0
            p = o + t * u
            h = (p - C) / R
            v = (o - p)
            v /= max(np.linalg.norm(v), 1e-12)
            l = 2.0 * (h @ v) * h - v
            l /= max(np.linalg.norm(l), 1e-12)
            dirs.append(l)
            origins.append(p)
            per_eye.append({'frame': r['frame'], 'side': side,
                            'light_dir_world': l.round(4).tolist(),
                            'angle_from_camera_axis_deg': float(np.degrees(
                                np.arccos(np.clip(l @ np.array([0.0, -1.0, 0.0]), -1, 1)))),
                            'corneal_half_angle_deg': float(np.degrees(
                                np.arccos(np.clip(h @ np.array(d['gaze_world'], float), -1, 1))))})
    if not dirs:
        return None
    D = np.array(dirs)
    O = np.array(origins)
    mean_dir = D.mean(0)
    mean_dir /= max(np.linalg.norm(mean_dir), 1e-12)
    spread = float(np.degrees(np.arccos(np.clip(D @ mean_dir, -1, 1))).std())
    # Least-squares intersection of all the rays. Well conditioned only if they are not
    # parallel, which is exactly the near-source case; the residual says which regime this is.
    A = np.zeros((3, 3))
    bb = np.zeros(3)
    for o, d in zip(O, D):
        P = np.eye(3) - np.outer(d, d)
        A += P
        bb += P @ o
    pos, res = None, None
    try:
        pos = np.linalg.solve(A + 1e-9 * np.eye(3), bb)
        res = float(np.mean([np.linalg.norm((np.eye(3) - np.outer(d, d)) @ (pos - o))
                             for o, d in zip(O, D)]))
    except np.linalg.LinAlgError:
        pass
    # Whether the triangulation means anything at all. The two corneas are ~63 mm apart, so the
    # rays are near parallel for any source more than a few tens of centimetres away, and the
    # least-squares point then collapses onto the rays' own origins -- i.e. onto the performer's
    # head -- with a SMALL residual. A residual test alone therefore passes on a nonsense answer,
    # which is exactly what it did here before this gate existed.
    baseline = float(np.linalg.norm(O.max(0) - O.min(0))) if len(O) > 1 else 0.0
    reach = float(np.linalg.norm(pos - O.mean(0))) if pos is not None else 0.0
    degenerate = pos is None or baseline < 1e-6 or reach < args_min_reach * baseline
    return {'samples': len(D),
            'mean_direction_world': mean_dir.round(4).tolist(),
            'direction_spread_deg': spread,
            'angle_from_camera_axis_deg': float(np.degrees(np.arccos(
                np.clip(mean_dir @ np.array([0.0, -1.0, 0.0]), -1, 1)))),
            'triangulation': {
                'status': 'DEGENERATE -- do not use' if degenerate else 'usable',
                'position_world': None if degenerate or pos is None else pos.round(4).tolist(),
                'raw_position_world': None if pos is None else pos.round(4).tolist(),
                'residual_m': res,
                'ray_baseline_m': baseline,
                'solved_reach_over_baseline': (reach / baseline) if baseline > 1e-6 else None,
                'why': 'the two corneas are ~63 mm apart, so for any source beyond a few tens of '
                       'centimetres the rays are near parallel and the least-squares point '
                       'collapses onto the corneas themselves WITH A SMALL RESIDUAL. A residual '
                       'test alone passes on that nonsense; this gate is reach over baseline.'},
            'direction_is_the_usable_output':
                'mean_direction_world and its spread are the measurement to consume. The spread '
                'is not pure noise: both eyes see the same source, so genuine parallax between '
                'the two corneas inflates it when the source is near.',
            'why_it_matters':
                'This is the only place in the frame where the plate photographed its own light '
                'sources directly. estimate_source_light.py fits SH to a surface whose normals '
                'all have ny < 0 and its own log warns that the resulting direction is an '
                'extrapolation; this is not.',
            'per_sample': per_eye[:40]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_shot_argument(ap)
    ap.add_argument('--min-aperture', type=float, default=0.9,
                    help='eyelid gap in iris radii below which the cornea is too covered to use')
    ap.add_argument('--ellipse-min-radius', type=float, default=40.0,
                    help='iris radius in px below which the ellipse gaze readout is not trusted. '
                         'At 12.9 px it returned a confident, stable 29.3 deg for a subject '
                         'looking down the lens; one pixel of asymmetry on a small circle is '
                         'already ~23 deg')
    ap.add_argument('--max-gaze-off-axis-deg', type=float, default=66.0,
                    help='beyond this the cornea is near edge-on, the sphere model degrades and '
                         'the far eye is usually part-occluded')
    ap.add_argument('--max-blink', type=float, default=0.5)
    args = ap.parse_args()
    shot = from_args(args)

    p = shot.relight / 'face_landmarks.npz'
    if not p.exists():
        raise SystemExit('No %s. Run scripts/face_track.py first (facetrack venv).' % p)
    lmk = np.load(p)
    cm = camera_model(shot)
    squeeze = squeeze_of(shot)
    rows = eye_records(shot, lmk, cm, squeeze, args)
    light = solve_plate_light(rows, cm, squeeze)

    vis = sum(1 for r in rows for d in (r.get('eyes') or {}).values() if d.get('visible'))
    glints = [d['plate_glint'] for r in rows for d in (r.get('eyes') or {}).values()
              if d.get('plate_glint')]
    err = [d['proxy_hit_px_error'] for r in rows for d in (r.get('eyes') or {}).values()
           if d.get('proxy_hit_px_error') is not None]
    out = {
        'shot': shot.name, 'frames': len(rows),
        'camera': cm, 'anamorphic_squeeze': squeeze,
        'anatomy': {'cornea_radius_mm': CORNEA_RADIUS_MM, 'iris_radius_mm': IRIS_RADIUS_MM,
                    'status': 'ASSUMED anatomical constants, ~5% population spread. Everything '
                              'that varies with the individual -- px per mm, gaze, aperture -- '
                              'is measured from the plate.'},
        'world_axes': '+X right, +Y away from camera, +Z up; actor centred at Y=0 '
                      '(identical to make_proxy_mesh.py, on purpose)',
        'visible_eye_frames': vis,
        'proxy_projection_error_px': {
            'mean': float(np.mean(err)) if err else None,
            'max': float(np.max(err)) if err else None,
            'meaning': 'distance between the iris centre and the projection of the proxy vertex '
                       'used as the corneal apex. Large values mean the mesh and the landmarks '
                       'disagree about where the eye is, and the catchlight will sit off it.'},
        'plate_glint': {
            'measured_on_eye_frames': len(glints),
            'mean_area_px': float(np.mean([g['area_px'] for g in glints])) if glints else None,
            'mean_peak_linear': float(np.mean([g['peak_linear'] for g in glints])) if glints else None,
            'clipped_fraction': float(np.mean([g['clipped'] for g in glints])) if glints else None,
            'mean_contrast_over_iris': float(np.mean(
                [g['contrast_over_iris_median'] for g in glints])) if glints else None,
            'clipping_note': 'an unclipped glint still has the iris diffuse recorded underneath '
                             'it, so separate_specular.py can subtract it and leave real '
                             'photographed texture. A clipped one cannot be recovered at all.'},
        'plate_key_light': light,
        'per_frame': rows}
    (shot.relight / 'eye_geometry.json').write_text(json.dumps(out, indent=2))
    shot.log('eye_geometry.json').write_text(json.dumps(
        {k: v for k, v in out.items() if k != 'per_frame'}, indent=2))
    print(json.dumps({k: v for k, v in out.items()
                      if k in ('frames', 'visible_eye_frames', 'proxy_projection_error_px',
                               'plate_glint')}, indent=2))
    if light:
        print(json.dumps({k: light[k] for k in
                          ('samples', 'mean_direction_world', 'direction_spread_deg',
                           'angle_from_camera_axis_deg', 'triangulation')}, indent=2))


if __name__ == '__main__':
    main()
