"""Where the corneal sphere goes, shared by the renderer and its validator.

Two things live here because two programs have to agree about them exactly, and they run
under different interpreters -- the renderer inside Blender, the validator in the pipeline
venv. If they disagree, the validator compares two different spheres and reports a
convention error that is really a bookkeeping error. (It did: moving the renderer to a
smoothed anchor without moving the validator dropped the agreement from NCC .83 to .04.)

1. **The anchor.** `eye_geometry.py` puts the corneal apex on the proxy mesh, and its own
   report says the proxy and the landmarks disagree about where the eye is by a mean of
   1.80 and a max of 3.43 native pixels. On a 12-pixel iris that is up to a quarter of the
   radius, and it walks the reflection off the iris onto the sclera. Everything else in
   the eye pipeline is anchored on the tracked landmark, so the cornea is too. Depth still
   comes from the proxy; only the direction is taken from the landmark.

2. **The smoothing.** A mirror DOUBLES angular error -- the reflected direction moves by
   twice the angle the normal does -- so a third of a pixel of landmark jitter becomes a
   couple of degrees of reflected direction, and a small bright lamp blinks on and off
   between frames. Round 33 measured this landmark's jitter against exactly this local
   quadratic fit (0.353 px rms left, 0.256 right) and established that it is noise rather
   than eye motion, so fitting it out costs no real movement.

Pure numpy; no Blender, no OpenCV.
"""
import numpy as np


def smoothed_iris_centres(eye_geo, sides=('left', 'right'), half=2):
    """Local quadratic fit of each eye's tracked iris centre, per frame.

    `eye_geo` maps frame index -> the `per_frame` record from `eye_geometry.json`.
    Returns {side: {frame: array([x, y])}}. A window of +/-2 frames: a quadratic through
    five samples follows a real saccade and rejects single-frame noise.
    """
    out = {}
    for side in sides:
        idx, pts = [], []
        for fi in sorted(eye_geo):
            e = ((eye_geo[fi].get('eyes') or {}).get(side)) or {}
            if e.get('visible') and e.get('iris_centre_px'):
                idx.append(int(fi))
                pts.append(e['iris_centre_px'])
        if len(idx) < 2 * half + 1:
            out[side] = {i: np.asarray(p, float) for i, p in zip(idx, pts)}
            continue
        t_all = np.asarray(idx, float)
        pts = np.asarray(pts, float)
        fitted = {}
        for k, fi in enumerate(idx):
            lo, hi = max(0, k - half), min(len(idx), k + half + 1)
            t = t_all[lo:hi] - t_all[k]
            deg = 2 if (hi - lo) >= 3 else 1
            v = np.polynomial.polynomial.polyvander(t, deg)
            coef, *_ = np.linalg.lstsq(v, pts[lo:hi], rcond=None)
            fitted[int(fi)] = coef[0]          # the fit evaluated at t = 0
        out[side] = fitted
    return out


def anchor_centre(cornea_centre, radius, gaze, iris_px, cam_pos, fx, width, height):
    """Move the corneal sphere onto the camera ray through `iris_px`, keeping its depth.

    `gaze` need not be normalized. Returns (centre, shift_metres); shift is how far the
    apex moved, which is the number worth logging -- it is the proxy/landmark
    disagreement expressed in metres.
    """
    centre = np.asarray(cornea_centre, float)
    gaze = np.asarray(gaze, float)
    n = np.linalg.norm(gaze)
    gaze = np.array([0.0, -1.0, 0.0]) if n < 1e-9 else gaze / n
    apex = centre + radius * gaze
    ray = np.array([(float(iris_px[0]) + 0.5 - width / 2.0) / fx, 1.0,
                    -(float(iris_px[1]) + 0.5 - height / 2.0) / fx])
    ray /= np.linalg.norm(ray)
    cam_pos = np.asarray(cam_pos, float)
    apex2 = cam_pos + ray * float(np.linalg.norm(apex - cam_pos))
    return apex2 - radius * gaze, float(np.linalg.norm(apex2 - apex))


# --- round 41 --------------------------------------------------------------------------------
# The limbus-centre anchor. `anchor_centre` puts the corneal APEX (centre + R * gaze) on the ray
# through the tracked pixel. The pixel the limbus track reports is the centre of the IRIS DISC,
# which sits on the corneal sphere's axis a distance sqrt(R^2 - r_iris^2) = 5.16 mm in front of
# the sphere centre, 2.64 mm behind the apex. With the gaze down the camera axis the two project
# to the same pixel; 18 degrees off axis they differ by 0.8 mm, about 1.7 px on this plate.
# Since the tracked point IS the iris centre, anchor that.
#
# The gaze decides where the sphere centre sits BEHIND the limbus, and the glint moves with the
# sphere centre one-for-one for a distant source. So the gaze model is not cosmetic, and
# `validate_tour_catchlight.py --plate-glint` scores the candidates against the photograph.

def anchor_limbus(depth_point, radius, iris_radius, gaze, iris_px, cam_pos, fx, width, height):
    """Corneal sphere centre such that the IRIS CENTRE lands on the camera ray through
    `iris_px` at the depth of `depth_point` (any point on the eye, e.g. the proxy apex), and
    the sphere axis follows `gaze`. Returns (centre, iris_centre_world)."""
    gaze = np.asarray(gaze, float)
    n = np.linalg.norm(gaze)
    gaze = np.array([0.0, -1.0, 0.0]) if n < 1e-9 else gaze / n
    ray = np.array([(float(iris_px[0]) + 0.5 - width / 2.0) / fx, 1.0,
                    -(float(iris_px[1]) + 0.5 - height / 2.0) / fx])
    ray /= np.linalg.norm(ray)
    cam_pos = np.asarray(cam_pos, float)
    depth = float(np.asarray(depth_point, float)[1] - cam_pos[1])      # along the camera axis
    iris_w = cam_pos + ray * (depth / ray[1])
    d = float(np.sqrt(max(radius * radius - iris_radius * iris_radius, 0.0)))
    return iris_w - d * gaze, iris_w


def reflect_point(centre, radius, cam_pos, source, at_infinity=False, iters=12):
    """Where a source reflects on a sphere toward the camera: the point p on the sphere whose
    normal bisects the directions to the camera and to the source (Alhazen, by fixed-point
    iteration -- converges in a few steps for a cap seen nearly head-on). `source` is a
    direction if `at_infinity`, else a position. Returns (p, normal)."""
    C = np.asarray(centre, float)
    cam = np.asarray(cam_pos, float)
    n = cam - C
    n /= np.linalg.norm(n)
    for _ in range(iters):
        p = C + radius * n
        v = cam - p
        v /= np.linalg.norm(v)
        if at_infinity:
            l = np.asarray(source, float)
            l = l / np.linalg.norm(l)
        else:
            l = np.asarray(source, float) - p
            l /= np.linalg.norm(l)
        n_new = v + l
        n_new /= np.linalg.norm(n_new)
        if np.linalg.norm(n_new - n) < 1e-12:
            n = n_new
            break
        n = n_new
    return C + radius * n, n


def project(p, cam_pos, fx, width, height):
    """World point -> plate pixel, the inverse of the ray in anchor_centre / eye_geometry."""
    p = np.asarray(p, float)
    cam = np.asarray(cam_pos, float)
    zc = p[1] - cam[1]
    return np.array([(p[0] - cam[0]) * fx / zc + width / 2.0 - 0.5,
                     (cam[2] - p[2]) * fx / zc + height / 2.0 - 0.5])
