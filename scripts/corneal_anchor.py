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
