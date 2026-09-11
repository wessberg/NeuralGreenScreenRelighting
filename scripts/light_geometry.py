"""Per-pixel world geometry of the plate, for lights that are CLOSE to the performer.

Round 44. A bulb 0.5-0.8 m from her face is not a distant light: across a 20 cm face the
direction to it swings by 15-25 deg, so a single light vector per frame puts the specular
lobe and the Lambert cosine on the wrong part of the face. Everything here is read from files
the pipeline already writes -- the proxy shell (`relight/proxy/*.npz`, a back-projected pixel
grid in Blender world axes with its UVs) and the neural normals -- so no stage grows a new
camera model of its own.

Axes are Blender world: +X right, +Y away from camera, +Z up; the camera sits at
(0, -actor_distance, height) looking down +Y (stress_proxy.py).
"""
import json
from pathlib import Path
import numpy as np
import cv2


def camera_world(shot):
    cam = (shot.config.get('camera') or {})
    return np.array([0.0, -float(cam.get('actor_distance_m', 1.30)), float(cam.get('height_m', 1.55))], np.float32)


def load_normals(shot, fi, shape):
    """Neural normals (world axes) resampled to `shape` = (h, w), unit length."""
    nrm = np.load(shot.relight / 'normals' / ('%06d.npz' % fi))['normal'].astype(np.float32)
    h, w = shape
    if nrm.shape[:2] != (h, w):
        nrm = cv2.resize(nrm, (w, h), interpolation=cv2.INTER_LINEAR)
    return nrm / np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-6)


def proxy_target(verts):
    """The tour's aim point, exactly as stress_proxy.py computes it from the proxy vertices."""
    zs, xs = verts[:, 2], verts[:, 0]
    head_z, mid_z = float(zs.max()), float(np.percentile(zs, 60))
    return np.array([float(np.median(xs)), 0.0, (head_z + mid_z) / 2], np.float32)


def bulb_world(shot, fi, offset, verts=None):
    if verts is None:
        verts = np.load(shot.relight / 'proxy' / ('%06d.npz' % fi))['verts'].astype(np.float32)
    return proxy_target(verts) + np.asarray(offset, np.float32)


def fill_world(shot, fi, offset, verts=None):
    """stress_proxy.py: fill at target - 0.6 * offset, 0.2 m lower."""
    if verts is None:
        verts = np.load(shot.relight / 'proxy' / ('%06d.npz' % fi))['verts'].astype(np.float32)
    t = proxy_target(verts)
    o = np.asarray(offset, np.float32)
    return np.array([t[0] - o[0] * 0.6, t[1] - o[1] * 0.6, t[2] - 0.2], np.float32)


def position_map(shot, fi, shape):
    """(h, w, 3) world position of the proxy's FRONT surface under every plate pixel.

    The front shell is a pixel grid at the proxy's stride, so each vertex lands on its own
    pixel (its UV); the map is that scatter, filled to every remaining pixel by nearest vertex
    (outside the matte the value is meaningless and only has to be finite) and smoothed over
    the stride so the stride does not print through as a staircase.
    """
    d = np.load(shot.relight / 'proxy' / ('%06d.npz' % fi))
    verts, uv, nf = d['verts'].astype(np.float32), d['uv'].astype(np.float32), int(d['n_front'])
    h, w = shape
    px = np.clip((uv[:nf, 0] * w).astype(int), 0, w - 1)
    py = np.clip(((1.0 - uv[:nf, 1]) * h).astype(int), 0, h - 1)
    grid = np.zeros((h, w, 3), np.float32)
    have = np.zeros((h, w), np.float32)
    grid[py, px] = verts[:nf]
    have[py, px] = 1.0
    # Nearest fill and smoothing at a quarter of the plate: the shell is a stride-2 grid, so
    # nothing is lost, and the distance transform costs 16x less.
    q = 4
    hq, wq = h // q, w // q
    gs = grid[:hq * q, :wq * q].reshape(hq, q, wq, q, 3).sum((1, 3))
    hs = have[:hq * q, :wq * q].reshape(hq, q, wq, q).sum((1, 3))
    gs = gs / np.maximum(hs, 1.0)[..., None]
    from scipy.ndimage import distance_transform_edt
    _, (iy, ix) = distance_transform_edt(hs == 0, return_indices=True)
    Ps = gs[iy, ix]
    Ps = cv2.GaussianBlur(Ps, (0, 0), 0.75)
    P = cv2.resize(Ps, (w, h), interpolation=cv2.INTER_LINEAR)
    return P


def light_vectors(P, light_pos, cam_pos):
    """Unit l (to the light), unit v (to the camera) and light distance, per pixel."""
    l = light_pos[None, None].astype(np.float32) - P
    dist = np.maximum(np.linalg.norm(l, axis=-1, keepdims=True), 1e-6)
    l = l / dist
    v = cam_pos[None, None].astype(np.float32) - P
    v = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-6)
    return l, v, dist[..., 0]
