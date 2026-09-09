"""Factored temporal optical hypotheses. No RGB input and no anatomy certification.

Each shared tuple constrains all observations of ONE eye. Pose and pupil size
may vary per observation. The finite grid is neither a posterior nor exhaustive.
"""
from dataclasses import replace
from itertools import product
import numpy as np
from connected_eye import trace_connected
from partial_eye_geometry import connected_candidate, project

RADII = np.round(np.arange(1.2, 3.01, .3), 8)
SHARED = np.array(list(product((7.2, 7.8, 8.4), (3.3, 3.6, 3.9),
                               (-.6, 0., .6), (-.6, 0., .6))))


def candidate(ring, shared, pupil, focal, centre):
    rc, depth, dx, dy = shared
    eye = connected_candidate(ring, focal, centre, rc, pupil)
    return replace(eye, iris_depth=depth), np.array([dx, dy])


def pupil_residuals(points, eye, offsets, radii, focal, centre):
    """Vectorized first-order boundary distances for all offsets and pupil sizes."""
    points = np.asarray(points, float)
    def trace(p): return trace_connected(p, eye, focal=focal, centre=centre)
    t = trace(points)
    uv = t['pupil_aperture_xy']
    jac = np.stack([(trace(points+d)['pupil_aperture_xy']-
                     trace(points-d)['pupil_aperture_xy'])/.1
                    for d in (np.array([.05, 0]), np.array([0, .05]))], -1)
    delta = uv[None]-np.asarray(offsets)[:, None]
    r = np.linalg.norm(delta, axis=-1)
    normal = delta/np.maximum(r[..., None], 1e-12)
    grad = np.einsum('opi,pij->opj', normal, jac)
    d = np.abs(r[:, None]-np.asarray(radii)[None, :, None]) / np.maximum(
        np.linalg.norm(grad, axis=-1)[:, None], 1e-8)
    return d, t['cap']


def build_table(observations, focal, radii=RADII):
    """Rows: shared, observation, ring, radius, max train pupil, sumsq, check0,check1.

    Only train-compatible ring profiles participate. Last two pupil points of a
    designated target are checks and excluded from every eligibility/ranking value.
    Check columns are diagnostic outputs only. No photometry is accepted here.
    """
    rows = []
    for oi, obs in enumerate(observations):
        points = np.asarray(obs['pupil'], float)
        train_n = 2 if obs['target'] else len(points)
        for ri, ring in enumerate(obs['family']):
            if not ring['compatible']: continue
            for base in range(0, len(SHARED), 9):
                eye, _ = candidate(ring, SHARED[base], 1.8, focal, obs['centre'])
                d, cap = pupil_residuals(points, eye, SHARED[base:base+9, 2:], radii,
                                         focal, obs['centre'])
                if not cap[:train_n].all(): continue
                for off in range(9):
                    for pi in range(len(radii)):
                        p = d[off, pi]
                        maximum = p[:train_n].max()
                        # Keep even incompatible profiles for fixed threshold sensitivity.
                        score = np.sum(p[:train_n]**2)+np.sum(np.array(ring['train_distances_px'])**2)
                        checks = p[2:4] if obs['target'] else np.zeros(2)
                        if obs['target'] and not cap[2:].all(): checks = np.full(2, 1e6)
                        rows.append([base+off, oi, ri, pi, maximum, score, *checks])
    return np.asarray(rows, float).reshape(-1, 8)


def joint_profiles(table, observation_count, tolerance=2., allowed=None):
    """Full factored compatible family and best train-only assignment per tuple."""
    profiles = []
    for si in range(len(SHARED)):
        if allowed is not None and not allowed(SHARED[si]): continue
        group = np.flatnonzero((table[:, 0] == si) & (table[:, 4] <= tolerance))
        choices = [group[table[group, 1] == oi] for oi in range(observation_count)]
        if any(len(c) == 0 for c in choices): continue
        best = [int(c[np.argmin(table[c, 5])]) for c in choices]
        profiles.append(dict(shared_index=si, choices=[c.tolist() for c in choices],
            best_rows=best, score=float(table[best, 5].sum())))
    return sorted(profiles, key=lambda p: (p['score'], p['shared_index']))


def iris_hit(xy, eye, offset, focal, centre):
    t = trace_connected(xy, eye, focal=focal, centre=centre)
    # Conservative base surface reachability; pupil aperture is decentered here.
    hit = (t['iris'] | t['pupil']) & (
        np.linalg.norm(t['iris_xy']-offset, axis=-1) >= eye.pupil_radius)
    return t['iris_xy'], hit


def inverse_iris(uv, eye, offset, focal, centre):
    """Invert refracted iris coordinates with batched Newton and explicit failure."""
    uv = np.asarray(uv, float).reshape(-1, 2)
    # An unrefracted iris projection can start beyond the corneal silhouette at
    # oblique gaze. Start at the visible apex, then use a descending line search.
    xy = np.broadcast_to(project(eye.apex, focal, centre), (len(uv), 2)).copy()
    for _ in range(20):
        got, _ = iris_hit(xy, eye, offset, focal, centre)
        error = np.linalg.norm(got-uv, axis=-1)
        if np.all(error < 1e-9): break
        jac = np.stack([(iris_hit(xy+d, eye, offset, focal, centre)[0]-
                         iris_hit(xy-d, eye, offset, focal, centre)[0])/.1
                        for d in (np.array([.05, 0]), np.array([0, .05]))], -1)
        det = jac[:, 0, 0]*jac[:, 1, 1]-jac[:, 0, 1]*jac[:, 1, 0]
        good = np.isfinite(det) & (np.abs(det) > 1e-10)
        safe = np.where(good, det, 1.)
        delta = got-uv
        step = np.stack([jac[:, 1, 1]*delta[:, 0]-jac[:, 0, 1]*delta[:, 1],
                         -jac[:, 1, 0]*delta[:, 0]+jac[:, 0, 0]*delta[:, 1]], -1)/safe[:, None]
        step = np.where(good[:, None], np.clip(step, -4, 4), 0)
        moved = error < 1e-9
        for scale in (1., .5, .25, .125, .0625):
            trial = xy-step*scale
            trial_uv, _ = iris_hit(trial, eye, offset, focal, centre)
            accept = ~moved & (np.linalg.norm(trial_uv-uv, axis=-1) < error)
            xy = np.where(accept[:, None], trial, xy)
            moved |= accept
            if moved.all(): break
    got, hit = iris_hit(xy, eye, offset, focal, centre)
    error = np.linalg.norm(got-uv, axis=-1)
    valid = hit & np.isfinite(xy).all(1) & (error < 1e-6)
    return xy, valid, error


def footprint(xy):
    xy = np.asarray(xy, float)
    ij = np.floor(xy).astype(int)
    return ij[:, None]+np.array([[0, 0], [1, 0], [0, 1], [1, 1]])


def box_iris(xy, eye, offset, focal, centre):
    nodes = (np.arange(3)+.5)/3-.5
    nodes = np.stack(np.meshgrid(nodes, nodes), -1).reshape(-1, 2)
    samples = np.asarray(xy)[:, None]+nodes
    return iris_hit(samples, eye, offset, focal, centre)[1].all(-1)


def donor_support(xy, safe, eye, offset, focal, centre):
    corners = footprint(xy)
    h, w = safe.shape
    inside = ((corners >= 0) & (corners < [w, h])).all((1, 2))
    clipped = np.clip(corners, [0, 0], [w-1, h-1])
    mask = safe[clipped[..., 1], clipped[..., 0]].all(1)
    material = box_iris(corners.reshape(-1, 2), eye, offset, focal, centre).reshape(-1, 4).all(1)
    return inside & mask & material


def sample_rgb(rgb, xy):
    corners = footprint(xy)
    h, w = rgb.shape[:2]
    if not ((corners >= 0) & (corners < [w, h])).all():
        raise ValueError('Out-of-image interpolation footprint')
    frac = np.asarray(xy)-np.floor(xy)
    x, y = frac.T
    weights = np.stack([(1-x)*(1-y), x*(1-y), (1-x)*y, x*y], -1)
    return np.sum(rgb[corners[..., 1], corners[..., 0]]*weights[..., None], axis=1)
