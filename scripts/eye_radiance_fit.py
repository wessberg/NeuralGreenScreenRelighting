"""Exposure-unit radiance regularization and explicit angular sampling coverage.

This experimental model fits source-lit appearance, not clean material. Angular
proximity does not establish layer identifiability. No production integration.
"""
import numpy as np
from scipy.optimize import nnls
from scipy.spatial import cKDTree
from connected_eye import trace_connected
from eye_appearance_fit import BOX, DIRECTIONS, light_basis, material_basis, MATERIAL_COLUMNS

APPEARANCE = MATERIAL_COLUMNS + 1  # Unknown constant recessed backing radiance.


def sphere_gram():
    """Exact (1/4pi) integral b_i(w)b_j(w) dw for constant + SG lobes."""
    vectors = np.concatenate([np.zeros((1, 3)), 8*DIRECTIONS, 32*DIRECTIONS])
    concentration = np.linalg.norm(vectors, axis=1)
    total = concentration[:, None] + concentration[None]
    radius = np.linalg.norm(vectors[:, None] + vectors[None], axis=-1)
    # exp(-total)*sinh(radius)/radius, evaluated without cancellation at zero.
    return np.where(radius > 1e-10,
        (np.exp(radius-total)-np.exp(-radius-total))/(2*np.maximum(radius, 1e-30)),
        np.exp(-total))


def penalty_root(material_count, reflection_ridge):
    gram = sphere_gram() + np.eye(53)*1e-6
    # R.T @ R = gram; use the full radiance field, including cross terms.
    return np.block([[np.eye(material_count)*np.sqrt(1e-3/material_count),
                      np.zeros((material_count, 53))],
                     [np.zeros((53, material_count)),
                      np.sqrt(reflection_ridge)*np.linalg.cholesky(gram).T]])


def fit_radiance(matrix, rgb, reflection_ridge, material_count=APPEARANCE):
    x, y = np.asarray(matrix, float), np.asarray(rgb, float)
    if x.ndim != 2 or x.shape[1] != material_count+53 or y.shape != (len(x), 3):
        raise ValueError('Expected appearance + 53 light columns and RGB rows')
    if not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all() or (x < 0).any():
        raise ValueError('Finite nonnegative design and nonempty observations required')
    if reflection_ridge is not None and (not np.isfinite(reflection_ridge) or reflection_ridge <= 0):
        raise ValueError('Radiance penalty must be positive')
    n = material_count if reflection_ridge is None else x.shape[1]
    root = (np.eye(n)*np.sqrt(1e-3/material_count) if reflection_ridge is None
            else penalty_root(material_count, reflection_ridge))
    augmented = np.concatenate([x[:, :n]/np.sqrt(len(x)), root])
    rhs = np.concatenate([y/np.sqrt(len(x)), np.zeros((n, 3))])
    scale = np.linalg.norm(augmented, axis=0)
    coefficients = np.zeros((x.shape[1], 3))
    coefficients[:n] = np.stack([nnls(augmented/scale, rhs[:, c], maxiter=20000)[0]/scale
                                for c in range(3)], 1)
    return coefficients


def angular_support(train_directions, query, degrees=5., min_frames=2):
    """Every query BOX node needs nearby directions in distinct supplied frames.

    Inputs: one (N, nodes, 3) unit-direction array per frame and query of same
    layout. Caller must supply only actually observed training footprints.
    """
    q = np.asarray(query, float)
    if q.ndim != 3 or q.shape[-1] != 3 or not np.isfinite(q).all():
        raise ValueError('Expected finite pixel x BOX x unit directions')
    if not 0 < degrees < 180 or min_frames < 1:
        raise ValueError('Invalid angular sampling criterion')
    if not np.allclose(np.linalg.norm(q, axis=-1), 1, atol=1e-7):
        raise ValueError('Directions must be unit length')
    distance = 2*np.sin(np.deg2rad(degrees)/2)
    counts = np.zeros(q.shape[:-1], int)
    nearest = []
    for directions in train_directions:
        d = np.asarray(directions, float).reshape(-1, 3)
        if not np.isfinite(d).all() or not np.allclose(np.linalg.norm(d, axis=-1), 1, atol=1e-7):
            raise ValueError('Training directions must be finite unit vectors')
        values = (cKDTree(d).query(q.reshape(-1, 3))[0].reshape(q.shape[:-1])
                  if len(d) else np.full(q.shape[:-1], np.inf))
        counts += values <= distance
        nearest.append(values)
    return (counts >= min_frames).all(1), counts


def design_with_backing(xy, eye, offset, focal, centre):
    xy = np.asarray(xy, float)
    samples = (xy[:, None]+BOX).reshape(-1, 2)
    ray = trace_connected(samples, eye, focal=focal, centre=centre, pupil_offset=offset)
    mat, grid = material_basis(ray['iris_xy'])
    f = ray['fresnel'][:, None]
    appearance = (1-f)*mat*ray['iris'][:, None]
    backing = (1-f)*ray['pupil'][:, None]
    reflected = f*light_basis(ray['reflected'])
    matrix = np.concatenate([appearance, backing, reflected], 1).reshape(len(xy), len(BOX), -1).mean(1)
    valid = ((ray['iris'] & grid) | ray['pupil']).reshape(len(xy), len(BOX)).all(1)
    return matrix, valid, ray['reflected'].reshape(len(xy), len(BOX), 3), ray['pupil'].reshape(len(xy), len(BOX)).mean(1)


def appearance_support(training, query):
    coverage = np.sum([np.any(d[:, :APPEARANCE] > 1e-12, axis=0) for d in training], axis=0)
    return np.all((query[:, :APPEARANCE] <= 1e-12) | (coverage >= 2), axis=1)
