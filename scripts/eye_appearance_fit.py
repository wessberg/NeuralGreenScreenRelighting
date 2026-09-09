"""Experimental source-lit iris appearance plus distant reflected radiance.

No clean material, calibrated illumination, anatomical confidence or production
repair is inferred here. Coefficients are fitted only from supplied observations.
"""
from itertools import product
import numpy as np
from scipy.optimize import nnls
from connected_eye import trace_connected
from joint_eye_geometry import iris_hit

GRID = 17
MATERIAL_COLUMNS = GRID*GRID
DIRECTIONS = np.array([p for p in product((-1., 0., 1.), repeat=3) if any(p)])
DIRECTIONS /= np.linalg.norm(DIRECTIONS, axis=1)[:, None]
BOX = np.stack(np.meshgrid(*[(np.arange(3)+.5)/3-.5]*2), -1).reshape(-1, 2)


def light_basis(direction):
    dot = np.asarray(direction)@DIRECTIONS.T
    return np.concatenate([np.ones((*dot.shape[:-1], 1)),
                           np.exp(8*(dot-1)), np.exp(32*(dot-1))], -1)


def material_basis(uv):
    """Bilinear grid; out-of-grid coordinates explicitly unsupported."""
    uv = np.asarray(uv, float); p = uv+8
    lower = np.floor(p).astype(int); frac = p-lower
    valid = ((lower >= 0) & (lower < GRID-1)).all(-1)
    lower = np.clip(lower, 0, GRID-2)
    result = np.zeros((len(uv), MATERIAL_COLUMNS))
    x, y = frac.T
    for delta, weight in zip(((0,0),(1,0),(0,1),(1,1)),
                            ((1-x)*(1-y),x*(1-y),(1-x)*y,x*y)):
        ij = lower+delta
        result[np.arange(len(uv)), ij[:,1]*GRID+ij[:,0]] += weight
    return result, valid


def design(xy, eye, offset, focal, centre):
    """Full native BOX integration of both paths; no RGB input."""
    xy = np.asarray(xy, float); samples = (xy[:,None]+BOX).reshape(-1,2)
    ray = trace_connected(samples, eye, focal=focal, centre=centre)
    uv, hit = iris_hit(samples, eye, offset, focal, centre)
    mat, grid_valid = material_basis(uv)
    f = ray['fresnel'][:,None]
    # Fixed eta scale is absorbed into source-lit A; A is not albedo.
    a = ((1-f)*mat).reshape(len(xy),len(BOX),-1).mean(1)
    # Open reflected visibility is an explicit unvalidated model assumption.
    b = (f*light_basis(ray['reflected'])).reshape(len(xy),len(BOX),-1).mean(1)
    valid = (hit & grid_valid).reshape(len(xy),len(BOX)).all(1)
    return np.concatenate([a,b],1), valid


def observed_support(train_designs, test_design):
    coverage = np.sum([np.any(d[:,:MATERIAL_COLUMNS] > 1e-12, axis=0)
                       for d in train_designs], axis=0)
    used = test_design[:,:MATERIAL_COLUMNS] > 1e-12
    return np.all(~used | (coverage[None] >= 2), axis=1)


def fit(design_matrix, rgb, reflection_ridge):
    """Convex NNLS, column-normalized ridge; None removes the reflected arm."""
    n = MATERIAL_COLUMNS if reflection_ridge is None else design_matrix.shape[1]
    x = design_matrix[:,:n]
    scale = np.linalg.norm(x, axis=0)
    active = scale > 1e-12
    normalized = x[:,active]/scale[active]
    penalty = np.full(n, 1e-3)
    if reflection_ridge is not None: penalty[MATERIAL_COLUMNS:] = reflection_ridge
    aug = np.concatenate([normalized, np.diag(np.sqrt(penalty[active]))])
    rhs = np.concatenate([rgb, np.zeros((active.sum(),3))])
    coef = np.zeros((design_matrix.shape[1],3))
    coef[np.flatnonzero(active)] = np.stack([
        nnls(aug, rhs[:,channel], maxiter=10000)[0]/scale[active] for channel in range(3)],1)
    return coef


def layers(design_matrix, coefficients):
    m = MATERIAL_COLUMNS
    return design_matrix[:,:m]@coefficients[:m], design_matrix[:,m:]@coefficients[m:]
