"""Observed-only layer fitting with an explicit, unknown-radiance pupil.

Supplied geometry is not validated here. No successful solve certifies material.
Separate pupil coordinates avoid smoothing across a known material boundary.
"""
import numpy as np
from scipy import sparse
from scipy.optimize import lsq_linear
from observation_eye_fit import bilinear_basis


def design(transport, size=12, *, split_pupil=True, pupil_radius=1.8, iris_radius=5.85):
    f = transport['fresnel'].ravel()
    v = transport['visible'].ravel()
    p = transport['pupil'].ravel()
    uv = transport['interior_uv'].reshape(-1, 2)
    weight = v*(1-f)
    if split_pupil:
        interior = bilinear_basis(uv, size).multiply((weight*~p)[:, None])
        pupil_uv = (uv-.5)*iris_radius/pupil_radius+.5
        pupil = bilinear_basis(pupil_uv, size).multiply((weight*p)[:, None])
        blocks = [interior, pupil]
    else:
        blocks = [bilinear_basis(uv, size).multiply(weight[:, None])]
    blocks.append(bilinear_basis(transport['environment_uv'], size).multiply(
        (f*transport['reflection_visible'].ravel())[:, None]))
    return sparse.hstack(blocks, format='csr')


def fit(matrices, observations, *, size=12, split_pupil=True):
    matrix = sparse.vstack(matrices, format='csr')
    rgb = np.concatenate(observations).reshape(-1, 3)
    interior_columns = size*size*(2 if split_pupil else 1)
    columns = interior_columns+size*size
    if matrix.shape != (len(rgb), columns) or not np.isfinite(rgb).all():
        raise ValueError('Finite observed RGB and matching transport required')
    active = np.asarray(matrix.power(2).sum(0)).ravel() > 0
    upper = np.r_[np.ones(interior_columns), np.full(size*size, 8.)]
    values = np.zeros((columns, 3))
    solvers = []
    for ch in range(3):
        result = lsq_linear(matrix[:, active], rgb[:, ch], bounds=(0, upper[active]),
                            tol=1e-9, lsmr_tol=1e-10, max_iter=200)
        values[active, ch] = result.x
        solvers.append(dict(success=bool(result.success), status=int(result.status),
            iterations=int(result.nit), cost=float(result.cost), optimality=float(result.optimality)))
    return values, active, dict(solvers=solvers, observations=len(rgb),
        active_parameters=int(active.sum()), total_parameters=columns,
        train_linear_mae=float(np.abs(matrix@values-rgb).mean()))


def components(transport, values, size=12, **kwargs):
    matrix = design(transport, size, **kwargs)
    split = values.shape[0]-size*size
    shape = (*transport['visible'].shape, 3)
    clean = (matrix[:, :split]@values[:split]).reshape(shape)
    reflection = (matrix[:, split:]@values[split:]).reshape(shape)
    return clean, reflection
