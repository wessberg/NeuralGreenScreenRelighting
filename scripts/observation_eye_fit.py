"""Sparse two-layer eye fitting from observed RGB and supplied transport only.

No clean texture, reflection map, pretrained decoder or generative prior is input.
Bounds and an optional known-black pupil are explicit assumptions. A successful
RGB fit is NOT evidence that the decomposition or geometry is correct.
"""
import numpy as np
from scipy import sparse
from scipy.optimize import lsq_linear


def bilinear_basis(uv, size):
    uv = np.asarray(uv, float).reshape(-1, 2)
    xy = np.clip(uv, 0, 1)*(size-1)
    lower = np.minimum(np.floor(xy).astype(int), size-2)
    frac = xy-lower
    x, y = lower.T
    a, b = frac.T
    cols = np.stack([y*size+x, y*size+x+1, (y+1)*size+x, (y+1)*size+x+1], 1)
    weights = np.stack([(1-a)*(1-b), a*(1-b), (1-a)*b, a*b], 1)
    rows = np.broadcast_to(np.arange(len(uv))[:, None], cols.shape)
    return sparse.csr_matrix((weights.ravel(), (rows.ravel(), cols.ravel())), shape=(len(uv), size*size))


def design(transport, size=16, *, known_black_pupil=False):
    f = transport['fresnel'].reshape(-1)
    vis = transport['visible'].reshape(-1)
    transmitted = vis*(1-f)
    if known_black_pupil:
        transmitted = transmitted*(~transport['pupil'].reshape(-1))
    reflected = f*transport['reflection_visible'].reshape(-1)
    a = bilinear_basis(transport['interior_uv'], size).multiply(transmitted[:, None])
    e = bilinear_basis(transport['environment_uv'], size).multiply(reflected[:, None])
    return sparse.hstack([a, e], format='csr')


def fit_observations(matrices, observations, *, size=16):
    matrix = sparse.vstack(matrices, format='csr')
    rgb = np.concatenate(observations).reshape(-1, 3)
    if matrix.shape != (len(rgb), 2*size*size) or not np.isfinite(rgb).all():
        raise ValueError('Observation/transport dimensions differ or RGB is nonfinite')
    active = np.asarray(matrix.power(2).sum(0)).ravel() > 0
    reduced = matrix[:, active]
    upper = np.r_[np.ones(size*size), np.full(size*size, 8.)][active]
    values, solver = np.zeros((2*size*size, 3)), []
    for ch in range(3):
        result = lsq_linear(reduced, rgb[:, ch], bounds=(0, upper), tol=1e-9,
                            max_iter=200, lsmr_tol=1e-10)
        values[active, ch] = result.x
        solver.append(dict(success=bool(result.success), status=int(result.status),
            iterations=int(result.nit), cost=float(result.cost), optimality=float(result.optimality)))
    return dict(values=values, active=active, solver=solver, observations=len(rgb),
                active_parameters=int(active.sum()), train_rgb_mae=float(np.abs(matrix@values-rgb).mean()))


def components(transport, values, size=16, *, known_black_pupil=False):
    matrix = design(transport, size, known_black_pupil=known_black_pupil)
    shape = (*transport['visible'].shape, 3)
    interior = (matrix[:, :size*size]@values[:size*size]).reshape(shape)
    reflection = (matrix[:, size*size:]@values[size*size:]).reshape(shape)
    return interior, reflection
