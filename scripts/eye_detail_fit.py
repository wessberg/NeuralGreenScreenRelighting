"""Observed RGB layer fit with explicit numerical and material-support limits.

This module accepts supplied transport only. Numerical convergence and repeated
coefficient support are necessary diagnostics, never proof of clean eye material.
"""
import numpy as np
from scipy import sparse
from scipy.optimize import minimize, linprog
from observation_eye_fit import bilinear_basis


def design(transports, iris_size=64, pupil_size=12, environment_size=32,
           pupil_radius=1.8, iris_radius=5.85):
    """Average physical subray operators before fitting native observed pixels."""
    result = None
    for t in transports:
        f, v, p = (t[k].ravel() for k in ('fresnel', 'visible', 'pupil'))
        uv = t['interior_uv'].reshape(-1, 2)
        blocks = [bilinear_basis(uv, iris_size).multiply((v*(1-f)*~p)[:, None]),
                  bilinear_basis((uv-.5)*iris_radius/pupil_radius+.5, pupil_size)
                  .multiply((v*(1-f)*p)[:, None]),
                  bilinear_basis(t['environment_uv'], environment_size)
                  .multiply((f*t['reflection_visible'].ravel())[:, None])]
        matrix = sparse.hstack(blocks, format='csr')
        result = matrix if result is None else result+matrix
    result = (result/len(transports)).tocsr()
    result.eliminate_zeros()
    return result


def upper_bounds(iris_size=64, pupil_size=12, environment_size=32):
    return np.r_[np.ones(iris_size**2+pupil_size**2), np.full(environment_size**2, 8.)]


def projected_gradient(matrix, rgb, values, upper):
    scale = np.sqrt(np.asarray(matrix.power(2).sum(0)).ravel())
    active = scale > 0
    normalized = matrix[:, active].multiply(1/scale[active]).tocsr()
    z, bound = values[active]*scale[active, None], upper[active]*scale[active]
    gradient = normalized.T@(matrix@values-rgb)
    # Gradient mapping accounts for distance to a bound, not a guessed active set.
    return np.max(np.abs(z-np.clip(z-gradient, 0, bound[:, None])), axis=0)


def fit(matrices, observations, upper):
    matrix = sparse.vstack(matrices, format='csr')
    rgb = np.concatenate(observations).reshape(-1, 3)
    if matrix.shape != (len(rgb), len(upper)) or not np.isfinite(rgb).all():
        raise ValueError('Finite RGB and matching transport required')
    scale = np.sqrt(np.asarray(matrix.power(2).sum(0)).ravel())
    active = scale > 0
    normalized = matrix[:, active].multiply(1/scale[active]).tocsr()
    transpose = normalized.T.tocsr()
    bound = upper[active]*scale[active]
    values = np.zeros((len(upper), 3))
    results = []
    for channel in range(3):
        observed = rgb[:, channel]
        def objective(z):
            residual = normalized@z-observed
            return .5*np.dot(residual, residual), transpose@residual
        result = minimize(objective, np.zeros(active.sum()), jac=True, method='L-BFGS-B',
            bounds=list(zip(np.zeros(len(bound)), bound)), options=dict(
                maxiter=20000, maxfun=50000, maxcor=30, ftol=0., gtol=1e-8))
        values[active, channel] = result.x/scale[active]
        results.append(dict(success=bool(result.success), message=str(result.message),
            iterations=int(result.nit), evaluations=int(result.nfev), cost=float(result.fun)))
    pg = projected_gradient(matrix, rgb, values, upper)
    for result, error in zip(results, pg):
        result.update(projected_gradient=float(error), certified=bool(error <= 1e-7))
    frames = np.sum([np.asarray(m.power(2).sum(0)).ravel()>0 for m in matrices], axis=0)
    return values, frames, dict(solvers=results, observations=len(rgb),
        active_parameters=int(active.sum()), total_parameters=len(upper),
        train_linear_mae=float(np.abs(matrix@values-rgb).mean()),
        train_max_abs=float(np.abs(matrix@values-rgb).max()),
        certified=all(r['certified'] for r in results))


def complete_support(matrix, frame_counts, minimum=3):
    """Every nonzero basis coefficient, including reflection, must be observed."""
    presence = matrix.copy()
    presence.eliminate_zeros()
    presence.data[:] = 1
    return (np.asarray(presence@(frame_counts < minimum)).ravel() == 0)


def interval(matrix, observed, upper, query, radius):
    """Conditional feasible-model interval, including unobserved coefficients.

    Both extremal coefficient vectors are returned for independent verification.
    A narrow result is not a statistical or real-material confidence guarantee.
    """
    constraints = sparse.vstack([matrix, -matrix], format='csr')
    rhs = np.r_[observed+radius, -observed+radius]
    answers, witnesses = [], []
    for sign in (1, -1):
        result = linprog(sign*query, A_ub=constraints, b_ub=rhs,
            bounds=list(zip(np.zeros(len(upper)), upper)), method='highs')
        if not result.success:
            return dict(success=False, message=result.message), None
        witness = result.x
        violation = max(float(np.max(np.abs(matrix@witness-observed)-radius)),
                        float(-witness.min()), float(np.max(witness-upper)), 0.)
        if violation > 1e-6:
            raise AssertionError(f'Infeasible uncertainty witness: {violation}')
        answers.append(float(query@witness))
        witnesses.append(witness)
    return dict(success=True, lower=answers[0], upper=answers[1],
        width=answers[1]-answers[0], rejected=bool(answers[1]-answers[0] > .01)), np.array(witnesses)
