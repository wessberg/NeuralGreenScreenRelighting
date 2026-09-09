"""Experimental local optical/light calibration of supplied same-eye observations.

Source-lit grids are appearance, not clean material. No compositor integration.
"""
from dataclasses import replace
import numpy as np
from scipy.optimize import least_squares
from connected_eye import trace_connected
from eye_shape_fit import box_nodes, vectors, basis, gram, fit, design

ARMS = ('fixed', 'light', 'coupled', 'spatial_backing')


def corrected(eye, focal, parameters):
    p = np.asarray(parameters, float)
    if p.shape != (4,) or not np.isfinite(p).all() or (np.abs(p) > 1.000001).any():
        raise ValueError('Four bounded finite optical corrections required')
    globe = np.array(eye.globe, float)
    globe[:2] += p[:2]*globe[2]/np.asarray(focal)
    return replace(eye, globe=tuple(globe), yaw=eye.yaw+np.deg2rad(5)*p[2],
                   pitch=eye.pitch+np.deg2rad(5)*p[3])


def compact_transport(xy, eye, offset, focal, centre, n):
    """Accumulate four bilinear weights directly, without dense per-node grids."""
    xy = np.asarray(xy, float)
    nodes = box_nodes(n); size = len(nodes)
    ray = trace_connected((xy[:, None]+nodes).reshape(-1, 2), eye,
                          focal=focal, centre=centre, pupil_offset=offset)
    f = ray['fresnel']; row = np.repeat(np.arange(len(xy)), size)
    def grid(uv, hit):
        p = uv+8; lower = np.floor(p).astype(int); frac = p-lower
        valid = ((lower >= 0) & (lower < 16)).all(-1)
        lower = np.clip(lower, 0, 15)
        a = np.zeros((len(xy), 289)); x, y = frac.T
        for (dx, dy), w in zip(((0,0),(1,0),(0,1),(1,1)),
                               ((1-x)*(1-y),x*(1-y),(1-x)*y,x*y)):
            col = (lower[:, 1]+dy)*17+lower[:, 0]+dx
            np.add.at(a, (row, col), (1-f)*hit*w/size)
        return a, valid
    iris, iv = grid(ray['iris_xy'], ray['iris'])
    backing, bv = grid(ray['backing_xy'], ray['pupil'])
    return dict(iris=iris, backing=backing,
        constant=((1-f)*ray['pupil']).reshape(len(xy), size).mean(1)[:, None],
        valid=((ray['iris'] & iv) | (ray['pupil'] & bv)).reshape(len(xy), size).all(1),
        f=f.reshape(len(xy), size), directions=ray['reflected'].reshape(len(xy), size, 3))


def light_vectors(seed, p):
    axis = np.asarray(seed, float)
    u = np.cross(axis, [0., 1., 0.])
    if np.linalg.norm(u) < .01:
        u = np.cross(axis, [1., 0., 0.])
    u /= np.linalg.norm(u); w = np.cross(axis, u)
    direction = axis+np.tan(np.deg2rad(20)*p[0])*u+np.tan(np.deg2rad(20)*p[1])*w
    direction /= np.linalg.norm(direction)
    concentration = 512*np.exp(np.log(4)*p[2])
    return np.concatenate([vectors(), (concentration*direction)[None]])


def penalty(co, m, v, p):
    return float(1e-3*np.sum(co[:m]**2)/m +
        1e-6*np.einsum('ic,ij,jc->', co[m:], gram(v)+1e-6*np.eye(len(v)), co[m:]) +
        1e-4*np.sum(p[:4]**2))


def predict(observations, focal, parameters, seed, coefficients, spatial=False, n=7):
    v = light_vectors(seed, parameters[4:]); result = []
    for o in observations:
        t = compact_transport(o['xy'], corrected(o['eye'], focal, parameters[:4]),
                              o['offset'], focal, o['centre'], n)
        x, m = design(t, v, spatial)
        result.append(dict(transport=t, matrix=x, material_count=m,
            layers=np.stack([x[:, :289]@coefficients[:289],
                             x[:, 289:m]@coefficients[289:m], x[:, m:]@coefficients[m:]])))
    return result


def solve(observations, values, focal, arm, sweeps=4, max_nfev=16):
    """No access to files or held observations; values and geometry are explicit."""
    if arm not in ARMS or not observations or len(values) != len(observations):
        raise ValueError('Matching nonempty observations/values and known arm required')
    y = np.concatenate(values)
    if y.shape != (sum(len(o['xy']) for o in observations), 3) or not np.isfinite(y).all():
        raise ValueError('Finite RGB matching every training coordinate required')
    initial = [compact_transport(o['xy'], o['eye'], o['offset'], focal, o['centre'], 7)
               for o in observations]
    directions = np.concatenate([t['directions'][:, 24] for t in initial])
    seed = directions[int(np.argmax(y.mean(1)))].copy()
    p = np.zeros(7); spatial = arm == 'spatial_backing'
    active = np.arange(4, 7) if arm == 'light' else np.arange(7)
    lower = np.full(7, -1.); upper = np.ones(7); upper[-1] = 2.
    cache_key, cache = None, None
    def matrices(params):
        nonlocal cache_key, cache
        key = tuple(params[:4])
        if key != cache_key:
            cache = [compact_transport(o['xy'], corrected(o['eye'], focal, params[:4]),
                        o['offset'], focal, o['centre'], 7) for o in observations]
            cache_key = key
        v = light_vectors(seed, params[4:])
        xs = [design(t, v, spatial) for t in cache]
        return np.concatenate([x for x, _ in xs]), xs[0][1], v, np.concatenate([t['valid'] for t in cache])
    def objective(params, co):
        x, m, v, valid = matrices(params)
        return float(np.sum((x@co-y)**2)/len(y)+penalty(co,m,v,params)+np.mean(~valid))
    x, m, v, valid = matrices(p); co = fit(x, y, m, v)
    history = [dict(objective=objective(p,co), invalid=int((~valid).sum()))]
    for sweep in range(sweeps if arm != 'fixed' else 0):
        fixed = co.copy()
        def residual(a):
            q = p.copy(); q[active] = a
            x, mc, vs, good = matrices(q)
            root = np.linalg.cholesky(gram(vs)+1e-6*np.eye(len(vs))).T
            return np.concatenate([(x@fixed-y).ravel()/np.sqrt(len(y)),
                (np.sqrt(1e-6)*root@fixed[mc:]).ravel(),
                np.sqrt(1e-4)*q[:4], (~good).astype(float)/np.sqrt(len(y))])
        # Explicit central steps avoid near-zero relative steps and silent ray
        # cancellation. This still samples a piecewise-smooth aperture objective.
        def jacobian(a):
            columns = []
            for k in range(len(a)):
                delta = np.zeros(len(a)); delta[k] = 1e-3
                columns.append((residual(a+delta)-residual(a-delta))/(2e-3))
            return np.stack(columns, 1)
        result = least_squares(residual, p[active], jac=jacobian,
            bounds=(lower[active]+.002, upper[active]-.002), max_nfev=max_nfev,
            ftol=1e-7, xtol=1e-7, gtol=1e-7)
        q = p.copy(); q[active] = result.x
        x, mc, vs, good = matrices(q); proposal = fit(x, y, mc, vs)
        before, after = objective(p, co), objective(q, proposal)
        accepted = after <= before+1e-14
        if accepted:
            p, co = q, proposal
        history.append(dict(objective=objective(p,co), accepted=accepted,
            status=int(result.status), nfev=int(result.nfev), optimality=float(result.optimality),
            invalid=int((~matrices(p)[3]).sum())))
    return dict(parameters=p, seed=seed, coefficients=co, history=history,
        bound_hits=np.flatnonzero(np.minimum(p-lower,upper-p)<.005).tolist(),
        arm=arm, source_material_validated=False)
