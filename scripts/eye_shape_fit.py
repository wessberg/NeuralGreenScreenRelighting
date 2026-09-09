"""Experimental supplied-geometry reflection shape; no production layer certification."""
import numpy as np
from scipy.optimize import nnls
from connected_eye import trace_connected
from eye_appearance_fit import DIRECTIONS, material_basis


def box_nodes(n):
    if not isinstance(n, int) or n < 1:
        raise ValueError('Positive integer sampling count required')
    a = (np.arange(n)+.5)/n-.5
    return np.stack(np.meshgrid(a, a), -1).reshape(-1, 2)


def vectors(training=None, count=24):
    base = np.concatenate([np.zeros((1, 3)), 8*DIRECTIONS, 32*DIRECTIONS])
    if training is None:
        return base
    d = np.concatenate(training)
    if d.ndim != 2 or d.shape[1] != 3 or not len(d):
        raise ValueError('Training unit directions required')
    if not np.isfinite(d).all() or not np.allclose(np.linalg.norm(d, axis=1), 1):
        raise ValueError('Invalid training directions')
    selected = [0]
    distance = np.full(len(d), np.inf)
    for _ in range(min(count, len(d))-1):
        distance = np.minimum(distance, np.sum((d-d[selected[-1]])**2, axis=1))
        selected.append(int(np.argmax(distance)))
    centres = d[selected]
    return np.concatenate([base, 128*centres, 512*centres])


def basis(d, v):
    return np.exp(np.minimum(np.asarray(d)@v.T-np.linalg.norm(v, axis=1), 0.))


def gram(v):
    k = np.linalg.norm(v, axis=1)
    total = k[:, None]+k[None]
    r = np.linalg.norm(v[:, None]+v[None], axis=-1)
    return np.where(r > 1e-10,
        (np.exp(r-total)-np.exp(-r-total))/(2*np.maximum(r, 1e-30)), np.exp(-total))


def fit(x, y, material_count, v, reflected=True):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.shape != (len(y), material_count+len(v)) or y.shape != (len(x), 3):
        raise ValueError('Incompatible appearance/light/RGB dimensions')
    if not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all() or (x < -1e-14).any():
        raise ValueError('Finite nonnegative design and nonempty observations required')
    size = x.shape[1] if reflected else material_count
    root = np.zeros((size, size))
    root[:material_count, :material_count] = np.eye(material_count)*np.sqrt(1e-3/material_count)
    if reflected:
        root[material_count:, material_count:] = np.linalg.cholesky(
            1e-6*(gram(v)+1e-6*np.eye(len(v)))).T
    a = np.concatenate([x[:, :size]/np.sqrt(len(x)), root])
    b = np.concatenate([y/np.sqrt(len(x)), np.zeros((size, 3))])
    scale = np.linalg.norm(a, axis=0)
    co = np.zeros((x.shape[1], 3))
    co[:size] = np.stack([nnls(a/scale, b[:, c], maxiter=30000)[0]/scale for c in range(3)], 1)
    return co


def transport(xy, eye, offset, focal, centre, n):
    nodes = box_nodes(n)
    ray = trace_connected((np.asarray(xy)[:, None]+nodes).reshape(-1, 2), eye,
                          focal=focal, centre=centre, pupil_offset=offset)
    m, mv = material_basis(ray['iris_xy'])
    b, bv = material_basis(ray['backing_xy'])
    f = ray['fresnel'][:, None]
    iris = ((1-f)*m*ray['iris'][:, None]).reshape(len(xy), len(nodes), -1).mean(1)
    backing = ((1-f)*b*ray['pupil'][:, None]).reshape(len(xy), len(nodes), -1).mean(1)
    constant = ((1-f[:, 0])*ray['pupil']).reshape(len(xy), len(nodes)).mean(1)[:, None]
    good = ((ray['iris'] & mv) | (ray['pupil'] & bv)).reshape(len(xy), len(nodes)).all(1)
    return dict(iris=iris, backing=backing, constant=constant, valid=good,
        f=f.reshape(len(xy), len(nodes)), directions=ray['reflected'].reshape(len(xy), len(nodes), 3))


def design(t, v, spatial=False):
    a = np.concatenate([t['iris'], t['backing'] if spatial else t['constant']], 1)
    r = (t['f'][..., None]*basis(t['directions'], v)).mean(1)
    return np.concatenate([a, r], 1), a.shape[1]


def occupied(training, query, m):
    seen = np.sum([np.any(x[:, :m] > 1e-12, axis=0) for x in training], axis=0)
    return np.all((query[:, :m] <= 1e-12) | (seen >= 2), axis=1)
