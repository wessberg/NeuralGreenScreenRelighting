"""Independent scalar spherical-eye transport and narrow-light sampling controls."""
import argparse
import math
from dataclasses import replace
import numpy as np
from numpy.polynomial.legendre import leggauss
from connected_eye import ConnectedEye, trace_connected
from eye_shape_fit import box_nodes, vectors, basis, gram, fit
from audit_r24_source import write, sha, ROOT
from shot_paths import add_shot_argument, from_args


def scalar(xy, globe):
    # Independent sphere quadratic; spherical cornea R=8, limbus=5.85.
    sag = 8-math.sqrt(64-5.85**2)
    depth = sag+math.sqrt(144-5.85**2)
    apex = [globe[0], globe[1], globe[2]-depth]
    centre = [apex[0], apex[1], apex[2]+8]
    d = [(xy[0]-47.5)/3000, (xy[1]-47.5)/3000, 1.]
    norm = math.sqrt(sum(a*a for a in d)); d = [a/norm for a in d]
    b = -sum(a*c for a, c in zip(d, centre))
    c = sum(a*a for a in centre)-64
    t = -b-math.sqrt(b*b-c)
    hit = [t*a for a in d]
    n = [(a-c)/8 for a, c in zip(hit, centre)]
    cosine = -sum(a*b for a, b in zip(d, n))
    ct = math.sqrt(1-(1-cosine*cosine)/1.376**2)
    refracted = [a/1.376+(cosine/1.376-ct)*b for a, b in zip(d, n)]
    reflected = [a+2*cosine*b for a, b in zip(d, n)]
    rs = (cosine-1.376*ct)/(cosine+1.376*ct)
    rp = (1.376*cosine-ct)/(1.376*cosine+ct)
    f = (rs*rs+rp*rp)/2
    ti = (apex[2]+3.6-hit[2])/refracted[2]
    uv = [hit[k]+ti*refracted[k]-apex[k] for k in (0, 1)]
    pupil = sum(a*a for a in uv) < 1.8**2
    return np.array(reflected), f, pupil


def independent(xy, globe, n, axis):
    result, layers = [], []
    for point in xy:
        a = np.zeros(3); r = np.zeros(3)
        # Separate quadrature construction from the production node helper.
        offsets = [np.array([(ix+.5)/n-.5, (iy+.5)/n-.5])
                   for iy in range(n) for ix in range(n)]
        for delta in offsets:
            d, f, pupil = scalar(point+delta, globe)
            a += (1-f)*(np.array([.015,.018,.021]) if pupil else np.array([.08,.12,.07]))
            dot = sum(float(x)*float(y) for x, y in zip(d, axis))
            r += f*(np.array([.06,.05,.04])+np.array([12.,9.,6.])*math.exp(512*(dot-1)))
        layers.append([a/(n*n), r/(n*n)])
        result.append((a+r)/(n*n))
    return np.array(result), np.array(layers)


def fixture_design(xy, eye, n, v):
    q = (xy[:, None]+box_nodes(n)).reshape(-1, 2)
    ray = trace_connected(q, eye)
    assert (ray['iris'] | ray['pupil']).all()
    f = ray['fresnel']
    a = np.stack([(1-f)*ray['iris'], (1-f)*ray['pupil']], 1)
    x = np.concatenate([a, f[:, None]*basis(ray['reflected'], v)], 1)
    return x.reshape(len(xy), n*n, -1).mean(1)


def controls():
    axis = np.array([-.09, .11, -1.]); axis /= np.linalg.norm(axis)
    narrow = np.concatenate([vectors(), 512*axis[None]])
    # Rotate independent quadrature so the narrow lobe sits at a polar axis.
    u = np.cross(axis, [0,1,0]); u /= np.linalg.norm(u); w = np.cross(axis, u)
    z, weights = leggauss(192); phi = (np.arange(384)+.5)*2*np.pi/384
    ds = (np.sqrt(1-z[:,None,None]**2)*(np.cos(phi)[None,:,None]*u
        + np.sin(phi)[None,:,None]*w)+z[:,None,None]*axis).reshape(-1,3)
    bb = basis(ds, narrow)
    actual = bb.T@(bb*np.repeat(weights/(2*384),384)[:,None])
    ge = float(np.max(np.abs(actual-gram(narrow))))
    assert ge < 1e-11
    eye = ConnectedEye(cornea_radius=8., eccentricity=0., lid_height=13., lid_width=13.)
    a = np.arange(-6,7,2); xx, yy = np.meshgrid(a, a)
    xy = np.stack([xx.ravel()+47.5, yy.ravel()+47.5], 1)
    maximum = 0.
    for shift in ((0,0),(.7,0),(-.7,0),(0,.7),(0,-.7)):
        e = replace(eye, globe=(shift[0], shift[1], 1313.))
        ray = trace_connected(xy, e)
        for i, point in enumerate(xy):
            d, f, pupil = scalar(point, e.globe)
            maximum = max(maximum, float(np.max(np.abs(d-ray['reflected'][i]))), abs(f-ray['fresnel'][i]))
            assert pupil == ray['pupil'][i]
    assert maximum < 1e-8
    dense, _ = independent(xy, eye.globe, 41, axis)
    integration = {}
    known = np.zeros((56,3)); known[:2] = [[.08,.12,.07],[.015,.018,.021]]
    known[2] = [.06,.05,.04]; known[-1] = [12,9,6]
    for n in (3,7,21):
        pred = fixture_design(xy, eye, n, narrow)@known
        truth, _ = independent(xy, eye.globe, n, axis)
        equality = float(np.max(np.abs(pred-truth)))
        assert equality < 1e-8
        integration[n] = dict(scalar_max_error=equality, dense41_max_error=float(np.max(np.abs(pred-dense))))
    assert integration[21]['dense41_max_error'] < .001
    # Same physical fixture, three fitted translations and a fourth held position.
    eyes = [replace(eye, globe=(x,y,1313.)) for x,y in ((-.7,0),(0,.7),(.7,0),(0,-.7))]
    truth = [independent(xy, e.globe, 21, axis) for e in eyes]
    models = {}
    for name, v in [('broad',vectors()), ('narrow',narrow)]:
        xs = [fixture_design(xy, e, 21, v) for e in eyes]
        co = fit(np.concatenate(xs[:3]), np.concatenate([t[0] for t in truth[:3]]), 2, v)
        pred = np.stack([xs[3][:,:2]@co[:2],xs[3][:,2:]@co[2:]],1)
        models[name] = dict(held_sum_max_error=float(np.max(np.abs(pred.sum(1)-truth[3][0]))),
            held_layer_max_error=float(np.max(np.abs(pred-truth[3][1]))),
            held_sum_mse=float(np.mean((pred.sum(1)-truth[3][0])**2)))
    # Independent per-observation material explanation: zero light, spatially varying
    # radiance chosen to match each sum. It need not be one shared physical texture.
    source, physical_layers = truth[3]
    no_reflection = np.stack([source, np.zeros_like(source)], 1)
    ambiguity = float(np.max(np.abs(no_reflection-physical_layers)))
    assert np.array_equal(no_reflection.sum(1), source) and ambiguity > .01
    return dict(scalar_signed_transport_max_error=maximum, signed_cases=5,
        sphere_gram_max_error=ge, integration=integration, independent_held_fit=models,
        single_view_spatial_material_sum_error=0., single_view_layer_difference=ambiguity,
        single_view_explanation_not_a_shared_temporal_texture=True,
        independent_truth_uses_production_ray_functions=False, performer_material_certified=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__); add_shot_argument(ap); shot=from_args(ap.parse_args())
    result=controls(); result['code_sha256']=sha(ROOT/'scripts/verify_r29_fixture.py')
    write(shot.logs/'r29_fixture.json',result); print('R29_FIXTURE', result, flush=True)


if __name__ == '__main__':
    main()
