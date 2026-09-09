"""Independent scalar, optical, renderer, transaction and source-preservation checks."""
import argparse
import ast
import copy
import hashlib
import json
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from pathlib import Path

import cv2
import numpy as np
from scipy import sparse

from audit_r20_eyes import HELDOUT, SIZE, measures, sha, training_inputs, write_image, write_json
from corneal_transport import EyeOptics, interface, rotation, trace
from coupled_eye_relight import apply_coupled
from observation_eye_fit import bilinear_basis, components
from shot_paths import add_shot_argument, from_args

ROOT = Path(__file__).resolve().parents[1]


def scalar_sample(grid, uv):
    """Deliberately do not use the sparse matrix's columns or weights."""
    result = []
    for u,v in np.asarray(uv).reshape(-1, 2):
        x,y = np.clip([u,v], 0, 1)*(len(grid)-1)
        ix,iy = min(int(x),len(grid)-2), min(int(y),len(grid)-2)
        dx,dy = x-ix,y-iy
        result.append(grid[iy,ix]*(1-dx)*(1-dy)+grid[iy,ix+1]*dx*(1-dy)
                      +grid[iy+1,ix]*(1-dx)*dy+grid[iy+1,ix+1]*dx*dy)
    return np.array(result)


def geometry_checks():
    y,x = np.mgrid[19:45, 19:45]
    xy = np.stack([x,y], -1)
    maxima = dict(conic=0., normal_unit=0., reflected_unit=0., refracted_unit=0.,
                  snell=0., reflection_cosine=0.)
    for yaw,pitch in ((0,0),(12,0),(-12,0),(0,8),(0,-8)):
        eye = EyeOptics(yaw=np.deg2rad(yaw),pitch=np.deg2rad(pitch))
        t = trace(xy, eye)
        valid = t['hit']
        p,n,d,refract = [t[k][valid] for k in ('local_point','local_normal','local_direction','refracted')]
        reflect = t['reflected'][valid]@rotation(eye.yaw,eye.pitch)
        residual = p[:,0]**2+p[:,1]**2+.75*p[:,2]**2-15.6*p[:,2]
        maxima['conic'] = max(maxima['conic'],float(np.abs(residual).max()))
        for name,v in (('normal_unit',n),('reflected_unit',reflect),('refracted_unit',refract)):
            maxima[name] = max(maxima[name],float(np.abs(np.linalg.norm(v,axis=-1)-1).max()))
        sine_in = np.linalg.norm(np.cross(d,n),axis=-1)
        sine_out = np.linalg.norm(np.cross(refract,n),axis=-1)
        maxima['snell'] = max(maxima['snell'],float(np.abs(sine_in-eye.ior*sine_out).max()))
        maxima['reflection_cosine'] = max(maxima['reflection_cosine'],
            float(np.abs(np.sum(d*n,axis=-1)+np.sum(reflect*n,axis=-1)).max()))
    assert maxima['conic'] < 1e-7 and max(v for k,v in maxima.items() if k!='conic') < 1e-12
    _,_,f = interface(np.array([[0.,0.,1.]]),np.array([[0.,0.,-1.]]),1.376)
    expected = ((1.376-1)/(1.376+1))**2
    assert abs(f[0]-expected) < 1e-15
    # Signed image displacements must have reflected x/y of the same sign.
    signed = trace(np.array([[35.5,31.5],[27.5,31.5],[31.5,35.5],[31.5,27.5]]),EyeOptics())
    for row,axis,sign in ((0,0,1),(1,0,-1),(2,1,1),(3,1,-1)):
        assert signed['reflected'][row,axis]*sign > 0
        assert (signed['interior_uv'][row,axis]-.5)*sign > 0
    assert not trace(xy,EyeOptics(),lid_height=0)['visible'].any()
    # Execute only the author's geometry functions; no neural model or dependency shim.
    import torch
    author = ROOT/'dependencies/eyenerf/eyenerf/eyenerf_datamanager.py'
    names = {'unpack_pts_xyz','compute_cornea_normals','batched_ray_ellipse_intersection'}
    tree = ast.parse(author.read_text(encoding='utf-8'))
    selected = ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
    scope = {'torch':torch}
    exec(compile(selected,str(author),'exec'),scope)
    t = trace(xy,EyeOptics())
    directions = torch.tensor(t['local_direction'].reshape(-1,3),dtype=torch.float64)
    origins = torch.tensor(np.tile([0.,0.,-1300.],(len(directions),1)),dtype=torch.float64)
    pts,_,mask = scope['batched_ray_ellipse_intersection'](origins,directions)
    assert np.array_equal(mask.numpy(),t['hit'].ravel())
    parity = float(np.abs(pts.numpy()-t['local_point'].reshape(-1,3)[mask.numpy()]).max())
    normals = scope['compute_cornea_normals'](pts).numpy()
    parity_n = float(np.abs(normals-t['local_normal'].reshape(-1,3)[mask.numpy()]).max())
    assert parity < 1e-8 and parity_n < 1e-8
    return dict(max_errors=maxima,normal_incidence_fresnel=float(f[0]),
        signed_image_controls=4,signed_pose_controls=4,closed_aperture_exact=True,
        author_geometry_only=dict(source_sha256=sha(author),position_max_error=parity,normal_max_error=parity_n,
                                  full_eyenerf_executed=False))


def transaction_checks():
    rng = np.random.default_rng(20)
    source = rng.uniform(0,.4,(10,12,3)).astype(np.float32)
    mask = np.zeros((10,12),bool)
    mask[3:7,3:8] = True
    clean, reflection = source*.6, source*.1
    flags = dict(geometry_reliable=True,correspondence_reliable=True,
                 extent_reliable=True,scene_visibility_reliable=True)
    checked = 0
    for flag in flags:
        options = dict(flags)
        options[flag] = False
        # Invalid payloads cannot be touched during fallback.
        result,info = apply_coupled(source,mask,object(),object(),support=object(),**options)
        assert np.array_equal(result,source) and info['changed_channels']==0
        checked += 1
    for strength in (0.,.25,.5,1.):
        result,info = apply_coupled(source,mask,clean,reflection,support=mask, strength=strength,**flags)
        expected = source.copy()
        expected[mask] = source[mask]*(1-strength)+(clean[mask]+reflection[mask])*strength
        assert np.array_equal(result,expected) and np.array_equal(result[~mask],source[~mask])
        checked += 1
    support = mask.copy()
    support[4,4] = False
    result,info = apply_coupled(source,mask,clean,reflection,support=support,**flags)
    assert np.array_equal(result,source) and info['reason']=='incomplete_or_nonfinite_evidence'
    for array in (clean,reflection):
        contaminated = array.copy()
        contaminated[4,4] = np.nan
        result,_ = apply_coupled(source,mask,contaminated if array is clean else clean,
            contaminated if array is reflection else reflection,support=mask,**flags)
        assert np.array_equal(result,source)
        checked += 1
    try:
        apply_coupled(source,mask,clean,reflection,support=np.full(mask.shape,np.nan),**flags)
    except ValueError:
        checked += 1
    else:
        raise AssertionError('Nonboolean support accepted')
    return dict(checks=checked+1,partial_coverage_exact_fallback=True,one_coupled_taper=True)


def static_ambiguity():
    f = np.linspace(.025,.1,30)[:,None]
    interior = np.tile([.1,.15,.2],(30,1))
    environment = np.tile([2.,1.5,1.],(30,1))
    shifted_environment = environment-.5
    shifted_interior = interior+f/(1-f)*.5
    before = (1-f)*interior+f*environment
    after = (1-f)*shifted_interior+f*shifted_environment
    assert min(shifted_environment.min(),shifted_interior.min()) >= 0
    difference = float(np.abs(before-after).max())
    assert difference < 1e-15
    return dict(static_rays=30,rgb_max_difference=difference,
                interior_mean_difference=float(np.abs(interior-shifted_interior).mean()),
                scope='Exact per-ray static counterexample; not a result on moving real eyes')


def saved_fit_checks(shot):
    out = shot.composite/'r20_eyes'
    report = json.loads((shot.logs/'r20_eyes.json').read_text())
    preflight = json.loads((shot.logs/'r20_eye_preflight.json').read_text())
    assert report['protocol_sha256'] == preflight['protocol_sha256'] == sha(ROOT/'docs/r20-eye-protocol.md')
    assert all(sha(ROOT/'scripts'/k)==v for k,v in report['code_sha256'].items())
    cases = []
    for i in range(36):
        with np.load(out/'observations'/f'{i:06d}.npz') as z:
            data = dict(z)
        assert {k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in data.items()} == preflight['source_arrays'][str(i)]
        cases.append(dict(frame=i,source=data['source'],clean=data['clean'],old=data['old'],
            new=data['new'],target=data['target'],score_mask=data['score_mask'],
            transport={k[10:]:v for k,v in data.items() if k.startswith('transport_')}))
    scalar_max = 0.
    iris_only = {}
    for name,known in (('unknown_pupil',False),('oracle_black_pupil',True)):
        with np.load(out/f'{name}_fit.npz') as fit:
            values,active = fit['values'],fit['active']
        matrices,rgb = training_inputs(cases,known)
        matrix = sparse.vstack(matrices,format='csr')
        observed = np.concatenate(rgb)
        assert report['arms'][name]['observations'] == len(observed)
        assert np.array_equal(active,np.asarray(matrix.power(2).sum(0)).ravel()>0)
        mae = float(np.abs(matrix@values-observed).mean())
        assert abs(mae-report['arms'][name]['train_rgb_mae']) < 1e-15
        assert values.min()>=0 and values[:SIZE*SIZE].max()<=1 and values[SIZE*SIZE:].max()<=8
        iris_estimates,iris_truth,iris_sources = [],[],[]
        for color in ([0,0,0],[1,0,1]):
            poisoned = [dict(c,source=np.full_like(c['source'],color)) if c['frame'] in HELDOUT else c for c in cases]
            pm,prgb = training_inputs(poisoned,known)
            assert (sparse.vstack(pm,format='csr')!=matrix).nnz == 0
            assert np.array_equal(np.concatenate(prgb),observed)
        for case,record in zip(cases,report['arms'][name]['records']):
            t = case['transport']
            a = scalar_sample(values[:SIZE*SIZE].reshape(SIZE,SIZE,3),t['interior_uv'])
            e = scalar_sample(values[SIZE*SIZE:].reshape(SIZE,SIZE,3),t['environment_uv'])
            f,v = t['fresnel'].ravel()[:,None],t['visible'].ravel()[:,None]
            clean = a*(1-f)*v
            if known:
                clean *= ~t['pupil'].ravel()[:,None]
            refl = e*f*t['reflection_visible'].ravel()[:,None]
            clean,refl = clean.reshape(64,64,3),refl.reshape(64,64,3)
            expected_clean,expected_refl = components(t,values,known_black_pupil=known)
            scalar_max = max(scalar_max,float(np.abs(clean-expected_clean).max()),float(np.abs(refl-expected_refl).max()))
            for key,est,truth in (('sum_fit',clean+refl,case['source']),('interior',clean,case['clean']),
                                  ('target_with_known_scene',clean+case['new'],case['target'])):
                recomputed = measures(est,truth,case['score_mask'])
                assert recomputed['pixels']==record[key]['pixels']
                if recomputed['pixels']:
                    assert abs(recomputed['display_mae']-record[key]['display_mae'])<1e-10
            if case['frame'] in HELDOUT:
                iris_mask = case['score_mask'] & t['iris']
                iris_estimates.append(clean[iris_mask])
                iris_truth.append(case['clean'][iris_mask])
                iris_sources.append(case['source'][iris_mask])
        estimate,truth,source = [np.concatenate(a) for a in (iris_estimates,iris_truth,iris_sources)]
        iris_only[name] = dict(reconstruction=measures(estimate,truth,np.ones(len(truth),bool)),
                              source_control=measures(source,truth,np.ones(len(truth),bool)))
    assert scalar_max < 1e-14
    return dict(scalar_interpolation_max_error=scalar_max,checked_frame_arm_records=72,
                heldout_substitutions=4,scope='Exact fitter-input equality under whole-heldout-RGB substitutions; no repeated solver runs',
                supplied_geometry=True,learned_confidence_earned=False,
                supplemental_heldout_iris_only=iris_only)


def cycles_checks(shot):
    report = json.loads((shot.logs/'r20_cycles_transport.json').read_text())
    y,x = np.mgrid[:64,:64]
    xy = np.stack([x,y],-1)
    results = []
    for record in report['cases']:
        image = cv2.imread(record['path'],cv2.IMREAD_UNCHANGED)[...,::-1]
        assert image.shape==(64,64,3) and np.isfinite(image).all()
        t = trace(xy,EyeOptics(yaw=np.deg2rad(record['yaw']),pitch=np.deg2rad(record['pitch'])),lid_height=record['lid_height'])
        f = t['fresnel'][...,None]
        prediction = (1-f)*np.array(report['emitter_interior'])*t['iris'][...,None]
        prediction += f*np.array(report['emitter_environment'])*t['reflection_visible'][...,None]
        mask = cv2.erode(t['visible'].astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
        mask &= np.abs(t['interior_radius']-1.8)>.7
        # Exclude light-visibility boundaries from constant-emitter MC comparison.
        reflected = t['reflection_visible'].astype(np.uint8)
        boundaries = cv2.dilate(reflected,np.ones((3,3),np.uint8)) != cv2.erode(reflected,np.ones((3,3),np.uint8))
        mask &= ~boundaries
        result = dict(name=record['name'],image_sha256=sha(record['path']),metrics=measures(image,prediction,mask),
                      camera_visible_light_blocked=int((t['visible'] & ~t['incident_open']).sum()))
        if record['name']=='closed':
            assert np.count_nonzero(image)==0
            result['closed_exact_zero']=True
        elif mask.any():
            # Numerical transport sanity bound, not a photorealism acceptance gate.
            assert result['metrics']['linear_mae'] < .01, result
        results.append(result)
        from audit_r20_eyes import panel
        board = np.concatenate([panel(prediction,'ANALYTIC TRANSPORT',record['name']),
                                panel(image,'CYCLES 4096 SAMPLES','constant emitters; known geometry')],1)
        write_image(shot.composite/'r20_eyes/cycles'/f"{record['name']}_comparison.png",board)
    assert sum(r['camera_visible_light_blocked'] for r in results)>0
    return dict(cases=results,render_code_sha256=sha(ROOT/'scripts/render_r20_transport.py'),
                scope='Constant emitters; independent mesh/path-tracing check, not real anatomical validation')


def video_checks(shot,name,expected_count,expected_fps,shape):
    cap = cv2.VideoCapture(str(shot.final/f'{name}.mp4'))
    fps,thumbs = cap.get(cv2.CAP_PROP_FPS),[]
    maximum_frame_mae = 0.
    frame_hashes = {}
    out = shot.composite/name
    while True:
        ok,frame = cap.read()
        if not ok:
            break
        assert frame.shape==shape
        i = len(thumbs)
        original_path = out/'review'/f'{i:06d}.png'
        original = cv2.imread(str(original_path))
        assert original is not None and original.shape==shape
        frame_hashes[str(i)] = sha(original_path)
        maximum_frame_mae = max(maximum_frame_mae,float(np.abs(frame.astype(float)-original.astype(float)).mean()))
        if i in (0,5,17,23,35,49):
            write_image(out/f'decoded_{i:03d}.png',frame)
        thumb = cv2.resize(frame,(240,round(240*shape[0]/shape[1])))
        cv2.putText(thumb,str(i),(4,15),0,.45,(0,255,255),1)
        thumbs.append(thumb)
    cap.release()
    assert len(thumbs)==expected_count and abs(fps-expected_fps)<.001
    assert maximum_frame_mae < 3
    for start in range(0,len(thumbs),24):
        batch = thumbs[start:start+24]
        batch += [np.zeros_like(thumbs[0])]*(24-len(batch))
        sheet = np.concatenate([np.concatenate(batch[i:i+4],1) for i in range(0,24,4)],0)
        write_image(out/f'contact_{start:03d}.jpg',sheet)
    return dict(frames=len(thumbs),fps=fps,sha256=sha(shot.final/f'{name}.mp4'),
                maximum_decoded_frame_mae=maximum_frame_mae,source_frame_sha256=frame_hashes)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_shot_argument(ap)
    args = ap.parse_args()
    shot = from_args(args)
    from verify_r15 import fallback
    result = dict(geometry=geometry_checks(),transaction=transaction_checks(),
                  static_ambiguity=static_ambiguity(),fits=saved_fit_checks(shot),
                  cycles=cycles_checks(shot),fallback=fallback(shot))
    result['videos'] = dict(synthetic=video_checks(shot,'r20_eyes',144,24.,(762,960,3)),
                           real=video_checks(shot,'r20_source_fallback',50,float(shot.frame_rate),(732,640,3)))
    result['verifier_sha256'] = sha(Path(__file__))
    write_json(shot.logs/'r20_eye_verification.json',result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
