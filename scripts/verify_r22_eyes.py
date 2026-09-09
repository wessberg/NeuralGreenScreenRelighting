"""Independent sampling, numerical/LP witnesses, RGB isolation and exact fallback."""
import argparse
import ast
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy import sparse

import audit_r22_eyes as audit
import eye_detail_fit as detail
from audit_r20_eyes import measures, sha, write_json
from verify_r21_eyes import scalar_grid
from verify_r20_eyes import transaction_checks, video_checks
from shot_paths import add_shot_argument, from_args


def independent_layers(values, options, frame, *, texture_pixels=False):
    si,sp,se = options['iris_size'],options.get('pupil_size',12),options.get('environment_size',32)
    a,b = si**2,si**2+sp**2
    iris,pupil,environment = values[:a].reshape(si,si,3),values[a:b].reshape(sp,sp,3),values[b:].reshape(se,se,3)
    clean,old = np.zeros((64,64,3)),np.zeros((64,64,3))
    for t in audit.geometry(frame):
        uv = t['interior_uv']
        base = scalar_grid(iris,uv,image_pixels=texture_pixels)
        pup = scalar_grid(pupil,(uv-.5)*5.85/1.8+.5,image_pixels=texture_pixels)
        clean += np.where(t['pupil'][...,None],pup,base)*(t['visible']*(1-t['fresnel']))[...,None]/4
        old += scalar_grid(environment,t['environment_uv'],image_pixels=texture_pixels)*(
            t['fresnel']*t['reflection_visible'])[...,None]/4
    return clean,old


def renderer(shot, frames):
    manifest = audit.manifest(shot)
    assert manifest['pixel_filter_type']=='BLACKMAN_HARRIS'
    assert abs(manifest['pixel_filter_width']-.01)<1e-8
    p = audit.ROOT/'scripts/render_r22_eyes.py'
    node = next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='textures')
    scope = {'np':np}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),scope)
    iris,pupil,old,_ = scope['textures']()
    values = np.concatenate([a.reshape(-1,3) for a in (iris,pupil,old)])
    cases = []
    for i in frames:
        c,r = independent_layers(values,dict(iris_size=512,pupil_size=512,environment_size=512),i,
                                  texture_pixels=True)
        clean,source = audit.load(shot,[i],'clean')[i],audit.load(shot,[i])[i]
        case = dict(frame=i,clean=measures(c,clean,audit.masks(i)['full_visible']),
            source=measures(c+r,source,audit.masks(i)['full_visible']))
        for k in ('clean','source'):
            if case[k]['pixels']:
                assert case[k]['linear_mae']<.01,case
        cases.append(case)
    return dict(cases=cases,quadrature_offset=audit.filter_offset(),
        scope='Independent fixture texture lookup; approximate footprint, shared simplified optical geometry')


def numerical_controls():
    rng = np.random.default_rng(2201)
    matrix = sparse.csr_matrix(rng.uniform(.02,1,(40,7))*np.array([1,.1,2,.03,1,2,.5]))
    truth = rng.uniform(.1,.8,(7,3))
    values,counts,info = detail.fit([matrix], [matrix@truth],np.ones(7))
    error = float(np.max(np.abs(values-truth)))
    assert info['certified'] and error<1e-6
    # A sum is perfectly known while its two nonnegative terms remain ambiguous.
    ambiguous = sparse.csr_matrix([[1.,1.,0.]])
    result,witness = detail.interval(ambiguous,np.array([.5]),np.ones(3),np.array([1.,0.,0.]),0.)
    assert result['success'] and abs(result['width']-.5)<1e-8
    # An unseen requested coefficient must remain free in uncertainty, never zero-filled.
    unseen,w = detail.interval(ambiguous,np.array([.5]),np.ones(3),np.array([0.,0.,1.]),0.)
    assert unseen['success'] and abs(unseen['width']-1.)<1e-8
    support = detail.complete_support(sparse.eye(3,format='csr'),np.array([3,2,0]))
    assert np.array_equal(support,[True,False,False])
    return dict(full_rank_max_coefficient_error=error,converged=info['certified'],
        indistinguishable_sum_interval=result['width'],unseen_coefficient_interval=unseen['width'])


def fits(shot):
    out = shot.composite/'r22_eyes'
    report = json.loads((shot.logs/'r22_eyes.json').read_text())
    pre = json.loads((shot.logs/'r22_preflight.json').read_text())
    assert report['code_sha256']==pre['code_sha256']
    assert all(sha(audit.ROOT/'scripts'/k)==v for k,v in report['code_sha256'].items())
    assert report['protocol_sha256']==pre['protocol_sha256']==sha(audit.ROOT/'docs/r22-eye-protocol.md')
    assert not set(audit.TRAIN)&set(audit.SELECTION+audit.TEST) and not set(audit.SELECTION)&set(audit.TEST)
    sources = audit.load(shot,range(36))
    predictions,maximum,substitutions = {},0.,0
    kkt, support_counts = {},{}
    for name,options in audit.ARMS.items():
        data = np.load(out/f'{name}_fit.npz')
        values,counts = data['values'],data['frame_counts']
        upper = detail.upper_bounds(**options)
        assert np.min(values)>=-1e-12 and np.max(values-upper[:,None])<=1e-12
        assert np.count_nonzero(values[counts==0])==0
        matrices,rgb = audit.training_inputs(sources,options)
        matrix,observed = sparse.vstack(matrices,format='csr'),np.concatenate(rgb)
        independent_counts = np.zeros(len(values),int)
        for m in matrices:
            independent_counts[np.unique(m.indices[m.data!=0])] += 1
        assert np.array_equal(counts,independent_counts)
        # Recompute normalized gradient mapping directly from saved coefficients.
        norm = np.sqrt(np.asarray(matrix.power(2).sum(0)).ravel())
        active = norm>0
        gradient = (matrix[:,active].T@(matrix@values-observed))/norm[active,None]
        z = values[active]*norm[active,None]
        pg = np.max(np.abs(z-np.clip(z-gradient,0,(upper[active]*norm[active])[:,None])),axis=0)
        assert np.allclose(pg,[s['projected_gradient'] for s in report['arms'][name]['solvers']],atol=1e-12,rtol=0)
        kkt[name] = pg.tolist()
        for color in ([0,0,0],[1,0,1]):
            altered = {i:np.full_like(s,color) if i not in audit.TRAIN else s for i,s in sources.items()}
            m,b = audit.training_inputs(altered,options)
            assert (sparse.vstack(m,format='csr')!=matrix).nnz==0 and np.array_equal(np.concatenate(b),observed)
            substitutions += 1
        class TrainingOnly:
            def __getitem__(self,index):
                assert index in audit.TRAIN
                return sources[index]
        _,guarded = audit.training_inputs(TrainingOnly(),options)
        assert np.array_equal(np.concatenate(guarded),observed)
        pred = dict(np.load(out/f'{name}_predictions.npz'))
        predictions[name] = pred
        for i in range(36):
            clean,old = independent_layers(values,options,i)
            maximum = max(maximum,float(np.abs(clean-pred['clean'][i]).max()),float(np.abs(old-pred['old'][i]).max()))
            m = detail.design(audit.geometry(i),**options)
            expected = np.array([all(counts[j]>=3 for j,v in zip(m.indices[m.indptr[r]:m.indptr[r+1]],
                m.data[m.indptr[r]:m.indptr[r+1]]) if v!=0) for r in range(m.shape[0])]).reshape(64,64)
            assert np.array_equal(expected,pred['support'][i])
        support_counts[name] = sum(int((pred['support'][i]&audit.masks(i)['full_visible']).sum()) for i in audit.TEST)
        assert support_counts[name]==report['arms'][name]['test_support']['complete_three_frame']
    assert maximum<1e-12
    selected,scores = audit.choose(predictions,sources,report['arms'])
    assert selected==report['selection']['selected'] and scores==report['selection']['scores']
    for color in ([0,0,0],[1,0,1]):
        altered = {i:np.full_like(s,color) if i in audit.TEST else s for i,s in sources.items()}
        assert audit.choose(predictions,altered,report['arms'])==(selected,scores)
    class SelectionOnly:
        def __getitem__(self,index):
            assert index in audit.SELECTION
            return sources[index]
    assert audit.choose(predictions,SelectionOnly(),report['arms'])==(selected,scores)
    # Poison actual paths, including heldout source files, in the real loader.
    manifest_path = shot.logs/'r22_cycles.json'
    manifest = json.loads(manifest_path.read_text())
    poisoned = 0
    for r in manifest['records']:
        for key in r['files']:
            if key!='source' or r['frame'] in audit.TEST:
                r['files'][key] = dict(path='FORBIDDEN_RGB_PATH',sha256='poison')
                poisoned += 1
    original = Path.read_text
    def guarded_read(p,*args,**kwargs):
        return json.dumps(manifest) if p==manifest_path else original(p,*args,**kwargs)
    with patch.object(Path,'read_text',guarded_read):
        loaded = audit.load(shot,audit.TRAIN+audit.SELECTION)
    assert all(np.array_equal(loaded[i],sources[i]) for i in loaded)
    assert audit.choose(predictions,loaded,report['arms'])==(selected,scores)
    clean,target = audit.load(shot,range(36),'clean'),audit.load(shot,range(36),'target')
    new = {i:target[i]-clean[i] for i in range(36)}
    for name,p in predictions.items():
        for region in audit.masks(0):
            for key,est,truth in (('sum_fit',p['clean']+p['old'],sources),('interior',p['clean'],clean),
                ('old_reflection',p['old'],{i:sources[i]-clean[i] for i in range(36)}),
                ('coupled_known_scene',{i:p['clean'][i]+new[i] for i in range(36)},target)):
                assert audit.metric(est,truth,region=region)==report['arms'][name]['test'][region][key]
    assert all(np.count_nonzero(s[17])==0 for s in (sources,clean,target))
    pupil_min = float(min(clean[i][audit.masks(i)['pupil']].min() for i in audit.TEST if audit.masks(i)['pupil'].any()))
    assert pupil_min>0
    witness_checks = []
    if selected:
        options = audit.ARMS[selected]
        matrices,rgb = audit.training_inputs(sources,options)
        m,b = sparse.vstack(matrices,format='csr'),np.concatenate(rgb)[:,1]
        upper = detail.upper_bounds(**options)
        radius = report['uncertainty']['radius']
        for index,(frame,y,x) in enumerate(audit.probes()):
            data = np.load(out/f'interval_{index:02d}.npz')
            w,q = data['values'],data['query']
            violation = max(float(np.max(np.abs(m@w.T-b[:,None])-radius)),
                float(-w.min()),float(np.max(w-upper)),0.)
            assert violation<=1e-6
            info = report['uncertainty']['records'][index]
            assert np.allclose(w@q,[info['lower'],info['upper']],atol=1e-9,rtol=0)
            witness_checks.append(violation)
    return dict(frame_arm_scalar_checks=144,scalar_max_error=maximum,kkt=kkt,
        support_counts=support_counts,training_substitutions=substitutions,training_access_guards=4,
        selection_substitutions=2,selection_access_guard=True,poisoned_loader_paths=poisoned,
        lp_witness_checks=len(witness_checks)*2,lp_max_violation=max(witness_checks,default=0),
        clean_pupil_min=pupil_min,closure_exact_zero=True,
        isolation_scope='Exact solver input invariance and guarded loader/selection; no repeated solver runs',
        complete_material_certification=False,production_repairs=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_shot_argument(ap)
    ap.add_argument('--transport-only',action='store_true')
    ap.add_argument('--skip-video',action='store_true')
    args = ap.parse_args()
    shot = from_args(args)
    if args.transport_only:
        result = renderer(shot,[0])
        write_json(shot.logs/'r22_transport_pretest.json',result)
    else:
        from verify_r15 import fallback
        result = dict(numerical_controls=numerical_controls(),fits=fits(shot),
            renderer=renderer(shot,[0,5,11,17,23,29,35]),transaction=transaction_checks(),source=fallback(shot))
        if not args.skip_video:
            result['video'] = video_checks(shot,'r22_eyes',108,24.,(762,960,3))
        result['verifier_sha256'] = sha(Path(__file__))
        write_json(shot.logs/'r22_verification.json',result)
    print('R22_VERIFICATION_COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':
    main()
