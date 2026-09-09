"""Independent saved-fit, residual, access-boundary, geometry and fallback checks."""
import argparse
import ast
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
import audit_r23_eyes as audit
import audit_r23_geometry as real
from audit_r20_eyes import sha, write_json
from coupled_eye_relight import apply_coupled
from eye_residual import preserve_residual
import eye_detail_fit as detail
from verify_r20_eyes import transaction_checks, video_checks
from verify_r22_eyes import numerical_controls
from shot_paths import add_shot_argument, from_args


def residual_controls():
    # High-frequency observed texture outside a coarse fitted-clean representation.
    rng=np.random.default_rng(2301)
    clean=rng.uniform(.01,.2,(8,9,3)); reflection=rng.uniform(.0,.04,clean.shape)
    source=clean+reflection
    retained,residual=preserve_residual(source,np.full_like(clean,.1),reflection)
    assert np.allclose(retained,clean,atol=1e-16,rtol=0)
    # The pupil's nonzero radiance can be misassigned even with a perfect source sum.
    wrong=reflection+clean*.5
    damaged,_=preserve_residual(source,clean*.5,wrong)
    assert np.allclose(damaged,clean*.5,atol=1e-16,rtol=0)
    noise=rng.normal(0,.002,clean.shape)
    noisy,_=preserve_residual(source+noise,clean,reflection)
    assert np.allclose(noisy-clean,noise,atol=1e-16,rtol=0)
    negative,_=preserve_residual(source,clean,np.ones_like(source))
    assert (negative<0).all()
    return dict(oracle_reflection_preserves_observed_detail=True,
        perfect_sum_can_remove_half_pupil_radiance=True,source_noise_retained=True,
        negatives_not_clamped=True)


def fit_checks(shot):
    out=shot.composite/'r23_eyes'; report=json.loads((shot.logs/'r23_eyes.json').read_text())
    pre=json.loads((shot.logs/'r23_preflight.json').read_text())
    assert report['code_sha256']==pre['code_sha256']
    assert all(sha(audit.ROOT/'scripts'/k)==v for k,v in report['code_sha256'].items())
    assert report['protocol_sha256']==sha(audit.ROOT/'docs/r23-eye-protocol.md')
    assert report['amendment_sha256']==sha(audit.ROOT/'docs/r23-transport-correction.md')
    assert not set(audit.TRAIN)&set(audit.SELECT+audit.TEST)
    sources,truth,target=[audit.load(shot,audit.FRAMES,k) for k in ('source','clean','target')]
    new={i:target[i]-truth[i] for i in audit.FRAMES}
    independent_max=0.; residual_identity=0.; scalar_checks=0; support_counts={}; gradients={}; coverage={}
    for arm in audit.ARMS:
        p=dict(np.load(out/f'{arm}.npz')); values=p['values']; upper=detail.upper_bounds(**audit.OPTIONS)
        assert values.min()>=0 and np.max(values-upper[:,None])<=0
        matrices,rgb=audit.inputs(sources,arm)
        m,b=sparse.vstack(matrices,format='csr'),np.concatenate(rgb)
        counts=np.zeros(len(values),int)
        for x in matrices: counts[np.unique(x.indices[x.data!=0])]+=1
        assert np.array_equal(counts,p['counts'])
        norm=np.sqrt(np.asarray(m.power(2).sum(0)).ravel()); active=norm>0
        g=(m[:,active].T@(m@values-b))/norm[active,None]
        z=values[active]*norm[active,None]
        pg=np.max(np.abs(z-np.clip(z-g,0,(upper[active]*norm[active])[:,None])),axis=0)
        assert np.allclose(pg,[s['projected_gradient'] for s in report['arms'][arm]['solvers']],atol=1e-12,rtol=0)
        gradients[arm]=pg.tolist()
        class TrainingOnly:
            def __getitem__(self,i):
                assert i in audit.TRAIN
                return sources[i]
        _,guarded=audit.inputs(TrainingOnly(),arm)
        assert np.array_equal(np.concatenate(guarded),b)
        for color in (0.,.71):
            altered={i:np.full_like(s,color) if i not in audit.TRAIN else s for i,s in sources.items()}
            _,sub=audit.inputs(altered,arm)
            assert np.array_equal(np.concatenate(sub),b)
        retained={}
        for i in audit.FRAMES:
            c,r=audit.direct(values,i,arm)
            independent_max=max(independent_max,float(np.abs(c-p['clean'][i]).max()),
                                 float(np.abs(r-p['old'][i]).max())); scalar_checks+=1
            expected_support=np.ones(4096,bool); x=audit.matrix(i,arm)
            for k in range(x.shape[0]):
                cols=x.indices[x.indptr[k]:x.indptr[k+1]]
                expected_support[k]=bool(np.all(counts[cols]>=3))
            assert np.array_equal(expected_support.reshape(64,64),p['support'][i])
            keep,residual=preserve_residual(sources[i],c,r); retained[i]=keep
            residual_identity=max(residual_identity,float(np.abs(keep-(c+residual)).max()),
                                  float(np.abs((keep-truth[i])-((sources[i]-truth[i])-r)).max()))
            result,info=apply_coupled(sources[i],audit.masks(i)['full_visible'],keep,new[i],
                support=p['support'][i],geometry_reliable=False)
            assert np.array_equal(result,sources[i]) and info['changed_channels']==0
        support_counts[arm]=int(sum((p['support'][i]&audit.masks(i)['full_visible']).sum() for i in audit.TEST))
        # The frozen saved flag is coefficient-count support only: an empty row
        # satisfies it vacuously. Explicitly reject rows with no modeled subray.
        modeled={i:(audit.matrix(i,arm).getnnz(axis=1)>0).reshape(64,64) for i in audit.TEST}
        coverage[arm]=dict(
            visible=sum(int(audit.masks(i)['full_visible'].sum()) for i in audit.TEST),
            raw_coefficient_count_support=support_counts[arm],
            visible_without_modeled_subray=sum(int((audit.masks(i)['full_visible']&~modeled[i]).sum()) for i in audit.TEST),
            nonempty_three_frame_support=sum(int((audit.masks(i)['full_visible']&modeled[i]&p['support'][i]).sum()) for i in audit.TEST),
            continuous_footprint_certified=False)
        for region in audit.masks(0):
            saved=report['arms'][arm]['test'][region]
            for key,est,ref in [('fitted_clean',p['clean'],truth),('residual_clean',retained,truth),
                ('fitted_coupled',{i:p['clean'][i]+new[i] for i in audit.FRAMES},target),
                ('residual_coupled',{i:retained[i]+new[i] for i in audit.FRAMES},target)]:
                computed=audit.metric(est,ref,region=region)
                assert all(np.isclose(computed[k],v,atol=1e-10,rtol=0) for k,v in saved[key].items())
    assert independent_max<1e-12 and residual_identity<1e-12
    # Poison the actual manifest's heldout/reference paths, then run the real loader.
    path=shot.logs/'r23_cycles.json'; manifest=json.loads(path.read_text()); poisoned=0
    for r in manifest['records']:
        for k in r['files']:
            if k!='source' or r['frame'] in audit.TEST:
                r['files'][k]=dict(path='FORBIDDEN_REFERENCE',sha256='poison'); poisoned+=1
    original=Path.read_text
    def guarded_read(p,*args,**kwargs):
        return json.dumps(manifest) if p==path else original(p,*args,**kwargs)
    with patch.object(Path,'read_text',guarded_read):
        loaded=audit.load(shot,audit.TRAIN+audit.SELECT)
    assert all(np.array_equal(loaded[i],sources[i]) for i in loaded)
    assert all(np.count_nonzero(d[17])==0 for d in (sources,truth,target))
    matrices,rgb=audit.inputs(sources,'nominal'); m=sparse.vstack(matrices,format='csr'); b=np.concatenate(rgb)[:,1]
    witness=np.load(out/'minimax_witness.npz'); w=witness['values']; radius=float(witness['radius'])
    violation=max(float(np.max(np.abs(m@w-b)-radius)),float(-w.min()),
                  float(np.max(w-detail.upper_bounds(**audit.OPTIONS))),0.)
    assert violation<1e-6
    # HiGHS logged an imprecise interior solution before reporting Optimal.
    # Independently form a feasible dual lower bound, not just a success flag.
    constraints=sparse.vstack([m,-m],format='csr')
    augmented=sparse.hstack([constraints,-np.ones((2*len(b),1))],format='csr')
    rhs=np.r_[b,-b]; upper=detail.upper_bounds(**audit.OPTIONS)
    objective=np.r_[np.zeros(len(upper)),1.]
    lp=linprog(objective,A_ub=augmented,b_ub=rhs,
        bounds=list(zip(np.zeros(len(upper)),upper))+[(0,None)],method='highs-ipm',
        options=dict(run_crossover='off',ipm_optimality_tolerance=1e-10))
    assert lp.success
    dual=np.minimum(lp.ineqlin.marginals,0.)
    dual*=min(1.,(1-1e-12)/max(1e-30,-dual.sum()))
    reduced=objective-augmented.T@dual
    assert reduced[-1]>=0 and np.max(dual)<=0
    lower=float(rhs@dual+np.minimum(reduced[:-1],0)@upper)
    assert 0<=radius-lower<1e-6 and lower>.002
    np.savez_compressed(out/'minimax_dual.npz',inequality_multipliers=dual,lower_bound=lower)
    return dict(scalar_frame_arm_checks=scalar_checks,independent_sampling_max=independent_max,
        residual_identity_max=residual_identity,projected_gradients=gradients,support_counts=support_counts,
        coverage=coverage,
        training_access_guards=4,heldout_input_substitutions=8,poisoned_loader_paths=poisoned,
        isolation_scope='Input invariance and actual loader checks; not repeated end-to-end fits',
        minimax_primal_witness_violation=violation,closed_frame_exact_zero=True,
        minimax_dual_lower_bound=lower,minimax_primal_dual_gap=radius-lower,
        fixed_radius_infeasible_by_dual_bound=True,
        disabled_coupled_transactions=96,production_repairs=0)


def geometry_checks(shot):
    report=json.loads((shot.logs/'r23_geometry.json').read_text())
    assert report['code_sha256']==sha(audit.ROOT/'scripts/audit_r23_geometry.py')
    root=shot.relight/'neural_iris_observations'
    assert report['manifest_sha256']==sha(root/'manifest.json')
    checks=0; maxerror=0.; invalid=0
    for row in report['records']:
        path=root/f"{row['frame']:06d}_{row['side']}.npz"; assert sha(path)==row['archive_sha256']
        d=dict(np.load(path)); origin=d['origin_xy']; focal=np.array(report['focal_pixels']); centre=np.array(report['centre_pixels'])
        iris,pupil=d['limbus_boundary_xy'][::12],d['pupil_boundary_xy'][::12]
        for p in row['profiles']:
            params=np.array(p['parameters'])
            e=real.residual(params,p['radius_mm'],origin,focal,centre,iris[1::2],pupil[1::2])
            maxerror=max(maxerror,abs(float(np.sqrt(np.mean(e*e)))-p['check_radial_rms_px']))
            t=real.rays(np.concatenate([iris,pupil]),params,p['radius_mm'],origin,focal,centre)
            assert bool(t['hit'].all())==p['complete_conic_support']
            invalid+=int(not p['complete_conic_support']); checks+=1
        assert not row['geometry_validated'] and not row['production_replacement']
    assert maxerror<1e-12
    return dict(profile_replays=checks,radial_score_max_difference=maxerror,
                rejected_incomplete_cap_profiles=invalid,validated_real_geometry=0,
                scope='Saved diagnostic replay; no independent physical truth')


def rendered_transport(shot):
    path=audit.ROOT/'scripts/render_r23_eyes.py'
    node=next(n for n in ast.parse(path.read_text(encoding='utf-8-sig')).body
              if isinstance(n,ast.FunctionDef) and n.name=='textures')
    scope={'np':np}; exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    values=np.concatenate([x.reshape(-1,3) for x in scope['textures']()[:3]])
    records=[]
    for i in (0,5,11,17,23):
        c,r=audit.direct(values,i,n=32,texture=True)
        clean,source=audit.load(shot,[i],'clean'),audit.load(shot,[i])
        scores={region:dict(clean=audit.metric({i:c},clean,[i],region),
                           source=audit.metric({i:c+r},source,[i],region)) for region in audit.masks(i)}
        records.append(dict(frame=i,scores=scores))
    return dict(cases=records,quadrature=32,
                scope='Post-fit independent texture integration; finite quadrature and Monte Carlo error retained')


def main():
    ap=argparse.ArgumentParser(description=__doc__); add_shot_argument(ap)
    ap.add_argument('--skip-video',action='store_true'); args=ap.parse_args(); shot=from_args(args)
    from verify_r15 import fallback
    report=dict(residual_controls=residual_controls(),numerical_controls=numerical_controls(),
        fits=fit_checks(shot),real_geometry=geometry_checks(shot),transaction_controls=transaction_checks(),
        rendered_transport=rendered_transport(shot),exact_source=fallback(shot),verifier_sha256=sha(Path(__file__)))
    previous=json.loads((shot.logs/'r20_eye_verification.json').read_text())['videos']['real']['sha256']
    report['real_fallback_review_sha256']=sha(shot.final/'r20_source_fallback.mp4')
    assert report['real_fallback_review_sha256']==previous
    if not args.skip_video:
        report['videos']=dict(synthetic=video_checks(shot,'r23_eyes',72,24.,(762,960,3)),
            real_geometry=video_checks(shot,'r23_geometry',48,24.,(696,640,3)))
    write_json(shot.logs/'r23_verification.json',report)
    print('R23_VERIFICATION_COMPLETE',json.dumps(report),flush=True)


if __name__=='__main__': main()
