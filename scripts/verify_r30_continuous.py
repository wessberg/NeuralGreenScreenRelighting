"""Independent transport control, saved model replay, source guards and video checks."""
import argparse
import json
from dataclasses import replace
from unittest.mock import patch
import numpy as np
from threadpoolctl import threadpool_limits
import audit_r30_continuous as audit
from audit_r24_source import ROOT, sha, write
from eye_continuous_fit import compact_transport, corrected, light_vectors, solve, predict, ARMS
from eye_shape_fit import transport, fit, design
from connected_eye import ConnectedEye
from partial_eye_geometry import project
from verify_r29_fixture import independent, scalar
from coupled_eye_relight import apply_coupled
from shot_paths import add_shot_argument, from_args


def controls():
    eye=ConnectedEye(cornea_radius=8.,eccentricity=0.,lid_width=13.,lid_height=13.)
    xy=np.stack(np.meshgrid(np.arange(41.5,54,2),np.arange(41.5,54,2)),-1).reshape(-1,2)
    maximum=0.; signed=0.; scalar_error=0.
    for k in range(4):
        for sign in (-1,1):
            p=np.zeros(4); p[k]=sign*.4; e=corrected(eye,[3000,3000],p)
            old=transport(xy,e,[0,0],[3000,3000],[47.5,47.5],7)
            new=compact_transport(xy,e,[0,0],[3000,3000],[47.5,47.5],7)
            for key in old:
                maximum=max(maximum,float(np.max(np.abs(old[key].astype(float)-new[key].astype(float)))))
            if k<2:
                displacement=project(e.globe,[3000,3000],[47.5,47.5])-project(eye.globe,[3000,3000],[47.5,47.5])
                assert np.allclose(displacement,p[:2],atol=1e-12)
                for i,q in enumerate(xy):
                    d,f,_=scalar(q,e.globe)
                    scalar_error=max(scalar_error,float(np.max(np.abs(d-new['directions'][i,24]))),abs(f-new['f'][i,24]))
            else:
                assert np.isclose([e.yaw-eye.yaw,e.pitch-eye.pitch][k-2],sign*np.deg2rad(2))
            signed+=1
    assert maximum<1e-12 and scalar_error<1e-8
    axis=np.array([-.09,.11,-1]); axis/=np.linalg.norm(axis)
    eyes=[replace(eye,globe=(x,y,1313.)) for x,y in ((-.7,0),(0,.7),(.7,0),(0,-.7))]
    obs=[dict(xy=xy,eye=e,offset=np.zeros(2),centre=np.array([47.5,47.5])) for e in eyes]
    truth=[independent(xy,e.globe,7,axis) for e in eyes]
    # Exact same-node RGB check using known radiance; truth uses scalar optics.
    v=np.concatenate([np.zeros((1,3)),(512*axis)[None]])
    parity=0.
    for o,(rgb,layers) in zip(obs,truth):
        t=compact_transport(xy,o['eye'],[0,0],[3000,3000],[47.5,47.5],7)
        x,m=design(t,v,False); co=np.zeros((m+2,3))
        co[:289]=[.08,.12,.07]; co[289]=[.015,.018,.021]
        co[m:]=[[.06,.05,.04],[12,9,6]]
        parity=max(parity,float(np.max(np.abs(x@co-rgb))))
    assert parity<1e-8
    findings={}
    for arm in ('fixed','light','coupled','spatial_backing'):
        r=solve(obs[:3],[t[0] for t in truth[:3]],np.array([3000.,3000.]),arm)
        pp=predict(obs,np.array([3000.,3000.]),r['parameters'],r['seed'],r['coefficients'],arm=='spatial_backing')[-1]['layers']
        findings[arm]=dict(held_sum_max=float(np.max(np.abs(pp.sum(0)-truth[-1][0]))),
            held_layer_max=float(np.max(np.abs(np.stack([pp[0]+pp[1],pp[2]],1)-truth[-1][1]))),
            final_objective=r['history'][-1]['objective'],parameters=r['parameters'].tolist(),
            localized_lobe_rgb=r['coefficients'][-1].tolist(),
            seed_true_axis_angle_degrees=float(np.rad2deg(np.arccos(np.clip(r['seed']@axis,-1,1)))),
            nonlinear_status=[h.get('status') for h in r['history'][1:]])
        assert all(b['objective']<=a['objective']+1e-14 for a,b in zip(r['history'],r['history'][1:]))
    return dict(signed_cases=signed,compact_dense_max_error=maximum,
        independent_scalar_direction_fresnel_max=scalar_error,independent_scalar_rgb_max=parity,
        unknown_axis_fixture=findings,fixture_material_certified=False)


def replay(shot):
    report=json.loads((shot.logs/'r30_continuous.json').read_text())
    assert audit.freeze(shot)==report['evidence']
    for name,value in report['code_sha256'].items(): assert sha(ROOT/'scripts'/name)==value
    for path,value in report['artifact_hashes'].items(): assert sha(ROOT/path)==value
    focal=np.array(report['focal']); results={}
    for side in ('left','right'):
        obs=audit.prepare(shot,side,focal)
        with np.load(shot.composite/f'r30_continuous/predictions_{side}.npz') as archive: z=dict(archive)
        saved=audit.read_rgb
        def guard(s,which,frame):
            assert frame not in (105,115),'Seen evaluation RGB entered fitting'
            return saved(s,which,frame)
        count=0; full_solver_replay=0; fallback=0
        with patch.object(audit,'read_rgb',guard):
            values=audit.training_values(shot,side,obs,range(4))
            for record in report['results'][side]['fits']:
                name=record['name']; arm=record['arm']; train=record['train']; held=record['held']
                p,seed,co=[z[f'{name}_{field}'] for field in ('parameters','seed','coefficients')]
                pred=predict(obs,focal,p,seed,co,arm=='spatial_backing')
                x=np.concatenate([pred[j]['matrix'] for j in train]); y=np.concatenate([values[j] for j in train])
                repeated=fit(x,y,pred[0]['material_count'],light_vectors(seed,p[4:]))
                assert np.array_equal(repeated,co),(side,name,'coefficient replay')
                for j in (range(6) if held==-1 else [held]):
                    assert np.array_equal(pred[j]['layers'],z[f'{name}_layers_{j}'])
                    assert np.array_equal(audit.support(pred,[k for k in train if k!=j],j),z[f'{name}_support_{j}'])
                assert all(b['objective']<=a['objective']+1e-14 for a,b in zip(record['history'],record['history'][1:]))
                if arm=='coupled' and held==0:
                    # Full nonlinear replay on a whole-frame withheld fold for
                    # each eye, with held RGB unavailable to the caller as well.
                    class GuardedValues(dict):
                        def __getitem__(self,key):
                            assert key in train,'Held RGB accessed'
                            return super().__getitem__(key)
                    guarded=GuardedValues(values)
                    r=solve([obs[j] for j in train],[guarded[j] for j in train],focal,arm)
                    for field in ('parameters','seed','coefficients'): assert np.array_equal(r[field],z[f'{name}_{field}'])
                    assert r['history']==record['history']; full_solver_replay+=1
                count+=1
                print('R30_REPLAY',side,name,flush=True)
        truth=audit.training_values(shot,side,obs,range(6)); scores,masks=audit.score_arrays(obs,z,truth)
        assert json.loads(json.dumps(scores))==report['results'][side]['evaluation']
        for o in obs:
            src=saved(shot,side,o['frame'])
            output,info=apply_coupled(src,np.ones(src.shape[:2],bool),object(),object(),geometry_reliable=False)
            assert np.array_equal(output,src) and info['changed_channels']==0; fallback+=1
        results[side]=dict(coefficient_forward_support_replays=count,nonlinear_fold_replays=full_solver_replay,
            held_rgb_guard=True,seen_fit_reads=0,disabled_complete_transactions=fallback)
    return results


def main():
    ap=argparse.ArgumentParser(description=__doc__); add_shot_argument(ap)
    ap.add_argument('--fixture-only',action='store_true'); ap.add_argument('--skip-videos',action='store_true')
    ap.add_argument('--videos-only',action='store_true'); args=ap.parse_args(); shot=from_args(args)
    with threadpool_limits(limits=1):
        if args.fixture_only:
            result=controls(); write(shot.logs/'r30_fixture.json',result); print('R30_FIXTURE',json.dumps(result)); return
        from verify_r15 import fallback
        if args.videos_only:
            result=json.loads((shot.logs/'r30_verification.json').read_text())
            report=json.loads((shot.logs/'r30_continuous.json').read_text())
            assert audit.freeze(shot)==report['evidence']
            for name,value in report['code_sha256'].items(): assert sha(ROOT/'scripts'/name)==value
            for path,value in report['artifact_hashes'].items(): assert sha(ROOT/path)==value
        else: result=dict(fixture=controls(),replay=replay(shot),fallback=fallback(shot))
        if not args.skip_videos:
            from verify_r20_eyes import video_checks
            result['videos']={name:video_checks(shot,name,count,fps,shape) for name,count,fps,shape in
                [('r30_continuous',150,float(shot.frame_rate),(960,768,3)),('r30_layers',144,24.,(580,1200,3))]}
            import cv2
            cap=cv2.VideoCapture(str(shot.final/'r30_layers.mp4'))
            for f in (0,24,48,72,96,120):
                cap.set(cv2.CAP_PROP_POS_FRAMES,f); ok,im=cap.read(); assert ok
                assert cv2.imwrite(str(shot.composite/f'r30_layers/decoded_{f:03d}.png'),im)
            cap.release()
        result['verifier_sha256']=sha(ROOT/'scripts/verify_r30_continuous.py')
        write(shot.logs/'r30_verification.json',result)
    print('R30_VERIFIED',flush=True)


if __name__=='__main__': main()
