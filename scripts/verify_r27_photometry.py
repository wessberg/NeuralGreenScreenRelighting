"""Independent numerical controls, frozen replay, source isolation and reviews."""
import argparse
import copy
import json
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from unittest.mock import patch
import cv2
import numpy as np
import audit_r27_photometry as audit
from audit_r24_source import ROOT, sha, write
from connected_eye import ConnectedEye, trace_connected
from corneal_transport import interface
from coupled_eye_relight import apply_coupled
from eye_appearance_fit import (BOX, MATERIAL_COLUMNS, DIRECTIONS, design, material_basis,
                                light_basis, fit, layers, observed_support)
from joint_eye_geometry import build_table
from partial_eye_geometry import fit_family, split_arc
from shot_paths import add_shot_argument, from_args


def controls():
    # Independent analytic mirror vectors for signed and mixed directions.
    d=np.array([[0,0,1],[.3,0,np.sqrt(.91)],[-.3,0,np.sqrt(.91)],
                [0,.4,np.sqrt(.84)],[0,-.4,np.sqrt(.84)],[.2,-.3,np.sqrt(.87)]])
    n=np.tile([0.,0.,-1.],(len(d),1)); _,r,f=interface(d,n,1.376)
    expected=d.copy(); expected[:,2]*=-1
    reflection_error=float(np.max(np.abs(r-expected))); assert reflection_error<1e-14
    assert abs(f[0]-((1.376-1)/(1.376+1))**2)<1e-14
    # Grid interpolation against an independent affine function.
    uv=np.array([[-3.71,2.62],[4.1,-2.38],[.12,.93]])
    basis,valid=material_basis(uv); yy,xx=np.mgrid[-8:9,-8:9]
    expected=3+uv[:,0]*.1+uv[:,1]*.2
    err=float(np.max(np.abs(basis@(3+xx*.1+yy*.2).ravel()-expected)))
    assert valid.all() and err<1e-14
    assert not material_basis(np.array([[8.1,0],[-8.1,0]]))[1].any()
    # Independent scalar 9-node integration, including both optical layers.
    eye=ConnectedEye(lid_height=13.,lid_width=13.)
    xy=np.array([[40.,47.],[54.,44.],[47.,56.]])
    matrix,valid=design(xy,eye,np.zeros(2),(3000.,2700.),(47.5,47.5))
    coefficients=np.zeros((342,3)); coefficients[:289]=np.stack(
        [(3+xx*.1+yy*.2).ravel()]*3,1)
    coefficients[289:]=np.arange(53)[:,None]*.002+.02
    expected=[]
    for point in xy:
        value=np.zeros(3)
        for dy in (-1/3,0.,1/3):
            for dx in (-1/3,0.,1/3):
                ray=trace_connected(point+[dx,dy],eye,focal=(3000.,2700.),centre=(47.5,47.5))
                u,v=ray['iris_xy']; a=3+.1*u+.2*v
                b=float(coefficients[289,0])
                for k,concentration in enumerate((8,32)):
                    for j,direction in enumerate(DIRECTIONS):
                        b+=coefficients[290+k*26+j,0]*np.exp(concentration*(np.dot(ray['reflected'],direction)-1))
                value+=(1-ray['fresnel'])*a+ray['fresnel']*b
        expected.append(value/9)
    box_error=float(np.max(np.abs(matrix@coefficients-expected)))
    assert valid.all() and box_error<1e-12
    # Overdetermined independent positive fixture: tests the solver, not material identifiability.
    rng=np.random.default_rng(27); x=np.zeros((120,342)); cols=[1,17,61,289,301]
    x[:,cols]=rng.uniform(.1,1,(120,len(cols)))
    co=np.zeros((342,3)); co[cols]=rng.uniform(.01,.1,(len(cols),3))
    y=x@co; got=fit(x,y,1e-4)
    reconstruction=float(np.max(np.abs(x@got-y)))
    assert (got>=0).all() and reconstruction<.001
    # Support requires two distinct frames, not repeated samples in one frame.
    train=np.zeros((2,342)); train[:,0]=1; query=train[:1]
    assert not observed_support([train],query).any()
    assert observed_support([train,train],query).all()
    closed,ok=design(xy,ConnectedEye(lid_height=0),np.zeros(2),(3000.,2700.),(47.5,47.5))
    assert not ok.any()
    return dict(signed_reflection_cases=len(d),reflection_max_error=reflection_error,
        affine_grid_max_error=err,independent_box_max_error=box_error,
        positive_fixture_prediction_max_error=reconstruction,distinct_frame_support=True,closure_rejected=True)


def replay(shot):
    out=shot.composite/'r27_photometry'; report=json.loads((shot.logs/'r27_photometry.json').read_text())
    assert audit.evidence(shot)==report['evidence']
    results={}; focal=np.array(report['focal'])
    for side in ('left','right'):
        for name,h in report['results'][side]['artifact_hashes'].items(): assert sha(ROOT/name)==h
        geometry=json.loads((out/f'geometry_{side}.json').read_text())
        obs,profiles=geometry['observations'],geometry['profiles']
        with np.load(out/f'geometry_{side}.npz') as d: table=d['table']
        matrices,xy,reflected,masks=audit.make_designs(obs,table,profiles,focal)
        with np.load(out/f'designs_{side}.npz') as d:
            for oi in range(6):
                assert np.array_equal(d[f'design_{oi}'],matrices[oi])
                assert np.array_equal(d[f'xy_{oi}'],xy[oi])
        expected=json.loads((out/f'selection_{side}.json').read_text())
        saved_read=audit.read_rgb; forbidden_reads=[]
        def guarded(s,which,frame):
            if frame==115:
                forbidden_reads.append(frame)
                raise AssertionError('Final target RGB requested by fitting')
            return saved_read(s,which,frame)
        # Replace every final archive RGB access at the NPZ boundary as well.
        saved_load=np.load; replacements=[]
        class ReplaceArchive:
            def __init__(self,obj,value): self.obj,self.value=obj,value
            def __enter__(self): return self
            def __exit__(self,*a): self.obj.close()
            def __getitem__(self,key):
                if key=='linear_rgb':
                    replacements.append(key)
                    return np.full_like(self.obj[key],self.value)
                return self.obj[key]
        for value in (0.,1000.):
            def substituted(path,*args,**kwargs):
                obj=saved_load(path,*args,**kwargs)
                return ReplaceArchive(obj,value) if f'000115_{side}.npz' in str(path) else obj
            with patch.object(audit,'read_rgb',guarded),patch.object(np,'load',substituted):
                selection,pred=audit.fit_predict(shot,side,matrices,xy)
            assert selection==expected
            with saved_load(out/f'predictions_{side}.npz') as data:
                for k,v in pred.items(): assert np.array_equal(v,data[k])
        assert not forbidden_reads and not replacements
        # Held pupil rows and limbus arcs cannot affect geometry conditioning.
        target=copy.deepcopy(obs[-1]); altered=copy.deepcopy(target)
        altered['pupil'][2:]=[[10000.,-10000.],[-10000.,10000.]]
        t1=build_table([target],focal); t2=build_table([altered],focal)
        assert np.array_equal(t1[:,:6],t2[:,:6])
        original=np.array(target['label']['limbus'],float); poisoned=original.copy()
        _,_,_,held=split_arc(original); poisoned[held]=[10000.,-10000.]
        assert fit_family(split_arc(original)[0],focal,target['centre'],angles=(-20,0,20))==fit_family(
            split_arc(poisoned)[0],focal,target['centre'],angles=(-20,0,20))
        assert audit.score(shot,side,out)==report['results'][side]['final']
        transactions=0
        for o in obs:
            source=audit.read_rgb(shot,side,o['frame']); mask=np.ones(source.shape[:2],bool)
            output,info=apply_coupled(source,mask,object(),object(),geometry_reliable=False)
            assert np.array_equal(source,output) and info['changed_channels']==0
            transactions+=1
        # Exploratory conditioning diagnostics AFTER primary scoring, no refit or new gate.
        conditioning=[]
        with np.load(out/f'predictions_{side}.npz') as data: coefficients=data['coefficients']; predictions=data['predictions']
        for g in range(len(profiles)):
            train=np.concatenate([matrices[j][g] for j in range(4)])
            light=train[:,289:]; maximum=light.max(0)
            test=matrices[5][g][:,289:]
            ratio=test/np.maximum(maximum[None],1e-30)
            norms=np.linalg.norm(train,axis=0); active=norms>1e-12
            singular=np.linalg.svd(train[:,active]/norms[active],compute_uv=False)
            conditioning.append(dict(geometry_index=g,active_columns=int(active.sum()),
                singular_max=float(singular[0]),singular_min=float(singular[-1]),
                numerical_rank_1e_8=int((singular>singular[0]*1e-8).sum()),
                final_over_train_light_basis_max=float(ratio.max()),
                final_pixels_any_light_basis_over_training_max=int(np.any(ratio>1+1e-10,axis=1).sum()),
                max_fitted_environment_coefficient=float(coefficients[g*4:g*4+4,289:].max()),
                max_final_total_linear=float(predictions[g*4:g*4+4].sum(1).max())))
        results[side]=dict(replayed_models=len(expected['records']),geometry_rows=len(table),
            final_rgb_substitution_replays=2,final_rgb_reads_during_fit=0,
            held_pupil_rows_unchanged=len(t1),held_arc_profiles_unchanged=9,
            disabled_complete_transactions=transactions,conditioning_exploratory=conditioning)
    return results


def main():
    ap=argparse.ArgumentParser(description=__doc__); add_shot_argument(ap)
    ap.add_argument('--skip-videos',action='store_true'); args=ap.parse_args(); shot=from_args(args)
    from verify_r15 import fallback
    result=dict(controls=controls(),source=replay(shot),exact_source=fallback(shot))
    if not args.skip_videos:
        from verify_r20_eyes import video_checks
        result['videos']={name:video_checks(shot,name,count,fps,shape) for name,count,fps,shape in
            [('r27_photometry',150,float(shot.frame_rate),(960,768,3)),('r27_layers',96,24.,(600,960,3))]}
        for name,frames in [('r27_photometry',(30,60,90,105,115,120)),('r27_layers',(0,24,48,72))]:
            cap=cv2.VideoCapture(str(shot.final/f'{name}.mp4'))
            for frame in frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES,frame); ok,im=cap.read(); assert ok
                assert cv2.imwrite(str(shot.composite/f'{name}/decoded_{frame:03d}.png'),im)
            cap.release()
    result['verifier_sha256']=sha(ROOT/'scripts/verify_r27_photometry.py')
    write(shot.logs/'r27_verification.json',result)
    print('R27_VERIFIED',json.dumps(result['controls']),flush=True)


if __name__=='__main__': main()
