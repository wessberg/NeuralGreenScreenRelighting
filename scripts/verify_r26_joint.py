"""Replay joint evidence, challenge holdout isolation, and verify source retention."""
import argparse
from copy import deepcopy
import json
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2
import numpy as np
from unittest.mock import patch
from audit_r24_source import ROOT, sha, write
from audit_r26_joint import (evidence, get_eye, mapping, predict, score_frozen,
                             witness_spread, TARGET, DONORS)
from joint_eye_geometry import (candidate, SHARED, RADII, build_table, joint_profiles,
    pupil_residuals, iris_hit, inverse_iris, donor_support, sample_rgb, footprint)
from partial_eye_geometry import split_arc, fit_family, conservative_open, pupil_distance
from coupled_eye_relight import apply_coupled
from shot_paths import add_shot_argument, from_args


def controls():
    focal = np.array([2400., 3100.]); centre = np.array([943., 531.])
    worst = 0.; ray_count = 0; centred_error = 0.
    for yaw, pitch in [(0,0), (30,0), (-30,0), (0,25), (0,-25), (30,-25), (-30,25)]:
        ring = dict(parameters=[919., 548., 1300.], yaw_deg=yaw, pitch_deg=pitch)
        eye, offset = candidate(ring, [7.8, 3.6, .6, -.6], 1.8, focal, centre)
        y, x = np.mgrid[535:565, 905:935]; xy = np.stack([x.ravel(), y.ravel()], -1)
        uv, hit = iris_hit(xy, eye, offset, focal, centre)
        assert hit.sum() > 10
        mapped, ok, error = inverse_iris(uv[hit], eye, offset, focal, centre)
        assert ok.all()
        worst = max(worst, float(np.abs(mapped-xy[hit]).max())); ray_count += int(hit.sum())
        stretch = np.array([1.37, 1.])
        again, valid, _ = inverse_iris(uv[hit], eye, offset, focal*stretch, centre*stretch)
        assert valid.all() and np.max(np.abs(again-mapped*stretch)) < 1e-6
        points = xy[hit][::max(1, hit.sum()//8)]
        a, cap = pupil_residuals(points, eye, [[0., 0.]], [1.8], focal, centre)
        b, oldcap = pupil_distance(points, eye, focal, centre)
        # Different finite-difference order: derivative of norm vs norm derivative.
        centred_error = max(centred_error, float(np.abs(a[0,0]-b).max()))
        assert np.array_equal(cap, oldcap)
    assert worst < 1e-6 and centred_error < .002
    # Independent front-on shifted aperture: a point at the displaced centre is pupil.
    eye, offset = candidate(dict(parameters=[943.,531.,1300.], yaw_deg=0,pitch_deg=0),
                            [7.8,3.6,.6,-.6],1.8,focal,centre)
    uv = np.array([[.6,-.6], [.6+1.9,-.6], [.6-1.9,-.6]])
    xy, valid, _ = inverse_iris(uv, eye, offset, focal, centre)
    _, material = iris_hit(xy, eye, offset, focal, centre)
    assert material.tolist() == [False, True, True] and valid.tolist() == material.tolist()
    # Fail one corner while centre remains open; the whole interpolation must reject.
    small_centre = np.array([31.5,31.5]); small_focal = np.array([2666.,2666.])
    e, off = candidate(dict(parameters=[31.5,31.5,1300.],yaw_deg=0,pitch_deg=0),
                       [7.8,3.6,0,0],1.8,small_focal,small_centre)
    q = np.array([[39.2,31.2]])
    safe = np.ones((64,64),bool)
    assert donor_support(q,safe,e,off,small_focal,small_centre).all()
    safe[32,40] = False
    assert not donor_support(q,safe,e,off,small_focal,small_centre).any()
    assert not donor_support([[-1.2,30]],np.ones((64,64),bool),e,off,small_focal,small_centre).any()
    _, closed = conservative_open((64,64), [[3,30],[60,30],[60,31],[3,31]], [])
    assert not closed.any()
    assert joint_profiles(np.empty((0,8)),5) == []
    # Bilinear sampler checked against an independently evaluated affine image.
    yy, xx = np.mgrid[:64,:64]
    rgb = np.stack([xx*.01+yy*.02, xx*.03-yy*.004, yy*.012], -1)
    p = np.array([[17.3,22.7],[42.9,18.2]])
    expected = np.stack([p[:,0]*.01+p[:,1]*.02,p[:,0]*.03-p[:,1]*.004,p[:,1]*.012],-1)
    assert np.max(np.abs(sample_rgb(rgb,p)-expected)) < 1e-12
    return dict(signed_camera_cases=7,inverse_rays=ray_count,inverse_max_px=worst,
        asymmetric_aspect_cases=7,centred_residual_difference_px=centred_error,
        decentration_sign=True,complete_footprint_rejection=True,closure=True,
        empty_joint_family_abstains=True,independent_bilinear_check=True)


def replay(shot):
    report = json.loads((shot.logs/'r26_joint.json').read_text())
    assert evidence(shot) == report['evidence']
    assert all(sha(ROOT/'scripts'/n) == h for n,h in report['code_sha256'].items())
    focal = np.array(report['focal']); out = shot.composite/'r26_joint'; results = {}
    for side in ('left','right'):
        summary = report['results'][side]
        path = out/f'geometry_{side}.json'; assert sha(path) == summary['geometry_sha256']
        geometry = json.loads(path.read_text()); obs = geometry['observations']
        path = out/f'family_{side}.npz'; assert sha(path) == summary['family_sha256']
        with np.load(path) as d: table = d['table']
        rebuilt = build_table(obs,focal)
        assert np.array_equal(rebuilt,table)
        profiles = joint_profiles(table,len(obs)); assert profiles == geometry['profiles']
        # An untouched target's held pupil labels must not change conditioning.
        altered = deepcopy(obs); altered[-1]['pupil'][2:] = [[10000.,-10000.],[-5000.,9000.]]
        poisoned = build_table(altered,focal)
        assert np.array_equal(poisoned[:,:6],table[:,:6])
        assert joint_profiles(poisoned,len(obs)) == profiles
        # Whole held limbus substitution preserves train-only target ring fits.
        arc = np.array(obs[-1]['label']['limbus'],float)
        _,_,ti,ci = split_arc(arc); other = arc.copy(); other[ci] = [10000.,-10000.]
        a = fit_family(split_arc(arc)[0],focal,obs[-1]['centre'],angles=(-20,0,20))
        b = fit_family(split_arc(other)[0],focal,obs[-1]['centre'],angles=(-20,0,20))
        assert a == b
        # Full table check columns cannot affect joint selection or compatible families.
        damaged = table.copy(); damaged[:,6:] = 99999
        assert joint_profiles(damaged,len(obs)) == profiles
        local = dict(replayed_table_rows=len(table),held_pupil_table_isolation=True,
            held_limbus_train_profiles=9,check_column_isolation=True)
        if profiles:
            maps = mapping(obs,table,profiles[0],focal)
            pred,control,paired = predict(shot,side,maps)
            path = out/f'prediction_{side}.npz'
            with np.load(path) as d:
                for k,v in maps.items(): assert np.array_equal(v,d[k])
                for k,v in [('predictions',pred),('control_predictions',control),('paired',paired)]:
                    assert np.array_equal(v,d[k])
                witnesses = witness_spread(obs,table,profiles,focal,maps['xy'])
                for k,v in witnesses.items(): assert np.array_equal(v,d[k])
            assert score_frozen(shot,side,path) == summary['rgb_check']
            # Intercept every target archive read with a replacement array and assert
            # it is never requested by mapping/prediction. No authoritative file edits.
            original_load = np.load; calls = []
            for value in (0.,1000.):
                def guarded(path,*args,**kwargs):
                    if str(path).endswith(f'000105_{side}.npz'):
                        calls.append(str(path))
                        return {'linear_rgb':np.full((*obs[-1]['shape'],3),value)}
                    return original_load(path,*args,**kwargs)
                with patch('numpy.load',guarded):
                    m = mapping(obs,table,profiles[0],focal)
                    p,c,mask = predict(shot,side,m)
                assert np.array_equal(pred,p) and np.array_equal(control,c) and np.array_equal(paired,mask)
            assert calls == []
            # Geometry witnesses separately from the two added torsion perturbations.
            idx = [1]+list(range(3,len(profiles)+2))
            wm,wv = witnesses['witness_maps'][idx],witnesses['witness_valid'][idx]
            lo=np.min(np.where(wv[...,None],wm,np.inf),axis=0)
            hi=np.max(np.where(wv[...,None],wm,-np.inf),axis=0)
            diam=np.where(wv.sum(0)>=2,np.max(hi-lo,axis=-1),0)
            mask=paired&(wv.sum(0)>=2)
            local.update(target_rgb_substitutions=2,target_rgb_reads_in_predictor=0,
                geometry_only_witness_median_px=float(np.median(diam[mask])),
                geometry_only_witness_max_px=float(diam[mask].max()),
                geometry_only_pairs_over_2px=int((mask&(diam>2)).sum()),
                geometry_only_comparable_pairs=int(mask.sum()))
        # Exploratory sensitivity, explicitly after the frozen primary evaluation:
        # systematically shift training pupil marks, then offset the pupil-size grid.
        sensitivity = {}
        for name,delta in [('x_minus',[-1,0]),('x_plus',[1,0]),('y_minus',[0,-1]),('y_plus',[0,1])]:
            shifted=deepcopy(obs)
            for o in shifted:
                n=2 if o['target'] else len(o['pupil'])
                o['pupil'][:n]=(np.array(o['pupil'][:n])+delta).tolist()
            t=build_table(shifted,focal); pp=joint_profiles(t,len(obs))
            sensitivity[name]=dict(shared_profiles=len(pp),
                best_shared=SHARED[pp[0]['shared_index']].tolist() if pp else None)
        radii=RADII[:-1]+.15
        t=build_table(obs,focal,radii=radii); pp=joint_profiles(t,len(obs))
        sensitivity['midpoint_pupil_radius_grid']=dict(radii_mm=radii.tolist(),shared_profiles=len(pp),
            best_shared=SHARED[pp[0]['shared_index']].tolist() if pp else None)
        local['exploratory_sensitivity']=sensitivity
        # Exercise the complete transaction at all ten observed keyed eyes.
        transactions=0
        for o in obs:
            with np.load(shot.relight/f'neural_iris_observations/{o["frame"]:06d}_{side}.npz') as d:
                rgb=d['linear_rgb']
            opening,_=conservative_open(o['shape'],o['label']['lid'],o['label']['reflection'])
            output,info=apply_coupled(rgb,opening,geometry_reliable=False)
            assert np.array_equal(output,rgb) and info['changed_channels']==0
            transactions+=1
        local['disabled_complete_transactions']=transactions
        results[side]=local
        print('R26_REPLAY',side,json.dumps(local),flush=True)
    return results


def main():
    ap=argparse.ArgumentParser(description=__doc__); add_shot_argument(ap)
    ap.add_argument('--controls-only',action='store_true'); ap.add_argument('--skip-videos',action='store_true')
    args=ap.parse_args(); shot=from_args(args)
    result=dict(controls=controls())
    if args.controls_only: print(json.dumps(result,indent=2)); return
    result['source']=replay(shot)
    from verify_r15 import fallback
    result['exact_source']=fallback(shot)
    if not args.skip_videos:
        from verify_r20_eyes import video_checks
        result['videos']={name:video_checks(shot,name,n,fps,shape) for name,n,fps,shape in
            [('r26_joint',150,float(shot.frame_rate),(960,768,3)),('r26_detail',120,24.,(992,768,3)),
             ('r26_transport',120,24.,(722,768,3))]}
        cap=cv2.VideoCapture(str(shot.final/'r26_joint.mp4'))
        for frame in (30,60,90,105,120):
            cap.set(cv2.CAP_PROP_POS_FRAMES,frame); ok,im=cap.read(); assert ok
            assert cv2.imwrite(str(shot.composite/f'r26_joint/decoded_{frame:03d}.png'),im)
        cap.release()
    result['verifier_sha256']=sha(ROOT/'scripts/verify_r26_joint.py')
    write(shot.logs/'r26_verification.json',result)
    print('R26_VERIFICATION_COMPLETE',flush=True)


if __name__=='__main__': main()
