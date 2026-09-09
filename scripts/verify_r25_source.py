"""Verify source-only inverse geometry evidence, conditional routes and exact fallback."""
import argparse
import json
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
import cv2
import numpy as np
from audit_r24_source import ROOT, sha, write
from audit_r25_source import camera, freeze
from partial_eye_geometry import (split_arc, fit_family, curve, distances, project,
    ring_world, connected_candidate, pupil_distance, conservative_open, route_family)
from shot_paths import add_shot_argument,from_args


def controls():
    focal=np.array([2400.,3100.]); centre=np.array([943.,531.])
    params=np.array([919.,548.,1300.]); worst=0.
    for yd,pd in [(0,0),(30,0),(-30,0),(0,25),(0,-25),(30,-25),(-30,25)]:
        yaw,pitch=np.deg2rad([yd,pd])
        row=dict(parameters=params,yaw_deg=yd,pitch_deg=pd)
        eye=connected_candidate(row,focal,centre)
        theta=np.arange(360)*2*np.pi/360
        local=np.stack([eye.limbus_radius*np.cos(theta),eye.limbus_radius*np.sin(theta),
                        np.full(360,eye.sag)],-1)
        expected=project(local@eye.rot.T+eye.apex,focal,centre)
        got=curve(params,yaw,pitch,focal,centre)
        worst=max(worst,float(np.abs(expected-got).max()))
        # Scale all world dimensions about the camera: identical image, different anatomy/depth.
        assert np.max(np.abs(curve(params*[1,1,1.2],yaw,pitch,focal,centre,radius=5.85*1.2)-got))<1e-10
        # Horizontal desqueeze changes only x pixels and focal/centre, not physical geometry.
        squeeze=1.37
        stretched=curve(params*[squeeze,1,1],yaw,pitch,focal*[squeeze,1],centre*[squeeze,1])
        assert np.max(np.abs(stretched-got*[squeeze,1]))<1e-10
    assert worst<1e-10
    # Independent signed projection of the rightmost ring point with pitch zero.
    for yaw in (-.4,.4):
        point=ring_world(params,yaw,0,focal,centre)[0]
        c=np.r_[(params[:2]-centre)/focal*params[2],params[2]]
        expected=c+np.array([5.85*np.cos(yaw),0.,-5.85*np.sin(yaw)])
        assert np.max(np.abs(point-expected))<1e-10
    # Empty families never create vacuous all-candidate support.
    a,b=route_family([[4,4]],[],focal,centre); assert not a.any() and not b.any()
    # Real 2D closure and the complete footprint uncertainty margin must reject pixels.
    opening,safe=conservative_open((32,32),[[2,16],[29,16],[29,17],[2,17]],[])
    assert opening.any() and not safe.any()
    opening,safe=conservative_open((32,32),[[2,2],[29,2],[29,29],[2,29]],[[14,14],[18,14],[18,18],[14,18]])
    assert safe.any() and not safe[16,16] and not safe[3,3] and safe[8,8]
    return dict(signed_geometry_cases=7,connected_seam_projection_max_px=worst,
                scale_ambiguity_cases=7,pixel_aspect_cases=7,closure_and_footprint_controls=True,
                empty_family_abstains=True)


def evidence_checks(shot):
    labels=json.loads((ROOT/'scripts/data/r25_source_boundaries.json').read_text())
    report=json.loads((shot.logs/'r25_source.json').read_text())
    assert freeze(shot,labels)==report['evidence']
    assert all(sha(ROOT/'scripts'/name)==digest for name,digest in report['code_sha256'].items())
    manifest=json.loads((shot.relight/'neural_iris_observations/manifest.json').read_text())
    focal,centre=camera(shot,manifest)
    counts=dict(train_points=0,check_points=0,best_check_within_2px=0,
        compatible_ring_profiles=0,compatible_optical_profiles=0,conservative_observed_pixels=0,
        iris_candidate_pixels=0,pupil_candidate_pixels=0,unknown_observed_pixels=0,
        no_optical_family_eyes=0)
    best_dist=[]; front_dist=[]; compatible_checks=[]; dense_discrepancy=0.; observation_rows=[]
    for label,record in zip(labels['records'],report['records']):
        assert (label['frame'],label['side'])==(record['frame'],record['side'])
        with np.load(shot.relight/f"neural_iris_observations/{label['frame']:06d}_{label['side']}.npz") as d:
            rgb,origin=d['linear_rgb'],d['origin_xy']
        train,check,ti,ci=split_arc(label['limbus']); local=centre-origin
        assert record['train_indices']==ti.tolist() and record['check_indices']==ci.tolist()
        # All geometry profiles and held arc scores replay independently at 4x contour density.
        for r in record['family']:
            c=curve(r['parameters'],*np.deg2rad([r['yaw_deg'],r['pitch_deg']]),focal,local)
            for pts,key in [(train,'train_distances_px'),(check,'check_distances_px')]:
                d=distances(pts,c); assert np.max(np.abs(d-r[key]))<1e-10
                dense=curve(r['parameters'],*np.deg2rad([r['yaw_deg'],r['pitch_deg']]),focal,local,samples=1440)
                dense_discrepancy=max(dense_discrepancy,float(np.abs(distances(pts,dense)-d).max()))
            assert r['compatible']==bool(r['success'] and max(r['train_distances_px'])<=2)
            if r['compatible']: compatible_checks.extend(r['check_distances_px'])
        assert record['best_train_index']==min(range(len(record['family'])),key=lambda i:record['family'][i]['train_rms_px'])
        best=record['family'][record['best_train_index']]['check_distances_px']; best_dist.extend(best)
        front_dist.extend(next(r['check_distances_px'] for r in record['family'] if r['yaw_deg']==r['pitch_deg']==0))
        eyes=[]
        for profile in record['optical_profiles']:
            row=record['family'][profile['family_index']]
            eye=connected_candidate(row,focal,local,profile['cornea_radius_mm'],profile['pupil_radius_mm'])
            d,hits=pupil_distance(label['pupil'],eye,focal,local)
            assert np.max(np.abs(d-profile['pupil_distances_px']))<1e-10
            assert hits.tolist()==profile['pupil_cap_hits']
            assert profile['compatible']==bool(hits.all() and (d<=2).all())
            if profile['compatible']: eyes.append(eye)
        assert len(eyes)==record['compatible_optical_count']
        opening,safe=conservative_open(rgb.shape[:2],label['lid'],label['reflection'])
        y,x=np.nonzero(safe); xy=np.stack([x,y],-1)
        iris,pupil=route_family(xy,eyes,focal,local)
        path=shot.composite/f"r25_source/observations_{label['frame']:06d}_{label['side']}.npz"
        assert sha(path)==record['observation_archive_sha256']
        with np.load(path) as d:
            assert np.array_equal(d['linear_rgb'],rgb) and np.array_equal(d['source_fallback'],rgb)
            assert np.array_equal(d['origin_xy'],origin) and np.array_equal(d['candidate_xy'],xy)
            assert np.array_equal(d['iris_candidate'],iris) and np.array_equal(d['pupil_candidate'],pupil)
            expected=np.zeros(rgb.shape[:2],np.uint8); expected[y[iris],x[iris]]=1; expected[y[pupil],x[pupil]]=2
            assert np.array_equal(d['route'],expected) and not d['clean_material_certified']
        counts['train_points']+=len(train); counts['check_points']+=len(check)
        counts['best_check_within_2px']+=int((np.array(best)<=2).sum())
        counts['compatible_ring_profiles']+=record['compatible_geometry_count']
        counts['compatible_optical_profiles']+=len(eyes)
        counts['no_optical_family_eyes']+=int(not eyes)
        for k in ['conservative_observed_pixels','iris_candidate_pixels','pupil_candidate_pixels','unknown_observed_pixels']:
            counts[k]+=record[k]
        # Store observed radiance summaries including nonzero candidate pupil, never as clean texture.
        observation_rows.append(dict(frame=label['frame'],side=label['side'],
            pupil_observed_linear_rgb=rgb[y[pupil],x[pupil]].tolist(),
            optical_yaw_deg=sorted(set(r['yaw_deg'] for p in record['optical_profiles'] if p['compatible']
                                     for r in [record['family'][p['family_index']]]))))
    # Substitute the whole withheld middle arc: all train-only solutions must remain identical.
    original=np.array(labels['records'][0]['limbus'],float); poisoned=original.copy()
    _,_,ti,ci=split_arc(original); poisoned[ci]=[10000.,-10000.]
    a=fit_family(split_arc(original)[0],focal,centre,angles=(-20,0,20))
    b=fit_family(split_arc(poisoned)[0],focal,centre,angles=(-20,0,20))
    assert a==b
    assert dense_discrepancy<.002 and counts['iris_candidate_pixels']>0 and counts['pupil_candidate_pixels']>0
    return dict(counts=counts,best_train_check_median_px=float(np.median(best_dist)),
        best_train_check_max_px=float(np.max(best_dist)),frontal_control_check_median_px=float(np.median(front_dist)),
        frontal_control_check_max_px=float(np.max(front_dist)),
        compatible_family_check_within_2px=int((np.array(compatible_checks)<=2).sum()),
        compatible_family_check_samples=len(compatible_checks),dense_contour_max_difference_px=dense_discrepancy,
        withheld_arc_substitution_profiles=9,rgb_absent_from_geometry_and_route_interfaces=True,
        observation_rows=observation_rows,clean_material_certified=False,geometry_validated=False)


def prepare_detail(shot):
    out=shot.composite/'r25_detail/review'; out.mkdir(parents=True,exist_ok=True)
    for k,frame in enumerate((30,60,90,120)):
        board=cv2.imread(str(shot.composite/f'r25_source/review/{frame:06d}.png'))
        header=np.zeros((32,board.shape[1],3),np.uint8)
        cv2.putText(header,'HELD SOURCE FRAME - 1 second per observation; not real motion',(8,22),0,.5,(255,255,255),1)
        board=np.concatenate([header,board],0)
        for j in range(24): assert cv2.imwrite(str(out/f'{k*24+j:06d}.png'),board)
    print('R25_DETAIL_PREPARED')


def main():
    ap=argparse.ArgumentParser(description=__doc__); add_shot_argument(ap)
    ap.add_argument('--prepare-detail',action='store_true'); args=ap.parse_args(); shot=from_args(args)
    if args.prepare_detail: prepare_detail(shot); return
    from verify_r15 import fallback
    from verify_r20_eyes import video_checks
    report=dict(controls=controls(),source=evidence_checks(shot),exact_source=fallback(shot))
    report['videos']={name:video_checks(shot,name,count,fps,shape) for name,count,fps,shape in [
        ('r25_source',150,float(shot.frame_rate),(920,768,3)),('r25_detail',96,24.,(952,768,3))]}
    cap=cv2.VideoCapture(str(shot.final/'r25_source.mp4'))
    for frame in (30,60,90,120):
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame); ok,board=cap.read(); assert ok
        assert cv2.imwrite(str(shot.composite/f'r25_source/decoded_{frame:03d}.png'),board)
    cap.release()
    report['verifier_sha256']=sha(ROOT/'scripts/verify_r25_source.py')
    write(shot.logs/'r25_verification.json',report)
    print('R25_VERIFICATION_COMPLETE',json.dumps(report['source'],indent=2))


if __name__=='__main__': main()
