"""Connected-eye geometry, provenance, independent renderer and exact fallback checks."""
import argparse
import ast
import dataclasses
import hashlib
import json
import os
from pathlib import Path
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
import cv2
import numpy as np
from connected_eye import ConnectedEye,fixture_eye,trace_connected,fixture_textures
from audit_r24_eyes import integrate,metric,sample
from audit_r24_source import sha,write,native_hash,arc_distance,polygon
from coupled_eye_relight import apply_coupled
from shot_paths import add_shot_argument,from_args

ROOT=Path(__file__).resolve().parents[1]


def geometry_checks():
    eye=ConnectedEye()
    # Independent closed-form seam constraints, including the inherited too-small cap.
    r=eye.limbus_radius; z=eye.sag
    conic=abs(r*r+(1-eye.eccentricity**2)*z*z-2*eye.cornea_radius*z)
    sphere=abs(r*r+(z-eye.centre_depth)**2-eye.globe_radius**2)
    assert max(conic,sphere)<1e-12
    nc=np.array([r,0,(1-eye.eccentricity**2)*z-eye.cornea_radius]); nc/=np.linalg.norm(nc)
    ns=np.array([r,0,z-eye.centre_depth]); ns/=np.linalg.norm(ns)
    seam_angle=float(np.rad2deg(np.arccos(np.clip(nc@ns,-1,1))))
    rows=[]; yy,xx=np.mgrid[:96,:96]; points=np.stack([xx,yy],-1)
    for yaw,pitch in [(0,0),(18,0),(-18,0),(0,10),(0,-10),(18,-10),(-18,10)]:
        e=dataclasses.replace(eye,yaw=np.deg2rad(yaw),pitch=np.deg2rad(pitch))
        t=trace_connected(points,e)
        assert t['iris'].any() and t['pupil'].any() and t['sclera'].any()
        assert not ((t['iris']|t['pupil'])&~t['cap']).any()
        assert not (t['iris']&t['pupil']).any()
        assert not ((t['cornea']|t['sclera'])&t['lid']).any()
        assert not (t['reflection']&t['light_blocked']).any()
        assert np.max(np.linalg.norm(t['pupil_aperture_xy'][t['pupil']],axis=-1))<e.pupil_radius
        assert np.max(np.abs(e.apex+np.array([0,0,e.centre_depth])@e.rot.T-e.globe))<1e-12
        q=t['point'][t['cap']]
        residual=q[:,0]**2+q[:,1]**2+(1-e.eccentricity**2)*q[:,2]**2-2*e.cornea_radius*q[:,2]
        assert np.max(np.abs(residual))<1e-7
        rows.append(dict(yaw=yaw,pitch=pitch,iris_pixels=int(t['iris'].sum()),pupil_pixels=int(t['pupil'].sum()),
            camera_visible_light_blocked=int((t['cornea']&t['light_blocked']).sum())))
    assert sum(r['camera_visible_light_blocked'] for r in rows)>0
    closed=trace_connected(points,dataclasses.replace(eye,lid_height=0))
    assert not any(closed[k].any() for k in ('iris','pupil','cornea','reflection','sclera'))
    for i in (15,16,17):
        t=trace_connected(points,fixture_eye(i))
        if i==16: assert not t['cornea'].any()
        else: assert t['cornea'].any()
    # Same physical rays under unequal fx/fy; no implicit spherical/aspect assumption.
    q=np.array([[45.,44.],[52.,48.],[42.,53.],[48.,38.]])
    centre=np.array([47.5,47.5]); stretched=(q-centre)*[1.37,.83]+centre
    a=trace_connected(q,eye); b=trace_connected(stretched,eye,focal=(4110.,2490.))
    assert np.allclose(a['world'],b['world'],atol=1e-10,rtol=0)
    assert np.allclose(a['reflected'],b['reflected'],atol=1e-12,rtol=0)
    invalid=0
    for kw in [dict(pupil_radius=7),dict(lid_radius=12),dict(iris_depth=2),dict(backing_depth=3),
               dict(lid_height=-1),dict(limbus_radius=8),dict(globe=(0,0,float('nan')))]:
        try: ConnectedEye(**kw)
        except ValueError: invalid+=1
    assert invalid==7
    # An independent affine texture interpolation has a closed-form result.
    y,x=np.mgrid[:9,:11]; ramp=np.stack([x+3*y,2*x-y,x+y],-1).astype(float)
    xy=np.array([[2.21,3.73],[8.1,6.27]])
    expected=np.stack([xy[:,0]+3*xy[:,1],2*xy[:,0]-xy[:,1],xy.sum(1)],-1)
    assert np.max(np.abs(sample(ramp,(xy+.5)/[11,9])-expected))<1e-12
    return dict(seam_conic_residual=conic,seam_sphere_residual=sphere,seam_normal_jump_degrees=seam_angle,
        cap_sag=z,cap_radius=r,rotation_cases=rows,closed_eye_zero_visible=True,
        invalid_geometry_rejections=invalid,asymmetric_pixel_aspect_same_physical_rays=True,
        continuous_anatomy_certified=False,posterior_cornea_modeled=False)


def provenance(shot):
    manifest=json.loads((shot.logs/'r24_cycles.json').read_text())
    current=(ROOT/'scripts/connected_eye.py').read_bytes()
    new=b"    wall=np.full(cap.shape,np.inf)\n    for g in (g1,g2):\n        wall_point=q+td*1e-6+g[...,None]*td\n        valid=gh&(g>0)&(wall_point[...,2]>=eye.sag-1e-8)\n        wall=np.minimum(wall,np.where(valid,g,np.inf))\n"
    old=b"    wall=np.where((g1>0)&gh,g1,np.where(gh,g2,np.inf))\n"
    assert new in current
    original=current.replace(new,old)
    original_hash=hashlib.sha256(original).hexdigest()
    current_hash=hashlib.sha256(current).hexdigest()
    recorded_hash=manifest['code_sha256']['connected_eye.py']
    assert recorded_hash in (original_hash,current_hash)
    path=shot.composite/'r24_eyes/frozen_connected_eye_before_wall_fix.py'; path.write_bytes(original)
    names={'ConnectedEye','fixture_eye','fixture_textures'}
    select=lambda code:[ast.dump(n) for n in ast.parse(code).body if getattr(n,'name',None) in names]
    assert select(current)==select(original)
    for key in ('protocol_sha256',): assert manifest[key]==sha(ROOT/'docs/r24-eye-protocol.md')
    assert manifest['filter']=='BOX' and manifest['filter_width']==1 and manifest['samples']==4096
    assert not manifest['persistent_data'] and not manifest['denoising']
    images=0
    for r in manifest['records']:
        for f in r['files'].values(): assert sha(f['path'])==f['sha256']; images+=1
    assert images==72
    return dict(original_module_hash_reconstructed=recorded_hash==original_hash,
        current_module_hash_matches_render=recorded_hash==current_hash,
        rendered_parameters_and_texture_functions_unchanged=True,
        corrected_analytic_module_sha256=sha(ROOT/'scripts/connected_eye.py'),verified_rendered_files=images,
        frozen_original_module=str(path.relative_to(shot.dir)))


def saved_scores(shot):
    report=json.loads((shot.logs/'r24_transport.json').read_text())
    manifest=json.loads((shot.logs/'r24_cycles.json').read_text()); maximum=0.; disabled=0
    assert all(sha(ROOT/'scripts'/n)==h for n,h in report['code_sha256'].items())
    assert report['correction_sha256']==sha(ROOT/'docs/r24-render-orientation-correction.md')
    for row in report['records']:
        i=row['frame']; d=np.load(shot.composite/f'r24_eyes/transport_{i:06d}.npz')
        files=manifest['records'][i]['files']
        truth={k:cv2.imread(v['path'],cv2.IMREAD_UNCHANGED)[...,::-1].astype(float) for k,v in files.items()}
        preds=dict(clean=d['clean'],source=d['clean']+d['old'],target=d['clean']+d['new'],
            coupled=truth['source']-d['old']+d['new'],add_only=truth['source']+d['new'])
        for region,scores in row['scores'].items():
            for k,saved in scores.items():
                m=metric(preds[k],truth[k] if k in truth else truth['target'],d[f'mask_{region}'])
                for key,v in saved.items():
                    if v is None: assert m[key] is None
                    else: maximum=max(maximum,abs(m[key]-v))
        result,info=apply_coupled(truth['source'],d['mask_cornea'],d['clean'],d['new'],
            support=d['mask_cornea'],geometry_reliable=False)
        assert np.array_equal(result,truth['source']) and info['changed_channels']==0
        disabled+=1
        if i==16:
            assert not d['old'].any() and not d['new'].any() and not d['mask_cornea'].any()
        else:
            assert d['mask_iris'].any() and d['mask_pupil'].any(),i
    assert maximum<1e-12
    # Independent denser quadrature on signed gaze and blink cases; no parameter changes.
    dense=[]
    for i in (0,6,12,16,20):
        sums,masks,info=integrate(i,32)
        files=manifest['records'][i]['files']
        truth={k:cv2.imread(v['path'],cv2.IMREAD_UNCHANGED)[...,::-1].astype(float) for k,v in files.items()}
        dense.append(dict(frame=i,regions={r:dict(clean=metric(sums['clean'],truth['clean'],mask),
            source=metric(sums['clean']+sums['old'],truth['source'],mask)) for r,mask in masks.items()}))
    return dict(score_replay_max_difference=maximum,disabled_transactions=disabled,dense_32x32=dense,
        no_empty_open_iris_or_pupil_masks=True,production_repairs=0)


def source_evidence(shot):
    report=json.loads((shot.logs/'r24_source.json').read_text())
    labels=json.loads((ROOT/'scripts/data/r24_source_boundaries.json').read_text())
    freeze=json.loads((shot.logs/'r24_source_freeze.json').read_text())
    assert freeze==report['evidence'] and freeze['labels_sha256']==sha(ROOT/'scripts/data/r24_source_boundaries.json')
    for r,h,scored in zip(labels['records'],freeze['records'],report['records']):
        p=shot.relight/f"neural_iris_observations/{r['frame']:06d}_{r['side']}.npz"; d=np.load(p)
        assert sha(p)==h['archive_sha256'] and native_hash(d['linear_rgb'])==h['rgb_sha256']
        for key in ('limbus','pupil'): assert arc_distance(r[key],d[key+'_boundary_xy'])==scored[key]
        lid=polygon(d['linear_rgb'].shape[:2],r['lid'])
        dist=cv2.distanceTransform((~lid).astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        assert int(((d['complete_ellipse_labels']>0)&(dist>2)).sum())==scored['outside_lid_uncertainty_pixels']
    return dict(frozen_source_records=6,independent_human_truth=False,real_geometry_validated=False,
        complete_boundary_evidence=False,thresholds_tuned=False)


def video(shot,name,count,fps,shape):
    from verify_r20_eyes import video_checks
    result=video_checks(shot,name,count,fps,shape)
    if name=='r24_source':
        cap=cv2.VideoCapture(str(shot.final/f'{name}.mp4'))
        for i in (15,75,135):
            cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,frame=cap.read(); assert ok
            cv2.imwrite(str(shot.composite/name/f'decoded_{i:03d}.png'),frame)
        cap.release()
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__); add_shot_argument(ap)
    ap.add_argument('--skip-video',action='store_true'); args=ap.parse_args(); shot=from_args(args)
    from verify_r15 import fallback
    report=dict(geometry=geometry_checks(),provenance=provenance(shot),transport=saved_scores(shot),
        source_evidence=source_evidence(shot),exact_source=fallback(shot),code_sha256=sha(Path(__file__)))
    if not args.skip_video:
        report['videos']=dict(synthetic=video(shot,'r24_eyes',72,24.,(672,864,3)),
            source=video(shot,'r24_source',150,float(shot.frame_rate),(876,768,3)))
    write(shot.logs/'r24_verification.json',report)
    print('R24_VERIFICATION_COMPLETE',flush=True)


if __name__=='__main__': main()
