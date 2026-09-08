"""Verify round 14 evidence, native search isolation, fallback, and encoded video."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2
import numpy as np

from anamorphic import SQUEEZE
from neural_iris_correspondence import Proposal, ellipse_geometry, fit_donor
from shot_paths import add_shot_argument, from_args
from temporal_iris import IrisGeometry, apply_eye_repair

ROOT = Path(__file__).resolve().parents[1]


def geometry_checks():
    residuals = []
    for aspect in (1., SQUEEZE):
        for angle in (-1.1, -.3, .45, 1.2):
            t = np.linspace(0, 2*np.pi, 720, endpoint=False)
            cs, sn = np.cos(angle), np.sin(angle)
            shape = np.array([[cs,sn],[-sn,cs]])
            centre = np.array([34., 37.])
            curve = np.column_stack([18*np.cos(t),12*np.sin(t)]) @ shape / [aspect,1]
            pred = dict(ellipses={n:dict(center_xy=centre,boundary_xy=centre+curve*scale)
                                  for n,scale in [('limbus',1.),('pupil',.4)]})
            geometry, offset = ellipse_geometry(pred)
            assert offset == 0 and abs(geometry.pupil_ratio-.4)<1e-12
            u = geometry.material(pred['ellipses']['limbus']['boundary_xy'])
            residuals.append(float(np.max(np.abs(u[:,0]-1))))
            assert residuals[-1]<1e-12
            assert np.max(np.abs(geometry.native(u)-pred['ellipses']['limbus']['boundary_xy']))<1e-10
    return dict(angles=4,pixel_aspects=[1.,SQUEEZE],max_limbus_radial_error=max(residuals))


def search_isolation():
    # Known synthetic geometry/texture exercises supported fitting, independently
    # of the real detector. Perturb all searched query/validation contributors;
    # selected angle and training score must stay exactly invariant.
    yy,xx = np.mgrid[:96,:96]
    g = IrisGeometry((48.,48.),(22.,18.),.2,.35)
    xy = np.column_stack([xx.ravel(),yy.ravel()])
    uv = g.material(xy).reshape(96,96,2)
    mask = (uv[...,0]>.1) & (uv[...,0]<.9)
    rgb = np.stack([100+30*np.sin(xx*.3)+20*np.cos(yy*.2),
                    100+35*np.cos(xx*.25+yy*.1),100+25*np.sin(yy*.3)],-1).astype(np.float32)
    ob = Proposal(g,mask,np.zeros((96,96),np.int8),np.zeros((96,96),np.int8),True,[],{})
    query = g.native(np.array([[.5,1.1],[.5,2.1]]))
    fit = fit_donor(rgb,ob,rgb,ob,query,rotation=True)
    assert fit is not None and fit['training_mae']==0 and fit['torsion_radians']==0
    poisoned = rgb.copy()
    footprint = np.concatenate([fit['footprints']['search_query'],fit['footprints']['search_validation']])
    for x,y in np.floor(footprint).astype(int):
        if 0<=x<95 and 0<=y<95:
            poisoned[y:y+2,x:x+2] = [255,0,255]
    other = fit_donor(rgb,ob,poisoned,ob,query,rotation=True)
    assert other is not None
    assert other['training_mae']==fit['training_mae']
    assert other['torsion_radians']==fit['torsion_radians']
    assert np.array_equal(other['footprints']['training'],fit['footprints']['training'])
    assert other['validation_mae']>fit['validation_mae']
    return dict(synthetic_known_geometry=True, training_pixels=fit['training_pixels'],
                searched_native_footprints=len(footprint), changed_validation_mae=other['validation_mae'],
                selected_angle_and_training_exact=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    add_shot_argument(ap)
    ap.add_argument('--unit-only',action='store_true')
    args=ap.parse_args()
    shot=from_args(args)
    check=dict(geometry=geometry_checks(),search_isolation=search_isolation())
    if args.unit_only:
        print(json.dumps(check,indent=2))
        return
    report=json.loads((shot.logs/'r14_neural.json').read_text())
    for key,path in [('protocol_sha256',ROOT/'docs/r14-evaluation-protocol.md'),
                     ('annotations_sha256',ROOT/'scripts/data/r14_native_labels.json')]:
        assert hashlib.sha256(path.read_bytes()).hexdigest()==report[key]
    assert not set(report['donor_ids']) & set(report['target_ids'])
    assert report['attempted_holes']==100 and report['attempted_pixels']==500
    assert len(report['hidden_rgb_invariance'])==100
    assert all(r['exact'] for r in report['hidden_rgb_invariance'])
    check['hidden_rgb_exact_cases']=len(report['hidden_rgb_invariance'])
    check['hidden_rgb_cases_with_support']=sum(r['supported_pixels']>0 for r in report['hidden_rgb_invariance'])
    check['real_fit_count']=sum(len(c[m]['info']['fits']) for c in report['cases'] for m in ('fixed','torsion'))
    assert check['hidden_rgb_cases_with_support']>0 and check['real_fit_count']>0
    labelled=report['supplementary_labelled_iris']['cases']
    assert len(labelled)==6 and sum(c['hidden_pixels'] for c in labelled)==30
    assert all(c['hidden_rgb_invariance'] for c in labelled)
    check['supplementary_labelled_hidden_rgb_exact_cases']=len(labelled)
    assert report['production_repairs']==0 and not report['scene_reflection_allowed']
    old=json.loads((shot.composite/'r10_iris/correspondence_detail.json').read_text())
    protected=old['checks']['protected_manifest']
    changed=[p for p,h in protected.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
    assert not changed,changed
    check['protected_hashes_unchanged']=len(protected)
    manifest=json.loads((shot.relight/'neural_iris_observations/manifest.json').read_text())
    count=0
    for row in manifest['frames']:
        raw=cv2.imread(str(shot.fg(row['frame'])),cv2.IMREAD_UNCHANGED)[...,::-1]
        for side,eye in row['eyes'].items():
            if 'origin_xy' not in eye:
                continue
            source=np.load(shot.relight/'neural_iris_observations'/f"{row['frame']:06d}_{side}.npz")['linear_rgb']
            h,w=source.shape[:2]
            x,y=eye['origin_xy']
            assert np.array_equal(source,raw[y:y+h,x:x+w])
            result,info=apply_eye_repair(source,np.ones((h,w),bool),geometry_reliable=False)
            assert np.array_equal(result,source) and info['changed_channels']==0
            count+=1
    assert count==300
    check['exact_keyed_eye_fallbacks']=count
    video=shot.final/'r14_neural.mp4'
    cap=cv2.VideoCapture(str(video))
    fps=cap.get(cv2.CAP_PROP_FPS)
    frames=[]
    while True:
        ok,im=cap.read()
        if not ok:
            break
        thumb=cv2.resize(im,(192,round(192*im.shape[0]/im.shape[1])))
        cv2.putText(thumb,str(950+len(frames)),(3,13),0,.4,(255,255,255),1)
        frames.append(thumb)
    cap.release()
    assert len(frames)==50 and abs(fps-float(shot.frame_rate))<.001,(len(frames),fps)
    sheet=np.concatenate([np.concatenate(frames[i:i+5],1) for i in range(0,50,5)],0)
    path=shot.composite/'r14_neural/contact_sheet.jpg'
    assert cv2.imwrite(str(path),sheet)
    check['video']=dict(file=str(video.relative_to(shot.dir)),decoded_frames=50,fps=fps,
                        visual_inspection='pending',artist_approval=False)
    for name in ('audit_r14_neural.py','neural_iris_correspondence.py','verify_r14.py'):
        compile((ROOT/'scripts'/name).read_text(),name,'exec')
    (shot.logs/'r14_verification.json').write_text(json.dumps(check,indent=2),encoding='utf-8')
    print(json.dumps(check,indent=2))


if __name__=='__main__':
    main()
