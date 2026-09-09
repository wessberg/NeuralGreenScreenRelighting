"""Fit partial source arcs and retain conditional geometry/material ambiguity."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from audit_r24_source import ROOT, sha, write, native_hash
from color_pipeline import linear_to_srgb
from coupled_eye_relight import apply_coupled
from partial_eye_geometry import (split_arc, fit_family, curve, distances,
    connected_candidate, pupil_distance, conservative_open, route_family)
from shot_paths import add_shot_argument, from_args


def camera(shot, manifest):
    config=json.loads((shot.dir/'shot.json').read_text())
    w,h=config['source_resolution']; cam=config['camera']
    fx=w*cam['lens_mm']/cam['sensor_width_mm']
    return np.array([fx,fx*manifest['pixel_aspect']]),np.array([(w-1)/2,(h-1)/2])


def freeze(shot, labels):
    prep=json.loads((shot.logs/'r25_source_preflight.json').read_text())
    assert prep['protocol_sha256']==sha(ROOT/'docs/r25-eye-protocol.md')
    assert [(r['frame'],r['side']) for r in prep['records']]==[(r['frame'],r['side']) for r in labels['records']]
    for r in prep['records']:
        p=shot.relight/f"neural_iris_observations/{r['frame']:06d}_{r['side']}.npz"
        assert sha(p)==r['archive_sha256']
        with np.load(p) as d: assert native_hash(d['linear_rgb'])==r['rgb_sha256']
    evidence=dict(preflight_sha256=sha(shot.logs/'r25_source_preflight.json'),
        labels_sha256=sha(ROOT/'scripts/data/r25_source_boundaries.json'),
        protocol_sha256=prep['protocol_sha256'],records=prep['records'])
    path=shot.logs/'r25_source_freeze.json'
    if path.exists(): assert json.loads(path.read_text())==evidence
    else: write(path,evidence)
    return evidence


def fit_record(label, rgb, origin, focal, centre, out):
    local_centre=centre-origin
    train,check,ti,ci=split_arc(label['limbus'])
    family=fit_family(train,focal,local_centre)
    compatible=[r for r in family if r['compatible']]
    best=min(family,key=lambda r:r['train_rms_px'])
    # Checking happens after all train-only parameters/selection are fixed.
    for r in family:
        contour=curve(r['parameters'],*np.deg2rad([r['yaw_deg'],r['pitch_deg']]),focal,local_centre)
        r['check_distances_px']=distances(check,contour).tolist()
    pupils=np.array(label['pupil'],float).reshape(-1,2)
    optical=[]; eyes=[]
    for idx,r in enumerate(family):
        if not r['compatible']: continue
        for radius in (7.2,7.8,8.4):
            for pupil in (1.2,1.8,2.4,3.0):
                eye=connected_candidate(r,focal,local_centre,radius,pupil)
                d,hits=pupil_distance(pupils,eye,focal,local_centre)
                accepted=bool(not len(pupils) or (hits.all() and (d<=2).all()))
                optical.append(dict(family_index=idx,cornea_radius_mm=radius,pupil_radius_mm=pupil,
                    pupil_distances_px=d.tolist(),pupil_cap_hits=hits.tolist(),compatible=accepted))
                if accepted: eyes.append(eye)
    opening,safe=conservative_open(rgb.shape[:2],label['lid'],label['reflection'])
    y,x=np.nonzero(safe); xy=np.stack([x,y],-1)
    iris,pupil=route_family(xy,eyes,focal,local_centre)
    routes=np.zeros(rgb.shape[:2],np.uint8)
    routes[y[iris],x[iris]]=1; routes[y[pupil],x[pupil]]=2
    fallback,info=apply_coupled(rgb,opening,geometry_reliable=False)
    assert np.array_equal(fallback,rgb) and info['changed_channels']==0
    path=out/f"observations_{label['frame']:06d}_{label['side']}.npz"
    np.savez_compressed(path,linear_rgb=rgb,origin_xy=origin,route=routes,
        candidate_xy=xy,iris_candidate=iris,pupil_candidate=pupil,
        source_fallback=fallback,clean_material_certified=np.array(False))
    contours=np.array([curve(r['parameters'],*np.deg2rad([r['yaw_deg'],r['pitch_deg']]),
                              focal,local_centre) for r in compatible])
    spread={}
    if len(compatible):
        params=np.array([r['parameters'] for r in compatible])
        for name,values in [('depth_mm',params[:,2]),('centre_x_px',params[:,0]),('centre_y_px',params[:,1]),
                            ('top_boundary_y_px',contours[:,:,1].min(1)),
                            ('yaw_deg',[r['yaw_deg'] for r in compatible]),
                            ('pitch_deg',[r['pitch_deg'] for r in compatible])]:
            spread[name]=[float(np.min(values)),float(np.max(values))]
    return dict(frame=label['frame'],side=label['side'],train_indices=ti.tolist(),check_indices=ci.tolist(),
        family=family,best_train_index=family.index(best),compatible_geometry_count=len(compatible),
        optical_profiles=optical,compatible_optical_count=len(eyes),conditional_ranges=spread,
        lid_open_pixels=int(opening.sum()),conservative_observed_pixels=len(xy),
        iris_candidate_pixels=int(iris.sum()),pupil_candidate_pixels=int(pupil.sum()),
        unknown_observed_pixels=int((~iris&~pupil).sum()),observation_archive_sha256=sha(path),
        geometry_validated=False,clean_material_certified=False,production_repairs=0)


def draw_review(shot,manifest,labels,records,focal,centre):
    out=shot.composite/'r25_source/review'; out.mkdir(parents=True,exist_ok=True)
    lookup={(r['frame'],r['side']):(l,r) for l,r in zip(labels['records'],records)}
    for row in manifest['frames']:
        panels=[]
        for side in ('left','right'):
            with np.load(shot.relight/f'neural_iris_observations/{row["frame"]:06d}_{side}.npz') as d:
                rgb,origin=d['linear_rgb'],d['origin_xy']
            raw=np.clip(linear_to_srgb(rgb)[...,::-1]*255,0,255).astype('uint8')
            source=cv2.resize(raw,(384,384),interpolation=cv2.INTER_NEAREST); overlay=source.copy()
            scale=np.array([384/raw.shape[1],384/raw.shape[0]])
            xy=lambda p:np.rint((np.asarray(p)+.5)*scale-.5).astype('int32')
            pair=lookup.get((row['frame'],side))
            status='NO FIT AT THIS FRAME - original eye retained'
            if pair:
                label,record=pair
                # Draw only the portion of each candidate ring inside the actual observed lid opening.
                for r in record['family']:
                    if not r['compatible']: continue
                    c=curve(r['parameters'],*np.deg2rad([r['yaw_deg'],r['pitch_deg']]),focal,centre-origin)
                    lid=np.array(label['lid'],np.float32)
                    visible=[cv2.pointPolygonTest(lid,tuple(map(float,p)),False)>=0 for p in c]
                    points=xy(c)
                    for j in range(len(c)):
                        if visible[j] and visible[(j+1)%len(c)]:
                            cv2.line(overlay,tuple(points[j]),tuple(points[(j+1)%len(c)]),(0,120,180),1,cv2.LINE_AA)
                cv2.polylines(overlay,[xy(label['lid'])],True,(255,160,0),1,cv2.LINE_AA)
                cv2.polylines(overlay,[xy(label['reflection'])],True,(0,0,255),1,cv2.LINE_AA)
                for j,p in enumerate(xy(label['limbus'])):
                    cv2.circle(overlay,tuple(p),3,(255,255,0) if j in record['check_indices'] else (0,255,0),-1)
                route=np.load(shot.composite/f'r25_source/observations_{row["frame"]:06d}_{side}.npz')['route']
                for value,color in [(1,(0,255,0)),(2,(255,0,255))]:
                    ys,xs=np.nonzero(route==value)
                    for p in xy(np.stack([xs,ys],-1)): cv2.circle(overlay,tuple(p),1,color,-1)
                status=f'{record["compatible_geometry_count"]} ring / {record["compatible_optical_count"]} optical candidates; UNVALIDATED'
            header=np.zeros((76,768,3),np.uint8)
            for y,text,color in [(20,f'{side.upper()} source {row["source_frame"]} | SOURCE / PARTIAL-ARC FIT',(255,255,255)),
                    (42,status,(220,220,220)),(64,'Green fit / cyan check; dots = observed RGB routes, NOT clean texture',(180,180,180))]:
                cv2.putText(header,text,(8,y),0,.43,color,1)
            panels.append(np.concatenate([header,np.concatenate([source,overlay],1)],0))
        assert cv2.imwrite(str(out/f'{row["frame"]:06d}.png'),np.concatenate(panels,0))


def main():
    ap=argparse.ArgumentParser(description=__doc__); add_shot_argument(ap)
    ap.add_argument('--freeze-only',action='store_true'); ap.add_argument('--review-only',action='store_true')
    args=ap.parse_args(); shot=from_args(args)
    labels=json.loads((ROOT/'scripts/data/r25_source_boundaries.json').read_text())
    evidence=freeze(shot,labels)
    if args.freeze_only: print('R25_SOURCE_FROZEN'); return
    manifest=json.loads((shot.relight/'neural_iris_observations/manifest.json').read_text())
    focal,centre=camera(shot,manifest); records=[]
    out=shot.composite/'r25_source'; out.mkdir(parents=True,exist_ok=True)
    if args.review_only:
        report=json.loads((shot.logs/'r25_source.json').read_text()); records=report['records']
    else:
        for label in labels['records']:
            with np.load(shot.relight/f"neural_iris_observations/{label['frame']:06d}_{label['side']}.npz") as d:
                rgb,origin=d['linear_rgb'],d['origin_xy']
            record=fit_record(label,rgb,origin,focal,centre,out); records.append(record)
            print('R25_FIT',label['frame'],label['side'],record['compatible_geometry_count'],
                  record['compatible_optical_count'],record['iris_candidate_pixels'],record['pupil_candidate_pixels'],flush=True)
        report=dict(evidence=evidence,records=records,focal_px=focal.tolist(),centre_px=centre.tolist(),
            camera_status='Shot camera assumed; 5.85mm ring is a supplied scale prior',
            geometry_validated=False,clean_material_certified=False,production_repairs=0,
            neural_contours_loaded=False,full_neural_method_run=False,
            filter_status='3x3 finite BOX quadrature diagnostic; actual camera PSF is unknown',
            lid_status='Source 2D aperture with uncertainty; 3D and reflected-light occlusion unresolved',
            code_sha256={n:sha(ROOT/'scripts'/n) for n in ['partial_eye_geometry.py','audit_r25_source.py',
                         'connected_eye.py','corneal_transport.py','coupled_eye_relight.py']})
        write(shot.logs/'r25_source.json',report)
    draw_review(shot,manifest,labels,records,focal,centre)
    print('R25_SOURCE_COMPLETE',flush=True)


if __name__=='__main__': main()
