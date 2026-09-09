"""Source-only sparse boundary evidence, scored without fitting or completing arcs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
import cv2
import numpy as np
from color_pipeline import linear_to_srgb
from shot_paths import add_shot_argument,from_args

ROOT=Path(__file__).resolve().parents[1]


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')


def native_hash(rgb): return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()


def polygon(shape,points):
    mask=np.zeros(shape,np.uint8)
    cv2.fillPoly(mask,[np.array(points,np.int32)],1)
    return mask.astype(bool)


def arc_distance(points,curve):
    if not points: return dict(samples=0,median_px=None,maximum_px=None,within_2px=0)
    points=np.array(points,float); a=np.array(curve,float); b=np.roll(a,-1,axis=0)
    delta=b-a
    u=np.clip(np.sum((points[:,None]-a)*delta,-1)/np.maximum(np.sum(delta*delta,-1),1e-15),0,1)
    distances=np.linalg.norm(points[:,None]-(a+u[...,None]*delta),axis=-1).min(1)
    return dict(samples=len(points),median_px=float(np.median(distances)),maximum_px=float(distances.max()),
                within_2px=int((distances<=2).sum()),distances_px=distances.tolist())


def main():
    ap=argparse.ArgumentParser(description=__doc__); add_shot_argument(ap)
    ap.add_argument('--freeze-only',action='store_true'); args=ap.parse_args(); shot=from_args(args)
    path=ROOT/'scripts/data/r24_source_boundaries.json'; labels=json.loads(path.read_text())
    root=shot.relight/'neural_iris_observations'; manifest=json.loads((root/'manifest.json').read_text())
    freeze_path=shot.logs/'r24_source_freeze.json'
    hashes=[]
    # Only RGB and origins are opened before this evidence/provenance record exists.
    for r in labels['records']:
        file=root/f"{r['frame']:06d}_{r['side']}.npz"
        with np.load(file) as d:
            hashes.append(dict(frame=r['frame'],side=r['side'],source_frame=manifest['frames'][r['frame']]['source_frame'],
                rgb_sha256=native_hash(d['linear_rgb']),origin_xy=d['origin_xy'].tolist(),archive_sha256=sha(file)))
    evidence=dict(labels_sha256=sha(path),protocol_sha256=sha(ROOT/'docs/r24-eye-protocol.md'),records=hashes,
        provenance=labels['provenance'],independent_human_truth=False,geometry_fitted=False)
    if args.freeze_only:
        if freeze_path.exists(): assert json.loads(freeze_path.read_text())==evidence
        else: write(freeze_path,evidence)
        print('R24_SOURCE_FROZEN'); return
    assert json.loads(freeze_path.read_text())==evidence
    records=[]; by_key={(r['frame'],r['side']):r for r in labels['records']}
    for r in labels['records']:
        d=np.load(root/f"{r['frame']:06d}_{r['side']}.npz")
        shape=d['linear_rgb'].shape[:2]; lid=polygon(shape,r['lid'])
        # Conservative rejection only beyond the 2px uncertainty band of the lid polygon.
        distance=cv2.distanceTransform((~lid).astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        proposed=d['complete_ellipse_labels']>0; pupil=d['complete_ellipse_labels']==2
        reflection=polygon(shape,r['reflection'])
        records.append(dict(frame=r['frame'],side=r['side'],
            limbus=arc_distance(r['limbus'],d['limbus_boundary_xy']),
            pupil=arc_distance(r['pupil'],d['pupil_boundary_xy']),
            proposed_eye_pixels=int(proposed.sum()),outside_lid_uncertainty_pixels=int((proposed&(distance>2)).sum()),
            proposed_pupil_pixels=int(pupil.sum()),reflection_overlap_pixels=int((pupil&reflection).sum()),
            reflection_pixels=int(reflection.sum()),complete_boundary_observed=False,
            geometry_validated=False,clean_material_certified=False))
    out=shot.composite/'r24_source/review'; out.mkdir(parents=True,exist_ok=True)
    for row in manifest['frames']:
        panels=[]
        for side in ('left','right'):
            d=np.load(root/f"{row['frame']:06d}_{side}.npz")
            raw=np.clip(linear_to_srgb(d['linear_rgb'])[...,::-1]*255,0,255).astype('uint8')
            source=cv2.resize(raw,(384,384),interpolation=cv2.INTER_NEAREST); overlay=source.copy()
            scale=np.array([384/raw.shape[1],384/raw.shape[0]])
            for field,color in [('limbus_boundary_xy',(0,200,255)),('pupil_boundary_xy',(255,80,255))]:
                points=np.rint((d[field]+.5)*scale-.5).astype('int32')
                cv2.polylines(overlay,[points],True,color,1,cv2.LINE_AA)
            r=by_key.get((row['frame'],side))
            if r:
                for field,color in [('limbus',(0,255,0)),('pupil',(255,255,0)),('lid',(255,160,0)),('reflection',(20,20,255))]:
                    if not r[field]: continue
                    points=np.rint((np.array(r[field])+.5)*scale-.5).astype('int32')
                    cv2.polylines(overlay,[points],field in ('lid','reflection'),color,2,cv2.LINE_AA)
            pair=np.concatenate([source,overlay],1)
            header=np.zeros((54,768,3),np.uint8)
            cv2.putText(header,f'{side.upper()} source {row["source_frame"]} | SOURCE / PROPOSALS',(8,21),0,.55,(255,255,255),1)
            cv2.putText(header,'Sparse source arcs on 715 / 775 / 835 only; geometry UNVALIDATED',(8,43),0,.45,(200,200,200),1)
            panels.append(np.concatenate([header,pair],0))
        board=np.concatenate(panels,0)
        assert cv2.imwrite(str(out/f'{row["frame"]:06d}.png'),board)
    report=dict(evidence=evidence,records=records,frames=len(manifest['frames']),review_shape=list(board.shape),
        source_fps=str(shot.frame_rate),code_sha256=sha(Path(__file__)),production_repairs=0,
        interpretation='Sparse visible-arc compatibility only; reflection overlap leaves pupil material unknown. No 3D fit.',
        thresholds_fitted=False,full_neural_method_run=False)
    write(shot.logs/'r24_source.json',report)
    print('R24_SOURCE_COMPLETE',json.dumps(records))


if __name__=='__main__': main()
