"""Collect pretrained EllSeg proposals for any prepared --shot, in isolated venv.

This is geometry evaluation only. Full ellipses do not identify visible material;
all keyed eyes retain exact source and receive no scene reflection replacement.
"""
import argparse
import json
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2
import numpy as np
from color_pipeline import linear_to_srgb, srgb_to_linear
from ellseg_iris import EllSeg, summary
from iris_observations import eye_crop, review_tile
from shot_paths import add_shot_argument, from_args
from temporal_iris import apply_eye_repair
from ritnet_iris import RITnet


def overlay(rgb, prediction, visible_labels=None):
    out = np.clip(rgb[...,::-1],0,255).astype(np.uint8)
    tint = np.zeros_like(out)
    if visible_labels is None:
        tint[prediction['labels']==1] = (0,110,0)
        tint[prediction['labels']==2] = (100,0,100)
    else:
        tint[visible_labels==1] = (150,100,0)
        tint[visible_labels==2] = (0,110,0)
        tint[visible_labels==3] = (100,0,100)
    out = cv2.addWeighted(out,.85,tint,.4,0)
    for name, e in prediction['ellipses'].items():
        pts = np.rint(e['boundary_xy']*16).astype(np.int32)
        cv2.polylines(out,[pts],True,(0,255,255) if name=='pupil' else (255,255,0),1,cv2.LINE_AA,4)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_shot_argument(ap)
    ap.add_argument('--review',action='store_true')
    args = ap.parse_args()
    shot = from_args(args)
    meta = json.loads((shot.relight/'face_landmarks.json').read_text())
    lm = np.load(shot.relight/'face_landmarks.npz')
    coords = meta['coordinates']
    aspect = float(coords['anamorphic_squeeze'])
    if len(lm['found']) != shot.frame_count or not np.isfinite(aspect) or aspect<=0:
        raise ValueError('Invalid tracking frame count or pixel aspect')
    model = EllSeg()
    visibility = RITnet()
    out = shot.relight/'neural_iris_observations'
    out.mkdir(parents=True,exist_ok=True)
    review = shot.composite/'r13_neural_observations'
    if args.review: review.mkdir(parents=True,exist_ok=True)
    rows=[]
    for i in range(shot.frame_count):
        raw=cv2.imread(str(shot.fg(i)),cv2.IMREAD_UNCHANGED)
        if raw is None or raw.dtype!=np.float32 or raw.ndim!=3 or raw.shape[2]!=3:
            raise ValueError('Expected scene-linear keyed RGB EXR')
        if list(raw.shape[1::-1])!=coords['resolution']:
            raise ValueError('Key and tracking coordinates disagree')
        eyes={}; tiles=[]
        for side in ('left','right'):
            crop,reason=(None,'missing_face_detection')
            if lm['found'][i]:
                crop,reason=eye_crop(raw[...,::-1],lm['points'][i],meta['topology']['iris_'+side],
                                     meta['topology']['eyelid_'+side],aspect)
            info={'geometry_validated':False,'visibility_validated':False,
                  'scene_reflection_allowed':False,'mode':'source_keyed','changed_channels':0}
            if crop is not None:
                rgb=crop['linear_rgb']
                pred=model.predict(linear_to_srgb(rgb)*255,pixel_aspect=aspect)
                visible=visibility.predict(linear_to_srgb(rgb)*255,pixel_aspect=aspect)
                result,fallback=apply_eye_repair(rgb,np.zeros(rgb.shape[:2],bool),
                    geometry_reliable=False,correspondence_reliable=False)
                assert np.array_equal(result,rgb)
                info.update(summary(pred),origin_xy=crop['origin'].tolist(),
                    seed_xy=crop['seed'].tolist(),radius=crop['radius'],
                    eye_rotation=crop['eye_rotation'],changed_channels=fallback['changed_channels'])
                np.savez_compressed(out/('%06d_%s.npz'%(i,side)),linear_rgb=rgb,
                    complete_ellipse_labels=pred['labels'],origin_xy=crop['origin'],
                    visible_part_proposal=visible,
                    **{k+'_boundary_xy':v['boundary_xy'] for k,v in pred['ellipses'].items()})
                if args.review:
                    source=review_tile(rgb,side+' SOURCE KEYED')
                    # Label overlay explicitly; it is never composited into performer pixels.
                    visual=srgb_to_linear(overlay(linear_to_srgb(rgb)*255,pred,visible)[...,::-1]/255.)
                    tiles.append(np.concatenate([source,review_tile(visual,'ELLSEG + RITNET PROPOSALS'),
                        review_tile(result,'EXACT SOURCE FALLBACK')],1))
            else:
                info['reason']=reason
                if args.review:
                    blank=review_tile(np.zeros((32,32,3),np.float32),side+' no valid ROI; source retained')
                    tiles.append(np.concatenate([blank]*3,1))
            eyes[side]=info
        rows.append({'frame':i,'source_frame':shot.source_start_frame+i,'eyes':eyes})
        if args.review and not cv2.imwrite(str(review/('%06d.png'%i)),np.concatenate(tiles,0)):
            raise RuntimeError('Review write failed')
        if i%25==0: print('keyed neural observations',i,flush=True)
    manifest={'schema':1,'provenance':model.provenance,'visibility_provenance':visibility.provenance,
              'frames':rows,'pixel_aspect':aspect,
              'status':'unvalidated geometry and visible-part proposals; exact source fallback throughout',
              'material_warning':'RITnet iris class can contain original reflections; it is not a clean donor or removal mask'}
    (out/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    report={k:v for k,v in manifest.items() if k!='frames'}
    report.update(frames=len(rows),eyes=sum(len(r['eyes']) for r in rows),
        changed_channels=sum(e['changed_channels'] for r in rows for e in r['eyes'].values()))
    (shot.logs/'r13_neural_observations.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
