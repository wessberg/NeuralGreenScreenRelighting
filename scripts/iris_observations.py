"""Collect experimental native iris observations for any prepared --shot.

Consumes scene-linear Rec.709 keyed EXRs and the existing landmark sidecar. A
landmark is a coarse ROI seed only. Resolution, temporal IDs, topology, eye roll
and pixel aspect come from the shot/track, never from a particular performer.

Writes reusable per-eye NPZ observations and an audit manifest. Geometry/visibility
are NOT certified by this experimental detector: source-keyed fallback remains
mandatory. --review writes a source/fallback comparison for visual inspection.
No production compositor, keyed source, colour ingest or camera config is modified.
"""
import argparse
from dataclasses import asdict
import json
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

import cv2
import numpy as np
from color_pipeline import linear_to_srgb
from native_iris import detect
from shot_paths import add_shot_argument, from_args
from temporal_iris import apply_eye_repair


def eye_crop(rgb, points, iris_ids, lid_ids, pixel_aspect):
    """Return a native crop and coarse orientation; no landmark polygon is filled."""
    iris = np.asarray(points)[iris_ids, :2]
    lid = np.asarray(points)[lid_ids, :2]
    if not np.isfinite(iris).all() or not np.isfinite(lid).all():
        return None, 'missing_eye_landmarks'
    radius = float(np.linalg.norm((iris[1:]-iris[0])*[pixel_aspect,1], axis=1).mean())
    if radius < 2:
        return None, 'iris_below_native_resolution_floor'
    # Bounding margins scale with the observation, including very small or macro eyes.
    half = np.ceil(2.5*radius/np.array([pixel_aspect,1])).astype(int)+3
    lo, hi = np.floor(iris[0]-half).astype(int), np.ceil(iris[0]+half+1).astype(int)
    if (lo < 0).any() or (hi > rgb.shape[1::-1]).any():
        return None, 'eye_roi_outside_keyed_frame'
    physical = (lid-lid.mean(0))*[pixel_aspect,1]
    _, _, axes = np.linalg.svd(physical, full_matrices=False)
    axis = axes[0] if axes[0,0] >= 0 else -axes[0]
    rotation = float(np.arctan2(axis[1],axis[0]))
    return {'linear_rgb':rgb[lo[1]:hi[1],lo[0]:hi[0]].copy(), 'origin':lo,
            'seed':iris[0]-lo, 'radius':radius, 'eye_rotation':rotation}, None


def observe_keyed(rgb, points, topology, *, pixel_aspect, found=True):
    """Array API shared by the CLI and tests; returns exact source on unvalidated views.

    The entire automatic detector is still experimental. A numerical candidate is
    never promoted to validated geometry merely because its donors agree.
    """
    results = {}
    for side in ('left','right'):
        if not found:
            results[side] = {'mode':'source_keyed','reason':'missing_face_detection'}
            continue
        crop, reason = eye_crop(rgb, points, topology['iris_'+side],
                               topology['eyelid_'+side], pixel_aspect)
        if reason:
            results[side] = {'mode':'source_keyed','reason':reason}
            continue
        display = linear_to_srgb(crop['linear_rgb'])*255
        ob = detect(display, crop['seed'], crop['radius'], pixel_aspect=pixel_aspect,
                    eye_rotation=crop['eye_rotation'])
        result, fallback = apply_eye_repair(crop['linear_rgb'], ob.reflection,
            geometry_reliable=False, correspondence_reliable=False)
        reason = ('unvalidated_native_visibility' if ob.candidate_ok else ','.join(ob.reasons))
        if ob.candidate_ok and ob.metrics['reflection_crosses_pupil']:
            reason = 'reflection_crosses_unvalidated_pupil'
        results[side] = {'mode':'source_keyed', 'reason':reason, 'crop':crop, 'ob':ob,
                         'output':result, 'fallback':fallback}
    return results


def review_tile(rgb, title, size=320):
    bgr = np.clip(linear_to_srgb(rgb)[..., ::-1]*255,0,255).astype(np.uint8)
    scale = min(size/bgr.shape[1], size/bgr.shape[0])
    bgr = cv2.resize(bgr, (max(1,round(bgr.shape[1]*scale)),max(1,round(bgr.shape[0]*scale))),
                     interpolation=cv2.INTER_NEAREST)
    out = np.zeros((size+28,size,3),np.uint8)
    y, x = 28+(size-bgr.shape[0])//2, (size-bgr.shape[1])//2
    out[y:y+bgr.shape[0],x:x+bgr.shape[1]] = bgr
    cv2.putText(out,title,(5,19),0,.4,(255,255,255),1)
    return out


def store_manifest(shot, manifest):
    out=shot.relight/'iris_observations'
    (out/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    summary={k:v for k,v in manifest.items() if k!='frames'}
    summary['detail']='relight/iris_observations/manifest.json'
    summary['frame_count']=len(manifest['frames'])
    eyes=[e for f in manifest['frames'] for e in f['eyes'].values()]
    summary['candidate_eyes']=sum(e.get('candidate_ok',False) for e in eyes)
    summary['changed_channels']=sum(e.get('changed_channels',0) for e in eyes)
    summary['fallback_reasons']={r:sum(e['reason']==r for e in eyes) for r in sorted({e['reason'] for e in eyes})}
    (shot.logs/'iris_observations.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_shot_argument(ap)
    ap.add_argument('--review',action='store_true')
    args = ap.parse_args()
    shot = from_args(args)
    sidecar = shot.relight/'face_landmarks.json'
    archive = shot.relight/'face_landmarks.npz'
    if not sidecar.exists() or not archive.exists():
        raise SystemExit('Run face_track.py --shot for this shot first; no tracking store exists.')
    meta = json.loads(sidecar.read_text(encoding='utf-8'))
    lm = np.load(archive)
    coords = meta['coordinates']
    pixel_aspect = float(coords['anamorphic_squeeze'])
    if not np.isfinite(pixel_aspect) or pixel_aspect <= 0:
        raise SystemExit('Tracking sidecar must declare a valid pixel aspect.')
    if len(lm['found']) != shot.frame_count:
        raise SystemExit('Tracking frame count disagrees with shot selection.')
    out = shot.relight/'iris_observations'
    out.mkdir(parents=True,exist_ok=True)
    rows = []
    for i in range(shot.frame_count):
        raw = cv2.imread(str(shot.fg(i)),cv2.IMREAD_UNCHANGED)
        if raw is None or raw.ndim != 3 or raw.shape[2] != 3 or raw.dtype != np.float32:
            raise SystemExit('Expected a scene-linear RGB keyed EXR: '+str(shot.fg(i)))
        if list(raw.shape[1::-1]) != coords['resolution']:
            raise SystemExit('Keyed/tracking pixel spaces disagree; align ingest/tracking before iris detection.')
        result = observe_keyed(raw[...,::-1],lm['points'][i],meta['topology'],
                               pixel_aspect=pixel_aspect, found=bool(lm['found'][i]))
        record = {'frame':i,'source_frame':shot.source_start_frame+i,'eyes':{}}
        tiles = []
        for side, value in result.items():
            details = {'mode':value['mode'],'reason':value['reason'], 'geometry_validated':False,
                       'correspondence_validated':False, 'scene_reflection_allowed':False}
            if 'ob' in value:
                ob, crop = value['ob'], value['crop']
                np.savez_compressed(out/('%06d_%s.npz'%(i,side)),linear_rgb=crop['linear_rgb'],
                    material_candidate=ob.material,reflection_candidate=ob.reflection,
                    excluded=ob.excluded,origin_xy=crop['origin'])
                details.update(candidate_ok=ob.candidate_ok, geometry_roi=asdict(ob.geometry),
                    origin_xy=crop['origin'].tolist(), metrics=ob.metrics,
                    seed_roi=crop['seed'].tolist(), radius=crop['radius'],
                    eye_rotation=crop['eye_rotation'],
                    changed_channels=value['fallback']['changed_channels'])
                if args.review:
                    tiles.append(np.concatenate([review_tile(crop['linear_rgb'],side+' source keyed'),
                        review_tile(value['output'],side+' source fallback')],1))
            elif args.review:
                black=np.zeros((32,32,3),np.float32)
                tiles.append(np.concatenate([review_tile(black,side+' no usable ROI'),
                                             review_tile(black,side+' source preserved')],1))
            record['eyes'][side]=details
        rows.append(record)
        if args.review:
            directory=shot.composite/'iris_observations_review'
            directory.mkdir(parents=True,exist_ok=True)
            im=np.concatenate(tiles,0)
            if not cv2.imwrite(str(directory/('%06d.png'%i)),im):
                raise RuntimeError('Review write failed')
        if i%25==0:
            print('observations',i,'/',shot.frame_count,flush=True)
    manifest = {'schema':1,'shot':shot.name,'frame_rate':str(shot.frame_rate),
        'working_space':'scene-linear Rec.709 keyed input; display conversion only for native detector/review',
        'pixel_aspect':pixel_aspect, 'status':'experimental observations; no automatic repair authorized',
        'scope':'single tracked performer; dynamic native ROIs; any iris colour; pupil/limbus model remains near-circular after pixel-aspect correction',
        'frames':rows}
    shot.logs.mkdir(parents=True,exist_ok=True)
    store_manifest(shot,manifest)
    print('Wrote',len(rows),'frames of observations; source-keyed fallback throughout.',flush=True)


if __name__=='__main__':
    main()
