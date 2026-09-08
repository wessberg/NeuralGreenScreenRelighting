"""Real reporter native-boundary/correspondence audit; isolated from production.

Uses round 10's decoded native crops and coarse landmark seed/scale ONLY. Never
uses its aperture as a semantic mask. Run without --shot to list available shots.
"""
import argparse
from dataclasses import asdict
import json
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

import cv2
import numpy as np
from native_iris import Observation, detect
from shot_paths import add_shot_argument, from_args
from temporal_iris import IrisGeometry


def write(path, im):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), im):
        raise RuntimeError(str(path))


def load(shot, stride, cached=False):
    cache = shot.composite / 'r10_iris'
    lm = np.load(cache / 'source_landmarks.npz')
    topo = json.loads((cache/'source_tracking.json').read_text())['topology']
    data = {s: [] for s in ('left', 'right')}
    prior = json.loads((shot.composite/'r12_native'/'observations_detail.json').read_text()) if cached else None
    masks = dict(np.load(shot.composite/'r12_native'/'observations.npz')) if cached else None
    for i in range(0, len(lm['found']), stride):
        raw = cv2.imread(str(cache/'source_crops'/('%06d.png' % i)))
        if raw is None:
            raise RuntimeError('Missing native crop %d' % i)
        for side in data:
            ids = topo['iris_'+side]
            points = lm['points'][i, ids, :2]-lm['origins'][i]
            centre = points[0]
            radius = float(np.linalg.norm(points[1:]-centre, axis=1).mean())
            origin = np.clip(np.rint(centre-32).astype(int), 0,
                             np.array(raw.shape[1::-1])-64)
            rgb = raw[origin[1]:origin[1]+64, origin[0]:origin[0]+64, ::-1].astype(np.float32)
            if rgb.shape != (64, 64, 3):
                raise RuntimeError('Eye crop outside cached ROI')
            seed = centre-origin
            if cached:
                p = prior['sides'][side][i]
                ob = Observation(IrisGeometry(**p['geometry']), masks[side+'_material'][i],
                    masks[side+'_excluded'][i], masks[side+'_reflection'][i],
                    p['candidate_ok'], p['reasons'], p['metrics'])
            else:
                ob = detect(rgb, seed, radius)
            data[side].append(dict(frame=i, rgb=rgb, seed=seed, radius=radius,
                origin=origin+lm['origins'][i], ob=ob))
        if i % 100 == 0:
            print('native boundaries', i, flush=True)
    return data


def annotation_board(shot, data):
    # Held-out semantic review samples chosen independently of detector acceptance.
    ids = [79, 217, 521, 779, 923, 1115]
    for i in ids:
        tiles = []
        for side in data:
            raw = data[side][i]['rgb'][..., ::-1].astype(np.uint8)
            im = cv2.resize(raw, (640, 640), interpolation=cv2.INTER_NEAREST)
            for c in range(0, 64, 4):
                cv2.line(im, (c*10, 0), (c*10, 639), (90, 90, 90), 1)
                cv2.line(im, (0, c*10), (639, c*10), (90, 90, 90), 1)
                cv2.putText(im, str(c), (c*10+1, 12), 0, .3, (255,255,255), 1)
                cv2.putText(im, str(c), (1, c*10+12), 0, .3, (255,255,255), 1)
            im = cv2.copyMakeBorder(im, 24, 0, 0, 0, cv2.BORDER_CONSTANT)
            cv2.putText(im, '%s source %d; NATIVE xy grid; no fitted overlay' % (side,i),
                        (5, 17), 0, .4, (255,255,255), 1)
            tiles.append(im)
        write(shot.composite/'r12_native'/('annotation_%04d.png'%i), np.concatenate(tiles,1))


def measure(estimate, truth, subset):
    if not subset.any():
        return None
    error = np.abs(estimate['rgb'][subset]-truth[subset])
    return {'pixels': int(subset.sum()), 'display_mae': float(error.mean()),
            'relative_rgb_error': float(error.sum()/max(truth[subset].sum(), 1e-8))}


def store_audit(shot, report, pilot=False):
    if pilot:
        (shot.logs/'r12_pilot.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        return
    (shot.composite/'r12_native'/'observations_detail.json').write_text(json.dumps(report),encoding='utf-8')
    summary={k:v for k,v in report.items() if k!='sides'}
    summary['detail']='composite/r12_native/observations_detail.json'
    summary['sides']={s:{'frames':len(rows), 'candidate_frames':sum(r['candidate_ok'] for r in rows),
        'pupil_crossing_candidate_masks':sum(r['metrics']['reflection_crosses_pupil'] for r in rows),
        'candidate_source_frame_ids':[r['frame'] for r in rows if r['candidate_ok']],
        'rejection_counts':{reason:sum(reason in r['reasons'] for r in rows)
                            for reason in sorted({q for r in rows for q in r['reasons']})}}
        for s,rows in report['sides'].items()}
    (shot.logs/'r12_native_audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')


def evaluate(shot, data):
    from iris_correspondence import predict
    records, invariance, before_after = [], [], []
    for side, rows in data.items():
        for i in range(shot.source_start_frame, shot.source_start_frame+shot.frame_count):
            row = rows[i]
            if not row['ob'].candidate_ok:
                continue
            for angle in [np.pi/3, 2*np.pi/3]:
                centre = row['seed']+row['radius']*.7*np.array([np.cos(angle), np.sin(angle)])
                yy, xx = np.mgrid[:64, :64]
                hidden = (xx-round(centre[0]))**2+(yy-round(centre[1]))**2 <= 1
                if not row['ob'].material[hidden].all():
                    continue
                results = {m: predict(rows, i, hidden, rotation=rotate)
                           for m, rotate in [('fixed', False), ('torsion', True)]}
                paired = results['fixed'][0]['supported'] & results['torsion'][0]['supported']
                truth = row['rgb'][hidden]
                record = {'side': side, 'source_frame': i, 'hidden_native_pixels': int(hidden.sum()),
                          'centre_native_xy': np.rint(centre).astype(int).tolist()}
                for method, (est, info) in results.items():
                    cert = est['agreement'] & info['context_pass']
                    record[method] = {'all': measure(est, truth, est['supported']),
                        'paired': measure(est, truth, paired),
                        'context_and_agreement': measure(est, truth, cert),
                        'info': {**info, 'geometry': asdict(info['geometry'])}}
                records.append(record)
                # Positive-support examples as well as abstentions; every upstream native
                # detector read is rerun after two extreme target-hole perturbations.
                if len(invariance) < 24:
                    original = row['rgb']
                    raw_original = detect(original, row['seed'], row['radius'])
                    raw_deltas = []
                    for colour in ([255., 0., 255.], [0., 0., 0.]):
                        altered = original.copy()
                        altered[hidden] = colour
                        row['rgb'] = altered
                        test, info = predict(rows, i, hidden, rotation=True)
                        ref, ri = results['torsion']
                        same = all(np.array_equal(test[k], ref[k]) for k in ref)
                        same &= info == ri
                        if not same:
                            raise AssertionError('Hidden native RGB leaked into detection/correspondence')
                        raw = detect(altered, row['seed'], row['radius'])
                        raw_deltas.append(float(np.linalg.norm(np.array(raw.geometry.centre)-raw_original.geometry.centre)))
                    row['rgb'] = original
                    invariance.append({'frame': i, 'side': side, 'passed': True,
                        'supported_pixels': int(results['torsion'][0]['supported'].sum()),
                        'unguarded_detector_centre_change_px': raw_deltas})
                shown = []
                for title, im in [('Source: hidden-pixel truth', row['rgb']),
                        ('Fixed coordinates: held-out pixels only', row['rgb'].copy()),
                        ('Fitted torsion: held-out pixels only', row['rgb'].copy())]:
                    shown.append((title, im))
                for j, method in enumerate(['fixed', 'torsion'], 1):
                    est, _ = results[method]
                    replacement = shown[j][1][hidden]
                    replacement[est['supported']] = est['rgb'][est['supported']]
                    replacement[~est['supported']] = [150, 0, 150]
                    shown[j][1][hidden] = replacement
                tiles = []
                for title, im in shown:
                    tile = cv2.resize(np.clip(im[..., ::-1],0,255).astype(np.uint8),
                                      (384,384), interpolation=cv2.INTER_NEAREST)
                    tile = cv2.copyMakeBorder(tile, 52,0,0,0,cv2.BORDER_CONSTANT)
                    cv2.putText(tile,title,(5,18),0,.39,(255,255,255),1)
                    cv2.putText(tile,'%s source %d; no reflection repair'%(side,i),(5,40),0,.38,(255,255,255),1)
                    tiles.append(tile)
                before_after.append(np.concatenate(tiles,1))
            print('held-out',side,i,'cases',len(records),flush=True)
    summary = {}
    for method in ['fixed', 'torsion']:
        summary[method] = {}
        for subset in ['all','paired','context_and_agreement']:
            cases = [r[method][subset] for r in records if r[method][subset]]
            summary[method][subset] = {'cases':len(cases), 'pixels':sum(c['pixels'] for c in cases),
                'display_mae_p50_p95': np.percentile([c['display_mae'] for c in cases],[50,95]).tolist() if cases else None,
                'relative_rgb_error_p50_p95':np.percentile([c['relative_rgb_error'] for c in cases],[50,95]).tolist() if cases else None}
    for k, im in enumerate(before_after):
        write(shot.composite/'r12_native'/'holdout'/('%06d.png'%k),im)
    report = {'status':'real held-out observations, NOT a reflection repair',
        'selection':'every selected source frame; 5 native pixels per hole; coarse seed at 60/120 degrees, 0.7 initial radius; conditional on preliminary material mask',
        'limits':'Coarse ROI/landmark seed fixed. Invariance includes native detector and correspondence, not upstream face tracking. Sparse 5-pixel holes do not validate large glint footprints or their unobserved material.',
        'summary':summary, 'hidden_rgb_invariance':invariance, 'cases':records}
    (shot.logs/'r12_correspondence.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


def panel(row):
    ob, rgb = row['ob'], row['rgb']
    raw = np.clip(rgb[..., ::-1], 0, 255).astype(np.uint8)
    overlay = raw.copy()
    overlay[ob.material] = (.5*overlay[ob.material]+.5*np.array([0, 255, 0])).astype(np.uint8)
    overlay[ob.excluded] = (.4*overlay[ob.excluded]+.6*np.array([0, 0, 255])).astype(np.uint8)
    for key, colour in [('limbus', (255, 255, 0)), ('pupil', (0, 255, 255))]:
        x, y, r = ob.metrics[key]
        cv2.circle(overlay, (round(x), round(y)), round(r), colour, 1)
    tiles = [cv2.resize(im, (320, 320), interpolation=cv2.INTER_NEAREST) for im in (raw, overlay)]
    result = np.concatenate(tiles, 1)
    result = cv2.copyMakeBorder(result, 48, 0, 0, 0, cv2.BORDER_CONSTANT)
    lines = ['source %04d  raw | native candidate masks' % row['frame'],
             'candidate OK (not certified)' if ob.candidate_ok else ', '.join(ob.reasons)]
    for i, line in enumerate(lines):
        cv2.putText(result, line, (5, 19+i*21), 0, .43, (255, 255, 255), 1)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_shot_argument(ap)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--pilot', action='store_true')
    ap.add_argument('--cached', action='store_true')
    ap.add_argument('--evaluate', action='store_true')
    args = ap.parse_args()
    shot = from_args(args)
    if shot.name != 'reporter_a' or shot.video_dir.name != 'reporter_test':
        raise SystemExit('This cache adapter is reporter-only; native_iris is generic')
    if args.stride < 1:
        raise SystemExit('stride must be positive')
    if args.stride != 1 and not args.pilot:
        raise SystemExit('Use --pilot for subsampled surveys; full evidence requires stride 1')
    if args.evaluate and (args.pilot or args.stride != 1):
        raise SystemExit('Correspondence evaluation requires the full native observation sequence')
    cv2.setNumThreads(1)
    data = load(shot, args.stride, args.cached)
    out = shot.composite/'r12_native'
    if not args.cached:
        for k in range(len(data['left'])):
            combined = np.concatenate([panel(data[s][k]) for s in data], 0)
            write(out/('pilot' if args.pilot else 'masks')/('%06d.png'%k), combined)
    report = {'status': 'experimental native boundaries; no reflection repair',
        'stride': args.stride, 'sides': {s: [dict(frame=r['frame'],
            geometry=asdict(r['ob'].geometry), candidate_ok=r['ob'].candidate_ok,
            reasons=r['ob'].reasons, metrics=r['ob'].metrics) for r in rows]
            for s, rows in data.items()}}
    store_audit(shot, report, args.pilot)
    if not args.pilot and not args.cached:
        np.savez_compressed(out/'observations.npz', **{s+'_'+name: np.stack([
            getattr(r['ob'], name) for r in rows]) for s, rows in data.items()
            for name in ['material', 'excluded', 'reflection']})
        annotation_board(shot, data)
    if args.evaluate:
        evaluate(shot, data)
    print(json.dumps({s: {'frames': len(rows), 'candidates': sum(r['ob'].candidate_ok for r in rows),
        'pupil_crossing': sum(r['ob'].metrics['reflection_crosses_pupil'] for r in rows)}
        for s, rows in data.items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()
