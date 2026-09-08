"""Reporter-only frozen neural correspondence experiment; never production repair."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from shot_paths import add_shot_argument, from_args

ROOT = Path(__file__).resolve().parents[1]
FRESH = [959, 977, 995]
SIDES = ('left', 'right')


def load_rows(shot, ids):
    cache = shot.composite / 'r10_iris'
    lm = np.load(cache / 'source_landmarks.npz')
    topology = json.loads((cache / 'source_tracking.json').read_text())['topology']
    rows = {side: {} for side in SIDES}
    for i in ids:
        raw = cv2.imread(str(cache / 'source_crops' / f'{i:06d}.png'))
        if raw is None:
            raise FileNotFoundError(f'Missing source crop {i}')
        for side in SIDES:
            pts = lm['points'][i, topology['iris_' + side], :2] - lm['origins'][i]
            origin = np.clip(np.rint(pts[0] - 32).astype(int), 0, np.array(raw.shape[1::-1]) - 64)
            rgb = raw[origin[1]:origin[1]+64, origin[0]:origin[0]+64, ::-1].astype(np.float32)
            rows[side][i] = dict(frame=i, rgb=rgb, seed=pts[0]-origin,
                                radius=float(np.linalg.norm(pts[1:]-pts[0], axis=1).mean()),
                                track_found=bool(lm['found'][i]))
    return rows


def write(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f'Image write failed: {path}')


def boards(shot):
    rows = load_rows(shot, FRESH)
    for i in FRESH:
        panels = []
        for side in SIDES:
            im = cv2.resize(rows[side][i]['rgb'][..., ::-1].astype(np.uint8), (640, 640),
                            interpolation=cv2.INTER_NEAREST)
            for c in range(0, 64, 4):
                cv2.line(im, (c*10, 0), (c*10, 639), (95, 95, 95), 1)
                cv2.line(im, (0, c*10), (639, c*10), (95, 95, 95), 1)
                cv2.putText(im, str(c), (c*10+1, 12), 0, .3, (255,255,255), 1)
                cv2.putText(im, str(c), (1, c*10+12), 0, .3, (255,255,255), 1)
            im = cv2.copyMakeBorder(im, 28, 0, 0, 0, cv2.BORDER_CONSTANT)
            cv2.putText(im, f'{side} SOURCE {i}; native xy; no predictions', (5,19), 0, .45, (255,255,255), 1)
            panels.append(im)
        write(shot.composite / 'r14_neural' / f'annotation_{i}.png', np.concatenate(panels, 1))


def score(est, truth, subset):
    if not subset.any():
        return None
    error = np.abs(est['rgb'][subset] - truth[subset])
    return dict(pixels=int(subset.sum()), display_mae=float(error.mean()),
                relative_rgb_error=float(error.sum()/max(truth[subset].sum(), 1e-8)))


def tile(rgb, title, subtitle):
    im = cv2.resize(np.clip(rgb[..., ::-1], 0, 255).astype(np.uint8), (320,320),
                    interpolation=cv2.INTER_NEAREST)
    im = cv2.copyMakeBorder(im, 48, 0, 0, 0, cv2.BORDER_CONSTANT)
    cv2.putText(im, title, (4,18), 0, .43, (255,255,255), 1)
    cv2.putText(im, subtitle, (4,39), 0, .38, (255,255,255), 1)
    return im


def render_review(shot, rows, target_ids):
    from temporal_iris import apply_eye_repair
    out = shot.composite / 'r14_neural'
    for k,i in enumerate(target_ids):
        panels = []
        for side in SIDES:
            row = rows[side][i]
            rgb, ob = row['rgb'], row['ob']
            overlay = rgb.copy()
            overlay[ob.material] = overlay[ob.material]*.5 + np.array([0,255,0])*.5
            result, fallback = apply_eye_repair(rgb,np.ones((64,64),bool),geometry_reliable=False)
            assert fallback['changed_channels'] == 0 and np.array_equal(result,rgb)
            status = 'CANDIDATE ONLY - NOT VALIDATED'
            if not ob.candidate_ok:
                status = ('REJECTED: too little iris / pupil'
                          if any('insufficient' in r for r in ob.reasons)
                          else 'REJECTED: uncertain eye geometry')
            panels.append(np.concatenate([tile(rgb,f'{side} SOURCE {i}','original source reflection'),
                tile(overlay,'NEURAL IRIS PROPOSAL',status),
                tile(result,'EXACT SOURCE FALLBACK','NO REFLECTION REPAIR')],1))
        board = np.concatenate(panels,0)
        write(out/'review'/f'{k:06d}.png',board)
        if i in FRESH:
            write(out/f'overlay_{i}.png',board)


def evaluate(shot):
    from ellseg_iris import EllSeg
    from ritnet_iris import RITnet
    from neural_iris_correspondence import observe, predict

    protocol = ROOT / 'docs/r14-evaluation-protocol.md'
    label_path = ROOT / 'scripts/data/r14_native_labels.json'
    labels = json.loads(label_path.read_text())
    donor_ids = [i for i in range(0,700,4) if all(abs(i-j)>8 for j in (79,217,521))]
    target_ids = list(range(950,1000))
    assert not set(donor_ids) & set(target_ids)
    rows = load_rows(shot, sorted(set(donor_ids+target_ids)))
    em, rm = EllSeg(), RITnet()
    out = shot.composite / 'r14_neural'
    out.mkdir(parents=True, exist_ok=True)
    observations, arrays = {}, {}
    for side in SIDES:
        observations[side] = {}
        for i, row in rows[side].items():
            ob = observe(row['rgb'], em, rm, track_found=row['track_found'])
            row['ob'] = ob
            observations[side][i] = dict(candidate_ok=ob.candidate_ok, reasons=ob.reasons, metrics=ob.metrics)
            for name, mask in [('material', ob.material), ('visible', ob.visible_labels), ('complete', ob.complete_labels)]:
                arrays[f'{i:06d}_{side}_{name}'] = mask
            if i % 100 == 0:
                print('neural proposals',side,i,flush=True)
    np.savez_compressed(out / 'proposals.npz', **arrays)
    semantic = []
    for lab in labels['frames']:
        ob = rows[lab['side']][lab['frame']]['ob']
        r = dict(frame=lab['frame'], side=lab['side'], view=lab['view'],
                 candidate_ok=ob.candidate_ok, reasons=ob.reasons,
                 pupil_crossing=lab['pupil_crossing'])
        for name in ('iris','noniris','reflection'):
            r[name+'_points'] = len(lab[name])
            for method, mask in [('ritnet',ob.visible_labels == 2), ('combined',ob.material)]:
                r[f'{method}_{name}_admitted'] = sum(int(mask[y,x]) for x,y in lab[name])
        semantic.append(r)
    cases, invariance, reflection_probes, labelled_iris = [], [], [], []
    yy, xx = np.mgrid[:64,:64]
    for side in SIDES:
        donors = [rows[side][i] for i in donor_ids]
        for i in target_ids[::2]:
            row = rows[side][i]
            for degrees in (60,120):
                theta = np.deg2rad(degrees)
                centre = np.rint(row['seed']+.7*row['radius']*np.array([np.cos(theta),np.sin(theta)])).astype(int)
                hidden = (xx-centre[0])**2+(yy-centre[1])**2 <= 1
                result = predict(row, donors, hidden, em, rm, verify_footprints=True)
                common = result['fixed'][0]['supported'] & result['torsion'][0]['supported']
                record = dict(side=side, frame=i, degrees=degrees, hole_xy=centre.tolist(),
                    hidden_pixels=int(hidden.sum()),
                    original_proposal_contains_hole=bool(row['ob'].material[hidden].all()))
                for method, (est, info) in result.items():
                    gate = est['agreement'] & info['context_pass']
                    record[method] = dict(all=score(est,row['rgb'][hidden],est['supported']),
                        paired=score(est,row['rgb'][hidden],common),
                        gated=score(est,row['rgb'][hidden],gate), info=info)
                # Recompute the complete path, including shortlist and ranking, for
                # EVERY hole. Fixed upstream face-track/crop provenance is explicit.
                for colour in ([0,0,0],[255,0,255]):
                    changed = dict(row, rgb=row['rgb'].copy())
                    changed['rgb'][hidden] = colour
                    test = predict(changed, donors, hidden, em, rm)
                    for method in result:
                        for key in result[method][0]:
                            assert np.array_equal(test[method][0][key],result[method][0][key],equal_nan=True), (i,side,method,key)
                        assert test[method][1] == result[method][1]
                invariance.append(dict(frame=i, side=side, degrees=degrees,
                    supported_pixels=int(result['torsion'][0]['supported'].sum()),
                    fitted_donors=len(result['torsion'][1]['fits']), exact=True))
                cases.append(record)
            print('correspondence',side,i,'cases',len(cases),flush=True)
        for lab in [r for r in labels['frames'] if r['side'] == side]:
            # Supplementary analysis, added after the primary result showed that
            # many fixed holes straddled the original candidate mask. Labels were
            # frozen before model inspection; thresholds/donors remain unchanged.
            # This is explicitly post-hoc analysis, not the preregistered test.
            iris_hidden = np.zeros((64,64),bool)
            for x,y in lab['iris']:
                iris_hidden[y,x] = True
            target = rows[side][lab['frame']]
            iris_result = predict(target,donors,iris_hidden,em,rm,verify_footprints=True)
            paired = iris_result['fixed'][0]['supported'] & iris_result['torsion'][0]['supported']
            iris_record = dict(frame=lab['frame'],side=side,hidden_pixels=int(iris_hidden.sum()))
            for method,(est,info) in iris_result.items():
                iris_record[method] = dict(all=score(est,target['rgb'][iris_hidden],est['supported']),
                    paired=score(est,target['rgb'][iris_hidden],paired),
                    gated=score(est,target['rgb'][iris_hidden],est['agreement'] & info['context_pass']), info=info)
            for colour in ([0,0,0],[255,0,255]):
                changed = dict(target,rgb=target['rgb'].copy())
                changed['rgb'][iris_hidden] = colour
                test = predict(changed,donors,iris_hidden,em,rm)
                for method in iris_result:
                    assert test[method][1] == iris_result[method][1]
                    for key in iris_result[method][0]:
                        assert np.array_equal(test[method][0][key],iris_result[method][0][key],equal_nan=True)
            iris_record['hidden_rgb_invariance'] = True
            labelled_iris.append(iris_record)
            hidden = np.zeros((64,64),bool)
            for x,y in lab['reflection']:
                hidden[y,x] = True
            result = predict(rows[side][lab['frame']],donors,hidden,em,rm,verify_footprints=True)
            reflection_probes.append(dict(frame=lab['frame'],side=side,
                sampled_points=int(hidden.sum()), pupil_crossing=lab['pupil_crossing'],
                repair_allowed=False, reason='unknown_underlying_material_and_incomplete_reflection_mask',
                methods={m:dict(supported_pixels=int(est['supported'].sum()),
                    agreement_pixels=int(est['agreement'].sum()),info=info) for m,(est,info) in result.items()}))
    summaries = {}
    for method in ('fixed','torsion'):
        summaries[method] = {}
        for subset in ('all','paired','gated'):
            selected = [r[method][subset] for r in cases if r[method][subset] is not None]
            summaries[method][subset] = dict(cases=len(selected), pixels=sum(r['pixels'] for r in selected),
                display_mae_p50_p95=np.percentile([r['display_mae'] for r in selected],[50,95]).tolist() if selected else None,
                relative_rgb_error_p50_p95=np.percentile([r['relative_rgb_error'] for r in selected],[50,95]).tolist() if selected else None)
    render_review(shot, rows, target_ids)
    supported_cases = [r for r in cases if r['fixed']['all'] or r['torsion']['all']]
    contained_cases = sum(r['original_proposal_contains_hole'] for r in supported_cases)
    report = dict(status='diagnostic only; no real reflection repairs earned',
        protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
        annotations_sha256=hashlib.sha256(label_path.read_bytes()).hexdigest(),
        annotation_provenance=labels['provenance'], model_provenance=dict(ellseg=em.provenance,ritnet=rm.provenance),
        donor_ids=donor_ids,target_ids=target_ids,observations=observations,
        semantic_scores=semantic,attempted_holes=len(cases),attempted_pixels=sum(r['hidden_pixels'] for r in cases),
        summary=summaries,cases=cases,hidden_rgb_invariance=invariance,reflection_probes=reflection_probes,
        supplementary_labelled_iris=dict(
            status='Post-hoc analysis of all 30 iris points whose source-only labels were frozen before model inspection; unchanged thresholds and donors.',
            motivation=f'Only {contained_cases} of {len(supported_cases)} supported primary holes lay wholly inside the original neural candidate mask; primary scores are not certified clean-iris accuracy.',
            cases=labelled_iris),
        limits=['Fresh labels and target interval on previously used reporter clip, not independent footage.',
            'Candidate donor masks are not independently clean or dense-visibility validated.',
            'Five-pixel iris holes do not establish complete glint removal; pupil/boundary material unknown.',
            'Hidden invariance covers both neural models and all correspondence; excludes upstream face tracking.',
            'Fixed experimental thresholds; no calibration probability or full photorealism claim.'],
        scene_reflection_allowed=False,production_repairs=0)
    (shot.logs/'r14_neural.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(summary=summaries,semantic_scores=semantic),indent=2),flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_shot_argument(ap)
    ap.add_argument('--boards', action='store_true')
    ap.add_argument('--review-only', action='store_true', help='Redraw cached diagnostics without rerunning inference')
    args = ap.parse_args()
    shot = from_args(args)
    if shot.name != 'reporter_a' or shot.video_dir.name != 'reporter_test':
        raise SystemExit('Reporter-only evaluation cache; reusable API is neural_iris_correspondence.py')
    cv2.setNumThreads(1)
    if args.boards:
        boards(shot)
    elif args.review_only:
        from types import SimpleNamespace
        report = json.loads((shot.logs/'r14_neural.json').read_text())
        rows = load_rows(shot, report['target_ids'])
        arrays = np.load(shot.composite/'r14_neural/proposals.npz')
        for side in SIDES:
            for i,row in rows[side].items():
                info = report['observations'][side][str(i)]
                row['ob'] = SimpleNamespace(material=arrays[f'{i:06d}_{side}_material'], **info)
        render_review(shot, rows, report['target_ids'])
    else:
        evaluate(shot)


if __name__ == '__main__':
    main()
