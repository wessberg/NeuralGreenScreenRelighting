"""Reporter-texture synthetic motion audit. No real repair or scene replacement."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from audit_r14_neural import load_rows, SIDES, tile, write
from audit_r15_dense import plain
from dense_iris_flow import DenseEyeFlow, query_flow
from reflection_motion import sequence
from shot_paths import add_shot_argument, from_args
from temporal_eye_tracker import TemporalEyeTracker

ROOT = Path(__file__).resolve().parents[1]
EVAL = (0, 4, 8, 12, 16, 20, 23)
FILES = ('reflection_motion.py', 'audit_r18_motion.py', 'temporal_eye_tracker.py',
         'dense_iris_flow.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(plain(value), indent=2, allow_nan=False), encoding='utf-8')


def definitions():
    labels = json.loads((ROOT/'scripts/data/r14_native_labels.json').read_text())
    return [dict(name=f'{v["side"]}_{i}_{axis}', side=v['side'], index=i,
                 point=v['iris'][i], axis=direction)
            for v in labels['frames'] if v['frame'] == 977
            for i in (0, 3) for axis, direction in (('x', [1, 0]), ('y', [0, 1]))]


def infer(tracker, flow, video, truth, anchor):
    point = truth[anchor]
    hidden = np.zeros(video.shape[:3], bool)
    x, y = np.floor(point).astype(int)
    hidden[anchor, y:y+2, x:x+2] = True
    query = np.array([[anchor, *point]], np.float32)
    tracks = tracker.track(video, query, excluded=hidden)
    result = dict(**tracks, excluded=hidden, query=query)
    for frame in EVAL:
        if frame == anchor:
            continue
        forward, backward = flow.pair(video[anchor], video[frame],
                                     target_excluded=hidden[anchor])
        result[f'{frame}_forward'] = forward
        result[f'{frame}_backward'] = backward
    return result


def records(data, synth, anchor):
    result = []
    for frame in EVAL:
        if frame == anchor:
            continue
        raft, inside, cycle = query_flow(data[f'{frame}_forward'],
                                        data[f'{frame}_backward'], synth['material'][anchor:anchor+1])
        for arm, xy, accepted in (
                ('raft', raft[0], inside[0] and cycle[0] <= .5),
                ('cotracker', data['xy'][frame, 0], data['usable_visibility'][frame, 0])):
            material_error = float(np.linalg.norm(xy-synth['material'][frame]))
            reflection_error = float(np.linalg.norm(xy-synth['reflection'][frame]))
            separation = float(np.linalg.norm(synth['material'][frame]-synth['reflection'][frame]))
            result.append(dict(frame=frame, arm=arm, xy=xy, accepted=bool(accepted),
                material_epe=material_error, reflection_epe=reflection_error,
                separation=separation, cycle=float(cycle[0]) if arm == 'raft' else None,
                accepted_wrong=bool(accepted and material_error > 1),
                reflection_lock=bool(accepted and separation >= 3 and reflection_error <= 1 and material_error > 1)))
    return plain(result)


def summarize(cases):
    summary = {}
    for condition in ('base', 'contaminated'):
        for anchor in (0, 12):
            for arm in ('raft', 'cotracker'):
                rows = [r for c in cases if c['condition'] == condition and c['anchor'] == anchor
                        for r in c['records'] if r['arm'] == arm]
                accepted = [r for r in rows if r['accepted']]
                summary[f'{condition}_{anchor}_{arm}'] = dict(
                    attempts=len(rows), accepted=len(accepted), rejected=len(rows)-len(accepted),
                    median_material_epe=float(np.median([r['material_epe'] for r in rows])),
                    p95_material_epe=float(np.percentile([r['material_epe'] for r in rows], 95)),
                    accepted_median_material_epe=float(np.median([r['material_epe'] for r in accepted])) if accepted else None,
                    accepted_wrong=sum(r['accepted_wrong'] for r in rows),
                    separated_attempts=sum(r['separation'] >= 3 for r in rows),
                    reflection_lock=sum(r['reflection_lock'] for r in rows))
    return summary


def render_review(shot, report):
    out = shot.composite/'r18_motion'
    index = {(c['name'], c['condition'], c['anchor']): c for c in report['cases']}
    for seq_index, definition in enumerate(definitions()):
        name = definition['name']
        with np.load(out/f'{name}_sequence.npz') as synth:
            caches = {(condition, anchor): np.load(out/index[name, condition, anchor]['cache'])
                      for condition in ('base', 'contaminated') for anchor in (0, 12)}
            for frame in range(24):
                boards = []
                for anchor in (0, 12):
                    panels = []
                    for condition in ('base', 'contaminated'):
                        for arm in ('raft', 'cotracker'):
                            case = index[name, condition, anchor]
                            rec = next((r for r in case['records'] if r['frame'] == frame and r['arm'] == arm), None)
                            status = 'FORCED QUERY - NOT SCORED' if frame == anchor else 'not scored this frame'
                            predicted, accept = None, False
                            if rec:
                                predicted, accept = rec['xy'], rec['accepted']
                                status = f'{"accept" if accept else "reject"}; EPE {rec["material_epe"]:.2f}px'
                            elif arm == 'cotracker':
                                cache = caches[condition, anchor]
                                predicted, accept = cache['xy'][frame, 0], cache['usable_visibility'][frame, 0]
                                if frame != anchor:
                                    status = 'temporal track; outside paired score'
                            im = tile(synth[condition][frame], f'{condition} / {arm} / anchor {anchor}', status)
                            # Native pixel centres under nearest-neighbour x5 magnification.
                            def pos(xy):
                                return tuple(np.rint(np.asarray(xy)*5+2+[0, 48]).astype(int))
                            cv2.drawMarker(im, pos(synth['material'][frame]), (255, 255, 0), cv2.MARKER_CROSS, 15, 1)
                            if condition == 'contaminated':
                                cv2.drawMarker(im, pos(synth['reflection'][frame]), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 13, 1)
                            if predicted is not None and np.isfinite(predicted).all():
                                cv2.circle(im, pos(predicted), 6, (0, 255, 0) if accept else (0, 0, 255), 1)
                            panels.append(im)
                    boards.append(np.concatenate(panels, 1))
                board = np.concatenate(boards, 0)
                board = cv2.copyMakeBorder(board, 54, 0, 0, 0, cv2.BORDER_CONSTANT)
                cv2.putText(board, f'SYNTHETIC MOTION ONLY | {name} | t={frame:02d} | NO REPAIR / SCENE REPLACEMENT',
                            (8, 20), 0, .53, (255, 255, 255), 1)
                cv2.putText(board, 'Cyan +: known material | Magenta x: added reflection | Green/red circle: model accept/reject',
                            (8, 43), 0, .5, (255, 255, 255), 1)
                write(out/'review'/f'{seq_index*24+frame:06d}.png', board)
            for cache in caches.values():
                cache.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_shot_argument(ap)
    ap.add_argument('--review-only', action='store_true')
    args = ap.parse_args()
    shot = from_args(args)
    if shot.dir != (ROOT/'music_videos/reporter_test/shots/reporter_a').resolve():
        raise SystemExit('Reporter-only audit; reflection_motion.sequence accepts arbitrary observed crops')
    out = shot.composite/'r18_motion'
    out.mkdir(parents=True, exist_ok=True)
    report_path = shot.logs/'r18_motion.json'
    if args.review_only:
        render_review(shot, json.loads(report_path.read_text()))
        return
    rows = load_rows(shot, [977])
    protocol = sha(ROOT/'docs/r18-evaluation-protocol.md')
    code = {n: sha(ROOT/'scripts'/n) for n in FILES}
    tracker, flow = TemporalEyeTracker(), DenseEyeFlow()
    cases = []
    for definition in definitions():
        synth = sequence(rows[definition['side']][977]['rgb'], definition['point'], definition['axis'])
        np.savez_compressed(out/f'{definition["name"]}_sequence.npz', **synth)
        for condition in ('base', 'contaminated'):
            video = synth[condition]
            for anchor in (0, 12):
                name = f'{definition["name"]}_{condition}_{anchor}'
                data = infer(tracker, flow, video, synth['material'], anchor)
                hashes = {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in data.items()}
                substitutions = []
                for color in ([0, 0, 0], [255, 0, 255]):
                    altered = video.copy()
                    altered[data['excluded']] = color
                    repeated = infer(tracker, flow, altered, synth['material'], anchor)
                    assert {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in repeated.items()} == hashes, name
                    substitutions.append(dict(color=color, exact_arrays=len(hashes)))
                np.savez_compressed(out/f'{name}.npz', **data)
                cases.append(dict(**definition, condition=condition, anchor=anchor,
                    cache=f'{name}.npz', arrays_sha256=hashes, substitutions=substitutions,
                    clipped_channels=int(synth['clipped'].sum()) if condition == 'contaminated' else 0,
                    records=records(data, synth, anchor)))
                print('Verified', name, flush=True)
    assert code == {n: sha(ROOT/'scripts'/n) for n in FILES}
    assert protocol == sha(ROOT/'docs/r18-evaluation-protocol.md')
    report = dict(protocol_sha256=protocol, code_sha256=code,
        labels_sha256=sha(ROOT/'scripts/data/r14_native_labels.json'),
        source_crops_sha256={side: hashlib.sha256(rows[side][977]['rgb'].tobytes()).hexdigest() for side in SIDES},
        models=[tracker.provenance, flow.provenance], cases=cases, summary=summarize(cases),
        production_repairs=0, scene_reflection_allowed=False, photometric_scores_computed=0,
        scope='Synthetic additive reflection over observed texture; no real clean-material truth')
    dump(report_path, report)
    render_review(shot, report)
    print(json.dumps(report['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()
