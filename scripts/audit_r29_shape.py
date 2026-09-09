"""Seen-source optical/light-shape hypotheses with reflection-balanced frame folds."""
import argparse
import json
import numpy as np
from audit_r24_source import ROOT, sha, write, polygon
from audit_r26_joint import get_eye
from audit_r27_photometry import read_rgb, error, FRAMES
from partial_eye_geometry import conservative_open
from eye_radiance_fit import angular_support
from eye_shape_fit import transport, vectors, design, fit, occupied
from shot_paths import add_shot_argument, from_args

ARMS = ('appearance', 'broad', 'narrow', 'spatial_backing')


def freeze(shot):
    paths = [ROOT/'docs/r29-eye-protocol.md', shot.dir/'shot.json',
             shot.logs/'r28_radiance.json', shot.logs/'r28_verification.json']
    for folder in ('r27_photometry', 'r28_radiance'):
        paths += sorted((shot.composite/folder).glob('*.npz'))
        paths += sorted((shot.composite/folder).glob('*.json'))
    paths += [shot.relight/f'neural_iris_observations/{f:06d}_{s}.npz'
              for s in ('left', 'right') for f in FRAMES]
    record = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    path = shot.logs/'r29_freeze.json'
    if path.exists():
        assert json.loads(path.read_text()) == record
    else:
        write(path, record)
    return record


def prepare(shot, side, focal):
    root = shot.composite/'r27_photometry'
    meta = json.loads((root/f'geometry_{side}.json').read_text())
    obs, profiles = meta['observations'], meta['profiles']
    with np.load(root/f'geometry_{side}.npz') as z:
        table = z['table']
    configs = [(g, n) for g in range(len(profiles)) for n in (3, 7)]
    trans, xy, labels = {}, {}, {}
    for j, o in enumerate(obs):
        _, safe = conservative_open(o['shape'], o['label']['lid'], [], 2.)
        y, x = np.nonzero(safe)
        q = np.stack([x, y], 1)
        ts = [transport(q, *get_eye(obs, table, profiles[g]['best_rows'][j], focal),
                        focal, o['centre'], n) for g, n in configs]
        common = np.all([t['valid'] for t in ts], axis=0)
        xy[j] = q[common]
        labels[j] = polygon(o['shape'], o['label']['reflection'])[y, x][common]
        trans[j] = [{k: v[common] for k, v in t.items()} for t in ts]
    # Common masks are constructed without any RGB; both backing arms participate.
    masks, support = {}, {}
    for target in range(6):
        train = [j for j in range(4) if j != target]
        a = np.ones(len(xy[target]), bool)
        ang = a.copy()
        for c in range(len(configs)):
            ts = [trans[j][c] for j in train]
            qt = trans[target][c]
            ang &= angular_support([t['directions'] for t in ts], qt['directions'])[0]
            for spatial in (False, True):
                v = vectors()
                dx, m = design(qt, v, spatial)
                a &= occupied([design(t, v, spatial)[0] for t in ts], dx, m)
        masks[target] = a & ang
        support[target] = dict(candidate=len(a), reflection_candidate=int(labels[target].sum()),
            appearance=int(a.sum()), angular=int(ang.sum()), joint=int((a & ang).sum()),
            reflection_joint=int((a & ang & labels[target]).sum()),
            other_joint=int((a & ang & ~labels[target]).sum()))
    return dict(obs=obs, configs=configs, trans=trans, xy=xy, labels=labels,
                masks=masks, support=support)


def training_values(shot, side, p, indices):
    values = {}
    for j in indices:
        rgb = read_rgb(shot, side, p['obs'][j]['frame'])
        xy = p['xy'][j]
        values[j] = rgb[xy[:, 1], xy[:, 0]]
    return values


def model(p, c, arm, train, truth):
    ts = [p['trans'][j][c] for j in train]
    v = vectors([t['directions'][:, t['directions'].shape[1]//2] for t in ts]
                if arm in ('narrow', 'spatial_backing') else None)
    matrices = [design(p['trans'][j][c], v, arm == 'spatial_backing')[0] for j in range(6)]
    m = 578 if arm == 'spatial_backing' else 290
    x = np.concatenate([matrices[j] for j in train])
    y = np.concatenate([truth[j] for j in train])
    co = fit(x, y, m, v, arm != 'appearance')
    return matrices, m, v, co


def predict(shot, side, p, out):
    truth = training_values(shot, side, p, range(4))
    eligible = [j for j in range(4) if p['support'][j]['reflection_joint'] >= 8
                and p['support'][j]['other_joint'] >= 8]
    records, arrays = [], {}
    for c, (g, n) in enumerate(p['configs']):
        for arm in ARMS:
            index = len(records)
            folds = []
            for held in range(4):
                train = [j for j in range(4) if j != held]
                x, m, v, co = model(p, c, arm, train, {j: truth[j] for j in train})
                pred = x[held]@co
                mask, label = p['masks'][held], p['labels'][held]
                r = error(pred[mask & label], truth[held][mask & label])
                o = error(pred[mask & ~label], truth[held][mask & ~label])
                balanced = float((np.mean((pred[mask & label]-truth[held][mask & label])**2)
                    + np.mean((pred[mask & ~label]-truth[held][mask & ~label])**2))/2) if held in eligible else None
                folds.append(dict(held_source=p['obs'][held]['frame']+700,
                    reflection=r, other=o, balanced_mse=balanced))
            x, m, v, co = model(p, c, arm, list(range(4)), truth)
            arrays[f'co_{index}'], arrays[f'vectors_{index}'] = co, v
            for j in range(6):
                arrays[f'layers_{index}_{j}'] = np.stack([x[j][:, :m]@co[:m], x[j][:, m:]@co[m:]])
            trainpred = np.concatenate([x[j]@co for j in range(4)])
            y = np.concatenate([truth[j] for j in range(4)])
            label = np.concatenate([p['labels'][j] for j in range(4)])
            records.append(dict(index=index, config=c, geometry=g, box_n=n, arm=arm,
                appearance_count=m, light_count=len(v), folds=folds,
                selection_score=float(np.mean([folds[j]['balanced_mse'] for j in eligible])) if len(eligible) >= 2 else None,
                train_all=error(trainpred, y), train_reflection=error(trainpred[label], y[label]),
                max_prediction=float(max(arrays[f'layers_{index}_{j}'].sum(0).max() for j in range(6))),
                light_rms=np.sqrt(np.einsum('ic,ij,jc->c', co[m:], __import__('eye_shape_fit').gram(v), co[m:])).tolist()))
            print('R29_MODEL', side, index, g, n, arm, records[-1]['selection_score'], flush=True)
    ranked = sorted(range(len(records)), key=lambda i: (records[i]['selection_score'], i)) if len(eligible) >= 2 else []
    choices = {arm: next((i for i in ranked if records[i]['arm'] == arm), None) for arm in ARMS}
    result = dict(records=records, eligible_folds=eligible, selected=ranked[0] if ranked else None,
                  choices=choices, support=p['support'])
    for j in range(6):
        arrays[f'xy_{j}'] = p['xy'][j]
        arrays[f'mask_{j}'] = p['masks'][j]
        arrays[f'label_{j}'] = p['labels'][j]
    np.savez_compressed(out/f'predictions_{side}.npz', **arrays)
    write(out/f'selection_{side}.json', result)
    return result


def score(shot, side, p, out, result):
    truth = training_values(shot, side, p, (4, 5))
    scores = {}
    with np.load(out/f'predictions_{side}.npz') as z:
        for j in (4, 5):
            mask, label = p['masks'][j], p['labels'][j]
            scores[j] = [dict(index=r['index'],
                candidate=error(z[f'layers_{r["index"]}_{j}'].sum(0), truth[j]),
                supported=error(z[f'layers_{r["index"]}_{j}'].sum(0)[mask], truth[j][mask]),
                reflection=error(z[f'layers_{r["index"]}_{j}'].sum(0)[mask & label], truth[j][mask & label]))
                for r in result['records']]
    return scores


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_shot_argument(ap)
    ap.add_argument('--preflight', action='store_true')
    args = ap.parse_args(); shot = from_args(args)
    evidence = freeze(shot)
    focal = np.array(json.loads((shot.logs/'r28_radiance.json').read_text())['focal'])
    out = shot.composite/'r29_shape'; out.mkdir(parents=True, exist_ok=True)
    prepared, result = {}, {}
    for side in ('left', 'right'):
        prepared[side] = prepare(shot, side, focal)
        print('R29_SUPPORT', side, json.dumps(prepared[side]['support']), flush=True)
    if args.preflight:
        write(shot.logs/'r29_preflight.json', {s: p['support'] for s, p in prepared.items()})
        return
    for side, p in prepared.items():
        result[side] = predict(shot, side, p, out)
    # Both eye predictions have been written before either seen diagnostic RGB read.
    for side, p in prepared.items():
        result[side]['seen_scores'] = score(shot, side, p, out, result[side])
    write(shot.logs/'r29_shape.json', dict(evidence=evidence, results=result, focal=focal.tolist(),
        artifact_hashes={str(f.relative_to(ROOT)): sha(f) for f in out.glob('*.*')},
        code_sha256={n: sha(ROOT/'scripts'/n) for n in ('eye_shape_fit.py', 'audit_r29_shape.py')},
        fresh_test=False, production_repairs=0, material_validated=False,
        scene_replacement=False, credible_3d_lids=False))
    print('R29_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
