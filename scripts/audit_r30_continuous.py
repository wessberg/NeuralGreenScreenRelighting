"""Seen-source continuous optical/light calibration with explicit backing competition."""
import argparse
import json
import time
import numpy as np
from threadpoolctl import threadpool_limits
from audit_r24_source import ROOT, sha, write
from audit_r26_joint import get_eye
from audit_r27_photometry import read_rgb, error
from eye_shape_fit import design, occupied
from eye_radiance_fit import angular_support
from eye_continuous_fit import ARMS, solve, predict, light_vectors
from shot_paths import add_shot_argument, from_args


def freeze(shot):
    paths = [ROOT/'docs/r30-eye-protocol.md', shot.dir/'shot.json',
             shot.logs/'r29_shape.json', shot.logs/'r29_verification.json']
    for folder in ('r27_photometry', 'r29_shape'):
        paths += sorted((shot.composite/folder).glob('*.npz'))
        paths += sorted((shot.composite/folder).glob('*.json'))
    paths += [shot.relight/f'neural_iris_observations/{f:06d}_{s}.npz'
              for s in ('left','right') for f in (30,60,90,120,105,115)]
    record = {str(p.relative_to(ROOT)):sha(p) for p in paths}
    path = shot.logs/'r30_freeze.json'
    if path.exists(): assert json.loads(path.read_text()) == record
    else: write(path,record)
    return record


def prepare(shot, side, focal):
    root = shot.composite/'r27_photometry'
    meta = json.loads((root/f'geometry_{side}.json').read_text())
    with np.load(root/f'geometry_{side}.npz') as z: table = z['table']
    observations = []
    with np.load(shot.composite/f'r29_shape/predictions_{side}.npz') as z:
        for j, o in enumerate(meta['observations']):
            eye, offset = get_eye(meta['observations'], table, meta['profiles'][0]['best_rows'][j], focal)
            observations.append(dict(o, eye=eye, offset=offset, xy=z[f'xy_{j}'],
                label=z[f'label_{j}'], prior_mask=z[f'mask_{j}']))
    return observations


def training_values(shot, side, observations, indices):
    result = {}
    for j in indices:
        o = observations[j]; xy = o['xy']; rgb = read_rgb(shot,side,o['frame'])
        result[j] = rgb[xy[:,1],xy[:,0]]
    return result


def support(predictions, train, target):
    q = predictions[target]; m = q['material_count']
    ts = [predictions[j] for j in train]
    matrices = [p['matrix'][p['transport']['valid']] for p in ts]
    directions = [p['transport']['directions'][p['transport']['valid']] for p in ts]
    return (q['transport']['valid'] & occupied(matrices,q['matrix'],m) &
            angular_support(directions,q['transport']['directions'])[0])


def fit_eye(shot, side, obs, focal, out):
    records = []; arrays = {}
    # Access only the four fitting frames. Each solver receives just its donors.
    truth = training_values(shot,side,obs,range(4))
    for held in (-1,0,1,2,3):
        train = [j for j in range(4) if j != held]
        for arm in ARMS:
            started = time.perf_counter()
            result = solve([obs[j] for j in train], [truth[j] for j in train], focal, arm)
            name = f'{held}_{arm}'
            for field in ('parameters','seed','coefficients'): arrays[f'{name}_{field}'] = result[field]
            pred = predict(obs,focal,result['parameters'],result['seed'],result['coefficients'],arm=='spatial_backing')
            targets = range(6) if held == -1 else [held]
            for j in targets:
                arrays[f'{name}_layers_{j}'] = pred[j]['layers']
                arrays[f'{name}_valid_{j}'] = pred[j]['transport']['valid']
                donors = [k for k in train if k != j]
                arrays[f'{name}_support_{j}'] = support(pred,donors,j)
            y = np.concatenate([truth[j] for j in train]); label = np.concatenate([obs[j]['label'] for j in train])
            fitted = np.concatenate([pred[j]['layers'].sum(0) for j in train])
            record = dict(name=name, held=held, arm=arm, train=train,
                history=result['history'], parameters=result['parameters'].tolist(),
                seed=result['seed'].tolist(), bound_hits=result['bound_hits'],
                concentration=float(np.linalg.norm(light_vectors(result['seed'],result['parameters'][4:])[-1])),
                train_all=error(fitted,y), train_reflection=error(fitted[label],y[label]),
                elapsed_seconds=time.perf_counter()-started)
            records.append(record)
            print('R30_FIT',side,name,record['train_reflection']['median_display'],
                  round(record['elapsed_seconds'],2),flush=True)
    for j,o in enumerate(obs):
        for name,value in [('xy',o['xy']),('label',o['label']),('prior_mask',o['prior_mask'])]:
            arrays[f'{name}_{j}'] = value
    np.savez_compressed(out/f'predictions_{side}.npz',**arrays)
    write(out/f'fits_{side}.json',records)
    return records


def score_arrays(obs, arrays, truth):
    folds = []; eligible = []; masks = {}; losses = {}
    for j,o in enumerate(obs):
        prefix = str(j) if j<4 else '-1'
        mask = o['prior_mask'].copy()
        for arm in ARMS: mask &= arrays[f'{prefix}_{arm}_support_{j}']
        masks[j] = mask
        label = o['label']; r = mask & label; other = mask & ~label
        losses[j] = dict(prior=int(o['prior_mask'].sum()),joint=int(mask.sum()),
            reflection=int(r.sum()),other=int(other.sum()),
            lost_from_prior=int((o['prior_mask'] & ~mask).sum()))
        if j<4 and r.sum()>=8 and other.sum()>=8: eligible.append(j)
        row = {}
        for arm in ARMS:
            layers = arrays[f'{prefix}_{arm}_layers_{j}']; pred = layers.sum(0)
            row[arm] = dict(reflection=error(pred[r],truth[j][r]),other=error(pred[other],truth[j][other]),
                supported=error(pred[mask],truth[j][mask]),
                reflection_layer_means=[float(a[r].mean()) if r.any() else None for a in layers])
        folds.append(row)
    scores = {arm:float(np.mean([(folds[j][arm]['reflection']['linear_mse']+
        folds[j][arm]['other']['linear_mse'])/2 for j in eligible])) if len(eligible)>=2 else None for arm in ARMS}
    selected = min(ARMS,key=lambda a:scores[a]) if len(eligible)>=2 else None
    return dict(folds=folds,coverage=losses,eligible_folds=eligible,scores=scores,selected=selected), masks


def sampling(obs, focal, z):
    results = {}
    for arm in ARMS:
        prefix = f'-1_{arm}'; estimates = {}; valid = {}; boundary = []
        for n in (3,7,15,31):
            rows = []; hits = []
            for o in obs:
                # Bound temporary SG arrays at dense quadrature.
                pp = [predict([dict(o,xy=o['xy'][start:start+64])],focal,
                    z[f'{prefix}_parameters'],z[f'{prefix}_seed'],z[f'{prefix}_coefficients'],
                    arm=='spatial_backing',n=n)[0] for start in range(0,len(o['xy']),64)]
                rows.append(np.concatenate([p['layers'] for p in pp],axis=1))
                hits.append(np.concatenate([p['transport']['valid'] for p in pp]))
                if n==31:
                    # Iris/pupil mixed footprint: constant equals transmitted
                    # pupil fraction; compare both nonzero material components.
                    boundary.append(np.concatenate([(p['transport']['iris'].sum(1)>1e-12)&
                        (p['transport']['constant'][:,0]>1e-12) for p in pp]))
            estimates[n] = np.concatenate(rows,axis=1); valid[n] = np.concatenate(hits)
        common = np.all(list(valid.values()),axis=0); edge = np.concatenate(boundary)
        record = dict(common_pixels=int(common.sum()),boundary_pixels=int((common&edge).sum()),comparisons={})
        for n in (3,7,15):
            delta = np.abs(estimates[n]-estimates[31]); total = np.abs(estimates[n].sum(0)-estimates[31].sum(0))
            record['comparisons'][n] = {}
            for label,mask in [('all',common),('boundary',common&edge),('interior',common&~edge)]:
                record['comparisons'][n][label] = dict(pixels=int(mask.sum()),
                    max_sum=float(total[mask].max()) if mask.any() else None,
                    mean_sum=float(total[mask].mean()) if mask.any() else None,
                    layer_max=[float(d[mask].max()) if mask.any() else None for d in delta])
        results[arm] = record
    return results


def main():
    ap=argparse.ArgumentParser(description=__doc__); add_shot_argument(ap)
    ap.add_argument('--preflight',action='store_true')
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args(); shot=from_args(args)
    evidence=freeze(shot); focal=np.array(json.loads((shot.logs/'r29_shape.json').read_text())['focal'])
    out=shot.composite/'r30_continuous'; out.mkdir(parents=True,exist_ok=True)
    observations={s:prepare(shot,s,focal) for s in ('left','right')}
    if args.preflight:
        write(shot.logs/'r30_preflight.json',{s:[dict(source=o['frame']+700,candidates=len(o['xy']),
            prior_supported=int(o['prior_mask'].sum())) for o in obs] for s,obs in observations.items()})
        print('R30_FROZEN',flush=True); return
    code={n:sha(ROOT/'scripts'/n) for n in ('eye_continuous_fit.py','audit_r30_continuous.py')}
    results={}
    with threadpool_limits(limits=1):
        for side,obs in observations.items():
            stamp=out/f'fit_code_{side}.json'
            if args.resume and stamp.exists():
                assert json.loads(stamp.read_text())==code
                results[side]=dict(fits=json.loads((out/f'fits_{side}.json').read_text()))
            else:
                results[side]=dict(fits=fit_eye(shot,side,obs,focal,out))
                write(stamp,code)
        # All predictions for both eyes exist before any seen 805/815 scoring RGB.
        for side,obs in observations.items():
            with np.load(out/f'predictions_{side}.npz') as archive: z=dict(archive)
            truth=training_values(shot,side,obs,range(6))
            scores,masks=score_arrays(obs,z,truth)
            results[side].update(evaluation=scores,sampling=sampling(obs,focal,z))
            np.savez_compressed(out/f'masks_{side}.npz',**{f'mask_{j}':m for j,m in masks.items()})
    write(shot.logs/'r30_continuous.json',dict(evidence=evidence,results=results,focal=focal.tolist(),
        code_sha256=code,artifact_hashes={str(p.relative_to(ROOT)):sha(p) for p in out.glob('*.*')},
        fresh_test=False,production_repairs=0,material_validated=False,scene_replacement=False,
        continuous_tracking=False,credible_3d_lids=False,source_driven_iris_illumination=False))
    print('R30_COMPLETE',flush=True)


if __name__=='__main__': main()
