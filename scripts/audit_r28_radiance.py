"""Seen-source radiance stability and explicit pupil/backing evidence diagnostic."""
import argparse
import json
import numpy as np
from audit_r24_source import ROOT, sha, write, polygon
from audit_r26_joint import get_eye
from audit_r27_photometry import read_rgb, error, RIDGES as OLD_RIDGES
from partial_eye_geometry import conservative_open
from eye_appearance_fit import MATERIAL_COLUMNS, fit as old_fit
from eye_radiance_fit import (APPEARANCE, fit_radiance, sphere_gram,
    design_with_backing, appearance_support, angular_support)
from shot_paths import add_shot_argument, from_args

RIDGES = (None, 1e-6, 1e-4, 1e-2)


def freeze(shot):
    paths = [ROOT/'docs/r28-eye-protocol.md', shot.dir/'shot.json',
             shot.logs/'r27_photometry.json', shot.logs/'r27_verification.json']
    paths += list((shot.composite/'r27_photometry').glob('*.npz'))
    paths += list((shot.composite/'r27_photometry').glob('*.json'))
    for side in ('left', 'right'):
        paths += [shot.relight/f'neural_iris_observations/{f:06d}_{side}.npz'
                  for f in (30, 60, 90, 120, 105, 115)]
    record = {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}
    path = shot.logs/'r28_freeze.json'
    if path.exists(): assert json.loads(path.read_text()) == record
    else: write(path, record)
    return record


def prepare(shot, side, focal):
    root = shot.composite/'r27_photometry'
    geometry = json.loads((root/f'geometry_{side}.json').read_text())
    obs, profiles = geometry['observations'], geometry['profiles']
    with np.load(root/f'geometry_{side}.npz') as d: table = d['table']
    matrices, coordinates, directions, reflections, pupils, support = {}, {}, {}, {}, {}, {}
    for j, o in enumerate(obs):
        _, safe = conservative_open(o['shape'], o['label']['lid'], [], 2.)
        y, x = np.nonzero(safe); xy = np.stack([x, y], 1)
        items = [design_with_backing(xy, *get_eye(obs, table, p['best_rows'][j], focal),
                                    focal, o['centre']) for p in profiles]
        valid = np.all([item[1] for item in items], axis=0)
        matrices[j] = np.array([item[0][valid] for item in items])
        directions[j] = np.array([item[2][valid] for item in items])
        pupils[j] = np.array([item[3][valid] for item in items])
        coordinates[j] = xy[valid]
        labelled = polygon(o['shape'], o['label']['reflection'])[y, x]
        reflections[j] = labelled[valid]
        support[j] = dict(lid_eligible=len(xy),labelled_lid_eligible=int(labelled.sum()),
            common_candidate=int(valid.sum()),labelled_candidate=int(reflections[j].sum()),
            pupil_mixed_or_full_all_witnesses=int(np.all(pupils[j] > 0, axis=0).sum()))
    masks = {}
    for j in (4, 5):
        appearance = np.all([appearance_support([matrices[k][g] for k in range(4)], matrices[j][g])
                             for g in range(len(profiles))], axis=0)
        angular = np.all([angular_support([directions[k][g] for k in range(4)], directions[j][g])[0]
                          for g in range(len(profiles))], axis=0)
        masks[j] = appearance & angular
        support[j].update(appearance_supported=int(appearance.sum()),
            angular_supported=int(angular.sum()),scored=int(masks[j].sum()),
            labelled_appearance=int((reflections[j]&appearance).sum()),
            labelled_angular=int((reflections[j]&angular).sum()),
            labelled_scored=int((reflections[j]&masks[j]).sum()))
    return obs, matrices, coordinates, directions, reflections, pupils, masks, support


def fit_predict(shot, side, obs, matrices, coordinates, masks):
    def values(j):
        a = read_rgb(shot, side, obs[j]['frame']); p = coordinates[j]
        return a[p[:,1], p[:,0]]
    truth = np.concatenate([values(j) for j in range(4)])
    development = values(4)[masks[4]]
    records, coefficients, predictions = [], [], []
    for g in range(len(matrices[0])):
        train = np.concatenate([matrices[j][g] for j in range(4)])
        for ridge in RIDGES:
            co = fit_radiance(train, truth, ridge)
            dp = (matrices[4][g]@co)[masks[4]]
            layers = np.stack([matrices[5][g][:,:APPEARANCE]@co[:APPEARANCE],
                               matrices[5][g][:,APPEARANCE:]@co[APPEARANCE:]])
            records.append(dict(geometry_index=g,reflection_ridge=ridge,
                train_linear_mse=float(np.mean((train@co-truth)**2)),
                development_linear_mse=float(np.mean((dp-development)**2)) if len(dp) else None,
                environment_rms=np.sqrt(np.einsum('ic,ij,jc->c',co[APPEARANCE:],sphere_gram(),co[APPEARANCE:])).tolist(),
                maximum_light_coefficient=float(co[APPEARANCE:].max()),
                max_candidate_prediction=float(layers.sum(0).max())))
            coefficients.append(co); predictions.append(layers)
    ranked = sorted(range(len(records)), key=lambda i:(records[i]['development_linear_mse'],i)) if len(development) else []
    selected = ranked[0] if ranked else None
    control = next((i for i in ranked if records[i]['reflection_ridge'] is None), None)
    reflected = next((i for i in ranked if records[i]['reflection_ridge'] is not None), None)
    return dict(records=records,selected=selected,control=control,best_reflected=reflected), dict(
        coefficients=np.array(coefficients),predictions=np.array(predictions))


def score(shot, side, output):
    with np.load(output/f'predictions_{side}.npz') as d:
        xy, mask, reflected, predictions = d['xy'], d['supported'], d['labelled_reflection'], d['predictions']
    a = read_rgb(shot, side, 115); truth = a[xy[:,1],xy[:,0]]
    return [dict(index=i,supported=error(p.sum(0)[mask],truth[mask]),
        labelled_reflection=error(p.sum(0)[mask&reflected],truth[mask&reflected]),
        unsupported_diagnostic=error(p.sum(0)[~mask],truth[~mask]),
        clipped_supported_pixels=int(np.any(p.sum(0)[mask] > 1, axis=-1).sum()))
        for i,p in enumerate(predictions)]


def old_support_replay(shot, side, out):
    root=shot.composite/'r27_photometry'
    with np.load(root/f'designs_{side}.npz') as d: data={k:d[k] for k in d.files}
    rgb=[]
    for j,frame in enumerate((30,60,90,120)):
        a=read_rgb(shot,side,frame); p=data[f'xy_{j}']; rgb.append(a[p[:,1],p[:,0]])
    truth=np.concatenate(rgb); results=[]; coefficients=[]; predictions=[]
    for g in range(len(data['design_0'])):
        train=np.concatenate([data[f'design_{j}'][g] for j in range(4)])
        for ridge in RIDGES:
            co=fit_radiance(train,truth,ridge,MATERIAL_COLUMNS)
            pred=data['design_5'][g]@co
            results.append(dict(geometry_index=g,reflection_ridge=ridge,
                max_predicted_linear=float(pred.max()),max_light_coefficient=float(co[289:].max())))
            coefficients.append(co); predictions.append(pred)
    np.savez_compressed(out/f'old_support_{side}.npz',coefficients=coefficients,predictions=predictions)
    return results


def main():
    ap=argparse.ArgumentParser(description=__doc__); add_shot_argument(ap)
    ap.add_argument('--freeze-only',action='store_true'); args=ap.parse_args(); shot=from_args(args)
    evidence=freeze(shot)
    if args.freeze_only: print('R28_FROZEN'); return
    out=shot.composite/'r28_radiance'; out.mkdir(parents=True,exist_ok=True)
    prior=json.loads((shot.logs/'r27_photometry.json').read_text()); focal=np.array(prior['focal'])
    results={}
    for side in ('left','right'):
        obs,matrices,xy,directions,reflections,pupils,masks,support=prepare(shot,side,focal)
        arrays={f'design_{j}':matrices[j] for j in matrices}
        arrays.update({f'xy_{j}':xy[j] for j in xy})
        arrays.update({f'directions_{j}':directions[j] for j in directions})
        np.savez_compressed(out/f'designs_{side}.npz',**arrays)
        print('R28_SUPPORT',side,json.dumps(support),flush=True)
        selection,pred=fit_predict(shot,side,obs,matrices,xy,masks)
        write(out/f'selection_{side}.json',selection)
        np.savez_compressed(out/f'predictions_{side}.npz',xy=xy[5],supported=masks[5],
            labelled_reflection=reflections[5],pupil_fraction=pupils[5],**pred)
        results[side]=dict(support=support,selection=selection,
            old_support_stability=old_support_replay(shot,side,out))
    for side in ('left','right'):
        results[side]['seen_815_scores']=score(shot,side,out)
        results[side]['artifact_hashes']={str(p.relative_to(ROOT)):sha(p) for p in out.glob(f'*_{side}.*')}
        idx=results[side]['selection']['selected']
        print('R28_SELECTED',side,idx,json.dumps(results[side]['seen_815_scores'][idx]) if idx is not None else 'NO SUPPORT',flush=True)
    write(shot.logs/'r28_radiance.json',dict(evidence=evidence,results=results,focal=focal.tolist(),
        fresh_final_test=False,production_repairs=0,clean_material_certified=False,
        actual_scene_light_computed=False,credible_3d_lids=False,
        code_sha256={n:sha(ROOT/'scripts'/n) for n in ('eye_radiance_fit.py','audit_r28_radiance.py','connected_eye.py')}))
    print('R28_RADIANCE_COMPLETE',flush=True)


if __name__=='__main__': main()
