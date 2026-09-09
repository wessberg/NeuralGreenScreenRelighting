"""Frozen source appearance/light witnesses and a fresh withheld RGB check."""
import argparse
import json
import numpy as np
from audit_r24_source import ROOT, sha, write, polygon
from audit_r25_source import camera
from audit_r26_joint import get_eye
from color_pipeline import linear_to_srgb
from eye_appearance_fit import design, observed_support, fit, layers
from joint_eye_geometry import build_table, SHARED
from partial_eye_geometry import split_arc, fit_family, conservative_open, curve, distances
from shot_paths import add_shot_argument, from_args

FRAMES = (30,60,90,120,105,115)
RIDGES = (None, 1e-4, 1e-2, 1.)


def evidence(shot):
    paths = [ROOT/'docs/r27-eye-protocol.md', ROOT/'scripts/data/r27_source_boundaries.json',
             shot.logs/'r27_preflight.json', shot.dir/'shot.json']
    for side in ('left','right'):
        paths += [shot.composite/f'r26_joint/geometry_{side}.json',
                  shot.composite/f'r26_joint/family_{side}.npz']
        paths += [shot.relight/f'neural_iris_observations/{f:06d}_{side}.npz' for f in FRAMES]
    pre = json.loads((shot.logs/'r27_preflight.json').read_text())
    assert pre['protocol_sha256'] == sha(paths[0])
    for r in pre['records']:
        assert r['archive_sha256'] == sha(shot.relight/f'neural_iris_observations/000115_{r["side"]}.npz')
    record = {str(p.relative_to(ROOT)):sha(p) for p in paths}
    path = shot.logs/'r27_freeze.json'
    if path.exists(): assert json.loads(path.read_text()) == record
    else: write(path,record)
    return record


def prepare_geometry(shot, side, focal, centre, out):
    prior = json.loads((shot.composite/f'r26_joint/geometry_{side}.json').read_text())
    obs = prior['observations']
    label = next(r for r in json.loads((ROOT/'scripts/data/r27_source_boundaries.json').read_text())['records'] if r['side']==side)
    meta = next(r for r in json.loads((shot.logs/'r27_preflight.json').read_text())['records'] if r['side']==side)
    train, check, _, _ = split_arc(label['limbus'])
    c = centre-np.array(meta['origin_xy'])
    target = dict(frame=115,side=side,target=True,shape=meta['shape'][:2],centre=c.tolist(),
                  origin=meta['origin_xy'],family=fit_family(train,focal,c),pupil=label['pupil'],
                  label=label,limbus_check=check.tolist())
    tt = build_table([target],focal); tt[:,1] = len(obs)
    with np.load(shot.composite/f'r26_joint/family_{side}.npz') as d: table=d['table']
    base = len(table); table=np.concatenate([table,tt]); obs=obs+[target]
    profiles=[]
    for profile in prior['profiles'][:6]:
        rows=np.flatnonzero((tt[:,0]==profile['shared_index']) & (tt[:,4]<=2))
        if not len(rows): continue
        row=int(rows[np.argmin(tt[rows,5])])+base
        profiles.append(dict(shared_index=profile['shared_index'],best_rows=profile['best_rows']+[row]))
    write(out/f'geometry_{side}.json',dict(observations=obs,profiles=profiles,
        original_witnesses=6, retained_witnesses=len(profiles),full_family_certified=False))
    np.savez_compressed(out/f'geometry_{side}.npz',table=table)
    return obs,table,profiles


def make_designs(obs, table, profiles, focal):
    result={}; masks={}; xy={}; reflected={}
    for oi,o in enumerate(obs):
        _, safe=conservative_open(o['shape'],o['label']['lid'],[],2.)
        y,x=np.nonzero(safe); coords=np.stack([x,y],1)
        designs=[]; valid=[]
        for p in profiles:
            eye,offset=get_eye(obs,table,p['best_rows'][oi],focal)
            matrix, good=design(coords,eye,offset,focal,o['centre'])
            designs.append(matrix); valid.append(good)
        common=np.all(valid,axis=0) if profiles else np.zeros(len(coords),bool)
        xy[oi]=coords[common]
        result[oi]=np.array(designs)[:,common] if profiles else np.zeros((0,0,342))
        r=polygon(o['shape'],o['label']['reflection'])
        reflected[oi]=r[y[common],x[common]]
        masks[oi]=dict(lid_eligible=len(coords),common_candidate=int(common.sum()),
                       labelled_reflection_candidate=int(reflected[oi].sum()))
    # A fixed common evaluation mask: every arm predicts exactly the same pixels.
    for oi in (4,5):
        support=[observed_support([result[j][g] for j in range(4)],result[oi][g])
                 for g in range(len(profiles))]
        good=np.all(support,axis=0) if support else np.zeros(0,bool)
        result[oi]=result[oi][:,good]; xy[oi]=xy[oi][good]; reflected[oi]=reflected[oi][good]
        masks[oi].update(scored=int(good.sum()),appearance_abstentions=int((~good).sum()),
                         labelled_reflection_scored=int(reflected[oi].sum()))
    return result,xy,reflected,masks


def read_rgb(shot, side, frame):
    with np.load(shot.relight/f'neural_iris_observations/{frame:06d}_{side}.npz') as d:
        return d['linear_rgb']


def fit_predict(shot,side,matrices,xy):
    """Only four fit frames and the development frame are read. Final RGB absent."""
    rgb=[]
    for oi,frame in enumerate(FRAMES[:4]):
        a=read_rgb(shot,side,frame); p=xy[oi]; rgb.append(a[p[:,1],p[:,0]])
    y=np.concatenate(rgb)
    a=read_rgb(shot,side,105); p=xy[4]; dev=a[p[:,1],p[:,0]]
    records=[]; coefficients=[]; predictions=[]; dev_predictions=[]
    for g in range(len(matrices[0])):
        x=np.concatenate([matrices[j][g] for j in range(4)])
        for ridge in RIDGES:
            co=fit(x,y,ridge)
            train=sum(layers(x,co)); dp=sum(layers(matrices[4][g],co))
            material,reflection=layers(matrices[5][g],co)
            records.append(dict(geometry_index=g,reflection_ridge=ridge,
                train_linear_mse=float(np.mean((train-y)**2)),
                development_linear_mse=float(np.mean((dp-dev)**2)) if len(dev) else None))
            coefficients.append(co); predictions.append([material,reflection]); dev_predictions.append(dp)
    ranked=sorted(range(len(records)),key=lambda i:(records[i]['development_linear_mse'],i)) if len(dev) else []
    controls=[i for i in ranked if records[i]['reflection_ridge'] is None]
    return dict(records=records,selected=ranked[0] if ranked else None,
                control=controls[0] if controls else None), dict(coefficients=np.array(coefficients),
                predictions=np.array(predictions),development_predictions=np.array(dev_predictions))


def error(pred, truth):
    if not len(truth): return dict(pixels=0,median_display=None)
    e=np.abs(linear_to_srgb(np.maximum(pred,0))-linear_to_srgb(np.maximum(truth,0))).mean(-1)*255
    return dict(pixels=len(truth),median_display=float(np.median(e)),mean_display=float(e.mean()),
                p95_display=float(np.percentile(e,95)),linear_mse=float(np.mean((pred-truth)**2)))


def score(shot,side,out):
    selection=json.loads((out/f'selection_{side}.json').read_text())
    with np.load(out/f'predictions_{side}.npz') as d:
        xy=d['xy']; predictions=d['predictions']; reflected=d['labelled_reflection']
    a=read_rgb(shot,side,115); truth=a[xy[:,1],xy[:,0]]
    records=[]
    for i,p in enumerate(predictions):
        records.append(dict(index=i,**error(p.sum(0),truth),
            labelled_reflection=error(p.sum(0)[reflected],truth[reflected])))
    sel=selection['selected']; control=selection['control']
    report=dict(models=records,selected=sel,control=control)
    if sel is not None:
        report.update(selected_error=records[sel],control_error=records[control],
            selected_reflected_mean_linear=float(predictions[sel,1].mean()) if len(truth) else None,
            material_range_median_linear=float(np.median(np.ptp(predictions[:,0],axis=0))) if len(truth) else None,
            reflection_range_median_linear=float(np.median(np.ptp(predictions[:,1],axis=0))) if len(truth) else None,
            total_range_median_linear=float(np.median(np.ptp(predictions.sum(1),axis=0))) if len(truth) else None)
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__); add_shot_argument(ap)
    ap.add_argument('--freeze-only',action='store_true'); args=ap.parse_args(); shot=from_args(args)
    frozen=evidence(shot)
    if args.freeze_only: print('R27_FROZEN'); return
    out=shot.composite/'r27_photometry'; out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((shot.relight/'neural_iris_observations/manifest.json').read_text())
    focal,centre=camera(shot,manifest); results={}
    for side in ('left','right'):
        obs,table,profiles=prepare_geometry(shot,side,focal,centre,out)
        matrices,xy,reflected,masks=make_designs(obs,table,profiles,focal)
        arrays={f'design_{j}':matrices[j] for j in matrices}
        arrays.update({f'xy_{j}':xy[j] for j in xy})
        np.savez_compressed(out/f'designs_{side}.npz',**arrays)
        print('R27_DESIGNS',side,len(profiles),json.dumps(masks),flush=True)
        selection,pred=fit_predict(shot,side,matrices,xy)
        write(out/f'selection_{side}.json',selection)
        np.savez_compressed(out/f'predictions_{side}.npz',xy=xy[5],labelled_reflection=reflected[5],**pred)
        # Both eyes' predictions are persisted before either final target is scored.
        checks=[]
        for p in profiles:
            row=table[p['best_rows'][-1]]; ring=obs[-1]['family'][int(row[2])]
            contour=curve(ring['parameters'],*np.deg2rad([ring['yaw_deg'],ring['pitch_deg']]),focal,obs[-1]['centre'])
            checks.append(dict(shared_index=p['shared_index'],limbus=distances(obs[-1]['limbus_check'],contour).tolist(),
                               pupil=row[6:8].tolist()))
        results[side]=dict(witnesses=len(profiles),support=masks,selection=selection,held_boundaries=checks)
    for side in ('left','right'):
        results[side]['final']=score(shot,side,out)
        results[side]['artifact_hashes']={str(p.relative_to(ROOT)):sha(p) for p in out.glob(f'*_{side}.*') if p.suffix in ('.json','.npz')}
        print('R27_FINAL',side,json.dumps(results[side]['final']),flush=True)
    write(shot.logs/'r27_photometry.json',dict(evidence=frozen,results=results,
        focal=focal.tolist(),centre=centre.tolist(),clean_material_certified=False,
        production_repairs=0,actual_scene_light_computed=False,
        code_sha256={n:sha(ROOT/'scripts'/n) for n in ('audit_r27_photometry.py','eye_appearance_fit.py')}))
    print('R27_PHOTOMETRY_COMPLETE',flush=True)


if __name__=='__main__': main()
