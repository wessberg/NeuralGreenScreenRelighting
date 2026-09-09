"""Post-fit diagnostics: physical boundary compatibility, local solve and layers."""
import argparse
import json
import numpy as np
from threadpoolctl import threadpool_limits
from audit_r24_source import write
from audit_r30_continuous import prepare
from eye_continuous_fit import corrected, ARMS
from joint_eye_geometry import pupil_residuals
from partial_eye_geometry import project, distances
from shot_paths import add_shot_argument, from_args


def main():
    ap=argparse.ArgumentParser(description=__doc__); add_shot_argument(ap); shot=from_args(ap.parse_args())
    report=json.loads((shot.logs/'r30_continuous.json').read_text()); focal=np.array(report['focal'])
    result={}
    with threadpool_limits(limits=1):
        for side in ('left','right'):
            obs=prepare(shot,side,focal); r=report['results'][side]
            source_geometry=json.loads((shot.composite/f'r27_photometry/geometry_{side}.json').read_text())['observations']
            with np.load(shot.composite/f'r30_continuous/predictions_{side}.npz') as z:
                rows={}
                for arm in ARMS:
                    p=z[f'-1_{arm}_parameters']; boundary=[]
                    for j,o in enumerate(obs):
                        eye=corrected(o['eye'],focal,p[:4]); a=np.arange(360)*2*np.pi/360
                        ring=np.stack([eye.limbus_radius*np.cos(a),eye.limbus_radius*np.sin(a),np.full(360,eye.sag)],1)
                        ring=project(ring@eye.rot.T+eye.apex,focal,o['centre'])
                        limbus=distances(source_geometry[j]['label']['limbus'],ring)
                        pupil,cap=pupil_residuals(o['pupil'],eye,[o['offset']],[eye.pupil_radius],focal,o['centre'])
                        boundary.append(dict(source=o['frame']+700,limbus_max_px=float(limbus.max()),
                            limbus_rms_px=float(np.sqrt(np.mean(limbus**2))),
                            pupil_max_px=float(pupil.max()),pupil_cap_all=bool(cap.all()),
                            note='All reused annotations; post-fit diagnostic, not independent geometry validation'))
                    fit=next(f for f in r['fits'] if f['held']==-1 and f['arm']==arm)
                    co=z[f'-1_{arm}_coefficients']; label=np.concatenate([o['label'] for o in obs[:4]])
                    layers=np.concatenate([z[f'-1_{arm}_layers_{j}'] for j in range(4)],axis=1)
                    steps=[h for f in r['fits'] if f['arm']==arm for h in f['history'][1:]]
                    rows[arm]=dict(train_reflection=fit['train_reflection'],parameters=p.tolist(),
                        concentration=fit['concentration'],bound_hits=fit['bound_hits'],
                        localized_lobe_rgb=co[-1].tolist(),
                        full_objectives=[h['objective'] for h in fit['history']],
                        solver_steps=len(steps),evaluation_limit_steps=sum(h['status']==0 for h in steps),
                        rejected_steps=sum(not h['accepted'] for h in steps),
                        invalid_training_pixels_max=max(h['invalid'] for f in r['fits'] if f['arm']==arm for h in f['history']),
                        train_reflection_layer_means=[float(a[label].mean()) for a in layers],
                        maximum_prediction=float(max(z[f'-1_{arm}_layers_{j}'].sum(0).max() for j in range(6))),
                        boundaries=boundary)
                result[side]=dict(arms=rows,coverage=r['evaluation']['coverage'],
                    selected=r['evaluation']['selected'],scores=r['evaluation']['scores'],
                    seen_815=r['evaluation']['folds'][5],sampling=r['sampling'])
    write(shot.logs/'r30_summary.json',result)
    for side,r in result.items():
        print(side,'coverage',r['coverage'],'selection',r['selected'],flush=True)
        for arm,a in r['arms'].items():
            print(arm,'train',round(a['train_reflection']['median_display'],3),'score',r['scores'][arm],
                '815',r['seen_815'][arm]['reflection'],'parameters',a['parameters'],
                'lobe',a['localized_lobe_rgb'],'limited',a['evaluation_limit_steps'],flush=True)


if __name__=='__main__': main()
