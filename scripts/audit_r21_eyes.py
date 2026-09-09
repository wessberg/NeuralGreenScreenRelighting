"""Frozen independent-render eye separation and finite geometry selection."""
import argparse
import json
import os
import time
from pathlib import Path

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2
import numpy as np

from audit_r20_eyes import measures, panel, sha, write_image, write_json
from corneal_transport import EyeOptics, trace
from coupled_eye_relight import apply_coupled
from eye_layer_fit import components, design, fit
from shot_paths import add_shot_argument, from_args

ROOT = Path(__file__).resolve().parents[1]
SIZE = 12
SELECTION = (3,9,15,21)
TEST = (5,11,17,23)
TRAIN = tuple(i for i in range(24) if i not in SELECTION+TEST)
ARMS = dict(continuous=dict(split_pupil=False), supplied=dict(),
    yaw_minus=dict(yaw_bias=-2), yaw_plus=dict(yaw_bias=2),
    radius_minus=dict(radius_scale=.95), radius_plus=dict(radius_scale=1.05),
    pupil_minus=dict(pupil_bias=-.3), pupil_plus=dict(pupil_bias=.3),
    lid_minus=dict(lid_bias=-.4), lid_plus=dict(lid_bias=.4))
CODE = ('render_r21_eyes.py','render_r20_transport.py','corneal_transport.py',
        'eye_layer_fit.py','observation_eye_fit.py','audit_r21_eyes.py','coupled_eye_relight.py')


def geometry(frame, *, split_pupil=True, yaw_bias=0., radius_scale=1., pupil_bias=0., lid_bias=0.):
    yy,xx = np.mgrid[:64,:64]
    phase = 2*np.pi*frame/24
    eye = EyeOptics(yaw=np.deg2rad(12*np.sin(phase)+yaw_bias),pitch=np.deg2rad(8*np.cos(phase)),
                    radius=7.8*radius_scale,pupil_radius=1.8+pupil_bias)
    height = 4*(1-np.exp(-((frame-11)/1.4)**2))
    height = max(0.,height+lid_bias) if height > 0 else 0.
    transport = trace(np.stack([xx,yy],-1),eye,lid_height=height)
    return transport, dict(split_pupil=split_pupil,pupil_radius=eye.pupil_radius,iris_radius=eye.iris_radius)


def masks(frame):
    t,_ = geometry(frame)
    score = cv2.erode(t['visible'].astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    score &= np.abs(t['interior_radius']-1.8) > .7
    return dict(interior=score,iris=score&t['iris'],pupil=score&t['pupil'],full_visible=t['visible'])


def read_rgb(path):
    result = cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if result is None or result.shape != (64,64,3) or not np.isfinite(result).all():
        raise ValueError(f'Invalid rendered image: {path}')
    return result[...,::-1].astype(np.float64)


def load_sources(shot):
    report = json.loads((shot.logs/'r21_cycles.json').read_text())
    assert report['protocol_sha256'] == sha(ROOT/'docs/r21-eye-protocol.md')
    assert all(sha(ROOT/'scripts'/k)==v for k,v in report['code_sha256'].items())
    records = report['records']
    sources = []
    for record in records:
        file = record['files']['source']
        assert sha(file['path']) == file['sha256']
        sources.append(read_rgb(file['path']))
    assert len(sources)==24 and [r['frame'] for r in records]==list(range(24))
    return sources, records


def training_inputs(sources, options):
    matrices, observations = [], []
    for i in TRAIN:
        t,kwargs = geometry(i,**options)
        select = masks(i)['interior'].ravel()
        matrices.append(design(t,SIZE,**kwargs)[select])
        observations.append(sources[i].reshape(-1,3)[select])
    return matrices, observations


def predict(values, options):
    result = []
    for i in range(24):
        t,kwargs = geometry(i,**options)
        clean,old = components(t,values,SIZE,**kwargs)
        result.append(dict(clean=clean,old=old,sum=clean+old,visible=t['visible']))
    return result


def metric(estimates, truths, frames, region='interior'):
    a,b = [],[]
    for i in frames:
        mask = masks(i)[region]
        a.append(estimates[i][mask])
        b.append(truths[i][mask])
    a,b = np.concatenate(a),np.concatenate(b)
    return measures(a,b,np.ones(len(a),bool))


def choose(predictions, sources):
    scores = {name:metric([p['sum'] for p in pred],sources,SELECTION)['display_mae']
              for name,pred in predictions.items() if name!='continuous'}
    return min(scores,key=scores.get),scores


def prepare(shot):
    sources,records = load_sources(shot)
    report = dict(protocol_sha256=sha(ROOT/'docs/r21-eye-protocol.md'),
        code_sha256={name:sha(ROOT/'scripts'/name) for name in CODE},
        train=TRAIN,selection=SELECTION,test=TEST,arms=ARMS,
        source_sha256={r['frame']:r['files']['source']['sha256'] for r in records},
        region_pixels={str(i):{k:int(v.sum()) for k,v in masks(i).items()} for i in range(24)},
        source_channels_above_one=sum(int((x>1).sum()) for x in sources),
        oracle_geometry_mask=True,reference_rgb_read=False)
    write_json(shot.logs/'r21_preflight.json',report)
    print('R21_PREFLIGHT_COMPLETE',flush=True)


def evaluate(shot):
    out = shot.composite/'r21_eyes'
    sources,records = load_sources(shot)
    pre = json.loads((shot.logs/'r21_preflight.json').read_text())
    assert pre['protocol_sha256']==sha(ROOT/'docs/r21-eye-protocol.md')
    assert all(sha(ROOT/'scripts'/k)==v for k,v in pre['code_sha256'].items())
    report = dict(protocol_sha256=pre['protocol_sha256'],code_sha256=pre['code_sha256'],arms={})
    predictions = {}
    for name,options in ARMS.items():
        matrices,rgb = training_inputs(sources,options)
        start = time.perf_counter()
        values,active,info = fit(matrices,rgb,size=SIZE,split_pupil=options.get('split_pupil',True))
        info['seconds'] = time.perf_counter()-start
        np.savez_compressed(out/f'{name}_fit.npz',values=values,active=active)
        predictions[name] = predict(values,options)
        report['arms'][name] = info
        print('R21_FIT',name,round(info['seconds'],2),info['solvers'],flush=True)
    # Selection has only source observations and predictions, never clean or target RGB.
    selected,scores = choose(predictions,sources)
    report['selection'] = dict(selected=selected,source_sum_display_mae=scores,
        frames=SELECTION,scope='Finite one-at-a-time hypotheses; not real geometry calibration')
    write_json(shot.logs/'r21_selection.json',report['selection'])
    # Reference access starts only after the finite candidate choice is saved.
    clean,targets = [],[]
    for record in records:
        for key,collection in (('clean',clean),('target',targets)):
            file = record['files'][key]
            assert sha(file['path']) == file['sha256']
            collection.append(read_rgb(file['path']))
    new = [target-base for target,base in zip(targets,clean)]
    old = [source-base for source,base in zip(sources,clean)]
    report['reference_subtraction'] = dict(new_negative_channels=sum(int((r<0).sum()) for r in new),
        new_min=float(min(r.min() for r in new)),old_negative_channels=sum(int((r<0).sum()) for r in old),
        old_min=float(min(r.min() for r in old)),raw_differences_scored=True)
    for name,pred in predictions.items():
        estimates = [p['clean'] for p in pred]
        reflections = [p['old'] for p in pred]
        sums = [p['sum'] for p in pred]
        coupled = [a+b for a,b in zip(estimates,new)]
        report['arms'][name]['test'] = {region:dict(
            sum_fit=metric(sums,sources,TEST,region), interior=metric(estimates,clean,TEST,region),
            old_reflection=metric(reflections,old,TEST,region),
            coupled_known_scene=metric(coupled,targets,TEST,region),
            missing_visibility=sum(int((masks(i)[region]&~pred[i]['visible']).sum()) for i in TEST))
            for region in ('interior','iris','pupil','full_visible')}
        np.savez_compressed(out/f'{name}_predictions.npz',clean=np.array(estimates),old=np.array(reflections))
    report['controls'] = {region:dict(source_as_interior=metric(sources,clean,TEST,region),
        add_only=metric([a+b for a,b in zip(sources,new)],targets,TEST,region))
        for region in ('interior','iris','pupil','full_visible')}
    disabled = []
    for i in range(24):
        state = f'frame {i:02d}; '+('FINAL TEST' if i in TEST else 'SELECTION' if i in SELECTION else 'training')
        estimated = predictions[selected][i]['clean']
        candidate,_ = apply_coupled(sources[i],masks(i)['full_visible'],estimated,np.maximum(new[i],0),
            support=masks(i)['full_visible'],geometry_reliable=False)
        assert np.array_equal(candidate,sources[i])
        disabled.append(i)
        board = np.concatenate([
            np.concatenate([panel(sources[i],'CYCLES SOURCE',state),
                panel(clean[i],'CYCLES CLEAN REFERENCE','known nonblack pupil; withheld truth'),
                panel(estimated,'FITTED INTERIOR',selected+'; NOT certified')],1),
            np.concatenate([panel(targets[i],'CYCLES TARGET REFERENCE','different environment; same interior'),
                panel(estimated+new[i],'FIT + ORACLE NEW REFLECTION','experimental separation, supplied transport'),
                panel(sources[i]+new[i],'ADD-ONLY CONTROL','old reflection remains')],1)],0)
        board = cv2.copyMakeBorder(board,30,0,0,0,cv2.BORDER_CONSTANT)
        cv2.putText(board,'NONPERFORMER / independent textured Cycles / simplified anatomy / no real repair',
                    (8,21),0,.48,(0,220,255),1,cv2.LINE_AA)
        for repeat in range(4):
            write_image(out/'review'/f'{repeat*24+i:06d}.png',board)
    report.update(disabled_coupled_fallbacks=len(disabled),production_repairs=0,
        scope='Independent rendered nonperformer RGB; supplied simplified geometry, oracle scoring region and new reflection',
        real_review='final/r20_source_fallback.mp4 (unchanged original catchlights)',
        reference_sha256={str(r['frame']):{k:f['sha256'] for k,f in r['files'].items()} for r in records})
    write_json(shot.logs/'r21_eyes.json',report)
    print('R21_EVALUATION_COMPLETE',selected,flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_shot_argument(ap)
    ap.add_argument('--prepare-only',action='store_true')
    args = ap.parse_args()
    shot = from_args(args)
    (prepare if args.prepare_only else evaluate)(shot)


if __name__ == '__main__':
    main()
