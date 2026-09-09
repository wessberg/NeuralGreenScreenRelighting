"""Fresh independent Cycles challenge: observed detail and conditional ambiguity."""
import argparse
from functools import lru_cache
import json
import os
import time
from pathlib import Path

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2
import numpy as np
from scipy import sparse

from audit_r20_eyes import measures, panel, sha, write_image, write_json
from audit_r21_eyes import read_rgb
from corneal_transport import EyeOptics, trace
from coupled_eye_relight import apply_coupled
import eye_detail_fit as detail
from eye_layer_fit import fit as legacy_fit
from shot_paths import add_shot_argument, from_args

ROOT = Path(__file__).resolve().parents[1]
SELECTION = (2,8,14,20,26,32)
TEST = (5,11,17,23,29,35)
TRAIN = tuple(i for i in range(36) if i not in SELECTION+TEST)
ARMS = dict(legacy=dict(iris_size=12,pupil_size=12,environment_size=12),
            coarse=dict(iris_size=12), medium=dict(iris_size=32), fine=dict(iris_size=64))
CODE = ('audit_r22_eyes.py','eye_detail_fit.py','eye_layer_fit.py','corneal_transport.py',
        'observation_eye_fit.py','render_r22_eyes.py','render_r20_transport.py','coupled_eye_relight.py')


def filter_offset():
    """Symmetric two-node quadrature matching the Blackman-Harris second moment.

    Cycles doubles its width before evaluating this window. Four tensor-product
    rays approximate the .01 filter, including material-boundary quadrature error.
    """
    x = np.linspace(-.01,.01,10001)
    phase = 2*np.pi*(x/.02+.5)
    weight = .35875-.48829*np.cos(phase)+.14128*np.cos(2*phase)-.01168*np.cos(3*phase)
    return float(np.sqrt(np.trapezoid(weight*x*x,x)/np.trapezoid(weight,x)))


@lru_cache(maxsize=36)
def geometry(frame):
    y,x = np.mgrid[:64,:64]
    xy = np.stack([x,y],-1)
    phase = 2*np.pi*frame/36+.19
    eye = EyeOptics(yaw=np.deg2rad(13*np.sin(phase)),pitch=np.deg2rad(7*np.cos(phase)))
    height = 4*(1-np.exp(-((frame-17)/1.4)**2))
    offset = filter_offset()
    return [trace(xy+np.array([dx,dy]),eye,lid_height=height)
            for dx in (-offset,offset) for dy in (-offset,offset)]


def masks(frame):
    t = geometry(frame)
    visible = np.any([a['visible'] for a in t],axis=0)
    iris = np.all([a['iris'] for a in t],axis=0)
    pupil = np.all([a['pupil'] for a in t],axis=0)
    return dict(full_visible=visible,iris=iris,pupil=pupil,mixed=visible&~(iris|pupil))


def manifest(shot):
    result = json.loads((shot.logs/'r22_cycles.json').read_text())
    assert result['protocol_sha256']==sha(ROOT/'docs/r22-eye-protocol.md')
    assert all(sha(ROOT/'scripts'/k)==v for k,v in result['code_sha256'].items())
    assert [r['frame'] for r in result['records']]==list(range(36))
    return result


def load(shot, frames, kind='source'):
    records = manifest(shot)['records']
    result = {}
    for i in frames:
        file = records[i]['files'][kind]
        assert sha(file['path'])==file['sha256']
        result[i] = read_rgb(file['path'])
    return result


def training_inputs(sources, options):
    matrices, observations = [], []
    for i in TRAIN:
        select = masks(i)['full_visible'].ravel()
        matrices.append(detail.design(geometry(i),**options)[select])
        observations.append(sources[i].reshape(-1,3)[select])
    return matrices, observations


def predict(values, options):
    split = options['iris_size']**2+options.get('pupil_size',12)**2
    clean, old = [], []
    for i in range(36):
        matrix = detail.design(geometry(i),**options)
        clean.append((matrix[:,:split]@values[:split]).reshape(64,64,3))
        old.append((matrix[:,split:]@values[split:]).reshape(64,64,3))
    return dict(clean=np.array(clean),old=np.array(old))


def metric(estimates, truths, frames=TEST, region='full_visible'):
    a = np.concatenate([estimates[i][masks(i)[region]] for i in frames])
    b = np.concatenate([truths[i][masks(i)[region]] for i in frames])
    return measures(a,b,np.ones(len(a),bool))


def choose(predictions, sources, infos):
    scores = {name:metric(p['clean']+p['old'],sources,SELECTION)['display_mae']
              for name,p in predictions.items() if name!='legacy'}
    eligible = [name for name in scores if infos[name]['certified']]
    return min(eligible,key=scores.get) if eligible else None, scores


def probes():
    result = []
    for i in TEST:
        for region in ('iris','pupil'):
            coords = np.argwhere(masks(i)[region])
            if len(coords):
                y,x = coords[len(coords)//2]
                result.append((i,int(y),int(x)))
    coords = np.argwhere(masks(TEST[0])['iris'])
    for j in (len(coords)//4,3*len(coords)//4):
        y,x = coords[j]
        result.append((TEST[0],int(y),int(x)))
    assert len(result)==12 and len(set(result))==12
    return result


def fit_stage(shot):
    out = shot.composite/'r22_eyes'
    out.mkdir(parents=True,exist_ok=True)
    sources = load(shot,TRAIN)
    report = dict(protocol_sha256=sha(ROOT/'docs/r22-eye-protocol.md'),
        code_sha256={name:sha(ROOT/'scripts'/name) for name in CODE},
        train=TRAIN,selection=SELECTION,test=TEST,arms={},probes=probes(),
        masks={str(i):{k:int(v.sum()) for k,v in masks(i).items()} for i in range(36)})
    write_json(shot.logs/'r22_preflight.json',report)
    predictions = {}
    for name,options in ARMS.items():
        matrices,rgb = training_inputs(sources,options)
        start = time.perf_counter()
        if name=='legacy':
            values,active,info = legacy_fit(matrices,rgb,size=12)
            counts = np.sum([np.asarray(m.power(2).sum(0)).ravel()>0 for m in matrices],axis=0)
            pg = detail.projected_gradient(sparse.vstack(matrices,format='csr'),np.concatenate(rgb),
                                         values,detail.upper_bounds(**options))
            for solver,g in zip(info['solvers'],pg):
                solver.update(projected_gradient=float(g),certified=bool(g<=1e-7))
            info['certified'] = all(s['certified'] for s in info['solvers'])
        else:
            values,counts,info = detail.fit(matrices,rgb,detail.upper_bounds(**options))
        info['seconds'] = time.perf_counter()-start
        np.savez_compressed(out/f'{name}_fit.npz',values=values,frame_counts=counts)
        predictions[name] = predict(values,options)
        support = np.array([detail.complete_support(detail.design(geometry(i),**options),counts)
                            .reshape(64,64) for i in range(36)])
        np.savez_compressed(out/f'{name}_predictions.npz',**predictions[name],support=support)
        info['test_support'] = dict(visible=sum(int(masks(i)['full_visible'].sum()) for i in TEST),
            complete_three_frame=sum(int((support[i]&masks(i)['full_visible']).sum()) for i in TEST))
        report['arms'][name] = info
        print('R22_FIT',name,info,flush=True)
    selection_source = load(shot,SELECTION)
    selected,scores = choose(predictions,selection_source,report['arms'])
    report['selection'] = dict(selected=selected,scores=scores,calibration=False)
    # Save selection before uncertainty and before ANY final-test/reference RGB read.
    write_json(shot.logs/'r22_selection.json',report)
    if selected:
        data = np.load(out/f'{selected}_fit.npz')
        values = data['values']
        options = ARMS[selected]
        matrices,rgb = training_inputs(sources,options)
        matrix,observed = sparse.vstack(matrices,format='csr'),np.concatenate(rgb)
        radius = max(.002,float(np.abs(matrix@values-observed).max())+1e-6)
        split = options['iris_size']**2+options.get('pupil_size',12)**2
        uncertainty = dict(selected=selected,radius=radius,channel='green',records=[],
                           scope='Conditional residual-box model intervals; not calibrated confidence')
        for index,(i,y,x) in enumerate(probes()):
            query = detail.design(geometry(i),**options).getrow(y*64+x).toarray().ravel()
            query[split:] = 0
            info,witness = detail.interval(matrix,observed[:,1],detail.upper_bounds(**options),query,radius)
            info.update(frame=i,y=y,x=x)
            uncertainty['records'].append(info)
            if witness is not None:
                np.savez_compressed(out/f'interval_{index:02d}.npz',values=witness,query=query)
            print('R22_INTERVAL',index,info,flush=True)
    else:
        uncertainty = dict(selected=None,records=[],reason='No numerically certified new arm')
    write_json(shot.logs/'r22_uncertainty.json',uncertainty)
    print('R22_FIT_COMPLETE',selected,flush=True)


def evaluate(shot):
    out = shot.composite/'r22_eyes'
    report = json.loads((shot.logs/'r22_selection.json').read_text())
    assert report['protocol_sha256']==sha(ROOT/'docs/r22-eye-protocol.md')
    assert all(sha(ROOT/'scripts'/k)==v for k,v in report['code_sha256'].items())
    uncertainty = json.loads((shot.logs/'r22_uncertainty.json').read_text())
    assert uncertainty['selected']==report['selection']['selected']
    sources,clean,target = [load(shot,range(36),kind) for kind in ('source','clean','target')]
    new = {i:target[i]-clean[i] for i in range(36)}
    old = {i:sources[i]-clean[i] for i in range(36)}
    predictions = {}
    for name in ARMS:
        pred = dict(np.load(out/f'{name}_predictions.npz'))
        predictions[name] = pred
        coupled = {i:pred['clean'][i]+new[i] for i in range(36)}
        report['arms'][name]['test'] = {region:dict(
            sum_fit=metric(pred['clean']+pred['old'],sources,region=region),
            interior=metric(pred['clean'],clean,region=region),
            old_reflection=metric(pred['old'],old,region=region),
            coupled_known_scene=metric(coupled,target,region=region)) for region in masks(0)}
    report['controls'] = {region:dict(source=metric(sources,clean,region=region),
        add_only=metric({i:sources[i]+new[i] for i in range(36)},target,region=region)) for region in masks(0)}
    selected = report['selection']['selected']
    # No candidate earns real geometry or complete uncertainty. Always exact source.
    for i in range(36):
        estimated = predictions[selected]['clean'][i] if selected else sources[i]
        candidate,_ = apply_coupled(sources[i],masks(i)['full_visible'],estimated,np.maximum(new[i],0),
            support=masks(i)['full_visible'],geometry_reliable=False)
        assert np.array_equal(candidate,sources[i])
        state = f'frame {i:02d}; '+('FINAL TEST' if i in TEST else 'SELECTION' if i in SELECTION else 'training')
        board = np.concatenate([
            np.concatenate([panel(sources[i],'CYCLES SOURCE',state),
                panel(clean[i],'KNOWN CLEAN REFERENCE','nonperformer; withheld truth'),
                panel(estimated,'FITTED INTERIOR',str(selected)+'; NOT certified material')],1),
            np.concatenate([panel(target[i],'KNOWN TARGET REFERENCE','same interior, new environment'),
                panel(estimated+new[i],'FIT + ORACLE NEW REFLECTION','experimental; supplied geometry'),
                panel(sources[i]+new[i],'ADD-ONLY CONTROL','old reflection remains')],1)],0)
        board = cv2.copyMakeBorder(board,30,0,0,0,cv2.BORDER_CONSTANT)
        cv2.putText(board,'NONPERFORMER / fresh textured Cycles / conditional fit / no real repair',
            (8,21),0,.48,(0,220,255),1,cv2.LINE_AA)
        for repeat in range(3):
            write_image(out/'review'/f'{repeat*36+i:06d}.png',board)
    report.update(uncertainty=uncertainty,disabled_coupled_fallbacks=36,production_repairs=0,
        reference_subtraction=dict(new_min=float(min(x.min() for x in new.values())),
            negative_channels=sum(int((x<0).sum()) for x in new.values()),raw_differences_scored=True),
        references={str(r['frame']):r['files'] for r in manifest(shot)['records']},
        scope='Independent rendering and new textures; shared simplified anatomy, oracle geometry and new reflection')
    write_json(shot.logs/'r22_eyes.json',report)
    print('R22_EVALUATION_COMPLETE',selected,flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_shot_argument(ap)
    ap.add_argument('--fit-only',action='store_true')
    ap.add_argument('--evaluate-only',action='store_true')
    args = ap.parse_args()
    shot = from_args(args)
    if not args.evaluate_only:
        fit_stage(shot)
    if not args.fit_only:
        evaluate(shot)


if __name__=='__main__':
    main()
