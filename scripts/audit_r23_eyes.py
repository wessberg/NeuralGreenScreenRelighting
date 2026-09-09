"""Fresh Cycles residual-preservation and optical-geometry sensitivity diagnostic."""
import argparse
import ast
from functools import lru_cache
import json
import os
from pathlib import Path
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from audit_r20_eyes import measures, panel, sha, write_image, write_json
from audit_r21_eyes import read_rgb
from corneal_transport import EyeOptics
from finite_cornea_transport import trace_fixture as trace
from coupled_eye_relight import apply_coupled
import eye_detail_fit as detail
from eye_residual import preserve_residual
from shot_paths import add_shot_argument, from_args
from verify_r21_eyes import scalar_grid

ROOT = Path(__file__).resolve().parents[1]
FRAMES = tuple(range(24))
SELECT, TEST = (2,8,14,20), (5,11,17,23)
TRAIN = tuple(i for i in FRAMES if i not in SELECT+TEST)
ARMS = ('nominal','cornea','pupil','lid')
OPTIONS = dict(iris_size=64,pupil_size=12,environment_size=32)
SPLIT = 64**2+12**2
CODE = ('audit_r23_eyes.py','eye_residual.py','eye_detail_fit.py','corneal_transport.py',
        'render_r23_eyes.py','render_r20_transport.py','observation_eye_fit.py','coupled_eye_relight.py',
        'finite_cornea_transport.py')


def geometry(frame, arm='nominal', n=8, corners=False):
    phase = 2*np.pi*frame/24+.47
    eye = EyeOptics(yaw=np.deg2rad(13*np.sin(phase)),pitch=np.deg2rad(7*np.cos(phase)),
        radius=8.1 if arm=='cornea' else 7.8,pupil_radius=2. if arm=='pupil' else 1.8)
    height = max(0.,4*(1-np.exp(-((frame-17)/1.4)**2))-(.15 if arm=='lid' else 0))
    y,x = np.mgrid[:64,:64]
    xy = np.stack([x,y],-1)
    offsets = (np.arange(n)+.5)/n-.5
    points = [(dx,dy) for dx in offsets for dy in offsets]
    if corners: points += [(dx,dy) for dx in (-.5,.5) for dy in (-.5,.5)]
    for dx,dy in points:
        yield trace(xy+[dx,dy],eye,lid_height=height)


@lru_cache(maxsize=24)
def masks(frame):
    visible,iris,pupil = np.zeros((64,64),bool),np.ones((64,64),bool),np.ones((64,64),bool)
    for t in geometry(frame,corners=True):
        visible |= t['visible']; iris &= t['iris']; pupil &= t['pupil']
    return dict(full_visible=visible,iris=iris,pupil=pupil,mixed=visible&~(iris|pupil))


@lru_cache(maxsize=96)
def matrix(frame,arm='nominal'):
    # Accumulate one subray at a time to avoid retaining gigabytes of trace arrays.
    result = None
    for t in geometry(frame,arm):
        m = detail.design([t],**OPTIONS,pupil_radius=2. if arm=='pupil' else 1.8)
        result = m if result is None else result+m
    result = (result/64).tocsr(); result.eliminate_zeros()
    return result


def manifest(shot):
    data = json.loads((shot.logs/'r23_cycles.json').read_text())
    assert data['protocol_sha256']==sha(ROOT/'docs/r23-eye-protocol.md')
    assert all(sha(ROOT/'scripts'/k)==v for k,v in data['code_sha256'].items())
    assert [r['frame'] for r in data['records']]==list(FRAMES)
    assert data['pixel_filter_type']=='BOX' and data['pixel_filter_width']==1.
    return data


def load(shot,frames,kind='source'):
    records = manifest(shot)['records']; result = {}
    for i in frames:
        f = records[i]['files'][kind]
        assert sha(f['path'])==f['sha256']
        result[i] = read_rgb(f['path'])
    return result


def inputs(sources,arm):
    return ([matrix(i,arm)[masks(i)['full_visible'].ravel()] for i in TRAIN],
            [sources[i][masks(i)['full_visible']] for i in TRAIN])


def metric(a,b,frames=TEST,region='full_visible'):
    x = np.concatenate([a[i][masks(i)[region]] for i in frames])
    y = np.concatenate([b[i][masks(i)[region]] for i in frames])
    return measures(x,y,np.ones(len(x),bool))


def direct(values,frame,arm='nominal',n=8,texture=False):
    sizes = (512,512,512) if texture else (64,12,32)
    a,b = sizes[0]**2,sum(s*s for s in sizes[:2])
    grids = [values[:a].reshape(sizes[0],sizes[0],3),
             values[a:b].reshape(sizes[1],sizes[1],3),values[b:].reshape(sizes[2],sizes[2],3)]
    clean,old = np.zeros((64,64,3)),np.zeros((64,64,3))
    for t in geometry(frame,arm,n):
        uv = t['interior_uv']
        base = scalar_grid(grids[0],uv,image_pixels=texture)
        pup = scalar_grid(grids[1],(uv-.5)*5.85/(2. if arm=='pupil' else 1.8)+.5,image_pixels=texture)
        clean += np.where(t['pupil'][...,None],pup,base)*(t['visible']*(1-t['fresnel']))[...,None]/n**2
        old += scalar_grid(grids[2],t['environment_uv'],image_pixels=texture)*(
            t['fresnel']*t['reflection_visible'])[...,None]/n**2
    return clean,old


def transport(shot):
    # Only training frame 0 clean truth is allowed for this pre-fit renderer check.
    path = ROOT/'scripts/render_r23_eyes.py'
    node = next(n for n in ast.parse(path.read_text(encoding='utf-8-sig')).body
                if isinstance(n,ast.FunctionDef) and n.name=='textures')
    scope = {'np':np}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    values = np.concatenate([a.reshape(-1,3) for a in scope['textures']()[:3]])
    clean,source = load(shot,[0],'clean'),load(shot,[0])
    results = {}; predictions = {}
    for n in (8,32):
        c,r = direct(values,0,n=n,texture=True); predictions[n]=(c,r)
        results[str(n)] = {region:dict(clean=metric({0:c},clean,[0],region),
                source=metric({0:c+r},source,[0],region)) for region in masks(0)}
    results['quadrature_max_difference'] = max(float(np.abs(a-b).max())
        for a,b in zip(predictions[8],predictions[32]))
    results['scope'] = 'Training frame 0 only; finite integration and Monte Carlo error, not camera PSF validation'
    write_json(shot.logs/'r23_transport_pretest.json',results)
    print('R23_TRANSPORT',json.dumps(results),flush=True)


def feasibility(m,b,upper,out):
    constraint = sparse.vstack([m,-m],format='csr')
    rhs = np.r_[b+.002,-b+.002]
    fixed = linprog(np.zeros(m.shape[1]),A_ub=constraint,b_ub=rhs,
        bounds=list(zip(np.zeros(len(upper)),upper)),method='highs')
    # A global minimax discrepancy distinguishes infeasible noise-scale model from
    # broad intervals. It is NOT a replacement noise estimate for every pixel.
    augmented = sparse.hstack([constraint,-np.ones((2*len(b),1))],format='csr')
    objective = np.r_[np.zeros(len(upper)),1.]
    result = linprog(objective,A_ub=augmented,b_ub=np.r_[b,-b],
        bounds=list(zip(np.zeros(len(upper)),upper))+[(0,None)],method='highs')
    info = dict(fixed_radius=.002,fixed_success=bool(fixed.success),fixed_status=int(fixed.status),
                fixed_message=fixed.message,minimax_success=bool(result.success),minimax_message=result.message)
    if result.success:
        radius = float(result.x[-1]); witness=result.x[:-1]
        violation = max(float(np.max(np.abs(m@witness-b)-radius)),
                        float(-witness.min()),float(np.max(witness-upper)),0.)
        assert violation<1e-6
        info.update(minimum_uniform_radius=radius,witness_violation=violation)
        np.savez_compressed(out/'minimax_witness.npz',values=witness,radius=radius)
    return info


def fit_stage(shot):
    out = shot.composite/'r23_eyes'; out.mkdir(parents=True,exist_ok=True)
    sources=load(shot,TRAIN)
    report=dict(protocol_sha256=sha(ROOT/'docs/r23-eye-protocol.md'),
        amendment_sha256=sha(ROOT/'docs/r23-transport-correction.md'),
        code_sha256={k:sha(ROOT/'scripts'/k) for k in CODE},train=TRAIN,selection=SELECT,test=TEST,arms={})
    write_json(shot.logs/'r23_preflight.json',report)
    for arm in ARMS:
        matrices,rgb=inputs(sources,arm)
        values,counts,info=detail.fit(matrices,rgb,detail.upper_bounds(**OPTIONS))
        clean,old,support=[],[],[]
        for i in FRAMES:
            m=matrix(i,arm)
            clean.append((m[:,:SPLIT]@values[:SPLIT]).reshape(64,64,3))
            old.append((m[:,SPLIT:]@values[SPLIT:]).reshape(64,64,3))
            support.append(detail.complete_support(m,counts).reshape(64,64))
        np.savez_compressed(out/f'{arm}.npz',values=values,counts=counts,clean=clean,old=old,support=support)
        info['selection_source']=metric(np.array(clean)+np.array(old),load(shot,SELECT),SELECT)
        report['arms'][arm]=info
        print('R23_FIT',arm,json.dumps(info),flush=True)
    matrices,rgb=inputs(sources,'nominal')
    report['feasibility']=feasibility(sparse.vstack(matrices,format='csr'),np.concatenate(rgb)[:,1],
                                     detail.upper_bounds(**OPTIONS),out)
    report['selection_policy']='All fixed sensitivity arms reported; none certifies material'
    write_json(shot.logs/'r23_selection.json',report)
    print('R23_FIT_COMPLETE',json.dumps(report['feasibility']),flush=True)


def temporal_error(estimate,truth):
    # Error change in native pixels, not material-aligned flicker or real motion.
    values=[]
    for i in range(1,24):
        mask=masks(i)['full_visible']&masks(i-1)['full_visible']
        values.append(((estimate[i]-truth[i])-(estimate[i-1]-truth[i-1]))[mask])
    a=np.concatenate(values)
    return dict(mean_absolute_linear_error_change=float(np.abs(a).mean()),pixels=len(a),
                scope='Seen full chronological sequence; screen-space error change, not tracked material flicker')


def evaluate(shot):
    out=shot.composite/'r23_eyes'
    report=json.loads((shot.logs/'r23_selection.json').read_text())
    assert report['protocol_sha256']==sha(ROOT/'docs/r23-eye-protocol.md')
    assert report['amendment_sha256']==sha(ROOT/'docs/r23-transport-correction.md')
    assert all(sha(ROOT/'scripts'/k)==v for k,v in report['code_sha256'].items())
    sources,clean,target=[load(shot,FRAMES,k) for k in ('source','clean','target')]
    new={i:target[i]-clean[i] for i in FRAMES}
    report['controls']={region:dict(source=metric(sources,clean,region=region),
        add_only=metric({i:sources[i]+new[i] for i in FRAMES},target,region=region)) for region in masks(0)}
    nominal=None
    for arm in ARMS:
        p=dict(np.load(out/f'{arm}.npz'))
        residual={i:preserve_residual(sources[i],p['clean'][i],p['old'][i])[0] for i in FRAMES}
        fitted_coupled={i:p['clean'][i]+new[i] for i in FRAMES}
        residual_coupled={i:residual[i]+new[i] for i in FRAMES}
        report['arms'][arm]['test']={region:dict(
            fitted_clean=metric(p['clean'],clean,region=region),residual_clean=metric(residual,clean,region=region),
            fitted_coupled=metric(fitted_coupled,target,region=region),
            residual_coupled=metric(residual_coupled,target,region=region),
            negative_residual_channels=sum(int((residual[i][masks(i)[region]]<0).sum()) for i in TEST),
            complete_support=sum(int((p['support'][i]&masks(i)[region]).sum()) for i in TEST))
            for region in masks(0)}
        report['arms'][arm]['temporal']=dict(fitted=temporal_error(p['clean'],clean),
                                                    residual=temporal_error(residual,clean))
        if arm=='nominal': nominal=(p,residual)
    p,residual=nominal
    for i in FRAMES:
        result,_=apply_coupled(sources[i],masks(i)['full_visible'],residual[i],new[i],
                              support=p['support'][i],geometry_reliable=False)
        assert np.array_equal(result,sources[i])
        state=f'frame {i:02d}; '+('FINAL TEST' if i in TEST else 'SELECTION' if i in SELECT else 'TRAIN')
        board=np.concatenate([
            np.concatenate([panel(sources[i],'SOURCE CONTROL',state),panel(clean[i],'CLEAN REFERENCE','nonperformer truth'),
                            panel(p['clean'][i],'FITTED CLEAN','nominal supplied geometry')],1),
            np.concatenate([panel(residual[i],'SOURCE - FITTED REFLECTION','raw subtraction; NOT certified'),
                            panel(target[i],'TARGET REFERENCE','same interior; new lighting'),
                            panel(residual[i]+new[i],'RESIDUAL + ORACLE REFLECTION','coupled experiment; NOT production')],1)],0)
        board=cv2.copyMakeBorder(board,30,0,0,0,cv2.BORDER_CONSTANT)
        cv2.putText(board,'R23 NONPERFORMER / 1px BOX / observed residual is NOT a material certificate',
                    (8,21),0,.46,(0,220,255),1,cv2.LINE_AA)
        for repeat in range(3): write_image(out/'review'/f'{repeat*24+i:06d}.png',board)
    report.update(production_repairs=0,disabled_coupled_fallbacks=24,
                  oracle_new_min=float(min(x.min() for x in new.values())),
                  raw_negative_values_retained=True,scope='Simplified geometry sensitivity; no inferred scene or real repair')
    write_json(shot.logs/'r23_eyes.json',report)
    print('R23_EVALUATION_COMPLETE',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__); add_shot_argument(ap)
    ap.add_argument('--transport-only',action='store_true'); ap.add_argument('--fit-only',action='store_true')
    ap.add_argument('--evaluate-only',action='store_true'); args=ap.parse_args(); shot=from_args(args)
    if args.transport_only: transport(shot)
    elif args.fit_only: fit_stage(shot)
    elif args.evaluate_only: evaluate(shot)
    else: transport(shot); fit_stage(shot); evaluate(shot)


if __name__=='__main__': main()
