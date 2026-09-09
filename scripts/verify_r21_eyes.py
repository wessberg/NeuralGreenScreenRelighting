"""Verify frozen eye fits, isolated inputs, renderer orientation and source fallback."""
import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

from audit_r21_eyes import (ARMS, ROOT, SELECTION, SIZE, TEST, TRAIN, choose, geometry,
    load_sources, masks, metric, predict, read_rgb, training_inputs)
from audit_r20_eyes import measures, sha, write_json
from eye_layer_fit import components
from verify_r20_eyes import transaction_checks, video_checks
from shot_paths import add_shot_argument, from_args


def scalar_grid(grid, uv, *, image_pixels=False):
    """Independent direct grid lookup, with separate image-texture centre convention."""
    size = len(grid)
    xy = np.asarray(uv)*size-.5 if image_pixels else np.asarray(uv)*(size-1)
    xy = np.clip(xy,0,size-1)
    lo = np.floor(xy).astype(int)
    hi = np.minimum(lo+1,size-1)
    u,v = np.moveaxis(xy-lo,-1,0)
    x,y = np.moveaxis(lo,-1,0)
    xx,yy = np.moveaxis(hi,-1,0)
    return (grid[y,x]*((1-u)*(1-v))[...,None]+grid[y,xx]*(u*(1-v))[...,None]
            +grid[yy,x]*((1-u)*v)[...,None]+grid[yy,xx]*(u*v)[...,None])


def check_fits(shot):
    sources,records = load_sources(shot)
    report = json.loads((shot.logs/'r21_eyes.json').read_text())
    preflight = json.loads((shot.logs/'r21_preflight.json').read_text())
    assert report['protocol_sha256']==preflight['protocol_sha256']==sha(ROOT/'docs/r21-eye-protocol.md')
    assert report['code_sha256']==preflight['code_sha256']
    assert all(sha(ROOT/'scripts'/k)==v for k,v in report['code_sha256'].items())
    assert not set(TRAIN)&set(SELECTION+TEST) and not set(SELECTION)&set(TEST)
    out = shot.composite/'r21_eyes'
    predictions,maximum = {},0.
    substitutions = 0
    for name,options in ARMS.items():
        data = np.load(out/f'{name}_fit.npz')
        values,active = data['values'],data['active']
        n = SIZE*SIZE
        split = len(values)-n
        assert values.min()>=0 and values[:split].max()<=1 and values[split:].max()<=8
        assert np.count_nonzero(values[~active])==0
        matrices,rgb = training_inputs(sources,options)
        matrix,observed = sparse.vstack(matrices,format='csr'),np.concatenate(rgb)
        assert np.array_equal(active,np.asarray(matrix.power(2).sum(0)).ravel()>0)
        info = report['arms'][name]
        assert info['observations']==len(observed) and info['active_parameters']==int(active.sum())
        assert abs(float(np.abs(matrix@values-observed).mean())-info['train_linear_mae'])<1e-14
        for color in ([0,0,0],[1,0,1]):
            altered = [np.full_like(s,color) if i in SELECTION+TEST else s for i,s in enumerate(sources)]
            altered_matrix,altered_rgb = training_inputs(altered,options)
            assert (sparse.vstack(altered_matrix,format='csr') != matrix).nnz == 0
            assert np.array_equal(np.concatenate(altered_rgb),observed)
            substitutions += 1
        class TrainingOnly:
            def __getitem__(self,index):
                assert index in TRAIN, 'Heldout RGB accessed during fitting'
                return sources[index]
        guarded_m,guarded_rgb = training_inputs(TrainingOnly(),options)
        assert np.array_equal(np.concatenate(guarded_rgb),observed)
        predictions[name] = predict(values,options)
        saved = np.load(out/f'{name}_predictions.npz')
        for i,pred in enumerate(predictions[name]):
            t,kwargs = geometry(i,**options)
            uv = t['interior_uv']
            interior = scalar_grid(values[:n].reshape(SIZE,SIZE,3),uv)
            if kwargs['split_pupil']:
                pupil_uv = (uv-.5)*kwargs['iris_radius']/kwargs['pupil_radius']+.5
                pupil = scalar_grid(values[n:2*n].reshape(SIZE,SIZE,3),pupil_uv)
                interior = np.where(t['pupil'][...,None],pupil,interior)
            interior *= (t['visible']*(1-t['fresnel']))[...,None]
            reflection = scalar_grid(values[split:].reshape(SIZE,SIZE,3),t['environment_uv'])
            reflection *= (t['fresnel']*t['reflection_visible'])[...,None]
            maximum = max(maximum,float(np.abs(interior-pred['clean']).max()),
                          float(np.abs(reflection-pred['old']).max()))
            assert np.array_equal(saved['clean'][i],pred['clean'])
            assert np.array_equal(saved['old'][i],pred['old'])
    assert maximum < 1e-12
    selected,scores = choose(predictions,sources)
    assert selected == report['selection']['selected'] and scores == report['selection']['source_sum_display_mae']
    for color in ([0,0,0],[1,0,1]):
        altered = [np.full_like(s,color) if i in TEST else s for i,s in enumerate(sources)]
        assert choose(predictions,altered)==(selected,scores)
    class SelectionOnly:
        def __getitem__(self,index):
            assert index in SELECTION, 'Final-test or training RGB accessed during selection'
            return sources[index]
    assert choose(predictions,SelectionOnly())==(selected,scores)
    # Poison both reference paths in the actual source-loader manifest. Any attempted
    # clean/target read or hash would now fail; source arrays and selection stay exact.
    from unittest.mock import patch
    manifest_path = shot.logs/'r21_cycles.json'
    manifest = json.loads(manifest_path.read_text())
    for record in manifest['records']:
        for key in ('clean','target'):
            record['files'][key] = dict(path='REFERENCE_RGB_MUST_NOT_BE_READ',sha256='poison')
    original_read = Path.read_text
    def poisoned_read(path,*args,**kwargs):
        return json.dumps(manifest) if path==manifest_path else original_read(path,*args,**kwargs)
    with patch.object(Path,'read_text',poisoned_read):
        isolated_sources,_ = load_sources(shot)
    assert all(np.array_equal(a,b) for a,b in zip(isolated_sources,sources))
    assert choose(predictions,isolated_sources)==(selected,scores)
    # References are loaded only here, after input isolation and selection checks.
    clean,targets = [],[]
    for record in records:
        for key,collection in (('clean',clean),('target',targets)):
            file = record['files'][key]
            assert sha(file['path'])==file['sha256']
            collection.append(read_rgb(file['path']))
    new = [a-b for a,b in zip(targets,clean)]
    for name,pred in predictions.items():
        estimated = [p['clean'] for p in pred]
        for region in ('interior','iris','pupil','full_visible'):
            actual = report['arms'][name]['test'][region]
            for key,est,truth in (
                ('sum_fit',[p['sum'] for p in pred],sources),
                ('interior',estimated,clean),
                ('old_reflection',[p['old'] for p in pred],[s-c for s,c in zip(sources,clean)]),
                ('coupled_known_scene',[a+b for a,b in zip(estimated,new)],targets)):
                assert metric(est,truth,TEST,region)==actual[key]
    assert all(np.count_nonzero(c)==0 for c in (sources[11],clean[11],targets[11]))
    pupil_values = np.concatenate([clean[i][masks(i)['pupil']] for i in TEST])
    assert pupil_values.min()>0, 'Nonblack-pupil challenge accidentally uses a black pupil'
    return dict(frame_arm_scalar_checks=240,scalar_max_error=maximum,
        heldout_rgb_substitutions=substitutions,training_access_guards=10,
        final_test_selection_substitutions=2,selection_access_guard=True,
        poisoned_reference_paths=48,
        reference_isolation='Poisoned clean/target paths preserve source-loader arrays and selection; exact fitter inputs, no repeated solves',
        closed_source_clean_target_exact_zero=True,clean_test_pupil_min=float(pupil_values.min()),
        supplied_geometry_and_oracle_masks=True,real_geometry_certified=False)


def check_renderer(shot):
    manifest = json.loads((shot.logs/'r21_cycles.json').read_text())
    assert abs(manifest['pixel_filter_width']-.01)<1e-8
    # Execute only the fixture's numeric texture definition, not Blender or any fitter.
    source = ROOT/'scripts/render_r21_eyes.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='textures')
    scope = {'np':np}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
    iris,pupil,old,new = scope['textures']()
    out = shot.composite/'r21_eyes/cycles'
    results = []
    for i in (0,5,6,11,12,17,18,23):
        t,kwargs = geometry(i)
        uv = t['interior_uv']
        interior = scalar_grid(iris,uv,image_pixels=True)
        pup = scalar_grid(pupil,(uv-.5)*5.85/1.8+.5,image_pixels=True)
        interior = np.where(t['pupil'][...,None],pup,interior)
        interior *= (t['visible']*(1-t['fresnel']))[...,None]
        old_image = scalar_grid(old,t['environment_uv'],image_pixels=True)
        old_image *= (t['fresnel']*t['reflection_visible'])[...,None]
        rendered_clean = read_rgb(out/f'{i:06d}_clean.exr')
        rendered_source = read_rgb(out/f'{i:06d}_source.exr')
        result = dict(frame=i,clean=measures(rendered_clean,interior,masks(i)['interior']),
                      source=measures(rendered_source,interior+old_image,masks(i)['interior']))
        # Inherited transport sanity bound, not a reconstruction acceptance threshold.
        for key in ('clean','source'):
            if result[key]['pixels']:
                assert result[key]['linear_mae'] < .01,result
        results.append(result)
    return dict(cases=results,texture_definition_sha256=sha(source),
        scope='512px physical UV and signed motion sanity; fixture texture is read only by this verifier')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_shot_argument(ap)
    ap.add_argument('--skip-video',action='store_true')
    args = ap.parse_args()
    shot = from_args(args)
    from verify_r15 import fallback
    result = dict(fits=check_fits(shot),renderer=check_renderer(shot),
                  transaction=transaction_checks(),source=fallback(shot))
    if not args.skip_video:
        result['video'] = video_checks(shot,'r21_eyes',96,24.,(762,960,3))
    result['verifier_sha256'] = sha(Path(__file__))
    write_json(shot.logs/'r21_verification.json',result)
    print('R21_VERIFICATION_COMPLETE',json.dumps(result['fits']),flush=True)


if __name__=='__main__':
    main()
