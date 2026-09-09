"""Frozen synthetic inverse-eye experiment and real exact-source fallback review."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from color_pipeline import linear_to_srgb
from corneal_transport import EyeOptics, trace
from coupled_eye_relight import apply_coupled
from observation_eye_fit import components, design, fit_observations
from shot_paths import add_shot_argument, from_args

ROOT = Path(__file__).resolve().parents[1]
SIZE = 16
HELDOUT = tuple(range(5, 36, 6))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')


def write_image(path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), rgb):
        raise IOError(path)


def grids():
    y, x = np.mgrid[-1:1:complex(SIZE), -1:1:complex(SIZE)]
    radius, angle = np.hypot(x, y), np.arctan2(y, x)
    pattern = .5+.25*np.sin(19*angle+7*radius)+.15*np.cos(37*angle-9*radius)
    iris = np.stack([.035+.055*pattern, .065+.11*pattern, .08+.15*pattern], -1)
    source = np.zeros_like(iris)+.08
    source[(x > -.6) & (x < -.05) & (y > -.5) & (y < .1)] = [5., 5., 5.]
    source[(x > .25) & (y > .15) & (y < .6)] = [1.5, 1.1, .7]
    target = np.zeros_like(iris)+.04
    target[(x > .1) & (x < .65) & (y > -.3) & (y < .3)] = [5., 2.8, 1.2]
    target[(x < -.3) & (y > .2) & (y < .65)] = [.8, 1.6, 3.5]
    return iris.reshape(-1, 3), source.reshape(-1, 3), target.reshape(-1, 3)


def synthesize():
    """Same-basis positive control, explicitly not an independent renderer benchmark."""
    yy, xx = np.mgrid[:64, :64]
    xy = np.stack([xx, yy], -1)
    iris, source_environment, target_environment = grids()
    source_values = np.concatenate([iris, source_environment])
    target_values = np.concatenate([iris, target_environment])
    cases = []
    for frame in range(36):
        phase = 2*np.pi*frame/36
        eye = EyeOptics(yaw=np.deg2rad(12)*np.sin(phase), pitch=np.deg2rad(8)*np.cos(phase))
        lid_height = 4*(1-np.exp(-((frame-17)/1.8)**2))
        transport = trace(xy, eye, lid_height=lid_height)
        clean, old = components(transport, source_values, SIZE, known_black_pupil=True)
        _, new = components(transport, target_values, SIZE, known_black_pupil=True)
        visible = transport['visible']
        score_mask = cv2.erode(visible.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        score_mask &= np.abs(transport['interior_radius']-eye.pupil_radius) > .7
        cases.append(dict(frame=frame, transport=transport, clean=clean, old=old, new=new,
                          source=clean+old, target=clean+new, score_mask=score_mask,
                          lid_height=lid_height))
    return cases


def measures(estimate, truth, mask):
    if not mask.any():
        return dict(pixels=0, linear_mae=None, display_mae=None, display_p95=None)
    error = np.abs(estimate[mask]-truth[mask])
    display = np.abs(linear_to_srgb(estimate[mask])-linear_to_srgb(truth[mask]))*255
    return dict(pixels=int(mask.sum()), linear_mae=float(error.mean()),
                display_mae=float(display.mean()), display_p95=float(np.percentile(display, 95)))


def training_inputs(cases, known_pupil):
    """Select whole training frames before reading any RGB; no target RGB is input."""
    matrices, observations = [], []
    for case in cases:
        if case['frame'] in HELDOUT:
            continue
        t = case['transport']
        select = t['visible'].ravel()
        matrices.append(design(t, SIZE, known_black_pupil=known_pupil)[select])
        observations.append(case['source'].reshape(-1, 3)[select])
    return matrices, observations


def panel(rgb, title, subtitle='', size=320):
    rgb = np.clip(linear_to_srgb(rgb)*255, 0, 255).astype(np.uint8)
    image = cv2.resize(rgb[..., ::-1], (size, size), interpolation=cv2.INTER_NEAREST)
    image = cv2.copyMakeBorder(image, 46, 0, 0, 0, cv2.BORDER_CONSTANT)
    cv2.putText(image, title, (5, 18), 0, .43, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, subtitle, (5, 38), 0, .36, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def prepare(shot):
    cases = synthesize()
    out = shot.composite/'r20_eyes'
    manifest = dict(protocol_sha256=sha(ROOT/'docs/r20-eye-protocol.md'),
        source='procedural nonperformer; exact bilinear two-layer basis', frames=36,
        heldout_frames=HELDOUT, samples={}, source_arrays={})
    for case in cases:
        i, t = case['frame'], case['transport']
        arrays = {k:case[k] for k in ('source', 'clean', 'old', 'new', 'target', 'score_mask')}
        arrays.update({f'transport_{k}':v for k,v in t.items()})
        path = out/'observations'/f'{i:06d}.npz'
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        manifest['source_arrays'][str(i)] = {k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()}
        manifest['samples'][str(i)] = int(t['visible'].sum())
    manifest['code_sha256'] = {name:sha(ROOT/'scripts'/name) for name in
        ('corneal_transport.py', 'observation_eye_fit.py', 'coupled_eye_relight.py', 'audit_r20_eyes.py')}
    write_json(shot.logs/'r20_eye_preflight.json', manifest)
    print('Prepared', sum(manifest['samples'].values()), 'synthetic eye observations; no fitting', flush=True)


def evaluate(shot):
    out = shot.composite/'r20_eyes'
    preflight = json.loads((shot.logs/'r20_eye_preflight.json').read_text())
    assert preflight['protocol_sha256'] == sha(ROOT/'docs/r20-eye-protocol.md')
    assert all(sha(ROOT/'scripts'/name) == value for name,value in preflight['code_sha256'].items())
    cases = []
    for i in range(36):
        with np.load(out/'observations'/f'{i:06d}.npz') as z:
            data = dict(z)
        assert {k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in data.items()} == preflight['source_arrays'][str(i)]
        transport = {k[len('transport_'):]:v for k,v in data.items() if k.startswith('transport_')}
        cases.append(dict(frame=i, transport=transport, **{k:v for k,v in data.items() if not k.startswith('transport_')}))
    result = dict(protocol_sha256=preflight['protocol_sha256'], code_sha256=preflight['code_sha256'],
        scope='Synthetic exact-basis experiment; no real repair or confidence certification', arms={})
    predictions = {}
    for name, known_pupil in (('unknown_pupil', False), ('oracle_black_pupil', True)):
        matrices, observations = training_inputs(cases, known_pupil)
        start = time.perf_counter()
        fitted = fit_observations(matrices, observations, size=SIZE)
        elapsed = time.perf_counter()-start
        np.savez_compressed(out/f'{name}_fit.npz', values=fitted['values'], active=fitted['active'])
        summaries = []
        predictions[name] = []
        for case in cases:
            t = case['transport']
            clean, reflected = components(t, fitted['values'], SIZE, known_black_pupil=known_pupil)
            predictions[name].append(clean)
            mask = case['score_mask']
            summaries.append(dict(frame=case['frame'], heldout=case['frame'] in HELDOUT,
                sum_fit=measures(clean+reflected, case['source'], mask),
                interior=measures(clean, case['clean'], mask),
                source_as_interior=measures(case['source'], case['clean'], mask),
                target_with_known_scene=measures(clean+case['new'], case['target'], mask),
                add_only=measures(case['source']+case['new'], case['target'], mask)))
        aggregate = {}
        for key in ('sum_fit', 'interior', 'source_as_interior', 'target_with_known_scene', 'add_only'):
            selected = [r[key] for r in summaries if r['heldout'] and r[key]['pixels']]
            count = sum(r['pixels'] for r in selected)
            aggregate[key] = dict(pixels=count,
                linear_mae=sum(r['linear_mae']*r['pixels'] for r in selected)/count,
                display_mae=sum(r['display_mae']*r['pixels'] for r in selected)/count)
        result['arms'][name] = {k:v for k,v in fitted.items() if k not in ('values', 'active')}
        result['arms'][name].update(seconds=elapsed, records=summaries, heldout=aggregate)
        print(name, 'seconds', round(elapsed, 2), 'heldout', aggregate, 'solver', fitted['solver'], flush=True)
    # Render all frames in chronological order, looped four times, with conditions named.
    for case in cases:
        i, t = case['frame'], case['transport']
        state = f'frame {i:02d}; '+('HELD OUT' if i in HELDOUT else 'training view')
        known = predictions['oracle_black_pupil'][i]
        unknown = predictions['unknown_pupil'][i]
        # The transaction gets oracle truth solely to exercise its algebra. The fitted
        # estimates above have not earned reliability and are displayed as experiments.
        oracle, info = apply_coupled(case['source'], t['visible'], case['clean'], case['new'],
            support=t['visible'], geometry_reliable=True, correspondence_reliable=True,
            extent_reliable=True, scene_visibility_reliable=True)
        assert np.allclose(oracle, case['target'], atol=1e-15)
        board = np.concatenate([
            np.concatenate([panel(case['source'], 'SYNTHETIC SOURCE', state),
                panel(unknown, 'FITTED INTERIOR', 'unknown pupil; NOT certified'),
                panel(known, 'FITTED INTERIOR', 'ORACLE black-pupil assumption')], 1),
            np.concatenate([panel(case['target'], 'KNOWN TARGET TRANSPORT', 'reference geometry + new scene'),
                panel(unknown+case['new'], 'FIT + KNOWN SCENE', 'experimental; geometry is supplied'),
                panel(case['source']+case['new'], 'ADD-ONLY CONTROL', 'old reflection still present')], 1)], 0)
        board = cv2.copyMakeBorder(board, 30, 0, 0, 0, cv2.BORDER_CONSTANT)
        cv2.putText(board, 'CONTROLLED MODEL: procedural texture, one interface, planar lids; NOT a real eye repair',
                    (8, 21), 0, .47, (0, 220, 255), 1, cv2.LINE_AA)
        for repeat in range(4):
            write_image(out/'review'/f'{repeat*36+i:06d}.png', board)
    result['synthetic_channels_above_one'] = sum(int((c['source'] > 1).sum()) for c in cases)
    result['production_repair_enabled'] = False
    write_json(shot.logs/'r20_eyes.json', result)


def real_review(shot):
    from audit_r14_neural import load_rows
    from color_pipeline import srgb_to_linear
    rows = load_rows(shot, range(950, 1000))
    out = shot.composite/'r20_source_fallback'
    hashes = {}
    for i in range(950, 1000):
        tiles = []
        for side in ('left', 'right'):
            source = rows[side][i]['rgb']/255
            output, info = apply_coupled(source, np.ones(source.shape[:2], bool))
            assert np.array_equal(source, output) and info['changed_channels'] == 0
            hashes[f'{i}_{side}'] = hashlib.sha256(source.tobytes()).hexdigest()
            tiles.append(np.concatenate([panel(srgb_to_linear(source), f'{side} SOURCE {i}', 'original catchlights'),
                panel(srgb_to_linear(output), 'EXACT SOURCE FALLBACK', 'no certified removal / no new reflection')], 1))
        write_image(out/'review'/f'{i-950:06d}.png', np.concatenate(tiles, 0))
    write_json(shot.logs/'r20_real_fallback.json', dict(cases=100, hashes=hashes,
        source='round-10 native display crops for review; keyed-EXR validation is separate',
        changed_channels=0, real_repairs=0))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_shot_argument(ap)
    ap.add_argument('--prepare-only', action='store_true')
    ap.add_argument('--real-review-only', action='store_true')
    args = ap.parse_args()
    shot = from_args(args)
    if args.prepare_only:
        prepare(shot)
    elif args.real_review_only:
        real_review(shot)
    else:
        evaluate(shot)
        real_review(shot)


if __name__ == '__main__':
    main()
