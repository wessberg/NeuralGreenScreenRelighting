"""Independent saved-map checks and full-model parity on an author's sample."""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
import unreflectanything.model_ as model_module
import models
import token_inpainters

from audit_r14_neural import load_rows, write
from shot_paths import add_shot_argument, from_args
from unreflect_highlights import UnreflectHighlights, file_hash

ROOT = Path(__file__).resolve().parents[1]
REVISION = '363ebb5c87d9f4f49b63f744fded8d3786a051b3'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_shot_argument(ap)
    ap.add_argument('--integration-image', type=Path, help='Optional ordinary object/scene image for a fresh full-model parity check')
    args = ap.parse_args()
    shot = from_args(args)
    out = shot.composite/'r19_unreflect'
    report = json.loads((shot.logs/'r19_unreflect.json').read_text())
    assert report['protocol_sha256'] == file_hash(ROOT/'docs/r19-unreflect-protocol.md')
    for name, value in report['code_sha256'].items():
        assert file_hash(ROOT/'scripts'/name) == value
    assert report['labels_sha256'] == file_hash(ROOT/'scripts/data/r14_native_labels.json')
    checkout = ROOT/'dependencies/UnReflectAnything'
    assert subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip() == REVISION
    assert not subprocess.check_output(['git', '-C', str(checkout), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    installed = {}
    for name, module in (('models.py', models), ('token_inpainters.py', token_inpainters), ('endpoints/model_.py', model_module)):
        installed[name] = file_hash(Path(inspect.getfile(module)))
        assert installed[name] == file_hash(checkout/name)
    rows = load_rows(shot, range(950,1000))
    assert len(report['records']) == 100
    assert {(r['frame'],r['side']) for r in report['records']} == {(f,s) for f in range(950,1000) for s in ('left','right')}
    map_error = 0.
    native_maps = {}
    for rec in report['records']:
        frame, side = rec['frame'], rec['side']
        assert hashlib.sha256(rows[side][frame]['rgb'].tobytes()).hexdigest() == rec['source_sha256']
        with np.load(out/f'{frame:06d}_{side}.npz') as data:
            assert {k:hashlib.sha256(data[k].tobytes()).hexdigest() for k in data.files} == rec['arrays_sha256']
            raw, native = data['model_probability'], data['native_probability']
            assert raw.shape == (896,896) and native.shape == (64,64)
            # Half-pixel downsampling 896->64 samples at (14*x+6.5,14*y+6.5).
            a = 14*np.arange(64)+6
            expected = (raw[a[:,None],a] + raw[(a+1)[:,None],a] +
                        raw[a[:,None],a+1] + raw[(a+1)[:,None],a+1])/4
            map_error = max(map_error, float(np.max(np.abs(native-expected))))
            assert map_error < 1e-7
            assert rec['native_pixels_above_03'] == int((native > .3).sum())
            assert rec['max_probability'] == float(native.max())
            assert rec['repeat_exact'] == (True if frame == 977 else None)
            native_maps[frame,side] = native
    for probe in report['probes']:
        x,y = probe['xy']
        p = float(native_maps[probe['frame'],probe['side']][y,x])
        assert p == probe['probability'] and probe['above_03'] == (p > .3)
    for category,count in (('iris',30),('reflection',12)):
        probes = [p for p in report['probes'] if p['category'] == category]
        assert len(probes) == count
        assert report['probe_summary'][category] == dict(attempts=count, above_03=sum(p['above_03'] for p in probes))
    # A fresh integration check is explicit; ordinary scenes/objects are the
    # preferred inputs. Preserve the initial check as historical evidence.
    previous_path = shot.logs/'r19_unreflect_verification.json'
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    sample_check = previous.get('author_sample')
    if args.integration_image:
        sample = args.integration_image.resolve()
        rgb = cv2.imread(str(sample))[...,::-1].copy().astype(np.float32)
        model = UnreflectHighlights()
        assert model.provenance == report['model']
        head = model.predict(rgb)['model_probability']
        with torch.inference_mode():
            x = torch.from_numpy(rgb).permute(2,0,1)[None].cuda()/255
            full = model.inner({'rgb':x, 'inpaint_mask_threshold':.3})
        assert np.array_equal(head, full['highlight'][0,0].cpu().numpy())
        assert torch.isfinite(full['diffuse']).all()
        sample_check = dict(source=str(sample), sha256=file_hash(sample),
            full_forward_finite=True, full_diffuse_shape=list(full['diffuse'].shape),
            highlight_head_exact=True, highlight_max=float(head.max()),
            highlight_pixels_above_03=int((head > .3).sum()),
            full_diffuse_saved=False, scope='Explicit integration image; no performance score')
    cap = cv2.VideoCapture(str(shot.final/'r19_unreflect.mp4'))
    fps, thumbs = cap.get(cv2.CAP_PROP_FPS), []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i = len(thumbs)
        assert frame.shape == (400,1280,3)
        if i in (0,9,27,31,32,45,49):
            write(out/f'decoded_{i:03d}.png',frame)
        thumb = cv2.resize(frame,(384,120))
        cv2.putText(thumb,str(i+950),(3,12),0,.4,(255,255,255),1)
        thumbs.append(thumb)
    cap.release()
    assert len(thumbs) == 50 and abs(fps-float(shot.frame_rate)) < .001
    for start in (0,25):
        sheet = np.concatenate([np.concatenate(thumbs[i:i+5],1) for i in range(start,start+25,5)],0)
        write(out/f'contact_{start:03d}.jpg',sheet)
    result = dict(cases=100, independently_checked_native_resize_max_error=map_error,
        probes=42, exact_repeat_cases=2, installed_source_sha256=installed,
        author_code_revision=REVISION, model=report['model'], author_sample=sample_check,
        fresh_integration_check=bool(args.integration_image),
        packages={n:importlib.metadata.version(n) for n in ('torch','torchvision','transformers','unreflectanything','numpy','opencv-python')},
        video=dict(frames=50,fps=fps,sha256=file_hash(shot.final/'r19_unreflect.mp4'),artist_approval=False),
        code_sha256=file_hash(Path(__file__)))
    (shot.logs/'r19_unreflect_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'model'},indent=2))


if __name__ == '__main__':
    main()
