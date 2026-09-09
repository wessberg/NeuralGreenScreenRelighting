"""Reporter-only released highlight-head diagnostic; no neural performer RGB."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from audit_r14_neural import load_rows, SIDES, tile, write
from shot_paths import add_shot_argument, from_args
from unreflect_highlights import UnreflectHighlights, file_hash

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_shot_argument(ap)
    args = ap.parse_args()
    shot = from_args(args)
    if shot.dir != (ROOT/'music_videos/reporter_test/shots/reporter_a').resolve():
        raise SystemExit('Reporter-only audit; UnreflectHighlights.predict is an arbitrary RGB array API')
    out = shot.composite/'r19_unreflect'
    out.mkdir(parents=True, exist_ok=True)
    protocol = file_hash(ROOT/'docs/r19-unreflect-protocol.md')
    code = {n: file_hash(ROOT/'scripts'/n) for n in ('unreflect_highlights.py', 'audit_r19_unreflect.py')}
    labels_path = ROOT/'scripts/data/r14_native_labels.json'
    labels = json.loads(labels_path.read_text())
    rows = load_rows(shot, range(950, 1000))
    model = UnreflectHighlights()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    records, probes = [], []
    for frame in range(950, 1000):
        source_panels, map_panels = [], []
        for side in SIDES:
            rgb = rows[side][frame]['rgb']
            result = model.predict(rgb)
            assert all(np.isfinite(v).all() and v.min() >= 0 and v.max() <= 1 for v in result.values())
            hashes = {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in result.items()}
            repeat_exact = None
            if frame == 977:
                repeat = model.predict(rgb)
                repeat_exact = all(np.array_equal(result[k], repeat[k]) for k in result)
                assert repeat_exact
            np.savez_compressed(out/f'{frame:06d}_{side}.npz', **result)
            native = result['native_probability']
            records.append(dict(frame=frame, side=side, source_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                arrays_sha256=hashes, repeat_exact=repeat_exact,
                native_pixels_above_03=int((native > .3).sum()), max_probability=float(native.max())))
            label = next((v for v in labels['frames'] if v['frame'] == frame and v['side'] == side), None)
            if label:
                for category in ('iris', 'reflection'):
                    for i, (x, y) in enumerate(label[category]):
                        # Existing sparse anchors are integer native pixel centres.
                        assert int(x) == x and int(y) == y
                        p = float(native[int(y), int(x)])
                        probes.append(dict(frame=frame, side=side, category=category, index=i,
                                           xy=[x,y], probability=p, above_03=p > .3))
            source_panels.append(tile(rgb, f'{side} SOURCE {frame}', 'original observed RGB; no removal'))
            gray = np.repeat(native[..., None]*255, 3, axis=-1)
            map_panels.append(tile(gray, f'{side} HIGHLIGHT PROBABILITY', f'unvalidated; pixels >0.3: {(native > .3).sum()}'))
        board = np.concatenate(source_panels+map_panels, 1)
        board = cv2.copyMakeBorder(board, 32, 0, 0, 0, cv2.BORDER_CONSTANT)
        cv2.putText(board, 'UnReflectAnything RELEASED HIGHLIGHT HEAD | SOURCE LEFT | NO DIFFUSE RECONSTRUCTION / NO REPAIR', (8, 22), 0, .52, (255,255,255), 1)
        write(out/'review'/f'{frame-950:06d}.png', board)
        print('UnReflectAnything', frame, flush=True)
    report = dict(protocol_sha256=protocol, code_sha256=code, labels_sha256=file_hash(labels_path),
        model=model.provenance, records=records, probes=probes,
        seconds=time.perf_counter()-start, peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        probe_summary={c: dict(attempts=sum(p['category'] == c for p in probes),
                              above_03=sum(p['category'] == c and p['above_03'] for p in probes))
                       for c in ('iris', 'reflection')},
        closure_frames=[r for r in records if r['frame'] in (981,982)],
        production_repairs=0, diffuse_reconstruction_tested=False,
        instruments_tested=False, scene_reflection_allowed=False)
    assert protocol == file_hash(ROOT/'docs/r19-unreflect-protocol.md')
    assert code == {n: file_hash(ROOT/'scripts'/n) for n in code}
    (shot.logs/'r19_unreflect.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('seconds', 'peak_cuda_bytes', 'probe_summary')}, indent=2))


if __name__ == '__main__':
    main()
