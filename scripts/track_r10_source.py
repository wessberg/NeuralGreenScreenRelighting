"""Reporter-only full-source eye crops and landmarks, in the isolated facetrack venv.

Sequentially decode the original; never overwrite the selected shot's landmark store.
Use cached target landmarks for 700:850 so widening donors does not change target geometry.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
from face_track import MODEL, TOPOLOGY
from shot_paths import add_shot_argument, from_args


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_shot_argument(ap)
    shot = from_args(ap.parse_args())
    if shot.name != 'reporter_a' or shot.video_dir.name != 'reporter_test':
        raise SystemExit('Reporter-only diagnostic')
    out = shot.composite / 'r10_iris'
    crops = out / 'source_crops'
    crops.mkdir(parents=True, exist_ok=True)
    old = np.load(shot.relight / 'face_landmarks.npz')
    start = shot.config['selection']['start_frame']
    n = shot.config['source_frame_count']
    points = np.full((n, 478, 3), np.nan, np.float32)
    found = np.zeros(n, bool)
    origins = np.zeros((n, 2), np.int32)
    fresh_target_deltas = []
    cap = cv2.VideoCapture(str(shot.plate))
    fps = cap.get(cv2.CAP_PROP_FPS)
    opts = FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(MODEL)),
                                 running_mode=RunningMode.VIDEO, num_faces=1)
    count = 0
    with FaceLandmarker.create_from_options(opts) as tracker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if count >= n:
                raise RuntimeError('More source frames than metadata')
            h, w = frame.shape[:2]
            rgb = np.ascontiguousarray(frame[..., ::-1])
            result = tracker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                              round(count * 1000 / float(shot.frame_rate)))
            if result.face_landmarks:
                points[count] = [[p.x*w, p.y*h, p.z*w] for p in result.face_landmarks[0]]
                found[count] = True
            if start <= count < start + shot.frame_count:
                k = count - start
                if found[count] and old['found'][k]:
                    ids = TOPOLOGY['iris_left'] + TOPOLOGY['iris_right']
                    fresh_target_deltas.append(float(np.linalg.norm(
                        points[count, ids, :2] - old['points'][k, ids, :2], axis=-1).mean()))
                points[count] = old['points'][k]
                found[count] = old['found'][k]
            if found[count]:
                ids = TOPOLOGY['eyelid_left'] + TOPOLOGY['eyelid_right']
                eye = points[count, ids, :2]
                lo = np.maximum(np.floor(eye.min(0) - 30), 0).astype(int)
                hi = np.minimum(np.ceil(eye.max(0) + 30), [w, h]).astype(int)
                origins[count] = lo
                crop = frame[lo[1]:hi[1], lo[0]:hi[0]]
            else:
                crop = np.zeros((80, 240, 3), np.uint8)
            if not cv2.imwrite(str(crops / ('%06d.png' % count)), crop):
                raise RuntimeError('Crop write failed')
            count += 1
            if count % 100 == 0:
                print('decoded/tracked', count, flush=True)
    cap.release()
    if count != n or abs(fps - float(shot.frame_rate)) > .001:
        raise RuntimeError('Source count/rate mismatch')
    np.savez_compressed(out / 'source_landmarks.npz', points=points, found=found, origins=origins)
    info = {'decoded_frames': count, 'fps': fps, 'missing': np.flatnonzero(~found).tolist(),
            'source': str(shot.plate), 'topology': TOPOLOGY,
            'target_geometry': 'existing selected-shot landmarks retained',
            'outside_geometry': 'fresh sequential VIDEO landmarks on decoded original',
            'fresh_vs_cached_target_iris_mean_distance_p50_p95_px':
                np.percentile(fresh_target_deltas, [50, 95]).tolist(),
            'decode': 'OpenCV VideoCapture, 8-bit BGR; isolated diagnostic, no production ingest change'}
    (out / 'source_tracking.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
