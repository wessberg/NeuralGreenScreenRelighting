"""Per-frame facial scale, for the metric distance track.

Runs in the ISOLATED dependencies/facetrack/.venv, not the pipeline venv: mediapipe 1.x
requires opencv-contrib 5.x, and replacing the pipeline's opencv 4.13 would touch every
EXR read in the project. This writes JSON; scripts/metric_track.py consumes it.

    dependencies/facetrack/.venv/Scripts/python.exe scripts/face_track.py --shot <dir>

Why facial scale: at bust/medium framing there is no floor contact, so ray-plane
intersection through the feet is unavailable. Apparent facial scale is the reliable
monocular depth cue, and the RELATIVE irradiance ratio needs no calibration whatsoever:

    E(t)/E(t0) = (Z(t0)/Z(t))^2 = (w(t)/w(t0))^2

Absolute distance additionally needs the subject's true interocular distance, which is one
ruler measurement per performer, for life. Without it the adult prior (63 mm, SD 3.5 mm)
gives about 5.5 percent 1-sigma distance error, i.e. ~0.5 dB of irradiance error.
"""
import sys, json, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np, cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
from shot_paths import add_shot_argument, from_args   # stdlib-only, safe in this venv

MODEL = ROOT / 'dependencies/facetrack/models/face_landmarker.task'

# 478-point FaceMesh topology. Iris centres exist only in the refined (478) model.
IRIS_L, IRIS_R = 468, 473
# Outer eye corners, used as a fallback scale when iris points are absent.
CANTHUS_L, CANTHUS_R = 33, 263
# Tragion-ish extremes of the face oval, for a wider (yaw-sensitive) cross-check.
TRAGION_L, TRAGION_R = 234, 454


def to_uint8_rgb(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise SystemExit('Cannot read %s' % path)
    if bgr.dtype == np.uint16:
        bgr = (bgr / 257.0).round().astype(np.uint8)
    return np.ascontiguousarray(bgr[..., ::-1])


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_shot_argument(p)
    args = p.parse_args()
    shot = from_args(args)
    if not MODEL.exists():
        raise SystemExit('Missing %s' % MODEL)

    n = shot.frame_count
    fps = shot.frame_rate            # exact Fraction
    H = W = None
    opts = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True,
    )
    rows, missing = [], []
    with FaceLandmarker.create_from_options(opts) as lm:
        for i in range(n):
            rgb = to_uint8_rgb(shot.frame(i))
            if H is None:
                H, W = rgb.shape[:2]
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            # VIDEO mode requires strictly increasing integer milliseconds.
            ts = int(round(i * 1000 * fps.denominator / fps.numerator))
            res = lm.detect_for_video(image, ts)
            if not res.face_landmarks:
                missing.append(i)
                rows.append({'frame': i, 'found': False})
                continue
            L = res.face_landmarks[0]
            pts = np.array([[p_.x * W, p_.y * H] for p_ in L], np.float64)

            def dist(a, b):
                return float(np.hypot(*(pts[a] - pts[b]))) if max(a, b) < len(pts) else None

            row = {'frame': i, 'found': True, 'n_landmarks': len(pts),
                   'ipd_px': dist(IRIS_L, IRIS_R),
                   'canthal_px': dist(CANTHUS_L, CANTHUS_R),
                   'bitragion_px': dist(TRAGION_L, TRAGION_R),
                   'face_centroid_px': [float(pts[:, 0].mean()), float(pts[:, 1].mean())],
                   'bbox_px': [float(pts[:, 0].min()), float(pts[:, 1].min()),
                               float(pts[:, 0].max()), float(pts[:, 1].max())]}
            if res.facial_transformation_matrixes:
                M = np.array(res.facial_transformation_matrixes[0]).reshape(4, 4)
                # Yaw about the vertical axis; IPD foreshortens as cos(yaw).
                row['yaw_deg'] = float(np.degrees(np.arctan2(-M[2, 0], np.hypot(M[2, 1], M[2, 2]))))
                row['pitch_deg'] = float(np.degrees(np.arcsin(np.clip(M[2, 1], -1, 1))))
            rows.append(row)
            if i % 25 == 0:
                print(json.dumps({k: row[k] for k in ('frame', 'ipd_px', 'bitragion_px') if k in row}), flush=True)

    found = [r for r in rows if r.get('found')]
    if not found:
        raise SystemExit('No face detected in any frame; facial scale is unavailable for this shot')
    ipd = np.array([r['ipd_px'] for r in found], float)
    out = {'shot': shot.name, 'frames': n, 'detected': len(found), 'missing_frames': missing,
           'resolution': [W, H], 'source_frame_rate': str(fps),
           'landmark_model': 'mediapipe face_landmarker.task (478-point, iris refined)',
           'ipd_px_stats': {'min': float(ipd.min()), 'median': float(np.median(ipd)),
                            'max': float(ipd.max()),
                            'range_pct_of_median': float(100 * (ipd.max() - ipd.min()) / np.median(ipd))},
           'relative_irradiance_note': 'E(t)/E(t0) = (ipd_px(t)/ipd_px(t0))**2 needs no calibration.',
           'per_frame': rows}
    shot.logs.mkdir(parents=True, exist_ok=True)
    shot.relight.mkdir(parents=True, exist_ok=True)
    (shot.relight / 'face_track.json').write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ('shot', 'frames', 'detected', 'ipd_px_stats')}, indent=2))
    if missing:
        print('WARNING: no face on %d frame(s): %s' % (len(missing), missing[:20]), file=sys.stderr)


if __name__ == '__main__':
    main()
