"""Per-frame facial scale AND the full 478-point landmark store.

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

**Round 5 additionally stores every landmark.** Rounds 1-4 computed all 478 points, used two of
them for an interocular distance and threw the rest away, so anything needing an eye, an eyelid
or a lip outline had to re-run mediapipe in a second venv. The store is a sidecar --
``relight/face_landmarks.npz`` plus a ``.json`` describing it -- and ``face_track.json`` keeps
its exact previous schema, so the existing IPD consumers are untouched.

Three things the store records that the caller cannot recover afterwards:

* **an explicit missing state.** ``found[i]`` is False and that frame's points are NaN. A frame
  with no detection must not silently inherit its neighbour's landmarks; a catchlight placed
  from a stale landmark is worse than no catchlight.
* **eye visibility.** ``blink`` is mediapipe's eyeBlink blendshape per eye, and the eyelid
  polygon's own aperture in iris radii is stored beside it, because the blendshape saturates
  late. Anything drawn on the eye must be gated by both.
* **the coordinate convention**, in the JSON: pixel coordinates of the SOURCE frames, origin
  top-left, y down, and whether those pixels are anamorphically squeezed. On a 1.6x squeezed
  plate an iris is an ellipse in stored coordinates and a circle in the world, and a corneal
  model that misses this is wrong by 60% in x.

MEDIAPIPE'S EYE NAMING, VERIFIED HERE: indices 468-472 are the SUBJECT'S RIGHT iris and 473-477
the subject's LEFT. Rounds 1-4 had ``IRIS_L, IRIS_R = 468, 473``, i.e. swapped. It never mattered
because the only consumer took a distance between them, but anything anatomical -- which eye is
nearer the key, which eyelid is drooping -- would have been silently mirrored.
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
# 468-472 is the subject's RIGHT iris, 473-477 the subject's LEFT -- verified against the plate,
# where 473 lands on the image-right eye and a subject facing the camera has their left eye there.
IRIS_R_CENTRE, IRIS_L_CENTRE = 468, 473
# Kept under the old names so the IPD expression below reads unchanged; the distance between the
# two iris centres does not care which is which.
IRIS_L, IRIS_R = IRIS_R_CENTRE, IRIS_L_CENTRE
# Outer eye corners, used as a fallback scale when iris points are absent.
CANTHUS_L, CANTHUS_R = 33, 263
# Tragion-ish extremes of the face oval, for a wider (yaw-sensitive) cross-check.
TRAGION_L, TRAGION_R = 234, 454

# Ordered rings of the canonical FaceMesh topology. Ordered, not sets: these are filled as
# polygons, and a set would produce a self-intersecting star. Naming is anatomical (the
# SUBJECT's left and right), which on a subject facing the camera puts "left" on the image right.
TOPOLOGY = {
    # Iris: centre first, then the four rim points, so rim[i] - centre gives the radius and the
    # squeeze. On a spherical lens the four are equidistant; on an anamorphic plate they are not.
    'iris_right':      [468, 469, 470, 471, 472],
    'iris_left':       [473, 474, 475, 476, 477],
    'eyelid_right':    [33, 7, 163, 144, 145, 153, 154, 155, 133,
                        173, 157, 158, 159, 160, 161, 246],
    'eyelid_left':     [263, 249, 390, 373, 374, 380, 381, 382, 362,
                        398, 384, 385, 386, 387, 388, 466],
    'lips_outer':      [61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
                        291, 375, 321, 405, 314, 17, 84, 181, 91, 146],
    'lips_inner':      [78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
                        308, 324, 318, 402, 317, 14, 87, 178, 88, 95],
    'face_oval':       [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
                        379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
                        234, 127, 162, 21, 54, 103, 67, 109],
}

# Blendshapes worth keeping. eyeBlink gates anything drawn on the eye; the four look* per eye
# give gaze direction without a separate gaze model, which is what aims the corneal reflection.
BLENDSHAPES = ['eyeBlinkLeft', 'eyeBlinkRight', 'eyeSquintLeft', 'eyeSquintRight',
               'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight',
               'eyeLookUpLeft', 'eyeLookUpRight', 'eyeLookDownLeft', 'eyeLookDownRight',
               'jawOpen', 'mouthClose']


def eyelid_aperture(pts, side):
    """Eyelid opening in iris radii. Zero when shut, ~2 wide open.

    mediapipe's eyeBlink blendshape saturates late -- it read 0.07 on a frame whose lids were
    plainly narrowed -- so a catchlight gated on the blendshape alone survives into frames where
    the pupil is half covered. This is the geometric measurement beside it: the lid gap in units
    of that frame's own iris radius, which makes it scale-free and needs no per-shot constant.
    """
    lid = pts[TOPOLOGY['eyelid_' + side]][:, :2]
    iris = pts[TOPOLOGY['iris_' + side]][:, :2]
    r = float(np.linalg.norm(iris[1:] - iris[0], axis=1).mean())
    if r < 1e-6:
        return 0.0
    # Height of the lid polygon perpendicular to its own long axis, so head roll cannot fake it.
    c = lid.mean(0)
    _, _, V = np.linalg.svd(lid - c, full_matrices=False)
    return float(np.ptp((lid - c) @ V[1]) / r)


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
    # Full-landmark store, parallel to rows. NaN marks a frame with no detection, so a consumer
    # that forgets to check `found` gets NaN rather than a plausible stale face.
    pts_all = np.full((n, 478, 3), np.nan, np.float32)
    blend_all = np.full((n, len(BLENDSHAPES)), np.nan, np.float32)
    aperture = np.full((n, 2), np.nan, np.float32)
    xform_all = np.full((n, 4, 4), np.nan, np.float32)
    detected = np.zeros(n, bool)
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
            # z is delivered in the same normalised units as x, so scaling it by W (not H) puts
            # all three axes in one consistent pixel scale. It is a relative depth about the head
            # centre, not metric, and nothing here treats it as metric.
            pts3 = np.array([[p_.x * W, p_.y * H, p_.z * W] for p_ in L], np.float32)
            pts_all[i] = pts3
            detected[i] = True
            if res.face_blendshapes:
                bs = {b.category_name: b.score for b in res.face_blendshapes[0]}
                blend_all[i] = [bs.get(k, np.nan) for k in BLENDSHAPES]
            if len(pts3) >= 478:
                aperture[i] = [eyelid_aperture(pts3, 'left'), eyelid_aperture(pts3, 'right')]

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
                xform_all[i] = M
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

    # ---- the full landmark store ------------------------------------------------------------
    # Arrays go in an npz (150 x 478 x 3 float32 is 860 kB; the same as JSON would be 5 MB of
    # text that every consumer then has to re-parse), and everything a consumer needs to
    # INTERPRET them goes in a JSON beside it. Splitting them this way means the metadata is
    # readable without numpy and greppable in the repo.
    anam = (shot.config.get('camera') or {}).get('anamorphic')
    squeeze = float(anam.get('squeeze', 1.6)) if isinstance(anam, dict) else (
        1.6 if anam else 1.0)
    np.savez_compressed(shot.relight / 'face_landmarks.npz',
                        points=pts_all, found=detected, blendshapes=blend_all,
                        eyelid_aperture=aperture, xform=xform_all)
    meta = {
        'shot': shot.name, 'frames': n, 'detected': int(detected.sum()),
        'missing_frames': [int(x) for x in np.nonzero(~detected)[0]],
        'arrays': {
            'points': 'float32 (frames, 478, 3). NaN on a frame with found=False.',
            'found': 'bool (frames,). False means NO DETECTION -- do not interpolate silently.',
            'blendshapes': 'float32 (frames, %d), order in blendshape_order' % len(BLENDSHAPES),
            'eyelid_aperture': 'float32 (frames, 2) = [subject-left, subject-right], lid gap in '
                               'IRIS RADII. Scale-free, so it needs no per-shot constant.',
            'xform': 'float32 (frames, 4, 4) mediapipe facial transformation matrix.'},
        'blendshape_order': BLENDSHAPES,
        'topology': TOPOLOGY,
        'topology_note': (
            'ORDERED rings, not sets -- fill them as polygons. Naming is anatomical (the '
            'SUBJECT, not the image), so for a subject facing camera the "left" ring is on the '
            'IMAGE RIGHT. mediapipe 468-472 is the RIGHT iris and 473-477 the LEFT; rounds 1-4 '
            'had those two swapped.'),
        'coordinates': {
            'space': 'pixels of source/frames/*.png, origin top-left, +x right, +y DOWN',
            'resolution': [W, H],
            'z': 'same units as x (scaled by width). RELATIVE depth about the head centre, '
                 'not metric. Nothing may treat it as metric.',
            'anamorphic_squeeze': squeeze,
            'anamorphic_warning': (
                'These pixels are ANAMORPHICALLY SQUEEZED by %.2fx in x. An iris is an ellipse '
                'here and a circle in the world. Desqueeze x (scripts/anamorphic.py) before any '
                'circular or angular model -- a corneal reflection solved in squeezed pixels is '
                'wrong by the full squeeze factor in x.' % squeeze) if squeeze != 1.0 else
                'Spherical lens; stored pixels are already square.'},
        'gates': 'Anything drawn on the eye must pass BOTH found=True and a visibility gate. '
                 'eyeBlink saturates late (it read 0.07 on a frame with visibly narrowed lids), '
                 'so gate on eyelid_aperture as well.',
        'why': 'Rounds 1-4 computed all 478 points, used two for an IPD and discarded the rest, '
               'so every eye/lip consumer had to re-run mediapipe in a second venv.'}
    (shot.relight / 'face_landmarks.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps({'landmark_store': str(shot.relight / 'face_landmarks.npz'),
                      'detected': int(detected.sum()),
                      'eyelid_aperture_median': [float(np.nanmedian(aperture[:, 0])),
                                                 float(np.nanmedian(aperture[:, 1]))]}, indent=2))
    print(json.dumps({k: out[k] for k in ('shot', 'frames', 'detected', 'ipd_px_stats')}, indent=2))
    if missing:
        print('WARNING: no face on %d frame(s): %s' % (len(missing), missing[:20]), file=sys.stderr)


if __name__ == '__main__':
    main()
