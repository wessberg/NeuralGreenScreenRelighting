"""Encode four native-pixel defect comparisons and decode videos for sequence inspection."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import cv2
import numpy as np
from shot_paths import add_shot_argument, from_args


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_shot_argument(ap)
    args = ap.parse_args()
    shot = from_args(args)
    lm = np.load(shot.relight / 'face_landmarks.npz')
    topo = json.loads((shot.relight / 'face_landmarks.json').read_text())['topology']
    points = lm['points'][lm['found']]
    crops = {}
    # Union over all frames, so motion never carries the mouth out of a frame-zero crop.
    for name, indices, pad in [('teeth', topo['lips_outer'], 36),
                               ('sclera', topo['eyelid_left'] + topo['eyelid_right'], 36),
                               ('eyelid', topo['eyelid_left'] + topo['eyelid_right'], 44)]:
        p = points[:, indices, :2].reshape(-1, 2)
        lo = np.maximum(np.floor(p.min(0) - pad), 0).astype(int)
        hi = np.ceil(p.max(0) + pad).astype(int)
        size = ((hi - lo + 1) // 2) * 2
        crops[name] = [*lo.tolist(), *size.tolist()]
    crops['hair'] = [780, 32, 460, 236]  # reporter-only crown ROI; recorded in every manifest
    videos = []
    for name, crop in crops.items():
        filename = 'r7_%s_1to1.mp4' % name
        cmd = [sys.executable, str(Path(__file__).with_name('review_video.py')),
               '--shot', str(shot.dir), '--name', 'ratio_r7', '--compare', 'ratio',
               '--compare-label', 'round 4', '--label', 'rollback', '--out', filename,
               '--crop', *map(str, crop), '--zoom', '1']
        subprocess.run(cmd, check=True)
        videos.append(filename)
    out = shot.composite / 'r7_audit'
    decoded = []
    # Read EVERY encoded frame, not just the source PNGs. Sheets retain consecutive frames
    # at native pixels. This is frame-sequence inspection, not a claim of real-time playback.
    for name in videos + ['r5_eyes_1to1.mp4', 'r5_hair_1to1.mp4', 'r5_mouth_1to1.mp4']:
        cap = cv2.VideoCapture(str(shot.final / name))
        fps = cap.get(cv2.CAP_PROP_FPS)
        count, tiles, pages = 0, [], []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # Reduce historical videos solely for the inspection sheet. This display resize
            # is not a native-pixel inspection and never changes the videos themselves.
            if name.startswith('r5_'):
                # Their stored encode manifests were overwritten; report the display size.
                display = cv2.resize(frame, (600, round(frame.shape[0]*600/frame.shape[1])), interpolation=cv2.INTER_AREA)
            else:
                display = frame.copy()
            cv2.putText(display, str(count), (3, display.shape[0]-4), cv2.FONT_HERSHEY_SIMPLEX,
                        .45, (255, 255, 255), 1)
            tiles.append(display)
            count += 1
            if len(tiles) == 30:
                page = np.concatenate([np.concatenate(tiles[i:i+3], 1) for i in range(0, 30, 3)], 0)
                path = out / ('%s_%03d.png' % (name[:-4], count-30))
                cv2.imwrite(str(path), page)
                pages.append(str(path.relative_to(shot.dir)))
                tiles = []
        cap.release()
        if count != 150 or abs(fps-24) > .001:
            raise RuntimeError('Bad review encode %s: %s frames, fps %s' % (name, count, fps))
        decoded.append({'video': name, 'frames_decoded': count, 'fps': fps, 'sequence_sheets': pages})
    shot.log('r7_video_review.json').write_text(json.dumps({'crops_xywh': crops, 'zoom': 1,
        'decoded': decoded, 'inspection': 'Consecutive-frame sheets; no real-time playback tool available.'}, indent=2))


if __name__ == '__main__':
    main()
