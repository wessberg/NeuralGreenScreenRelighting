"""Draw the fitted limbus and the mediapipe landmark on EVERY frame, at eye scale.

Round 34's rule, in the artist's words: put frames in front of him constantly, and
inspect every frame at eye scale. Two of round 33's four bugs were invisible in the logs.
A tracker is a temporal object, so this writes both contact sheets and a per-frame video:
a still sheet at low zoom will hide a 1.5 px error, which is exactly how round 23's
review tiles -- 70 px crops shown at 1.7x on the sheet -- read as perfect while carrying
the same order of error the artist can now see at 16x.

Yellow is the mediapipe four-point ellipse every fill since round 32 is anchored on.
Green is `limbus_fit.py`'s fit to the image edge. Red crosses are the per-angle sub-pixel
edge samples the fit actually used, so a wrong fit shows WHY.

    dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/review_r35_track.py \
        --shot music_videos/reporter_test/shots/reporter_a --track limbus_track.json
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from color_pipeline import linear_to_srgb
from shot_paths import add_shot_argument, from_args
import eye_reflection_repair as r32
from master_export import ffmpeg_path

SHIFT = 4               # cv2 sub-pixel fixed point, so a 0.1 px change is visible
UNIT = 1 << SHIFT


def _pt(x, y, x0, y0, zoom):
    return (int(round(((x - x0) * zoom + zoom / 2) * UNIT)),
            int(round(((y - y0) * zoom + zoom / 2) * UNIT)))


def eye_panel(disp, row, side, zoom, half, show_landmark=True):
    eye = next((e for e in row['eyes'] if e['side'] == side), None)
    if eye is None:
        return None
    cx, cy = eye['landmark_centre']
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    crop = disp[y0:y0 + 2 * half, x0:x0 + 2 * half]
    if crop.shape[:2] != (2 * half, 2 * half):
        return None
    big = cv2.resize((np.clip(crop, 0, 1)[..., ::-1] * 255).astype(np.uint8), None,
                     fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    if show_landmark:
        cv2.ellipse(big, _pt(cx, cy, x0, y0, zoom),
                    (int(round(eye['landmark_axes'][0] * zoom * UNIT)),
                     int(round(eye['landmark_axes'][1] * zoom * UNIT))),
                    0, 0, 360, (0, 210, 255), 1, cv2.LINE_AA, SHIFT)
    if eye.get('fitted'):
        fx, fy = eye.get('smoothed_centre', eye['centre'])
        ax, ay = eye.get('smoothed_axes', eye['axes'])
        cv2.ellipse(big, _pt(fx, fy, x0, y0, zoom),
                    (int(round(ax * zoom * UNIT)), int(round(ay * zoom * UNIT))),
                    0, 0, 360, (80, 255, 80), 1, cv2.LINE_AA, SHIFT)
        for px, py in eye.get('edge_points', []):
            p = _pt(px, py, x0, y0, zoom)
            cv2.drawMarker(big, (p[0] >> SHIFT, p[1] >> SHIFT), (60, 60, 255),
                           cv2.MARKER_TILTED_CROSS, max(3, zoom // 2), 1, cv2.LINE_AA)
        text = 'r %.2f  rms %.2f  arc %.2f' % (ax, eye.get('rms', 0), eye.get('open_arc', 0))
    else:
        text = eye.get('reason', 'no fit')
    cv2.rectangle(big, (0, 0), (big.shape[1] - 1, 15), (0, 0, 0), -1)
    cv2.putText(big, '%d %s %s' % (row['source_frame'], side[0].upper(), text),
                (3, 12), 0, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
    return big


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_shot_argument(ap)
    ap.add_argument('--track', default='limbus_track.json')
    ap.add_argument('--name', default='r35_track')
    ap.add_argument('--zoom', type=int, default=10)
    ap.add_argument('--half', type=int, default=26)
    ap.add_argument('--columns', type=int, default=6)
    ap.add_argument('--per-sheet', type=int, default=24)
    ap.add_argument('--no-landmark', action='store_true')
    args = ap.parse_args()
    shot = from_args(args)
    data = json.loads((shot.relight / args.track).read_text(encoding='utf-8'))
    out = shot.composite / args.name
    (out / 'frames').mkdir(parents=True, exist_ok=True)
    (out / 'sheets').mkdir(parents=True, exist_ok=True)

    panels = []
    for row in data['rows']:
        raw = cv2.imread(str(shot.fg(row['frame'])), cv2.IMREAD_UNCHANGED)
        disp = np.clip(linear_to_srgb(raw[..., ::-1].astype(np.float32)), 0, 1)
        pair = [eye_panel(disp, row, s, args.zoom, args.half, not args.no_landmark)
                for s in r32.SIDES]
        pair = [p for p in pair if p is not None]
        if not pair:
            continue
        frame = np.hstack(pair)
        cv2.imwrite(str(out / 'frames' / ('%06d.png' % row['frame'])), frame)
        panels.extend(pair)

    written = []
    for start in range(0, len(panels), args.per_sheet):
        block = panels[start:start + args.per_sheet]
        h, w = block[0].shape[:2]
        block = block + [np.zeros_like(block[0])] * (-len(block) % args.columns)
        rows_ = [np.hstack(block[k:k + args.columns])
                 for k in range(0, len(block), args.columns)]
        sheet = np.vstack(rows_)
        name = 'sheet_%03d.jpg' % (start // args.per_sheet)
        cv2.imwrite(str(out / 'sheets' / name), sheet, [cv2.IMWRITE_JPEG_QUALITY, 95])
        written.append(name)

    video = shot.final / ('%s.mp4' % args.name)
    video.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg_path(), '-y', '-hide_banner', '-loglevel', 'error', '-framerate', '12', '-i',
           str(out / 'frames' / '%06d.png'), '-c:v', 'libx264', '-crf', '12',
           '-pix_fmt', 'yuv420p', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
           '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
           str(video)]
    subprocess.run(cmd, check=True, capture_output=True)
    print('%d panels, %d sheets in %s' % (len(panels), len(written), out / 'sheets'))
    print('video %s' % video)


if __name__ == '__main__':
    main()
