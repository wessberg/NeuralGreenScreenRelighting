"""FACE-SCALE rows, one per result, following both fitted eyes, every frame, plus a video.

The eye-scale review (`review_r35_eyes.py`) is where a defect is found; this is where it is
JUDGED -- at the scale the shot actually plays, where round 36's lid shadow was killed in
one look and where "removal alone reads as a dead eye" was first seen. Rounds 35 and 36
built these panels ad hoc and nothing recorded the crop; here it follows the midpoint of
the two fitted limbus centres, so the face stays put while she moves.

    dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/review_r37_panel.py \
        --shot music_videos/reporter_test/shots/reporter_a --name r37h_panel \
        --dirs r37h/source,r36g/removal,r37h/removal --labels "SOURCE|R36g|R37h"
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from master_export import ffmpeg_path
from shot_paths import add_shot_argument, from_args
import limbus_fit as lf


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_shot_argument(ap)
    ap.add_argument('--dirs', required=True)
    ap.add_argument('--labels', default='')
    ap.add_argument('--name', required=True)
    ap.add_argument('--width', type=int, default=530, help='crop width, native px')
    ap.add_argument('--height', type=int, default=320, help='crop height, native px')
    ap.add_argument('--drop', type=int, default=70,
                    help='native px the crop centre sits BELOW the eyes')
    ap.add_argument('--zoom', type=float, default=2.0)
    ap.add_argument('--fps', type=int, default=12)
    args = ap.parse_args()
    shot = from_args(args)
    track = lf.Track.load(shot.relight / 'limbus_track.json')
    dirs = [shot.composite / d.strip() for d in args.dirs.split(',')]
    labels = args.labels.split('|') if args.labels else [d.name for d in dirs]
    out = shot.composite / args.name
    out.mkdir(parents=True, exist_ok=True)
    last = None
    n = 0
    for i in range(shot.frame_count):
        if not all((d / ('%06d.png' % i)).exists() for d in dirs):
            continue
        pts = [track.get(i, s) for s in ('left', 'right')]
        pts = [p[0] for p in pts if p is not None]
        if pts:
            last = np.mean(np.asarray(pts, float), axis=0)
        if last is None:
            continue
        cx, cy = int(round(last[0])), int(round(last[1])) + args.drop
        x0, y0 = cx - args.width // 2, cy - args.height // 2
        rows = []
        for d, lab in zip(dirs, labels):
            img = cv2.imread(str(d / ('%06d.png' % i)))
            crop = img[max(0, y0):y0 + args.height, max(0, x0):x0 + args.width]
            crop = cv2.copyMakeBorder(crop, 0, args.height - crop.shape[0], 0,
                                      args.width - crop.shape[1], cv2.BORDER_CONSTANT)
            big = cv2.resize(crop, None, fx=args.zoom, fy=args.zoom,
                             interpolation=cv2.INTER_CUBIC)
            bar = np.zeros((34, big.shape[1], 3), np.uint8)
            cv2.putText(bar, '%s   source %d' % (lab, i + int(shot.source_start_frame)),
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1,
                        cv2.LINE_AA)
            rows.append(np.vstack([bar, big]))
        frame = np.vstack(rows)
        frame = frame[:frame.shape[0] // 2 * 2, :frame.shape[1] // 2 * 2]
        cv2.imwrite(str(out / ('%06d.png' % i)), frame)
        n += 1
    video = shot.final / ('%s.mp4' % args.name)
    subprocess.run([ffmpeg_path(), '-y', '-hide_banner', '-loglevel', 'error',
                    '-framerate', str(args.fps), '-i', str(out / '%06d.png'),
                    '-c:v', 'libx264', '-crf', '12', '-pix_fmt', 'yuv420p',
                    '-colorspace', 'bt709', '-color_primaries', 'bt709',
                    '-color_trc', 'bt709', str(video)], check=True, capture_output=True)
    (out / 'panel.json').write_text(json.dumps(dict(
        rows=labels, dirs=[str(d) for d in dirs], width=args.width, height=args.height,
        drop=args.drop, zoom=args.zoom, frames=n, video=str(video),
        purpose='REVIEW ONLY, display-referred sRGB'), indent=2) + '\n', encoding='utf-8')
    print('%d frames -> %s' % (n, video))


if __name__ == '__main__':
    main()
