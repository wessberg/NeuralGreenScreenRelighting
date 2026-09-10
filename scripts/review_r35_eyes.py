"""Stack any number of results at EYE SCALE, on every frame, following the tracked eye.

The r29 contact-sheet standard, with the one thing it was missing: the crop follows the
FITTED limbus centre, so the eye stays in the middle of the tile while she moves. Round
35's first comparison used a fixed crop taken from frame 700 and by frame 760 it was
looking at her eyebrow -- an hour of reading nothing.

Each row is `<composite dir>/<sub>`, so a previous round's result and this round's sit
against the same source pixels. Writes every frame, contact sheets, and a video, because
the artefacts that get rejected here are temporal.

    dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/review_r35_eyes.py \
        --shot music_videos/reporter_test/shots/reporter_a \
        --dirs r35_newiris/source,r34_newiris2/removal,r35_newiris/removal \
        --labels "SOURCE|R34 landmark anchor|R35 image-fitted anchor" --name r35_eyes
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from master_export import ffmpeg_path
from shot_paths import add_shot_argument, from_args
import eye_reflection_repair as r32


def centres(shot, track_file):
    """Per-frame, per-eye crop centres: the fitted limbus where there is one, the
    landmark where there is not, so a frame is never dropped from a review."""
    data = json.loads((shot.relight / track_file).read_text(encoding='utf-8'))
    out = {}
    for row in data['rows']:
        for eye in row['eyes']:
            c = eye.get('smoothed_centre') or eye.get('centre') or eye['landmark_centre']
            out[(row['frame'], eye['side'])] = (float(c[0]), float(c[1]))
    return out, {row['frame']: row['source_frame'] for row in data['rows']}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_shot_argument(ap)
    ap.add_argument('--dirs', required=True,
                    help='comma-separated <composite dir>/<sub>, one per row')
    ap.add_argument('--labels', default='', help='PIPE-separated row labels')
    ap.add_argument('--track', default='limbus_track.json')
    ap.add_argument('--name', default='r35_eyes')
    ap.add_argument('--zoom', type=int, default=10)
    ap.add_argument('--half', type=int, default=24)
    ap.add_argument('--columns', type=int, default=5)
    ap.add_argument('--per-sheet', type=int, default=20)
    ap.add_argument('--fps', type=int, default=12)
    args = ap.parse_args()
    shot = from_args(args)
    names = [x.strip() for x in args.dirs.split(',') if x.strip()]
    dirs = [shot.composite / n for n in names]
    for d in dirs:
        if not d.is_dir():
            raise SystemExit('No %s' % d)
    labels = ([x.strip() for x in args.labels.split('|')] if args.labels else names)
    if len(labels) != len(dirs):
        raise SystemExit('Give one label per row')
    cen, source_frame = centres(shot, args.track)
    out = shot.composite / args.name
    (out / 'frames').mkdir(parents=True, exist_ok=True)
    (out / 'sheets').mkdir(parents=True, exist_ok=True)

    common = None
    for d in dirs:
        got = {p.name for p in d.glob('*.png')}
        common = got if common is None else (common & got)
    files = sorted(common or [])
    if not files:
        raise SystemExit('The rows share no frames')

    z, h = args.zoom, args.half
    panels, written = [], []
    for fn in files:
        i = int(Path(fn).stem)
        rows_ = []
        for d, label in zip(dirs, labels):
            im = cv2.imread(str(d / fn))
            if im is None:
                raise SystemExit('Could not read %s' % (d / fn))
            tiles = []
            for side in r32.SIDES:
                c = cen.get((i, side))
                if c is None:
                    tiles.append(np.zeros((2 * h * z, 2 * h * z, 3), np.uint8))
                    continue
                x0 = int(round(c[0])) - h
                y0 = int(round(c[1])) - h
                crop = im[max(0, y0):y0 + 2 * h, max(0, x0):x0 + 2 * h]
                if crop.shape[:2] != (2 * h, 2 * h):
                    crop = cv2.copyMakeBorder(crop, 0, 2 * h - crop.shape[0], 0,
                                              2 * h - crop.shape[1], cv2.BORDER_CONSTANT)
                tiles.append(cv2.resize(crop, None, fx=z, fy=z,
                                        interpolation=cv2.INTER_NEAREST))
            band = np.hstack(tiles)
            cv2.rectangle(band, (0, 0), (band.shape[1] - 1, 17), (0, 0, 0), -1)
            cv2.putText(band, '%s   source %d' % (label, source_frame.get(i, i)),
                        (4, 13), 0, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            rows_.append(band)
        frame = np.vstack(rows_)
        if frame.shape[0] % 2:
            frame = frame[:-1]
        if frame.shape[1] % 2:
            frame = frame[:, :-1]
        cv2.imwrite(str(out / 'frames' / fn), frame)
        panels.append(frame)

    for start in range(0, len(panels), args.per_sheet):
        block = panels[start:start + args.per_sheet]
        block = block + [np.zeros_like(block[0])] * (-len(block) % args.columns)
        sheet = np.vstack([np.hstack(block[k:k + args.columns])
                           for k in range(0, len(block), args.columns)])
        name = 'sheet_%03d.jpg' % (start // args.per_sheet)
        cv2.imwrite(str(out / 'sheets' / name), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        written.append(name)

    video = shot.final / ('%s.mp4' % args.name)
    video.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([ffmpeg_path(), '-y', '-hide_banner', '-loglevel', 'error',
                    '-framerate', str(args.fps), '-i', str(out / 'frames' / '%06d.png'),
                    '-c:v', 'libx264', '-crf', '12', '-pix_fmt', 'yuv420p',
                    '-colorspace', 'bt709', '-color_primaries', 'bt709',
                    '-color_trc', 'bt709', str(video)], check=True, capture_output=True)
    (out / 'panel.json').write_text(json.dumps(
        dict(rows=labels, dirs=names, track=args.track, zoom=z, half=h,
             frames=len(panels), sheets=written, video=str(video),
             purpose='REVIEW ONLY, display-referred sRGB'), indent=2) + '\n',
        encoding='utf-8')
    print('%d frames, %d sheets -> %s' % (len(panels), len(written), out))
    print('video %s' % video)


if __name__ == '__main__':
    main()
