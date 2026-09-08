"""Turn a folder of composited frames into something the artist can actually watch.

    dependencies/CorridorKey/.venv/Scripts/python.exe scripts/review_video.py --shot <dir> --name ratio
    ... --name r4_ratio --compare ratio          # side by side, old on the left

Every round so far has re-derived this with an ad-hoc ffmpeg line, and re-derived the settings by
probing the previous round's file. It is a pipeline stage like any other, so it lives here.

Two things about it are conventions rather than facts, and both are easy to get wrong:

* **Review runs at 24 fps regardless of the shot's rate.** The stress tour is 150 rendered frames,
  not 150 frames of a 59.94 fps performance, and every review video in `final/` so far is 24 fps.
  Judging motion in the plate is a different job and wants `--fps shot`.
* **The frames are already display-referred sRGB PNGs.** No transform belongs here. This is a
  REVIEW encode, not a delivery: delivery is `master_export.py` then `hdr_mux.py`, which carry the
  Rec.2020 / ST2084 pipeline and the HDR10 metadata. Never hand one of these files to a grade.

A side-by-side is the useful form for judging a fix, so `--compare` takes the dir to put on the
LEFT (the before) and labels both halves, because six months later nobody remembers which side was
which. Labels are drawn with OpenCV rather than ffmpeg's drawtext, which needs a font path that has
to be escaped differently on Windows and fails silently when it is wrong.
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import sys, json, shutil, argparse, subprocess, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np, cv2
from shot_paths import add_shot_argument, from_args
from master_export import ffmpeg_path

REVIEW_FPS = 24.0


def frames_in(d):
    return sorted(p for p in d.glob('*.png') if p.stem.isdigit())


def apply_crop(img, crop, zoom):
    if crop:
        x, y, w, h = crop
        img = img[y:y + h, x:x + w]
        if img.size == 0:
            raise SystemExit('Crop %s falls outside the frame' % (crop,))
    if zoom > 1:
        img = cv2.resize(img, (img.shape[1] * zoom, img.shape[0] * zoom),
                         interpolation=cv2.INTER_NEAREST)
    return img


def label_bar(img, text):
    h = max(28, img.shape[0] // 30)
    cv2.rectangle(img, (0, 0), (img.shape[1] - 1, h), (0, 0, 0), -1)
    cv2.putText(img, text, (10, int(h * 0.72)), cv2.FONT_HERSHEY_SIMPLEX,
                h / 42.0, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_shot_argument(ap)
    ap.add_argument('--name', default='ratio', help='dir under composite/ holding the PNGs')
    ap.add_argument('--compare', default='',
                    help='another dir under composite/ to place on the LEFT, as the before')
    ap.add_argument('--label', default='', help='defaults to --name')
    ap.add_argument('--compare-label', default='', help='defaults to --compare')
    ap.add_argument('--out', default='', help='filename under final/; defaults from the names')
    ap.add_argument('--fps', default=str(int(REVIEW_FPS)),
                    help="review rate, or 'shot' to use the shot's own frame rate")
    # A full-frame side-by-side is 3840 wide and every player scales it to fit, so each half
    # arrives at roughly half resolution -- which is exactly where despill and edge artefacts
    # live. --crop keeps both halves at 1:1 by looking at a region instead of the whole frame.
    ap.add_argument('--crop', type=int, nargs=4, metavar=('X', 'Y', 'W', 'H'), default=None,
                    help='crop both sequences to this region before comparing, so no downscale')
    ap.add_argument('--zoom', type=int, default=1, help='nearest-neighbour magnification of the crop')
    ap.add_argument('--crf', type=int, default=12,
                    help='low by default: these are judged for fine artefacts, and h264 at a '
                         'normal crf hides exactly the sub-pixel detail a round is arguing about')
    args = ap.parse_args()
    shot = from_args(args)

    src = shot.composite / args.name
    frames = frames_in(src)
    if not frames:
        raise SystemExit('No numbered PNGs in %s' % src)
    fps = float(shot.frame_rate) if args.fps == 'shot' else float(args.fps)

    cmp_dir, cmp_frames = None, []
    if args.compare:
        cmp_dir = shot.composite / args.compare
        cmp_frames = frames_in(cmp_dir)
        if not cmp_frames:
            raise SystemExit('No numbered PNGs in %s' % cmp_dir)
        shared = sorted({p.stem for p in frames} & {p.stem for p in cmp_frames})
        if not shared:
            raise SystemExit('%s and %s share no frame numbers' % (src, cmp_dir))
        if len(shared) != len(frames) or len(shared) != len(cmp_frames):
            print('note: comparing the %d frames the two dirs share' % len(shared))
        frames = [src / (s + '.png') for s in shared]
        cmp_frames = [cmp_dir / (s + '.png') for s in shared]

    out = shot.final / (args.out or
                        ('%s_vs_%s.mp4' % (args.compare, args.name) if args.compare
                         else '%s.mp4' % args.name))
    out.parent.mkdir(parents=True, exist_ok=True)

    tmp = Path(tempfile.mkdtemp(prefix='review_'))
    try:
        if cmp_dir is None and not args.crop and args.zoom == 1:
            feed, pattern = src, '%06d.png'
            # A dir of bare frames needs no rewrite; point ffmpeg straight at it.
            start = int(frames[0].stem)
        elif cmp_dir is None:
            for i, f in enumerate(frames):
                im = cv2.imread(str(f))
                if im is None:
                    raise SystemExit('Could not read %s' % f)
                cv2.imwrite(str(tmp / ('%06d.png' % i)), apply_crop(im, args.crop, args.zoom))
            feed, pattern, start = tmp, '%06d.png', 0
        else:
            left_label = args.compare_label or args.compare
            right_label = args.label or args.name
            for i, (a, b) in enumerate(zip(cmp_frames, frames)):
                la, lb = cv2.imread(str(a)), cv2.imread(str(b))
                if la is None or lb is None:
                    raise SystemExit('Could not read %s or %s' % (a, b))
                if la.shape != lb.shape:
                    raise SystemExit('%s and %s differ in size, %s vs %s'
                                     % (a.name, b.name, la.shape, lb.shape))
                la, lb = (apply_crop(x, args.crop, args.zoom) for x in (la, lb))
                pair = np.concatenate([label_bar(la, left_label),
                                       label_bar(lb, right_label)], 1)
                # h264 needs even dimensions in yuv420p.
                if pair.shape[1] % 2 or pair.shape[0] % 2:
                    pair = pair[:pair.shape[0] // 2 * 2, :pair.shape[1] // 2 * 2]
                cv2.imwrite(str(tmp / ('%06d.png' % i)), pair)
            feed, pattern, start = tmp, '%06d.png', 0

        cmd = [ffmpeg_path(), '-y', '-hide_banner', '-loglevel', 'error',
               '-framerate', repr(fps), '-start_number', str(start),
               '-i', str(feed / pattern),
               '-c:v', 'libx264', '-crf', str(args.crf), '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', str(out)]
        subprocess.run(cmd, check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    size = out.stat().st_size
    summary = {'shot': shot.name, 'output': str(out), 'frames': len(frames),
               'fps': fps, 'crf': args.crf, 'bytes': size,
               'crop': args.crop, 'zoom': args.zoom,
               'source': str(src), 'compare': str(cmp_dir) if cmp_dir else None,
               'purpose': 'REVIEW ONLY, display-referred sRGB. Delivery is master_export + hdr_mux.'}
    shot.log('review_video.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
