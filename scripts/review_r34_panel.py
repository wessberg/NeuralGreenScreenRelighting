"""Build a three-up panel dir -- SOURCE | REMOVED | REPLACED -- for `review_video.py`.

    dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/review_r34_panel.py \\
        --shot music_videos/reporter_test/shots/reporter_a

`review_video.py --compare` puts one before beside one after, which is the right tool when
a round has one result. Round 34 has two that have to be judged separately: the removal on
its own, which is what rounds 32 and 33 were rejected on, and the removal plus the scene's
own corneal reflection, which is the thing the eye pipeline has never actually shown. Put
side by side against the same source pixels, the second row answers "is the highlight
gone" and the third answers "does the eye still look like an eye".

Writes numbered PNGs under `composite/<name>` so `review_video.py --name <name>` encodes
them with the round's normal settings; it writes no video itself.
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shot_paths import add_shot_argument, from_args

LABELS = ('SOURCE', 'R34 REMOVED', 'R34 REPLACED')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_shot_argument(ap)
    ap.add_argument('--from-name', default='r34_removal')
    ap.add_argument('--name', default='r34_panel')
    ap.add_argument('--crop', type=int, nargs=4, metavar=('X', 'Y', 'W', 'H'),
                    default=[848, 204, 265, 178],
                    help="the r33 eye band, so this is directly comparable to the file "
                         "the artist rejected")
    ap.add_argument('--zoom', type=int, default=4)
    ap.add_argument('--rows', default='source,removal,replace',
                    help="subdirectories of --from-name to stack, in order. An entry "
                         "containing a slash is read as '<composite dir>/<sub>' instead, "
                         "so two rounds' results can be stacked against one source -- "
                         "which is the only honest way to show that a change helped.")
    ap.add_argument('--labels', default='',
                    help='PIPE-separated row labels (a label may contain commas); defaults to the built-in three')
    ap.add_argument('--stack', choices=('h', 'v'), default='v',
                    help='v stacks the three rows, which keeps each at 1:1 on a 16:9 screen')
    args = ap.parse_args()
    shot = from_args(args)
    src = shot.composite / args.from_name
    names = [x.strip() for x in args.rows.split(',') if x.strip()]
    dirs = [(shot.composite / n) if '/' in n else (src / n) for n in names]
    for d in dirs:
        if not d.is_dir():
            raise SystemExit('No %s. Run the producing stage first.' % d)
    labels = ([x.strip() for x in args.labels.split('|')] if args.labels
              else list(LABELS)[:len(dirs)])
    if len(labels) != len(dirs):
        raise SystemExit('Give one label per row')
    out = shot.composite / args.name
    out.mkdir(parents=True, exist_ok=True)

    common = None
    for d in dirs:
        got = {p.name for p in d.glob('*.png')}
        common = got if common is None else (common & got)
    names = sorted(common or [])
    if not names:
        raise SystemExit('The three dirs share no frames')
    x, y, w, h = args.crop
    written = 0
    for n in names:
        panes = []
        for d, label in zip(dirs, labels):
            im = cv2.imread(str(d / n))
            if im is None:
                raise SystemExit('Could not read %s' % (d / n))
            im = im[y:y + h, x:x + w]
            if im.size == 0:
                raise SystemExit('Crop falls outside the frame')
            im = cv2.resize(im, (im.shape[1] * args.zoom, im.shape[0] * args.zoom),
                            interpolation=cv2.INTER_NEAREST)
            bar = max(26, im.shape[0] // 22)
            cv2.rectangle(im, (0, 0), (im.shape[1] - 1, bar), (0, 0, 0), -1)
            cv2.putText(im, label, (10, int(bar * 0.75)), cv2.FONT_HERSHEY_SIMPLEX,
                        bar / 40.0, (255, 255, 255), 1, cv2.LINE_AA)
            panes.append(im)
        axis = 0 if args.stack == 'v' else 1
        gap = ((np.zeros((3, panes[0].shape[1], 3), np.uint8),) if axis == 0
               else (np.zeros((panes[0].shape[0], 3, 3), np.uint8),))
        seq = []
        for k, pane in enumerate(panes):
            if k:
                seq.append(gap[0])
            seq.append(pane)
        stacked = np.concatenate(seq, axis=axis)
        # h264 needs even dimensions.
        if stacked.shape[0] % 2:
            stacked = stacked[:-1]
        if stacked.shape[1] % 2:
            stacked = stacked[:, :-1]
        cv2.imwrite(str(out / n), stacked)
        written += 1
    (out / 'panel.json').write_text(json.dumps(
        dict(source=str(src), rows=labels, crop=args.crop, zoom=args.zoom,
             stack=args.stack, frames=written), indent=2) + '\n', encoding='utf-8')
    print('R34_PANEL', written, 'frames ->', out, flush=True)


if __name__ == '__main__':
    main()
