"""Where does a result FLICKER? Over a steady stretch, in the eye's own coordinates, in CIELAB.

The artist's R37h verdict was *"quite a bit of flicker in the eyes from frame to frame,
like small color variation here and there ... not as easily detectable in your atlas
tables"*. Every static tool agreed with the atlas. This is the one that saw it.

Each frame is resampled onto a fixed grid through the fitted limbus, so a cell is the same
piece of eye in every frame. Over cells replaced in EVERY frame of the window it reports

* the frame-to-frame step of the fill's GLOBAL mean L* and a*b* (a level or hue pulse of
  the whole fill -- round 37's per-frame gain), p50 / p95 / max;
* the median over cells of the rms frame-to-frame change of the residual after removing
  that global mean (local flicker "here and there"), L* and chroma separately;

and the same numbers for the SOURCE on the same cells as the control -- her own eye moving
and the plate's codec noise. A map per run is written beside the numbers: where the local
flicker lives is usually the diagnosis (round 37: the fill's top band and the softbox
region, exactly where a per-frame shading read its values).

The window must be one where the limbus track is good: a slid track makes a still fill
look like it jumps (round 37 section 5).

    dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/audit_r37_flicker.py \
        --shot music_videos/reporter_test/shots/reporter_a --names r36g,r37w --side left \
        --frames 46-78
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shot_paths import add_shot_argument, from_args
import limbus_fit as lf


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_shot_argument(ap)
    ap.add_argument('--names', required=True, help='comma-separated composite run names')
    ap.add_argument('--side', default='left')
    ap.add_argument('--frames', required=True, help='first-last shot frame indices')
    ap.add_argument('--source', default=None,
                    help='composite dir holding source/ PNGs (default: the first run)')
    ap.add_argument('--cells', type=int, default=49)
    args = ap.parse_args()
    shot = from_args(args)
    names = [n.strip() for n in args.names.split(',') if n.strip()]
    src_dir = shot.composite / (args.source or names[0]) / 'source'
    track = lf.Track.load(shot.relight / 'limbus_track.json')
    lo, hi = (int(x) for x in args.frames.split('-'))
    t = np.linspace(-1.2, 1.2, args.cells).astype(np.float32)
    gu, gv = np.meshgrid(t, t)
    rho = np.hypot(gu, gv)
    frames = [i for i in range(lo, hi + 1) if track.get(i, args.side) is not None]
    grids = []
    for i in frames:
        (cx, cy), ax, ay = track.get(i, args.side)
        grids.append(((cx + gu * ax).astype(np.float32), (cy + gv * ay).astype(np.float32)))

    def lab(path, mx, my):
        im = cv2.imread(str(path))
        tile = cv2.remap(im, mx, my, cv2.INTER_LINEAR).astype(np.float32) / 255
        return cv2.cvtColor(tile, cv2.COLOR_BGR2Lab)

    out_dir = shot.composite / 'r37_flicker'
    out_dir.mkdir(parents=True, exist_ok=True)
    report, rows = {}, []
    for n in names + ['__source__']:
        ref = names[-1] if n == '__source__' else n
        stack, ws = [], []
        for i, (mx, my) in zip(frames, grids):
            p = (src_dir if n == '__source__' else shot.composite / n / 'removal') \
                / ('%06d.png' % i)
            stack.append(lab(p, mx, my))
            w = cv2.imread(str(shot.composite / ref / 'weight' / ('%06d.png' % i)),
                           cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
            ws.append(cv2.remap(w, mx, my, cv2.INTER_LINEAR))
        S, W = np.stack(stack), np.stack(ws)
        m = (W >= 0.99).all(0) & (rho < 0.95)
        if m.sum() < 20:
            raise SystemExit('fewer than 20 cells replaced in every frame of %s' % args.frames)
        g = S[:, m].mean(1)
        steps = np.abs(np.diff(g, axis=0))
        chroma_steps = np.hypot(steps[:, 1], steps[:, 2])
        resid = S - g[:, None, None, :]
        sd = np.sqrt((np.diff(resid, axis=0) ** 2).mean(0))
        local_l, local_c = sd[..., 0], np.hypot(sd[..., 1], sd[..., 2])
        key = 'source' if n == '__source__' else n
        report[key] = dict(
            cells=int(m.sum()),
            global_step_L=[round(float(np.median(steps[:, 0])), 4),
                           round(float(np.percentile(steps[:, 0], 95)), 4),
                           round(float(steps[:, 0].max()), 4)],
            global_step_chroma=[round(float(np.median(chroma_steps)), 4),
                                round(float(np.percentile(chroma_steps, 95)), 4),
                                round(float(chroma_steps.max()), 4)],
            local_L_median=round(float(np.median(local_l[m])), 4),
            local_chroma_median=round(float(np.median(local_c[m])), 4))
        print('R37_FLICKER %-10s %s' % (key, json.dumps(report[key])), flush=True)
        tiles = []
        for mp, scale, what in ((local_l, 3.0, 'L*'), (local_c, 1.5, 'chroma')):
            v = np.where(rho < 1.2, mp, 0)
            im = cv2.applyColorMap(np.clip(v / scale * 255, 0, 255).astype(np.uint8),
                                   cv2.COLORMAP_INFERNO)
            im = cv2.resize(im, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)
            edge = cv2.resize(m.astype(np.uint8) * 255, None, fx=6, fy=6,
                              interpolation=cv2.INTER_NEAREST)
            cnt, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(im, cnt, -1, (0, 255, 0), 1)
            cv2.putText(im, '%s %s' % (key, what), (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (255, 255, 255), 1)
            tiles.append(im)
        rows.append(np.hstack(tiles))
    tag = '%s_%s_%s' % ('_'.join(names), args.side, args.frames)
    cv2.imwrite(str(out_dir / ('flicker_%s.png' % tag)), np.vstack(rows))
    (shot.logs / ('r37_flicker_%s.json' % tag)).write_text(
        json.dumps(dict(window=args.frames, side=args.side, runs=report), indent=1),
        encoding='utf-8')
    print('R37_FLICKER_WROTE', out_dir / ('flicker_%s.png' % tag), flush=True)


if __name__ == '__main__':
    main()
