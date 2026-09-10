"""Measure the LID-SHADOW SEAM: the luminance step across the fill boundary.

Round 35 section 7 measured this by hand and shipped the number in a markdown table. This
is that measurement as a script, because a number that cannot be re-run is a number the
next round has to take on trust.

**What the artist is seeing.** Her iris darkens toward the lash -- it is in shadow under
the lid -- and the canonical iris is painted with ONE scalar gain per channel per frame,
so it cannot carry a gradient. Where the fill meets her own lid-shadowed iris there is a
step. Round 35 measured +0.0824 display in her photograph against -0.0039 in the result:
the repair does not merely get the shadow slightly wrong, it removes it.

**What this reads.** The SOURCE photograph, the RESULT, and the written WEIGHT map. The
weight is the repair's own declared audit trail -- the file that says what was changed --
so using it to locate the boundary is not the repair grading its own homework about
LEVELS, which is what is measured here and comes from the two photographs only.

Two statistics, both reported with p95 and max as AGENTS.md requires:

* **step** -- median display luma just inside the boundary minus just outside it, in the
  UPPER half of the iris, where the lid shadow is. A source that steps and a result that
  does not is the seam.
* **profile** -- the observed clean iris divided by the level the repair painted beside
  it, binned by depth below the fitted upper lid margin. This is the shape the repair has
  to carry, and it is what `--lid-shadow` in `eye_soft_removal.py` fits.

    dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/audit_r36_seam.py \
        --shot music_videos/reporter_test/shots/reporter_a --name r35_final
"""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shot_paths import add_shot_argument, from_args
import eye_reflection_repair as r32
import limbus_fit as lf

BAND = 2.0        # native px either side of the boundary that the step is read over


def luma(bgr):
    return bgr[..., ::-1].astype(np.float32).max(-1) / 255.0


def stat(name, xs):
    xs = np.asarray(xs, float)
    if xs.size == 0:
        return {name: None}
    return {name: dict(n=int(xs.size), median=round(float(np.median(xs)), 4),
                       p95=round(float(np.percentile(xs, 95)), 4),
                       max=round(float(xs.max()), 4), min=round(float(xs.min()), 4))}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_shot_argument(ap)
    ap.add_argument('--name', default='r35_final')
    ap.add_argument('--track', default='limbus_track.json')
    ap.add_argument('--eyelid-track', default='eyelid_track.json')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    shot = from_args(args)
    root = shot.composite / args.name
    track = lf.Track.load(shot.relight / args.track)
    lids = {}
    for row in json.loads((shot.relight / args.eyelid_track)
                          .read_text(encoding='utf-8'))['rows']:
        for e in row['eyes']:
            if e.get('fitted'):
                lids[(int(row['frame']), e['side'])] = e

    steps = {s: dict(source=[], result=[], delta=[]) for s in r32.SIDES}
    prof = {s: {} for s in r32.SIDES}
    treated = 0
    for path in sorted((root / 'weight').glob('*.png')):
        i = int(path.stem)
        w = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        src = cv2.imread(str(root / 'source' / path.name), cv2.IMREAD_UNCHANGED)
        res = cv2.imread(str(root / 'removal' / path.name), cv2.IMREAD_UNCHANGED)
        if w is None or src is None or res is None:
            continue
        ls, lr = luma(src), luma(res)
        wf = w.astype(np.float32) / 255.0
        for side in r32.SIDES:
            got = track.get(i, side)
            if got is None:
                continue
            centre, ax, ay = got[0], got[1], got[2]
            cx, cy = float(centre[0]), float(centre[1])
            r = max(ax, ay)
            y0, y1 = int(cy - 3 * r), int(cy + 3 * r) + 1
            x0, x1 = int(cx - 3 * r), int(cx + 3 * r) + 1
            if y0 < 0 or x0 < 0 or y1 > w.shape[0] or x1 > w.shape[1]:
                continue
            sub = (slice(y0, y1), slice(x0, x1))
            ww = wf[sub]
            if ww.max() <= 0:
                continue
            treated += 1
            yy, xx = np.mgrid[y0:y1, x0:x1]
            p = (xx - cx) / ax
            q = (yy - cy) / ay
            rho = np.hypot(p, q)
            upper = q < -0.05          # above the iris centre: where the lid shadow is
            core = ww > 0.5
            outs = ww <= 0.0
            k = int(2 * np.ceil(BAND) + 1)
            kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            grown = cv2.dilate(core.astype(np.uint8), kern) > 0
            shrunk = cv2.erode(core.astype(np.uint8), kern) > 0
            inner = core & ~shrunk & upper & (rho < 1.0)
            outer = grown & outs & upper & (rho < 1.0)
            if inner.sum() < 12 or outer.sum() < 12:
                continue
            s_step = float(np.median(ls[sub][inner]) - np.median(ls[sub][outer]))
            r_step = float(np.median(lr[sub][inner]) - np.median(lr[sub][outer]))
            steps[side]['source'].append(s_step)
            steps[side]['result'].append(r_step)
            steps[side]['delta'].append(r_step - s_step)
            # The profile the fix needs: her own CLEAN iris against the level the repair
            # painted right beside it, binned by depth below the fitted upper lid.
            cfg = lids.get((i, side))
            if cfg and cfg.get('upper'):
                a, b, c = cfg['upper']['coef']
                u = xx - cfg['centre'][0]
                depth = yy - (a * u ** 2 + b * u + c)
                clean = outs & (rho < 0.95) & (ls[sub] > 0.02)
                ref = core & (rho < 0.95)
                if clean.sum() >= 30 and ref.sum() >= 30:
                    level = float(np.median(lr[sub][ref]))
                    if level > 0.02:
                        d = depth[clean]
                        v = ls[sub][clean] / level
                        for db, vv in zip(np.floor(d).astype(int), v):
                            if 0 <= db < 24:
                                prof[side].setdefault(int(db), []).append(float(vv))

    out = dict(shot=shot.name, name=args.name, band_px=BAND, eyes_measured=treated,
               reads='source photograph, result, and the written weight map only')
    for side in r32.SIDES:
        d = {}
        for k in ('source', 'result', 'delta'):
            d.update(stat(k, steps[side][k]))
        d['profile_by_depth'] = {
            str(k): dict(n=len(v), median=round(float(np.median(v)), 4))
            for k, v in sorted(prof[side].items()) if len(v) >= 40}
        out[side] = d
        s = d.get('source') or {}
        r = d.get('result') or {}
        print('R36_SEAM', side,
              'source', None if not s else s['median'],
              'result', None if not r else r['median'],
              'n', None if not s else s['n'], flush=True)
    dest = shot.logs / (args.out or ('r36_seam_%s.json' % args.name))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print('R36_SEAM_WROTE', dest, flush=True)


if __name__ == '__main__':
    main()
