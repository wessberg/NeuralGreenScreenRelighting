"""Measure what the artist actually rejected, and whether round 34 fixes it.

    dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/audit_r34_eyes.py \\
        --shot music_videos/reporter_test/shots/reporter_a

Round 33 scored well and looked wrong, so this audit is built around the three things
that are visible at 8x rather than around the atlas's accuracy, which round 33 already
established and round 34 does not change (the atlas construction is byte-identical; only
the removal weight and the added reflection are new).

**1. Residue.** How much of the highlight survives, in the eye's own noise units, and --
the part that makes it visible -- how much that varies from frame to frame. A residue
that is always there reads as a slightly wrong pupil; a residue that comes and goes reads
as flicker, which is what round 32 was rejected for and what round 33 still did.

**2. Jitter, in a registered eye frame.** Every result is resampled onto the same 33x33
normalized eye grid, so the same cell means the same piece of iris in every frame no
matter where the eye is. Per cell, the departure from a five-frame rolling median of
itself: a smooth drift as the eye turns costs nothing, a one-frame excursion costs
everything. The SOURCE's own cells are the paired control, over the same cells and the
same frames -- an unanchored figure here is worthless, which is how round 32 reported a
median below its own control while plainly flickering. Median, p95 and max, never a
median alone.

**3. Specular presence.** The measure round 33 had no reason to take and would have
failed: the peak display level inside the iris. The source has a catchlight, round 33's
output has nothing at all, and an eye with no specular reads as dead however stable it
is. Reported for the source, the removal and the removal-plus-scene-reflection.

Withheld-pixel accuracy is NOT re-derived here. The prediction is round 33's atlas built
by the same code with the same parameters, so `logs/r33_withheld.json` still describes
it: 29 % / 18 % better than round 32 on 5,739 / 1,599 withheld pixels across a temporal
split, at 1.36x / 1.24x the plate's own texture floor.
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

GRID = 33
REACH = 0.95


def eye_grid():
    t = np.linspace(-REACH, REACH, GRID).astype(np.float32)
    return np.meshgrid(t, t)


def sample(image, limbus, ax, ay, u, v):
    mx = (limbus[0] + u * ax).astype(np.float32)
    my = (limbus[1] + v * ay).astype(np.float32)
    return cv2.remap(image, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def roughness_stack(stack, cells):
    """Per-cell departure from a five-frame rolling median of the same cell.

    `stack` is (frames, GRID, GRID, 3) in display levels. Returns one value per
    (frame, cell) pair over `cells`, so the distribution is over the whole shot rather
    than one number per frame.
    """
    a = np.asarray(stack, np.float32)
    n = len(a)
    out = []
    for i in range(n):
        lo, hi = max(0, i - 2), min(n, i + 3)
        med = np.median(a[lo:hi], axis=0)
        d = np.abs(a[i] - med).mean(axis=-1)
        out.append(d[cells])
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def summarize(values, scale=1.0):
    a = np.asarray(values, float) * scale
    if not a.size:
        return None
    return dict(n=int(a.size), median=round(float(np.median(a)), 4),
                p95=round(float(np.percentile(a, 95)), 4),
                max=round(float(a.max()), 4))


def load(path):
    im = cv2.imread(str(path))
    return None if im is None else im[..., ::-1].astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_shot_argument(ap)
    ap.add_argument('--new', default='r34_removal')
    ap.add_argument('--old', default='r33_repair')
    ap.add_argument('--count', type=int, default=150)
    ap.add_argument('--out', default='r34_eyes.json')
    args = ap.parse_args()
    shot = from_args(args)
    new_root = shot.composite / args.new
    old_root = shot.composite / args.old
    log = json.loads((shot.log(args.new + '.json')).read_text())
    rows = {r['frame']: r for r in log['rows']}
    has_replace = (new_root / 'replace').is_dir()
    topology = json.loads((shot.relight / 'face_landmarks.json').read_text())['topology']
    with np.load(shot.relight / 'face_landmarks.npz') as z:
        points, found = z['points'], z['found']

    u, v = eye_grid()
    rho = np.hypot(u, v)
    # rho < 0.80, not 0.88. At 0.88 the sampled cell straddles the limbus wherever the
    # landmark centre is a pixel off, and the sclera behind it is the brightest thing in
    # the eye -- which made an earlier pass report an identical 177.77 "peak after
    # removal" for both rounds, on a pixel neither of them touched.
    iris = rho < 0.80

    stacks = {s: {k: [] for k in ('source', 'new', 'old', 'replace')} for s in r32.SIDES}
    changed = {s: np.zeros((GRID, GRID), bool) for s in r32.SIDES}
    # A cell inside rho < 0.88 of the LIMBUS is not automatically iris: when the lid comes
    # down it is bright skin, and skin is the brightest thing anywhere near the eye. An
    # earlier pass of this audit skipped the gate and reported a 214-level "iris peak"
    # identical in both rounds, because the brightest pixel was an eyelid neither round
    # touched. Every measure here is over cells the lid leaves open.
    visible_count = {s: np.zeros((GRID, GRID), np.int32) for s in r32.SIDES}
    frames_used = {s: [] for s in r32.SIDES}
    per_frame = []
    for i in range(args.count):
        row = rows.get(i)
        if row is None:
            continue
        images = {'source': load(new_root / 'source' / ('%06d.png' % i)),
                  'new': load(new_root / 'removal' / ('%06d.png' % i)),
                  'old': load(old_root / 'repair' / ('%06d.png' % i))}
        if has_replace:
            images['replace'] = load(new_root / 'replace' / ('%06d.png' % i))
        if images['source'] is None or images['new'] is None:
            continue
        wnew = cv2.imread(str(new_root / 'weight' / ('%06d.png' % i)), 0)
        wold = cv2.imread(str(old_root / 'mask' / ('%06d.png' % i)), 0)
        entry = dict(frame=i, source_frame=row['source_frame'], eyes=[])
        for side, rec in zip(r32.SIDES, row['eyes']):
            if 'limbus_centre' not in rec:
                continue
            limbus = np.asarray(rec['limbus_centre'], float)
            ax, ay = rec['limbus_axes']
            if not (found[i] and np.isfinite(points[i]).all()):
                continue
            geom, _ = r32.eye_frame(images['source'] / 255.0, points[i], topology, side)
            if geom is None:
                continue
            win = geom['window']
            lid_full = np.zeros(images['source'].shape[:2], np.float32)
            lid_full[win] = r32.lid_distance(points[i], topology['eyelid_' + side], win)
            open_cells = sample(lid_full, limbus, ax, ay, u, v) > 0.85
            treated = {}
            for tag, wimg in (('new', wnew), ('old', wold)):
                treated[tag] = (np.zeros((GRID, GRID), bool) if wimg is None else
                                sample(wimg.astype(np.float32), limbus, ax, ay, u, v) > 8)
            g = {k: (None if im is None else sample(im, limbus, ax, ay, u, v))
                 for k, im in images.items()}
            if g['source'] is None or g['new'] is None or g['old'] is None:
                continue
            for k in ('source', 'new', 'old', 'replace'):
                stacks[side][k].append(g[k] if g.get(k) is not None
                                       else np.zeros_like(g['source']))
            frames_used[side].append(i)
            visible_count[side] += (open_cells & iris).astype(np.int32)
            open_iris = open_cells & iris
            if open_iris.sum() < 20:
                continue
            for w in (wnew, wold):
                if w is None:
                    continue
                m = sample(w.astype(np.float32), limbus, ax, ay, u, v)
                changed[side] |= (m > 8)
            # Residue and specular presence, in display levels over the iris disc.
            e = dict(side=side,
                     source_peak=round(float(g['source'][open_iris].max()), 2),
                     new_peak=round(float(g['new'][open_iris].max()), 2),
                     old_peak=round(float(g['old'][open_iris].max()), 2),
                     iris_median=round(float(np.median(g['source'][open_iris])), 2))
            if g.get('replace') is not None:
                e['replace_peak'] = round(float(g['replace'][open_iris].max()), 2)
            # Residue is measured ONLY where that round actually changed pixels, against
            # the iris it is supposed to match. Measured over the whole iris it reports
            # the limbal ring in every frame and says nothing about the highlight.
            iris_level = float(np.median(g['source'][open_iris]))
            e['iris_level'] = round(iris_level, 2)
            # The UNION of the two rounds' treated cells, so residue is paired: measured
            # over each round's own footprint the two are not comparable, because round
            # 34's continuous weight deliberately touches a wider, softer region.
            cells = (treated['new'] | treated['old']) & open_iris
            e['treated_cells'] = int(cells.sum())
            # NO display-space residue metric is reported here, and that is deliberate.
            # Three were tried. Peak-minus-iris-median reports the iris's own centre-to-
            # limbus gradient in every frame; peak-minus-collar reports the black coding
            # undershoot that sits beside the highlight and is left alone by design; and
            # both are dominated by structure neither round touches. Residue against the
            # ATLAS PREDICTION is the only reference that means anything, both rounds
            # compute it against the same atlas, and both log it per eye
            # (`residual_pixels_above_r33_threshold` in round 34's log is round 33's own
            # criterion recomputed, so the two are directly comparable). What is measured
            # here instead is what the artist actually sees: the peak inside the treated
            # region, and the jitter of every cell.
            if cells.sum() >= 3:
                e['source_peak_treated'] = round(float(g['source'][cells].max()), 2)
                for tag, arr in (('new', g['new']), ('old', g['old'])):
                    e[tag + '_treated_peak'] = round(float(arr[cells].max()), 2)
                if g.get('replace') is not None:
                    e['replace_treated_peak'] = round(float(g['replace'][cells].max()), 2)
            entry['eyes'].append(e)
        per_frame.append(entry)
        print('R34_AUDIT frame', i, flush=True)

    result = dict(new=args.new, old=args.old, frames=len(per_frame),
                  grid=GRID, reach=REACH, per_frame=per_frame, by_side={})
    for side in r32.SIDES:
        st = stacks[side]
        if len(st['source']) < 6:
            continue
        always_open = visible_count[side] >= int(0.9 * len(st['source']))
        cells = changed[side] & iris & always_open
        if cells.sum() < 4:
            cells = iris & always_open
        entry = dict(frames=len(st['source']), changed_cells=int(cells.sum()),
                     cells_open_in_90pct=int(always_open.sum()))
        for k in ('source', 'old', 'new', 'replace'):
            if k == 'replace' and not has_replace:
                continue
            entry['jitter_' + k] = summarize(roughness_stack(st[k], cells))
        eyes = [e for f in per_frame for e in f['eyes'] if e['side'] == side]
        for key in ('source_peak', 'old_peak', 'new_peak', 'replace_peak',
                    'source_peak_treated', 'old_treated_peak', 'new_treated_peak',
                    'replace_treated_peak', 'iris_level'):
            vals = [e[key] for e in eyes if key in e]
            if vals:
                entry[key] = summarize(vals)
        for key in ('residual_pixels_above_r33_threshold', 'weighted_pixels',
                    'pixels_touched', 'peak_excess_sigmas'):
            vals = [e.get(key) for f in log['rows'] for e in f['eyes']
                    if e['side'] == side and e.get(key) is not None]
            if vals:
                entry[key] = summarize(vals)
        catch = [e.get('catchlight_lid_kept') for f in log['rows'] for e in f['eyes']
                 if e['side'] == side and e.get('catchlight_lid_kept') is not None]
        if catch:
            entry['catchlight_lid_kept'] = summarize(catch)
        peaks = [e.get('catchlight_peak_added') for f in log['rows'] for e in f['eyes']
                 if e['side'] == side and e.get('catchlight_peak_added') is not None]
        if peaks:
            entry['catchlight_peak_added_linear'] = summarize(peaks)
        result['by_side'][side] = entry
        print('R34_AUDIT', side, json.dumps(entry, indent=1), flush=True)

    result['note'] = (
        'Jitter is per cell in a limbus-registered eye grid, against a five-frame rolling '
        'median of the same cell, over cells either round ever changed. The source is the '
        'paired control on identical cells and frames. Peaks and residues are display '
        'levels 0-255 inside the iris disc.')
    r32.write_json(shot.log(args.out), result)
    print('R34_AUDIT_DONE ->', shot.log(args.out), flush=True)


if __name__ == '__main__':
    main()
