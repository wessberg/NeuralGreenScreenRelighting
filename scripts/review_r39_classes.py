"""Put every eye of an audit class in front of a person: what the pass believed, on her image.

For each listed eye (and its neighbours), one column: SOURCE, RESULT, and SOURCE again with
what the pass used drawn on it -- the limbus circle it painted on (cyan), the fitted upper
and lower lids (yellow), the lash line bounding the fill (green), the fill's outline
(magenta) and the softbox region replaced whole (red). Crops are STABILISED on the eye
corners (their midpoint plus the eye's median offset from them over +/-1 s), not on the
limbus track: round 37 section 5 -- any view that rides the tracker hides the tracker's
error, and a slide then looks like the eye moving.

    dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/review_r39_classes.py \
        --shot music_videos/reporter_test/shots/reporter_full --name r39_gpu \
        --classes logs/r39_classes_r39_gpu.json --cls painted_skin
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shot_paths import add_shot_argument, from_args
import eye_reflection_repair as r32
import eye_soft_removal as esr
import eye_removal_gpu as R
from limbus_fit import Track

Z = 8          # zoom
HALF = 16      # crop half-size, native px


def stabilised_centres(track, corners, frames, side, rate):
    """Corner midpoint + the median (limbus - corners) offset over +/- 1 s."""
    off = {}
    for i in frames:
        got = track.get(i, side)
        c = corners.get((i, side))
        if got is not None and c is not None:
            off[i] = np.asarray(got[0], float) - c
    out = {}
    half = max(1, int(round(float(rate))))
    for i in frames:
        c = corners.get((i, side))
        if c is None:
            got = track.get(i, side)
            if got is not None:
                out[i] = np.asarray(got[0], float)
            continue
        near = [off[j] for j in range(i - half, i + half + 1) if j in off]
        out[i] = c + (np.median(near, axis=0) if near else 0.0)
    return out


def tile(img, cx, cy, label):
    x0, y0 = int(round(cx)) - HALF, int(round(cy)) - HALF
    crop = img[max(0, y0):y0 + 2 * HALF, max(0, x0):x0 + 2 * HALF]
    crop = cv2.copyMakeBorder(crop, 0, 2 * HALF - crop.shape[0], 0, 2 * HALF - crop.shape[1],
                              cv2.BORDER_CONSTANT)
    t = cv2.resize(crop, None, fx=Z, fy=Z, interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(t, (0, 0), (t.shape[1], 14), (0, 0, 0), -1)
    cv2.putText(t, label, (3, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return t, x0, y0


def P(x, y, x0, y0):
    """Native coords -> zoomed tile pixel centre, with 4 bits of sub-pixel shift."""
    return (int(round(((x - x0) + 0.5) * Z * 16)), int(round(((y - y0) + 0.5) * Z * 16)))


def overlay(t, x0, y0, d, rec):
    if rec.get('limbus_centre'):
        cx, cy = rec['limbus_centre']
        r = rec['limbus_axes'][0]
        cv2.circle(t, P(cx, cy, x0, y0), int(round(r * Z * 16)), (255, 255, 0), 1, cv2.LINE_AA, 4)
    if d is None:
        return t
    lids = d['lids']
    xs = np.arange(d['x0'], d['x0'] + d['w'], 0.25)
    for off, colr in ((0, (0, 230, 255)), (3, (0, 160, 255))):
        a, b, c = lids[off:off + 3]
        if np.isfinite(a):
            u = xs - lids[6]
            ys = a * u ** 2 + b * u + c
            pts = np.array([P(x, y, x0, y0) for x, y in zip(xs, ys)], np.int32)
            cv2.polylines(t, [pts], False, colr, 1, cv2.LINE_AA, 4)
    fl = d.get('floor_rows')
    if fl is not None and np.isfinite(fl).any():
        # the lower lid over her iris (--lid-floor), white; drawn only inside the window
        cols = np.arange(d['w'])
        ok = np.isfinite(fl) & (fl > -2) & (fl < d['h'] + 2)
        if ok.sum() > 1:
            pts = np.array([P(d['x0'] + x, d['y0'] + y - 0.5, x0, y0)
                            for x, y in zip(cols[ok], fl[ok])], np.int32)
            cv2.polylines(t, [pts], False, (255, 255, 255), 1, cv2.LINE_AA, 4)
    ce = d['ceiling']
    if np.isfinite(ce).any():
        cols = np.arange(d['w'])
        ok = np.isfinite(ce)
        pts = np.array([P(d['x0'] + x, d['y0'] + y - 0.5, x0, y0) for x, y in zip(cols[ok], ce[ok])], np.int32)
        cv2.polylines(t, [pts], False, (60, 255, 60), 1, cv2.LINE_AA, 4)
    for plane, colr in (('w_fill', (255, 60, 255)), ('wb', (60, 60, 255))):
        m = (d[plane] >= 128).astype(np.uint8)
        if not m.any():
            continue
        big = np.zeros((t.shape[0], t.shape[1]), np.uint8)
        ox, oy = (d['x0'] - x0) * Z, (d['y0'] - y0) * Z
        mz = cv2.resize(m, None, fx=Z, fy=Z, interpolation=cv2.INTER_NEAREST)
        ys_, xs_ = slice(max(0, oy), min(t.shape[0], oy + mz.shape[0])), slice(max(0, ox), min(t.shape[1], ox + mz.shape[1]))
        big[ys_, xs_] = mz[ys_.start - oy:ys_.stop - oy, xs_.start - ox:xs_.stop - ox]
        cnt, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(t, cnt, -1, colr, 1)
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_shot_argument(ap)
    ap.add_argument('--name', required=True, help='the run under composite/')
    ap.add_argument('--ref', default=None, help='a second run shown as an extra RESULT row')
    ap.add_argument('--classes', default=None, help='audit_r39_classes JSON (relative to shot)')
    ap.add_argument('--cls', default=None, help='class name in that JSON')
    ap.add_argument('--items', default='', help="explicit 'frame:side,...' instead of a class")
    ap.add_argument('--neighbours', type=int, default=1)
    ap.add_argument('--cols', type=int, default=10)
    ap.add_argument('--limit', type=int, default=400)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    shot = from_args(args)
    run = shot.composite / args.name
    rec = json.loads((run / 'removal.json').read_text(encoding='utf-8'))
    rows = {r['frame']: r for r in rec['rows']}
    diag = R.load_diag(run / 'diag.npz') if (run / 'diag.npz').exists() else {}
    track = Track.load(shot.relight / 'limbus_track.json')
    meta = json.loads((shot.relight / 'face_landmarks.json').read_text())
    with np.load(shot.relight / 'face_landmarks.npz') as z:
        corners = esr.eye_corners_mid(z['points'], z['found'], meta['topology'])
    if args.items:
        items = [(int(a), b) for a, b in (x.split(':') for x in args.items.split(','))]
    else:
        cl = json.loads((shot.dir / args.classes).read_text(encoding='utf-8'))['classes'][args.cls]
        items = [(c['frame'], c['side']) for c in cl]
    items = items[:args.limit]
    out_dir = shot.composite / 'r39_review'
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.out or (args.cls or 'items')
    for side in r32.SIDES:
        want = sorted({f for f, s in items if s == side})
        if not want:
            continue
        frames = sorted({j for f in want for j in range(f - args.neighbours, f + args.neighbours + 1)
                         if 0 <= j < shot.frame_count})
        cen = stabilised_centres(track, corners, frames, side, shot.frame_rate)
        tiles = []
        for i in frames:
            if i not in cen:
                continue
            cx, cy = cen[i]
            src = cv2.imread(str(run / 'source' / ('%06d.png' % i)))
            res = cv2.imread(str(run / 'removal' / ('%06d.png' % i)))
            erec = next((e for e in rows[i]['eyes'] if e['side'] == side), {})
            mark = '*' if i in want else ''
            state = ('ok' if erec.get('repaired') else (erec.get('reason') or '-'))
            if erec.get('fill_abstained'):
                state += ' ' + erec['fill_abstained'].split('_')[0]
            a, x0, y0 = tile(src, cx, cy, 'SRC %d%s' % (i, mark))
            b, _, _ = tile(res, cx, cy, '%s %s' % (args.name, state[:22]))
            col = [a, b]
            if args.ref:
                ref = cv2.imread(str(shot.composite / args.ref / 'removal' / ('%06d.png' % i)))
                col.append(tile(ref, cx, cy, args.ref)[0])
            c, _, _ = tile(src, cx, cy, 'what it used')
            col.append(overlay(c, x0, y0, diag.get((i, side)), erec))
            tiles.append(np.vstack(col))
        for k in range(0, len(tiles), args.cols):
            blk = tiles[k:k + args.cols]
            blk += [np.zeros_like(blk[0])] * (args.cols - len(blk))
            sheet = np.hstack([np.pad(t, ((0, 0), (0, 3), (0, 0))) for t in blk])
            p = out_dir / ('%s_%s_%02d.png' % (tag, side, k // args.cols))
            cv2.imwrite(str(p), sheet)
        print(side, len(tiles), 'tiles ->', out_dir / ('%s_%s_*.png' % (tag, side)))


if __name__ == '__main__':
    main()
