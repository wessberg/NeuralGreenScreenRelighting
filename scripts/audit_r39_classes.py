"""Find EVERY eye in each of round 38's failure classes, over the whole clip.

Round 38 section 6 and its evidence frames (rounds/evidence/round38/): the artist caught
three on her left eye -- 137 (the whole disc painted over a squint), 243 (a double iris
from a limbus slide), 304 (iris painted over the lid) -- and round 38 found more by looking
(e4 sideways gaze, e5 fast motion). His direction for round 39: *find ALL frames of each
class automatically, not just the ones he screenshotted.*

Each class is measured from what the pass WROTE (source / removal / weight PNGs), plus the
pass's own dump (`eye_removal_gpu.py --dump`, for where it believed the lids and the
softbox were), and one estimator that shares nothing with the tracker:

* **painted_skin** (e1, e3): fill painted where the SOURCE is lid skin. Iris, pupil and lash
  are dark on this plate and lid skin is bright; the corneal highlight is bright too, so
  the pass's own geometric softbox region is excluded. The skin threshold is per eye,
  halfway between her iris level (inside the disc) and her skin level (a band above it).
* **limbus_offset** (e2, e5): the tracked centre against `audit_r35_limbus.measure` -- a
  max-channel level-crossing estimator with a first-harmonic displacement model, which
  shares no feature, statistic or sampling with `limbus_fit.py`.
* **softbox_left** (e4, e5): a compact bright blob with a dark surround still in the
  RESULT inside the eye, found by the same test the atlas used for surviving softboxes
  (`corneal_specular`, read off the result alone). Split by whether the pass treated the
  eye, and whether the eye was open (landmark aperture).

    dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/audit_r39_classes.py \
        --shot music_videos/reporter_test/shots/reporter_full --name r39_gpu
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from color_pipeline import srgb_to_linear
from shot_paths import add_shot_argument, from_args
import eye_reflection_repair as r32
import eye_gpu as G
import eye_removal_gpu as R
import audit_r35_limbus as a35
from limbus_fit import Track


def crop(img, x0, y0, w, h):
    return img[y0:y0 + h, x0:x0 + w]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_shot_argument(ap)
    ap.add_argument('--name', required=True)
    ap.add_argument('--limbus-track', default='limbus_track.json')
    ap.add_argument('--skin-px', type=int, default=6,
                    help='painted skin pixels at which an eye is listed')
    ap.add_argument('--offset-px', type=float, default=1.5,
                    help='independent limbus offset at which an eye is listed')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    shot = from_args(args)
    run = shot.composite / args.name
    rec = json.loads((run / 'removal.json').read_text(encoding='utf-8'))
    diag = R.load_diag(run / 'diag.npz') if (run / 'diag.npz').exists() else {}
    track = Track.load(shot.relight / args.limbus_track)
    radius = json.loads((shot.relight / args.limbus_track).read_text())['shot_radius']
    meta = json.loads((shot.relight / 'face_landmarks.json').read_text())
    topology = meta['topology']
    with np.load(shot.relight / 'face_landmarks.npz') as z:
        points, found, aperture = z['points'], z['found'], z['eyelid_aperture']
    th = np.arange(a35.ANGLES, dtype=np.float32) * (2 * np.pi / a35.ANGLES)
    cos, sin = np.cos(th), np.sin(th)
    eyes = []
    for row in rec['rows']:
        i = row['frame']
        if not (found[i] and np.isfinite(points[i]).all()):
            continue
        src = cv2.imread(str(run / 'source' / ('%06d.png' % i)))[..., ::-1]
        res = cv2.imread(str(run / 'removal' / ('%06d.png' % i)))[..., ::-1]
        wgt = cv2.imread(str(run / 'weight' / ('%06d.png' % i)), cv2.IMREAD_UNCHANGED)
        srcm = src.max(-1).astype(np.float32) / 255.0
        resm = res.max(-1).astype(np.float32) / 255.0
        for side, er in zip(r32.SIDES, row['eyes']):
            k = r32.SIDES.index(side)
            ap_ = float(aperture[i][k]) if np.isfinite(aperture[i][k]) else None
            if er.get('limbus_centre'):
                cx, cy = er['limbus_centre']
            else:
                got = track.get(i, side)
                if got is None:
                    lm, _, _ = r32.limbus_ellipse(points[i], topology['iris_' + side])
                    cx, cy = float(lm[0]), float(lm[1])
                else:
                    cx, cy = got[0]
            r = float(radius[side])
            x0, y0 = int(np.floor(cx - 2.2 * r)), int(np.floor(cy - 2.2 * r))
            w = h = int(np.ceil(4.4 * r)) + 1
            if x0 < 0 or y0 < 0 or x0 + w > src.shape[1] or y0 + h > src.shape[0]:
                continue
            e = dict(frame=i, side=side, repaired=bool(er.get('repaired')),
                     reason=er.get('reason'), fill_abstained=er.get('fill_abstained'),
                     aperture=ap_, centre=[cx, cy], x0=x0, y0=y0, w=w, h=h)
            e['src_m'] = crop(srcm, x0, y0, w, h)
            e['res_m'] = crop(resm, x0, y0, w, h)
            e['wgt'] = crop(wgt, x0, y0, w, h)
            # independent limbus estimate, on the source crop's max channel
            lin = srgb_to_linear(e['src_m'].astype(np.float32))
            lid = r32.lid_distance(points[i], topology['eyelid_' + side],
                                   (slice(y0, y0 + h), slice(x0, x0 + w)))
            m = a35.measure(e['src_m'], lid, 0, 0, (cx - x0, cy - y0), r, cos, sin)
            e['limbus_offset_px'] = None if m is None or not m.get('refined') else m['offset_px']
            e['limbus_offset_xy'] = None if m is None or not m.get('refined') else m['offset_xy']
            d = diag.get((i, side))
            e['painted_skin_px'] = 0
            e['painted_px'] = 0
            if d is not None:
                ox, oy = d['x0'] - x0, d['y0'] - y0
                sm = e['src_m'][oy:oy + d['h'], ox:ox + d['w']]
                if sm.shape == d['w_fill'].shape:
                    yy, xx = np.mgrid[:d['h'], :d['w']]
                    rho = np.hypot(xx + d['x0'] - d['limbus'][0], yy + d['y0'] - d['limbus'][1]) / d['ax']
                    iris = (rho > 0.45) & (rho < 0.85) & d['visible'] & (d['wb'] == 0)
                    band = (rho > 1.35) & (rho < 1.9) & (yy + d['y0'] < d['limbus'][1])
                    iris_lv = float(np.median(sm[iris])) if iris.sum() > 20 else 0.25
                    skin_lv = float(np.percentile(sm[band], 75)) if band.sum() > 20 else 0.7
                    thr = 0.5 * (iris_lv + skin_lv)
                    painted = (d['w_fill'] >= 128) & (d['wb'] == 0)
                    e['painted_px'] = int(painted.sum())
                    e['painted_skin_px'] = int((painted & (sm > thr)).sum())
                    e['skin_threshold'] = round(thr, 3)
            eyes.append(e)
    # softbox left in the RESULT (and in the source, for context), batched on the GPU
    for key in ('res_m', 'src_m'):
        by_shape = {}
        for n, e in enumerate(eyes):
            by_shape.setdefault(e[key].shape, []).append(n)
        for shape, idx in by_shape.items():
            h, w = shape
            lum = G.as_t(np.stack([eyes[n][key] for n in idx]))
            lx = torch.tensor([eyes[n]['centre'][0] - eyes[n]['x0'] for n in idx], device=G.DEV, dtype=torch.float64).view(-1, 1, 1)
            ly = torch.tensor([eyes[n]['centre'][1] - eyes[n]['y0'] for n in idx], device=G.DEV, dtype=torch.float64).view(-1, 1, 1)
            rr = torch.tensor([float(radius[eyes[n]['side']]) for n in idx], device=G.DEV, dtype=torch.float64).view(-1, 1, 1)
            ys = torch.arange(h, device=G.DEV, dtype=torch.float64).view(1, h, 1)
            xs = torch.arange(w, device=G.DEV, dtype=torch.float64).view(1, 1, w)
            rho = torch.hypot((xs - lx) / rr, (ys - ly) / rr)
            core, fnd = R.corneal_specular(lum, rho)
            px = core.sum((1, 2)).cpu().numpy()
            pk = torch.where(core, lum, torch.zeros_like(lum)).reshape(len(idx), -1).max(1).values.cpu().numpy()
            for k_, n in enumerate(idx):
                eyes[n]['softbox_px_' + key[:3]] = int(px[k_])
                eyes[n]['softbox_peak_' + key[:3]] = round(float(pk[k_]), 3)
    classes = dict(painted_skin=[], limbus_offset=[], softbox_left_treated=[],
                   softbox_left_untreated_open=[], softbox_left_untreated_closed=[])
    for e in eyes:
        tag = dict(frame=e['frame'], side=e['side'], repaired=e['repaired'], reason=e['reason'],
                   fill_abstained=e['fill_abstained'], aperture=e['aperture'])
        if e['painted_skin_px'] >= args.skin_px:
            classes['painted_skin'].append(dict(tag, painted_skin_px=e['painted_skin_px'],
                                                painted_px=e['painted_px']))
        if e['limbus_offset_px'] is not None and e['limbus_offset_px'] >= args.offset_px:
            classes['limbus_offset'].append(dict(tag, offset_px=round(e['limbus_offset_px'], 2),
                                                 offset_xy=[round(x, 2) for x in e['limbus_offset_xy']]))
        if e.get('softbox_px_res', 0) > 0:
            t = dict(tag, softbox_px=e['softbox_px_res'], peak=e['softbox_peak_res'])
            if e['repaired']:
                classes['softbox_left_treated'].append(t)
            elif e['aperture'] is not None and e['aperture'] >= 0.8:
                classes['softbox_left_untreated_open'].append(t)
            else:
                classes['softbox_left_untreated_closed'].append(t)

    def runs(lst):
        out = {}
        for side in r32.SIDES:
            fr = sorted(x['frame'] for x in lst if x['side'] == side)
            rr, start, prev = [], None, None
            for f in fr:
                if prev is None or f != prev + 1:
                    if start is not None:
                        rr.append('%d-%d' % (start, prev) if prev > start else str(start))
                    start = f
                prev = f
            if start is not None:
                rr.append('%d-%d' % (start, prev) if prev > start else str(start))
            out[side] = rr
        return out
    summary = {k: dict(eyes=len(v), runs=runs(v)) for k, v in classes.items()}
    off = [e['limbus_offset_px'] for e in eyes if e['limbus_offset_px'] is not None]
    report = dict(name=args.name, eyes_audited=len(eyes), summary=summary,
                  limbus_offset_px=dict(median=float(np.median(off)), p95=float(np.percentile(off, 95)),
                                        max=float(np.max(off)), n=len(off)),
                  painted_skin_px_total=int(sum(e['painted_skin_px'] for e in eyes)),
                  classes=classes,
                  thresholds=dict(skin_px=args.skin_px, offset_px=args.offset_px),
                  estimators=dict(
                      painted_skin='fill >= 0.5 outside the geometric softbox region where the '
                                   'SOURCE max channel is above halfway between her iris and '
                                   'her skin level (per eye)',
                      limbus_offset='audit_r35_limbus.measure: max-channel level crossing, '
                                    'first harmonic, two passes -- independent of limbus_fit',
                      softbox_left='corneal_specular on the RESULT alone'))
    out = Path(args.out) if args.out else shot.log('r39_classes_%s.json' % args.name)
    r32.write_json(out, report)
    print(json.dumps(dict(eyes=len(eyes), summary=summary,
                          limbus_offset_px=report['limbus_offset_px'],
                          painted_skin_px_total=report['painted_skin_px_total']), indent=1))
    print('wrote', out)


if __name__ == '__main__':
    main()
