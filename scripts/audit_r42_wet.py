"""Where does the capture room's light show on the WET parts of her eye that are not the iris?

Round 42 starts from *"maybe the eye didn't become 'wet' enough"*. Round 41's reflection stops
at the limbus because the mirror is a corneal cap; the white of the eye and the tear line along
the lower lid are photographed under the OLD light and the ratio can only recolour them. Before
a second reflecting surface is built, this measures, on every open eye of the untouched source:

1. **Bright blobs outside the limbus and inside the fitted opening** (round 38's local-contrast
   test), each classified by where it sits: on the BULK sclera (>= `--margin` px from both
   lids), on the LOWER LID MARGIN (within `--margin` px above the lower parabola), on the upper
   margin, or touching a corner. Position in limbus radii, clock angle, nasal/temporal.
2. **The lower-lid band**: display luma sampled 1 px above the lower parabola, column by
   column across the opening, against the sclera 4.5 px above it. Where along the lid the
   tear line is brightest, and by how much.
3. **Two geometric predictions**, per eye, from the same sphere the renderer places
   (`corneal_anchor.anchor_limbus`) and the softbox direction round 41 fitted on the corneal
   glint: (a) where that softbox would reflect in a scleral GLOBE (radius `--globe-mm`, centre
   `--globe-back-mm` behind the corneal centre along the gaze) -- as a limbus radius; if it is
   under the cornea, no scleral glint is expected and none should be measured; (b) the column
   where a tear MENISCUS along the lower lid (a thin tube on the globe following the fitted
   parabola) would reflect it: the half-vector perpendicular to the tube's tangent. Compared
   with the measured column of the lower-lid band maximum.

Diagnostic only; writes logs/<out> and an annotated crop sheet. Pure numpy/OpenCV.

    dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/audit_r42_wet.py \
        --shot music_videos/reporter_test/shots/reporter_a
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shot_paths import add_shot_argument, from_args
from corneal_anchor import smoothed_iris_centres, anchor_limbus, reflect_point, project
from eyelid_fit import evaluate as lid_evaluate
from validate_tour_catchlight import load_track, camera_of, sphere_for, open_eyes

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def ray_sphere(cam, ray, C, R):
    """First hit of cam + t*ray on the sphere (C, R), or None."""
    oc = cam - C
    b = ray @ oc
    c = oc @ oc - R * R
    disc = b * b - c
    if disc < 0:
        return None
    t = -b - np.sqrt(disc)
    return cam + t * ray if t > 0 else None


def pixel_ray(px, py, cam):
    ray = np.array([(px + 0.5 - cam['W'] / 2.0) / cam['fx'], 1.0,
                    -(py + 0.5 - cam['H'] / 2.0) / cam['fx']])
    return ray / np.linalg.norm(ray)


def meniscus_columns(lid, cx, r, Ce, Re, cam, l_dir, span=1.7):
    """Columns (in limbus radii from the iris centre) where a distant source `l_dir` reflects
    in a tube that follows the fitted lower lid on the globe: sign changes of h . t."""
    us = np.arange(-span * r, span * r + 0.5, 1.0)
    pts, cols = [], []
    for u in us:
        x = cx + u
        y = lid_evaluate(lid['lower']['coef'], x, lid['centre'])
        p = ray_sphere(cam['pos'], pixel_ray(x, y, cam), Ce, Re)
        if p is None:
            continue
        pts.append(p)
        cols.append(u / r)
    if len(pts) < 5:
        return [], None
    pts = np.asarray(pts)
    t = np.gradient(pts, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
    v = cam['pos'] - pts
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    h = v + l_dir[None, :]
    h /= np.linalg.norm(h, axis=1, keepdims=True)
    dots = (h * t).sum(1)
    out = []
    for k in range(len(dots) - 1):
        if dots[k] == 0 or dots[k] * dots[k + 1] < 0:
            out.append(float(cols[k] + (cols[k + 1] - cols[k]) * abs(dots[k])
                             / (abs(dots[k]) + abs(dots[k + 1]) + 1e-12)))
    return out, dots


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_shot_argument(ap)
    ap.add_argument('--limbus-track', default='limbus_track_tv_import.json')
    ap.add_argument('--eyelid-track', default='eyelid_track_wtv40_import.json')
    ap.add_argument('--source-run', default='r40_floorB_import',
                    help='composite/<run>/source holds the untouched source PNGs')
    ap.add_argument('--softbox', default='',
                    help='logs/<json> of validate_tour_catchlight --plate-glint; the geometry '
                         "model's fitted direction is the capture room's softbox")
    ap.add_argument('--globe-mm', type=float, default=12.0)
    ap.add_argument('--globe-back-mm', type=float, default=5.6,
                    help='globe centre this far behind the corneal centre, along the gaze')
    ap.add_argument('--margin', type=float, default=2.5, help='px from a lid = margin, not bulk')
    ap.add_argument('--excess', type=float, default=0.10)
    ap.add_argument('--floor', type=float, default=0.80)
    ap.add_argument('--sheet-frames', default='0,20,40,60,80,100,120,140')
    ap.add_argument('--out', default='r42_wet_audit.json')
    args = ap.parse_args()
    shot = from_args(args)
    cam = camera_of(shot)
    eye_geo = {r['frame']: r for r in json.loads(
        (shot.relight / 'eye_geometry.json').read_text())['per_frame']}
    track = load_track(shot, args.limbus_track)
    smooth = smoothed_iris_centres(track)
    lids = open_eyes(shot, args.eyelid_track)
    src_dir = shot.composite / args.source_run / 'source'
    sb = None
    if args.softbox:
        v = json.loads((shot.logs / args.softbox).read_text())
        sb = np.asarray(v['plate_glint']['geometry']['fitted_direction_world'], float)
        sb /= np.linalg.norm(sb)
    sheet_frames = {int(x) for x in args.sheet_frames.split(',') if x}
    tiles = []
    rows = []
    for fi in sorted(track):
        rec = eye_geo.get(fi) or {}
        img = None
        for side in ('left', 'right'):
            tr = (track[fi].get('eyes') or {}).get(side)
            e = ((rec.get('eyes') or {}).get(side)) or {}
            lid = lids.get((fi, side))
            if tr is None or not e.get('visible') or lid is None:
                continue
            px = tr['iris_centre_px']
            sm = smooth.get(side, {}).get(fi)
            if sm is not None:
                px = (float(sm[0]), float(sm[1]))
            cx, cy = px
            r = float(tr['iris_radius_px'])
            lo_c = lid_evaluate(lid['lower']['coef'], cx, lid['centre'])
            up_c = lid_evaluate(lid['upper']['coef'], cx, lid['centre'])
            if lo_c - up_c < 1.4 * r:
                continue
            if img is None:
                im = cv2.imread(str(src_dir / ('%06d.png' % fi)), -1)
                img = (im[..., ::-1].astype(np.float32) / 255.0)
                lum = img @ LUMA
            reach = 3.0 * r + 6
            x0, x1 = int(cx - reach), int(cx + reach) + 1
            y0, y1 = int(cy - 2.2 * r), int(cy + 2.2 * r) + 1
            if x0 < 0 or y0 < 0 or x1 > lum.shape[1] or y1 > lum.shape[0]:
                continue
            lw = lum[y0:y1, x0:x1]
            yy, xx = np.mgrid[y0:y1, x0:x1]
            up = lid_evaluate(lid['upper']['coef'], xx.astype(float), lid['centre'])
            lo = lid_evaluate(lid['lower']['coef'], xx.astype(float), lid['centre'])
            if lid.get('lower_iris'):
                lo = np.minimum(lo, lid_evaluate(lid['lower_iris']['coef'], xx.astype(float),
                                                 lid['centre']))
            below_up = yy - up            # px below the upper margin
            above_lo = lo - yy            # px above the lower margin
            opening = (below_up > 1.0) & (above_lo > 1.0)
            rho = np.hypot(xx - cx, yy - cy) / r
            region = opening & (rho > 1.10) & (rho < 2.8)
            med = cv2.medianBlur((lw * 255).astype(np.uint8), 9).astype(np.float32) / 255.
            hot = region & (lw - med > args.excess) & (lw > args.floor)
            blobs = []
            if hot.any():
                n, lab, stats, cent = cv2.connectedComponentsWithStats(hot.astype(np.uint8))
                edge = cv2.dilate((~opening).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
                for k in range(1, n):
                    area = int(stats[k, cv2.CC_STAT_AREA])
                    if area > 60:
                        continue
                    m = lab == k
                    bx, by = cent[k][0] + x0, cent[k][1] + y0
                    a_lo = float(lid_evaluate(lid['lower']['coef'], bx, lid['centre']) - by)
                    b_up = float(by - lid_evaluate(lid['upper']['coef'], bx, lid['centre']))
                    ang = float(np.degrees(np.arctan2(-(by - cy), bx - cx)))
                    # her left eye is on image right; its nasal side is image-left of it
                    nasal = (bx < cx) if side == 'left' else (bx > cx)
                    where = ('corner' if (m & edge).any() and abs(bx - cx) > 1.3 * r else
                             'lower_margin' if a_lo < args.margin else
                             'upper_margin' if b_up < args.margin else 'bulk')
                    blobs.append(dict(area=area, excess=round(float((lw - med)[m].max()), 3),
                                      peak=round(float(lw[m].max()), 3),
                                      rho=round(float(np.hypot(bx - cx, by - cy) / r), 2),
                                      angle_deg=round(ang, 1), where=where,
                                      nasal=bool(nasal), above_lower_px=round(a_lo, 2),
                                      below_upper_px=round(b_up, 2),
                                      xy=[round(float(bx), 1), round(float(by), 1)]))
            # the lower-lid band: 1 px above the parabola vs 4.5 px above, per column
            us = np.arange(-1.6 * r, 1.6 * r + 0.5, 1.0)
            xs = cx + us
            ys_lo = lid_evaluate(lid['lower']['coef'], xs, lid['centre'])
            if lid.get('lower_iris'):
                ys_lo = np.minimum(ys_lo, lid_evaluate(lid['lower_iris']['coef'], xs,
                                                       lid['centre']))
            ys_up = lid_evaluate(lid['upper']['coef'], xs, lid['centre'])
            ok = (ys_lo - ys_up) > 3.0

            def samp(xq, yq):
                return cv2.remap(lum, xq.astype(np.float32)[None], yq.astype(np.float32)[None],
                                 cv2.INTER_LINEAR)[0]
            band = samp(xs, ys_lo - 1.0)
            scl = samp(xs, ys_lo - 4.5)
            band[~ok] = 0
            k = int(np.argmax(band))
            band_rec = dict(max=round(float(band[k]), 3), col_r=round(float(us[k] / r), 2),
                            excess_over_sclera=round(float(band[k] - scl[k]), 3),
                            mean_excess=round(float(np.mean((band - scl)[ok])) if ok.any()
                                              else 0.0, 3),
                            bright_cols=int(((band - scl > args.excess) & ok).sum()))
            pred = {}
            if sb is not None:
                C, R, g = sphere_for(e, tr, px, 'geometry', cam)
                Ce = C - (args.globe_back_mm / 1000.0) * g
                Re = args.globe_mm / 1000.0
                p, _ = reflect_point(Ce, Re, cam['pos'], sb, at_infinity=True)
                q = project(p, cam['pos'], cam['fx'], cam['W'], cam['H'])
                pred['globe_glint_px'] = [round(float(q[0]), 2), round(float(q[1]), 2)]
                pred['globe_glint_rho'] = round(float(np.hypot(q[0] - cx, q[1] - cy) / r), 3)
                pred['globe_under_cornea'] = bool(pred['globe_glint_rho'] < 1.07)
                cols, dots = meniscus_columns(lid, cx, r, Ce, Re, cam, sb)
                pred['meniscus_cols_r'] = [round(c, 2) for c in cols]
                if dots is not None:
                    pred['meniscus_min_abs_dot'] = round(float(np.abs(dots).min()), 3)
            rows.append(dict(frame=fi, side=side, iris_r=round(r, 2), blobs=blobs,
                             lower_band=band_rec, predicted=pred))
            if fi in sheet_frames:
                Z = 8
                crop = (np.clip(img[y0:y1, x0:x1], 0, 1) * 255).astype(np.uint8)[..., ::-1]
                big = cv2.resize(crop, None, fx=Z, fy=Z, interpolation=cv2.INTER_NEAREST)
                for xq, yq in zip(xs, ys_lo):
                    cv2.circle(big, (int((xq - x0 + 0.5) * Z), int((yq - y0 + 0.5) * Z)), 1,
                               (0, 255, 0), -1)
                for xq, yq in zip(xs, ys_up):
                    cv2.circle(big, (int((xq - x0 + 0.5) * Z), int((yq - y0 + 0.5) * Z)), 1,
                               (0, 200, 255), -1)
                cv2.circle(big, (int((cx - x0 + 0.5) * Z), int((cy - y0 + 0.5) * Z)),
                           int(r * Z), (255, 255, 0), 1)
                for b in blobs:
                    col = {'bulk': (0, 0, 255), 'lower_margin': (255, 0, 255),
                           'upper_margin': (255, 128, 0), 'corner': (0, 128, 255)}[b['where']]
                    cv2.circle(big, (int((b['xy'][0] - x0 + 0.5) * Z),
                                     int((b['xy'][1] - y0 + 0.5) * Z)), 3 * Z, col, 1)
                if pred.get('globe_glint_px'):
                    gx, gy = pred['globe_glint_px']
                    cv2.drawMarker(big, (int((gx - x0 + 0.5) * Z), int((gy - y0 + 0.5) * Z)),
                                   (255, 255, 255), cv2.MARKER_CROSS, 12, 1)
                for c in pred.get('meniscus_cols_r', []):
                    xq = cx + c * r
                    yq = lid_evaluate(lid['lower']['coef'], xq, lid['centre'])
                    cv2.drawMarker(big, (int((xq - x0 + 0.5) * Z), int((yq - y0 + 0.5) * Z)),
                                   (255, 0, 0), cv2.MARKER_TRIANGLE_UP, 12, 1)
                cv2.putText(big, '%d %s' % (fi, side[0].upper()), (4, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                tiles.append(big)
        if fi % 30 == 0:
            print('R42_WET frame', fi, flush=True)

    # ---- pool ---------------------------------------------------------------------------
    summary = {}
    for side in ('left', 'right'):
        mine = [r for r in rows if r['side'] == side]
        by_where = {}
        for r in mine:
            for b in r['blobs']:
                by_where.setdefault(b['where'], []).append(b)
        wsum = {}
        for w, bl in by_where.items():
            ang = np.array([b['angle_deg'] for b in bl])
            wsum[w] = dict(blobs=len(bl), frames=len({id(b) for b in bl}),
                           nasal_fraction=round(float(np.mean([b['nasal'] for b in bl])), 2),
                           rho_median=round(float(np.median([b['rho'] for b in bl])), 2),
                           angle_median_deg=round(float(np.median(ang)), 1),
                           angle_p5_p95=[round(float(np.percentile(ang, 5)), 1),
                                         round(float(np.percentile(ang, 95)), 1)],
                           above_lower_px_median=round(float(np.median(
                               [b['above_lower_px'] for b in bl])), 2),
                           excess_median=round(float(np.median([b['excess'] for b in bl])), 3),
                           area_median=float(np.median([b['area'] for b in bl])))
        band = np.array([r['lower_band']['max'] for r in mine])
        col = np.array([r['lower_band']['col_r'] for r in mine])
        exc = np.array([r['lower_band']['excess_over_sclera'] for r in mine])
        bright = [r for r in mine if r['lower_band']['excess_over_sclera'] > args.excess]
        s = dict(open_eyes=len(mine), blobs_by_where=wsum,
                 frames_with_any_blob=sum(1 for r in mine if r['blobs']),
                 frames_with_bulk_blob=sum(1 for r in mine
                                           if any(b['where'] == 'bulk' for b in r['blobs'])),
                 lower_band=dict(max_median=round(float(np.median(band)), 3),
                                 excess_median=round(float(np.median(exc)), 3),
                                 excess_p95=round(float(np.percentile(exc, 95)), 3),
                                 frames_bright=len(bright),
                                 bright_col_r_median=round(float(np.median(
                                     [r['lower_band']['col_r'] for r in bright])), 2)
                                 if bright else None,
                                 bright_col_r_p5_p95=[round(float(np.percentile(
                                     [r['lower_band']['col_r'] for r in bright], q)), 2)
                                     for q in (5, 95)] if bright else None))
        if sb is not None:
            g_rho = np.array([r['predicted']['globe_glint_rho'] for r in mine])
            s['globe_prediction'] = dict(rho_median=round(float(np.median(g_rho)), 3),
                                         rho_max=round(float(g_rho.max()), 3),
                                         under_cornea_fraction=round(float(np.mean(
                                             g_rho < 1.07)), 3))
            errs = []
            for r in bright:
                cols = r['predicted'].get('meniscus_cols_r') or []
                if cols:
                    errs.append(min(abs(c - r['lower_band']['col_r']) for c in cols) * r['iris_r'])
            s['meniscus_prediction'] = dict(
                eyes_with_solution=sum(1 for r in mine if r['predicted'].get('meniscus_cols_r')),
                bright_eyes_scored=len(errs),
                col_error_px_median=round(float(np.median(errs)), 2) if errs else None,
                col_error_px_p95=round(float(np.percentile(errs, 95)), 2) if errs else None,
                col_error_px_max=round(float(np.max(errs)), 2) if errs else None)
        summary[side] = s
    dest = shot.logs / args.out
    dest.write_text(json.dumps(dict(summary=summary, softbox_dir=(sb.tolist() if sb is not None
                                                                   else None),
                                    globe_mm=args.globe_mm, globe_back_mm=args.globe_back_mm,
                                    margin_px=args.margin, rows=rows), indent=1),
                    encoding='utf-8')
    if tiles:
        hmax = max(t.shape[0] for t in tiles)
        wmax = max(t.shape[1] for t in tiles)
        padded = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, wmax - t.shape[1],
                                     cv2.BORDER_CONSTANT) for t in tiles]
        per_row = 4
        rows_img = [np.concatenate(padded[i:i + per_row] + [np.zeros_like(padded[0])] *
                                   (per_row - len(padded[i:i + per_row])), 1)
                    for i in range(0, len(padded), per_row)]
        sheet = np.concatenate(rows_img, 0)
        sp = shot.composite / 'r42_wet_audit'
        sp.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(sp / 'sheet.png'), sheet)
        print('R42_WET_SHEET', sp / 'sheet.png')
    print(json.dumps(summary, indent=1))
    print('R42_WET_WROTE', dest)


if __name__ == '__main__':
    main()
