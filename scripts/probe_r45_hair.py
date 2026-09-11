"""Round 45: measure the capture room's band in her hair BEFORE building anything to remove it.

    dependencies/CorridorKey/.venv/Scripts/python.exe scripts/probe_r45_hair.py --shot <dir> \\
        --frames 30,73,97,120 --out r45_hair_probe.json --sheet r45_hair_probe.png

Round 44 SS5 established two things that this probe does not re-derive: the crown band is 23x
the dark hair beside it, and it sits at the SAME chroma, so no dichromatic split can touch it.
What round 45 needs to know before it writes a repair is different, and all of it is here:

1. LEVEL. Against a local dark-hair baseline (a low percentile inside the hair mask, over a
   window in face-radius units, so it does not scale with this clip), how cleanly does a
   threshold separate the band? Reported as the baseline, the band fraction, and the ratio
   profile -- if the band is 95 % specular the excess over the baseline IS the room's sheen.
2. STRANDS. Does the structure tensor actually find an orientation in this hair? Reported as
   the coherence distribution over the region and over the band alone. A replacement specular
   needs a tangent; if the coherence is low the tangent is a guess and the round must say so.
3. GEOMETRY. Does the pipeline have what a rendered hair specular needs on those pixels --
   proxy coverage, a rendered irradiance, a neural normal that is not the silhouette's edge?
4. TIME. In head-stabilised coordinates, does the band's footprint stay put? That decides
   whether the removal can be held in time the way sclera_glint_removal.py holds its glint.

Nothing is written but the report and a sheet to look at.
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import sys, json, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np, cv2
from shot_paths import add_shot_argument, from_args
from color_pipeline import linear_to_srgb
import hair_region as hr

LUMA = np.array([.2126, .7152, .0722], np.float32)
RATIO_ALBEDO = 0.6


def local_baseline(lum, mask, sigma, pct=25.0):
    """Local low percentile of the hair's luminance -- the dark hair the band sits on.

    A percentile has no closed form under a Gaussian, so this is the standard cheap stand-in:
    the masked mean minus k standard deviations, with k chosen for the requested percentile of
    a normal. It is computed on the LOG so it is a ratio statistic, and it is normalised by the
    blurred mask so the silhouette does not pull it toward zero.
    """
    from scipy.stats import norm
    k = float(norm.ppf(pct / 100.0))
    m = mask.astype(np.float32)
    l = np.log2(np.maximum(lum, 1e-5)).astype(np.float32) * m
    den = np.maximum(cv2.GaussianBlur(m, (0, 0), sigma), 1e-4)
    mu = cv2.GaussianBlur(l, (0, 0), sigma) / den
    mu2 = cv2.GaussianBlur(l * l * 0 + (l ** 2), (0, 0), sigma) / den
    sd = np.sqrt(np.maximum(mu2 - mu * mu, 0.0))
    return np.exp2(mu + k * sd)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_shot_argument(ap)
    ap.add_argument('--frames', default='30,73,97,120')
    ap.add_argument('--temporal', default='70-80', help='first-last for the footprint stability test')
    ap.add_argument('--render-dir', default='ratio_render')
    ap.add_argument('--baseline-sigma', type=float, default=0.30,
                    help='baseline window, in FACE RADII (her own face, not pixels)')
    ap.add_argument('--baseline-pct', type=float, default=25.0)
    ap.add_argument('--out', default='r45_hair_probe.json')
    ap.add_argument('--sheet', default='r45_hair_probe.png')
    args = ap.parse_args()
    shot = from_args(args)
    topo = json.loads((shot.relight / 'face_landmarks.json').read_text())['topology']
    with np.load(shot.relight / 'face_landmarks.npz') as z:
        pts_all, found = z['points'], z['found']
    frames = [int(x) for x in args.frames.split(',')]
    rep = {'frames': {}, 'why': __doc__.splitlines()[0]}
    tiles = []
    for f in frames:
        if not found[f]:
            continue
        p = pts_all[f]
        c, rad, chin = hr.face_frame(p, topo)
        alpha = cv2.imread(str(shot.alpha(f)), -1)
        alpha = alpha[..., 0] if alpha.ndim == 3 else alpha
        h, w = alpha.shape
        raw = cv2.imread(str(shot.relight / 'skin_mask' / ('%06d.png' % f)), cv2.IMREAD_UNCHANGED)
        scale = {np.dtype(np.uint16): 65535.0, np.dtype(np.uint8): 255.0}.get(raw.dtype, 1.0)
        skin = (raw[..., 0] if raw.ndim == 3 else raw).astype(np.float32) / scale
        hair = hr.hair_mask(alpha, skin, p, topo)
        crown = hr.crown_mask(hair)
        fg = cv2.imread(str(shot.fg(f)), -1)[..., ::-1].astype(np.float32)
        lum = fg @ LUMA
        sig = args.baseline_sigma * rad
        base = local_baseline(lum, hair, sig, args.baseline_pct)
        ratio = lum / np.maximum(base, 1e-5)
        row = {'face_radius_px': rad, 'hair_px': int(hair.sum()), 'crown_px': int(crown.sum()),
               'baseline_sigma_px': sig}
        for nm, m in (('hair', hair), ('crown', crown)):
            if m.sum() < 100:
                continue
            v, b, rt = lum[m], base[m], ratio[m]
            row[nm] = {'lum_p10': float(np.percentile(v, 10)), 'lum_p50': float(np.median(v)),
                       'lum_p90': float(np.percentile(v, 90)), 'lum_p99': float(np.percentile(v, 99)),
                       'baseline_p50': float(np.median(b)),
                       'ratio_p50': float(np.median(rt)), 'ratio_p90': float(np.percentile(rt, 90)),
                       'ratio_p99': float(np.percentile(rt, 99)),
                       'frac_over_2x': float((rt > 2).mean()), 'frac_over_4x': float((rt > 4).mean()),
                       'frac_over_8x': float((rt > 8).mean())}
        # 2. strands
        t2, coh = hr.structure_tangent(lum, hair)
        band = hair & (ratio > 4.0)
        row['coherence'] = {'hair_p50': float(np.median(coh[hair])) if hair.any() else 0.0,
                            'hair_p90': float(np.percentile(coh[hair], 90)) if hair.any() else 0.0,
                            'band_p50': float(np.median(coh[band])) if band.any() else 0.0,
                            'band_px': int(band.sum())}
        # A tangent has no head and no tail, so its angle lives mod 180 deg and an ordinary
        # median of it is meaningless (the first run of this probe reported an IQR of 92-100 deg
        # on a field that is visibly ordered). Circular statistics on the DOUBLE angle instead:
        # the resultant length R is 0 for a uniform field and 1 for one direction.
        if band.any():
            ang2 = 2.0 * np.arctan2(t2[..., 1], t2[..., 0])
            cs, sn = np.cos(ang2[band]).mean(), np.sin(ang2[band]).mean()
            R = float(np.hypot(cs, sn))
            row['coherence']['band_angle_mean_deg'] = float(np.degrees(np.arctan2(sn, cs)) / 2.0 % 180.0)
            row['coherence']['band_angle_resultant'] = R
            row['coherence']['band_angle_circ_sd_deg'] = float(np.degrees(np.sqrt(max(-2.0 * np.log(max(R, 1e-9)), 0.0))) / 2.0)
        # 3. geometry
        rimg = cv2.imread(str(shot.composite / args.render_dir / ('%06d.exr' % f)), -1)
        E = rimg[..., 2::-1].astype(np.float32) / RATIO_ALBEDO
        cov = rimg[..., 3].astype(np.float32) if rimg.shape[2] == 4 else np.ones((h, w), np.float32)
        if E.shape[:2] != (h, w):
            E = cv2.resize(E, (w, h), interpolation=cv2.INTER_LINEAR)
            cov = cv2.resize(cov, (w, h), interpolation=cv2.INTER_LINEAR)
        nrm = np.load(shot.relight / 'normals' / ('%06d.npz' % f))['normal'].astype(np.float32)
        nrm = cv2.resize(nrm, (w, h), interpolation=cv2.INTER_LINEAR)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-6)
        row['geometry'] = {'cov_over_half_frac_hair': float((cov[hair] > 0.5).mean()),
                           'cov_over_half_frac_band': float((cov[band] > 0.5).mean()) if band.any() else 0.0,
                           'E_lum_p50_hair': float(np.median(E[hair] @ LUMA)),
                           'E_lum_p50_band': float(np.median(E[band] @ LUMA)) if band.any() else 0.0,
                           'normal_z_p50_crown': float(np.median(nrm[crown][:, 2])) if crown.any() else 0.0,
                           'normal_len_ok': float((np.abs(np.linalg.norm(nrm[hair], axis=-1) - 1) < 1e-3).mean())}
        rep['frames'][f] = row
        # sheet tile: plate crop, the band in red, the tangent as short strokes
        vis = (np.clip(linear_to_srgb(np.clip(fg, 0, 1)), 0, 1) * 255).astype(np.uint8)[..., ::-1].copy()
        ov = vis.copy()
        ov[band] = (0, 0, 255)
        vis = cv2.addWeighted(vis, 0.55, ov, 0.45, 0)
        step = max(int(rad / 14), 6)
        for y in range(0, h, step):
            for x in range(0, w, step):
                if hair[y, x] and coh[y, x] > 0.3:
                    dx, dy = t2[y, x] * step * 0.45
                    cv2.line(vis, (int(x - dx), int(y - dy)), (int(x + dx), int(y + dy)), (0, 255, 255), 1, cv2.LINE_AA)
        x0, x1 = int(c[0] - 1.25 * rad), int(c[0] + 1.25 * rad)
        y0, y1 = int(c[1] - 1.55 * rad), int(c[1] + 0.35 * rad)
        crop = vis[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
        crop = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
        cv2.putText(crop, 'f%d  band>4x  coh>0.3' % f, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(crop, 'f%d  band>4x  coh>0.3' % f, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(crop)

    # 4. TIME: the band's footprint in head-stabilised coordinates.
    a, b = (int(x) for x in args.temporal.split('-'))
    seq, ref = [], None
    for f in range(a, b + 1):
        if not found[f]:
            continue
        p = pts_all[f]
        alpha = cv2.imread(str(shot.alpha(f)), -1)
        alpha = alpha[..., 0] if alpha.ndim == 3 else alpha
        h, w = alpha.shape
        raw = cv2.imread(str(shot.relight / 'skin_mask' / ('%06d.png' % f)), cv2.IMREAD_UNCHANGED)
        scale = {np.dtype(np.uint16): 65535.0, np.dtype(np.uint8): 255.0}.get(raw.dtype, 1.0)
        skin = (raw[..., 0] if raw.ndim == 3 else raw).astype(np.float32) / scale
        hair = hr.hair_mask(alpha, skin, p, topo)
        _, rad, _ = hr.face_frame(p, topo)
        fg = cv2.imread(str(shot.fg(f)), -1)[..., ::-1].astype(np.float32)
        lum = fg @ LUMA
        base = local_baseline(lum, hair, args.baseline_sigma * rad, args.baseline_pct)
        rt = lum / np.maximum(base, 1e-5)
        band = (hair & (rt > 4.0)).astype(np.float32)
        # A strand-level threshold is a wiry footprint that 1 px of registration error already
        # halves, so the same test is also run on the ENVELOPE: the sheen field low-passed at
        # 0.05 face radii, which is what a removal keys on.
        env = cv2.GaussianBlur(np.maximum(rt - 1.0, 0.0) * hair, (0, 0), 0.05 * rad)
        envb = (hair & (env > np.percentile(env[hair], 85))).astype(np.float32)
        if ref is None:
            ref, refe, refp = band, envb, pts_all[f][:, :2].astype(np.float32)
            seq.append({'frame': f, 'iou': 1.0, 'iou_envelope': 1.0})
            continue
        M, _ = cv2.estimateAffinePartial2D(pts_all[f][:, :2].astype(np.float32), refp, method=cv2.LMEDS)
        row = {'frame': f}
        for key, cur, r0f in (('iou', band, ref), ('iou_envelope', envb, refe)):
            bw = cv2.warpAffine(cur, M, (w, h), flags=cv2.INTER_NEAREST) > 0.5
            r0 = r0f > 0.5
            row[key] = float((bw & r0).sum()) / max(float((bw | r0).sum()), 1.0)
        seq.append(row)
    rep['footprint_stability'] = {'window': args.temporal, 'iou_vs_first': seq,
                                  'iou_p50': float(np.median([s['iou'] for s in seq[1:]])) if len(seq) > 1 else 0.0,
                                  'iou_envelope_p50': float(np.median([s['iou_envelope'] for s in seq[1:]])) if len(seq) > 1 else 0.0,
                                  'note': 'band>4x warped onto the first frame by a similarity fit to all landmarks'}
    (shot.logs / args.out).write_text(json.dumps(rep, indent=2) + '\n')
    if tiles:
        hm = min(t.shape[0] for t in tiles)
        sheet = np.concatenate([t[:hm] for t in tiles], 1)
        out = shot.composite / 'panels' / args.sheet
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), sheet)
        print('sheet:', out, sheet.shape)
    print(json.dumps({k: v for k, v in rep.items() if k != 'frames'}, indent=2)[:1200])
    for f, row in rep['frames'].items():
        print('f%s: hair %d px, crown %d px, crown ratio p90 %.2f p99 %.2f, >4x %.3f, coh hair %.2f band %.2f, cov %.2f, angle %.0f+-%.0f' % (
            f, row['hair_px'], row['crown_px'], row.get('crown', {}).get('ratio_p90', 0),
            row.get('crown', {}).get('ratio_p99', 0), row.get('crown', {}).get('frac_over_4x', 0),
            row['coherence']['hair_p50'], row['coherence']['band_p50'],
            row['geometry']['cov_over_half_frac_band'],
            row['coherence'].get('band_angle_mean_deg', -1), row['coherence'].get('band_angle_circ_sd_deg', -1)))


if __name__ == '__main__':
    main()
