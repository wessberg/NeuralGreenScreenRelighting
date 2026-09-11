"""Round 45: does the shine in her hair RESPOND to the new light -- measured two ways, because
round 44's statistic cannot answer it once the band is removed.

    dependencies/CorridorKey/.venv/Scripts/python.exe scripts/audit_r45_hair.py --shot <dir> \\
        --dirs r44_m50_s05,r45_rmonly,r45_hair35 --light-pass r44_light_bulb --out r45_hair_audit.json

ROUND 44'S TEST, AND WHY IT BREAKS HERE. `audit_r44_sheen.py` reports `sheen` = p95 / p50 of
the region's luminance and correlates it with log E over the tour. The ratio is deliberately
invariant to a global multiply, which is the whole point: `plate x R` rescales a photographed
highlight along with the hair under it, so a photographed band holds its ratio and a real one
does not. That works while the denominator is a real number. It stops working the moment the
band is SUBTRACTED: the region's median then falls to the composite's own floor at the dark
stations, and p95 / p50 becomes the ratio of two near-zero numbers -- measured on this shot as
a "sheen" of 14 000 at the dark stations against 11 at the bright ones, which then drags the
correlation hard negative no matter what the highlight does. The number is reported here beside
the median that produces it, so the artefact is visible rather than inherited.

TWO STATISTICS THAT DO SURVIVE THE REMOVAL.

1. SPECULAR FRACTION. `(sum(lum) - p20(lum) * n) / sum(lum)` over the region: the share of the
   region's light that stands above its own dark-hair level. Numerator and denominator both
   scale with R, so it is invariant to a global multiply exactly as round 44's ratio is, but it
   is bounded in [0, 1] and cannot explode. A photographed band holds it constant across the
   tour (correlation ~ 0 or negative); a highlight made by the new light rises where that light
   reaches the hair.

2. DOES THE BAND MOVE? The centroid of the region's brightest decile, in IPD units relative to
   the face centre, against the AZIMUTH of the rendered source read from the light pass's own
   manifest. A photographed band sits still while the source swings through 360 degrees; a
   rendered one tracks it. Reported as the correlation of the centroid's horizontal offset with
   sin(azimuth) and the peak-to-peak swing, which is the least circular test available: the
   source position comes from the render rig and the centroid from the output picture, and the
   two share nothing.

The region is `audit_r44_sheen.py`'s, unchanged and deliberately NOT the mask `hair_sheen.py`
acts through -- an acceptance test must not share its subject's definition of where to look.
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import sys, json, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np, cv2
from shot_paths import add_shot_argument, from_args
from color_pipeline import srgb_to_linear

LUMA = np.array([.2126, .7152, .0722], np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_shot_argument(ap)
    ap.add_argument('--dirs', required=True)
    ap.add_argument('--render-dir', default='ratio_render')
    ap.add_argument('--light-pass', default='r44_light_bulb',
                    help='a stress_proxy --only-light run; its manifest gives the source position')
    ap.add_argument('--out', default='r45_hair_audit.json')
    args = ap.parse_args()
    shot = from_args(args)
    topo = json.loads((shot.relight / 'face_landmarks.json').read_text())['topology']
    with np.load(shot.relight / 'face_landmarks.npz') as z:
        pts_all, found = z['points'], z['found']
    runs = args.dirs.split(',')

    azim = {}
    man = shot.composite / args.light_pass / 'render_manifest.json'
    if man.exists():
        for r in json.loads(man.read_text())['per_frame']:
            b = r.get('bulb_world') or r.get('fill_world')
            t = r.get('target_world') or [0.0, 0.0, 0.0]
            if b:
                d = np.array(b, float) - np.array(t, float)
                azim[r['out_frame']] = float(np.degrees(np.arctan2(d[0], d[1])))

    frames = sorted(int(p.stem) for p in (shot.composite / runs[0]).glob('[0-9]*.png'))
    rows = []
    for f in frames:
        if not found[f]:
            continue
        p = pts_all[f]
        ipd = float(np.linalg.norm(p[33, :2] - p[263, :2]))
        alpha = cv2.imread(str(shot.alpha(f)), -1)
        alpha = alpha[..., 0] if alpha.ndim == 3 else alpha
        h, w = alpha.shape
        skin = cv2.imread(str(shot.relight / 'skin_mask' / ('%06d.png' % f)), -1)
        skin = skin[..., 0] if skin.ndim == 3 else skin
        yy, xx = np.mgrid[0:h, 0:w]
        fc = (p[10, :2] + p[151, :2]) / 2
        brow_y = min(p[105, 1], p[334, 1])
        hair = ((alpha > 0.99) & (skin < 32767) & (yy < brow_y - 0.15 * ipd)
                & (np.hypot(xx - fc[0], yy - fc[1]) < 0.9 * ipd))
        if hair.sum() < 200:
            continue
        E = cv2.imread(str(shot.composite / args.render_dir / ('%06d.exr' % f)), -1)[..., 2::-1] @ LUMA
        row = {'frame': f, 'E_hair': float(np.median(E[hair])), 'azimuth_deg': azim.get(f)}
        for run in runs:
            a = cv2.imread(str(shot.composite / run / ('%06d.png' % f)))
            if a is None:
                continue
            if a.shape[:2] != (h, w):
                a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
            lum = srgb_to_linear(a[..., ::-1].astype(np.float32) / 255.0) @ LUMA
            v = lum[hair]
            med, p20, p95 = float(np.median(v)), float(np.percentile(v, 20)), float(np.percentile(v, 95))
            total = float(v.sum())
            top = v >= np.percentile(v, 90)
            row[run] = {'level_p50': med, 'sheen_r44': p95 / max(med, 1e-9),
                        'specular_fraction': max(total - p20 * v.size, 0.0) / max(total, 1e-12),
                        'bright_x_ipd': float((xx[hair][top].mean() - fc[0]) / ipd),
                        'bright_y_ipd': float((yy[hair][top].mean() - fc[1]) / ipd)}
        rows.append(row)

    Ev = np.array([r['E_hair'] for r in rows])
    lo, hi = np.percentile(Ev, 20), np.percentile(Ev, 80)
    # Every ratio statistic here is a ratio of percentiles INSIDE the region, and at the dark
    # half of the tour the composite puts that whole region on its own floor -- so the ratio is
    # a ratio of floor values and says nothing about a highlight that is not there to see. The
    # lit half is where the question "does the shine follow the light" is answerable at all, and
    # it is reported separately rather than averaged into the noise.
    lit = Ev >= np.percentile(Ev, 50)
    az = np.array([r['azimuth_deg'] if r['azimuth_deg'] is not None else np.nan for r in rows])
    summary = {}
    for run in runs:
        recs = [r[run] for r in rows if run in r]
        if len(recs) != len(rows):
            continue
        sf = np.array([r['specular_fraction'] for r in recs])
        sh = np.array([r['sheen_r44'] for r in recs])
        lv = np.array([r['level_p50'] for r in recs])
        bx = np.array([r['bright_x_ipd'] for r in recs])
        ok = np.isfinite(az)
        summary[run] = {
            'specular_fraction_p50': float(np.median(sf)),
            'specular_fraction_dark_stations': float(np.median(sf[Ev <= lo])),
            'specular_fraction_bright_stations': float(np.median(sf[Ev >= hi])),
            'corr_specular_fraction_logE': float(np.corrcoef(sf, np.log(np.maximum(Ev, 1e-6)))[0, 1]),
            'sheen_r44_p50': float(np.median(sh)),
            'sheen_r44_dark_stations': float(np.median(sh[Ev <= lo])),
            'corr_sheen_r44_logE': float(np.corrcoef(sh, np.log(np.maximum(Ev, 1e-6)))[0, 1]),
            'region_level_p50_dark_stations': float(np.median(lv[Ev <= lo])),
            'region_level_p50_bright_stations': float(np.median(lv[Ev >= hi])),
            'bright_x_swing_ipd': float(np.percentile(bx, 95) - np.percentile(bx, 5)),
            'corr_bright_x_with_sin_azimuth': (
                float(np.corrcoef(bx[ok], np.sin(np.radians(az[ok])))[0, 1]) if ok.sum() > 5 else None),
            'lit_half': {
                'frames': int(lit.sum()),
                'specular_fraction_p50': float(np.median(sf[lit])),
                'corr_specular_fraction_logE': float(np.corrcoef(sf[lit], np.log(np.maximum(Ev[lit], 1e-6)))[0, 1]),
                'sheen_r44_p50': float(np.median(sh[lit])),
                'corr_sheen_r44_logE': float(np.corrcoef(sh[lit], np.log(np.maximum(Ev[lit], 1e-6)))[0, 1]),
                'corr_bright_x_with_sin_azimuth': (
                    float(np.corrcoef(bx[lit & ok], np.sin(np.radians(az[lit & ok])))[0, 1])
                    if (lit & ok).sum() > 5 else None)}}
    out = {'summary': summary, 'per_frame': rows, 'runs': runs,
           'region': "audit_r44_sheen.py's hair crown, unchanged -- an acceptance test must not "
                     'share its subject definition of where to look',
           'why': 'round 44 p95/p50 explodes once the band is subtracted because the median falls '
                  'to the composite floor at the dark stations; specular_fraction is invariant to a '
                  'global multiply in the same way but is bounded in [0,1]'}
    (shot.logs / args.out).write_text(json.dumps(out, indent=2) + '\n')
    print('LIT HALF (the frames where the statistic is answerable):')
    for run, v in summary.items():
        h = v['lit_half']
        print('  %-16s corr(r44 sheen, logE) %+.3f   corr(spec fraction, logE) %+.3f   '
              'corr(bright-x, sin azimuth) %s   [%d frames]'
              % (run, h['corr_sheen_r44_logE'], h['corr_specular_fraction_logE'],
                 ('%+.3f' % h['corr_bright_x_with_sin_azimuth']) if h['corr_bright_x_with_sin_azimuth'] is not None else 'n/a',
                 h['frames']))
    print('WHOLE TOUR (the dark half puts the region on the composite floor):')
    for run, v in summary.items():
        print('%-16s sf p50 %.3f  dark %.3f  bright %.3f  corr(sf,logE) %+.3f | r44 sheen %8.1f (dark %10.1f) corr %+.3f | '
              'level dark %.4f bright %.4f | bright-x swing %.3f IPD corr(azim) %s'
              % (run, v['specular_fraction_p50'], v['specular_fraction_dark_stations'],
                 v['specular_fraction_bright_stations'], v['corr_specular_fraction_logE'],
                 v['sheen_r44_p50'], v['sheen_r44_dark_stations'], v['corr_sheen_r44_logE'],
                 v['region_level_p50_dark_stations'], v['region_level_p50_bright_stations'],
                 v['bright_x_swing_ipd'],
                 ('%+.3f' % v['corr_bright_x_with_sin_azimuth']) if v['corr_bright_x_with_sin_azimuth'] is not None else 'n/a'))


if __name__ == '__main__':
    main()
