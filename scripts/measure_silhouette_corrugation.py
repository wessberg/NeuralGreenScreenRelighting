"""Measure the dotted rim on the proxy silhouette, and show it side by side.

    dependencies/CorridorKey/.venv/Scripts/python.exe scripts/measure_silhouette_corrugation.py \
        --shot <dir> --candidate-render composite/r4_mesh_fixed \
        --baseline-composite composite/r4_baseline_comp \
        --candidate-composite composite/r4_mesh_fixed_comp

The artefact the artist named -- dots along the image-left shoulder in 000079 and along the hand
in 000051 -- is a corrugation at the proxy's own grid pitch. The mesh is sampled every `step`
depth-map pixels, so on a 768x432 depth grid at step 2 the facets land every 5 plate pixels. That
is a specific spatial frequency, which makes the artefact measurable rather than a matter of
opinion.

Method: band-pass the rendered irradiance around the grid pitch, express it as a fraction of the
local irradiance so that a brightness difference between two runs cannot masquerade as a texture
difference, and take the RMS inside a rim band around the silhouette.

Two controls, because a rim number alone proves nothing:

* the matte CORE, well inside the silhouette, where the same statistic measures ordinary shading
  variation rather than an edge artefact;
* the keyed PLATE through the identical filter, which is the photograph's own micro-contrast in
  the same band. That is a reference, not a floor the composite has to reach: the plate has hair
  and fabric at this frequency and a smooth irradiance field should sit well below it.

The crop for the visual panels is chosen by finding the worst rim patch in the BASELINE, so the
comparison is centred on the artefact rather than on a region picked by eye, and both runs are
shown at the same place.
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

LUMA = np.array([.2126, .7152, .0722], np.float32)


def relative_corrugation(rgb, pitch_px):
    """Band-passed modulation around the grid pitch, as a fraction of the local level."""
    luma = np.ascontiguousarray(rgb[..., :3].astype(np.float32)) @ LUMA
    bp = (cv2.GaussianBlur(luma, (0, 0), pitch_px / 4.0)
          - cv2.GaussianBlur(luma, (0, 0), pitch_px / 1.5))
    local = np.maximum(cv2.GaussianBlur(luma, (0, 0), pitch_px * 2.0), 1e-4)
    return np.abs(bp) / local


def rms(x, m):
    return float(np.sqrt(np.mean(np.square(x[m])))) if m.any() else float('nan')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_shot_argument(ap)
    ap.add_argument('--baseline-render', default='', help='default composite/ratio_render')
    ap.add_argument('--candidate-render', required=True)
    ap.add_argument('--baseline-composite', required=True, help='composited PNGs for the baseline')
    ap.add_argument('--candidate-composite', required=True)
    ap.add_argument('--frames', default='51,79')
    ap.add_argument('--pitch-px', type=float, default=0.0,
                    help='proxy grid pitch in plate pixels; derived from the depth grid if unset')
    ap.add_argument('--step', type=int, default=2, help='the step make_proxy_mesh was run with')
    ap.add_argument('--crop', type=int, nargs=2, default=(360, 330))
    ap.add_argument('--zoom', type=int, default=2)
    ap.add_argument('--name', default='r4_compare')
    # The panels outlive the run that made them, so what they are comparing has to be on
    # the image rather than in the command line that produced it.
    ap.add_argument('--baseline-label', default='original mesh')
    ap.add_argument('--candidate-label', default='candidate mesh')
    args = ap.parse_args()
    shot = from_args(args)

    base_r = Path(args.baseline_render) if args.baseline_render else shot.composite / 'ratio_render'
    cand_r = Path(args.candidate_render)
    base_c = Path(args.baseline_composite)
    cand_c = Path(args.candidate_composite)
    out = shot.composite / args.name
    out.mkdir(parents=True, exist_ok=True)
    cw, ch = args.crop
    per_frame = []

    for fi in [int(f) for f in args.frames.split(',')]:
        alpha = cv2.imread(str(shot.alpha(fi)), cv2.IMREAD_UNCHANGED)
        if alpha.ndim == 3:
            alpha = alpha[..., 0]
        alpha = np.clip(alpha.astype(np.float32), 0, 1)
        H, W = alpha.shape
        pitch = args.pitch_px
        if not pitch:
            dh, dw = np.load(shot.relight / 'depth' / ('%06d.npy' % fi)).shape[:2]
            pitch = args.step * (W / float(dw))

        b_e = cv2.imread(str(base_r / ('%06d.exr' % fi)), cv2.IMREAD_UNCHANGED)
        c_e = cv2.imread(str(cand_r / ('%06d.exr' % fi)), cv2.IMREAD_UNCHANGED)
        fg = cv2.imread(str(shot.fg(fi)), cv2.IMREAD_UNCHANGED)
        b_p = cv2.imread(str(base_c / ('%06d.png' % fi)), cv2.IMREAD_UNCHANGED)
        c_p = cv2.imread(str(cand_c / ('%06d.png' % fi)), cv2.IMREAD_UNCHANGED)
        for label, im in (('baseline render', b_e), ('candidate render', c_e), ('plate', fg),
                          ('baseline composite', b_p), ('candidate composite', c_p)):
            if im is None:
                raise SystemExit('Missing %s for frame %d' % (label, fi))

        # The rim is the band over which the proxy silhouette can disagree with the photographed
        # one; the core is far enough inside that only real shading lives there.
        solid = (alpha > 0.99).astype(np.uint8)
        rim = (alpha > 0.05) & (cv2.erode(solid, np.ones((13, 13), np.uint8)) == 0)
        core = cv2.erode(solid, np.ones((41, 41), np.uint8)) > 0

        b_map = relative_corrugation(b_e, pitch)
        c_map = relative_corrugation(c_e, pitch)
        p_map = relative_corrugation(fg, pitch)
        row = {'frame': fi, 'grid_pitch_plate_px': round(float(pitch), 2),
               'rim_px': int(rim.sum()),
               'rim_corrugation': {'baseline': round(rms(b_map, rim), 5),
                                   'candidate': round(rms(c_map, rim), 5),
                                   'plate_reference': round(rms(p_map, rim), 5)},
               'core_control': {'baseline': round(rms(b_map, core), 5),
                                'candidate': round(rms(c_map, core), 5)}}
        row['rim_reduction_pct'] = round(
            100 * (1 - row['rim_corrugation']['candidate']
                   / max(row['rim_corrugation']['baseline'], 1e-9)), 1)

        # Locate the worst rim patch in the baseline, and show both runs at that same place.
        energy = cv2.boxFilter(np.where(rim, b_map, 0).astype(np.float32), -1, (cw // 2, ch // 2))
        energy[cv2.dilate(rim.astype(np.uint8), np.ones((9, 9), np.uint8)) == 0] = 0
        cy, cx = np.unravel_index(int(np.argmax(energy)), energy.shape)
        y0 = int(np.clip(cy - ch // 2, 0, H - ch))
        x0 = int(np.clip(cx - cw // 2, 0, W - cw))
        row['crop_xywh'] = [x0, y0, cw, ch]

        def panel(img, label, to_srgb=True):
            c = img[y0:y0 + ch, x0:x0 + cw]
            c = linear_to_srgb(c[..., :3]) if to_srgb else c[..., :3]
            c = cv2.resize(np.clip(c * 255, 0, 255).astype(np.uint8),
                           (cw * args.zoom, ch * args.zoom), interpolation=cv2.INTER_NEAREST)
            cv2.rectangle(c, (0, 0), (c.shape[1] - 1, 26), (0, 0, 0), -1)
            cv2.putText(c, label, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1,
                        cv2.LINE_AA)
            return c

        def gray(m):
            return np.repeat(np.clip(m * 12.0, 0, 1)[..., None], 3, 2)

        top = np.concatenate([
            panel(fg * alpha[..., None], 'plate (keyed)'),
            panel(b_p.astype(np.float32) / 255.0, 'composite: ' + args.baseline_label, False),
            panel(c_p.astype(np.float32) / 255.0, 'composite: ' + args.candidate_label, False)], 1)
        bottom = np.concatenate([
            panel(gray(p_map), 'plate corrugation x12', False),
            panel(gray(b_map), 'corrugation: ' + args.baseline_label, False),
            panel(gray(c_map), 'corrugation: ' + args.candidate_label, False)], 1)
        cv2.imwrite(str(out / ('%06d_rim.png' % fi)), np.concatenate([top, bottom], 0))
        per_frame.append(row)
        print(json.dumps(row))

    summary = {'shot': shot.name,
               'baseline_render': str(base_r), 'candidate_render': str(cand_r),
               'measures': ('RMS of band-passed irradiance around the proxy grid pitch, relative '
                            'to the local irradiance, inside a rim band around the matte'),
               'reading': ('The fix works if rim corrugation falls toward the core control. The '
                           'plate reference is photographed micro-contrast in the same band, not '
                           'a floor the composite has to reach.'),
               'per_frame': per_frame}
    shot.log('silhouette_corrugation.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({'panels': str(out), 'reading': summary['reading']}, indent=2))


if __name__ == '__main__':
    main()
