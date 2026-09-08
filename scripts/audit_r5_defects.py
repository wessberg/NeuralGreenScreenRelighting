"""Audit rejected round 5 against round 4 and a newly computed rollback; requires --shot.

Integrity means preservation relative to the approved control, not photorealism. Teeth and
sclera masks are anatomical candidates selected using the ORIGINAL plate only, not the result.
Hair temporal error uses current-to-previous optical flow from the plate, never from S.
No metric here proves that a removed signal was physically specular.
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import argparse
import json
from types import SimpleNamespace
import cv2
import numpy as np
from shot_paths import add_shot_argument, from_args
from separate_specular import local_base, separate, LUMA
from review_panel import region_box


def polygon(shape, points):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.rint(points[:, :2]).astype(np.int32)], 1)
    return m.astype(bool)


def read(path):
    a = cv2.imread(str(path), -1)
    if a is None:
        raise RuntimeError('Cannot read %s' % path)
    return a


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_shot_argument(ap)
    ap.add_argument('--result', default='ratio_r7')
    args = ap.parse_args()
    shot = from_args(args)
    manifest = json.loads((shot.composite / args.result / 'composite_manifest.json').read_text())
    if manifest['plate_specular']['mode'] != 'keep' or manifest['catchlight']['gain'] != 0:
        raise RuntimeError('This rollback audit requires keep mode and catchlight gain zero')
    lm = np.load(shot.relight / 'face_landmarks.npz')
    topo = json.loads((shot.relight / 'face_landmarks.json').read_text())['topology']
    tour = json.loads(shot.log('stress_proxy.json').read_text())['per_frame']
    sep_log = json.loads(shot.log('specular_separation.json').read_text())
    G = np.array(sep_log['illuminant']['rgb'], np.float32)
    Gh = G / np.linalg.norm(G)
    settings = SimpleNamespace(cos_max=.9995, floor_fraction=.25)
    out = shot.composite / 'r7_audit'
    out.mkdir(exist_ok=True)
    rows, sensitivity, temporal, crops = [], [], [], {}
    previous = None
    max_difference, changed_channels = 0, 0
    selected = {0, 21, 40, 51, 79, 100, 120, 149}
    sheets = {name: [] for name in ('mouth', 'eyes', 'hair')}
    for rec in tour:
        n, fi = rec['out_frame'], rec['shot_frame']
        P = lm['points'][fi]
        if not lm['found'][fi]:
            raise RuntimeError('Missing landmarks: %d' % fi)
        fg = read(shot.fg(fi))[..., :3][..., ::-1].astype(np.float32)
        S = read(shot.key / 'specular' / ('%06d.exr' % fi))[..., ::-1].astype(np.float32)
        alpha = read(shot.alpha(fi))
        if alpha.ndim == 3:
            alpha = alpha[..., 0]
        L, SL = fg @ LUMA, S @ LUMA
        fraction = SL / np.maximum(L, 1e-5)
        h, w = L.shape
        yy, xx = np.mgrid[:h, :w]
        mouth = polygon((h, w), P[topo['lips_inner']])
        eyes, iris, cap = np.zeros_like(mouth), np.zeros_like(mouth), np.zeros_like(L)
        for side in ('left', 'right'):
            eyes |= polygon((h, w), P[topo['eyelid_' + side]])
            ids = topo['iris_' + side]
            c = P[ids[0], :2]
            radius = np.linalg.norm(P[ids[1:], :2] - c, axis=1).mean()
            dist = np.hypot(xx - c[0], yy - c[1])
            iris |= dist < radius * 1.15
            rr = radius * .8
            cap = np.maximum(cap, np.clip((rr - dist) / max(rr * .45, 1), 0, 1))
        sclera = eyes & ~iris
        # Remove polygon borders and keep the bright part of each anatomical candidate.
        masks = {}
        for name, m in [('teeth', mouth), ('sclera', sclera)]:
            m = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            masks[name] = m & (L >= np.percentile(L[m], 50)) if m.any() else m
        hair = (alpha > .99) & (yy < P[[105, 334, 107, 336], 1].mean() - 12)
        hair &= (xx > P[topo['face_oval'], 0].min()) & (xx < P[topo['face_oval'], 0].max())
        masks['dark_hair'] = hair & (L < np.percentile(L[hair], 50))
        images = {k: read(shot.composite / d / ('%06d.png' % n))
                  for k, d in [('round4', 'ratio'), ('round5', 'ratio_r5'), ('result', args.result)]}
        diff = np.abs(images['result'].astype(np.int16) - images['round4'].astype(np.int16))
        max_difference = max(max_difference, int(diff.max()))
        changed_channels += int(np.count_nonzero(diff))
        row = {'output_frame': n, 'shot_frame': fi, 'regions': {}}
        for name, m in masks.items():
            if not m.any():
                row['regions'][name] = {'pixels': 0}
                continue
            v = {'pixels': int(m.sum()), 'r5_removed_energy_fraction': float(SL[m].sum() / L[m].sum()),
                 'r5_pixels_losing_over_25pct': float((fraction[m] > .25).mean()),
                 'result_removed_energy_fraction': 0.0}
            # Display RGB comparison is independent of the separator's reconstruction identity.
            for key in ('round5', 'result'):
                delta = images[key].astype(np.float32) - images['round4'].astype(np.float32)
                v[key + '_mean_abs_srgb_8bit'] = float(np.abs(delta[m]).mean())
            row['regions'][name] = v
        row['old_cap_weight_outside_eyelids'] = float(cap[~eyes].sum() / max(cap.sum(), 1e-8))
        rows.append(row)
        # Flow on a fixed head crop. Warp previous removal fraction into current plate space.
        if not crops:
            for name in sheets:
                crops[name] = list(region_box(P, name, alpha))
        x, y, cw, ch = crops['hair']
        gray = cv2.cvtColor(np.clip(fg[y:y+ch, x:x+cw] * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        f = fraction[y:y+ch, x:x+cw]
        mask = hair[y:y+ch, x:x+cw]
        if previous is not None:
            oldgray, oldf, oldmask = previous
            flow = cv2.calcOpticalFlowFarneback(gray, oldgray, None, .5, 3, 21, 3, 5, 1.2, 0)
            gy, gx = np.mgrid[:ch, :cw].astype(np.float32)
            mx, my = gx + flow[..., 0], gy + flow[..., 1]
            warped = cv2.remap(oldf, mx, my, cv2.INTER_LINEAR)
            valid = mask & (cv2.remap(oldmask.astype(np.uint8), mx, my, cv2.INTER_NEAREST) > 0)
            valid &= (mx >= 1) & (mx < cw-2) & (my >= 1) & (my < ch-2)
            error = np.abs(f - warped)[valid]
            temporal.append({'output_frame': n, 'pixels': int(valid.sum()),
                             'r5_mean_abs_removed_fraction_change': float(error.mean()),
                             'r5_p95_abs_removed_fraction_change': float(np.percentile(error, 95)),
                             'result_mean_abs_removed_fraction_change': 0.0})
        previous = gray, f, mask
        if n in selected:
            base = local_base(fg, alpha, 15)
            bh = base / np.maximum(np.linalg.norm(base, axis=-1, keepdims=True), 1e-8)
            cosine = (bh * Gh).sum(-1)
            diag = {'output_frame': n, 'regions': {}, 'kernel_widths': {}}
            for name, m in masks.items():
                if m.any():
                    diag['regions'][name] = {'base_illuminant_angle_deg_p5_p50_p95': np.percentile(
                        np.degrees(np.arccos(np.clip(cosine[m], -1, 1))), [5, 50, 95]).tolist(),
                        'guarded_fraction': float((cosine[m] > .9995).mean())}
            for width in (7, 15, 31):
                est, _, _, _ = separate(fg, alpha, G, width, settings)
                ef = (est @ LUMA) / np.maximum(L, 1e-5)
                m = masks['dark_hair']
                _, _, stats, _ = cv2.connectedComponentsWithStats((m & (ef > .25)).astype(np.uint8))
                sizes = stats[1:]
                sizes = sizes[sizes[:, cv2.CC_STAT_AREA] >= 4]
                diag['kernel_widths'][str(width)] = {
                    'dark_hair_mean_removed_fraction': float(ef[m].mean()),
                    'mean_abs_change_from_saved_fraction': float(np.abs(ef[m] - fraction[m]).mean()),
                    'patch_median_width_height': np.median(sizes[:, 2:4], axis=0).tolist() if len(sizes) else None}
            sensitivity.append(diag)
            for name in sheets:
                bx, by, bw, bheight = crops[name]
                tiles = [im[by:by+bheight, bx:bx+bw].copy() for im in images.values()]
                panel = np.concatenate(tiles, 1)
                cv2.putText(panel, '%d: round4 | round5 | rollback' % n, (4, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
                sheets[name].append(panel)
            # Mask overlays are reviewable evidence of what the numbers actually sampled.
            overlay = images['round4'].copy()
            for m, colour in zip(masks.values(), [(0, 255, 255), (255, 0, 255), (255, 255, 0)]):
                overlay[m] = .5 * overlay[m] + .5 * np.array(colour)
            cv2.imwrite(str(out / ('%06d_masks.png' % n)), overlay)
        if n % 40 == 0:
            print('audited', n, flush=True)
    for name, panels in sheets.items():
        cv2.imwrite(str(out / (name + '_sequence.png')), np.concatenate(panels, 0))
    summary = {'frames': len(rows), 'result': args.result,
               'result_vs_round4': {'max_abs_8bit_channel_difference': max_difference,
                                    'changed_channels': changed_channels},
               'regions': {}, 'hair_temporal': {
                   'pairs': len(temporal),
                   'r5_mean_abs_removed_fraction_change': float(np.mean([v['r5_mean_abs_removed_fraction_change'] for v in temporal])),
                   'result_mean_abs_removed_fraction_change': 0.0},
               'mean_old_cap_weight_outside_eyelids': float(np.mean([v['old_cap_weight_outside_eyelids'] for v in rows]))}
    for name in masks:
        vals = [r['regions'][name] for r in rows if r['regions'][name]['pixels']]
        summary['regions'][name] = {k: float(np.mean([v[k] for v in vals])) for k in vals[0] if k != 'pixels'}
    report = {'summary': summary, 'method': __doc__, 'crops_xywh': crops,
              'limitations': ['Teeth and sclera are bright anatomical candidates, not ground-truth segmentation; inspect mask overlays.',
                              'Flow residual includes tracking error and physical appearance changes; it measures instability, not its sole cause.',
                              'Result removal is zero by disabled-stage policy; image comparison verifies the actual output separately.',
                              'Kernel sensitivity does not prove the blocks have exactly the kernel footprint.',
                              'Eyelid measure is mask weight, not rendered reflection energy.'],
              'illuminant_record': sep_log['illuminant'], 'sensitivity': sensitivity,
              'per_frame': rows, 'temporal': temporal}
    shot.log('r7_defect_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2))
    if len(rows) != manifest['frames'] or max_difference != 0:
        raise RuntimeError('Rollback differs from round 4; inspect the report before shipping')


if __name__ == '__main__':
    main()
