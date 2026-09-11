"""Round 44: does the meso-shading or the specular FLICKER? Face-scale, head-stabilised.

    dependencies/CorridorKey/.venv/Scripts/python.exe scripts/audit_r44_temporal.py --shot <dir> \\
        --dirs r43_nospec,r44_meso --out r44_temporal.json

Both new terms ride the neural normals, which crawl about 2 deg per frame (constraint row 14),
and a GGX lobe of roughness 0.3 moves 4 deg for a 2 deg normal change. A per-frame step alone
conflates her motion with flicker (round 32 scored below its own control that way), so every
pair of frames is registered first: a similarity transform fitted to ALL mediapipe landmarks
warps frame f+1 onto f, and the step is |log2 lum(f+1 warped) - log2 lum(f)| over the eroded
skin core. Reported per run as the p95 and max over frames of the per-frame p95 step, and the
same for the PLATE (the control: what her own motion and the codec leave after registration)
and for the tour's light motion (the rendered E_t, which moves between stations by design).
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


def lum_of_run(shot, run, f, shape):
    a = cv2.imread(str(shot.composite / run / ('%06d.png' % f)))
    if a is None:
        return None
    if a.shape[:2] != shape:
        a = cv2.resize(a, shape[::-1], interpolation=cv2.INTER_AREA)
    return srgb_to_linear(a[..., ::-1].astype(np.float32) / 255.0) @ LUMA


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_shot_argument(ap)
    ap.add_argument('--dirs', required=True)
    ap.add_argument('--render-dir', default='ratio_render')
    ap.add_argument('--frames', default='', help='first-last; default all of the first run')
    ap.add_argument('--out', default='r44_temporal.json')
    args = ap.parse_args()
    shot = from_args(args)
    runs = args.dirs.split(',')
    with np.load(shot.relight / 'face_landmarks.npz') as z:
        pts_all, found = z['points'], z['found']
    frames = sorted(int(p.stem) for p in (shot.composite / runs[0]).glob('[0-9]*.png'))
    if args.frames:
        a, b = (int(x) for x in args.frames.split('-'))
        frames = [f for f in frames if a <= f <= b]
    per = {r: [] for r in runs + ['plate', 'E_target']}
    per_hp = {}
    rows = []
    for f0, f1 in zip(frames[:-1], frames[1:]):
        if not (found[f0] and found[f1]):
            continue
        p0, p1 = pts_all[f0][:, :2].astype(np.float32), pts_all[f1][:, :2].astype(np.float32)
        M, _ = cv2.estimateAffinePartial2D(p1, p0, method=cv2.LMEDS)
        if M is None:
            continue
        alpha = cv2.imread(str(shot.alpha(f0)), -1)
        alpha = alpha[..., 0] if alpha.ndim == 3 else alpha
        h, w = alpha.shape
        skin = cv2.imread(str(shot.relight / 'skin_mask' / ('%06d.png' % f0)), -1)
        skin = skin[..., 0] if skin.ndim == 3 else skin
        core = cv2.erode(((skin > 32767) & (alpha > 0.99)).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        fields = {}
        for r in runs:
            a0, a1 = lum_of_run(shot, r, f0, (h, w)), lum_of_run(shot, r, f1, (h, w))
            if a0 is not None and a1 is not None:
                fields[r] = (a0, a1)
        fg0 = cv2.imread(str(shot.fg(f0)), -1)[..., ::-1].astype(np.float32) @ LUMA
        fg1 = cv2.imread(str(shot.fg(f1)), -1)[..., ::-1].astype(np.float32) @ LUMA
        fields['plate'] = (fg0, fg1)
        e0 = cv2.imread(str(shot.composite / args.render_dir / ('%06d.exr' % f0)), -1)[..., 2::-1] @ LUMA
        e1 = cv2.imread(str(shot.composite / args.render_dir / ('%06d.exr' % f1)), -1)[..., 2::-1] @ LUMA
        fields['E_target'] = (e0, e1)
        row = {'frame': f0}
        for name, (a0, a1) in fields.items():
            a1w = cv2.warpAffine(a1, M, (w, h), flags=cv2.INTER_LINEAR)
            l0, l1 = np.log2(np.maximum(a0, 1e-4)), np.log2(np.maximum(a1w, 1e-4))
            d = np.abs(l1 - l0)[core]
            # High-passed at 12 px: the tour's beat pulse and the moving light are global and
            # smooth; anything the meso term or the specular flickers lives in this band.
            hp = np.abs((l1 - cv2.GaussianBlur(l1, (0, 0), 12)) - (l0 - cv2.GaussianBlur(l0, (0, 0), 12)))[core]
            row[name] = {'p50': float(np.median(d)), 'p95': float(np.percentile(d, 95)), 'max': float(d.max()),
                         'hp12_p95': float(np.percentile(hp, 95)), 'hp12_max': float(hp.max())}
            per[name].append(row[name]['p95'])
            per_hp.setdefault(name, []).append(row[name]['hp12_p95'])
        rows.append(row)
    summary = {}
    for name, v in per.items():
        if v:
            v = np.array(v)
            hv = np.array(per_hp[name])
            summary[name] = {'p95_of_frame_p95': float(np.percentile(v, 95)), 'max_of_frame_p95': float(v.max()),
                             'median_of_frame_p95': float(np.median(v)),
                             'worst_frame': int(rows[int(np.argmax(v))]['frame']),
                             'hp12_median_of_frame_p95': float(np.median(hv)), 'hp12_p95_of_frame_p95': float(np.percentile(hv, 95)),
                             'hp12_max_of_frame_p95': float(hv.max()), 'hp12_worst_frame': int(rows[int(np.argmax(hv))]['frame'])}
    out = {'summary': summary, 'per_frame': rows, 'unit': 'stops (|delta log2 luminance|), head-stabilised, eroded skin core'}
    (shot.logs / args.out).write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
