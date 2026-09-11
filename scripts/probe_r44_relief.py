"""Round 44: how much meso-scale shading relief each field carries, on skin, per frame.

    dependencies/CorridorKey/.venv/Scripts/python.exe scripts/probe_r44_relief.py --shot <dir> \\
        --frames 30,73,97 --dirs r43_nospec,r44_meso

Round 43 SS3 measured the std of the 12 px high-passed log2 luminance over skin for the plate,
the rendered irradiance and a Lambert of the neural normals under the new light, and found the
proxy at 40-60 % of the Lambert. This reproduces that table from the repo, adds the composite
OUTPUTS (what the artist sees), and is the gate for the meso-shading stage: E_t' must reach the
Lambert column. Every column is read the same way; nothing here shares code with the stage.

Columns:
  plate      the photograph (old light's shading)
  E_source   the measured SH-L2 of the old light on the neural normals
  E_target   the proxy render (ratio_render), luminance
  lambert    max(n . l, 0) on the NEURAL normals with l per pixel from the bulb's world position
  <run>      each composite under --dirs, linearised
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
from sh_project import sh_basis
from light_geometry import position_map, bulb_world, load_normals

LUMA = np.array([.2126, .7152, .0722], np.float32)


def relief(lum, mask, sigma):
    lo = np.log2(np.maximum(lum, 1e-6))
    hp = lo - cv2.GaussianBlur(lo, (0, 0), sigma)
    return float(hp[mask].std())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_shot_argument(ap)
    ap.add_argument('--frames', default='30,73,97')
    ap.add_argument('--dirs', default='')
    ap.add_argument('--sigma', type=float, default=12.0, help='high-pass scale, plate px')
    ap.add_argument('--render-dir', default='ratio_render')
    ap.add_argument('--out', default='r44_relief.json')
    args = ap.parse_args()
    shot = from_args(args)
    C = np.array(json.loads(shot.log('source_light.json').read_text())['sh_coefficients_rgb'], np.float32)
    tour = json.loads(shot.log('stress_proxy.json').read_text())
    pf = {r['out_frame']: r for r in tour['per_frame']}
    runs = [d for d in args.dirs.split(',') if d]
    rows = []
    for f in (int(x) for x in args.frames.split(',')):
        fi = pf[f]['shot_frame'] if f in pf else f
        alpha = cv2.imread(str(shot.alpha(fi)), -1)
        alpha = alpha[..., 0] if alpha.ndim == 3 else alpha
        h, w = alpha.shape
        skin = cv2.imread(str(shot.relight / 'skin_mask' / ('%06d.png' % fi)), -1)
        skin = skin[..., 0] if skin.ndim == 3 else skin
        core = cv2.erode(((skin > 32767) & (alpha > 0.99)).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        fg = cv2.imread(str(shot.fg(fi)), -1)[..., ::-1].astype(np.float32)
        nrm = load_normals(shot, fi, (h, w))
        Y = sh_basis(nrm[..., 0], nrm[..., 1], nrm[..., 2])
        E_s = np.maximum(np.einsum('hwk,kc->hwc', Y, C), 1e-4) @ LUMA
        E_t = cv2.imread(str(shot.composite / args.render_dir / ('%06d.exr' % f)), -1)[..., 2::-1] @ LUMA
        P = position_map(shot, fi, (h, w))
        B = bulb_world(shot, fi, pf[f]['bulb_offset_m'])
        l = B[None, None] - P
        l /= np.maximum(np.linalg.norm(l, axis=-1, keepdims=True), 1e-6)
        lam = np.maximum(np.sum(nrm * l, -1), 0.0)
        row = dict(frame=f, shot_frame=fi, station=pf[f]['station'], bulb_world=[round(float(v), 3) for v in B],
                   skin_px=int(core.sum()),
                   plate=relief(fg @ LUMA, core, args.sigma), E_source=relief(E_s, core, args.sigma),
                   E_target=relief(E_t, core, args.sigma), lambert=relief(lam + 1e-3, core, args.sigma))
        for run in runs:
            a = cv2.imread(str(shot.composite / run / ('%06d.png' % f)))
            if a is None:
                continue
            if a.shape[:2] != (h, w):
                a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
            row[run] = relief(srgb_to_linear(a[..., ::-1].astype(np.float32) / 255.0) @ LUMA, core, args.sigma)
        rows.append(row)
        print(json.dumps(row), flush=True)
    (shot.logs / args.out).write_text(json.dumps(dict(sigma_px=args.sigma, rows=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
