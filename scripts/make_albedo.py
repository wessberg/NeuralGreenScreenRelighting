"""Strip the plate's own lighting off, leaving something that can be re-lit: an albedo estimate.

    dependencies/CorridorKey/.venv/Scripts/python.exe scripts/make_albedo.py --shot <dir>

The keyed foreground already contains the studio lighting it was shot under. Projecting it onto
the proxy and lighting that in Cycles would light it twice: the original key would still be
visible, baked into the texture, fighting whatever the new scene does.

So divide it out:

    albedo = plate / E_source(n)

``E_source`` is the same assumed soft frontal studio key that ``relight_sh.py`` uses -- one model
of the plate's lighting, used consistently, so proxy and gain map cannot disagree about what the
original light was. It is evaluated against the **neural normals**, which is what makes this
better than a flat divide: the brow, nose and cheeks each get the divisor their own orientation
earns, so the recovered albedo is flatter than the plate in exactly the places the original key
sculpted.

**This is an estimate, and the weakest link in every ratio method.** `E_source` is assumed, not
measured -- a chrome ball and an 18% grey ball at the head of the take would turn it into a
measurement. Where the assumption is wrong, some of the original modelling survives into the
albedo and will be visible as lighting that does not respond to the new rig.

What this is NOT: a generative reconstruction. Every output pixel is an input pixel divided by a
renderer-side scalar, so the performer's identity, texture and detail are carried through
unaltered. That property is the whole reason this architecture was chosen.
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

W, H = 768, 432                       # the normals' working resolution
# Same model as relight_sh.py. Kept identical on purpose: two different E_source models would
# make the proxy render and the gain map disagree about the plate's original lighting.
SRC_AMBIENT, SRC_KEY = 0.55, 0.45
L_SRC = np.array([-0.28, -0.92, 0.28], np.float32)
L_SRC /= np.linalg.norm(L_SRC)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_shot_argument(ap)
    ap.add_argument('--frames', type=int, default=0)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--gain', type=float, default=1.0,
                    help='overall scale on the recovered albedo')
    ap.add_argument('--max-albedo', type=float, default=0.92,
                    help='no real surface reflects more than this; clamps division blowups')
    args = ap.parse_args()
    shot = from_args(args)

    out = shot.relight / 'albedo'
    out.mkdir(parents=True, exist_ok=True)
    skin_dir = shot.relight / 'skin_mask'
    skin_dir.mkdir(parents=True, exist_ok=True)
    n = args.frames or shot.frame_count
    stats = []
    for i in range(args.start, args.start + n):
        npz = shot.relight / 'normals' / ('%06d.npz' % i)
        if not npz.exists():
            raise SystemExit('No normals for frame %d. Run scripts/neural_normals.py first.' % i)
        nrm = np.load(npz)['normal'].astype(np.float32)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-6)

        fg = cv2.imread(str(shot.fg(i)), -1)[..., ::-1].astype(np.float32)     # linear RGB
        alpha = cv2.imread(str(shot.alpha(i)), -1)
        if alpha.ndim == 3:
            alpha = alpha[..., 0]
        alpha = np.clip(alpha, 0, 1)
        fh, fw = alpha.shape

        E = SRC_AMBIENT + SRC_KEY * np.maximum(np.sum(nrm * L_SRC, -1), 0)
        E = cv2.resize(E, (fw, fh), interpolation=cv2.INTER_LINEAR)[..., None]
        albedo = np.clip(fg / np.maximum(E, 1e-3) * args.gain, 0, args.max_albedo)

        # Outside the matte the divisor is meaningless; hold the edge colour so the texture does
        # not fringe to black where bilinear sampling reaches just past the silhouette.
        m = (alpha > 0.02).astype(np.uint8)
        if m.any():
            albedo = cv2.inpaint((np.clip(albedo, 0, 1) * 255).astype(np.uint8),
                                 (1 - m) * 255, 3, cv2.INPAINT_TELEA).astype(np.float32) / 255.0

        srgb = np.clip(linear_to_srgb(albedo), 0, 1)
        cv2.imwrite(str(out / ('%06d.png' % i)),
                    (srgb[..., ::-1] * 65535).astype(np.uint16))

        # Skin mask. A HEURISTIC STAND-IN for facebook/sapiens2-seg-1b, whose 29-class taxonomy
        # is the real answer: skin is strongly red-dominant while hair, a dark suit and a foam
        # microphone are not. It exists because applying a subsurface skin shader to fabric and
        # hair is visibly wrong -- cloth would glow. Replace it with the segmentation model.
        lum = albedo @ np.array([.2126, .7152, .0722], np.float32)
        redness = (albedo[..., 0] - albedo[..., 2]) / np.maximum(lum, 1e-3)
        skin = np.clip((redness - 0.25) / 0.45, 0, 1) * np.clip((lum - 0.06) / 0.10, 0, 1)
        skin = cv2.GaussianBlur(skin.astype(np.float32), (0, 0), 3.0) * (alpha > 0.5)
        cv2.imwrite(str(skin_dir / ('%06d.png' % i)), (skin * 65535).astype(np.uint16))

        core = alpha > 0.99
        if core.any():
            stats.append({'frame': i,
                          'albedo_mean': albedo[core].mean(0).round(4).tolist(),
                          'plate_mean': fg[core].mean(0).round(4).tolist(),
                          'clipped_pct': float(100 * (albedo[core] >= args.max_albedo).mean()),
                          'skin_fraction': float((skin[core] > 0.5).mean())})
        if i % 25 == 0 and stats:
            print(json.dumps(stats[-1]), flush=True)

    a = np.array([s['albedo_mean'] for s in stats])
    summary = {'shot': shot.name, 'frames': len(stats), 'output': str(out),
               'method': 'albedo = keyed foreground / E_source(neural normals)',
               'E_source': {'model': 'soft frontal studio key (ASSUMED, not measured)',
                            'direction': L_SRC.tolist(),
                            'ambient': SRC_AMBIENT, 'key': SRC_KEY,
                            'shared_with': 'scripts/relight_sh.py, deliberately identical',
                            'how_to_measure': 'chrome ball + 18% grey ball at the head of the take'},
               'max_albedo': args.max_albedo,
               'encoding': '16-bit sRGB PNG; base colour is a reflectance in [0,1]',
               'identity': 'every pixel is a plate pixel divided by a scalar; no generative model '
                           'touches the output, so identity cannot drift',
               'skin_mask': {'path': str(skin_dir),
                             'method': 'red-dominance heuristic, blurred',
                             'purpose': 'gate subsurface and roughness so a skin shader is not '
                                        'applied to hair, suit and microphone',
                             'replace_with': 'facebook/sapiens2-seg-1b (29 classes, and it '
                                             'splits lips/teeth/tongue for free)',
                             'mean_skin_fraction': float(np.mean(
                                 [s['skin_fraction'] for s in stats])) if stats else 0.0},
               'result': {'mean_albedo': a.mean(0).round(4).tolist(),
                          'mean_clipped_pct': float(np.mean([s['clipped_pct'] for s in stats]))},
               'per_frame': stats}
    shot.log('albedo.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ('frames', 'result')}, indent=2))


if __name__ == '__main__':
    main()
