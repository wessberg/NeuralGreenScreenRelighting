"""Did the plate's own highlights leave, and did the scene's arrive? Three regions, measured.

    dependencies/CorridorKey/.venv/Scripts/python.exe scripts/measure_specular_transfer.py \\
        --shot <dir> --before composite/<dir> --after composite/<dir>

The artist named three things on frame 21 of the reporter test, on three separate occasions, and
they turned out to be one bug:

* *"I can kind of see the shine in her hair from the original green screen room."*
* *"the lips are somehow lit up more than the skin around them"*
* and, when asked directly, the corneal catchlight -- which was still sitting at 11 o'clock on
  both irises in the round-4 composite, in exactly the position the green-screen room's softbox
  put it.

All three are the same thing: ``out = plate x R`` with a smooth R can recolour a specular
highlight but cannot move, reshape or delete one, so every one of them survived every relight.
This measures whether they leave, in the three places he pointed at:

**iris** -- residual glint contrast inside the corneal disc, against the iris's own median. The
plate runs about 10x. A composite that has removed the photographed catchlight and added a
rendered one should show the CONTRAST surviving (there is still a catchlight) while its POSITION
moves to agree with the scene's key rather than the studio's. Both are reported, because a
number that only says "still bright" cannot tell those apart.

**hair crown** -- the sheen band's contrast against the hair around it, in a band above the brow
line. This is the one he saw first.

**lip gloss** -- the p97/median ratio inside the vermilion, which is how far the wet band on the
lower lip stands above the lip body.

Regions come from the landmark store, so nothing here is hand-placed on this one plate, and
every figure is normalised by its own local level so a round that simply changed the exposure
cannot score as an improvement.
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import sys, json, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np, cv2
from shot_paths import add_shot_argument, from_args
from color_pipeline import srgb_to_linear, linear_to_srgb

LUMA = np.array([.2126, .7152, .0722], np.float32)
LIPS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
        291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
        377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
BROW = [105, 334, 107, 336]


def poly(P, idx, shape, scale=1.0):
    # The landmark store is (N, 3): x, y and a relative z. Polygons want (N, 2), and cv2 rejects
    # the 3-column array with an assertion about checkVector rather than a shape message.
    p = P[idx][:, :2].astype(np.float64)
    c = p.mean(0)
    p = c + (p - c) * scale
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [p.round().astype(np.int32)], 1)
    return m.astype(bool)


def load_linear(path, shape):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.dtype == np.uint8:
        lin = srgb_to_linear(img[..., 2::-1].astype(np.float32) / 255.0)
    elif img.dtype == np.uint16:
        lin = srgb_to_linear(img[..., 2::-1].astype(np.float32) / 65535.0)
    else:
        lin = img[..., 2::-1].astype(np.float32)
    if lin.shape[:2] != shape:
        lin = cv2.resize(lin, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    return lin


def regions(shot, P, alpha, skin, eyes):
    h, w = alpha.shape
    core = alpha > 0.99
    face = poly(P, OVAL, (h, w)) & core
    lip = poly(P, LIPS, (h, w), 0.92) & core
    peri = (poly(P, LIPS, (h, w), 1.45) & ~poly(P, LIPS, (h, w), 1.12)) & core
    yy, xx = np.mgrid[0:h, 0:w]
    brow_y = float(P[BROW][:, 1].mean())
    # Hair: inside the matte, above the brow, not skin, and near the head rather than the torso.
    ext = np.ptp(P[OVAL][:, 0])
    cx = float(P[OVAL][:, 0].mean())
    hair = (core & (skin < 0.2) & (yy < brow_y - 4)
            & (np.abs(xx - cx) < ext * 0.9) & ~face)
    iris = np.zeros((h, w), bool)
    for d in eyes.values():
        cxx, cyy = d['iris_centre_px']
        r = max(d['iris_radius_px_desqueezed'], 1.0)
        iris |= ((xx - cxx) ** 2 + (yy - cyy) ** 2) <= (r * 1.05) ** 2
    return {'face': face, 'lips': lip, 'perioral': peri, 'hair_crown': hair, 'iris': iris}


def measure(img, R, eyes):
    L = img @ LUMA
    out = {}
    if R['iris'].any():
        v = L[R['iris']]
        med = float(np.median(v))
        out['iris_glint_contrast'] = float(np.percentile(v, 99.5) / max(med, 1e-9))
        # Where the brightest thing in each iris sits, in iris radii from its centre. This is
        # what separates "the studio's catchlight is still there" from "the scene's is".
        pos = {}
        h, w = L.shape
        yy, xx = np.mgrid[0:h, 0:w]
        for side, d in eyes.items():
            cx, cy = d['iris_centre_px']
            r = max(d['iris_radius_px_desqueezed'], 1.0)
            disc = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (r * 1.05) ** 2
            if disc.sum() < 20:
                continue
            m = disc & (L >= np.percentile(L[disc], 99.0))
            if m.sum() < 2:
                continue
            wy, wx = np.nonzero(m)
            ww = L[m]
            gx = float((wx * ww).sum() / ww.sum())
            gy = float((wy * ww).sum() / ww.sum())
            pos[side] = [round((gx - cx) / r, 3), round((gy - cy) / r, 3)]
        out['glint_offset_iris_radii'] = pos
    for key, name in (('hair_crown', 'hair_sheen_contrast'), ('lips', 'lip_gloss_contrast')):
        if R[key].sum() > 200:
            v = L[R[key]]
            out[name] = float(np.percentile(v, 97) / max(np.median(v), 1e-9))
    if R['lips'].sum() > 50 and R['perioral'].sum() > 50:
        out['lip_over_perioral_luma'] = float(
            np.median(L[R['lips']]) / max(np.median(L[R['perioral']]), 1e-9))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_shot_argument(ap)
    ap.add_argument('--before', default='', help='composite dir to compare against (optional)')
    ap.add_argument('--after', default='composite/ratio')
    ap.add_argument('--frames', default='21,51,79')
    ap.add_argument('--panel', default='', help='write a labelled crop panel here')
    args = ap.parse_args()
    shot = from_args(args)

    lp = shot.relight / 'face_landmarks.npz'
    if not lp.exists():
        raise SystemExit('No %s. Run scripts/face_track.py first.' % lp)
    lmk = np.load(lp)
    eg = {}
    egp = shot.relight / 'eye_geometry.json'
    if egp.exists():
        eg = {r['frame']: r for r in json.loads(egp.read_text())['per_frame']}

    rows = []
    for f in [int(x) for x in args.frames.split(',')]:
        alpha = cv2.imread(str(shot.alpha(f)), -1)
        alpha = alpha[..., 0] if alpha.ndim == 3 else alpha
        h, w = alpha.shape
        plate = cv2.imread(str(shot.fg(f)), -1)[..., ::-1].astype(np.float32)
        raw = cv2.imread(str(shot.relight / 'skin_mask' / ('%06d.png' % f)), cv2.IMREAD_UNCHANGED)
        sc = {np.dtype(np.uint16): 65535.0, np.dtype(np.uint8): 255.0}.get(raw.dtype, 1.0)
        skin = np.clip(cv2.resize((raw[..., 0] if raw.ndim == 3 else raw).astype(np.float32) / sc,
                                  (w, h)), 0, 1)
        eyes = {k: v for k, v in ((eg.get(f) or {}).get('eyes') or {}).items() if v.get('visible')}
        R = regions(shot, lmk['points'][f], alpha, skin, eyes)
        row = {'frame': f, 'region_px': {k: int(v.sum()) for k, v in R.items()},
               'plate': measure(plate, R, eyes)}
        for tag, d in (('before', args.before), ('after', args.after)):
            if not d:
                continue
            img = load_linear(shot.dir / d / ('%06d.png' % f), (h, w))
            if img is None:
                img = load_linear(shot.dir / d / ('%06d.exr' % f), (h, w))
            if img is not None:
                row[tag] = measure(img, R, eyes)
        rows.append(row)
        print(json.dumps(row), flush=True)

    def agg(tag, key):
        v = [r[tag][key] for r in rows if tag in r and key in r[tag]]
        return float(np.mean(v)) if v else None

    summary = {
        'shot': shot.name, 'frames': args.frames,
        'before': args.before or None, 'after': args.after,
        'what_each_number_means': {
            'iris_glint_contrast': 'p99.5 / median inside the corneal disc. The PLATE is the '
                                   'reference: a composite with no catchlight at all scores low '
                                   'and is wrong; one that kept the studio catchlight scores '
                                   'like the plate and is also wrong. Read it with '
                                   'glint_offset_iris_radii, which says WHERE the light is',
            'glint_offset_iris_radii': 'brightest point in each iris, in iris radii from its '
                                       'centre. This is the number that distinguishes a studio '
                                       'softbox still sitting in the eye from the scene light '
                                       'having replaced it',
            'hair_sheen_contrast': 'p97 / median over the hair above the brow',
            'lip_gloss_contrast': 'p97 / median inside the vermilion',
            'lip_over_perioral_luma': 'median lip luminance over median perioral skin. READ '
                                      'THIS ONLY WHERE THE NEW KEY RESEMBLES THE OLD ONE. Under '
                                      'a key from a different direction the lips and the '
                                      'perioral skin legitimately diverge -- the lips protrude '
                                      'and the philtrum sits in the nose shadow -- so the plate '
                                      'value is NOT a target there. It is a fidelity check on a '
                                      'frontal frame and nothing more. Measured on frame 21 with '
                                      'the panel registered to correlation 1.0000: plate 0.8189, '
                                      'round 3 0.9936 (+21.3%), round 4 0.8764 (+7.0%), round 5 '
                                      '0.8259 (+0.9%). Over frames 51 and 79, where the bulb is '
                                      'below the chin and off to one side, it moves a long way '
                                      'in both rounds and means nothing',
            'sensitivity_warning': 'these region figures move by ~12% between mediapipe VIDEO '
                                   'and IMAGE landmarks on the same frame. Compare runs of this '
                                   'script against each other, never against a number quoted '
                                   'from a differently-masked measurement'},
        'means': {t: {k: agg(t, k) for k in ('iris_glint_contrast', 'hair_sheen_contrast',
                                             'lip_gloss_contrast', 'lip_over_perioral_luma')}
                  for t in ('plate', 'before', 'after') if any(t in r for r in rows)},
        'per_frame': rows}
    shot.log('specular_transfer.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary['means'], indent=2))


if __name__ == '__main__':
    main()
