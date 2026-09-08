"""Acceptance test for hard constraint 1: does the proxy actually cast SELF-SHADOWS?

    dependencies/CorridorKey/.venv/Scripts/python.exe scripts/validate_proxy_shadows.py --shot <dir>

The first version of this file was wrong, and an external review proved it: it thresholded the
image below the face median and checked that the dark region moved between an uplight and a
toplight. **A shading-only hemisphere -- max(dot(n,l),0) with no shadow computation whatsoever --
passes all five of those checks**, with centroids 0.291 and 0.705 and contrast above 0.86. The
test measured shading, which is exactly the thing this project keeps insisting is NOT shadowing.

So it now measures occlusion directly, by difference. Each light is rendered twice: once
normally, and once with the proxy's ``visible_shadow`` disabled, which leaves shading identical
and removes only the shadow-ray contribution. Their difference IS the cast shadow, and nothing
else. A flat card, a normal map or a gain map all produce a difference of exactly zero here,
which is the property the old test lacked.

The checks are all on that difference rather than on the image:

* the occlusion contribution must be a real fraction of the lit image, not noise;
* the shadow PATTERN must change when the light moves -- low overlap and low correlation between
  the two occlusion maps.

It deliberately does NOT assert which way the shadow moves. That was a second overclaim in the
first version: under an uplight the chin occludes the neck, and that area dwarfs the nose shadow
thrown up onto the brow, so the centroid of total occlusion sits low under BOTH lights. Measured
here: 0.859 for the uplight and 0.735 for the toplight, both in the lower face. The thing that
separates occlusion from shading is not direction, it is that the pattern depends on the light at
all -- a normal map produces the identical map every time.

Renders go through ``render_proxy.py`` because it owns the Blender scene; this script only
measures. The previews are tone-mapped, and the measurement is spatial and differential, so tone
mapping cannot bias the verdict.
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import sys, json, argparse, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np, cv2
from shot_paths import add_shot_argument, from_args

BLENDER = Path(r'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe')


def render(shot, frame, test, samples, no_shadow=False):
    cmd = [str(BLENDER), '-b', '--factory-startup', '--python',
           str(ROOT / 'scripts' / 'render_proxy.py'), '--',
           '--shot', str(shot.dir), '--frame', str(frame), '--test', test,
           '--samples', str(samples)]
    if no_shadow:
        cmd.append('--no-cast-shadow')
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit('render failed (%s%s):\n%s'
                         % (test, ' no-shadow' if no_shadow else '',
                            r.stdout[-2000:] + r.stderr[-2000:]))
    tag = test + ('_noshadow' if no_shadow else '')
    return shot.relight / 'proxy_render' / ('%06d_%s_preview.png' % (frame, tag))


def load(png):
    img = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit('cannot read %s' % png)
    if img.shape[2] == 4:
        a = img[..., 3].astype(np.float32) / 255.0
        rgb = img[..., :3].astype(np.float32) / 255.0
    else:
        a = np.ones(img.shape[:2], np.float32)
        rgb = img.astype(np.float32) / 255.0
    return rgb @ np.array([.0722, .7152, .2126], np.float32), a      # imread gives BGR


def face_mask(shape, face_box, src_shape, alpha):
    H, W = shape
    x0, y0, x1, y1 = [int(round(v)) for v in face_box]
    sx, sy = W / src_shape[1], H / src_shape[0]
    x0, x1, y0, y1 = int(x0 * sx), int(x1 * sx), int(y0 * sy), int(y1 * sy)
    sel = np.zeros((H, W), bool)
    sel[y0:y1, x0:x1] = True
    return (sel & (alpha > 0.5)), (y0, y1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_shot_argument(ap)
    ap.add_argument('--frame', type=int, default=0)
    ap.add_argument('--samples', type=int, default=128)
    ap.add_argument('--no-render', action='store_true')
    args = ap.parse_args()
    shot = from_args(args)

    track = json.loads((shot.relight / 'face_track.json').read_text())
    row = next(r for r in track['per_frame'] if r.get('found') and r['frame'] == args.frame)
    src = shot.config.get('source_resolution', [1920, 1080])

    out, results = {}, []
    for test in ('uplight', 'toplight'):
        pngs = {}
        for ns in (False, True):
            tag = test + ('_noshadow' if ns else '')
            p = shot.relight / 'proxy_render' / ('%06d_%s_preview.png' % (args.frame, tag))
            if not args.no_render or not p.exists():
                p = render(shot, args.frame, test, args.samples, ns)
            pngs[ns] = p
        lit, alpha = load(pngs[False])
        unshadowed, _ = load(pngs[True])

        # The shadow is what disappears when the object stops casting. Positive = light lost.
        occl = np.clip(unshadowed - lit, 0, None)
        m, (fy0, fy1) = face_mask(lit.shape, row['bbox_px'], (src[1], src[0]), alpha)
        if m.sum() < 100:
            raise SystemExit('face region empty in %s' % pngs[False])

        energy = float(occl[m].sum() / max(unshadowed[m].sum(), 1e-9))
        ys = np.indices(lit.shape)[0].astype(np.float32)
        w = occl * m
        cy = float((ys * w).sum() / max(w.sum(), 1e-9))
        out.setdefault('_maps', {})[test] = (occl * m)
        out[test] = {'occlusion_energy_fraction': round(energy, 5),
                     'occluded_centroid_frac_of_face_height': round((cy - fy0) / max(fy1 - fy0, 1), 4),
                     'render': str(pngs[False]), 'control': str(pngs[True])}

    up = out['uplight']['occluded_centroid_frac_of_face_height']
    top = out['toplight']['occluded_centroid_frac_of_face_height']
    sep = abs(top - up)
    # Where the shadow lands is NOT predictable from the light's elevation alone: under an
    # uplight the chin occludes the neck, which is a far larger area than the nose shadow thrown
    # up onto the brow, so the centroid of total occlusion sits LOW for both lights. Asserting a
    # direction here was an overclaim carried over from the old shading-based test. What actually
    # distinguishes occlusion from shading is that the shadow PATTERN changes with the light, so
    # that is what is checked: the two occlusion maps must be substantially decorrelated.
    a = out['_maps']['uplight']
    b = out['_maps']['toplight']
    ov = float((np.minimum(a, b).sum()) / max(np.maximum(a, b).sum(), 1e-9))
    corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if a.std() > 1e-9 and b.std() > 1e-9 else 1.0
    del out['_maps']
    out['pattern_overlap_iou'] = round(ov, 4)
    out['pattern_correlation'] = round(corr, 4)
    checks = [
        ('uplight casts a measurable shadow', out['uplight']['occlusion_energy_fraction'], '>', 0.01),
        ('toplight casts a measurable shadow', out['toplight']['occlusion_energy_fraction'], '>', 0.01),
        ('the shadow PATTERN changes with the light (IoU)', ov, '<', 0.70),
        ('the two shadow maps are not the same picture (corr)', corr, '<', 0.90),
        ('shadow centroid also shifts', sep, '>', 0.02),
    ]
    results = [{'test': t, 'value': round(float(v), 5), 'requires': '%s %.2f' % (op, ref),
                'pass': bool(v < ref if op == '<' else v > ref)} for t, v, op, ref in checks]
    verdict = 'PASS' if all(r['pass'] for r in results) else 'FAIL'
    report = {
        'shot': shot.name, 'frame': args.frame,
        'constraint': 'AGENTS.md hard constraint 1: self-shadowing',
        'method': 'DIFFERENCE between a normal render and one with the proxy casting no shadow. '
                  'Shading is identical in both, so the difference is pure occlusion.',
        'why_the_old_test_was_invalid': 'It thresholded the image below the face median, which a '
                                        'shading-only hemisphere with no shadows at all passes '
                                        '(centroids 0.291 / 0.705, contrast > 0.86). It measured '
                                        'shading, not shadowing.',
        'measurements': out, 'centroid_shift_frac_of_face_height': round(sep, 4),
        'note_on_direction': 'The occlusion centroid sits LOW under both lights because the chin '
                             'shadowing the neck covers far more area than the nose shadow. A '
                             'direction assertion here would be an overclaim; what distinguishes '
                             'occlusion from shading is that the PATTERN changes.',
        'checks': results, 'verdict': verdict}
    shot.log('proxy_shadows.json').write_text(json.dumps(report, indent=2))

    for t in ('uplight', 'toplight'):
        print('%-9s occlusion energy %.4f of the lit image, centroid %.3f of face height'
              % (t, out[t]['occlusion_energy_fraction'],
                 out[t]['occluded_centroid_frac_of_face_height']))
    print('shadow pattern: IoU %.4f, correlation %.4f (low = the shadow really moves)'
          % (out['pattern_overlap_iou'], out['pattern_correlation']))
    print('centroid shift %.3f of face height' % sep)
    for r in results:
        print('  %-46s %+.5f  %s' % (r['test'], r['value'], 'OK' if r['pass'] else 'FAIL'))
    print('VERDICT %s' % verdict)
    return 0 if verdict == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
