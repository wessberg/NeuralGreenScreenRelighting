"""Acceptance test for hard constraint 1: does the proxy actually cast self-shadows?

    dependencies/CorridorKey/.venv/Scripts/python.exe scripts/validate_proxy_shadows.py --shot <dir>

Renders the same proxy under an uplight and a toplight and checks that the shadow MOVES. A flat
card, a normal map, or a gain map all produce *some* light-and-dark under a moving source; only
real occlusion moves a shadow to the opposite side of the nose when the light crosses the face.

So the test is not "is there contrast" but "does the dark region change position with the
light", which is the property no per-pixel method has. Three checks:

* the shadow centroid inside the face must sit ABOVE the nose under an uplight and BELOW it
  under a toplight;
* the two centroids must be separated by a usable fraction of the face height;
* both renders must have real dark pixels, so a uniformly lit mannequin cannot pass by having
  no shadow at all in either.

Renders are driven through Blender because ``render_proxy.py`` runs there; this script only
measures the previews it writes. That is deliberate -- the previews are tone-mapped, and the
measurement is spatial (WHERE is the shadow) rather than photometric (how dark), so tone mapping
cannot bias the verdict.
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


def render(shot, frame, test, samples):
    cmd = [str(BLENDER), '-b', '--factory-startup', '--python',
           str(ROOT / 'scripts' / 'render_proxy.py'), '--',
           '--shot', str(shot.dir), '--frame', str(frame), '--test', test,
           '--samples', str(samples)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit('render failed for %s:\n%s' % (test, r.stdout[-2000:] + r.stderr[-2000:]))
    return shot.relight / 'proxy_render' / ('%06d_%s_preview.png' % (frame, test))


def shadow_centroid(png, face_box, alpha_full):
    """Vertical centroid of shadowed pixels inside the face, as a fraction of face height.

    0 is the top of the face box and 1 the bottom, so an uplight should score LOW (shadow thrown
    upward onto brow and eyes) and a toplight HIGH (shadow thrown down onto lip and chin).
    """
    img = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit('cannot read %s' % png)
    if img.shape[2] == 4:
        a = img[..., 3].astype(np.float32) / 255.0
        rgb = img[..., :3].astype(np.float32) / 255.0
    else:
        a = np.ones(img.shape[:2], np.float32)
        rgb = img.astype(np.float32) / 255.0
    lum = rgb @ np.array([.0722, .7152, .2126], np.float32)      # BGR order from imread
    H, W = lum.shape
    x0, y0, x1, y1 = [int(round(v)) for v in face_box]
    sx, sy = W / alpha_full[1], H / alpha_full[0]
    x0, x1 = int(x0 * sx), int(x1 * sx)
    y0, y1 = int(y0 * sy), int(y1 * sy)
    sel = np.zeros_like(lum, bool)
    sel[y0:y1, x0:x1] = True
    sel &= a > 0.5
    if sel.sum() < 100:
        raise SystemExit('face region empty in %s' % png)
    v = lum[sel]
    # "Shadowed" = below the face's own median, so exposure differences between renders cannot
    # move the verdict.
    thr = float(np.median(v))
    ys = np.indices(lum.shape)[0].astype(np.float32)
    dark = sel & (lum < thr)
    frac_dark = float(dark.sum() / sel.sum())
    cy = float(ys[dark].mean())
    return {'centroid_frac_of_face_height': (cy - y0) / max(y1 - y0, 1),
            'dark_fraction': frac_dark,
            'face_median_luma': thr,
            'contrast_p10_p90': float(np.percentile(v, 90) - np.percentile(v, 10))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_shot_argument(ap)
    ap.add_argument('--frame', type=int, default=0)
    ap.add_argument('--samples', type=int, default=96)
    ap.add_argument('--no-render', action='store_true', help='measure existing previews only')
    args = ap.parse_args()
    shot = from_args(args)

    track = json.loads((shot.relight / 'face_track.json').read_text())
    row = next(r for r in track['per_frame'] if r.get('found') and r['frame'] == args.frame)
    src = shot.config.get('source_resolution', [1920, 1080])

    out = {}
    for test in ('uplight', 'toplight'):
        png = (shot.relight / 'proxy_render' / ('%06d_%s_preview.png' % (args.frame, test)))
        if not args.no_render or not png.exists():
            png = render(shot, args.frame, test, args.samples)
        out[test] = shadow_centroid(png, row['bbox_px'], (src[1], src[0]))
        out[test]['preview'] = str(png)

    up = out['uplight']['centroid_frac_of_face_height']
    top = out['toplight']['centroid_frac_of_face_height']
    separation = top - up
    checks = [
        ('uplight throws the shadow into the upper face', up, '<', 0.55),
        ('toplight throws the shadow into the lower face', top, '>', 0.45),
        ('the shadow MOVES when the light moves', separation, '>', 0.10),
        ('uplight actually has shadow', out['uplight']['contrast_p10_p90'], '>', 0.05),
        ('toplight actually has shadow', out['toplight']['contrast_p10_p90'], '>', 0.05),
    ]
    results = [{'test': t, 'value': round(float(v), 4), 'requires': '%s %.2f' % (op, ref),
                'pass': bool(v < ref if op == '<' else v > ref)} for t, v, op, ref in checks]
    verdict = 'PASS' if all(r['pass'] for r in results) else 'FAIL'
    report = {'shot': shot.name, 'frame': args.frame,
              'constraint': 'CLAUDE.md hard constraint 1: self-shadowing',
              'method': 'shadow centroid inside the face box, as a fraction of face height, '
                        'under an uplight and a toplight',
              'measurements': out, 'shadow_travel_frac_of_face_height': round(separation, 4),
              'checks': results, 'verdict': verdict}
    shot.log('proxy_shadows.json').write_text(json.dumps(report, indent=2))

    print('uplight  shadow centroid %.3f of face height (0 = top)' % up)
    print('toplight shadow centroid %.3f' % top)
    print('shadow travel %.3f of face height' % separation)
    for r in results:
        print('  %-52s %+.4f  %s' % (r['test'], r['value'], 'OK' if r['pass'] else 'FAIL'))
    print('VERDICT %s' % verdict)
    return 0 if verdict == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
