"""Measure flicker and crawl in a composited sequence, against another sequence as control.

    dependencies/CorridorKey/.venv/Scripts/python.exe scripts/measure_temporal_stability.py \
        --shot <dir> --name r4_ratio --compare ratio

Constraint 14 is temporal stability, and it is the one a per-frame estimator fails by
construction: nothing in the pipeline sees more than one frame at a time, so any error that is
independent per frame shows up as crawl. Looking at stills cannot find it and a single number for
the whole frame hides it, because the subject is moving and real motion swamps the artefact.

Method: deviation from the 3-frame temporal median. A pixel on a smoothly moving surface is close
to the median of its own past and future; a pixel that flickers is not. The plate runs at 59.94
fps, so adjacent frames are 17 ms apart and genuine motion is small at this scale -- which is why
the temporal median is a fair reference here and would not be at 24 fps.

Reported in two regions, because they fail for different reasons:

* the matte CORE, eroded well inside, where deviation means the lighting solution itself is
  unstable frame to frame;
* the RIM, where it means the silhouette is crawling -- the proxy and the matte disagreeing
  differently on each frame.

Both are normalised by the local level, so a bright frame cannot look stabler than a dim one.
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import sys, json, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np, cv2
from shot_paths import add_shot_argument, from_args

LUMA = np.array([.2126, .7152, .0722], np.float32)


def luma_of(p):
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise SystemExit('Could not read %s' % p)
    scale = {np.dtype(np.uint8): 255.0, np.dtype(np.uint16): 65535.0}.get(im.dtype, 1.0)
    return np.ascontiguousarray(im[..., :3].astype(np.float32) / scale) @ LUMA


def masks_for(alpha_dir, fi):
    a = cv2.imread(str(alpha_dir / ('%06d.exr' % fi)), cv2.IMREAD_UNCHANGED)
    if a is None:
        return None, None
    if a.ndim == 3:
        a = a[..., 0]
    a = np.clip(a.astype(np.float32), 0, 1)
    solid = (a > 0.99).astype(np.uint8)
    core = cv2.erode(solid, np.ones((41, 41), np.uint8)) > 0
    rim = (a > 0.05) & (cv2.erode(solid, np.ones((13, 13), np.uint8)) == 0)
    return core, rim


def sequence_stats(alpha_dir, d, stride):
    frames = sorted(p for p in d.glob('*.png') if p.stem.isdigit())
    if len(frames) < 3:
        raise SystemExit('Need at least 3 numbered PNGs in %s' % d)
    rows = []
    prev, cur = luma_of(frames[0]), luma_of(frames[1])
    for i in range(1, len(frames) - 1, stride):
        if stride != 1 and i != 1:
            prev, cur = luma_of(frames[i - 1]), luma_of(frames[i])
        nxt = luma_of(frames[i + 1])
        med = np.median(np.stack([prev, cur, nxt]), axis=0)
        # Relative, so a dark frame is not flattered by its own dimness.
        dev = np.abs(cur - med) / np.maximum(med, 1e-3)
        core, rim = masks_for(alpha_dir, int(frames[i].stem))
        if core is None:
            continue
        rows.append({'frame': int(frames[i].stem),
                     'core': float(np.sqrt(np.mean(np.square(dev[core])))) if core.any() else None,
                     'rim': float(np.sqrt(np.mean(np.square(dev[rim])))) if rim.any() else None})
        prev, cur = cur, nxt
    def agg(key):
        v = [r[key] for r in rows if r[key] is not None]
        return {'mean': round(float(np.mean(v)), 5), 'p95': round(float(np.percentile(v, 95)), 5),
                'max': round(float(np.max(v)), 5),
                'worst_frame': int(rows[int(np.argmax([r[key] for r in rows]))]['frame'])} if v else None
    return {'dir': str(d), 'frames_measured': len(rows),
            'core': agg('core'), 'rim': agg('rim')}, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_shot_argument(ap)
    ap.add_argument('--name', default='ratio', help='dir under composite/ to measure')
    ap.add_argument('--compare', default='', help='another dir under composite/, as control')
    ap.add_argument('--stride', type=int, default=1, help='measure every Nth frame')
    # The mattes define the regions but are a separate pipeline output, so point at a snapshot
    # when the key is being rewritten -- reading a file mid-write is a silent, unrepeatable wrong.
    ap.add_argument('--alpha-dir', default='', help='defaults to the shot key/alpha')
    args = ap.parse_args()
    shot = from_args(args)

    out = {'shot': shot.name,
           'measures': ('RMS relative deviation from the 3-frame temporal median, by region. '
                        'Lower is stabler; the number is a fraction of the local level.')}
    alpha_dir = Path(args.alpha_dir) if args.alpha_dir else shot.key_alpha
    out['alpha_dir'] = str(alpha_dir)
    main_stats, rows = sequence_stats(alpha_dir, shot.composite / args.name, args.stride)
    out['sequence'] = main_stats
    out['per_frame'] = rows
    if args.compare:
        out['control'], _ = sequence_stats(alpha_dir, shot.composite / args.compare, args.stride)
        for region in ('core', 'rim'):
            a, b = out['control'][region], out['sequence'][region]
            if a and b:
                out.setdefault('change_vs_control', {})[region] = \
                    '%+.1f%%' % (100 * (b['mean'] / max(a['mean'], 1e-9) - 1))
    shot.log('temporal_stability.json').write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in out if k != 'per_frame'}, indent=2))


if __name__ == '__main__':
    main()
