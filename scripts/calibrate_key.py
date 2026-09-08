"""Derive screen-detection thresholds from the plate instead of hand-tuning them.

The proof-of-concept thresholds (green > 0.68, green-dominance > 0.48) were fitted to a
CLIPPED backing measuring 1.000 / 0.929. A correctly exposed backing can sit far below that
-- the reporter plate measures 0.859 / 0.467 -- so those constants classify none of it as
background, the hint claims the whole frame is foreground, and the key silently fails with a
plausible-looking interior alpha near 1.0.

This samples the frame border, isolates the backing cluster, and places thresholds a relative
margin below it, then sanity-checks the result against pixels that cannot be backing.

    python scripts/calibrate_key.py --shot <dir> [--write]

--write records the thresholds into the shot's shot.json. Evidence is always written to
logs/key_calibration.json. Shots with deliberately hand-set thresholds keep them: this never
overwrites without --write.
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import sys, json, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np, cv2
from shot_paths import add_shot_argument, from_args

BORDER_PX = 24
MARGIN = 0.20          # place thresholds this fraction below the measured backing
SAMPLE_FRAMES = 5


def load(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise SystemExit('Cannot read %s' % path)
    scale = 65535.0 if bgr.dtype == np.uint16 else 255.0
    return np.ascontiguousarray(bgr[..., ::-1].astype(np.float32) / scale)


def border_mask(shape):
    """Ring around the frame. Includes all four sides: a subject cropped at frame bottom
    contributes non-backing pixels there, which the cluster split below removes."""
    h, w = shape
    m = np.zeros((h, w), bool)
    m[:BORDER_PX, :] = True; m[-BORDER_PX:, :] = True
    m[:, :BORDER_PX] = True; m[:, -BORDER_PX:] = True
    return m


def measure(rgb):
    r, g, b = np.moveaxis(rgb, -1, 0)
    return g, g - np.maximum(r, b)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_shot_argument(p)
    p.add_argument('--write', action='store_true', help='record thresholds into shot.json')
    p.add_argument('--margin', type=float, default=MARGIN)
    args = p.parse_args()
    shot = from_args(args)

    total = shot.frame_count
    ids = sorted({0, total // 4, total // 2, 3 * total // 4, total - 1})[:SAMPLE_FRAMES]
    per_frame, g_lo, d_lo = [], [], []

    for i in ids:
        rgb = load(shot.frame(i))
        g, dom = measure(rgb)
        ring = border_mask(g.shape)
        dr = dom[ring]
        # Split the border into backing (green-dominant) and not, at the midpoint of its range.
        mid = (float(dr.min()) + float(dr.max())) / 2
        backing = ring & (dom > mid)
        if backing.sum() < 500:
            raise SystemExit('Frame %d: no green-dominant border cluster found; is this a green-screen plate?' % i)
        gb, db = g[backing], dom[backing]
        # 1st percentile: tolerate lighting falloff and noise across the screen.
        g1, d1 = float(np.percentile(gb, 1)), float(np.percentile(db, 1))
        g_lo.append(g1); d_lo.append(d1)
        per_frame.append({'frame': i, 'border_backing_pixels': int(backing.sum()),
                          'backing_green_p01': round(g1, 4), 'backing_green_median': round(float(np.median(gb)), 4),
                          'backing_dominance_p01': round(d1, 4), 'backing_dominance_median': round(float(np.median(db)), 4)})

    green_min = min(g_lo) * (1 - args.margin)
    dominance_min = min(d_lo) * (1 - args.margin)

    # Sanity checks on the middle sampled frame.
    rgb = load(shot.frame(ids[len(ids) // 2]))
    g, dom = measure(rgb)
    bg = (g > green_min) & (dom > dominance_min)
    ring = border_mask(g.shape)
    luma = rgb @ np.array([.2126, .7152, .0722], np.float32)
    dark = luma < np.percentile(luma, 10)          # hair and dark wardrobe: never backing
    skinish = (dom < -0.05) & (luma > 0.15)        # r/b above g: skin, never backing
    checks = {
        'frame_classified_background_pct': round(100 * float(bg.mean()), 2),
        'border_classified_background_pct': round(100 * float(bg[ring].mean()), 2),
        'darkest_decile_misclassified_pct': round(100 * float(bg[dark].mean()), 4),
        'skin_like_misclassified_pct': round(100 * float(bg[skinish].mean()), 4) if skinish.sum() else None,
    }
    problems = []
    if checks['frame_classified_background_pct'] < 5:
        problems.append('almost nothing classified as background; thresholds still too strict')
    if checks['frame_classified_background_pct'] > 95:
        problems.append('almost everything classified as background; thresholds too loose')
    if checks['border_classified_background_pct'] < 50:
        problems.append('less than half the frame border reads as backing')
    if checks['darkest_decile_misclassified_pct'] > 1:
        problems.append('dark subject pixels classified as backing')
    if checks['skin_like_misclassified_pct'] and checks['skin_like_misclassified_pct'] > 1:
        problems.append('skin-like pixels classified as backing')

    result = {'shot': shot.name, 'sampled_frames': ids, 'margin': args.margin,
              'derived': {'green_min': round(green_min, 4), 'green_dominance_min': round(dominance_min, 4)},
              'previous': shot.key_hint, 'per_frame': per_frame, 'sanity_checks': checks,
              'problems': problems,
              'method': 'Border ring split at the midpoint of its dominance range to isolate the '
                        'backing cluster; thresholds set %d%% below the lowest 1st-percentile '
                        'backing value across sampled frames.' % round(args.margin * 100)}
    shot.logs.mkdir(parents=True, exist_ok=True)
    shot.log('key_calibration.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ('shot', 'derived', 'previous', 'sanity_checks', 'problems')}, indent=2))

    if problems:
        print('\nNOT writing thresholds: sanity checks failed.', file=sys.stderr)
        raise SystemExit(1)
    if args.write:
        cfg_path = shot.dir / 'shot.json'
        cfg = json.loads(cfg_path.read_text(encoding='utf-8')) if cfg_path.exists() else {}
        keying = cfg.setdefault('keying', {})
        keying['hint'] = result['derived']
        keying['hint_source'] = 'Derived by scripts/calibrate_key.py from the plate border; see logs/key_calibration.json'
        cfg_path.write_text(json.dumps(cfg, indent=2) + '\n', encoding='utf-8')
        print('\nWrote thresholds into %s' % cfg_path)


if __name__ == '__main__':
    main()
