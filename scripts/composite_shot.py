"""Composite a keyed shot over its per-frame environment and encode a preview video.

    dependencies/CorridorKey/.venv/Scripts/python.exe scripts/composite_shot.py --shot <dir>

Loads the background PER FRAME. composite.py loads one EXR and reuses it for the whole shot,
which freezes the room behind a moving performer; with animated lights that is unmistakable.

Writes two videos:
  <shot>/final/<name>_relit.mp4       the composite as delivered
  <shot>/final/<name>_compare.mp4     relight off | relight on, side by side

The comparison is the one to judge by. Under static light a flat gain and a correct one look
the same; it is only when the scene's lighting moves that a relighter's failures show.
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import sys, json, argparse, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np, cv2
from shot_paths import add_shot_argument, from_args
from color_pipeline import sdr_preview

FFMPEG = next((ROOT / 'dependencies/ffmpeg').rglob('ffmpeg.exe'))
LUMA = np.array([.2126, .7152, .0722], np.float32)


def load_exr(p):
    v = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if v is None:
        raise SystemExit('Cannot read %s' % p)
    return v


def label(img, text):
    cv2.putText(img, text, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_shot_argument(ap)
    ap.add_argument('--frames', type=int, default=0)
    ap.add_argument('--width', type=int, default=0, help='output width; defaults to the background')
    ap.add_argument('--name', default='stress_test')
    args = ap.parse_args()
    shot = from_args(args)
    env = shot.environment
    bg_dir = env / 'background'
    if not bg_dir.is_dir():
        raise SystemExit('No per-frame background in %s; run scripts/render_background.py' % bg_dir)

    n = args.frames or shot.frame_count
    fps = shot.frame_rate
    out_dir = shot.composite
    out_dir.mkdir(parents=True, exist_ok=True)
    (shot.final).mkdir(parents=True, exist_ok=True)

    probe = load_exr(bg_dir / '000000.exr')
    bh, bw = probe.shape[:2]
    if args.width:
        bw, bh = args.width, round(args.width * bh / probe.shape[1])

    rel_dir = out_dir / 'relit'
    cmp_dir = out_dir / 'compare'
    for d in (rel_dir, cmp_dir):
        d.mkdir(parents=True, exist_ok=True)

    stats = []
    for i in range(n):
        bgf = bg_dir / ('%06d.exr' % i)
        bg = load_exr(bgf)[..., :3][..., ::-1].astype(np.float32)
        if bg.shape[1] != bw:
            bg = cv2.resize(bg, (bw, bh), interpolation=cv2.INTER_AREA)

        fg = load_exr(shot.fg(i))[..., ::-1].astype(np.float32)
        alpha = np.clip(load_exr(shot.alpha(i)).astype(np.float32), 0, 1)
        gain = np.load(shot.control(i))['gain'].astype(np.float32)
        gain = cv2.resize(gain, fg.shape[1::-1], interpolation=cv2.INTER_LINEAR)

        lit = fg * gain
        # Coverage-weighted resize keeps transparent RGB from bleeding into the outline.
        def over(front):
            pre = front * alpha[..., None]
            if front.shape[1::-1] != (bw, bh):
                pre = cv2.resize(pre, (bw, bh), interpolation=cv2.INTER_LANCZOS4)
                a = np.clip(cv2.resize(alpha, (bw, bh), interpolation=cv2.INTER_LANCZOS4), 0, 1)
            else:
                a = alpha
            return np.maximum(pre, 0) + bg * (1 - a[..., None])

        comp_on = over(lit)
        comp_off = over(fg)
        sdr_on = np.clip(sdr_preview(comp_on), 0, 1)
        sdr_off = np.clip(sdr_preview(comp_off), 0, 1)
        cv2.imwrite(str(rel_dir / ('%06d.jpg' % i)), (sdr_on[..., ::-1] * 255).astype(np.uint8),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        left = label((sdr_off[..., ::-1] * 255).astype(np.uint8).copy(), 'relight OFF')
        right = label((sdr_on[..., ::-1] * 255).astype(np.uint8).copy(), 'relight ON')
        cv2.imwrite(str(cmp_dir / ('%06d.jpg' % i)), np.hstack([left, right]),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

        m = alpha > .99
        if m.any():
            stats.append({'frame': i,
                          'actor_luma_off': float(np.median((fg[m] @ LUMA))),
                          'actor_luma_on': float(np.median((lit[m] @ LUMA))),
                          'room_luma': float(np.median(bg @ LUMA)),
                          'actor_rb_on': float(np.median(lit[m][:, 0]) / max(np.median(lit[m][:, 2]), 1e-6)),
                          'room_rb': float(np.median(bg[..., 0]) / max(np.median(bg[..., 2]), 1e-6))})
        if i % 25 == 0:
            print('COMP', i, flush=True)

    def encode(src_dir, dest, w):
        cmd = [str(FFMPEG), '-hide_banner', '-y', '-loglevel', 'error',
               '-framerate', '%d/%d' % (fps.numerator, fps.denominator),
               '-start_number', '0', '-i', str(src_dir / '%06d.jpg'),
               '-c:v', 'libx264', '-preset', 'slow', '-crf', '17',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
               '-r', '%d/%d' % (fps.numerator, fps.denominator), str(dest)]
        subprocess.run(cmd, check=True)
        return dest

    relit = encode(rel_dir, shot.final / ('%s_relit.mp4' % args.name), bw)
    compare = encode(cmp_dir, shot.final / ('%s_compare.mp4' % args.name), bw * 2)

    s = {'shot': shot.name, 'frames': n, 'fps': str(fps), 'resolution': [bw, bh],
         'videos': {'relit': str(relit), 'compare': str(compare)},
         'tracking': {}}
    if stats:
        a_on = np.array([x['actor_luma_on'] for x in stats])
        a_off = np.array([x['actor_luma_off'] for x in stats])
        room = np.array([x['room_luma'] for x in stats])
        rb_a = np.array([x['actor_rb_on'] for x in stats])
        rb_r = np.array([x['room_rb'] for x in stats])
        def corr(x, y):
            return float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 1e-9 and np.std(y) > 1e-9 else float('nan')
        s['tracking'] = {
            'actor_luma_swing_pct_off': float(100 * np.ptp(a_off) / a_off.mean()),
            'actor_luma_swing_pct_on': float(100 * np.ptp(a_on) / a_on.mean()),
            'room_luma_swing_pct': float(100 * np.ptp(room) / room.mean()),
            'corr_actor_vs_room_luma_off': corr(a_off, room),
            'corr_actor_vs_room_luma_on': corr(a_on, room),
            'corr_actor_vs_room_warmth_on': corr(rb_a, rb_r),
            'note': 'A relighter that tracks the scene should correlate with the room. Relight OFF is the control: the plate cannot change, so its correlation is meaningless and its swing should be ~0.'}
        s['per_frame'] = stats
    shot.log('composite_shot.json').write_text(json.dumps(s, indent=2))
    print(json.dumps({k: s[k] for k in ('frames', 'resolution', 'videos', 'tracking')}, indent=2))


if __name__ == '__main__':
    main()
