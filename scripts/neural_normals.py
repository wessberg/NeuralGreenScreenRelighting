"""Per-pixel surface normals for the keyed performer, from Sapiens2.

    dependencies/CorridorKey/.venv/Scripts/python.exe scripts/neural_normals.py --shot <dir>

Replaces ``relight_sh.normals_from_depth``. Depth-gradient normals are structurally flat: a
monocular depth model is trained to get *ordering* right, not slope, and differentiating a
smooth depth field over a face yields a surface whose normals barely leave the optical axis.
That is why the stress rig's key could sweep 200 degrees without visibly moving across the
actor's face -- with every normal pointing at the camera, ``dot(n, l)`` is nearly constant in
azimuth, so the relighter could only change level and colour, never direction.

Sapiens2 is trained on humans specifically and predicts normals directly, so the nose, brow,
cheeks and jaw carry real, differing orientations.

Three things about the input matter, and none of them is optional:

* **Crop to the matte bounding box.** The model runs at 1024x768 (3:4 portrait). A raw 16:9
  1080p frame letterboxes into that and leaves the face about 150 px across; the crop puts it
  nearer 600 px. This is the single highest-leverage decision in the stage.
* **Composite onto neutral mid-grey.** A saturated green backing is far out of distribution and
  bleeds into the estimate near the matte edge, which is exactly where hair detail lives.
* **Feed it sRGB.** The keyed foreground is scene-linear. Handing linear values to a model
  normalised with ImageNet statistics silently darkens the midtones by roughly a stop and
  flattens the shading cues the network reads.

The camera-space -> Blender-world convention is NOT assumed here. It is read from
``logs/normals_convention.json``, which ``scripts/validate_normals_convention.py`` writes after
searching all 48 signed axis permutations against rendered ground truth. Refusing to run
without that file is deliberate: an axis or sign error in normals is invisible in a single
frame and ungradeable afterwards.
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import sys, json, argparse, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np, cv2
import torch
from shot_paths import add_shot_argument, from_args
from color_pipeline import linear_to_srgb

MODEL_ID = 'facebook/sapiens2-normal-1b'
IN_H, IN_W = 1024, 768                  # the checkpoint's native inference resolution
OUT_W = OUT_H = None                    # set per shot from the plate aspect
MID_GREY_SRGB = 0.5                     # neutral, mid-range, and out of the green gamut


def matte_bbox(alpha, thresh=0.05, pad=0.08):
    """Bounding box of the matte, padded, then expanded to the model's 3:4 aspect.

    Padded before the aspect fit so hair and shoulder edges are not clipped: the model has no
    way to infer a normal for geometry that was cropped away, and an edge-of-frame subject
    produces a visible seam in the normal field.
    """
    ys, xs = np.where(alpha > thresh)
    if not len(ys):
        return 0, 0, alpha.shape[1], alpha.shape[0]
    H, W = alpha.shape
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = (x1 - x0) * (1 + 2 * pad), (y1 - y0) * (1 + 2 * pad)
    # Grow the short side to reach 3:4; never shrink, or padded content is lost again.
    if w / h > IN_W / IN_H:
        h = w * IN_H / IN_W
    else:
        w = h * IN_W / IN_H
    x0, y0 = cx - w / 2, cy - h / 2
    # Clamp into frame without changing the aspect ratio, shrinking only if genuinely too big.
    w, h = min(w, W), min(h, H)
    if w / h > IN_W / IN_H:
        w = h * IN_W / IN_H
    else:
        h = w * IN_H / IN_W
    x0 = int(round(min(max(x0, 0), W - w)))
    y0 = int(round(min(max(y0, 0), H - h)))
    return x0, y0, int(round(w)), int(round(h))


def prepare(fg_linear, alpha, box):
    """Crop, composite onto mid-grey, convert to sRGB, resize to the model's input."""
    x0, y0, w, h = box
    fg = fg_linear[y0:y0 + h, x0:x0 + w]
    a = alpha[y0:y0 + h, x0:x0 + w][..., None]
    srgb = np.clip(linear_to_srgb(np.clip(fg, 0, None)), 0, 1)
    over = srgb * a + MID_GREY_SRGB * (1 - a)
    return cv2.resize(over, (IN_W, IN_H), interpolation=cv2.INTER_AREA).astype(np.float32)


class Sapiens2Normals:
    """Thin wrapper: fp16 on CUDA, ImageNet normalisation, unit-normalised output."""

    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, model_id=MODEL_ID, device=None, dtype=torch.float16):
        from transformers import Sapiens2ForNormalEstimation
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype if self.device == 'cuda' else torch.float32
        self.model = Sapiens2ForNormalEstimation.from_pretrained(
            model_id, dtype=self.dtype).to(self.device).eval()

    @torch.inference_mode()
    def __call__(self, img_srgb):
        """(H,W,3) sRGB float in [0,1] -> (H,W,3) unit normals in the model's camera frame."""
        x = (img_srgb - self.MEAN) / self.STD
        t = torch.from_numpy(x.transpose(2, 0, 1)[None]).to(self.device, self.dtype)
        n = self.model(pixel_values=t).normals[0].float().cpu().numpy().transpose(1, 2, 0)
        return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)


def load_convention(path):
    """The camera->world mapping, as validated. Never defaulted."""
    if not path.exists():
        raise SystemExit(
            "No %s.\n"
            "Run scripts/validate_normals_convention.py first: the camera-space to Blender-world\n"
            "mapping is inherited and undocumented upstream, and a sign error in it is invisible\n"
            "in a single frame and ungradeable afterwards." % path)
    c = json.loads(path.read_text())
    return np.array(c['matrix'], np.float32), c


def to_world(n_cam, M):
    """Apply the validated 3x3 signed permutation. Rows of M map camera axes to world axes."""
    return n_cam @ M.T


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_shot_argument(ap)
    ap.add_argument('--frames', type=int, default=0)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--model', default=MODEL_ID)
    args = ap.parse_args()
    shot = from_args(args)
    global OUT_W, OUT_H
    OUT_W, OUT_H = shot.working_resolution

    M, conv = load_convention(ROOT / 'logs' / 'normals_convention.json')
    out = shot.relight / 'normals'
    out.mkdir(parents=True, exist_ok=True)

    n_frames = args.frames or shot.frame_count
    print('loading %s ...' % args.model, flush=True)
    t0 = time.time()
    model = Sapiens2Normals(args.model)
    print('loaded in %.1f s on %s (%s)' % (time.time() - t0, model.device, model.dtype), flush=True)

    stats, t_total = [], 0.0
    for i in range(args.start, args.start + n_frames):
        fg = cv2.imread(str(shot.fg(i)), -1)[..., ::-1].astype(np.float32)
        alpha = cv2.imread(str(shot.alpha(i)), -1)
        if alpha.ndim == 3:
            alpha = alpha[..., 0]
        alpha = np.clip(alpha, 0, 1)

        box = matte_bbox(alpha)
        t1 = time.time()
        n_cam = model(prepare(fg, alpha, box))
        t_total += time.time() - t1
        n_world = to_world(n_cam, M)

        # Back into full-frame layout, then down to the relighter's working resolution.
        x0, y0, w, h = box
        full = np.zeros((alpha.shape[0], alpha.shape[1], 3), np.float32)
        full[..., 1] = -1.0                                  # outside the matte: face camera
        full[y0:y0 + h, x0:x0 + w] = cv2.resize(n_world, (w, h), interpolation=cv2.INTER_LINEAR)
        small = cv2.resize(full, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
        small /= np.maximum(np.linalg.norm(small, axis=-1, keepdims=True), 1e-6)
        np.savez_compressed(out / ('%06d.npz' % i), normal=small.astype(np.float16),
                            box=np.array(box, np.int32))

        a_small = cv2.resize(alpha, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
        m = a_small > 0.99
        if m.any():
            mn = small[m]
            stats.append({'frame': i, 'box': list(map(int, box)),
                          'mean_normal': mn.mean(0).tolist(),
                          # Spread of the normal field: the flatness that depth gradients could
                          # not fix. A hemisphere facing the camera has azimuth spread near 90.
                          'azimuth_spread_deg': float(np.degrees(np.std(
                              np.arctan2(mn[:, 0], -mn[:, 1])))),
                          'elevation_spread_deg': float(np.degrees(np.std(
                              np.arcsin(np.clip(mn[:, 2], -1, 1)))))})
        if i % 25 == 0 and stats:
            print(json.dumps(stats[-1]), flush=True)

    az = np.array([s['azimuth_spread_deg'] for s in stats])
    el = np.array([s['elevation_spread_deg'] for s in stats])
    summary = {'shot': shot.name, 'model': args.model, 'frames': len(stats),
               'convention': conv, 'input': {'resolution': [IN_W, IN_H],
                                             'crop': 'matte bbox, 8% pad, expanded to 3:4',
                                             'background': 'neutral mid-grey %.2f sRGB' % MID_GREY_SRGB,
                                             'transfer': 'linear -> sRGB before ImageNet normalisation'},
               'seconds_per_frame': t_total / max(len(stats), 1),
               'result': {'azimuth_spread_deg_mean': float(az.mean()),
                          'azimuth_spread_deg_min': float(az.min()),
                          'elevation_spread_deg_mean': float(el.mean())},
               'per_frame': stats}
    shot.log('neural_normals.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ('frames', 'seconds_per_frame', 'result')}, indent=2))


if __name__ == '__main__':
    main()
