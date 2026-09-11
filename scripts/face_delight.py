"""Round 43: take the capture room's SHINE off her skin with a learned separation, so the ratio
composite relights a diffuse face and the scene's own specular is the only one in the picture.

    P  = dependencies/CorridorKey/.venv/Scripts/python.exe          (prepare, analyse)
    PD = dependencies/diffusion-runtime/.venv/Scripts/python.exe    (inverse)
    PU = dependencies/unreflect-runtime/.venv/Scripts/python.exe    (unreflect)

    P  scripts/face_delight.py --shot <dir> --name r43_delight --stage prepare
    PD scripts/face_delight.py --shot <dir> --name r43_delight --stage inverse   [--frames N]
    PU scripts/face_delight.py --shot <dir> --name r43_delight --stage unreflect
    P  scripts/face_delight.py --shot <dir> --name r43_delight --stage analyse

WHY. The delivery composite is ``out = plate x R``. The plate is ``D + S``; a smooth lighting
ratio scales both, so the green-screen room's specular on her forehead, nose and lips dims in
proportion with the diffuse and never moves. The artist saw it: *"her skin is shining equally
even as we reduce the light or move it away"*. Round 5 tried to split D from S by colour on a
single frame and destroyed teeth, sclera and hair (skin reddens as it darkens and mimics the
specular signature). This uses two released models instead, on the FACE CROP only:

* **UnReflectAnything** (single image): its own highlight head + diffuse decoder. Gives a
  DIFFUSE RADIANCE under the same light, directly comparable with the plate.
* **DiffusionRenderer inverse** (SVD video model): basecolor + roughness + normal. Basecolor is
  an ALBEDO with an arbitrary scale, so `analyse` fits one per-channel gain per frame between
  ``basecolor x E_source(n)`` and the plate over non-highlight skin (a robust median), and the
  fitted diffuse is what the plate is compared against.

WHAT COMES OUT, and what does not. Neither model's picture goes into the output. The composite
(`composite_ratio.py --face-delight <name>`) subtracts ``S = max(plate - D, 0)`` only inside
the skin mask and the model's highlight footprint; every pixel outside stays photographed.
That is the round-30 rule (original pixels outside the selected repair region) and it is what
keeps teeth, sclera and hair out of the model's hands by construction.

Both model runtimes are research/evaluation licensed (DiffusionRenderer: NVIDIA licence 3.3,
non-commercial; UnReflectAnything code MIT, DINOv3 weights under Meta's DINO licence). This is
an evaluation round, like round 31.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2
import numpy as np
from shot_paths import add_shot_argument, from_args
from color_pipeline import linear_to_srgb, srgb_to_linear

PASSES = ('basecolor', 'roughness', 'normal')
LUMA = np.array([.2126, .7152, .0722], np.float32)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, default=str) + '\n', encoding='utf-8')


def read_rgb8(path):
    a = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if a is None:
        raise FileNotFoundError(path)
    return a[..., ::-1]


def save_rgb8(path, rgb):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.ascontiguousarray(np.asarray(rgb)[..., ::-1]))


def read_mask(path):
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(path)
    if raw.ndim == 3:
        raw = raw[..., 0]
    scale = {np.dtype(np.uint16): 65535.0, np.dtype(np.uint8): 255.0}.get(raw.dtype, 1.0)
    return raw.astype(np.float32) / scale


# ----------------------------------------------------------------------------------------
# prepare: one square face crop for the whole range, display sRGB over mid-grey
# ----------------------------------------------------------------------------------------

def plate_size(shot):
    """(width, height) of the keyed plate on disk. NOT shot.resolution, which is the DELIVERY
    size (3840x2160 here against a 1920x1080 plate) -- using it put the S canvas at twice the
    plate and cropped the normals from the wrong quadrant."""
    a = cv2.imread(str(shot.fg(0)), -1)
    if a is None:
        raise FileNotFoundError(shot.fg(0))
    return a.shape[1], a.shape[0]


def face_crop(shot, indices, pad):
    """One square crop, in native plate pixels, holding the face track's bbox union, padded."""
    track = json.loads((shot.relight / 'face_track.json').read_text())
    boxes = np.array([track['per_frame'][j]['bbox_px'] for j in indices
                      if track['per_frame'][j]['found']], np.float64)
    if not len(boxes):
        raise SystemExit('No face track in the requested range')
    lo, hi = boxes[:, :2].min(0), boxes[:, 2:].max(0)
    mid = (lo + hi) / 2
    half = float(max(hi - lo)) * (0.5 + pad)
    W, H = plate_size(shot)
    x0, y0 = int(np.floor(mid[0] - half)), int(np.floor(mid[1] - half))
    x1, y1 = int(np.ceil(mid[0] + half)), int(np.ceil(mid[1] + half))
    # Keep the crop square and inside the plate: slide it rather than clip it.
    side = min(x1 - x0, y1 - y0, W, H)
    x0 = int(np.clip(x0, 0, W - side)); y0 = int(np.clip(y0, 0, H - side))
    return [x0, y0, x0 + side, y0 + side]


def stage_prepare(shot, root, args):
    n = shot.frame_count
    indices = list(range(args.start, n if args.frames <= 0 else min(n, args.start + args.frames)))
    crop = face_crop(shot, indices, args.pad)
    x0, y0, x1, y1 = crop
    rows = []
    for i in indices:
        fg = cv2.imread(str(shot.fg(i)), -1)
        al = cv2.imread(str(shot.alpha(i)), -1)
        if fg is None or al is None:
            raise FileNotFoundError(shot.fg(i))
        fg = fg[y0:y1, x0:x1, ::-1].astype(np.float32)
        al = al[y0:y1, x0:x1]
        if al.ndim == 3:
            al = al[..., 0]
        al = np.clip(al, 0, 1)[..., None].astype(np.float32)
        # Neutral mid-grey behind her, as the normals stage does: a black backing is out of
        # distribution for both encoders and bleeds into the estimate along the hairline.
        grey = np.float32(srgb_to_linear(np.float32(0.5)))
        lin = fg + (1.0 - al) * grey
        disp = np.rint(np.clip(linear_to_srgb(lin), 0, 1) * 255).astype(np.uint8)
        save_rgb8(root / 'input' / ('%06d.png' % i), disp)
        skin_p = shot.relight / 'skin_mask' / ('%06d.png' % i)
        skin = read_mask(skin_p)[y0:y1, x0:x1] if skin_p.exists() else np.zeros(al.shape[:2], np.float32)
        (root / 'skin').mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(root / 'skin' / ('%06d.png' % i)), np.rint(skin * 255).astype(np.uint8))
        rows.append(dict(frame=i, fg_sha256=sha(shot.fg(i)), skin_pixels=int((skin > 0.5).sum())))
    rec = dict(crop_xyxy=crop, native_side=x1 - x0, frames=[r['frame'] for r in rows], rows=rows,
               pad=args.pad, background='mid-grey (display 0.5) under the matte',
               input_transfer='display-sRGB', source_transfer='scene-linear Rec.709',
               code_sha256=sha(__file__))
    write_json(root / 'input.json', rec)
    write_json(shot.logs / ('%s_input.json' % root.name), rec)
    print('PREPARED', len(rows), 'frames, crop', crop, flush=True)


# ----------------------------------------------------------------------------------------
# inverse: DiffusionRenderer basecolor / roughness / normal, chunked with cross-faded overlap
# ----------------------------------------------------------------------------------------

def stage_inverse(shot, root, args):
    import torch
    from diffusion_eye_trial import load_pipeline
    meta = json.loads((root / 'input.json').read_text())
    frames = meta['frames'] if args.frames <= 0 else meta['frames'][:args.frames]
    size = args.size
    pipe = load_pipeline('inverse')
    # Windows of `chunk` frames stepping by chunk - overlap; overlapping frames are cross-faded
    # linearly so a chunk boundary never lands as a cut.
    starts = list(range(0, max(1, len(frames) - args.overlap), args.chunk - args.overlap))
    windows = []
    for s in starts:
        e = min(len(frames), s + args.chunk)
        windows.append((s, e))
        if e == len(frames):
            break
    passes = [p for p in PASSES if p in args.passes.split(',')]
    inputs = np.stack([read_rgb8(root / 'input' / ('%06d.png' % f)) for f in frames])
    if inputs.shape[1] != size:
        inputs = np.stack([cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA if x.shape[0] > size
                                      else cv2.INTER_CUBIC) for x in inputs])
    acc = {p: np.zeros((len(frames), size, size, 3), np.float32) for p in passes}
    wsum = np.zeros(len(frames), np.float32)
    log = []
    for wi, (s, e) in enumerate(windows):
        x = inputs[s:e][None].astype(np.float32) / 255.0
        count = e - s
        ramp = np.ones(count, np.float32)
        if wi > 0:
            ramp[:args.overlap] = np.linspace(0, 1, args.overlap + 2)[1:-1]
        if wi < len(windows) - 1:
            ramp[count - args.overlap:] = np.linspace(1, 0, args.overlap + 2)[1:-1]
        for p in passes:
            t0 = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                out = pipe(dict(rgb=x, input_context=p), {'rgb': 'vae'}, height=size, width=size,
                           num_frames=count, num_inference_steps=args.steps,
                           min_guidance_scale=1.0, max_guidance_scale=1.0, fps=7,
                           motion_bucket_id=127, noise_aug_strength=0,
                           generator=torch.Generator(device='cuda').manual_seed(args.seed),
                           decode_chunk_size=4).frames[0]
            arr = np.stack([np.asarray(f) for f in out]).astype(np.float32)
            if arr.max() <= 1.0:
                arr *= 255.0
            acc[p][s:e] += arr * ramp[:, None, None, None]
            log.append(dict(window=[s, e], pass_=p, seconds=time.perf_counter() - t0,
                            peak_cuda_bytes=torch.cuda.max_memory_allocated()))
            print('INVERSE', p, 'frames', frames[s], '..', frames[e - 1], '%.1fs' % log[-1]['seconds'], flush=True)
        wsum[s:e] += ramp
    for p in passes:
        for k, f in enumerate(frames):
            save_rgb8(root / 'inverse' / p / ('%06d.png' % f),
                      np.rint(np.clip(acc[p][k] / wsum[k], 0, 255)).astype(np.uint8))
    rec = dict(size=size, steps=args.steps, seed=args.seed, chunk=args.chunk, overlap=args.overlap,
               windows=windows, passes=passes, frames=frames, timings=log,
               model_revision=json.loads((ROOT / 'dependencies/diffusion-weights/inverse_pin.json').read_text())['revision'],
               licence='NVIDIA licence, section 3.3: non-commercial (research or evaluation) only',
               code_sha256=sha(__file__))
    write_json(root / 'inverse' / 'complete.json', rec)
    write_json(shot.logs / ('%s_inverse.json' % root.name), rec)
    print('INVERSE_DONE', len(frames), 'frames', sum(l['seconds'] for l in log), 's', flush=True)


# ----------------------------------------------------------------------------------------
# unreflect: the model's own highlight head and diffuse decoder, per frame
# ----------------------------------------------------------------------------------------

def stage_unreflect(shot, root, args):
    import torch
    import torch.nn.functional as F
    from unreflect_diffuse import load_model
    meta = json.loads((root / 'input.json').read_text())
    frames = meta['frames'] if args.frames <= 0 else meta['frames'][:args.frames]
    model, prov = load_model(args.device)
    size = args.unreflect_size
    t0 = time.perf_counter()
    for k, f in enumerate(frames):
        rgb = read_rgb8(root / 'input' / ('%06d.png' % f))
        h, w = rgb.shape[:2]
        x = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].float().to(args.device) / 255.0
        x = F.interpolate(x, size=(size, size), mode='bicubic', align_corners=False).clamp(0, 1)
        with torch.inference_mode():
            out = model({'rgb': x, 'inpaint_mask_dilation': args.dilation,
                         'inpaint_mask_threshold': args.threshold})
        hl = F.interpolate(out['highlight'].float(), size=(h, w), mode='bilinear', align_corners=False)[0, 0]
        d = F.interpolate(out['diffuse'].float().clamp(0, 1), size=(h, w), mode='bicubic', align_corners=False)[0]
        d = d.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        save_rgb8(root / 'unreflect' / 'diffuse' / ('%06d.png' % f), np.rint(d * 255).astype(np.uint8))
        (root / 'unreflect' / 'highlight').mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(root / 'unreflect' / 'highlight' / ('%06d.png' % f)),
                    np.rint(hl.clamp(0, 1).cpu().numpy() * 65535).astype(np.uint16))
        if k % 25 == 0:
            print('UNREFLECT', k, f, flush=True)
    prov.update(frames=frames, size=size, dilation=args.dilation, threshold=args.threshold,
                mask_source="the model's OWN highlight head (no override)",
                diffuse_decoder_executed=True, seconds=time.perf_counter() - t0,
                licence='code MIT; DINOv3 encoder weights under the Meta DINO licence',
                code_sha256=sha(__file__))
    write_json(root / 'unreflect' / 'complete.json', prov)
    write_json(shot.logs / ('%s_unreflect.json' % root.name), prov)
    print('UNREFLECT_DONE', len(frames), '%.0fs' % prov['seconds'], flush=True)


# ----------------------------------------------------------------------------------------
# analyse: put both models against the plate, in linear light, on skin only
# ----------------------------------------------------------------------------------------

def load_source_sh(shot):
    p = shot.log('source_light.json')
    return np.array(json.loads(p.read_text())['sh_coefficients_rgb'], np.float32) if p.exists() else None


def e_source(shot, i, crop, C):
    """E_source(n) over the crop, from the neural normals and the measured SH, luminance only
    (the composite divides by luminance too: composite_ratio.py, E_s @ LUMA)."""
    from sh_project import sh_basis
    z = np.load(shot.relight / 'normals' / ('%06d.npz' % i))
    nrm = z['normal'].astype(np.float32)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-6)
    W, H = plate_size(shot)
    nrm = cv2.resize(nrm, (W, H), interpolation=cv2.INTER_LINEAR)
    x0, y0, x1, y1 = crop
    nrm = nrm[y0:y1, x0:x1]
    nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-6)
    Y = sh_basis(nrm[..., 0], nrm[..., 1], nrm[..., 2])
    E = np.maximum(np.einsum('hwk,kc->hwc', Y, C), 1e-4)
    return E @ LUMA


def robust_gain(plate, model, weight):
    """Per-channel median of plate/model over the weighted pixels: the scale that puts an
    albedo-like map into the plate's units, insensitive to the highlight minority."""
    g = np.ones(3, np.float32)
    m = weight > 0.5
    if m.sum() < 100:
        return g
    for c in range(3):
        r = plate[..., c][m] / np.maximum(model[..., c][m], 1e-4)
        g[c] = float(np.median(r))
    return g


def region_boxes(shot, i, crop):
    """Forehead, nose, upper lip and cheek boxes from mediapipe landmarks, in crop pixels.
    Landmark ids: 10 forehead top / 151 brow centre, 4 nose tip / 6 bridge, 0 upper lip centre,
    50 / 280 cheeks."""
    with np.load(shot.relight / 'face_landmarks.npz') as z:
        pts = z['points'][i][:, :2].copy()
        found = bool(z['found'][i])
    if not found:
        return {}
    pts -= np.array(crop[:2], np.float32)
    ipd = float(np.linalg.norm(pts[33] - pts[263]))  # outer eye corners
    def box(cx, cy, rw, rh):
        return [int(cx - rw * ipd), int(cy - rh * ipd), int(cx + rw * ipd), int(cy + rh * ipd)]
    fore = (pts[10] + pts[151]) / 2
    return dict(forehead=box(*fore, 0.18, 0.10), nose=box(*pts[4], 0.07, 0.07),
                nose_bridge=box(*pts[6], 0.06, 0.08), upper_lip=box(*pts[0], 0.12, 0.05),
                cheek_l=box(*pts[50], 0.09, 0.07), cheek_r=box(*pts[280], 0.09, 0.07))


def stage_analyse(shot, root, args):
    meta = json.loads((root / 'input.json').read_text())
    crop = meta['crop_xyxy']; x0, y0, x1, y1 = crop
    frames = meta['frames']
    C = load_source_sh(shot)
    have_inv = (root / 'inverse' / 'basecolor').is_dir()
    have_unr = (root / 'unreflect' / 'diffuse').is_dir()
    inv_frames = sorted(int(p.stem) for p in (root / 'inverse' / 'basecolor').glob('*.png')) if have_inv else []
    unr_frames = sorted(int(p.stem) for p in (root / 'unreflect' / 'diffuse').glob('*.png')) if have_unr else []
    frames = [f for f in frames if (not have_inv or f in inv_frames) and (not have_unr or f in unr_frames)]
    if not frames:
        raise SystemExit('nothing to analyse: run inverse and/or unreflect first')
    per = []
    stacks = dict(plate=[], unreflect=[], inverse=[], hl=[], skin=[])
    sheet_rows = []
    for f in frames:
        fg = cv2.imread(str(shot.fg(f)), -1)[y0:y1, x0:x1, ::-1].astype(np.float32)
        skin = read_mask(root / 'skin' / ('%06d.png' % f))
        # Skin mask with the eyes, brows and mouth interior cut out by erosion: highlights ARE
        # skin, but a mask edge on a lash or a lip line is not something to fit a gain on.
        core = cv2.erode((skin > 0.5).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        row = dict(frame=f, skin_px=int(core.sum()))
        lum_p = fg @ LUMA
        boxes = region_boxes(shot, f, crop)
        cand = {}
        if have_unr:
            d = srgb_to_linear(read_rgb8(root / 'unreflect' / 'diffuse' / ('%06d.png' % f)).astype(np.float32) / 255.0)
            if d.shape[:2] != fg.shape[:2]:
                d = cv2.resize(d, fg.shape[1::-1], interpolation=cv2.INTER_CUBIC)
            hl = read_mask(root / 'unreflect' / 'highlight' / ('%06d.png' % f))
            if hl.shape != fg.shape[:2]:
                hl = cv2.resize(hl, fg.shape[1::-1], interpolation=cv2.INTER_LINEAR)
            cand['unreflect'] = d
            stacks['hl'].append(hl)
            row['unreflect_highlight_skin_frac'] = float((hl[core] > 0.3).mean()) if core.any() else 0.0
        if have_inv:
            bc = srgb_to_linear(read_rgb8(root / 'inverse' / 'basecolor' / ('%06d.png' % f)).astype(np.float32) / 255.0)
            bc = cv2.resize(bc, fg.shape[1::-1], interpolation=cv2.INTER_CUBIC)
            if C is not None:
                E = e_source(shot, f, crop, C)[..., None]
            else:
                E = np.ones(fg.shape[:2] + (1,), np.float32)
            model = bc * E
            # Fit the albedo's scale on the DARKER half of the skin (below the skin median
            # luminance): the highlight lives in the upper tail and must not pull the gain.
            w = core & (lum_p <= np.median(lum_p[core])) if core.any() else core
            g = robust_gain(fg, model, w.astype(np.float32))
            cand['inverse'] = model * g
            row['inverse_gain_rgb'] = g.tolist()
            rgh = read_rgb8(root / 'inverse' / 'roughness' / ('%06d.png' % f)).astype(np.float32) / 255.0
            rgh = cv2.resize(rgh[..., 0], fg.shape[1::-1], interpolation=cv2.INTER_LINEAR)
            row['inverse_roughness_skin_p10_p50_p90'] = [float(np.percentile(rgh[core], q)) for q in (10, 50, 90)] if core.any() else None
        stacks['plate'].append(lum_p); stacks['skin'].append(core)
        for name, d in cand.items():
            S = np.maximum(fg - d, 0.0)
            lum_d = d @ LUMA; lum_s = S @ LUMA
            row['%s_skin_S_over_plate' % name] = float(lum_s[core].sum() / max(lum_p[core].sum(), 1e-6)) if core.any() else None
            row['%s_skin_S_p50_p95_max' % name] = [float(np.percentile(lum_s[core], q)) for q in (50, 95, 100)] if core.any() else None
            # Removed energy that is NOT above the local diffuse level would be texture, not
            # shine: report how much of S sits where the plate is below its own 15 px mean.
            local = cv2.GaussianBlur(lum_p, (0, 0), 15)
            below = core & (lum_p < local)
            row['%s_S_energy_frac_below_local_mean' % name] = float(lum_s[below].sum() / max(lum_s[core].sum(), 1e-6)) if core.any() else None
            for rname, (bx0, by0, bx1, by1) in boxes.items():
                sub = (slice(max(by0, 0), max(by1, 0)), slice(max(bx0, 0), max(bx1, 0)))
                m = core[sub]
                if m.sum() < 20:
                    continue
                row['%s_%s' % (name, rname)] = dict(plate=float(np.median(lum_p[sub][m])),
                                                   diffuse=float(np.median(lum_d[sub][m])),
                                                   S_p95=float(np.percentile(lum_s[sub][m], 95)))
            stacks[name].append(lum_d)
        per.append(row)
        if f in frames[:: max(1, len(frames) // args.sheet_frames)][:args.sheet_frames]:
            tiles = [np.rint(np.clip(linear_to_srgb(fg), 0, 1) * 255).astype(np.uint8)]
            for name in ('unreflect', 'inverse'):
                if name in cand:
                    tiles.append(np.rint(np.clip(linear_to_srgb(cand[name]), 0, 1) * 255).astype(np.uint8))
                    Sv = np.clip(linear_to_srgb(np.maximum(fg - cand[name], 0) * 3.0), 0, 1)
                    tiles.append(np.rint(Sv * 255).astype(np.uint8))
            if have_unr:
                tiles.append(np.repeat(np.rint(stacks['hl'][-1] * 255).astype(np.uint8)[..., None], 3, -1))
            strip = np.concatenate(tiles, 1)
            cv2.putText(strip, 'f%d  plate | unreflect D | S x3 | inverse D | S x3 | hl' % f, (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            sheet_rows.append(strip)
    # Temporal stability on the static crop, skin only, per pixel over frames: p5-p95 spread of
    # the diffuse estimate versus the plate's own. The head barely moves on this plate; report
    # BOTH so a model that flickers shows up as spread above the plate's.
    summary = dict(frames=len(frames), crop_xyxy=crop)
    skin_all = np.mean(np.stack(stacks['skin']), 0) > 0.9
    for name in ('plate', 'unreflect', 'inverse'):
        if not stacks[name]:
            continue
        st = np.stack(stacks[name])
        spread = np.percentile(st, 95, axis=0) - np.percentile(st, 5, axis=0)
        step = np.abs(np.diff(st, axis=0))
        summary['%s_temporal' % name] = dict(
            spread_p50=float(np.median(spread[skin_all])), spread_p95=float(np.percentile(spread[skin_all], 95)),
            step_p95=float(np.percentile(step[:, skin_all], 95)), step_max=float(step[:, skin_all].max()))
    for key in ('unreflect_skin_S_over_plate', 'inverse_skin_S_over_plate',
                'unreflect_S_energy_frac_below_local_mean', 'inverse_S_energy_frac_below_local_mean'):
        vals = [r[key] for r in per if r.get(key) is not None]
        if vals:
            summary[key] = dict(p50=float(np.median(vals)), p95=float(np.percentile(vals, 95)), max=float(max(vals)))
    if have_inv:
        gains = np.array([r['inverse_gain_rgb'] for r in per])
        summary['inverse_gain_rgb_p50'] = np.median(gains, 0).tolist()
        summary['inverse_gain_step_p95'] = float(np.percentile(np.abs(np.diff(gains, axis=0)), 95)) if len(gains) > 1 else 0.0
        summary['inverse_roughness_skin_p50_median'] = float(np.median([r['inverse_roughness_skin_p10_p50_p90'][1] for r in per if r.get('inverse_roughness_skin_p10_p50_p90')]))
    for rname in ('forehead', 'nose', 'nose_bridge', 'upper_lip', 'cheek_l', 'cheek_r'):
        for name in ('unreflect', 'inverse'):
            vals = [r['%s_%s' % (name, rname)] for r in per if '%s_%s' % (name, rname) in r]
            if vals:
                summary['%s_%s' % (name, rname)] = dict(
                    plate_p50=float(np.median([v['plate'] for v in vals])),
                    diffuse_p50=float(np.median([v['diffuse'] for v in vals])),
                    S_p95_p50=float(np.median([v['S_p95'] for v in vals])))
    write_json(root / 'analysis.json', dict(summary=summary, per_frame=per))
    write_json(shot.logs / ('%s_analysis.json' % root.name), dict(summary=summary, per_frame=per))
    if sheet_rows:
        sheet = np.concatenate(sheet_rows, 0)
        cv2.imwrite(str(root / 'sheet.jpg'), sheet[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(json.dumps(summary, indent=2))



# ----------------------------------------------------------------------------------------
# separate: the shine, as a per-pixel illuminant-coloured excess over a MODEL diffuse
# ----------------------------------------------------------------------------------------
#
# plate = bc * E(n) + S.  bc is DiffusionRenderer's basecolor (an albedo, scale unknown), E(n)
# is SH-L2 shading refitted PER FRAME from the plate itself against the neural normals with a
# one-sided robust weight (bright residuals are shine and must not pull the fit), and the
# residual r = plate - D is split as r = a*b_hat + s*g_hat, a,s >= 0 -- the round-5 dichromatic
# decomposition, but with the diffuse reference b_hat taken from the MODEL's diffuse at that
# pixel rather than from a local morphological opening (which is what mixed materials across
# the mouth and the eye and destroyed teeth and sclera). g_hat is the plate's illuminant,
# measured from her own corneal glints (logs/specular_separation.json).
#
# What is NOT removed: the pore-scale texture the 512 px model cannot reproduce (s is low-passed
# at a fraction of the interocular distance), the residual bias of the fit (the skin median of
# s is subtracted as a floor), anything outside the skin mask, and the eyes and the mouth
# interior (landmark polygons, dilated), which the skin mask includes and the eye passes own.

def polygon_mask(points, ids, shape, grow_px):
    m = np.zeros(shape, np.uint8)
    poly = points[ids, :2]
    if not np.isfinite(poly).all():
        return m.astype(bool)
    cv2.fillPoly(m, [np.rint(poly).astype(np.int32)], 1)
    if grow_px > 0:
        m = cv2.dilate(m, np.ones((2 * grow_px + 1, 2 * grow_px + 1), np.uint8))
    return m > 0


def fit_shading(Y, bc, fg, mask, iters=6, order=1, ridge=0.05, xy=None, alternations=2):
    """SH coefficients per channel, one-sided IRLS, returned as (9, 3) with unused orders zero.

    Unregularised SH-L2 over a FACE is ill-conditioned: her normals span a narrow cone about
    the camera axis, the nine basis functions are nearly collinear on that set, and the fit
    returns coefficients of +-70 that cancel to a shading of order 1 -- and flip sign between
    adjacent frames (measured: the DC term ran +70 to -24 over 24 frames). The shading of a
    large soft source is L1 (ambient plus a cosine lobe), so that is the default; the
    directional terms carry a ridge toward zero, in units of the DC term.
    """
    nb = {0: 1, 1: 4, 2: 9}[order]
    C = np.zeros((9, 3), np.float32)
    # Spatial falloff: a near source is brighter on the parts of the face nearer to it, which
    # no function of the normal can express (measured: the hand, nearer the softbox than the
    # face, came out as one solid block of "shine"). Modelled as 1 + quadratic in the crop's
    # normalised (x, y), fitted alternately with the SH terms; P is shared by the channels.
    P = np.zeros(5, np.float32)
    Q = None
    if xy is not None:
        u, v = xy[mask][:, 0], xy[mask][:, 1]
        Q = np.stack([u, v, u * u, u * v, v * v], 1).astype(np.float32)
    for c in range(3):
        A0 = Y[mask][:, :nb] * bc[mask][:, c:c + 1]
        b = fg[mask][:, c]
        w = np.ones(len(b), np.float32)
        scale = float(np.mean(np.abs(A0[:, 0]))) + 1e-6
        R = np.zeros((nb, nb), np.float32)
        R[1:, 1:] = np.eye(nb - 1, dtype=np.float32) * (ridge * scale * np.sqrt(len(b)))
        coef = None
        for alt in range(alternations if Q is not None else 1):
            fall = (1.0 + Q @ P) if Q is not None else 1.0
            A = A0 * (fall[:, None] if Q is not None else 1.0)
            for _ in range(iters):
                Aw = np.concatenate([A * w[:, None], R], 0)
                bw = np.concatenate([b * w, np.zeros(nb, np.float32)], 0)
                coef, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
                r = b - A @ coef
                sd = 1.4826 * float(np.median(np.abs(r))) + 1e-6
                w = np.where(r > 0, 1.0 / (1.0 + (r / (1.5 * sd)) ** 2), 1.0).astype(np.float32)
            if Q is not None and c == 0 and alt < alternations - 1:
                # Refit the falloff for the SH just found: b ~ base * (1 + Q P), linear in P.
                base = A0 @ coef
                rq = b - base
                Pw, *_ = np.linalg.lstsq((Q * base[:, None]) * w[:, None], rq * w, rcond=None)
                P = Pw.astype(np.float32)
        C[:nb, c] = coef
    return C, P


def stage_separate(shot, root, args):
    import torch
    from fractions import Fraction
    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    meta = json.loads((root / 'input.json').read_text())
    crop = meta['crop_xyxy']; x0, y0, x1, y1 = crop
    side = x1 - x0
    W, H = plate_size(shot)
    frames = [f for f in meta['frames'] if (root / 'inverse' / 'basecolor' / ('%06d.png' % f)).exists()]
    if args.frames > 0:
        frames = frames[:args.frames]
    if not frames:
        raise SystemExit('no inverse basecolor: run --stage inverse first')
    sep = json.loads((shot.log('specular_separation.json')).read_text())
    g = np.array(sep['illuminant']['candidates'][sep['illuminant']['source']]['rgb'], np.float32)
    g_hat = g / np.linalg.norm(g)
    from sh_project import sh_basis
    topo = json.loads((shot.relight / 'face_landmarks.json').read_text())['topology']
    with np.load(shot.relight / 'face_landmarks.npz') as z:
        lm_pts, lm_found = z['points'], z['found']
    fps = float(Fraction(str(shot.frame_rate)))
    win = max(1, int(round(args.shading_window * fps)))
    win += (win + 1) % 2   # odd

    # Pass 1: per-frame data and the shading fit.
    per = {}
    for f in frames:
        fg = cv2.imread(str(shot.fg(f)), -1)[y0:y1, x0:x1, ::-1].astype(np.float32)
        bc = srgb_to_linear(read_rgb8(root / 'inverse' / 'basecolor' / ('%06d.png' % f)).astype(np.float32) / 255.0)
        bc = cv2.resize(bc, (side, side), interpolation=cv2.INTER_CUBIC)
        z = np.load(shot.relight / 'normals' / ('%06d.npz' % f))
        n = z['normal'].astype(np.float32)
        n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)
        n = cv2.resize(n, (W, H), interpolation=cv2.INTER_LINEAR)[y0:y1, x0:x1]
        n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)
        Y = sh_basis(n[..., 0], n[..., 1], n[..., 2]).astype(np.float32)
        skin = read_mask(root / 'skin' / ('%06d.png' % f))
        core = cv2.erode((skin > 0.5).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        excl = np.zeros(core.shape, bool)
        ipd = None
        if lm_found[f]:
            pts = lm_pts[f].copy(); pts[:, :2] -= np.array([x0, y0], np.float32)
            ipd = float(np.linalg.norm(pts[33, :2] - pts[263, :2]))
            grow = int(round(args.exclude_grow * ipd))
            for key in ('eyelid_left', 'eyelid_right', 'lips_inner'):
                excl |= polygon_mask(pts, topo[key], core.shape, grow)
        if lm_found[f] and args.face_only:
            # The stage is about the FACE: the hand and the neck sit at other distances from
            # the source and their shading is not the face's. The skin mask keeps them.
            core &= polygon_mask(pts, topo['face_oval'], core.shape, int(round(args.oval_grow * ipd)))
        fitmask = core & ~excl
        uv = np.stack(np.meshgrid((np.arange(side) - side / 2) / side, (np.arange(side) - side / 2) / side), -1).astype(np.float32)
        if fitmask.sum() > 500:
            C, P = fit_shading(Y, bc, fg, fitmask, order=args.sh_order, ridge=args.ridge, xy=uv if args.falloff else None)
        else:
            C, P = None, None
        per[f] = dict(fg=fg, bc=bc, Y=Y, core=core, excl=excl, C=C, P=P, ipd=ipd, skin=skin, uv=uv)

    # Temporal filter of the shading coefficients: symmetric median over `win` frames, so a
    # single frame's fit cannot flicker the whole face (round 37's lesson, applied to C).
    Cs = np.array([per[f]['C'] if per[f]['C'] is not None else np.full((9, 3), np.nan, np.float32) for f in frames])
    Ps = np.array([per[f]['P'] if per[f]['P'] is not None else np.full(5, np.nan, np.float32) for f in frames])
    Cf, Pf = Cs.copy(), Ps.copy()
    h = win // 2
    for k in range(len(frames)):
        lo, hi = max(0, k - h), min(len(frames), k + h + 1)
        block = Cs[lo:hi]
        if np.isfinite(block).all(axis=(1, 2)).any():
            Cf[k] = np.nanmedian(block, axis=0)
            Pf[k] = np.nanmedian(Ps[lo:hi], axis=0)

    ipd_med = float(np.median([per[f]['ipd'] for f in frames if per[f]['ipd']])) if any(per[f]['ipd'] for f in frames) else side / 4
    sigma = max(0.5, args.smooth * ipd_med)
    gt = torch.tensor(g_hat, device=dev)
    rows = []
    S_stack = []
    for k, f in enumerate(frames):
        d = per[f]
        if not np.isfinite(Cf[k]).all():
            S_full = np.zeros((H, W, 3), np.float32); wfull = np.zeros((H, W), np.uint8)
            rows.append(dict(frame=f, fitted=False)); S_stack.append(np.zeros((side, side), np.float32))
        else:
            fg = torch.from_numpy(d['fg']).to(dev)
            uv = d['uv']
            fall = 1.0 + np.stack([uv[..., 0], uv[..., 1], uv[..., 0] ** 2, uv[..., 0] * uv[..., 1], uv[..., 1] ** 2], -1) @ Pf[k]
            D = torch.from_numpy(d['bc']).to(dev) * torch.from_numpy((d['Y'] @ Cf[k]) * fall[..., None]).to(dev).clamp_min(0)
            r = fg - D
            b_hat = D / D.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            # 2x2 NNLS per pixel: r ~ a*b_hat + s*g_hat, a,s >= 0.
            bb = (b_hat * b_hat).sum(-1); bg = (b_hat * gt).sum(-1); gg = float(gt @ gt)
            rb = (r * b_hat).sum(-1); rg = (r * gt).sum(-1)
            det = (bb * gg - bg * bg).clamp_min(1e-6)
            a = (rb * gg - rg * bg) / det
            s = (rg * bb - rb * bg) / det
            # Clamp branches: if a < 0, refit s alone; if s < 0, nothing to remove.
            s_only = (rg / gg).clamp_min(0)
            s = torch.where(a < 0, s_only, s).clamp_min(0)
            mask_np = (d['core'] & ~d['excl']).astype(np.uint8)
            mask = torch.from_numpy(mask_np.astype(np.float32)).to(dev)
            # Feather: S fades to zero over `feather` IPD inside the mask edge, so the face
            # oval, the eye and mouth cut-outs and the skin mask never draw a step.
            feather_px = max(1.0, args.feather * (d['ipd'] or ipd_med))
            dist = cv2.distanceTransform(mask_np, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            feather = torch.from_numpy(np.clip(dist / feather_px, 0, 1).astype(np.float32)).to(dev)
            s = s * mask
            # Low-pass at a fraction of the IPD, normalised by the mask so the edge does not bleed.
            ksz = int(6 * sigma) | 1
            gk = torch.exp(-0.5 * (torch.arange(ksz, device=dev, dtype=torch.float32) - ksz // 2) ** 2 / sigma ** 2)
            gk = gk / gk.sum()
            def blur(x):
                x = x[None, None]
                x = torch.nn.functional.conv2d(x, gk.view(1, 1, 1, -1), padding=(0, ksz // 2))
                x = torch.nn.functional.conv2d(x, gk.view(1, 1, -1, 1), padding=(ksz // 2, 0))
                return x[0, 0]
            s_lp = blur(s) / blur(mask).clamp_min(1e-3) * (mask > 0)
            inside = mask > 0
            floor = float(s_lp[inside].median()) if inside.any() else 0.0
            s_lp = (s_lp - floor).clamp_min(0) * mask * feather
            S = s_lp[..., None] * gt
            S = torch.minimum(S, fg.clamp_min(0))       # never below black
            S_np = S.cpu().numpy()
            S_full = np.zeros((H, W, 3), np.float32); S_full[y0:y1, x0:x1] = S_np
            wfull = np.zeros((H, W), np.uint8); wfull[y0:y1, x0:x1] = (s_lp.cpu().numpy() > 1e-4).astype(np.uint8) * 255
            lum_p = d['fg'] @ LUMA; lum_s = S_np @ LUMA
            core = d['core']
            rows.append(dict(frame=f, fitted=True, floor=floor, sigma_px=sigma,
                             S_over_plate_skin=float(lum_s[core].sum() / max(lum_p[core].sum(), 1e-6)),
                             S_p50_p95_max=[float(np.percentile(lum_s[core], q)) for q in (50, 95, 100)],
                             painted_frac_skin=float((lum_s[core] > 1e-4).mean()),
                             C=Cf[k].tolist(), P=Pf[k].tolist()))
            S_stack.append(lum_s)
        for sub, arr in (('S', np.rint(np.clip(S_full, 0, 1) * 65535).astype(np.uint16)), ('weight', wfull)):
            (root / 'separate' / sub).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(root / 'separate' / sub / ('%06d.png' % f)), np.ascontiguousarray(arr[..., ::-1]) if arr.ndim == 3 else arr)
        # Review tiles: plate | plate - S | S x3, crop scale.
        prev = np.concatenate([np.rint(np.clip(linear_to_srgb(d['fg']), 0, 1) * 255),
                               np.rint(np.clip(linear_to_srgb(np.maximum(d['fg'] - S_full[y0:y1, x0:x1], 0)), 0, 1) * 255),
                               np.rint(np.clip(linear_to_srgb(S_full[y0:y1, x0:x1] * 3), 0, 1) * 255)], 1).astype(np.uint8)
        save_rgb8(root / 'separate' / 'preview' / ('%06d.png' % f), prev)
    st = np.stack(S_stack)
    skin_all = np.mean(np.stack([per[f]['core'] for f in frames]), 0) > 0.9
    step = np.abs(np.diff(st, axis=0))
    fitted = [r for r in rows if r['fitted']]
    summary = dict(frames=len(frames), sigma_px=sigma, ipd_px=ipd_med, shading_window_frames=win,
                   illuminant_rgb=g.tolist(),
                   S_over_plate_skin=dict(p50=float(np.median([r['S_over_plate_skin'] for r in fitted])),
                                          max=float(max(r['S_over_plate_skin'] for r in fitted))),
                   S_temporal_step_skin=dict(p95=float(np.percentile(step[:, skin_all], 95)) if len(frames) > 1 else 0.0,
                                             max=float(step[:, skin_all].max()) if len(frames) > 1 else 0.0),
                   S_temporal_spread_skin_p95=float(np.percentile(np.percentile(st, 95, 0)[skin_all] - np.percentile(st, 5, 0)[skin_all], 95)),
                   floor_p50=float(np.median([r['floor'] for r in fitted])),
                   code_sha256=sha(__file__))
    write_json(root / 'separate' / 'separate.json', dict(summary=summary, per_frame=rows))
    write_json(shot.logs / ('%s_separate.json' % root.name), dict(summary=summary, per_frame=rows))
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_shot_argument(ap)
    ap.add_argument('--name', default='r43_delight')
    ap.add_argument('--stage', choices=('prepare', 'inverse', 'unreflect', 'analyse', 'separate'), required=True)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--frames', type=int, default=0, help='0 = the whole shot (prepare) / all prepared (models)')
    ap.add_argument('--pad', type=float, default=0.18, help='crop padding as a fraction of the bbox side')
    ap.add_argument('--size', type=int, default=512, help='DiffusionRenderer inference side')
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--seed', type=int, default=43)
    ap.add_argument('--chunk', type=int, default=24)
    ap.add_argument('--overlap', type=int, default=6)
    ap.add_argument('--passes', default='basecolor,roughness,normal')
    ap.add_argument('--unreflect-size', type=int, default=896)
    ap.add_argument('--dilation', type=int, default=5)
    ap.add_argument('--threshold', type=float, default=0.3)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--sheet-frames', type=int, default=6)
    ap.add_argument('--sh-order', type=int, default=1, choices=(0, 1, 2))
    ap.add_argument('--no-falloff', dest='falloff', action='store_false', help='drop the spatial quadratic falloff term')
    ap.add_argument('--all-skin', dest='face_only', action='store_false', help='fit and paint the whole skin mask, not just the face oval')
    ap.add_argument('--oval-grow', type=float, default=0.08, help='face-oval growth, fraction of IPD')
    ap.add_argument('--feather', type=float, default=0.06, help='S fades to zero over this fraction of the IPD inside every mask edge')
    ap.add_argument('--ridge', type=float, default=0.05, help='ridge on the directional SH terms, in DC units')
    ap.add_argument('--smooth', type=float, default=0.02, help='low-pass sigma on the shine, as a fraction of the IPD')
    ap.add_argument('--shading-window', type=float, default=0.1, help='SECONDS of symmetric median on the shading fit')
    ap.add_argument('--exclude-grow', type=float, default=0.06, help='eye / mouth-interior exclusion growth, fraction of IPD')
    args = ap.parse_args()
    shot = from_args(args)
    root = shot.composite / args.name
    root.mkdir(parents=True, exist_ok=True)
    dict(prepare=stage_prepare, inverse=stage_inverse, unreflect=stage_unreflect,
         analyse=stage_analyse, separate=stage_separate)[args.stage](shot, root, args)


if __name__ == '__main__':
    main()
