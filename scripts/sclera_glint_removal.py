"""Remove the capture room's reflections from the WHITE of her eye, on top of the accepted iris pass.

Round 42. The iris pass (`eye_removal_gpu.py`, accepted as `r40_floorB`) stops at the limbus,
and round 38 logged a second capture-room glint on the sclera. `audit_r42_wet.py` measured it
on this shot: a 1-3 px blob on the lower-temporal white of her right eye in 100 of 103 open
frames, a nasal-lower one on the left eye in 58 of 109, both at the same image angle
(about -150 deg), plus a tear-lake glint at the right eye's nasal corner in 95. They are the
OLD room's light; the ratio can only recolour them, and the wet globe rendered by
`stress_proxy.py --corneal --globe` puts the NEW scene's reflection where physics says.

What this does, per open eye, on the GPU:

* finds the blobs exactly as the audit does (local contrast over a 9x9 median of the display
  luma, above a floor, inside the fitted opening and outside 1.10 limbus radii, area <= 60),
  classes a blob touching the corner of the opening as `corner` and leaves it alone unless
  `--corners` (a wet caruncle glints too, nothing rendered replaces it yet);
* holds the footprint in TIME: a pixel is in it when hot now, or hot in the previous or next
  frame (registered on the limbus centre) and still warm now -- round 32's lesson, a
  footprint that is refit per frame flickers at its edges;
* fills it by normalised convolution of the sclera around it (sigma `--sigma` px, the
  footprint dilated 2 px excluded from the support) and blends through a 1 px feather.
  The sclera is smooth, so the fill is a local level, never texture; nothing is invented.

Writes composite/<name>/{removal,weight,source}/ in the eye pass's own layout (the run's
frames with this applied, the union of the weights, the untouched source), so
`composite_ratio.py --eye-removal <name>` substitutes both in one go. Reports per-eye
counts, the mean fill level against the sclera it was taken from, and the frame-to-frame step
of the filled region's level against an untouched control ring (p95 and max, never a median
alone).

    dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/sclera_glint_removal.py \
        --shot music_videos/reporter_test/shots/reporter_a --run r40_floorB_import --name r42_wet_removal
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shot_paths import add_shot_argument, from_args
from corneal_anchor import smoothed_iris_centres
from eyelid_fit import evaluate as lid_evaluate
from eye_reflection_repair import lid_distance as landmark_lid_distance
from validate_tour_catchlight import load_track, open_eyes

LUMA = torch.tensor([0.2126, 0.7152, 0.0722])


def gauss_kernel(sigma, device):
    r = int(np.ceil(3 * sigma))
    x = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return (k / k.sum()), r


def blur(img, k, r):
    """Separable Gaussian on (C, H, W)."""
    c = img.shape[0]
    x = F.pad(img[None], (r, r, r, r), mode='replicate')
    x = F.conv2d(x, k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    x = F.conv2d(x, k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)
    return x[0]


def dilate(mask, n):
    m = mask[None, None].float()
    for _ in range(n):
        m = F.max_pool2d(m, 3, 1, 1)
    return m[0, 0] > 0.5


def median9(luma):
    """9x9 median of a (H, W) float image, on the GPU, replicate-padded."""
    x = F.pad(luma[None, None], (4, 4, 4, 4), mode='replicate')
    p = F.unfold(x, 9)                       # (1, 81, H*W)
    return p.median(1).values.view(luma.shape)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_shot_argument(ap)
    ap.add_argument('--run', default='r40_floorB_import', help='composite/<run> of the iris pass')
    ap.add_argument('--name', default='r42_wet_removal')
    ap.add_argument('--limbus-track', default='limbus_track_tv_import.json')
    ap.add_argument('--eyelid-track', default='eyelid_track_wtv40_import.json')
    ap.add_argument('--excess', type=float, default=0.10,
                    help='display above the 9x9 median to be a glint (the audit\'s test)')
    ap.add_argument('--floor', type=float, default=0.80)
    ap.add_argument('--warm', type=float, default=0.5,
                    help='fraction of --excess that keeps a pixel in the footprint when a '
                         'neighbouring frame had it hot')
    ap.add_argument('--corners', action='store_true', help='also remove corner (tear-lake) glints')
    ap.add_argument('--sigma', type=float, default=2.5, help='fill support, px')
    ap.add_argument('--max-area', type=int, default=60)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--sheet-every', type=int, default=1, help='changed eyes per sheet stride')
    ap.add_argument('--eye-patches', default='', help='EyeClean float run; retain its iris pixels and export sclera-only corrections')
    args = ap.parse_args()
    shot = from_args(args)
    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    t0 = time.time()
    run = shot.composite / args.run
    patch_run = None
    repaired = {}
    if args.eye_patches:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
        from ngsr.eyes import EyeCleanPatches
        patch_run = EyeCleanPatches(args.eye_patches)
        if dev.type != 'cuda':
            raise SystemExit('Float EyeClean/sclera integration requires CUDA')
        run = patch_run.path
    out = shot.composite / args.name
    for sub in ('removal', 'weight', 'source'):
        (out / sub).mkdir(parents=True, exist_ok=True)
    track = load_track(shot, args.limbus_track)
    smooth = smoothed_iris_centres(track)
    lids = open_eyes(shot, args.eyelid_track)
    lm_points = lm_found = lm_topo = None
    if (shot.relight / 'face_landmarks.npz').exists():
        with np.load(shot.relight / 'face_landmarks.npz') as z:
            lm_points, lm_found = z['points'], z['found']
        lm_topo = json.loads((shot.relight / 'face_landmarks.json').read_text())['topology']
    frames = (sorted(patch_run.frames) if patch_run is not None else
              sorted(int(p.stem) for p in (run / 'removal').glob('*.png')))
    if patch_run is not None:
        (out / 'correction').mkdir(parents=True, exist_ok=True)
    n_frames = len(frames)
    kern, kr = gauss_kernel(args.sigma, dev)

    # ---- pass 1: per-eye windows, hot masks and warm masks --------------------------------
    eyes = {}          # (frame, side) -> dict(win, hot, warm, region, opening, cx, cy, r)
    imgs = {}
    for fi in frames:
        if patch_run is not None:
            fg = cv2.imread(str(shot.fg(fi)), -1)[..., ::-1].copy()
            clean, repaired[fi], _ = patch_run.apply(fg, fi, shot.fg(fi))
            disp = torch.as_tensor(clean, device=dev).clamp(min=0)
            disp = torch.where(disp <= .0031308, disp*12.92, 1.055*disp.pow(1/2.4)-.055)
            img = (disp.clamp(0,1)*255).round().byte().cpu().numpy()[..., ::-1].copy()
        else:
            img = cv2.imread(str(run / 'removal' / ('%06d.png' % fi)), -1)
        imgs[fi] = img
        t_img = torch.from_numpy(img[..., ::-1].copy()).to(dev).float() / 255.0     # RGB
        for side in ('left', 'right'):
            if patch_run is not None and side not in repaired[fi]:
                continue
            tr = (track.get(fi, {}).get('eyes') or {}).get(side)
            if tr is None:
                continue
            px = tr['iris_centre_px']
            sm = smooth.get(side, {}).get(fi)
            if sm is not None:
                px = (float(sm[0]), float(sm[1]))
            cx, cy = px
            r = float(tr['iris_radius_px'])
            reach = 3.0 * r + 8
            x0, x1 = int(cx - reach), int(cx + reach) + 1
            y0, y1 = int(cy - 2.4 * r), int(cy + 2.4 * r) + 1
            if x0 < 0 or y0 < 0 or x1 > img.shape[1] or y1 > img.shape[0]:
                continue
            win = (slice(y0, y1), slice(x0, x1))
            yy, xx = torch.meshgrid(torch.arange(y0, y1, device=dev, dtype=torch.float32),
                                    torch.arange(x0, x1, device=dev, dtype=torch.float32),
                                    indexing='ij')
            lid = lids.get((fi, side))
            if lid is not None:
                up = torch.from_numpy(np.asarray(lid_evaluate(lid['upper']['coef'],
                                                              xx.cpu().numpy(), lid['centre']),
                                                 np.float32)).to(dev)
                lo = torch.from_numpy(np.asarray(lid_evaluate(lid['lower']['coef'],
                                                              xx.cpu().numpy(), lid['centre']),
                                                 np.float32)).to(dev)
                if lid.get('lower_iris'):
                    lo = torch.minimum(lo, torch.from_numpy(np.asarray(lid_evaluate(
                        lid['lower_iris']['coef'], xx.cpu().numpy(), lid['centre']),
                        np.float32)).to(dev))
                opening = ((yy - up) > 1.0) & ((lo - yy) > 1.0)
                lid_source = 'image_fit'
            elif (lm_points is not None and fi < len(lm_found) and lm_found[fi]
                  and np.isfinite(lm_points[fi]).all()):
                d = landmark_lid_distance(lm_points[fi], lm_topo['eyelid_' + side], win)
                opening = torch.from_numpy(np.asarray(d > 1.0)).to(dev)
                lid_source = 'mediapipe_polygon'
            else:
                continue
            crop = t_img[y0:y1, x0:x1]
            luma = crop @ LUMA.to(dev)
            rho = torch.hypot(xx - cx, yy - cy) / r
            region = opening & (rho > 1.10) & (rho < 2.8)
            if region.sum() < 20:
                continue
            med = median9(luma)
            exc = luma - med
            hot = region & (exc > args.excess) & (luma > args.floor)
            warm = region & (exc > args.warm * args.excess) & (luma > args.floor - 0.05)
            # drop big components and corners on the CPU (tiny masks)
            hot_np = hot.cpu().numpy().astype(np.uint8)
            keep = np.zeros_like(hot_np)
            corner_px = 0
            if hot_np.any():
                n, lab, stats, cent = cv2.connectedComponentsWithStats(hot_np)
                edge = cv2.dilate((~opening).cpu().numpy().astype(np.uint8),
                                  np.ones((3, 3), np.uint8)) > 0
                for k in range(1, n):
                    area = int(stats[k, cv2.CC_STAT_AREA])
                    if area > args.max_area:
                        continue
                    m = lab == k
                    bx = cent[k][0] + x0
                    is_corner = bool((m & edge).any()) and abs(bx - cx) > 1.3 * r
                    if is_corner and not args.corners:
                        corner_px += area
                        continue
                    keep[m] = 1
            eyes[(fi, side)] = dict(win=win, hot=torch.from_numpy(keep).to(dev).bool(),
                                    warm=warm, region=region, opening=opening,
                                    cx=cx, cy=cy, r=r, lid_source=lid_source,
                                    corner_px_left=corner_px)
        if fi % 30 == 0:
            print('SCLERA pass1 frame %d/%d' % (fi, n_frames), flush=True)

    # ---- pass 2: temporal footprint, fill, write --------------------------------------
    def registered(src_key, dst):
        """The neighbour frame's hot mask shifted onto this frame's window by the integer
        limbus-centre offset (a globe glint rides the eye, not the frame)."""
        s = eyes.get(src_key)
        if s is None:
            return None
        dx = int(round(dst['cx'] - s['cx']))
        dy = int(round(dst['cy'] - s['cy']))
        m = s['hot']
        outm = torch.zeros_like(dst['hot'])
        sy0, sx0 = s['win'][0].start, s['win'][1].start
        dy0, dx0 = dst['win'][0].start, dst['win'][1].start
        # source pixel (y, x) lands at (y + dy, x + dx) in frame coords
        ys, xs = torch.where(m)
        ty = ys + sy0 + dy - dy0
        tx = xs + sx0 + dx - dx0
        ok = (ty >= 0) & (ty < outm.shape[0]) & (tx >= 0) & (tx < outm.shape[1])
        outm[ty[ok], tx[ok]] = True
        return outm

    per_eye = []
    level_series = {'left': {}, 'right': {}}
    control_series = {'left': {}, 'right': {}}
    union_fp = {'left': None, 'right': None}
    sheet_tiles = []
    changed_total = 0
    for fi in frames:
        img = imgs[fi]
        wgt = (np.zeros(img.shape[:2], np.uint8) if patch_run is not None else
               cv2.imread(str(run / 'weight' / ('%06d.png' % fi)), -1))
        if wgt.ndim == 3:
            wgt = wgt[..., 0]
        out_img = img.copy()
        out_w = wgt.copy()
        for side in ('left', 'right'):
            e = eyes.get((fi, side))
            if e is None:
                continue
            fp = e['hot'].clone()
            for nb in (fi - 1, fi + 1):
                rm = registered((nb, side), e)
                if rm is not None:
                    fp |= rm & e['warm']
            fp &= e['region']
            n_fp = int(fp.sum())
            rec = dict(frame=fi, side=side, footprint_px=n_fp, lid_source=e['lid_source'],
                       corner_px_left=e['corner_px_left'])
            if n_fp == 0:
                per_eye.append(rec)
                continue
            y0, y1 = e['win'][0].start, e['win'][0].stop
            x0, x1 = e['win'][1].start, e['win'][1].stop
            crop = torch.from_numpy(img[y0:y1, x0:x1, ::-1].copy()).to(dev).float() / 255.0
            crop = crop.permute(2, 0, 1)                        # (3, H, W) RGB display
            support = (e['opening'] & ~dilate(fp, 2) & (torch.hypot(
                torch.arange(x0, x1, device=dev, dtype=torch.float32)[None] - e['cx'],
                torch.arange(y0, y1, device=dev, dtype=torch.float32)[:, None] - e['cy'])
                / e['r'] > 1.05)).float()
            num = blur(crop * support[None], kern, kr)
            den = blur(support[None], kern, kr)
            fill = num / den.clamp_min(1e-6)
            have = den[0] > 0.05
            w = dilate(fp, 1).float()
            w = F.avg_pool2d(w[None, None], 3, 1, 1)[0, 0]
            w = torch.where(have, w, torch.zeros_like(w))
            if patch_run is not None:
                w = w * e['region']  # float mode cannot quantize EyeClean iris or lids
            res = crop * (1 - w[None]) + fill * w[None]
            changed = w > 0
            n_changed = int(changed.sum())
            changed_total += n_changed
            res8 = (res.permute(1, 2, 0).clamp(0, 1) * 255).round().byte().cpu().numpy()[..., ::-1]
            out_img[y0:y1, x0:x1][changed.cpu().numpy()] = res8[changed.cpu().numpy()]
            w8 = (w * 255).round().byte().cpu().numpy()
            out_w[y0:y1, x0:x1] = np.maximum(out_w[y0:y1, x0:x1], w8)
            lum_before = (crop.permute(1, 2, 0) @ LUMA.to(dev))
            lum_after = (res.permute(1, 2, 0) @ LUMA.to(dev))
            rec.update(changed_px=n_changed,
                       level_before=round(float(lum_before[fp].mean()), 4),
                       level_after=round(float(lum_after[fp].mean()), 4),
                       fill_support_px=int(support.sum()))
            # flicker series in eye coordinates: the filled level and a control ring
            ring = (support > 0) & (dilate(fp, 5) & ~dilate(fp, 3))
            level_series[side][fi] = float(lum_after[fp].mean())
            control_series[side][fi] = float(lum_after[ring].mean()) if ring.any() else None
            per_eye.append(rec)
            if len(sheet_tiles) < 400 and (fi % args.sheet_every == 0):
                Z = 6
                a = cv2.resize(img[y0:y1, x0:x1], None, fx=Z, fy=Z,
                               interpolation=cv2.INTER_NEAREST)
                b = cv2.resize(out_img[y0:y1, x0:x1], None, fx=Z, fy=Z,
                               interpolation=cv2.INTER_NEAREST)
                cv2.putText(a, '%d %s' % (fi, side[0].upper()), (3, 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                sheet_tiles.append(np.concatenate([a, b], 0))
        if patch_run is not None:
            fg = cv2.imread(str(shot.fg(fi)), -1)[..., ::-1].copy()
            clean, _, _ = patch_run.apply(fg, fi, shot.fg(fi))
            target = torch.as_tensor(out_img[..., ::-1].copy(), device=dev).float()/255
            target = torch.where(target <= .04045, target/12.92, ((target+.055)/1.055).pow(2.4))
            base = torch.as_tensor(clean, device=dev)
            support = torch.as_tensor(out_w > 0, device=dev)[..., None]
            original = torch.as_tensor(fg, device=dev)
            eye_edited = torch.any(base.view(torch.int32) != original.view(torch.int32), dim=-1)
            support = support & ~eye_edited[..., None]
            correction = torch.where(support, target-base, torch.zeros_like(base)).cpu().numpy()
            flags = [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT,
                     cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_ZIP]
            if not cv2.imwrite(str(out / 'correction' / ('%06d.exr' % fi)), correction[..., ::-1], flags):
                raise RuntimeError('Cannot write sclera float correction')
        cv2.imwrite(str(out / 'removal' / ('%06d.png' % fi)), out_img)
        cv2.imwrite(str(out / 'weight' / ('%06d.png' % fi)), out_w)
        src = run / 'source' / ('%06d.png' % fi)
        if src.exists():
            shutil.copyfile(src, out / 'source' / ('%06d.png' % fi))
        if fi % 30 == 0:
            print('SCLERA pass2 frame %d/%d' % (fi, n_frames), flush=True)

    # ---- report --------------------------------------------------------------------
    def steps(series):
        ks = sorted(k for k, v in series.items() if v is not None)
        st = [abs(series[b] - series[a]) for a, b in zip(ks, ks[1:]) if b == a + 1]
        if not st:
            return None
        return dict(n=len(st), p95=round(float(np.percentile(st, 95)), 4),
                    max=round(float(np.max(st)), 4))
    summary = {}
    for side in ('left', 'right'):
        mine = [r for r in per_eye if r['side'] == side]
        ch = [r for r in mine if r.get('changed_px')]
        summary[side] = dict(
            eyes_seen=len(mine), eyes_changed=len(ch),
            changed_px_total=int(sum(r['changed_px'] for r in ch)),
            footprint_px_median=float(np.median([r['footprint_px'] for r in ch])) if ch else 0,
            footprint_px_max=int(max([r['footprint_px'] for r in ch] or [0])),
            level_before_median=round(float(np.median([r['level_before'] for r in ch])), 4)
            if ch else None,
            level_after_median=round(float(np.median([r['level_after'] for r in ch])), 4)
            if ch else None,
            corner_px_left_total=int(sum(r['corner_px_left'] for r in mine)),
            filled_level_step=steps(level_series[side]),
            control_ring_step=steps(control_series[side]),
            on_polygon=sum(1 for r in ch if r['lid_source'] == 'mediapipe_polygon'))
    if sheet_tiles:
        hmax = max(t.shape[0] for t in sheet_tiles)
        wmax = max(t.shape[1] for t in sheet_tiles)
        pad = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, wmax - t.shape[1],
                                  cv2.BORDER_CONSTANT) for t in sheet_tiles]
        per_row = 8
        for si in range(0, len(pad), per_row * 5):
            chunk = pad[si:si + per_row * 5]
            rows = []
            for i in range(0, len(chunk), per_row):
                rr = chunk[i:i + per_row]
                rr += [np.zeros_like(pad[0])] * (per_row - len(rr))
                rows.append(np.concatenate(rr, 1))
            cv2.imwrite(str(out / ('sheet_%03d.jpg' % (si // (per_row * 5)))),
                        np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 92])
    code_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    src_json = run / 'removal.json'
    report = dict(shot=shot.name, run=args.run, name=args.name, frames=n_frames,
                  method='round 42 scleral glint removal: audit_r42_wet blob test, footprint '
                         'held in time (hot now, or hot in a neighbour frame and warm now), '
                         'normalised-convolution fill of the surrounding sclera, 1 px feather; '
                         'applied on top of the iris pass frames, weights unioned',
                  excess=args.excess, floor=args.floor, warm=args.warm, sigma_px=args.sigma,
                  corners=args.corners, max_area=args.max_area,
                  limbus_track=args.limbus_track, eyelid_track=args.eyelid_track,
                  device=str(dev), code_sha256=code_sha,
                  iris_run_removal_json_sha256=(hashlib.sha256(src_json.read_bytes()).hexdigest()
                                                if src_json.exists() else None),
                  changed_pixels_total=changed_total, summary=summary, per_eye=per_eye,
                  seconds=round(time.time() - t0, 1))
    if patch_run is not None:
        report['eye_patches'] = str(patch_run.path)
        report['eye_binding_sha256'] = hashlib.sha256((patch_run.path / 'ngsr.json').read_bytes()).hexdigest()
        report['correction_color'] = 'linear-rec709'
        report['float_contract'] = 'ngsr.sclera/1: add correction/ EXRs after EyeClean; its float iris stays untouched'
    (out / 'removal.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(summary, indent=1))
    print('SCLERA_WROTE', out, '%.1f s' % (time.time() - t0))


if __name__ == '__main__':
    main()
