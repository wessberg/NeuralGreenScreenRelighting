"""Hand the eye to a temporally-coherent VIDEO inpainter, using this project's own mask.

Two steps, because the model runs in its own interpreter:

    # 1. export a limbus-stabilized eye crop sequence and its mask
    dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/eye_video_inpaint.py \\
        --shot music_videos/reporter_test/shots/reporter_a --export

    # 2. run ProPainter on each eye (its own venv), then
    dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/eye_video_inpaint.py \\
        --shot music_videos/reporter_test/shots/reporter_a --apply

**Why this model class, after round 31 tried the wrong one.** Round 31 ran
DiffusionRenderer, which is a *relighting* model: it re-renders a surface under new
light. Removing a softbox reflection from a photographed eye is not relighting, it is
**masked video inpainting**, and that has purpose-built pretrained answers. This project's
own prior-art table has listed ProPainter as the "temporal removal baseline" since round
12 and never run it.

It is also the tool whose core assumption this project has already MEASURED to hold.
ProPainter works by flow-guided propagation: it fills a masked region from the same
material seen in other frames, and only synthesises what no frame shows. Round 33's
occupancy map (`logs/r33_occupancy.json`) established exactly that precondition -- the
highlight sweeps the eye's own frame, so 506 of 507 ever-corrupt cells left and 550 of
621 right are observed CLEAN in other frames. Temporal consistency is its design goal,
and temporal instability is what got rounds 32, 33 and 34's first pass rejected.

**Everything hard-won in rounds 12-34 is what makes the input usable**, and none of it is
thrown away:

* the tracked limbus (`eye_reflection_repair.limbus_ellipse`, mediapipe landmarks) gives
  the crop centre and scale, so the sequence handed to the model is **stabilized**: the
  eye barely moves in it and the model's flow module only has to follow eye rotation, not
  head motion;
* the sequence atlas (`eye_sequence_atlas.py`) defines the footprint against ONE fixed
  reference rather than a per-frame refit, which is what stops the mask breathing;
* round 34's continuous weight supplies the mask, including the coding undershoot rim --
  one artefact, both lobes;
* the eyelid polygon and the limbus bound where anything may be written;
* and the crop is upscaled, because a 24-pixel iris is far below what any video model
  resolves.

The output is composited back through round 34's own soft weight, so **every pixel
outside the reported weight stays bit-exact source** and a frame that abstains is
untouched. What the model contributes is confined to the footprint.

**Licence.** ProPainter is released under S-Lab Licence 1.0, which is NONCOMMERCIAL. This
is an evaluation, exactly as `docs/iris-prior-art.md` says it must be. Do not adopt it as
a production dependency for the artist's own videos without resolving that.
"""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from color_pipeline import linear_to_srgb, srgb_to_linear
from shot_paths import add_shot_argument, from_args
import eye_reflection_repair as r32
from eye_soft_removal import lid_matte

SIDES = r32.SIDES


def smooth_series(values, half=2):
    """Local quadratic fit of a per-frame vector series, the same smoother the corneal
    anchor uses. The limbus landmark's own jitter was measured against exactly this fit
    in round 33 (0.353 px rms left, 0.256 right), so fitting it out costs no real eye
    motion -- and an unstabilized crop hands the inpainter jitter to inpaint.
    """
    idx = np.asarray(sorted(values), float)
    pts = np.asarray([values[int(i)] for i in idx], float)
    out = {}
    for k, fi in enumerate(idx):
        lo, hi = max(0, k - half), min(len(idx), k + half + 1)
        t = idx[lo:hi] - fi
        deg = 2 if (hi - lo) >= 3 else (1 if (hi - lo) >= 2 else 0)
        v = np.polynomial.polynomial.polyvander(t, deg)
        coef, *_ = np.linalg.lstsq(v, pts[lo:hi], rcond=None)
        out[int(fi)] = coef[0]
    return out


def crop_maps(centre, ax, ay, half_radii, n):
    """Sampling grid for a stabilized crop, in native pixels.

    Isotropic in LIMBUS RADII, so the eye keeps a constant size in the crop even as the
    tracked ellipse changes with gaze -- the model then sees a nearly static eye.
    """
    t = np.linspace(-half_radii, half_radii, n).astype(np.float32)
    u, v = np.meshgrid(t, t)
    mx = (centre[0] + u * ax).astype(np.float32)
    my = (centre[1] + v * ay).astype(np.float32)
    return mx, my, u, v


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_shot_argument(ap)
    ap.add_argument('--from-name', default='r34_removal',
                    help='the run whose source frames and removal weight are used')
    ap.add_argument('--name', default='r34_inpaint')
    ap.add_argument('--export', action='store_true')
    ap.add_argument('--collect', action='store_true',
                    help='gather the per-phase ProPainter output into <side>/result')
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--strides', default='5,7,11',
                    help="temporal strides to run, comma separated. More than one, and "
                         "the darkest result per pixel is kept, because whether a given "
                         "pass removes the highlight at all is PHASE DEPENDENT: at a "
                         "single stride of 5 the left eye's phases 2 and 3 came back with "
                         "a median output/input peak ratio of 0.95 and 0.96 -- the "
                         "highlight essentially intact -- while phases 0, 1 and 4 sat at "
                         "0.59-0.68. A failed pass leaves the region BRIGHT, so a minimum "
                         "over independent passes is the estimator that matches the "
                         "failure mode, and it is the same logic as the standing rule "
                         "that a removal must darken.")
    ap.add_argument('--stride', type=int, default=5,
                    help="temporal stride of each exported phase. THIS IS THE WHOLE "
                         "TRICK. ProPainter fills a hole by propagating along completed "
                         "optical flow between ADJACENT frames, so it needs the occluder "
                         "to move between them. At 59.94 fps this highlight creeps about "
                         "0.34 px per frame in the eye's own coordinates -- consecutive "
                         "frames are nearly identical, nothing is ever uncovered, and the "
                         "model falls back on its prior. Measured: run at stride 1 it put "
                         "the catchlight BACK, peak 224 display against 215 in the "
                         "source. At stride 5 the same model and the same masks take the "
                         "peak down to 110-150. Every frame is still processed; they are "
                         "just processed in `stride` interleaved passes.")
    ap.add_argument('--crop', type=int, default=192, help='stabilized crop size in pixels')
    ap.add_argument('--half-radii', type=float, default=2.0,
                    help='crop half-width in limbus radii')
    ap.add_argument('--mask-dilate', type=int, default=1,
                    help='native-pixel dilation of the removal weight before export')
    ap.add_argument('--catchlight', default=None,
                    help='directory of rendered corneal-reflection EXRs to ADD after the '
                         'inpaint, through the same eyelid-and-limbus matte')
    ap.add_argument('--catchlight-gain', type=float, default=0.61502,
                    help='measured by eye_soft_removal.py --catchlight-gain auto as '
                         '1 / median(E_target/E_source) at the eyes')
    ap.add_argument('--sheet-zoom', type=int, default=4)
    ap.add_argument('--sheet-frames', type=int, default=25)
    args = ap.parse_args()
    if not (args.export or args.apply or args.collect):
        raise SystemExit('Pass --export, --collect or --apply')
    strides = [int(x) for x in str(args.strides).split(',') if x.strip()]
    shot = from_args(args)
    src_root = shot.composite / args.from_name
    root = shot.composite / args.name
    log = json.loads(shot.log(args.from_name + '.json').read_text())
    rows = {r['frame']: r for r in log['rows']}

    # Per side: the frames that were actually repaired, and their smoothed limbus.
    geom = {s: {} for s in SIDES}
    for i, row in rows.items():
        for e in row['eyes']:
            if e.get('repaired') and 'limbus_centre' in e:
                geom[e['side']][i] = e
    smoothed = {s: smooth_series({i: e['limbus_centre'] for i, e in geom[s].items()})
                for s in SIDES}

    if args.export:
        meta = {}
        for side in SIDES:
            frames = sorted(geom[side])
            if len(frames) < 10:
                continue
            for st in strides:
                for ph in range(st):
                    tag = 's%dp%d' % (st, ph)
                    (root / side / tag / 'frames').mkdir(parents=True, exist_ok=True)
                    (root / side / tag / 'masks').mkdir(parents=True, exist_ok=True)
            # ONE crop scale for the whole shot, from the median tracked semi-axes, so
            # the exported sequence has no per-frame zoom of its own for the model's
            # flow to chase.
            ax = float(np.median([geom[side][i]['limbus_axes'][0] for i in frames]))
            ay = float(np.median([geom[side][i]['limbus_axes'][1] for i in frames]))
            recs = []
            for k, i in enumerate(frames):
                img = cv2.imread(str(src_root / 'source' / ('%06d.png' % i)))
                w8 = cv2.imread(str(src_root / 'weight' / ('%06d.png' % i)),
                                cv2.IMREAD_GRAYSCALE)
                if img is None or w8 is None:
                    raise SystemExit('Missing source or weight for frame %d' % i)
                c = smoothed[side][i]
                mx, my, _, _ = crop_maps(c, ax, ay, args.half_radii, args.crop)
                crop = cv2.remap(img, mx, my, cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
                # The weight belongs to ONE eye; isolate it before resampling, or the
                # other eye's footprint can bleed in through a wide crop.
                lim = np.asarray(geom[side][i]['limbus_centre'], float)
                yy, xx = np.mgrid[:w8.shape[0], :w8.shape[1]]
                own = (np.hypot((xx - lim[0]) / max(ax, 1e-6),
                                (yy - lim[1]) / max(ay, 1e-6)) < args.half_radii)
                m = np.where(own, w8, 0).astype(np.uint8)
                if args.mask_dilate > 0:
                    k2 = 2 * args.mask_dilate + 1
                    m = cv2.dilate(m, np.ones((k2, k2), np.uint8))
                mcrop = cv2.remap(m, mx, my, cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT)
                places = []
                for st in strides:
                    ph, sub = k % st, k // st
                    pdir = root / side / ('s%dp%d' % (st, ph))
                    cv2.imwrite(str(pdir / 'frames' / ('%05d.png' % sub)), crop)
                    cv2.imwrite(str(pdir / 'masks' / ('%05d.png' % sub)),
                                ((mcrop > 0) * 255).astype(np.uint8))
                    places.append(dict(stride=st, phase=ph, sub=sub))
                recs.append(dict(index=k, places=places, frame=i,
                                 centre=[float(c[0]), float(c[1])]))
            meta[side] = dict(frames=recs, ax=ax, ay=ay, crop=args.crop,
                              half_radii=args.half_radii, strides=strides)
            print('R34_EXPORT', side, len(recs), 'frames,', sum(strides), 'passes ->',
                  root / side, flush=True)
        r32.write_json(root / 'export.json',
                       dict(source=args.from_name, sides=meta,
                            crop=args.crop, half_radii=args.half_radii,
                            strides=strides, mask_dilate=args.mask_dilate))
        print('R34_EXPORT_DONE ->', root, flush=True)
        return

    meta = json.loads((root / 'export.json').read_text())
    if args.collect:
        for side in SIDES:
            m = meta['sides'].get(side)
            if m is None:
                continue
            out = root / side / 'result'
            out.mkdir(parents=True, exist_ok=True)
            missing, kept = 0, []
            for rec in m['frames']:
                got = []
                for pl in rec['places']:
                    src = (root / side / ('s%dp%d' % (pl['stride'], pl['phase'])) / 'pp' /
                           'frames' / 'frames' / ('%04d.png' % pl['sub']))
                    if src.exists():
                        img = cv2.imread(str(src))
                        if img is not None:
                            got.append(img.astype(np.float32))
                if not got:
                    missing += 1
                    continue
                # Darkest per pixel across the passes. A pass that failed to remove the
                # highlight is BRIGHT there; a pass that succeeded is not.
                best = np.min(np.stack(got), axis=0)
                kept.append(len(got))
                cv2.imwrite(str(out / ('%05d.png' % rec['index'])),
                            np.rint(best).astype(np.uint8))
            print('R34_COLLECT', side, len(m['frames']) - missing, 'of',
                  len(m['frames']), 'frames, passes per frame median',
                  int(np.median(kept)) if kept else 0, flush=True)
        return

    # --- apply ------------------------------------------------------------------------
    subs = ['source', 'inpaint', 'weight', 'sheets']
    catch = Path(args.catchlight) if args.catchlight else None
    if catch is not None:
        if not catch.is_dir():
            raise SystemExit('No catchlight render directory at %s' % catch)
        subs.append('replace')
    for sub in subs:
        (root / sub).mkdir(parents=True, exist_ok=True)
    filled = {}
    for side in SIDES:
        m = meta['sides'].get(side)
        if m is None:
            continue
        res = root / side / 'result'
        if not res.is_dir():
            raise SystemExit('No %s. Run ProPainter into that folder first.' % res)
        for rec in m['frames']:
            p = res / ('%05d.png' % rec['index'])
            if not p.exists():
                raise SystemExit('Missing inpainted frame %s' % p)
            filled[(rec['frame'], side)] = (cv2.imread(str(p)), rec, m)

    rows_out, tiles = [], []
    topology = json.loads((shot.relight / 'face_landmarks.json').read_text())['topology']
    with np.load(shot.relight / 'face_landmarks.npz') as z:
        points, found = z['points'], z['found']
    for i in sorted(rows):
        raw = cv2.imread(str(shot.fg(i)), cv2.IMREAD_UNCHANGED)
        values = raw[..., ::-1].astype(np.float32)
        w8 = cv2.imread(str(src_root / 'weight' / ('%06d.png' % i)), cv2.IMREAD_GRAYSCALE)
        out = values.copy()
        combined = np.zeros(values.shape[:2], np.float32)
        eyes = []
        for side in SIDES:
            got = filled.get((i, side))
            if got is None:
                eyes.append(dict(side=side, inpainted=False, reason='not_repaired'))
                continue
            crop, rec, m = got
            e = geom[side][i]
            lim = np.asarray(e['limbus_centre'], float)
            ax, ay = m['ax'], m['ay']
            c = np.asarray(rec['centre'], float)
            h, wdt = values.shape[:2]
            yy, xx = np.mgrid[:h, :wdt]
            # Inverse of the export map: native pixel -> crop pixel.
            cu = ((xx - c[0]) / ax) / m['half_radii']
            cv_ = ((yy - c[1]) / ay) / m['half_radii']
            gx = ((cu + 1.0) * 0.5 * (m['crop'] - 1)).astype(np.float32)
            gy = ((cv_ + 1.0) * 0.5 * (m['crop'] - 1)).astype(np.float32)
            inside = (np.abs(cu) < 1.0) & (np.abs(cv_) < 1.0)
            back = cv2.remap(crop, gx, gy, cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
            lin = srgb_to_linear(np.clip(back[..., ::-1].astype(np.float32) / 255.0, 0, 1))
            # Round 34's own soft weight, unchanged, and only where this eye is.
            own = (np.hypot((xx - lim[0]) / max(ax, 1e-6),
                            (yy - lim[1]) / max(ay, 1e-6)) < m['half_radii'])
            w = np.where(own & inside, w8.astype(np.float32) / 255.0, 0.0).astype(np.float32)
            out = out * (1 - w[..., None]) + lin * w[..., None]
            combined = np.maximum(combined, w)
            eyes.append(dict(side=side, inpainted=True, reason=None,
                             changed_pixels=int((w > 0).sum())))
        untouched = combined == 0
        out = np.maximum(out, 0.0)
        out[untouched] = values[untouched]
        assert np.array_equal(out[untouched], values[untouched])
        cv2.imwrite(str(root / 'source' / ('%06d.png' % i)), r32.rgb8(values)[..., ::-1])
        cv2.imwrite(str(root / 'inpaint' / ('%06d.png' % i)), r32.rgb8(out)[..., ::-1])
        if catch is not None:
            add = cv2.imread(str(catch / ('%06d.exr' % i)), cv2.IMREAD_UNCHANGED)
            if add is None:
                raise SystemExit('No catchlight frame %06d.exr' % i)
            add = add[..., :3][..., ::-1].astype(np.float32) * args.catchlight_gain
            matte = np.zeros(values.shape[:2], np.float32)
            if found[i] and np.isfinite(points[i]).all():
                for side in SIDES:
                    g, _ = r32.eye_frame(values, points[i], topology, side)
                    if g is None:
                        continue
                    matte[g['window']] = np.maximum(
                        matte[g['window']],
                        lid_matte(points[i], topology, side, g['window'], g))
            rep = np.maximum(out + add * matte[..., None], 0.0)
            cv2.imwrite(str(root / 'replace' / ('%06d.png' % i)),
                        r32.rgb8(rep)[..., ::-1])
        cv2.imwrite(str(root / 'weight' / ('%06d.png' % i)),
                    np.rint(np.clip(combined, 0, 1) * 255).astype(np.uint8))
        if found[i] and np.isfinite(points[i]).all():
            tiles.append(r32.eye_tile(values, out, combined, points[i], topology,
                                      args.sheet_zoom, shot.source_start_frame + i,
                                      [dict(side=e['side'], repaired=e.get('inpainted'),
                                            anomaly_pixels=e.get('changed_pixels', 0),
                                            reason=e.get('reason'))
                                       for e in eyes]))
        rows_out.append(dict(frame=i, source_frame=shot.source_start_frame + i,
                             source_sha256=r32.sha(shot.fg(i)),
                             changed_pixels=int((combined > 0).sum()), eyes=eyes))
        print('R34_INPAINT frame', i, flush=True)

    sheets = []
    for k in range(0, len(tiles), args.sheet_frames):
        p = root / 'sheets' / ('contact_%03d.jpg' % k)
        cv2.imwrite(str(p), r32.contact_sheet(tiles[k:k + args.sheet_frames]),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        sheets.append(p.name)
    record = dict(rows=rows_out, contact_sheets=sheets, frames=len(rows_out),
                  source=args.from_name,
                  model='ProPainter (S-Lab Licence 1.0, NONCOMMERCIAL -- evaluation only)',
                  method='limbus-stabilized upscaled eye crop + round 34 removal weight, '
                         'flow-guided video inpainting, composited back through the same '
                         'soft weight',
                  generated_material_used=True,
                  scene_reflection_added=catch is not None,
                  catchlight_render=None if catch is None else str(catch),
                  catchlight_gain=args.catchlight_gain,
                  source_outside_weight_exact=True,
                  production_enabled=False, artist_accepted=False,
                  code_sha256=r32.sha(__file__))
    r32.write_json(root / 'inpaint.json', record)
    r32.write_json(shot.log(args.name + '.json'), record)
    print('R34_INPAINT_DONE', len(rows_out), flush=True)


if __name__ == '__main__':
    main()
