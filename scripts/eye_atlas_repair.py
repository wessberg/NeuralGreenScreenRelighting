"""Remove the capture-room corneal highlight using ONE sequence-wide eye estimate.

This is round 33's answer to round 32's rejection. The artist's words were that the
result was not temporally stable "in terms of neither shape nor color", and that "the
pupil was sometimes brown". Each of those maps onto a specific thing that is different
here:

* **Colour.** The filled radiance is `atlas x gain`, where the atlas is estimated ONCE
  over the whole shot from observed clean pixels and never refit per frame. Round 32
  rebuilt its radial profile, its angular gain and its donors from a few dozen pixels of
  each frame, which is why its fill colour traversed R 36->132 and B 2->66 across the
  shot. Nothing in the atlas can move frame to frame, so that mechanism is gone.
* **Shape.** The footprint is detected against that same fixed prediction in every
  frame, not against a model refit per frame, and the resulting mask is then smoothed
  over a five-frame window in the eye's own coordinates. Round 32's mask breathed
  because its reference breathed.
* **The brown pupil.** The dark undershoot beside the highlight is left as observed
  source and never filled, exactly as in round 32's shipped method. What is new is that
  the fill's own colour at the centre now comes from pooled observations of that part of
  the eye rather than from a per-frame radial median.

The illumination term is a constant plus a linear ramp in eye coordinates, per channel,
temporally smoothed. It is deliberately the smallest term that can track the upper lid's
moving shadow across the top of the iris; a single scalar per channel cannot, and
anything richer starts refitting the eye per frame again, which is the failure this
module exists to avoid.

Why any of this is possible was measured first, in `eye_sequence_occupancy.py`: the
highlight SWEEPS the eye's own frame over the shot, so 506 of 507 ever-corrupt cells
left and 550 of 621 right are observed clean in other frames.

Everything outside the reported mask is bit-exact source, and a frame whose eye is
closed, untracked or unpredicted abstains and keeps exact source.

Writes no production default.

    dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/eye_atlas_repair.py \
        --shot music_videos/reporter_test/shots/reporter_a
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
from color_pipeline import linear_to_srgb
from shot_paths import add_shot_argument, from_args
import eye_reflection_repair as r32
import eye_sequence_atlas as atl


def predict(geom, atlas, aw, du, dv, scale, coef):
    """Render the sequence atlas back into this frame's native eye patch.

    Inverse of the pooling map: a frame's pixel at eye coordinate (p, q) was pooled into
    the atlas at ((p - du) / scale, (q - dv) / scale), so that is where it is read back.
    """
    xx, yy = geom['xx'], geom['yy']
    p = ((xx - geom['limbus'][0]) / geom['ax']).astype(np.float32)
    q = ((yy - geom['limbus'][1]) / geom['ay']).astype(np.float32)
    ai = (((p - du) / scale) / atl.ATLAS_STEP + (atl.ATLAS_N - 1) / 2).astype(np.float32)
    aj = (((q - dv) / scale) / atl.ATLAS_STEP + (atl.ATLAS_N - 1) / 2).astype(np.float32)
    base = cv2.remap(atlas, ai, aj, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    support = cv2.remap(aw.astype(np.float32), ai, aj, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT) > 0.98
    gain = atl.apply_gain(coef, atl.gain_basis(p, q))
    return np.maximum(base * gain, 0.0), support, (p, q)


def detect(patch, base, inside, sigmas=4.0, floor=0.012, reach=7.0, grow=1):
    """The footprint of the highlight against the FIXED sequence prediction.

    Same sign-aware bounded growth as round 32 -- the highlight is a positive core, the
    coding undershoot beside it a negative rim, and iris texture is both signs at random
    so growing on |residual| cannot tell them apart -- but the reference is the
    sequence atlas rather than a model refit from this frame's own few dozen pixels.
    Only the positive core is taken: the dark rim stays observed source.
    """
    signed = (patch - base).max(axis=-1)
    sample = signed[inside]
    if sample.size < 40:
        return None, None, 'too_few_usable_pixels'
    spread = float(np.median(np.abs(sample - np.median(sample))) * 1.4826)
    threshold = max(sigmas * spread, floor)
    low = max(2.5 * spread, floor / 2)
    peak = np.where(inside, signed, -np.inf)
    if not np.isfinite(peak).any() or peak.max() <= threshold:
        return None, None, 'no_reflection_found'
    seed = np.unravel_index(int(np.argmax(peak)), peak.shape)
    rows, cols = np.mgrid[:inside.shape[0], :inside.shape[1]]
    near = np.hypot(cols - seed[1], rows - seed[0]) <= reach
    kernel = np.ones((3, 3), np.uint8)
    core = inside & near & (signed > threshold)
    if not core.any():
        return None, None, 'seed_outside_candidates'
    for _ in range(6):
        grown = (cv2.dilate(core.astype(np.uint8), kernel) > 0) & inside & near & (signed > low)
        if grown.sum() == core.sum():
            break
        core = grown
    if grow:
        k = np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)
        core = (cv2.dilate(core.astype(np.uint8), k) > 0) & inside
    return core, dict(threshold=threshold, low=low, spread=spread), None


def to_atlas(mask, p, q, du, dv, scale):
    """Scatter a native-resolution mask into atlas cells, for temporal smoothing."""
    out = np.zeros((atl.ATLAS_N, atl.ATLAS_N), np.float32)
    if not mask.any():
        return out
    ai = np.rint(((p[mask] - du) / scale) / atl.ATLAS_STEP + (atl.ATLAS_N - 1) / 2).astype(int)
    aj = np.rint(((q[mask] - dv) / scale) / atl.ATLAS_STEP + (atl.ATLAS_N - 1) / 2).astype(int)
    ok = (ai >= 0) & (ai < atl.ATLAS_N) & (aj >= 0) & (aj < atl.ATLAS_N)
    out[aj[ok], ai[ok]] = 1.0
    # The scatter is sparser than the atlas grid (about 0.27 native pixels per cell), so
    # close it before it is used as a temporal vote.
    return (cv2.dilate(out, np.ones((5, 5), np.uint8)) > 0).astype(np.float32)


def feather(mask, radius=2.0):
    """A ramp at the mask edge, so the repair has no hard seam.

    Radius 1 is a no-op: `distanceTransform` gives an edge pixel distance 1, so
    clipping at 1 returns a full-weight hard mask. The ramp needs radius > 1 to exist.
    """
    if not mask.any():
        return mask.astype(np.float32)
    d = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    return np.clip(d / max(radius, 1e-6), 0, 1).astype(np.float32)


def eye_tile(source, repaired, weight, points, topology, zoom, source_frame, eyes, half=17):
    """Both eyes, source above repair, at eye scale. The r29 contact-sheet standard."""
    cols = []
    for side, rec in zip(r32.SIDES, eyes):
        centre, _, _ = r32.limbus_ellipse(points, topology['iris_' + side])
        cx, cy = int(round(centre[0])), int(round(centre[1]))
        y0, y1 = max(0, cy - half), min(source.shape[0], cy + half)
        x0, x1 = max(0, cx - half), min(source.shape[1], cx + half)
        s = r32.rgb8(source)[y0:y1, x0:x1, ::-1].copy()
        r = r32.rgb8(repaired)[y0:y1, x0:x1, ::-1].copy()
        edge = cv2.dilate((weight[y0:y1, x0:x1] > 0).astype(np.uint8), np.ones((3, 3), np.uint8))
        edge = (edge > 0) & ~(weight[y0:y1, x0:x1] > 0)
        s[edge] = (0, 200, 255)
        pair = np.concatenate([s, r], axis=0)
        cols.append(cv2.resize(pair, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST))
    h = max(c.shape[0] for c in cols)
    cols = [np.pad(c, ((0, h - c.shape[0]), (0, 4), (0, 0))) for c in cols]
    tile = np.concatenate(cols, axis=1)
    label = np.zeros((22, tile.shape[1], 3), np.uint8)
    text = '%d %s' % (source_frame, ''.join('.' if e.get('repaired') else 'x' for e in eyes))
    cv2.putText(label, text, (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1,
                cv2.LINE_AA)
    return np.concatenate([label, tile], axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_shot_argument(ap)
    ap.add_argument('--name', default='r33_repair')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--count', type=int, default=150)
    ap.add_argument('--iterations', type=int, default=5)
    ap.add_argument('--gain-window', type=int, default=5)
    ap.add_argument('--mask-window', type=int, default=2,
                    help='half-width, in frames, of the temporal vote on the mask')
    ap.add_argument('--min-ncc', type=float, default=0.5,
                    help='abstain when the frame does not register to the atlas this well')
    ap.add_argument('--mask-votes', type=int, default=3,
                    help='frames in that window a cell must be flagged in to be repaired')
    ap.add_argument('--sheet-zoom', type=int, default=4)
    ap.add_argument('--sheet-frames', type=int, default=25)
    args = ap.parse_args()
    shot = from_args(args)
    if Path(args.name).name != args.name:
        raise ValueError('Expected a directory name, not a path')
    root = shot.composite / args.name
    for sub in ('source', 'repair', 'mask', 'sheets'):
        (root / sub).mkdir(parents=True, exist_ok=True)

    meta = json.loads((shot.relight / 'face_landmarks.json').read_text())
    topology = meta['topology']
    with np.load(shot.relight / 'face_landmarks.npz') as z:
        landmarks = {k: z[k] for k in ('points', 'found', 'eyelid_aperture')}
    indices = list(range(args.start, args.start + args.count))
    if min(indices) < 0 or max(indices) >= shot.frame_count:
        raise ValueError('Requested interval leaves the shot')

    u, v = atl.atlas_grid()
    sources, geoms = {}, {}
    collected = {s: [] for s in r32.SIDES}
    for i in indices:
        raw = cv2.imread(str(shot.fg(i)), cv2.IMREAD_UNCHANGED)
        if raw is None or raw.dtype != np.float32 or raw.ndim != 3:
            raise ValueError('Expected scene-linear keyed RGB EXR at %s' % shot.fg(i))
        sources[i] = raw[..., ::-1].astype(np.float32)
        if not (landmarks['found'][i] and np.isfinite(landmarks['points'][i]).all()):
            continue
        for side in r32.SIDES:
            geom, record = r32.eye_frame(sources[i], landmarks['points'][i], topology, side)
            if geom is None:
                continue
            patch = sources[i][geom['window']]
            state = r32.solve_eye(patch, geom)
            if state is None:
                continue
            anomaly, model, reason = r32.detect_anomaly(patch, state['rho'], state['theta'],
                                                        state['inside'])
            if anomaly is None:
                continue
            collar = cv2.dilate(anomaly.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            rgb, w = atl.sample_eye(patch, geom, state['inside'] & ~collar, u, v)
            geoms[(i, side)] = geom
            collected[side].append(dict(frame=i, rgb=rgb, w=w,
                                        luma=rgb.max(-1).astype(np.float32),
                                        du=0.0, dv=0.0, scale=1.0, ncc=0.0,
                                        ax=geom['ax'], ay=geom['ay']))
        print('R33_COLLECT frame', i, flush=True)

    # --- one atlas per eye for the whole shot -----------------------------------------
    fitted = {}
    for side in r32.SIDES:
        frames = collected[side]
        if len(frames) < 20:
            fitted[side] = None
            continue
        atlas, aw, history, _, rho, seed_sharp = atl.build(frames, args.iterations,
                                                           args.gain_window)
        # The FILL region reaches past RHO_MAX -- see FILL_REACH. Interpolate the atlas
        # across cells the sweep never uncovered, once, on the pooled atlas, so nothing
        # it introduces can vary between frames.
        core = rho < atl.FILL_REACH
        holes_before = int((core & ~aw).sum())
        atlas, aw = atl.fill_atlas_holes(atlas, aw, core)
        fitted[side] = dict(atlas=atlas, aw=aw, history=history,
                            atlas_holes_interpolated=holes_before - int((core & ~aw).sum()),
                            atlas_holes_remaining=int((core & ~aw).sum()),
                            seed_sharpness=seed_sharp,
                            sharpness=atl.sharpness(atlas, aw & (rho < r32.RHO_MAX)),
                            by_frame={f['frame']: f for f in frames})
        print('R33_ATLAS_READY', side, len(frames), 'frames', flush=True)

    # --- pass A: detect against the fixed prediction, in every frame ------------------
    raw_masks = {s: {} for s in r32.SIDES}
    native_masks = {s: {} for s in r32.SIDES}
    detail = {s: {} for s in r32.SIDES}
    for side in r32.SIDES:
        fit = fitted[side]
        if fit is None:
            continue
        for i, f in fit['by_frame'].items():
            geom = geoms[(i, side)]
            patch = sources[i][geom['window']]
            state = r32.solve_eye(patch, geom)
            base, support, (p, q) = predict(geom, fit['atlas'], fit['aw'],
                                            f['du'], f['dv'], f['scale'], f['gain_coef'])
            # Detect over the whole visible iris. Gating this on the atlas's support
            # mask was a bug that cost most of a pass: the highlight dwells in roughly
            # one place for many consecutive frames, so its own cells are exactly the
            # ones with the fewest clean observations, and only 7 of 85 footprint pixels
            # survived the gate. The detector could not see the thing it looks for.
            # Support is a constraint on what may be FILLED, not on what may be found.
            mask, info, reason = detect(patch, base, state['inside'])
            detail[side][i] = dict(reason=reason, info=info, base=base, support=support,
                                   inside=state['inside'], p=p, q=q, geom=geom)
            native_masks[side][i] = mask
            raw_masks[side][i] = (to_atlas(mask, p, q, f['du'], f['dv'], f['scale'])
                                  if mask is not None
                                  else np.zeros((atl.ATLAS_N, atl.ATLAS_N), np.float32))

    # --- pass B: temporal vote on the mask, in the eye's own coordinates --------------
    voted = {s: {} for s in r32.SIDES}
    for side in r32.SIDES:
        keys = sorted(raw_masks[side])
        for k, i in enumerate(keys):
            window = [raw_masks[side][keys[j]] for j in
                      range(max(0, k - args.mask_window), min(len(keys), k + args.mask_window + 1))]
            votes = np.sum(window, axis=0)
            need = min(args.mask_votes, len(window))
            voted[side][i] = votes >= need

    # --- pass C: repair ---------------------------------------------------------------
    rows, sheet_tiles = [], []
    for i in indices:
        values = sources[i]
        out = values.copy()
        combined = np.zeros(values.shape[:2], np.float32)
        eyes = []
        for side in r32.SIDES:
            fit = fitted[side]
            d = detail[side].get(i) if fit is not None else None
            if d is None:
                eyes.append(dict(side=side, repaired=False, reason='no_eye_frame'))
                continue
            geom = d['geom']
            f = fit['by_frame'][i]
            record = dict(side=side, limbus_centre=[round(float(c), 3) for c in geom['limbus']],
                          limbus_axes=[round(geom['ax'], 3), round(geom['ay'], 3)],
                          registration_shift_px=[round(float(f['du'] * geom['ax']), 4),
                                                 round(float(f['dv'] * geom['ay']), 4)],
                          registration_ncc=round(float(f['ncc']), 5))
            if d['reason'] is not None:
                record.update(repaired=False, reason=d['reason'])
                eyes.append(record)
                continue
            # No reliable registration means no reliable prediction. At the blinks the
            # visible iris is a sliver, the correlation collapses to 0.0-0.08, and the
            # atlas lands in the wrong place -- which made the fill BRIGHTER than the
            # source it was meant to darken, a bright blob mid-blink. Abstain instead.
            if not f['registered'] or f['ncc'] < args.min_ncc:
                record.update(repaired=False, reason='registration_below_threshold')
                eyes.append(record)
                continue
            # Project the temporally voted mask back to native pixels.
            ai = ((d['p'] - f['du']) / f['scale'] / atl.ATLAS_STEP
                  + (atl.ATLAS_N - 1) / 2).astype(np.float32)
            aj = ((d['q'] - f['dv']) / f['scale'] / atl.ATLAS_STEP
                  + (atl.ATLAS_N - 1) / 2).astype(np.float32)
            back = cv2.remap(voted[side][i].astype(np.float32), ai, aj, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT)
            # UNION with this frame's own detection, never intersection. A 3-of-5
            # temporal vote on its own erodes the mask, and an eroded mask leaves the
            # core of the highlight behind as a bright speck inside the pupil -- which
            # is the artefact round 31 was rejected for. The vote may only ADD
            # neighbouring frames' footprint, to stop the shape breathing; it may not
            # take away what this frame actually shows.
            own = native_masks[side].get(i)
            found = ((back > 0.5) | (own if own is not None else False)) & d['inside']
            # Only pixels the atlas actually predicts may be replaced; the rest keep
            # exact source, and the count is reported rather than hidden.
            mask = found & d['support']
            record['unsupported_pixels'] = int((found & ~d['support']).sum())
            if mask.sum() < 3:
                record.update(repaired=False, reason='no_supported_pixels',
                              detected_pixels=int(found.sum()))
                eyes.append(record)
                continue
            limit = min(3.4 * geom['ax'] ** 2 / 4 * np.pi / 4, 0.55 * d['inside'].sum())
            if mask.sum() > limit:
                record.update(repaired=False, reason='footprint_implausibly_large',
                              footprint_pixels=int(mask.sum()))
                eyes.append(record)
                continue
            # Fill, then look for what the fill did not cover, extend the mask over it
            # and fill again. A single pass leaves the highlight's sub-threshold tail
            # behind as a light speck at the footprint edge -- the artefact round 31 was
            # rejected for -- because the detector's threshold is set by the eye's own
            # spread and the tail sits just under it. Growth is confined to a 3-pixel
            # collar of the existing mask, so this closes the footprint rather than
            # walking out into real iris texture.
            patch = out[geom['window']].copy()
            passes = 0
            for passes in range(1, 4):
                w = feather(mask) * mask
                blended = patch * (1 - w[..., None]) + d['base'] * w[..., None]
                near = cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
                after = blended.max(axis=-1) - d['base'].max(axis=-1)
                leftover = d['inside'] & d['support'] & near & (after > d['info']['threshold'])
                if not leftover.any():
                    break
                grown = (mask | leftover) & d['inside'] & d['support']
                if grown.sum() == mask.sum() or grown.sum() > limit:
                    break
                mask = grown
            # A repair of a highlight must darken it. If the prediction is brighter
            # than the source over the footprint, the atlas is in the wrong place and
            # filling would ADD light to the eye, so keep exact source and say so.
            if float(d['base'][mask].max()) >= float(patch[mask].max()):
                record.update(repaired=False, reason='prediction_not_darker_than_source',
                              footprint_pixels=int(mask.sum()))
                eyes.append(record)
                continue
            w = feather(mask) * mask
            blended = patch * (1 - w[..., None]) + d['base'] * w[..., None]
            out[geom['window']] = np.maximum(blended, 0.0)
            reg = geom['window']
            combined[reg] = np.maximum(combined[reg], w)
            fill = d['base'][mask]
            # Explicit check that the highlight is actually gone: how much of the eye
            # is still above the detection threshold after the fill, and how bright.
            # Restricted to the footprint's own neighbourhood. Measured over the whole
            # iris it counts ordinary limbal-ring and texture residual as leftover
            # highlight and reads 16 pixels on every eye, which says nothing.
            near = cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
            after = blended.max(axis=-1) - d['base'].max(axis=-1)
            leftover = d['inside'] & near & (after > d['info']['threshold'])
            record.update(
                fill_passes=passes,
                residual_pixels_above_threshold=int(leftover.sum()),
                residual_peak_display=round(float(linear_to_srgb(np.clip(
                    blended[d['inside'] & near].max(), 0, 1))), 4),
                surround_peak_display=round(float(linear_to_srgb(np.clip(
                    blended[d['inside'] & ~near].max(), 0, 1))), 4),
                repaired=True, reason=None,
                footprint_pixels=int(mask.sum()),
                threshold_linear=round(float(d['info']['threshold']), 6),
                peak_source_display=round(float(linear_to_srgb(patch[mask].max())), 4),
                peak_repaired_display=round(float(linear_to_srgb(
                    np.clip(blended[mask].max(), 0, 1))), 4),
                fill_display=[int(round(float(linear_to_srgb(np.clip(c, 0, 1)) * 255)))
                              for c in fill.mean(axis=0)])
            eyes.append(record)

        untouched = combined == 0
        assert np.isfinite(out).all() and (out >= 0).all()
        assert np.array_equal(out[untouched], values[untouched]), 'source changed outside the mask'
        cv2.imwrite(str(root / 'source' / ('%06d.png' % i)), r32.rgb8(values)[..., ::-1])
        cv2.imwrite(str(root / 'repair' / ('%06d.png' % i)), r32.rgb8(out)[..., ::-1])
        cv2.imwrite(str(root / 'mask' / ('%06d.png' % i)),
                    np.rint(np.clip(combined, 0, 1) * 255).astype(np.uint8))
        if landmarks['found'][i] and np.isfinite(landmarks['points'][i]).all():
            sheet_tiles.append(eye_tile(values, out, combined, landmarks['points'][i],
                                        topology, args.sheet_zoom,
                                        shot.source_start_frame + i, eyes))
        else:
            sheet_tiles.append(np.zeros((22 + 68 * args.sheet_zoom,
                                         2 * 34 * args.sheet_zoom + 8, 3), np.uint8))
        rows.append(dict(frame=i, source_frame=shot.source_start_frame + i,
                         source_sha256=r32.sha(shot.fg(i)),
                         changed_pixels=int((combined > 0).sum()), eyes=eyes))
        print('R33_REPAIR frame', i, 'changed', int((combined > 0).sum()), flush=True)

    sheets = []
    for k in range(0, len(sheet_tiles), args.sheet_frames):
        path = root / 'sheets' / ('contact_%03d.jpg' % indices[k])
        assert cv2.imwrite(str(path), r32.contact_sheet(sheet_tiles[k:k + args.sheet_frames]),
                           [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        sheets.append(path.name)
        print('R33_SHEET', path.name, flush=True)

    repaired = [r for r in rows if r['changed_pixels']]
    record = dict(
        rows=rows, contact_sheets=sheets,
        frames=len(rows), frames_repaired=len(repaired),
        frames_exact_source=len(rows) - len(repaired),
        total_changed_pixels=int(sum(r['changed_pixels'] for r in rows)),
        atlas={s: (None if fitted[s] is None else dict(
            frames=len(fitted[s]['by_frame']), iterations=fitted[s]['history'],
            sharpness=round(fitted[s]['sharpness'], 5),
            unregistered_seed_sharpness=round(fitted[s]['seed_sharpness'], 5),
            holes_interpolated=fitted[s]['atlas_holes_interpolated'],
            holes_remaining=fitted[s]['atlas_holes_remaining']))
            for s in r32.SIDES},
        method='one sequence-wide registered appearance atlas per eye, times a smooth '
               'temporally-smoothed illumination field; mask voted over five frames',
        material_source='observed clean pixels of the same eye, pooled over the shot',
        generated_material_used=False, scene_reflection_added=False,
        dark_rim_repaired=False,
        production_enabled=False, artist_accepted=False,
        source_outside_mask_exact=True,
        mask_window_frames=args.mask_window, mask_votes=args.mask_votes,
        code_sha256=r32.sha(__file__), atlas_code_sha256=r32.sha(atl.__file__),
        track_sha256=r32.sha(shot.relight / 'face_landmarks.npz'))
    r32.write_json(root / 'repair.json', record)
    r32.write_json(shot.log(args.name + '.json'), record)
    print('R33_REPAIR_DONE', len(rows), 'frames,', len(repaired), 'repaired', flush=True)


if __name__ == '__main__':
    main()
