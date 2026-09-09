"""Does the corneal highlight MOVE in the eye's own stabilized frame?

Round 32's transferable finding is that no per-frame-refit fill can be temporally
stable at a 24-pixel iris, and that a next attempt needs the eye estimated ONCE over
the sequence in a stabilized frame. That is only possible if the corrupted footprint
sweeps across the stabilized frame, so that every cell it ever covers is observed
CLEAN in some other frames. If instead the footprint sits still in eye coordinates,
there is no observed material for those cells anywhere in the shot and no
sequence-wide observed-pixel model can exist -- which would be a real result, and the
point at which round 30's authorized selective generation is the only remaining route.

This script measures that and nothing else. It reuses round 32's limbus polar frame
and highlight detector unchanged (`eye_reflection_repair`), resamples every frame's
eye onto ONE fixed polar grid, and reports per-cell clean/corrupt observation counts
plus the footprint's trajectory in eye coordinates.

It writes no production default and repairs nothing.

    dependencies/CorridorKey/.venv/Scripts/python.exe -B scripts/eye_sequence_occupancy.py \
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
from shot_paths import add_shot_argument, from_args
import eye_reflection_repair as r32

# The polar grid. About 11.3 native pixels of radius at RHO_MAX on this plate, so 64
# angular cells is roughly one pixel of arc at the outer edge; inner rings are
# oversampled, which is what pooling over frames is for.
RHO_BINS = 16
THETA_BINS = 64


def cell_index(rho, theta):
    ri = np.clip((rho / r32.RHO_MAX * RHO_BINS).astype(int), 0, RHO_BINS - 1)
    ti = np.clip(((theta + np.pi) / (2 * np.pi) * THETA_BINS).astype(int), 0, THETA_BINS - 1)
    return ri * THETA_BINS + ti


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_shot_argument(ap)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--count', type=int, default=150)
    ap.add_argument('--name', default='r33_occupancy')
    args = ap.parse_args()
    shot = from_args(args)

    meta = json.loads((shot.relight / 'face_landmarks.json').read_text())
    topology = meta['topology']
    with np.load(shot.relight / 'face_landmarks.npz') as z:
        landmarks = {k: z[k] for k in ('points', 'found', 'eyelid_aperture')}
    indices = list(range(args.start, args.start + args.count))
    if min(indices) < 0 or max(indices) >= shot.frame_count:
        raise ValueError('Requested interval leaves the shot')

    ncell = RHO_BINS * THETA_BINS
    clean = {s: np.zeros(ncell, int) for s in r32.SIDES}
    corrupt = {s: np.zeros(ncell, int) for s in r32.SIDES}
    # Pooled clean radiance per cell, for the stationarity question. Sum and sum of
    # squares per channel, so a spread comes out without holding every sample.
    csum = {s: np.zeros((ncell, 3)) for s in r32.SIDES}
    csq = {s: np.zeros((ncell, 3)) for s in r32.SIDES}
    tracks = {s: [] for s in r32.SIDES}

    for i in indices:
        path = shot.fg(i)
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None or raw.dtype != np.float32 or raw.ndim != 3:
            raise ValueError('Expected scene-linear keyed RGB EXR at %s' % path)
        values = raw[..., ::-1].astype(np.float32)
        if not (landmarks['found'][i] and np.isfinite(landmarks['points'][i]).all()):
            for s in r32.SIDES:
                tracks[s].append(dict(frame=i, reason='no_face_track'))
            continue
        for side in r32.SIDES:
            geom, record = r32.eye_frame(values, landmarks['points'][i], topology, side)
            if geom is None:
                tracks[side].append(dict(frame=i, reason=record.get('reason')))
                continue
            patch = values[geom['window']]
            state = r32.solve_eye(patch, geom)
            if state is None:
                tracks[side].append(dict(frame=i, reason='no_eye_frame'))
                continue
            rho, theta, inside = state['rho'], state['theta'], state['inside']
            anomaly, model, reason = r32.detect_anomaly(patch, rho, theta, inside)
            if anomaly is None:
                tracks[side].append(dict(frame=i, reason=reason))
                continue
            # The two-pixel collar carries the sub-threshold tail of the same
            # highlight; it is neither clean material nor part of the footprint.
            collar = cv2.dilate(anomaly.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            good = inside & ~collar
            idx = cell_index(rho, theta)
            np.add.at(corrupt[side], idx[anomaly], 1)
            np.add.at(clean[side], idx[good], 1)
            np.add.at(csum[side], idx[good], patch[good].astype(np.float64))
            np.add.at(csq[side], idx[good], patch[good].astype(np.float64) ** 2)
            fr, ft = rho[anomaly], theta[anomaly]
            # Angular centroid on the circle, so the +/- pi seam does not average to 0.
            tracks[side].append(dict(
                frame=i, source_frame=shot.source_start_frame + i,
                pixels=int(anomaly.sum()),
                rho=round(float(fr.mean()), 4),
                theta=round(float(np.arctan2(np.sin(ft).mean(), np.cos(ft).mean())), 4),
                rho_span=[round(float(fr.min()), 4), round(float(fr.max()), 4)],
                limbus_axes=[round(geom['ax'], 3), round(geom['ay'], 3)],
                limbus_centre=[round(float(geom['limbus'][0]), 3),
                               round(float(geom['limbus'][1]), 3)]))
        print('R33_OCCUPANCY frame', i, flush=True)

    report = dict(rho_bins=RHO_BINS, theta_bins=THETA_BINS, rho_max=r32.RHO_MAX,
                  frames=len(indices), tracks=tracks, sides={})
    for side in r32.SIDES:
        ever = corrupt[side] > 0
        n = int(ever.sum())
        cl = clean[side][ever]
        rows = [t for t in tracks[side] if 'pixels' in t]
        rho = np.array([t['rho'] for t in rows])
        th = np.array([t['theta'] for t in rows])
        # Unwrap theta about its circular mean before quoting a spread.
        mu = np.arctan2(np.sin(th).mean(), np.cos(th).mean())
        dth = np.angle(np.exp(1j * (th - mu)))
        mean = np.divide(csum[side], np.maximum(clean[side], 1)[:, None])
        var = np.maximum(csq[side] / np.maximum(clean[side], 1)[:, None] - mean ** 2, 0)
        solid = clean[side] >= 8
        report['sides'][side] = dict(
            detected_frames=len(rows),
            cells_ever_corrupt=n,
            cells_corrupt_with_zero_clean=int((cl == 0).sum()),
            clean_count_in_corrupt_cells=dict(
                min=int(cl.min()) if n else None,
                p5=float(np.percentile(cl, 5)) if n else None,
                median=float(np.median(cl)) if n else None,
                max=int(cl.max()) if n else None),
            corrupt_count=dict(median=float(np.median(corrupt[side][ever])) if n else None,
                               max=int(corrupt[side].max())),
            footprint_track=dict(
                rho_mean=round(float(rho.mean()), 4),
                rho_range=[round(float(rho.min()), 4), round(float(rho.max()), 4)],
                rho_p5_p95=[round(float(np.percentile(rho, 5)), 4),
                            round(float(np.percentile(rho, 95)), 4)],
                theta_mean_rad=round(float(mu), 4),
                theta_spread_rad=[round(float(dth.min()), 4), round(float(dth.max()), 4)],
                theta_p5_p95_rad=[round(float(np.percentile(dth, 5)), 4),
                                  round(float(np.percentile(dth, 95)), 4)],
                theta_std_rad=round(float(dth.std()), 4),
                arc_travel_px=round(float(np.ptp(dth) * rho.mean() * np.median(
                    [t['limbus_axes'][0] for t in rows])), 3),
                pixels=dict(median=float(np.median([t['pixels'] for t in rows])),
                            max=int(max(t['pixels'] for t in rows)))),
            clean_cell_spread=dict(
                cells_with_8plus_clean=int(solid.sum()),
                relative_sd_median=round(float(np.median(
                    np.sqrt(var[solid]).sum(1) / np.maximum(mean[solid].sum(1), 1e-6))), 4)
                if solid.any() else None))
        print('R33_SIDE', side, json.dumps(report['sides'][side]), flush=True)

    r32.write_json(shot.log(args.name + '.json'), report)
    np.savez_compressed(shot.log(args.name + '.npz'),
                        **{'clean_' + s: clean[s] for s in r32.SIDES},
                        **{'corrupt_' + s: corrupt[s] for s in r32.SIDES},
                        **{'sum_' + s: csum[s] for s in r32.SIDES},
                        **{'sq_' + s: csq[s] for s in r32.SIDES})
    print('R33_OCCUPANCY_DONE', flush=True)


if __name__ == '__main__':
    main()
