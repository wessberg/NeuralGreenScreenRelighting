"""Experimental native-image iris observations; never a production reliability oracle.

Only near-circular, well exposed views are modelled. A coarse eye ROI/scale is an
input, NOT a semantic mask. Pupil and limbus are fitted from native radial contrast.
Occluded upper sectors are deliberately unavailable. No appearance is generated.
All reads, including boundary fitting, require four non-hidden native contributors.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from temporal_iris import IrisGeometry, footprint_valid


@dataclass
class Observation:
    geometry: IrisGeometry
    material: np.ndarray
    excluded: np.ndarray
    reflection: np.ndarray
    candidate_ok: bool
    reasons: list
    metrics: dict


def sample(im, xy):
    xy = np.asarray(xy, np.float32)
    shape = xy.shape[:-1]
    flat = xy.reshape(-1, 2)
    parts = [cv2.remap(im, p[:, 0:1], p[:, 1:2], cv2.INTER_LINEAR)[:, 0]
             for p in np.array_split(flat, max(1, int(np.ceil(len(flat)/16000))))]
    return np.concatenate(parts).reshape(*shape, *im.shape[2:])


def circle_fit(luma, readable, seed, radius_range, centre_span, angles, pixel_aspect=1.):
    """Search in native pixels; missing RGB never enters objective or selection.

    Six independent angular sectors expose fits supported by only one bright edge.
    Circle is intentional: unreliable off-angle ellipses must not invent material.
    """
    offsets = np.arange(-centre_span, centre_span+.01, max(.5, centre_span/8))
    yy, xx, rr = np.meshgrid(offsets+seed[1], offsets/pixel_aspect+seed[0], radius_range, indexing='ij')
    candidates = np.column_stack([xx.ravel(), yy.ravel(), rr.ravel()]).astype(np.float32)
    direction = np.column_stack([np.cos(angles)/pixel_aspect, np.sin(angles)])
    xy = candidates[:, None, :2] + candidates[:, None, 2:]*direction
    a, b = xy-direction, xy+direction
    av = footprint_valid(readable, a.reshape(-1, 2)).reshape(a.shape[:2])
    bv = footprint_valid(readable, b.reshape(-1, 2)).reshape(b.shape[:2])
    diff = sample(luma, b)-sample(luma, a)
    diff[~(av & bv)] = np.nan
    sectors = []
    for ids in np.array_split(np.arange(len(angles)), 6):
        v = diff[:, ids]
        good = np.isfinite(v).sum(1) >= max(3, len(ids)//2)
        score = np.nansum(v, 1)/np.maximum(np.isfinite(v).sum(1), 1)
        sectors.append(np.where(good, score, -1000))
    sectors = np.stack(sectors, 1)
    # Lower quartile requires multiple arcs, not just a sclera or softbox edge.
    objective = np.percentile(sectors, 25, axis=1)
    best = int(np.argmax(objective))
    return candidates[best], sectors[best], float(objective[best])


def detect(rgb, seed, radius, hidden=None, *, pixel_aspect=1., eye_rotation=0.):
    """RGB display levels [0,255]; results are candidates requiring real validation.

    hidden is scrubbed BEFORE every statistic and fitting read. Seed/ROI are held
    fixed externally; this does not establish upstream face-tracker independence.
    """
    rgb = np.asarray(rgb, np.float32).copy()
    hidden = np.zeros(rgb.shape[:2], bool) if hidden is None else np.asarray(hidden, bool)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or hidden.shape != rgb.shape[:2]:
        raise ValueError('Expected native RGB and same-size hidden mask')
    if not np.isfinite([*seed, radius, pixel_aspect, eye_rotation]).all() or radius < 2 or pixel_aspect <= 0:
        raise ValueError('Invalid native eye seed, radius or pixel aspect')
    readable = ~hidden & np.isfinite(rgb).all(-1)
    rgb[~readable] = 0
    lum = rgb @ np.array([.2126, .7152, .0722], np.float32)
    # Only lateral/lower limbus: upper boundary is frequently covered by lashes.
    angles = np.linspace(0, np.pi, 48)+eye_rotation
    step = max(.5, radius/50)
    limbus, ls, lscore = circle_fit(lum, readable, seed,
        np.arange(radius*.7, radius*1.26, step), max(3, radius*.2), angles, pixel_aspect)
    yy, xx = np.mgrid[:lum.shape[0], :lum.shape[1]]
    dx, dy = (xx-limbus[0])*pixel_aspect, yy-limbus[1]
    preliminary_radius = np.hypot(dx, dy)
    lower_ring = (preliminary_radius > limbus[2]*.65) & (preliminary_radius < limbus[2]*.85)
    lower_ring &= (-np.sin(eye_rotation)*dx+np.cos(eye_rotation)*dy > 2) & readable
    baseline = lum[lower_ring]
    threshold = max(float(np.median(baseline)*1.8), float(np.percentile(baseline, 90))) if len(baseline) else 255
    preliminary_bright = (preliminary_radius < limbus[2]*.85) & (lum > threshold) & readable
    preliminary_exclusion = cv2.dilate(preliminary_bright.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    pupil_radii = np.arange(limbus[2]*.2, limbus[2]*.71, step)
    pupil, ps, pscore = circle_fit(lum, readable & ~preliminary_exclusion, limbus[:2],
        pupil_radii, max(2, radius*.12), angles, pixel_aspect)
    concentric_error = float(np.linalg.norm((pupil[:2]-limbus[:2])*[pixel_aspect,1]))
    g = IrisGeometry(tuple(limbus[:2].astype(float)), (float(limbus[2]/pixel_aspect), float(limbus[2])),
                     0., float(pupil[2]/limbus[2]), eye_rotation)
    delta = np.stack([xx, yy], -1)-limbus[:2]
    delta[..., 0] *= pixel_aspect
    rr = np.linalg.norm(delta, axis=-1)
    theta = np.arctan2(np.sin(np.arctan2(delta[..., 1], delta[..., 0])-eye_rotation),
                       np.cos(np.arctan2(delta[..., 1], delta[..., 0])-eye_rotation))
    # Conservative material INTERIOR with a native-pixel margin at both boundaries.
    # Never claim the old landmark aperture, upper iris or eyelids as clean material.
    interior = (rr > pupil[2]+1.5) & (rr < limbus[2]-1.5)
    lower = (theta > .15) & (theta < np.pi-.15)
    material = interior & lower & readable
    baseline = lum[material]
    cutoff = max(float(np.median(baseline)*1.8), float(np.percentile(baseline, 90))) if len(baseline) else 255
    disc = rr <= limbus[2]-1
    bright = disc & readable & (lum > cutoff)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(bright.astype(np.uint8))
    reflection = np.zeros(lum.shape, bool)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= 2:
            reflection |= labels == i
    excluded = cv2.dilate(reflection.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    material &= ~excluded
    reasons = []
    if radius < 4:
        reasons.append('iris_below_native_resolution_floor')
    if lscore < 10 or np.count_nonzero(ls > 8) < 5:
        reasons.append('insufficient_limbus_arcs')
    if pscore < 3 or np.count_nonzero(ps > 2) < 5:
        reasons.append('insufficient_pupil_arcs')
    if concentric_error > 1:
        reasons.append('nonconcentric_or_unresolved_pupil')
    if pupil[2] <= pupil_radii[0]+step+.01 or pupil[2] >= pupil_radii[-1]-.01:
        reasons.append('pupil_radius_at_search_boundary')
    if material.sum() < 30:
        reasons.append('insufficient_visible_material')
    pupil_crossing = bool((excluded & (rr <= pupil[2]+1.5)).any())
    return Observation(g, material, excluded, reflection, not reasons, reasons,
        {'limbus': limbus.tolist(), 'pupil': pupil.tolist(),
         'limbus_sector_contrast': ls.tolist(), 'pupil_sector_contrast': ps.tolist(),
         'limbus_score': lscore, 'pupil_score': pscore,
         'concentric_error_px': concentric_error,
         'material_pixels': int(material.sum()), 'reflection_pixels': int(reflection.sum()),
         'reflection_crosses_pupil': pupil_crossing,
         'reflection_outside_visible_material': bool((excluded & ~interior).any())})
