"""Deterministic temporal iris sampling, independent of shot, colour and tracker.

Geometry describes projected concentric pupil/limbus ellipses. Iris material uses
angle and fractional distance between those boundaries (a linear rubber sheet).
This approximation does not model corneal refraction or nonlinear iris deformation.
The caller MUST provide validated geometry, material visibility and reflection
exclusions. This module cannot certify those inputs, or infer unseen texture.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class IrisGeometry:
    centre: tuple[float, float]
    axes: tuple[float, float]  # limbus SEMI axes in native pixels
    rotation: float  # radians in image coordinates, positive toward image-down
    pupil_ratio: float
    texture_rotation: float = 0.0  # material torsion relative to ellipse

    def __post_init__(self):
        values = (*self.centre, *self.axes, self.rotation, self.pupil_ratio, self.texture_rotation)
        if not np.isfinite(values).all() or min(self.axes) <= 0 or not 0 < self.pupil_ratio < 1:
            raise ValueError('Invalid pupil/limbus geometry')

    def material(self, xy):
        """Native xy -> iris radial fraction, material angle. Pupil is NOT iris."""
        c, s = np.cos(self.rotation), np.sin(self.rotation)
        local = (np.asarray(xy) - self.centre) @ np.array([[c, -s], [s, c]])
        local = local / self.axes
        radius = np.linalg.norm(local, axis=-1)
        angle = np.arctan2(local[..., 1], local[..., 0]) - self.texture_rotation
        return np.stack([(radius-self.pupil_ratio)/(1-self.pupil_ratio), angle], -1)

    def native(self, uv):
        """Iris material coordinates -> native xy, with dilation and torsion."""
        uv = np.asarray(uv)
        radius = self.pupil_ratio + uv[..., 0]*(1-self.pupil_ratio)
        angle = uv[..., 1] + self.texture_rotation
        local = np.stack([radius*np.cos(angle)*self.axes[0],
                          radius*np.sin(angle)*self.axes[1]], -1)
        c, s = np.cos(self.rotation), np.sin(self.rotation)
        return local @ np.array([[c, s], [-s, c]]) + self.centre


def footprint_valid(mask, xy):
    """Require all four native bilinear contributors, including zero-weight neighbours."""
    mask = np.asarray(mask, bool)
    if mask.ndim != 2 or min(mask.shape) < 2:
        raise ValueError('Native visibility mask must be a 2D image')
    finite = np.isfinite(xy).all(-1)
    safe = np.where(finite[..., None], xy, -10)
    x, y = np.floor(safe).astype(np.int64).T
    inside = finite & (x >= 0) & (y >= 0) & (x < mask.shape[1]-1) & (y < mask.shape[0]-1)
    x, y = np.clip(x, 0, mask.shape[1]-2), np.clip(y, 0, mask.shape[0]-2)
    return inside & mask[y, x] & mask[y+1, x] & mask[y, x+1] & mask[y+1, x+1]


def reconstruct(images, geometries, usable_masks, frame_ids, target_index, target_xy,
                *, max_mad, min_donors=3, temporal_guard=3, donor_spacing=4):
    """Sample each material point from every eligible frame, retaining provenance.

    RGB and masks from the target and its temporal guard are never read. Confidence
    means donor agreement only; it is NOT proof of correct correspondence or absence
    of source reflection. max_mad is expressed in the caller's RGB units. Adjacent
    selected donors must be at least donor_spacing frame IDs apart.
    """
    n = len(images)
    if not n or not (len(geometries) == len(usable_masks) == len(frame_ids) == n):
        raise ValueError('One geometry, mask and unique frame ID required per image')
    if len(set(frame_ids)) != n or min_donors < 1 or donor_spacing < 1 or temporal_guard < 0 or max_mad < 0:
        raise ValueError('Invalid reconstruction controls')
    if not 0 <= target_index < n:
        raise ValueError('Target outside sequence')
    target_xy = np.asarray(target_xy, np.float32).reshape(-1, 2)
    uv = geometries[target_index].material(target_xy)
    material = np.isfinite(uv).all(-1) & (uv[:, 0] >= 0) & (uv[:, 0] <= 1)
    samples = np.zeros((n, len(uv), 3), np.float32)
    valid = np.zeros((n, len(uv)), bool)
    last = np.full(len(uv), -np.inf)
    for j in np.argsort(frame_ids):
        if abs(frame_ids[j]-frame_ids[target_index]) <= temporal_guard:
            continue
        rgb = np.asarray(images[j], np.float32)
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or usable_masks[j].shape != rgb.shape[:2]:
            raise ValueError('Donor image and native mask shapes disagree')
        xy = geometries[j].native(uv).astype(np.float32)
        ok = material & footprint_valid(usable_masks[j], xy)
        ok &= frame_ids[j]-last >= donor_spacing
        # OpenCV remap has a 32767-pixel dimension limit; chunk query vectors.
        for lo in range(0, len(uv), 16000):
            part = xy[lo:lo+16000]
            samples[j, lo:lo+16000] = cv2.remap(rgb, part[:, 0:1], part[:, 1:2],
                                                cv2.INTER_LINEAR)[:, 0]
        ok &= np.isfinite(samples[j]).all(-1)
        valid[j] = ok
        last[ok] = frame_ids[j]
    data = np.ma.array(samples, mask=np.broadcast_to(~valid[..., None], samples.shape))
    prediction = np.ma.median(data, axis=0).filled(0).astype(np.float32)
    deviation = np.ma.abs(data-prediction).mean(-1)
    mad = np.ma.median(deviation, axis=0).filled(np.inf).astype(np.float32)
    count = valid.sum(0)
    supported = material & (count >= min_donors)
    return {'rgb': prediction, 'supported': supported, 'agreement': supported & (mad <= max_mad),
            'donor_count': count, 'mad': mad, 'donor_used': valid,
            'frame_ids': np.asarray(frame_ids), 'material_uv': uv}


def repair_strength(reliable, ramp_frames=3):
    """Offline, per-eye taper within trusted runs. Unreliable frames stay EXACTLY zero.

    This reduces a hard switch; it does not prove that a changing reflection will be
    imperceptible. Use the same weights for removal and eventual replacement layers.
    """
    reliable = np.asarray(reliable, bool)
    if reliable.ndim != 1 or ramp_frames < 1:
        raise ValueError('Expected a reliability sequence and positive ramp length')
    strength = reliable.astype(np.float32)
    for i in range(len(strength)):
        if not reliable[i]:
            continue
        left, right = i, i
        while left > 0 and i-left < ramp_frames and reliable[left-1]:
            left -= 1
        while right+1 < len(strength) and right-i < ramp_frames and reliable[right+1]:
            right += 1
        strength[i] = min(1.0, (i-left+1)/ramp_frames, (right-i+1)/ramp_frames)
    return strength


def apply_eye_repair(source_keyed, repair_mask, estimate=None, *, geometry_reliable=False,
                     correspondence_reliable=False, strength=1.0):
    """Apply a complete supported eye-region repair or keep the keyed source verbatim.

    Call separately for each eye. estimate query order must be np.where(repair_mask)
    (row major native pixels). Geometry AND material correspondence need independent
    validation by the caller; donor agreement alone never turns either flag on.
    No mask pixel may be repaired when another requested pixel lacks evidence. There
    is no spatial fill, forced removal, or synthetic catchlight on a fallback frame.
    """
    source = np.asarray(source_keyed)
    mask = np.asarray(repair_mask, bool)
    if source.ndim != 3 or source.shape[-1] != 3 or mask.shape != source.shape[:2]:
        raise ValueError('Expected keyed RGB and a matching native repair mask')
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('Repair strength must be between zero and one')
    result = source.copy()
    reason = None
    if not geometry_reliable:
        reason = 'unreliable_iris_detection'
    elif not correspondence_reliable:
        reason = 'unreliable_material_correspondence'
    elif not mask.any():
        reason = 'no_reflection_to_repair'
    elif strength == 0:
        reason = 'temporal_gate'
    elif estimate is None:
        reason = 'no_temporal_evidence'
    else:
        count = int(mask.sum())
        rgb = np.asarray(estimate['rgb'])
        support, agreement = np.asarray(estimate['supported']), np.asarray(estimate['agreement'])
        if rgb.shape != (count, 3) or support.shape != (count,) or agreement.shape != (count,):
            raise ValueError('Estimate must match the native repair-mask query order')
        if not np.isfinite(rgb).all() or not support.all() or not agreement.all():
            reason = 'incomplete_or_inconsistent_donors'
        else:
            result[mask] = source[mask]*(1-strength) + rgb*strength
    return result, {'mode': 'source_keyed' if reason else 'temporal_repair',
                    'reason': reason, 'strength': 0.0 if reason else float(strength),
                    'original_catchlight_kept': bool(reason),
                    'changed_channels': int(np.count_nonzero(result != source))}
