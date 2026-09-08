"""Experimental neural-proposal correspondence; never a clean-material certificate.

Native display RGB in 0..255. The caller supplies same-eye rows, pixel aspect,
exclusions and fixed crop tracking. Geometry/visibility are model proposals only.
No generated RGB is used: predictions sample original donor pixels. Reflections
inside candidate masks remain a blocker until independently excluded/validated.
"""
from dataclasses import dataclass, replace, asdict

import cv2
import numpy as np

from iris_correspondence import disjoint_footprints
from temporal_iris import IrisGeometry, footprint_valid, reconstruct


@dataclass
class Proposal:
    geometry: IrisGeometry | None
    material: np.ndarray
    visible_labels: np.ndarray
    complete_labels: np.ndarray
    candidate_ok: bool
    reasons: list
    metrics: dict


def ellipse_geometry(prediction):
    """Recover native ellipse semi-axes; preserve a deterministic image-axis sign."""
    limbus = prediction['ellipses']['limbus']
    pupil = prediction['ellipses']['pupil']
    centre = np.asarray(limbus['center_xy'])
    points = limbus['boundary_xy'] - centre
    _, singular, vt = np.linalg.svd(points, full_matrices=False)
    axes = singular * np.sqrt(2 / len(points))
    direction = vt[0] if vt[0, 0] >= 0 else -vt[0]
    rotation = float(np.arctan2(direction[1], direction[0]))
    # Ratio in the normalized limbus plane; center displacement is gated below.
    local = (pupil['boundary_xy'] - pupil['center_xy']) @ vt.T / axes
    ratio = float(np.linalg.norm(local, axis=-1).mean())
    offset = float(np.linalg.norm(pupil['center_xy'] - centre))
    g = IrisGeometry(tuple(centre), tuple(axes), rotation, ratio)
    return g, offset


def observe(rgb, ellseg, ritnet, *, excluded=None, pixel_aspect=1., track_found=True):
    rgb = np.asarray(rgb, np.float32)
    hidden = np.zeros(rgb.shape[:2], bool) if excluded is None else np.asarray(excluded, bool)
    if hidden.shape != rgb.shape[:2]:
        raise ValueError('Native exclusion shape mismatch')
    pred = ellseg.predict(rgb, excluded=hidden, pixel_aspect=pixel_aspect)
    visible = ritnet.predict(rgb, excluded=hidden, pixel_aspect=pixel_aspect)
    reasons, metrics, g = [], {}, None
    candidate = np.zeros(hidden.shape, bool)
    try:
        g, offset = ellipse_geometry(pred)
        yy, xx = np.mgrid[:hidden.shape[0], :hidden.shape[1]]
        uv = g.material(np.column_stack([xx.ravel(), yy.ravel()]))
        annulus = ((uv[:, 0] >= 0) & (uv[:, 0] <= 1)).reshape(hidden.shape)
        candidate = (visible == 2) & (pred['labels'] == 1) & annulus & ~hidden
        candidate = cv2.erode(candidate.astype(np.uint8), np.ones((3,3), np.uint8),
                              borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        metrics.update(pupil_offset_px=offset, pupil_ratio=g.pupil_ratio,
                       candidate_pixels=int(candidate.sum()), pupil_pixels=int((visible == 3).sum()))
        if offset > 2:
            reasons.append('nonconcentric_pupil')
        if not .15 <= g.pupil_ratio <= .75:
            reasons.append('pupil_ratio')
        if candidate.sum() < 40:
            reasons.append('insufficient_visible_iris')
        if (visible == 3).sum() < 12:
            reasons.append('insufficient_visible_pupil')
    except (KeyError, ValueError, np.linalg.LinAlgError):
        reasons.append('invalid_neural_geometry')
    if not track_found:
        reasons.append('missing_face_track')
    return Proposal(g, candidate, visible, pred['labels'], not reasons, reasons, metrics)


def sample(rgb, xy):
    xy = np.asarray(xy, np.float32)
    return cv2.remap(np.asarray(rgb, np.float32), xy[:, 0:1], xy[:, 1:2], cv2.INTER_LINEAR)[:, 0]


def fit_donor(target_rgb, target, donor_rgb, donor, query_xy, *, rotation=False):
    if not target.candidate_ok or not donor.candidate_ok:
        return None
    yy, xx = np.where(target.material)
    xy = np.column_stack([xx, yy]).astype(np.float32)
    # Query context is excluded in target native pixels, independently of inference.
    context = disjoint_footprints(xy, query_xy)
    xy, yy, xx = xy[context], yy[context], xx[context]
    uv, query_uv = target.geometry.material(xy), target.geometry.material(query_xy)
    values = target_rgb[yy, xx]
    candidates, plans = [], []
    for angle in (np.linspace(-.18, .18, 19) if rotation else [0.]):
        g = replace(donor.geometry, texture_rotation=float(angle))
        dxy, qxy = g.native(uv), g.native(query_uv)
        wrong = replace(g, texture_rotation=g.texture_rotation+.35).native(uv)
        plans.append((g, dxy, qxy, wrong))
    # Exclude unions across the entire search. Per-hypothesis exclusions would
    # let another torsion hypothesis read query/validation contributors during
    # selection, even if the winning hypothesis looked locally disjoint.
    query_union = np.unique(np.floor(np.concatenate([p[2] for p in plans])), axis=0)
    validation_union = []
    prepared = []
    for g, dxy, qxy, wrong in plans:
        usable = footprint_valid(donor.material, dxy) & disjoint_footprints(dxy, query_union)
        validation = (uv[:, 1] > 1.9) & usable
        training = (uv[:, 1] < 1.25) & usable
        common = validation & footprint_valid(donor.material, wrong) & disjoint_footprints(wrong, query_union)
        validation_union.extend([dxy[validation], wrong[common]])
        prepared.append((g, dxy, qxy, wrong, training, validation, common))
    validation_union = np.unique(np.floor(np.concatenate(validation_union)), axis=0)
    for g, dxy, qxy, wrong, training, validation, common in prepared:
        training &= disjoint_footprints(dxy, validation_union)
        if training.sum() < 15 or validation.sum() < 10:
            continue
        error = np.abs(sample(donor_rgb, dxy)-values).mean(-1)
        negative_error = np.abs(sample(donor_rgb, wrong)-values).mean(-1)
        train_error = float(error[training].mean())
        validation_error = float(error[validation].mean())
        margin = float((negative_error[common]-error[common]).mean()) if common.sum() >= 8 else None
        # Retain the actual footprints for independent verification, not only counts.
        candidates.append(dict(geometry=g, training_mae=train_error,
            validation_mae=validation_error, training_pixels=int(training.sum()),
            validation_pixels=int(validation.sum()), negative_common_pixels=int(common.sum()),
            wrong_rotation_margin=margin, torsion_radians=g.texture_rotation,
            context_pass=train_error <= 3 and validation_error <= 3 and margin is not None and margin >= 1,
            footprints=dict(training=dxy[training], validation=dxy[validation],
                            negative=wrong[common], query=qxy,
                            search_query=query_union, search_validation=validation_union)))
    return min(candidates, key=lambda c: (c['training_mae'], abs(c['torsion_radians']), c['torsion_radians'])) if candidates else None


def predict(target_row, donors, hidden, ellseg, ritnet, *, verify_footprints=False):
    """Rerun target models with exclusions; return paired diagnostic arms.

    Donor rows need rgb/frame/ob; target metadata includes optional pixel_aspect,
    excluded and track_found. Excluded donor RGB must already have been scrubbed
    for model inference by observe(). No cached unmasked target proposals are read.
    """
    source = np.asarray(target_row['rgb'], np.float32)
    excluded = np.asarray(target_row.get('excluded', np.zeros(hidden.shape, bool)), bool) | hidden
    target = observe(source, ellseg, ritnet, excluded=excluded,
                     pixel_aspect=target_row.get('pixel_aspect', 1.),
                     track_found=target_row.get('track_found', True))
    rgb = source.copy()
    rgb[excluded] = 0
    yy, xx = np.where(hidden)
    query = np.column_stack([xx, yy]).astype(np.float32)
    shortlist = []
    if target.candidate_ok:
        for row in donors:
            if abs(row['frame']-target_row['frame']) <= 3:
                continue
            fit = fit_donor(rgb, target, row['rgb'], row['ob'], query)
            if fit:
                shortlist.append((row, fit))
    shortlist.sort(key=lambda item: (item[1]['training_mae'], item[0]['frame']))
    shortlist = shortlist[:24]
    output = {}
    for method in ('fixed', 'torsion'):
        fits = shortlist
        if method == 'torsion':
            fits = [(row, fit_donor(rgb, target, row['rgb'], row['ob'], query, rotation=True))
                    for row, _ in shortlist]
            fits = [(row, fit) for row, fit in fits if fit is not None]
        fits = sorted(fits, key=lambda item: (item[1]['training_mae'], item[0]['frame']))[:12]
        if target.geometry is None:
            n = len(query)
            result = dict(rgb=np.zeros((n,3), np.float32), supported=np.zeros(n,bool),
                          agreement=np.zeros(n,bool), donor_count=np.zeros(n,int),
                          mad=np.full(n,np.inf), donor_used=np.zeros((1,n),bool),
                          frame_ids=np.array([target_row['frame']]), material_uv=np.full((n,2),np.nan))
        else:
            result = reconstruct([rgb]+[r['rgb'] for r, _ in fits],
                [target.geometry]+[f['geometry'] for _, f in fits],
                [target.material]+[r['ob'].material for r, _ in fits],
                [target_row['frame']]+[r['frame'] for r, _ in fits], 0, query, max_mad=2)
        provenance = []
        for row, fit in fits:
            fp = fit['footprints']
            if verify_footprints:
                assert disjoint_footprints(fp['training'], fp['validation']).all()
                assert disjoint_footprints(fp['training'], fp['negative']).all()
                assert disjoint_footprints(fp['training'], fp['search_validation']).all()
                for part in ('training', 'validation', 'negative'):
                    assert disjoint_footprints(fp[part], fp['search_query']).all()
            provenance.append(dict(frame=row['frame'], **{k:v for k,v in fit.items()
                                                          if k not in ('geometry','footprints')}))
        used = result['donor_used'][1:].any(1)
        context = sum(used) >= 3 and all(f['context_pass'] for f, u in zip(provenance, used) if u)
        output[method] = (result, dict(candidate_ok=target.candidate_ok, reasons=target.reasons,
            geometry=asdict(target.geometry) if target.geometry else None, metrics=target.metrics,
            fits=provenance, context_pass=bool(context),
            clean_material_validated=False, scene_reflection_allowed=False))
    return output
