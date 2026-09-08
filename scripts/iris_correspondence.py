"""Experimental torsion fit and disjoint-context checks in iris material space.

Native geometry/visibility still require validation. Passing these tests measures
agreement on observed texture; it cannot certify an unobserved reflection region.
"""
from dataclasses import replace

import numpy as np
from native_iris import sample
from temporal_iris import footprint_valid, reconstruct


def disjoint_footprints(xy, held):
    """True where a sample shares NO bilinear contributor with validation samples."""
    a, b = np.floor(xy).astype(int), np.floor(held).astype(int)
    shared = (np.abs(a[:, None, :]-b[None, :, :]) <= 1).all(-1)
    return ~shared.any(1)


def fit_donor(target_rgb, target_ob, donor_rgb, donor_ob, *, rotation=True):
    """Fit on one spatial region; validate on another, with disjoint donor footprints.

    No hidden target RGB enters here: caller passes a scrubbed image and recomputed
    observation. Selection uses training pixels only; validation is a separate gate.
    """
    yy, xx = np.where(target_ob.material)
    xy = np.column_stack([xx, yy]).astype(np.float32)
    if len(xy) < 40:
        return None
    uv = target_ob.geometry.material(xy)
    # Spatially separated sectors, with a wide angular gap against interpolation overlap.
    train = uv[:, 1] < 1.25
    validation = uv[:, 1] > 1.9
    if train.sum() < 15 or validation.sum() < 10:
        return None
    values = target_rgb[yy, xx]
    candidates = []
    for angle in (np.linspace(-.18, .18, 19) if rotation else [0.]):
        g = replace(donor_ob.geometry,
                    texture_rotation=donor_ob.geometry.texture_rotation+float(angle))
        dxy = g.native(uv).astype(np.float32)
        usable = footprint_valid(donor_ob.material, dxy)
        v = validation & usable
        t = train & usable
        if v.sum() < 10 or t.sum() < 15:
            continue
        t &= disjoint_footprints(dxy, dxy[v])
        if t.sum() < 15:
            continue
        pred = sample(donor_rgb, dxy)
        error = np.abs(pred-values).mean(-1)
        candidates.append((float(error[t].mean()), float(angle), g, dxy, v, error))
    if not candidates:
        return None
    candidates.sort(key=lambda c: (c[0], abs(c[1]), c[1]))
    training, angle, g, dxy, v, error = candidates[0]
    # A displaced map is a negative correspondence control, with identical validation pixels.
    negative = replace(g, texture_rotation=g.texture_rotation+.35).native(uv).astype(np.float32)
    common = v & footprint_valid(donor_ob.material, negative)
    negative_error = np.abs(sample(donor_rgb, negative)-values).mean(-1)
    validation_error = float(error[v].mean())
    margin = float((negative_error[common]-error[common]).mean()) if common.sum() >= 8 else None
    return {'geometry': g, 'training_mae': training, 'validation_mae': validation_error,
            'validation_pixels': int(v.sum()), 'negative_common_pixels': int(common.sum()),
            'wrong_rotation_margin': margin, 'torsion_radians': angle,
            # Fixed experimental gate. Donor agreement is not used to establish correspondence.
            'context_pass': training <= 3 and validation_error <= 3 and
                            margin is not None and margin >= 1}


def predict(rows, index, hidden, *, rotation, max_donors=12):
    """Rerun native detection with hidden pixels excluded, then fit eligible donors."""
    from native_iris import detect
    target = rows[index]
    scrubbed = target['rgb'].copy()
    scrubbed[hidden] = 0
    ob = detect(scrubbed, target['seed'], target['radius'], hidden,
                pixel_aspect=target.get('pixel_aspect', 1.),
                eye_rotation=target.get('eye_rotation', 0.))
    fits = []
    if ob.candidate_ok:
        for j, row in enumerate(rows):
            if abs(row['frame']-target['frame']) <= 3 or not row['ob'].candidate_ok:
                continue
            fit = fit_donor(scrubbed, ob, row['rgb'], row['ob'], rotation=False)
            if fit is not None:
                fits.append((fit['training_mae'], j, fit))
    fits.sort(key=lambda f: (f[0], rows[f[1]]['frame']))
    # Both arms start with the same shortlist, chosen without hidden RGB.
    fits = fits[:24]
    if rotation:
        refined = []
        for _, j, _ in fits:
            row = rows[j]
            fit = fit_donor(scrubbed, ob, row['rgb'], row['ob'], rotation=True)
            if fit is not None:
                refined.append((fit['training_mae'], j, fit))
        fits = sorted(refined, key=lambda f: (f[0], rows[f[1]]['frame']))
    fits = fits[:max_donors]
    images, geometries, masks, ids = [scrubbed], [ob.geometry], [ob.material], [target['frame']]
    provenance = []
    for _, j, fit in fits:
        row = rows[j]
        images.append(row['rgb'])
        geometries.append(fit['geometry'])
        masks.append(row['ob'].material)
        ids.append(row['frame'])
        provenance.append({'frame': row['frame'], **{k: v for k, v in fit.items() if k != 'geometry'}})
    yy, xx = np.where(hidden)
    result = reconstruct(images, geometries, masks, ids, 0, np.column_stack([xx, yy]), max_mad=2)
    # Both arms use same support rules. This certificate remains experimental and local.
    used = result['donor_used'][1:].any(1)
    context_ok = sum(used) >= 3 and all(p['context_pass'] for p, u in zip(provenance, used) if u)
    return result, {'geometry': ob.geometry, 'candidate_ok': ob.candidate_ok,
                    'reasons': ob.reasons, 'fits': provenance, 'context_pass': bool(context_ok)}
