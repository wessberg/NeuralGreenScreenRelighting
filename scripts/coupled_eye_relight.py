"""All-or-nothing source-removal/scene-reflection transaction for one native eye.

Inputs must already be validated by the caller. This module does not earn support,
infer geometry or repair partial coverage. It has no production CLI integration.
"""
import numpy as np


def apply_coupled(source, mask, clean_transmitted=None, scene_reflection=None, *,
                  support=None, geometry_reliable=False, correspondence_reliable=False,
                  extent_reliable=False, scene_visibility_reliable=False, strength=1.):
    source, mask = np.asarray(source), np.asarray(mask, bool)
    if source.ndim != 3 or source.shape[-1] != 3 or mask.shape != source.shape[:2]:
        raise ValueError('Expected keyed RGB and a matching native eye-region mask')
    if not np.issubdtype(source.dtype, np.floating) or not np.isfinite(source).all():
        raise ValueError('Source must be finite floating-point linear RGB')
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('Strength must be within [0,1]')
    result = source.copy()
    gates = [('geometry', geometry_reliable), ('correspondence', correspondence_reliable),
             ('reflection_extent', extent_reliable), ('scene_visibility', scene_visibility_reliable)]
    reason = next((name for name, ok in gates if not ok), None)
    if reason is None and (not mask.any() or strength == 0):
        reason = 'empty_or_temporally_disabled'
    if reason is None:
        if clean_transmitted is None or scene_reflection is None or support is None:
            reason = 'missing_complete_evidence'
        else:
            clean, reflection, supported = map(np.asarray, (clean_transmitted, scene_reflection, support))
            if clean.shape != source.shape or reflection.shape != source.shape or supported.shape != mask.shape:
                raise ValueError('Transport/evidence shapes must match native keyed eye')
            if supported.dtype != np.bool_:
                raise ValueError('Support must be a boolean evidence mask')
            if not supported[mask].all() or not np.isfinite(clean[mask]).all() or not np.isfinite(reflection[mask]).all():
                reason = 'incomplete_or_nonfinite_evidence'
            elif (clean[mask] < 0).any() or (reflection[mask] < 0).any():
                reason = 'negative_radiance'
            else:
                result[mask] = source[mask]*(1-strength)+(clean[mask]+reflection[mask])*strength
    return result, dict(mode='source_keyed' if reason else 'coupled_replacement', reason=reason,
                        strength=0. if reason else float(strength),
                        changed_channels=int(np.count_nonzero(result != source)))
