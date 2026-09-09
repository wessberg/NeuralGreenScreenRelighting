"""Observed-pixel residual transport, not a certificate of separated material."""
import numpy as np


def preserve_residual(source, fitted_clean, fitted_reflection):
    """Keep the entire observed residual; never clamp an unresolved negative value.

    Error against true clean equals true reflection minus estimated reflection.
    Consequently this avoids texture-grid smoothing but cannot fix layer ambiguity.
    The caller still needs complete geometry/material evidence for replacement.
    """
    source, clean, reflection = [np.asarray(a) for a in
                                 (source, fitted_clean, fitted_reflection)]
    if source.shape != clean.shape or source.shape != reflection.shape:
        raise ValueError('Source and fitted layers must have identical shapes')
    if not all(np.issubdtype(a.dtype, np.floating) and np.isfinite(a).all()
               for a in (source, clean, reflection)):
        raise ValueError('Finite floating point radiance required')
    residual = source - (clean + reflection)
    return source - reflection, residual
