"""Extend reliable interior irradiance across a coarse proxy's silhouette.

The photographic matte remains authoritative. Proxy coverage is not a second matte.
"""
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


def extend_irradiance(rgb, coverage, radius=8):
    coverage = np.clip(coverage.astype(np.float32), 0, 1)
    straight = rgb / np.maximum(coverage[..., None], 1e-6)
    reliable = coverage >= .999
    if radius:
        reliable = cv2.erode(reliable.astype(np.uint8),
                             np.ones((2*radius+1, 2*radius+1), np.uint8)) > 0
    if not reliable.any():
        raise ValueError('No reliable proxy interior for irradiance extension')
    # Full precision nearest extension avoids the previous uint8 inpaint quantisation and
    # its arbitrary max-dependent scaling. Never change a reliable interior sample.
    indices = distance_transform_edt(~reliable, return_distances=False, return_indices=True)
    return straight[indices[0], indices[1]], reliable
