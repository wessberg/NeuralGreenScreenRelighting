"""RGB-free context capacity after source-label centroid registration."""
import numpy as np

from iris_correspondence import disjoint_footprints
from temporal_iris import footprint_valid


def registered_capacity(target, donor, query):
    target, donor = np.asarray(target, bool), np.asarray(donor, bool)
    q = np.asarray(query, np.float32).reshape(-1, 2)
    if target.ndim != 2 or target.shape != donor.shape or min(target.shape) < 2:
        raise ValueError('Expected matching native material masks')
    if q.shape != (1, 2) or not np.isfinite(q).all():
        raise ValueError('Expected one finite native query')
    if not target.any() or not donor.any():
        raise ValueError('Centroids require nonempty material masks')
    shift = np.rint(np.argwhere(donor).mean(0)[::-1] -
                    np.argwhere(target).mean(0)[::-1]).astype(np.float32)
    yy, xx = np.mgrid[:target.shape[0]-1, :target.shape[1]-1]
    grid = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)
    delta = np.array([3, 0], np.float32)
    union = np.concatenate([q+shift, q+shift+delta])
    common = footprint_valid(target, grid) & disjoint_footprints(grid, q)
    for mapped in (grid+shift, grid+shift+delta):
        common &= footprint_valid(donor, mapped) & disjoint_footprints(mapped, union)
    return dict(shift=shift, context=grid[common], query_union=union,
                context_upper_bound=int(common.sum()), capacity_eligible=bool(common.sum() >= 8))
