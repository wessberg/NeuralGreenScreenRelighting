"""RGB-free upper bound for native all-arm correspondence context capacity."""
import numpy as np

from iris_correspondence import disjoint_footprints
from temporal_iris import footprint_valid


def fixed_control_capacity(target_mask, donor_mask, query, wrong_delta=(3., 0.)):
    """Any intersection adding learned maps can only REDUCE these samples.

    Full 2x2 footprints are conservative even at integer sampling coordinates.
    This measures capacity under a fixed control, not clean material or accuracy.
    """
    target_mask, donor_mask = np.asarray(target_mask, bool), np.asarray(donor_mask, bool)
    if target_mask.ndim != 2 or target_mask.shape != donor_mask.shape:
        raise ValueError('Expected matching native material masks')
    query = np.asarray(query, np.float32).reshape(-1, 2)
    if query.shape != (1, 2) or not np.isfinite(query).all():
        raise ValueError('Capacity is evaluated one finite query at a time')
    delta = np.asarray(wrong_delta, np.float32)
    if delta.shape != (2,) or not np.isfinite(delta).all():
        raise ValueError('Expected finite native xy wrong-map displacement')
    yy, xx = np.mgrid[:target_mask.shape[0]-1, :target_mask.shape[1]-1]
    grid = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)
    interior = footprint_valid(target_mask, grid)
    unhidden = interior & disjoint_footprints(grid, query)
    donor_union = np.concatenate([query, query+delta])
    common = unhidden.copy()
    for mapped in (grid, grid+delta):
        common &= footprint_valid(donor_mask, mapped)
        common &= disjoint_footprints(mapped, donor_union)
    return dict(target_footprints=int(interior.sum()),
                after_single_query_exclusion=int(unhidden.sum()),
                fixed_control_upper_bound=int(common.sum()),
                capacity_eligible=bool(common.sum() >= 8),
                context=grid[common], query_union=donor_union)
