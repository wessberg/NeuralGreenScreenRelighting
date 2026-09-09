"""Known local observed-texture motion against a stationary surround; no repair."""
import numpy as np

from reflection_motion import sequence as global_sequence


def decode(display):
    v = np.asarray(display, np.float32)/255
    return np.where(v <= .04045, v/12.92, ((v+.055)/1.055)**2.4)


def encode(linear):
    v = np.asarray(linear, np.float32)
    return np.clip(np.where(v <= .0031308, v*12.92,
                           1.055*np.maximum(v, 0)**(1/2.4)-.055)*255, 0, 255)


def sequence(rgb, point, axis):
    """24 frames; exact moving interior, linear blend, exact stationary exterior.

    The aperture is artificial, not an anatomical material/visibility mask.
    The point's full 2x2 footprint is guaranteed inside the unblended interior.
    """
    result = global_sequence(rgb, point, axis)
    a = np.asarray(rgb, np.float32)
    yy, xx = np.mgrid[:a.shape[0], :a.shape[1]]
    radius = np.hypot(xx-point[0], yy-point[1])
    weight = (.5+.5*np.cos(np.pi*np.clip((radius-8)/4, 0, 1))).astype(np.float32)
    moving = result['base']
    local = encode(decode(moving)*weight[None, ..., None] +
                   decode(a)[None]*(1-weight[None, ..., None]))
    local[:, weight == 1] = moving[:, weight == 1]
    local[:, weight == 0] = a[weight == 0]
    # Frame zero is entirely the same observed image, including the blend band.
    local[0] = a
    for xy in result['material']:
        x, y = np.floor(xy).astype(int)
        if not (weight[y:y+2, x:x+2] == 1).all():
            raise ValueError('Query footprint leaves exact local-motion interior')
    summed = decode(local)+result['addition'][..., None]
    contaminated = encode(summed)
    contaminated[result['addition'] == 0] = local[result['addition'] == 0]
    result.update(base=local, contaminated=contaminated, clipped=summed > 1,
                  local_weight=weight)
    return result
