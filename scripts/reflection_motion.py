"""Controlled additive-reflection motion over observed RGB; diagnostic only.

The translated base includes every original photographed highlight. It is not
clean performer texture, and the synthetic reflection is not a corneal render.
"""
import numpy as np


def sequence(rgb, point, axis, *, frames=24, peak=.6):
    a = np.asarray(rgb, np.float32)
    q = np.asarray(point, np.float32)
    direction = np.asarray(axis, np.float32)
    if a.ndim != 3 or a.shape[-1] != 3 or not np.isfinite(a).all():
        raise ValueError('Expected finite native HxWx3 display RGB')
    if a.min() < 0 or a.max() > 255 or frames != 24 or peak < 0:
        raise ValueError('Expected display RGB 0..255, 24 frames, nonnegative peak')
    if q.shape != (2,) or direction.shape != (2,) or not np.isfinite(q).all():
        raise ValueError('Expected one finite xy point and integer unit axis')
    if tuple(direction) not in ((1., 0.), (0., 1.), (-1., 0.), (0., -1.)):
        raise ValueError('Expected signed native unit axis')
    h, w = a.shape[:2]
    offsets = (np.arange(frames)//4)[:, None]*direction
    material = q[None]+offsets
    reflection = q[None]+3*direction+(12-np.arange(frames))[:, None]*direction
    if ((material < 0) | (material >= [w-1, h-1])).any():
        raise ValueError('Material query must retain complete native footprints')
    # Reflect-101 indexing, exact integer translation: no interpolated texture.
    def reflect(indices, length):
        period = 2*(length-1)
        folded = indices % period
        return np.minimum(folded, period-folded).astype(int)
    base = np.stack([a[reflect(np.arange(h)-dy, h)[:, None],
                       reflect(np.arange(w)-dx, w)[None, :]] for dx, dy in offsets])
    yy, xx = np.mgrid[:h, :w]
    delta = np.stack([xx, yy], -1)[None]-reflection[:, None, None]
    lobes = np.where(np.abs(delta) < 3, .5+.5*np.cos(np.pi*delta/3), 0)
    addition = (peak*np.prod(lobes, axis=-1)).astype(np.float32)
    display = base/255
    linear = np.where(display <= .04045, display/12.92,
                      ((display+.055)/1.055)**2.4)
    summed = linear+addition[..., None]
    clipped = summed > 1
    encoded = np.where(summed <= .0031308, summed*12.92,
                       1.055*np.maximum(summed, 0)**(1/2.4)-.055)*255
    contaminated = np.clip(encoded, 0, 255).astype(np.float32)
    # Avoid a transfer round-trip changing the unmodified control pixels.
    contaminated[addition == 0] = base[addition == 0]
    return dict(base=base, contaminated=contaminated, material=material.astype(np.float32),
                reflection=reflection.astype(np.float32), addition=addition,
                clipped=clipped, offsets=offsets.astype(np.float32))
