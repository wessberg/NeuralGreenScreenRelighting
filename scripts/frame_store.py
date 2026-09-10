"""Hold a whole shot's frames in RAM compactly, and hand each one out as float32.

The eye passes (`eye_soft_removal.py`, `limbus_fit.py`) revisit every frame several times,
so they keep them all in memory. As full-frame float32 that is 25 MB a frame at 1080p:
fine for reporter_a's 150 frames, **about 30 GB for the 1200-frame clip**, and twice that
when an atlas sequence is pooled beside the source (round 37 section 6).

Nothing about the pixels changes, and that is checked rather than assumed:

* The keyed foreground is written by `key_plate.py` as HALF-float EXR, so every value
  `cv2.imread` returns is exactly representable in float16. `half()` stores it that way
  and ASSERTS the round trip is bit exact on every frame it is given; a float32 source
  (a future full-float EXR) fails loudly instead of being rounded.
* An 8-bit sequence (an inpainter's PNGs) is stored as its 8 bits and decoded on access
  by the caller's own conversion, so it is the identical computation, just later.

Frames come back READ-ONLY. A consumer that wrote into a frame it was handed would
otherwise corrupt the stored copy for every later pass; with the flag it raises at once.
A small cache keeps the last few decoded frames, because the passes visit both eyes of a
frame back to back.
"""
from collections import OrderedDict

import numpy as np


class FrameStore:
    """dict-like: frame index -> read-only float32 image, stored packed."""

    def __init__(self, pack, unpack, cache=3):
        self._pack, self._unpack = pack, unpack
        self._data = {}
        self._cache = OrderedDict()
        self._cache_size = cache

    def __setitem__(self, i, image):
        self._data[i] = self._pack(image)
        self._cache.pop(i, None)

    def __getitem__(self, i):
        hit = self._cache.get(i)
        if hit is not None:
            self._cache.move_to_end(i)
            return hit
        out = np.asarray(self._unpack(self._data[i]), np.float32)
        out.setflags(write=False)
        self._cache[i] = out
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return out

    def __contains__(self, i):
        return i in self._data

    def __len__(self):
        return len(self._data)

    def keys(self):
        return self._data.keys()

    def nbytes(self):
        return int(sum(v.nbytes for v in self._data.values()))


def _pack_half(image):
    image = np.asarray(image, np.float32)
    half = image.astype(np.float16)
    if not np.array_equal(half.astype(np.float32), image):
        raise ValueError('frame is not exactly representable in float16; refusing to round '
                         'it (key_plate.py writes HALF EXR -- was this written another way?)')
    return half


def half():
    """Store for scene-linear frames read from half-float EXR: exact, half the memory."""
    return FrameStore(_pack_half, lambda h: h.astype(np.float32))


def encoded(decode):
    """Store raw encoded pixels (e.g. 8-bit PNG) and run `decode` on access."""
    return FrameStore(lambda raw: np.ascontiguousarray(raw), decode)
