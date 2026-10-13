"""State keys and context descriptors over grids, shared by the hosts.

key(grid id)        masked-content digest (HUD pixels learned by VolatileMask are excluded)
ctx(key, action)    (class token, local token) for the algebra:
                      keys    -> ((a,), None)
                      clicks  -> ((6, colour under cursor, component-size bucket), 5x5 patch digest)
inventory(key)      multiset digest of object signatures (colour + shape): a mode / inventory proxy
"""
from __future__ import annotations

import hashlib

import numpy as np
from scipy import ndimage

from .objects import parse

FOUR = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def _digest(b: bytes, n: int = 10) -> bytes:
    return hashlib.blake2b(b, digest_size=n).digest()


class GridContext:
    def __init__(self, log, vmask):
        self.log = log
        self.vm = vmask
        self.rep: dict = {}          # key -> grid
        self._key_cache: dict = {}   # (grid id, mask version) -> key
        self._mask_bytes = b""

    def key_of_grid(self, g: np.ndarray) -> bytes:
        gg = np.array(g, dtype=np.uint8, copy=True)
        gg[self.vm.mask] = 255
        k = _digest(gg.tobytes())
        self.rep.setdefault(k, np.asarray(g, dtype=np.uint8))
        return k

    def key(self, gid: int) -> bytes:
        mb = self.vm.mask.tobytes()
        if mb != self._mask_bytes:
            self._mask_bytes = mb
            self._key_cache.clear()
        k = self._key_cache.get(gid)
        if k is None:
            k = self._key_cache[gid] = self.key_of_grid(self.log.grid(gid))
        return k

    def ctx(self, k, a):
        if a[0] != 6:
            return (a, None)
        g = self.rep.get(k)
        if g is None:
            return ((6,), None)
        x, y = int(a[1]), int(a[2])
        col = int(g[y, x])
        lab, _ = ndimage.label(g == col, structure=FOUR)
        size = int((lab == lab[y, x]).sum())
        y0, y1, x0, x1 = max(0, y - 2), min(64, y + 3), max(0, x - 2), min(64, x + 3)
        patch = _digest(np.ascontiguousarray(g[y0:y1, x0:x1]).tobytes() + bytes([y - y0, x - x0]), 6)
        return ((6, col, size.bit_length()), patch)

    def inventory(self, k):
        g = self.rep.get(k)
        if g is None:
            return None
        objs = parse(g, mask=self.vm.mask, max_objects=200)
        sigs = sorted((o.color, o.shape) for o in objs)
        return _digest(repr(sigs).encode(), 6)
