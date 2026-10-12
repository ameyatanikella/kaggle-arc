"""Grid perception: components, background, HUD/volatile masking, diffs.

Colors are categorical labels; nothing here treats color ids as magnitudes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

FOUR = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


@dataclass(frozen=True)
class Component:
    color: int
    size: int
    y0: int
    x0: int
    y1: int  # inclusive
    x1: int  # inclusive
    cy: int  # a cell inside the component closest to its centroid
    cx: int

    @property
    def key(self):
        return (self.color, self.y0, self.x0, self.y1, self.x1)


def background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def components(grid: np.ndarray, skip_colors=(), min_size: int = 1) -> list[Component]:
    """Same-color 4-connected components (excluding `skip_colors`)."""
    out: list[Component] = []
    for c in np.unique(grid):
        c = int(c)
        if c in skip_colors:
            continue
        lab, n = ndimage.label(grid == c, structure=FOUR)
        if n == 0:
            continue
        objs = ndimage.find_objects(lab)
        for i, sl in enumerate(objs, start=1):
            if sl is None:
                continue
            m = lab[sl] == i
            size = int(m.sum())
            if size < min_size:
                continue
            ys, xs = np.nonzero(m)
            my, mx = ys.mean(), xs.mean()
            k = int(np.argmin((ys - my) ** 2 + (xs - mx) ** 2))
            out.append(Component(c, size, sl[0].start, sl[1].start, sl[0].stop - 1, sl[1].stop - 1,
                                 int(ys[k] + sl[0].start), int(xs[k] + sl[1].start)))
    return out


def diff_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a != b


class VolatileMask:
    """Learns pixels whose changes are not gameplay (step bars, timers).

    A changed region counts as HUD-like when it lies entirely within an edge band.
    A shrinking bar changes a *different* pixel each step, so hits are counted per
    edge line (a row of the top/bottom band, a column of the left/right band), not
    per pixel. Once a line has shown HUD-like change in >= `min_hits` transitions,
    the whole line is masked for state identity. Interior pixels are never masked.
    """

    def __init__(self, shape=(64, 64), band: int = 4, min_hits: int = 2):
        self.band = band
        self.min_hits = min_hits
        self.shape = shape
        h, w = shape
        self.edge = np.zeros(shape, dtype=bool)
        self.edge[:band, :] = True
        self.edge[-band:, :] = True
        self.edge[:, :band] = True
        self.edge[:, -band:] = True
        self.row_hits = np.zeros(h, dtype=np.int32)
        self.col_hits = np.zeros(w, dtype=np.int32)
        self.mask = np.zeros(shape, dtype=bool)

    def update(self, before: np.ndarray, after: np.ndarray) -> None:
        d = before != after
        if not d.any():
            return
        h, w = self.shape
        lab, n = ndimage.label(d, structure=np.ones((3, 3), bool))
        rows_hit, cols_hit = set(), set()
        for i in range(1, n + 1):
            ys, xs = np.nonzero(lab == i)
            if not np.all(self.edge[ys, xs]):
                continue
            # attribute to the nearest edge: horizontal bands -> rows, vertical -> cols
            dist = {"top": ys.max(), "bottom": h - 1 - ys.min(), "left": xs.max(), "right": w - 1 - xs.min()}
            side = min(dist, key=dist.get)
            if side in ("top", "bottom"):
                rows_hit.update(int(y) for y in np.unique(ys))
            else:
                cols_hit.update(int(x) for x in np.unique(xs))
        for y in rows_hit:
            self.row_hits[y] += 1
        for x in cols_hit:
            self.col_hits[x] += 1
        m = np.zeros(self.shape, dtype=bool)
        m[self.row_hits >= self.min_hits, :] = True
        m[:, self.col_hits >= self.min_hits] = True
        self.mask = m & self.edge

    def key(self, grid: np.ndarray) -> bytes:
        g = grid.copy()
        g[self.mask] = 255
        return g.tobytes()

    def gameplay_changed(self, before: np.ndarray, after: np.ndarray) -> bool:
        d = (before != after) & ~self.mask
        return bool(d.any())
