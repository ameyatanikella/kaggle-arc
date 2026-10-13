"""Object parse with abstention, and correspondence between consecutive decision states.

An object is a same-colour 4-connected component (colours are categorical labels). The parse is
uncertain, so every consumer receives explicit abstentions:
  * identity: when several identical objects (same colour and shape) exist and more than one of
    them could be the source of a moved object, the match is AMBIGUOUS and no displacement is
    claimed for it;
  * touching objects of one colour merge into one component (reported as a size change, never as
    a move);
  * a masked region (HUD) is excluded before parsing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage

FOUR = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


@dataclass(frozen=True)
class Obj:
    color: int
    y0: int
    x0: int
    y1: int
    x1: int
    size: int
    shape: bytes          # canonical shape signature: packed cell mask of the bbox + its dims

    @property
    def sig(self) -> tuple:
        return (self.color, self.shape)

    @property
    def pos(self) -> tuple:
        return (self.y0, self.x0)

    @property
    def center(self) -> tuple:
        return ((self.y0 + self.y1) // 2, (self.x0 + self.x1) // 2)


def background(grid: np.ndarray) -> int:
    v, c = np.unique(grid, return_counts=True)
    return int(v[np.argmax(c)])


def parse(grid: np.ndarray, mask: Optional[np.ndarray] = None, bg: Optional[int] = None,
          max_objects: int = 400) -> list:
    """Same-colour 4-connected components except the background. Returns [] (abstains) when the
    frame has more than max_objects components (texture, not objects)."""
    g = np.asarray(grid)
    if bg is None:
        bg = background(g)
    out = []
    for c in np.unique(g):
        c = int(c)
        if c == bg:
            continue
        m = g == c
        if mask is not None:
            m &= ~mask
        lab, n = ndimage.label(m, structure=FOUR)
        if n == 0:
            continue
        for i, sl in enumerate(ndimage.find_objects(lab), start=1):
            if sl is None:
                continue
            cell = lab[sl] == i
            h, w = cell.shape
            out.append(Obj(c, sl[0].start, sl[1].start, sl[0].stop - 1, sl[1].stop - 1, int(cell.sum()),
                           bytes([h, w]) + np.packbits(cell).tobytes()))
            if len(out) > max_objects:
                return []
    return out


@dataclass
class Match:
    moved: list          # (obj_before, obj_after, dy, dx), unique identity
    ambiguous: list      # (obj_before, [candidates after]) identity not determined: no displacement
    appeared: list       # objects after with no same-signature source
    vanished: list       # objects before with no same-signature target
    recolored: list      # (obj_before, obj_after): same bbox and shape, different colour
    unchanged: int


def match(before: list, after: list) -> Match:
    """Correspondence by signature (colour + shape). Objects present at the same position in both
    frames are unchanged. Remaining objects of a signature are paired only when the pairing is
    forced (one source and one target); otherwise abstain (ambiguous)."""
    set_b = {(o.sig, o.pos) for o in before}
    set_a = {(o.sig, o.pos) for o in after}
    rem_b = [o for o in before if (o.sig, o.pos) not in set_a]
    rem_a = [o for o in after if (o.sig, o.pos) not in set_b]
    unchanged = len(before) - len(rem_b)
    by_sig_b: dict = {}
    by_sig_a: dict = {}
    for o in rem_b:
        by_sig_b.setdefault(o.sig, []).append(o)
    for o in rem_a:
        by_sig_a.setdefault(o.sig, []).append(o)
    moved, amb, appeared, vanished = [], [], [], []
    for sig in set(by_sig_b) | set(by_sig_a):
        bs, as_ = by_sig_b.get(sig, []), by_sig_a.get(sig, [])
        if len(bs) == 1 and len(as_) == 1:
            ob, oa = bs[0], as_[0]
            moved.append((ob, oa, oa.y0 - ob.y0, oa.x0 - ob.x0))
        elif bs and as_:
            for ob in bs:
                amb.append((ob, list(as_)))
        else:
            appeared += as_
            vanished += bs
    # recolouring: a vanished and an appeared object with the same bbox and shape mask
    rec = []
    left_a = list(appeared)
    for ob in list(vanished):
        for oa in left_a:
            if (oa.y0, oa.x0, oa.y1, oa.x1) == (ob.y0, ob.x0, ob.y1, ob.x1) and oa.shape == ob.shape:
                rec.append((ob, oa))
                left_a.remove(oa)
                vanished.remove(ob)
                break
    return Match(moved, amb, left_a, vanished, rec, unchanged)


def describe_match(m: Match, limit: int = 6) -> list:
    """Compact text lines for a model-facing report."""
    out = []
    for ob, oa, dy, dx in m.moved[:limit]:
        out.append(f"colour {ob.color} object (size {ob.size}) at (row {ob.y0}, col {ob.x0}) moved by (dy={dy}, dx={dx})")
    for ob, oa in m.recolored[:limit]:
        out.append(f"object at (row {ob.y0}, col {ob.x0}) size {ob.size} changed colour {ob.color} -> {oa.color}")
    for o in m.appeared[:limit]:
        out.append(f"colour {o.color} object (size {o.size}) appeared at (row {o.y0}, col {o.x0})")
    for o in m.vanished[:limit]:
        out.append(f"colour {o.color} object (size {o.size}) vanished from (row {o.y0}, col {o.x0})")
    if m.ambiguous:
        out.append(f"{len(m.ambiguous)} identical-object moves not attributed (identity ambiguous)")
    return out
