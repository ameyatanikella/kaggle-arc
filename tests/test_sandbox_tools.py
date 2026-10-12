import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle" / "b2"))
import sandbox_tools as T  # noqa: E402


def test_bfs_plan_grid():
    walls = {(1, 1), (1, 2)}
    def step(s, a):
        d = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}[a]
        n = (s[0] + d[0], s[1] + d[1])
        if not (0 <= n[0] < 3 and 0 <= n[1] < 4) or n in walls:
            return None
        return n
    r = T.bfs_plan((0, 0), step, lambda s: s == (2, 2), "UDLR")
    assert r["plan"] is not None and len(r["plan"]) == 4
    r2 = T.bfs_plan((0, 0), step, lambda s: s == (9, 9), "UDLR")
    assert r2["plan"] is None and r2["exhausted"]


def test_distinguish_two_wall_hypotheses():
    def mk(wall):
        return lambda x, a: (min(wall, max(0, x + (1 if a == "R" else -1))),) * 2
    r = T.distinguish([mk(3), mk(4)], [0, 0], ["L", "R"], max_depth=8)
    assert r["sequence"] == ["R", "R", "R", "R"] and len(r["groups"]) == 2
    same = T.distinguish([mk(3), mk(3)], [0, 0], ["L", "R"], max_depth=8)
    assert same["sequence"] is None and same["exhausted"]


def test_gf2_and_zm():
    r = T.gf2_solve([[1, 0, 1], [0, 1, 1]], [1, 1], cost=[1, 1, 5])
    assert r["presses"] == [0, 1] and r["cost"] == 2 and r["exact"]
    assert T.gf2_solve([[1, 1], [1, 1]], [1, 0]) is None
    assert T.zm_solve([[2]], [1], 4) is None
    assert T.zm_solve([[1, 1]], [3], 4, cost=[1, 2])["presses"] == [3, 0]


def test_fit_boolean_or_xor_and_guarded():
    two = T.fit_boolean([[1, 0], [0, 1]], [1, 1])
    assert (0, 1) in two["unidentified"]                      # interaction not pinned yet
    # without (0,0) the sparser rule 1 - b1*b2 also fits: MDL picks it (genuine ambiguity)
    amb = T.fit_boolean([[1, 0], [0, 1], [1, 1]], [1, 1, 0])
    assert amb["terms"] == {(): 1, (0, 1): -1}
    xor = T.fit_boolean([[0, 0], [1, 0], [0, 1], [1, 1]], [0, 1, 1, 0])
    assert xor["terms"] == {(0,): 1, (1,): 1, (0, 1): -2} and xor["unidentified"] == []
    g = T.fit_guarded(list(range(9)), [min(x, 5) for x in range(9)])
    assert g["kind"] == "guarded" and g["value_after"] == 5
