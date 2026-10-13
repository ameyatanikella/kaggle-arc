"""HX event log, goal evidence and triggers on synthetic frame sequences (no engine)."""
import numpy as np

from arc3.hx.events import DEATH, MOVE, RESET, WIN, EventLog
from arc3.hx.goals import GoalModel, facts
from arc3.hx.objects import match, parse
from arc3.hx.session import HXSession


def board(player, exit_=(10, 10), exit_present=True):
    g = np.zeros((64, 64), np.uint8)
    if exit_present:
        g[exit_[0]:exit_[0] + 2, exit_[1]:exit_[1] + 2] = 14
    g[player[0]:player[0] + 2, player[1]:player[1] + 2] = 9
    return g


def test_event_kinds_and_win_frame():
    log = EventLog()
    s0 = board((20, 20))
    log.start(s0)
    s1 = board((20, 18))
    st = log.record((3,), [s1], "NOT_FINISHED", 0, 0)
    assert st.kind == MOVE and log.grid(st.post) is not None
    win = board((10, 10), exit_present=False)
    nxt = board((40, 40), exit_=(5, 5))
    st = log.record((3,), [s1, win, nxt], "NOT_FINISHED", 0, 1)
    assert st.kind == WIN
    assert np.array_equal(log.grid(st.terminal), win)        # frames[-2]
    assert np.array_equal(log.grid(st.init), nxt)            # frames[-1]
    st = log.record((1,), [nxt], "GAME_OVER", 1, 1)
    assert st.kind == DEATH
    st = log.record((0,), [nxt], "NOT_FINISHED", 1, 1)
    assert st.kind == RESET
    tr = list(log.transitions())
    assert len(tr) == 1 and tr[0][3] == MOVE                 # the win, death and reset are not dynamics
    tr = list(log.transitions(include_terminal=True))
    assert {k for *_, k in tr} == {MOVE, WIN, DEATH}


def test_win_frame_abstains_with_one_frame():
    log = EventLog()
    log.start(board((20, 20)))
    st = log.record((1,), [board((40, 40))], "NOT_FINISHED", 0, 1)
    assert st.kind == WIN and st.terminal is None


def test_object_match_abstains_on_identical_objects():
    a = np.zeros((64, 64), np.uint8)
    a[10, 10] = a[10, 20] = 5
    b = np.zeros((64, 64), np.uint8)
    b[11, 10] = b[11, 20] = 5
    m = match(parse(a), parse(b))
    assert not m.moved and len(m.ambiguous) == 2


def test_goal_transfer_absent_exit():
    gm = GoalModel()
    start = board((20, 20))
    visited = [start, board((20, 18)), board((15, 12))]
    win = board((10, 10), exit_present=False)
    explain = gm.observe_win(0, start, visited, win)
    assert ("absent", 14) in explain
    init2 = board((40, 40), exit_=(5, 5))
    rep = gm.at_level_start(1, init2)
    assert ("absent", 14) in [f for f, _ in rep["open"]]


def test_session_triggers():
    hx = HXSession()
    s = board((20, 20))
    hx.start(s)
    # four no-ops in a row -> one noops report
    for _ in range(4):
        hx.record((1,), [s], "NOT_FINISHED", 0, 0, pre_grid=s)
    rep = hx.take_report()
    assert "changed nothing" in rep and hx.take_report() == ""
    # a hidden-state conflict: same board and action, different result
    s2 = board((20, 22))
    hx.record((2,), [s2], "NOT_FINISHED", 0, 0, pre_grid=s)
    hx.record((2,), [s], "NOT_FINISHED", 0, 0, pre_grid=s)
    assert "does not show" in hx.take_report()
    # a level completion with a win frame -> level report
    win = board((10, 10), exit_present=False)
    nxt = board((40, 40), exit_=(5, 5))
    hx.record((3,), [win, nxt], "NOT_FINISHED", 0, 1, pre_grid=s)
    rep = hx.take_report()
    assert "level 1 was completed" in rep and "absent(colour 14)" in rep
    out = hx.request("goals", {})
    assert any(c["fact"] == "absent(colour 14)" for c in out["candidates"])
    assert hx.request("wins", {})["wins"][0]["win_grid"] is not None
