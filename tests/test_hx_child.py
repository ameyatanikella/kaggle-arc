"""Sandbox-side HX helpers (kaggle/b2/hx_child.py): decision-valued probes, replay checks, doc keys.

The module is pure Python and is loaded here exactly as the sandbox would execute it (module level).
"""
import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location("hx_child_t", ROOT / "kaggle/b2/hx_child.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class F:  # minimal FrameView stand-in
    def __init__(self, rows, level=1, step=0):
        self.ascii = "\n".join(rows)
        self.level = level
        self.step = step


class Tr:
    def __init__(self, action, before, after, result):
        self.action, self.before_frame, self.after_frame, self.result = action, before, after, result


def test_probe_acts_when_plans_agree():
    m = load()
    models = {"a": (0, lambda s, a: s + (1 if a == "R" else -1)), "b": (0, lambda s, a: s + (2 if a == "R" else -1))}
    out = m.hx_probe(models, {"g": lambda s: s >= 4}, ["R", "L"])
    assert out["decision"] == "act" and out["action"] == "R"


def test_probe_splits_disagreeing_models_when_it_pays():
    m = load()
    # model a: R moves +1, L moves -1; model b: R moves -1, L moves +1 (mirrored). Goal: position 3.
    a = (0, lambda s, x: s + (1 if x == "R" else -1))
    b = (0, lambda s, x: s + (-1 if x == "R" else 1))
    out = m.hx_probe({"a": a, "b": b}, {"g": lambda s: s == 3}, ["R", "L"], depth=1)
    # first actions differ (R vs L); one action separates the models and is also progress under one
    assert out["decision"] == "probe"
    assert len(out["probe"]) == 1 and out["gain"] > 0
    assert len(out["groups"]) == 2


def test_probe_follows_when_disagreement_is_not_worth_testing():
    m = load()
    # both models reach the goal with plan [X] but differ on an irrelevant action Y; plans agree
    a = (0, lambda s, x: 5 if x == "X" else s)
    b = (0, lambda s, x: 5 if x == "X" else s + 1)
    out = m.hx_probe({"a": a, "b": b}, {"g": lambda s: s == 5}, ["X", "Y"])
    assert out["decision"] == "act" and out["action"] == "X"


def test_probe_reports_none_when_no_model_reaches_a_goal():
    m = load()
    a = (0, lambda s, x: s)
    out = m.hx_probe({"a": a}, {"g": lambda s: s == 1}, ["X"])
    assert out["decision"] == "none"


def test_kind_and_moves_exclude_level_up_death_reset():
    m = load()
    f0, f1, f2, f3 = F(["ab"], 1), F(["ba"], 1), F(["zz"], 2), F(["zy"], 2)
    trs = [Tr("ACTION1", f0, f1, {"action_name": "ACTION1"}),
           Tr("ACTION2", f1, f2, {"action_name": "ACTION2", "level_completed": True}),
           Tr("ACTION1", f2, f3, {"action_name": "ACTION1"}),
           Tr("ACTION3", f3, f3, {"action_name": "ACTION3", "game_over": True}),
           Tr("RESET", f3, f2, {"action_name": "RESET", "automatic": True})]
    rg = {"transitions": trs, "current_frame": f2, "frame_diff": lambda b=None, a=None: {"diff": True}}
    m._hx_refresh(rg)
    assert [t.kind for t in trs] == ["move", "level_up", "move", "death", "reset"]
    assert rg["moves"] == [trs[2]]
    assert "note" in rg["frame_diff"]()                      # last action was a reset
    assert rg["frame_diff"](f0, f1) == {"diff": True}         # explicit frames still work
    assert "hx_check" in rg and "hx_probe" in rg


def test_check_classifies_hidden_state_and_border():
    m = load()
    A, B, C = F(["....", "....", "....", "...."]), F(["x...", "....", "....", "...."]), F(["....", ".x..", "....", "...."])
    trs = [Tr("ACTION1", A, B, {}), Tr("ACTION1", A, C, {})]
    rg = {"transitions": trs, "current_frame": C, "frame_diff": None}
    m._hx_refresh(rg)
    out = m.hx_check(lambda before, a: before.ascii)        # identity model
    causes = [x["cause"] for x in out["mismatches"]]
    assert out["checked"] == 2 and out["correct"] == 0
    assert causes[0] == "border"                              # only (0,0) changed: inside the 4-cell border
    assert causes[1] == "hidden_state"                        # same board + action, another result


def test_doc_lists_returned_keys():
    m = load()
    doc = m.HX_DOC
    for name in ("hx_check", "hx_probe", "hx_goals", "hx_relations", "hx_wins"):
        assert f"`{name}(" in doc
    for k in ("checked", "correct", "mismatches", "errors", "decision", "action", "probe", "gain", "plans", "groups",
              "candidates", "false_now", "true_now", "hazards", "last_win", "relations", "conflicts", "wins"):
        assert f"'{k}'" in doc, k
