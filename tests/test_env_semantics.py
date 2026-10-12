"""Official reset/terminal semantics, offline loading and scoring replica, pinned
against the installed arc_agi 0.9.9 / arcengine 0.9.3 (no network)."""
import os
import random

import numpy as np
import pytest

os.environ["OPERATION_MODE"] = "offline"
import arc_agi  # noqa: E402
from arc_agi.scorecard import EnvironmentScoreCalculator  # noqa: E402

from arc3.env import ActionRefused, CompetitionEnv  # noqa: E402
from arc3.perception import VolatileMask  # noqa: E402
from arc3.runner import ENV_DIR, _quiet_logger, list_games  # noqa: E402
from arc3.scoring import game_score  # noqa: E402

pytestmark = pytest.mark.skipif(not ENV_DIR.exists(), reason="public games not downloaded")


def _env(game):
    os.environ.pop("ONLY_RESET_LEVELS", None)
    arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=str(ENV_DIR),
                         logger=_quiet_logger())
    return arc, CompetitionEnv(arc, game)


def test_offline_loading_all_public_games():
    games = list_games()
    assert len(games) == 25


def test_game_over_reset_costs_one_action_and_restores_level():
    arc, env = _env("tr87")
    first = env.obs.grid.copy()
    i = 0
    while env.obs.state != "GAME_OVER" and i < 1000:
        env.step([1, 2, 3, 4][i % 4])
        i += 1
    assert env.obs.state == "GAME_OVER"
    with pytest.raises(ActionRefused):
        env.step(1)                               # non-RESET after GAME_OVER is refused
    n = env.actions_used
    env.step(0)
    assert env.actions_used == n + 1             # level reset is charged
    assert np.array_equal(env.obs.grid, first)
    run = arc.close_scorecard().environments[0].runs[0]
    assert run.actions == env.actions_used and run.resets == 1


def test_unavailable_action_refused():
    arc, env = _env("vc33")                      # click-only game
    assert env.obs.available_actions == (6,)
    with pytest.raises(ActionRefused):
        env.step(1)
    with pytest.raises(ActionRefused):
        env.step(6)                              # missing coordinates


def test_scoring_replica_matches_official_calculator():
    rng = random.Random(1)
    for _ in range(300):
        n = rng.randint(1, 10)
        base = [rng.randint(5, 300) for _ in range(n)]
        k = rng.randint(0, n)
        acts = [rng.randint(1, 900) for _ in range(k)]
        calc = EnvironmentScoreCalculator()
        for i in range(n):
            calc.add_level(i + 1, i < k, acts[i] if i < k else rng.randint(0, 50), base[i])
        assert abs(calc.to_score().score - game_score(base, acts)) < 1e-9


def test_volatile_mask_edge_vs_interior():
    vm = VolatileMask(band=4, min_hits=2)
    a = np.zeros((64, 64), np.uint8)
    b = a.copy(); b[62, 10] = 3            # HUD-like change at the bottom edge
    c = b.copy(); c[62, 11] = 3
    vm.update(a, b); vm.update(b, c)
    assert vm.mask[62, 11] or vm.mask[62, 10]
    d = c.copy(); d[30, 30] = 7            # interior gameplay change is never masked
    vm.update(c, d); vm.update(d, c)
    assert not vm.mask[30, 30]
    assert vm.gameplay_changed(c, d)
