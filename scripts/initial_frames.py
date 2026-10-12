import os, sys, json, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["OPERATION_MODE"] = "offline"
import arc_agi
from arc3.viz.render import contact_sheet
from arc3.runner import list_games, ENV_DIR, _quiet_logger
games = list_games()
grids, titles = [], []
for g in games:
    arc = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=str(ENV_DIR), logger=_quiet_logger())
    w = arc.make(g)
    ob = w.observation_space
    grids.append(ob.frame[-1]); titles.append(f"{g[:4]} a={ob.available_actions}")
contact_sheet(grids, titles, scale=4, cols=5).save(sys.argv[1])
print("saved", sys.argv[1])
