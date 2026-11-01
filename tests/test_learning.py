"""Learned force window and the LeRobot export."""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "sim_mujoco"), str(ROOT / "learning"), str(ROOT / "scripts")]


def _x(meta):
    import tactile as T
    return np.r_[[meta["vision_features"][k] for k in T.VISION_FEATS], [meta["plunger_features"][k] for k in T.PLUNGER_FEATS]]


def test_force_window_is_physically_ordered():
    """p(crush) must rise and p(slip) must fall with F, and the chosen F must sit inside each object's safe range"""
    import tactile as T
    fw = T.ForceWindow.from_json(json.load(open(ROOT / "learning/force_window.json")))
    metas = [json.loads(line) for line in open(ROOT / "data/tactile_dataset/index.jsonl")]
    for obj, lo, hi in [("strawberry_overripe", 0.5, 1.5), ("tomato_overripe", 0.6, 2.0), ("tomato", 1.0, 5.0)]:
        xs = [_x(m) for m in metas if m["object"] == obj][:10]
        for x in xs:
            g, pc, ps = fw.curves(x)
            assert np.all(np.diff(pc) >= 0) and np.all(np.diff(ps) <= 0) and ps[0] > ps[-1]
        Fs = [fw.choose(x) for x in xs]
        assert lo <= np.median(Fs) <= hi, (obj, Fs)


def test_lerobot_export(tmp_path):
    import export_lerobot as E
    info = E.export(ROOT / "data/tactile_dataset", tmp_path, limit=3)
    assert info["codebase_version"] == "v2.1" and info["total_episodes"] == 3
    eps = [json.loads(line) for line in open(tmp_path / "meta/episodes.jsonl")]
    n = 0
    for e in eps:
        t = pq.read_table(tmp_path / f"data/chunk-000/episode_{e['episode_index']:06d}.parquet").to_pydict()
        assert len(t["index"]) == e["length"] and t["index"][0] == n and t["next.done"][-1]
        assert len(t["observation.state"][0]) == 6 and len(t["observation.tactile"][0]) == 24
        n += e["length"]
    assert n == info["total_frames"]
