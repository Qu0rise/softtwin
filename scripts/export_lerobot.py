"""Export the tactile dataset to the LeRobotDataset v2.1 layout (parquet per episode + meta/*.json[l]).

    python scripts/export_lerobot.py [--src data/tactile_dataset] [--out data/lerobot_softtwin] [--limit N]

Per frame (50 Hz, the grasp phase that the dataset records):
    observation.state    float32[6]   p [kPa], bend_model [deg], bend_meas x3 [deg], f_est [N]
    observation.tactile  float32[24]  fabric skin after the sensor model [N]
    action               float32[1]   p_cmd [kPa]
    + timestamp, frame_index, episode_index, index, task_index, next.done
The three camera stills per episode stay in the original .npz (LeRobot expects one image per frame);
the outcome annotation, hidden physics and F* go into meta/episodes.jsonl next to the episode length.
Only numpy and pyarrow are needed; the lerobot package is not required to write or read this layout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
FPS = 50
STATE_NAMES = ["p_kpa", "bend_model_deg", "bend_f0_deg", "bend_f1_deg", "bend_f2_deg", "f_est_n"]
TAX_NAMES = [f"f{f}_s{s}_{side}" for f in range(3) for s in range(4) for side in ("prox", "dist")]


def task_text(meta):
    return f"pick up the {meta['looks_like'].replace('_', ' ')} without crushing it and place it in the tray"


def _stats(a):
    a = np.asarray(a, np.float64).reshape(len(a), -1)
    return dict(min=a.min(0).tolist(), max=a.max(0).tolist(), mean=a.mean(0).tolist(), std=a.std(0).tolist(), count=[len(a)])


def export(src: Path, out: Path, limit: int | None = None):
    metas = [json.loads(line) for line in open(src / "index.jsonl")][:limit]
    tasks: dict[str, int] = {}
    (out / "meta").mkdir(parents=True, exist_ok=True)
    ep_lines, st_lines, g = [], [], 0
    for ei, m in enumerate(metas):
        d = np.load(src / m["file"])
        T = len(d["t"])
        state = np.c_[d["p"], d["bend_model"], d["bend_meas"], d["f_est"]].astype(np.float32)
        tax = d["skin"].astype(np.float32)
        act = d["p_cmd"].astype(np.float32).reshape(T, 1)
        ts = (np.arange(T) / FPS).astype(np.float32)
        ti = tasks.setdefault(task_text(m), len(tasks))
        tbl = pa.table({
            "observation.state": pa.array(list(state), type=pa.list_(pa.float32(), 6)),
            "observation.tactile": pa.array(list(tax), type=pa.list_(pa.float32(), 24)),
            "action": pa.array(list(act), type=pa.list_(pa.float32(), 1)),
            "timestamp": ts, "frame_index": np.arange(T, dtype=np.int64),
            "episode_index": np.full(T, ei, np.int64), "index": np.arange(g, g + T, dtype=np.int64),
            "task_index": np.full(T, ti, np.int64), "next.done": np.r_[np.zeros(T - 1, bool), True],
        })
        p = out / "data" / f"chunk-{ei // 1000:03d}" / f"episode_{ei:06d}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(tbl, p)
        ep_lines.append(dict(episode_index=ei, tasks=[task_text(m)], length=T, source_id=m["id"], object=m["object"],
                             true_class=m["true_class"], F_target=m["F_target"], outcome=m["outcome"], physics=m["physics"]))
        st_lines.append(dict(episode_index=ei, stats={"observation.state": _stats(state), "observation.tactile": _stats(tax),
                                                       "action": _stats(act), "timestamp": _stats(ts)}))
        g += T
    feats = {
        "observation.state": dict(dtype="float32", shape=[6], names=STATE_NAMES),
        "observation.tactile": dict(dtype="float32", shape=[24], names=TAX_NAMES),
        "action": dict(dtype="float32", shape=[1], names=["p_cmd_kpa"]),
        "timestamp": dict(dtype="float32", shape=[1], names=None),
        "frame_index": dict(dtype="int64", shape=[1], names=None),
        "episode_index": dict(dtype="int64", shape=[1], names=None),
        "index": dict(dtype="int64", shape=[1], names=None),
        "task_index": dict(dtype="int64", shape=[1], names=None),
        "next.done": dict(dtype="bool", shape=[1], names=None),
    }
    info = dict(codebase_version="v2.1", robot_type="softtwin_3f_pneumatic_sim", total_episodes=len(metas), total_frames=g,
                total_tasks=len(tasks), total_videos=0, total_chunks=(len(metas) - 1) // 1000 + 1, chunks_size=1000, fps=FPS,
                splits={"train": f"0:{len(metas)}"}, data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                video_path=None, features=feats, license="CC-BY-4.0", source="data/tactile_dataset (see meta/info.json there)")
    json.dump(info, open(out / "meta" / "info.json", "w"), indent=1)
    with open(out / "meta" / "tasks.jsonl", "w") as f:
        for t, i in sorted(tasks.items(), key=lambda kv: kv[1]):
            f.write(json.dumps(dict(task_index=i, task=t)) + "\n")
    for name, lines in (("episodes.jsonl", ep_lines), ("episodes_stats.jsonl", st_lines)):
        with open(out / "meta" / name, "w") as f:
            f.writelines(json.dumps(x) + "\n" for x in lines)
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(REPO / "data" / "tactile_dataset"))
    ap.add_argument("--out", default=str(REPO / "data" / "lerobot_softtwin"))
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    info = export(Path(a.src), Path(a.out), a.limit)
    print(f"{info['total_episodes']} episodes, {info['total_frames']} frames, {info['total_tasks']} tasks -> {a.out}")
