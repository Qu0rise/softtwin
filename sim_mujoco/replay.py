"""Draw a run recorded in Isaac Sim / PhysX (sim/isaac_headless/q1_isaac.py, v2_replay_*.json) with MuJoCo's renderer.

Nothing is simulated here: palm position, the 12 finger joint angles and the object pose of every control step are
written into a MuJoCo model of the same cell (same seed -> same object instance) and only drawn (mj_forward).
The motion and the contact forces are PhysX's; the pixels are MuJoCo's (the RTX renderer does not start on the
machine that ran PhysX: its GPU driver is older than Isaac Sim 5.0 needs).

    python sim_mujoco/replay.py isaac_io/v2_replay_0_tomato_closed_learned.json --cam close --out media/x.mp4
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import softtwin_mj as S  # first: it picks the OpenGL backend before mujoco is imported
import mujoco  # noqa: E402


def frames(path, cam="close", size=(960, 540), every=1):
    d = json.load(open(path))
    name = d["object"]
    cell = S.Cell(seed=d["seed"], objects=(name,))
    m, data = cell.m, cell.d
    sl = [m.joint(n).qposadr[0] for n in ("slide_x", "slide_y", "slide_z")]
    fj = cell.jadr
    oj = m.joint(f"{name}_free").qposadr[0]
    r = mujoco.Renderer(m, size[1], size[0])
    out, ts = [], []
    for i, row in enumerate(d["rows"]):
        if i % every:
            continue
        data.qpos[sl] = np.array(row[1:4]) - cell.home
        data.qpos[fj] = row[4:16]
        data.qpos[oj:oj + 3] = row[16:19]
        data.qpos[oj + 3:oj + 7] = row[19:23]
        mujoco.mj_forward(m, data)
        r.update_scene(data, camera=cam)
        out.append(r.render().copy())
        ts.append(row[0])
    r.close()
    return out, ts, d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("replay")
    ap.add_argument("--cam", default="close")
    ap.add_argument("--out", default="media/physx_replay.mp4")
    a = ap.parse_args()
    fr, _, _ = frames(a.replay, a.cam)
    S.save_video(fr, a.out, fps=50)
    print(a.out, len(fr))
