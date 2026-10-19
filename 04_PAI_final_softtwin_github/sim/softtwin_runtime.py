"""SoftTwin runtime inside Isaac Sim: actuator model -> finger targets, pick-and-place episodes, metrics.

Usage from MCP (after softtwin_scene.py built the cell):
    isaac_run_file('sim/softtwin_runtime.py')                    # defines helpers in the shared namespace
    isaac_exec("await run_episode('strawberry', mode='closed_loop'); _result = LAST")
    isaac_exec("await benchmark(seeds=range(20)); _result = SUMMARY")

Physics/control model
  pressure p [kPa] --actuator model--> free bend angle per joint alpha(p) [deg]
  joint drive target = alpha(p), stiffness = silicone bending stiffness (PRBM)
  grip-force estimate (proprioceptive, no extra sensor):
      tau_j = k_b * (alpha_target - alpha_measured)      [N*m]
      F_finger ~= sum_j tau_j / lever_j                    [N]
  closed loop: inflate until F_est reaches the target force chosen for the object class, then hold.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import time

import numpy as np
import omni.kit.app
import omni.timeline
import omni.usd
from pxr import Gf, Sdf, UsdGeom

REPO = os.environ.get("PAI_REPO") or os.path.dirname(os.path.dirname(os.path.abspath(globals().get("__file__", "."))))
RESULTS = os.path.join(REPO, "results")
os.makedirs(RESULTS, exist_ok=True)

# ---------------------------------------------------------------- actuator models
class LinearActuator:
    """Datasheet-style baseline: alpha = c * p (what you'd assume without identification)."""

    def __init__(self, c=None, p_max=80.0):
        if c is None:  # best linear fit stored by learning/actuator_net.py (total bend per kPa)
            try:
                c = json.load(open(os.path.join(REPO, "learning", "actuator_net.json"), encoding="utf-8"))["linear_c"]
            except Exception:
                c = 1.5
        self.c, self.p_max = c, p_max

    def reset(self):
        pass

    def __call__(self, p, dt):
        return self.c * max(0.0, min(p, self.p_max)) / 4.0  # per joint (4 joints share the bend)


class LearnedActuator:
    """Small MLP identified from logged (pressure, bend) data (learning/actuator_net.py -> learning/actuator_net.json)."""

    def __init__(self, path=os.path.join(REPO, "learning", "actuator_net.json")):
        d = json.load(open(path, encoding="utf-8"))
        self.W = [np.array(w) for w in d["W"]]
        self.b = [np.array(b) for b in d["b"]]
        self.xm, self.xs = np.array(d["x_mean"]), np.array(d["x_std"])
        self.ym, self.ys = d["y_mean"], d["y_std"]
        self.n_joints = d.get("n_joints", 4)
        self.reset()

    def reset(self):
        self.prev_p, self.prev_a = 0.0, 0.0

    def __call__(self, p, dt):
        dp = (p - self.prev_p) / max(dt, 1e-3)
        x = (np.array([p, dp, self.prev_a]) - self.xm) / self.xs
        h = x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < len(self.W) - 1:
                h = np.tanh(h)
        a = float(h[0] * self.ys + self.ym)
        self.prev_p, self.prev_a = p, a
        return a / self.n_joints


# ---------------------------------------------------------------- scene access
def scene_info():
    st = omni.usd.get_context().get_stage()
    return json.loads(st.GetPrimAtPath("/World").GetAttribute("softtwin:scene_info").Get())


def _frames():
    return omni.kit.app.get_app().next_update_async()


class Twin:
    """Thin wrapper over omni.physics.tensors (stable low-level API under isaacsim.core)."""

    def __init__(self):
        import omni.physics.tensors as tensors

        self.info = scene_info()
        self.view = tensors.create_simulation_view("numpy")
        self.view.set_subspace_roots("/")
        self.art = self.view.create_articulation_view("/World/Gripper/root")
        self.dof_names = list(self.art.shared_metatype.dof_names)
        self.idx = {n: i for i, n in enumerate(self.dof_names)}
        self.finger_idx = [i for n, i in self.idx.items() if n.startswith("f")]
        self.obj = {}
        for name, m in self.info["objects"].items():
            self.obj[name] = self.view.create_rigid_body_view(m["path"])
        self.cfg = self.info["cfg"]
        self.home = np.array(self.info["cfg"]["gantry_home"])

    def q(self):
        return self.art.get_dof_positions()[0].copy()

    def set_targets(self, xyz_world=None, bend_deg=None):
        tgt = self.art.get_dof_position_targets()[0].copy()
        if xyz_world is not None:
            d = np.array(xyz_world) - self.home
            for k, n in zip(range(3), ("slide_x", "slide_y", "slide_z")):
                tgt[self.idx[n]] = d[k]
        if bend_deg is not None:
            for i in self.finger_idx:
                tgt[i] = math.radians(bend_deg)  # tensors API uses radians
        self.art.set_dof_position_targets(tgt[None, :], np.array([0], dtype=np.int32))

    def palm_xyz(self):
        q = self.q()
        return self.home + np.array([q[self.idx["slide_x"]], q[self.idx["slide_y"]], q[self.idx["slide_z"]]])

    def finger_bend_deg(self):
        q = self.q()
        return np.degrees(q[self.finger_idx])

    def grip_force_est(self, bend_target_deg):
        """Proprioceptive estimate (same as sim_mujoco): joint torque not explained by the free bend,
        least-squares fit of a fingertip force per finger (lever l_j = (n - j) * L), summed over fingers [N]."""
        kb = self.cfg["bend_stiffness"] * 180.0 / math.pi  # N*m/deg -> N*m/rad
        L, n, nf = self.cfg["seg_len"], self.cfg["n_segments"], self.cfg["n_fingers"]
        q = np.radians(self.finger_bend_deg().reshape(nf, n))
        ref = math.radians(bend_target_deg)
        tau = kb * (ref - q)
        lever = np.array([(n - k) * L for k in range(n)])
        per_finger = np.clip(tau @ lever / (lever @ lever), 0, None)
        return float(per_finger.sum()), per_finger.tolist()

    def obj_pos(self, name):
        tr = self.obj[name].get_transforms()[0]
        return tr[:3].copy()


def mark_crushed(name, crushed=True):
    st = omni.usd.get_context().get_stage()
    g = st.GetPrimAtPath(f"/World/Objects/{name}/geom")
    if g and crushed:
        UsdGeom.Gprim(g).GetDisplayColorAttr().Set([Gf.Vec3f(0.25, 0.05, 0.05)])


# ---------------------------------------------------------------- episodes
TARGET_FORCE = {"very_fragile": 0.8, "fragile": 1.6, "rigid": 4.0}  # N, chosen per class (VLM/planner output)
LAST, SUMMARY = None, None
_TWIN = None


def twin():
    global _TWIN
    if _TWIN is None:
        _TWIN = Twin()
    return _TWIN


async def _move(tw, xyz, bend, steps, log, t0, p=None):
    for _ in range(steps):
        tw.set_targets(xyz, bend)
        await _frames()
        if log is not None:
            f, _ = tw.grip_force_est(bend)
            log.append({"t": round(time.time() - t0, 3), "palm": tw.palm_xyz().round(4).tolist(), "bend_cmd": bend, "bend": tw.finger_bend_deg().mean().round(2).item(), "force_est": round(f, 3), "p": p})


async def run_episode(name, mode="closed_loop", actuator="learned", p_open=60.0, dt=1 / 60, record=True):
    """Pick object `name` and place it in the tray.
    mode: 'open_loop' (inflate to p_open regardless) | 'closed_loop' (stop at class target force)."""
    global LAST
    tl = omni.timeline.get_timeline_interface()
    if not tl.is_playing():
        tl.play()
        for _ in range(10):
            await _frames()
    tw = twin()
    info = tw.info
    m = info["objects"][name]
    act = LearnedActuator() if actuator == "learned" else LinearActuator()
    act.reset()
    t0 = time.time()
    log = []
    top = info["cfg"]["table_top"]
    obj0 = tw.obj_pos(name)
    above = [obj0[0], obj0[1], top + 0.20]
    grasp_z = obj0[2] + 0.055
    # approach
    await _move(tw, above, 0.0, 90, log, t0, 0.0)
    await _move(tw, [obj0[0], obj0[1], grasp_z], 0.0, 90, log, t0, 0.0)
    # inflate
    p, f_target = 0.0, TARGET_FORCE[m["class"]]
    peak_f, bend, hold = 0.0, 0.0, 0
    for _ in range(360):
        f, _ = tw.grip_force_est(bend)
        if mode == "open_loop":
            p = min(p + 0.4, p_open)
        elif f < f_target:
            p = min(p + 0.4, 80.0)  # inflate until the class target force is reached, then hold
        else:
            hold += 1
        bend = act(p, dt)
        tw.set_targets(None, bend)
        await _frames()
        f, _ = tw.grip_force_est(bend)
        peak_f = max(peak_f, f)
        log.append({"t": round(time.time() - t0, 3), "phase": "inflate", "p": round(p, 2), "bend_cmd": round(bend, 2), "bend": tw.finger_bend_deg().mean().round(2).item(), "force_est": round(f, 3)})
        if (mode == "closed_loop" and hold > 30) or (mode == "open_loop" and p >= p_open and len(log) > 300):
            break
    crushed = peak_f > m["max_force"]
    if crushed:
        mark_crushed(name)
    # lift and transport
    tray = info["tray"]
    await _move(tw, [obj0[0], obj0[1], top + 0.22], bend, 120, log, t0, p)
    lifted = tw.obj_pos(name)[2] > top + 0.08
    await _move(tw, [tray[0], tray[1], top + 0.22], bend, 150, log, t0, p)
    await _move(tw, [tray[0], tray[1], top + 0.12], bend, 60, log, t0, p)
    await _move(tw, [tray[0], tray[1], top + 0.12], 0.0, 90, log, t0, 0.0)
    pos = tw.obj_pos(name)
    in_tray = abs(pos[0] - tray[0]) < 0.09 and abs(pos[1] - tray[1]) < 0.07 and pos[2] < top + 0.06
    await _move(tw, above, 0.0, 60, None, t0)
    LAST = {
        "object": name, "class": m["class"], "mode": mode, "actuator": actuator, "lifted": bool(lifted), "in_tray": bool(in_tray),
        "peak_force_est": round(peak_f, 3), "max_safe_force": m["max_force"], "crushed": bool(crushed),
        "success": bool(lifted and in_tray and not crushed), "final_pressure": round(p, 2), "steps": len(log),
    }
    if record:
        with open(os.path.join(RESULTS, f"episode_{name}_{mode}_{actuator}.json"), "w", encoding="utf-8") as fp:
            json.dump({"summary": LAST, "log": log}, fp)
    return LAST


async def reset_scene(seed):
    """Rebuild the cell with a new seed (object layout) and restart physics."""
    global _TWIN
    tl = omni.timeline.get_timeline_interface()
    tl.stop()
    await _frames()
    ns = {"__name__": "softtwin_scene_import"}
    exec(open(os.path.join(REPO, "sim", "softtwin_scene.py"), encoding="utf-8").read(), ns)
    ns["build"](seed=seed)
    _TWIN = None
    tl.play()
    for _ in range(30):
        await _frames()


async def benchmark(seeds=range(10), objects=("strawberry", "egg", "tofu", "tomato", "bolt"), modes=("open_loop", "closed_loop"), actuator="learned"):
    """Fixed-seed closed-loop evaluation (lecture 2: judge by success rate, not loss)."""
    global SUMMARY
    rows = []
    for s in seeds:
        for mode in modes:
            await reset_scene(s)
            for o in objects:
                r = await run_episode(o, mode=mode, actuator=actuator, record=False)
                r["seed"] = s
                rows.append(r)
    out = {}
    for mode in modes:
        rs = [r for r in rows if r["mode"] == mode]
        out[mode] = {
            "n": len(rs), "success_rate": round(sum(r["success"] for r in rs) / max(1, len(rs)), 3),
            "crush_rate": round(sum(r["crushed"] for r in rs) / max(1, len(rs)), 3),
            "drop_rate": round(sum((not r["lifted"]) for r in rs) / max(1, len(rs)), 3),
            "by_object": {o: round(sum(r["success"] for r in rs if r["object"] == o) / max(1, sum(1 for r in rs if r["object"] == o)), 3) for o in objects},
        }
    SUMMARY = {"seeds": list(seeds), "actuator": actuator, "results": out}
    with open(os.path.join(RESULTS, "benchmark.json"), "w", encoding="utf-8") as fp:
        json.dump({"summary": SUMMARY, "rows": rows}, fp, indent=1)
    return SUMMARY


print("softtwin_runtime loaded: run_episode(), benchmark(), reset_scene(), twin()")
_result = {"repo": REPO, "results": RESULTS}
