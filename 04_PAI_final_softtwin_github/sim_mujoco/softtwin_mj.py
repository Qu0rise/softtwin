"""SoftTwin cell in MuJoCo (CPU): same PRBM soft gripper, actuator model, controller and metrics as sim/softtwin_runtime.py.

Used (1) as a second physics engine for cross-simulator validation of the Isaac Sim results and
(2) to produce results/video when no GPU is available.  MuJoCo additionally gives ground-truth
contact forces, which we use to validate the proprioceptive grip-force estimate.

  python sim_mujoco/softtwin_mj.py episode strawberry --mode closed_loop --video media/mj_strawberry.mp4
  python sim_mujoco/softtwin_mj.py bench --seeds 10
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "osmesa" if sys.platform.startswith("linux") else "glfw")
import mujoco  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "learning"))
from actuator_net import SyntheticMuscle  # noqa: E402

# ------------------------------------------------------------------ configuration (shared with Isaac Sim cell)
CFG = dict(table_top=0.75, home=(0.0, 0.0, 1.00), n_fingers=3, n_seg=4, seg_len=0.026, seg_w=0.020, seg_t=0.012,
           finger_r=0.045, palm=(0.06, 0.06, 0.01), k_b=0.5, d_b=0.006, tray=(0.30, 0.20))
# name: shape, size (MuJoCo half sizes / radius), mass [kg], rgba, max safe force [N] (assumed), class
OBJECTS = {
    "strawberry": ("ellipsoid", (0.022, 0.017, 0.017), 0.018, (0.85, 0.08, 0.12, 1), 3.0, "fragile"),
    "egg": ("ellipsoid", (0.028, 0.021, 0.021), 0.060, (0.95, 0.90, 0.80, 1), 25.0, "fragile"),
    "tofu": ("box", (0.022, 0.022, 0.016), 0.035, (0.97, 0.97, 0.92, 1), 1.2, "very_fragile"),
    "tomato": ("sphere", (0.027,), 0.090, (0.90, 0.25, 0.10, 1), 10.0, "fragile"),
    "bolt": ("cylinder", (0.008, 0.026), 0.035, (0.55, 0.57, 0.60, 1), 1e9, "rigid"),
}
# look-alike variants: same appearance as a base object, different hidden physics (what a camera / VLM cannot see)
OBJECTS["tomato_overripe"] = ("sphere", (0.027,), 0.090, (0.90, 0.25, 0.10, 1), 1.5, "soft")
OBJECTS["sample_strawberry"] = ("ellipsoid", (0.022, 0.017, 0.017), 0.040, (0.85, 0.08, 0.12, 1), 1e9, "rigid")  # resin food sample
BASE_OBJECTS = ("strawberry", "egg", "tofu", "tomato", "bolt")
VISUAL_ID = {"tomato_overripe": "tomato", "sample_strawberry": "strawberry"}
# hidden contact physics (not visible to the camera): contact time constant tc [s] (larger = softer surface,
# MuJoCo solref "tc 1"), Coulomb friction mu. Look-alikes share the appearance but not these values.
PHYS = {  # tc: contact time constant [s]; dmin/width: MuJoCo solimp (low dmin + wide width = the surface gives way = soft flesh)
    "strawberry": dict(tc=0.030, dmin=0.50, width=0.003, mu=1.0), "egg": dict(tc=0.008, dmin=0.90, width=0.001, mu=0.9),
    "tofu": dict(tc=0.040, dmin=0.50, width=0.003, mu=1.0), "tomato": dict(tc=0.020, dmin=0.60, width=0.002, mu=1.0),
    "bolt": dict(tc=0.004, dmin=0.95, width=0.001, mu=1.0),
    "tomato_overripe": dict(tc=0.045, dmin=0.30, width=0.006, mu=1.0), "sample_strawberry": dict(tc=0.004, dmin=0.95, width=0.001, mu=0.6),
}
TARGET_FORCE = {"very_fragile": 0.7, "soft": 1.3, "fragile": 1.8, "rigid": 3.5}  # N, chosen per class by the planner
VISUAL_CLASS = {"strawberry": "fragile", "egg": "fragile", "tofu": "very_fragile", "tomato": "fragile", "bolt": "rigid"}
SPOTS = [(-0.28, -0.16), (-0.12, -0.18), (0.04, -0.16), (-0.24, 0.06), (-0.06, 0.08), (0.10, -0.02)]


def obj_half_height(shape, size):
    return size[0] if shape == "sphere" else (size[1] if shape == "cylinder" else size[2])


def quat_z(deg):
    h = math.radians(deg) / 2
    return f"{math.cos(h):.6f} 0 0 {math.sin(h):.6f}"


def build_xml(seed=0, objects=BASE_OBJECTS):
    rng = np.random.default_rng(seed)
    top = CFG["table_top"]
    hx, hy, hz = CFG["home"]
    spots = [SPOTS[i] for i in rng.permutation(len(SPOTS))]
    fingers = []
    for f in range(CFG["n_fingers"]):
        th = 90 + f * 360 / CFG["n_fingers"]
        r = CFG["finger_r"]
        L, w, t = CFG["seg_len"], CFG["seg_w"], CFG["seg_t"]
        x, y, z = r * math.cos(math.radians(th)), r * math.sin(math.radians(th)), -CFG["palm"][2]
        body = ""
        for k in reversed(range(CFG["n_seg"])):
            col = f"{0.95:.2f} {0.45 + 0.08 * k:.2f} {0.20:.2f} 1"
            inner = body
            body = (f'<body name="f{f}s{k}" pos="0 0 {-L if k else 0}">'
                    f'<joint name="f{f}j{k}" type="hinge" axis="0 1 0" range="-10 80" damping="{CFG["d_b"]}" armature="0.0002"/>'
                    f'<geom type="box" size="{t/2} {w/2} {L*0.48}" pos="0 0 {-L/2}" rgba="{col}" mass="0.006" friction="1.6 0.02 0.002" condim="4"/>'
                    + "".join(f'<site name="tx_f{f}s{k}_{m}" type="box" size="0.008 {w/2*0.98:.4f} {L/4*0.99:.4f}" pos="{-t/2-0.002:.4f} 0 {-L*(0.25+0.5*m):.4f}" rgba="0.1 0.8 0.9 0.0"/>'
                              f'<geom name="pad_f{f}s{k}_{m}" type="box" size="0.0007 {w/2*0.92:.4f} {L/4*0.92:.4f}" pos="{-t/2-0.0007:.4f} 0 {-L*(0.25+0.5*m):.4f}" rgba="0.20 0.22 0.30 1" contype="0" conaffinity="0" mass="0"/>' for m in range(2))
                    + (f'<geom type="capsule" size="0.004 {w/2*0.8}" pos="{-t/2-0.002} 0 {-L+0.004}" euler="90 0 0" rgba="{col}" mass="0.001" friction="1.6 0.02 0.002" condim="4"/>' if k == CFG["n_seg"] - 1 else "")
                    + f'{inner}</body>')
        fingers.append(f'<body name="f{f}" pos="{x:.4f} {y:.4f} {z:.4f}" quat="{quat_z(th)}">{body}</body>')
    acts = "".join(f'<position name="a_f{f}j{k}" joint="f{f}j{k}" kp="{CFG["k_b"]}" ctrlrange="-0.2 1.4"/>'
                   for f in range(CFG["n_fingers"]) for k in range(CFG["n_seg"]))
    objs = []
    for i, name in enumerate(objects):
        shape, size, mass, rgba, fmax, cls = OBJECTS[name]
        ph = PHYS[name]
        mass = mass * rng.uniform(0.75, 1.25)          # domain randomisation: mass +-25 %
        fr = ph["mu"] * rng.uniform(0.8, 1.2)          # friction +-20 %
        tc = ph["tc"] * rng.uniform(0.85, 1.15)        # surface compliance +-15 %
        sx, sy = spots[i]
        sx += rng.uniform(-0.015, 0.015)
        sy += rng.uniform(-0.015, 0.015)
        h = obj_half_height(shape, size)
        sz = " ".join(str(s) for s in size)
        objs.append(f'<body name="{name}" pos="{sx:.4f} {sy:.4f} {top + h + 0.001:.4f}"><freejoint name="{name}_free"/>'
                    f'<geom name="{name}_g" type="{shape}" size="{sz}" mass="{mass:.4f}" rgba="{" ".join(map(str, rgba))}" friction="{fr:.3f} 0.02 0.002" condim="4" priority="1" solref="{tc:.4f} 1" solimp="{min(0.95, ph['dmin'] * rng.uniform(0.85, 1.15)):.3f} 0.95 {ph['width'] * rng.uniform(0.85, 1.15):.4f}"/></body>')
    tx, ty = CFG["tray"]
    walls = "".join(f'<geom type="box" size="{a}" pos="{b}" rgba="0.15 0.35 0.65 1"/>' for a, b in [
        ("0.09 0.07 0.005", f"{tx} {ty} {top+0.005}"), ("0.005 0.07 0.02", f"{tx-0.09} {ty} {top+0.02}"),
        ("0.005 0.07 0.02", f"{tx+0.09} {ty} {top+0.02}"), ("0.09 0.005 0.02", f"{tx} {ty-0.07} {top+0.02}"),
        ("0.09 0.005 0.02", f"{tx} {ty+0.07} {top+0.02}")])
    return f"""<mujoco model="softtwin">
<option timestep="0.001" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic" impratio="10" noslip_iterations="5"/>
<visual><global offwidth="1280" offheight="720"/><quality shadowsize="4096"/><headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5"/></visual>
<asset><texture type="skybox" builtin="gradient" rgb1="0.82 0.86 0.92" rgb2="0.35 0.38 0.45" width="512" height="512"/><texture name="grid" type="2d" builtin="checker" rgb1="0.30 0.31 0.33" rgb2="0.26 0.27 0.29" width="512" height="512"/>
<material name="grid" texture="grid" texrepeat="8 8" reflectance="0.05"/>
<material name="wood" rgba="0.62 0.50 0.38 1"/></asset>
<worldbody>
<light pos="0.6 -0.8 2.2" dir="-0.3 0.4 -1" diffuse="0.8 0.8 0.8" castshadow="true"/>
<geom type="plane" size="3 3 0.1" material="grid"/>
<geom type="box" size="0.5 0.35 0.02" pos="0 0 {top-0.02}" material="wood" friction="0.8 0.02 0.002"/>
{''.join(f'<geom type="box" size="0.02 0.02 {(top-0.04)/2}" pos="{a} {b} {(top-0.04)/2}" rgba="0.35 0.28 0.22 1"/>' for a,b in [(-0.46,-0.31),(0.46,-0.31),(-0.46,0.31),(0.46,0.31)])}
{walls}
<geom type="cylinder" size="0.01 0.6" pos="-0.55 0 0.6" rgba="0.6 0.6 0.62 1" contype="0" conaffinity="0"/>
<body name="palm" pos="{hx} {hy} {hz}" gravcomp="1">
  <joint name="slide_x" type="slide" axis="1 0 0" range="-0.45 0.45" damping="40"/>
  <joint name="slide_y" type="slide" axis="0 1 0" range="-0.35 0.35" damping="40"/>
  <joint name="slide_z" type="slide" axis="0 0 1" range="-0.3 0.2" damping="40"/>
  <inertial pos="0 0 0" mass="0.5" diaginertia="0.002 0.002 0.002"/>
  <geom type="box" size="{CFG['palm'][0]} {CFG['palm'][1]} {CFG['palm'][2]}" rgba="0.85 0.85 0.88 1" contype="0" conaffinity="0" mass="0"/>
  <geom type="cylinder" size="0.012 0.12" pos="0 0 0.13" rgba="0.6 0.6 0.62 1" contype="0" conaffinity="0" mass="0"/>
  <camera name="wrist" pos="0 0 -0.012" xyaxes="1 0 0 0 1 0" fovy="70"/>
  {''.join(fingers)}
</body>
{''.join(objs)}
<camera name="hero" pos="0.75 -0.75 1.22" xyaxes="0.707 0.707 0 -0.28 0.28 0.92"/>
<camera name="close" pos="0.22 -0.30 0.92" xyaxes="0.81 0.59 0 -0.25 0.34 0.91"/>
<camera name="overhead" pos="0 0 1.9" xyaxes="1 0 0 0 1 0"/>
</worldbody>
<sensor>{"".join(f'<touch name="tx_f{f}s{k}_{m}" site="tx_f{f}s{k}_{m}"/>' for f in range(CFG["n_fingers"]) for k in range(CFG["n_seg"]) for m in range(2))}</sensor>
<actuator>{acts}<position name="a_x" joint="slide_x" kp="4000" kv="200"/><position name="a_y" joint="slide_y" kp="4000" kv="200"/><position name="a_z" joint="slide_z" kp="6000" kv="250"/></actuator>
</mujoco>"""


# ------------------------------------------------------------------ models
class Actuator:
    """pressure [kPa] -> free bend per finger [deg] (learned MLP or best linear fit), see learning/actuator_net.py"""

    def __init__(self, kind="learned"):
        d = json.load(open(REPO / "learning" / "actuator_net.json"))
        self.kind = kind
        self.W, self.b = [np.array(w) for w in d["W"]], [np.array(b) for b in d["b"]]
        self.xm, self.xs, self.ym, self.ys, self.c = np.array(d["x_mean"]), np.array(d["x_std"]), d["y_mean"], d["y_std"], d["linear_c"]
        self.reset()

    def reset(self):
        self.prev_p, self.prev_a = 0.0, 0.0

    def __call__(self, p, dt):
        if self.kind == "linear":
            return self.c * p
        h = (np.array([p, (p - self.prev_p) / dt, self.prev_a]) - self.xm) / self.xs
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < len(self.W) - 1:
                h = np.tanh(h)
        a = float(h[0] * self.ys + self.ym)
        self.prev_p, self.prev_a = p, a
        return max(0.0, a)


class Supply:
    """pump + valve: first-order pressure response (tau [s]), rate-limited by the pump"""

    def __init__(self, tau=0.25):
        self.p, self.tau = 0.0, tau

    def step(self, cmd, dt):
        self.p += (cmd - self.p) * (1 - math.exp(-dt / self.tau))
        return self.p


@dataclass
class Episode:
    name: str
    mode: str
    actuator: str
    log: list = field(default_factory=list)
    retries_left: int = 0


class Cell:
    def __init__(self, seed=0, objects=BASE_OBJECTS, render=None, cam="hero", size=(960, 540)):
        self.m = mujoco.MjModel.from_xml_string(build_xml(seed, objects))
        self.d = mujoco.MjData(self.m)
        mujoco.mj_forward(self.m, self.d)
        self.objects = objects
        self.home = np.array(CFG["home"])
        self.ref_hat = None
        self.palm_target = self.home.copy()
        self.jids = [self.m.joint(f"f{f}j{k}").qposadr[0] for f in range(CFG["n_fingers"]) for k in range(CFG["n_seg"])]
        self.finger_geoms = {self.m.body(f"f{f}s{k}").id for f in range(CFG["n_fingers"]) for k in range(CFG["n_seg"])}
        self.frames, self.renderer, self.cam = [], None, cam
        if render:
            self.renderer = mujoco.Renderer(self.m, size[1], size[0])
            self.render_every = int(round(1 / 30 / self.m.opt.timestep))
        self.t_step = 0
        for _ in range(300):  # settle objects
            self._step()

    # -------- low level
    def _step(self):
        mujoco.mj_step(self.m, self.d)
        self.t_step += 1
        if self.renderer is not None and self.t_step % self.render_every == 0:
            self.renderer.update_scene(self.d, camera=self.cam)
            self.frames.append(self.renderer.render().copy())

    def taxels(self):
        """24 taxels (3 fingers x 4 segments x 2): normal force [N] on the inner silicone surface (touch sensors)"""
        return self.d.sensordata[: CFG["n_fingers"] * CFG["n_seg"] * 2].copy()

    def skin_read(self):
        """tactile skin output: raw touch taxels, or the fabric-skin sensor model if one is attached (tactile.py)"""
        return self.skin.read() if getattr(self, "skin", None) is not None else self.taxels()

    def on_frame(self, tag, ep):
        """camera hook (dataset generator overrides via cell.frame_cb)"""
        cb = getattr(self, "frame_cb", None)
        if cb is not None:
            cb(self, tag, ep)

    def obj_pos(self, name):
        return self.d.body(name).xpos.copy()

    def set_bend(self, bend_deg_per_finger):
        tgt = math.radians(bend_deg_per_finger / CFG["n_seg"])
        self.d.ctrl[: len(self.jids)] = tgt

    def q(self):
        return np.array([self.d.qpos[a] for a in self.jids]).reshape(CFG["n_fingers"], CFG["n_seg"])

    def qdot(self):
        return np.array([self.d.qvel[self.m.joint(f"f{f}j{k}").dofadr[0]] for f in range(CFG["n_fingers"]) for k in range(CFG["n_seg"])]).reshape(CFG["n_fingers"], CFG["n_seg"])

    def force_est(self, compensate=True):
        """proprioceptive grip-force estimate (N): joint torques not explained by free motion, divided by lever arm.
        tau_contact ~= k_b*(theta_ref - q) - d_b*qdot   (quasi-static; compensate=False drops the damping term)"""
        L, n = CFG["seg_len"], CFG["n_seg"]
        ref = self.d.ctrl[: len(self.jids)].reshape(CFG["n_fingers"], n)
        if getattr(self, "ref_hat", None) is not None:  # estimator only knows the model's free-bend prediction
            ref = np.full_like(ref, self.ref_hat)
        tau = CFG["k_b"] * (ref - self.q())
        if compensate:
            tau = tau - CFG["d_b"] * self.qdot()
        # contact assumed at the fingertip: tau_j = F * l_j with l_j = (n - j) * L  ->  least squares per finger
        lever = np.array([(n - k) * L for k in range(n)])
        per_finger = np.clip(tau @ lever / (lever @ lever), 0, None)
        return float(per_finger.sum())

    def force_true(self, name):
        """ground truth: sum of contact normal forces between finger geoms and the object (N)"""
        gid = self.m.geom(f"{name}_g").id
        tot = 0.0
        f6 = np.zeros(6)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            bodies = {self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]}
            if gid in (c.geom1, c.geom2) and bodies & self.finger_geoms:
                mujoco.mj_contactForce(self.m, self.d, i, f6)
                tot += abs(f6[0])
        return tot

    def move_palm(self, target, seconds, bend=None, ep=None, phase=""):
        start = self.palm_target.copy()
        n = max(1, int(seconds / self.m.opt.timestep))
        for i in range(n):
            a = (i + 1) / n
            s = a * a * (3 - 2 * a)
            self.palm_target = start + s * (np.array(target) - start)
            self.d.ctrl[len(self.jids):len(self.jids) + 3] = self.palm_target - self.home
            if callable(bend):
                if i % 20 == 0:
                    self._cur_bend = bend()
                self.set_bend(self._cur_bend)
            elif bend is not None:
                self.set_bend(bend)
            self._step()
            if ep is not None and i % 20 == 0:
                ep.log.append(dict(t=round(self.d.time, 3), phase=phase, f_est=round(self.force_est(), 3), f_true=round(self.force_true(ep.name), 3)))


def _attempt(cell: Cell, name: str, mode="closed_loop", actuator="learned", p_fixed=60.0, F_target=None, ep=None, plant="synthetic", slot=(0.0, 0.0),
             probe_fn=None, F_probe=0.4):
    """approach -> descend -> inflate (open loop to p_fixed, or closed loop to F*) -> lift -> transport -> release."""
    ep = ep or Episode(name, mode, actuator)
    top = CFG["table_top"]
    cls = OBJECTS[name][5]
    F_star = TARGET_FORCE[cls] if F_target is None else F_target
    dt = 0.02
    act, sup = Actuator(actuator), Supply()
    # plant: "synthetic" = the stand-in real muscle (hysteresis, lag, dead band) drives the fingers, the controller only sees the model;
    #        "model" = the model itself drives the fingers (pure twin)
    muscle = SyntheticMuscle(seed=hash(name) % 1000, noise=0.0) if plant == "synthetic" else None

    def actuate(p):
        b_hat = act(p, dt)
        cell.ref_hat = math.radians(b_hat / CFG["n_seg"]) if muscle is not None else None
        return muscle.step(p, dt) if muscle is not None else b_hat
    o = cell.obj_pos(name)
    hz = CFG["home"][2]
    shape, size = OBJECTS[name][0], OBJECTS[name][1]
    h = obj_half_height(shape, size)
    finger_len = CFG["n_seg"] * CFG["seg_len"]
    # fingertips end ~40 % of the object's half height below its centre (straight fingers)
    grasp_z = o[2] - 0.4 * h + 2 * CFG["palm"][2] + finger_len
    cell.move_palm([o[0], o[1], top + 0.22], 1.0, 0.0, ep, "approach")
    o = cell.obj_pos(name)  # re-measure (perception update) right before descending
    # fingertips go below the object's equator (cradle grasp) but stay 4 mm above the table
    grasp_z = max(top + 0.004, o[2] - 0.9 * h) + 2 * CFG["palm"][2] + finger_len
    cell.move_palm([o[0], o[1], top + 0.22], 0.3, 0.0, ep, "align")
    cell.on_frame("pregrasp", ep)  # wrist camera looks straight down at the object
    cell.move_palm([o[0], o[1], grasp_z], 0.8, 0.0, ep, "descend")
    dt = 0.02
    sub = int(dt / cell.m.opt.timestep)
    p_cmd, peak_true, peak_est, hold = 0.0, 0.0, 0.0, 0
    bend, contact, p = 0.0, False, 0.0
    # haptic probing: squeeze gently to F_probe first, read the skin + proprioception, then let probe_fn choose F*
    probing = mode == "closed_loop" and probe_fn is not None
    F_final, F_star = F_star, (F_probe if probing else F_star)
    for k in range(int((8.0 if probing else 5.0) / dt)):
        f_est = cell.force_est()
        if mode == "open_loop":
            p_cmd = p_fixed
        else:
            # force controller: fast approach until first contact, then proportional regulation of p_cmd on the force error
            e = F_star - f_est
            # first contact: proprioception (estimate above 15 % of F*) or, while probing, the tactile skin (model-error free)
            touched = (cell.skin_read().sum() > 0.05 or f_est > 0.25) if probing else f_est >= 0.15 * F_star
            if not touched and not contact:
                dp = 12.0                                   # free closing: fast
            else:
                if not contact:                             # first contact: drop the command to the measured pressure (kills lag overshoot)
                    contact, p_cmd = True, p
                dp = float(np.clip(4.0 * e, -15.0, 3.0))    # then slow proportional force regulation
            p_cmd = float(np.clip(p_cmd + dp * dt, 0.0, 80.0))
            hold = hold + 1 if abs(e) < 0.15 * F_star else 0
            if probing and hold > 10:
                F_star = float(probe_fn(cell, ep))
                ep.F_star_chosen, probing, hold = F_star, False, 0
        p = sup.step(p_cmd, dt)
        bend = actuate(p)
        for _ in range(sub):
            cell.set_bend(bend)
            cell._step()
        ft = cell.force_true(name)
        peak_true, peak_est = max(peak_true, ft), max(peak_est, cell.force_est())
        ep.log.append(dict(t=round(cell.d.time, 3), phase="inflate", p_cmd=round(p_cmd, 2), p=round(p, 2), bend=round(bend, 2),
                           f_est=round(cell.force_est(), 3), f_est_naive=round(cell.force_est(False), 3), f_true=round(ft, 3),
                           q_mean=round(float(np.degrees(cell.q().sum(1).mean())), 3), q_f=np.round(np.degrees(cell.q().sum(1)), 3).tolist(), tax=np.round(cell.skin_read(), 4).tolist(), tax_true=np.round(cell.taxels(), 4).tolist(), probing=probing))
        if mode == "closed_loop" and hold > 20:
            break
        if mode == "open_loop" and k * dt > 2.0:
            break
    cell.on_frame("grasp", ep)
    tx, ty = CFG["tray"][0] + slot[0], CFG["tray"][1] + slot[1]
    # place so that the object's bottom stops ~12 mm above the tray floor (object hangs (o_z - grasp_z) below the palm)
    place_z = (top + 0.010 + 0.012 + h) - (o[2] - grasp_z)
    state = {"p_cmd": p_cmd}

    def hold_ctrl():
        """keep regulating the grip while the object is carried (load changes with acceleration); open loop keeps p fixed"""
        if mode == "closed_loop":
            e = F_star - cell.force_est()
            state["p_cmd"] = float(np.clip(state["p_cmd"] + np.clip(4.0 * e, -15.0, 3.0) * dt, 0.0, 80.0))
        return actuate(sup.step(state["p_cmd"], dt))

    cell._cur_bend = bend
    for tgt, sec, ph in [([o[0], o[1], top + 0.20], 1.0, "lift"), ([tx, ty, top + 0.20], 1.4, "transport"), ([tx, ty, place_z], 0.8, "lower")]:
        cell.move_palm(tgt, sec, hold_ctrl, ep, ph)
        for e in ep.log[-5:]:
            peak_true = max(peak_true, e.get("f_true", 0))
        if ph == "lift":
            cell.on_frame("lift", ep)
            lifted = cell.obj_pos(name)[2] > top + 0.08
            if not lifted and mode == "closed_loop" and ep.retries_left > 0:
                # task-level recovery: the object stayed on the table (slip) -> release and regrasp with a higher target force
                cell.move_palm([o[0], o[1], top + 0.20], 0.4, 0.0, ep, "regrasp")
                return None, ep
    sup_release = 0.0
    cell.ref_hat = None
    cell.move_palm([tx, ty, place_z], 0.8, 0.0, ep, "release")
    cell.move_palm([tx, ty, top + 0.22], 0.6, 0.0, ep, "retract")
    pos = cell.obj_pos(name)
    in_tray = abs(pos[0] - CFG["tray"][0]) < 0.09 and abs(pos[1] - CFG["tray"][1]) < 0.07 and pos[2] < top + 0.07
    fmax = OBJECTS[name][4]
    # damage criterion: peak normal force while the object is held (20 ms moving average to drop single-step impulses)
    held = np.array([e["f_true"] for e in ep.log if e["phase"] in ("inflate", "lift", "transport", "lower")])
    held_f = np.convolve(held, np.ones(3) / 3, mode="same") if len(held) >= 3 else held
    peak_true = float(held_f.max()) if len(held_f) else 0.0
    crushed = peak_true > fmax
    return dict(object=name, cls=cls, F_star_final=round(F_star, 3), mode=mode, actuator=actuator, p_fixed=p_fixed if mode == "open_loop" else None, F_star=F_star,
                lifted=bool(lifted), in_tray=bool(in_tray), crushed=bool(crushed), success=bool(lifted and in_tray and not crushed),
                peak_force_true=round(peak_true, 3), peak_force_est=round(peak_est, 3), max_safe=fmax), ep


def run_episode(cell: Cell, name: str, mode="closed_loop", actuator="learned", p_fixed=60.0, F_target=None, max_retries=2, plant="synthetic", slot=(0.0, 0.0),
                probe_fn=None):
    """one pick-and-place with up to `max_retries` regrasps (closed loop only); F* grows x1.4 per retry, capped at 70 % of the safe force."""
    ep = Episode(name, mode, actuator)
    ep.retries_left = max_retries if mode == "closed_loop" else 0
    F = TARGET_FORCE[OBJECTS[name][5]] if F_target is None else F_target
    attempts = 0
    while True:
        attempts += 1
        r, ep = _attempt(cell, name, mode, actuator, p_fixed, F, ep, plant, slot, probe_fn if attempts == 1 else None)
        if r is None and getattr(ep, "F_star_chosen", None):
            F = ep.F_star_chosen  # regrasp starts from the touch-chosen target
        if r is not None:
            r["attempts"] = attempts
            return r, ep
        ep.retries_left -= 1
        F = min(F * 1.4, 0.7 * OBJECTS[VISUAL_ID.get(name, name)][4])  # cap from what the planner believes the object is


def save_video(frames, path, fps=30):
    import imageio.v2 as imageio

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=8) as w:
        for f in frames:
            w.append_data(f)


def bench(seeds, out):
    rows = []
    configs = [("open_loop", 20.0, "learned"), ("open_loop", 60.0, "learned"), ("closed_loop", None, "linear"), ("closed_loop", None, "learned")]
    for s in seeds:
        for mode, pf, actk in configs:
            for name in BASE_OBJECTS:
                cell = Cell(seed=s, objects=(name,))
                r, ep = run_episode(cell, name, mode=mode, p_fixed=pf or 0.0, actuator=actk)
                r["seed"] = s
                rows.append(r)
                ftrue = np.array([e["f_true"] for e in ep.log if e.get("phase") == "inflate"])
                fest = np.array([e["f_est"] for e in ep.log if e.get("phase") == "inflate"])
                r["est_corr"] = float(np.corrcoef(ftrue, fest)[0, 1]) if ftrue.std() > 1e-6 and fest.std() > 1e-6 else None
                r["est_mae"] = float(np.mean(np.abs(ftrue - fest))) if len(ftrue) else float("nan")
                print(json.dumps(r), flush=True)
    summary = {}
    for mode, pf, actk in configs:
        key = f"{mode}" + (f"_{int(pf)}kPa" if pf else f"_{actk}")
        rs = [r for r in rows if r["mode"] == mode and (r["p_fixed"] == pf) and (pf or r["actuator"] == actk)]
        summary[key] = {
            "n": len(rs), "success_rate": round(np.mean([r["success"] for r in rs]), 3), "crush_rate": round(np.mean([r["crushed"] for r in rs]), 3),
            "drop_rate": round(np.mean([not r["lifted"] for r in rs]), 3),
            "mean_attempts": round(float(np.mean([r.get("attempts", 1) for r in rs])), 2),
            "est_abs_err_N": round(float(np.nanmean([r["est_mae"] for r in rs])), 3),
            "by_object": {o: round(np.mean([r["success"] for r in rs if r["object"] == o]), 3) for o in BASE_OBJECTS},
        }
    res = {"engine": f"MuJoCo {mujoco.__version__}", "seeds": list(seeds), "summary": summary, "rows": rows}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd")
    a = sp.add_parser("episode")
    a.add_argument("name")
    a.add_argument("--mode", default="closed_loop")
    a.add_argument("--p_fixed", type=float, default=60.0)
    a.add_argument("--actuator", default="learned")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--video")
    a.add_argument("--cam", default="close")
    b = sp.add_parser("bench")
    b.add_argument("--seeds", type=int, default=10)
    b.add_argument("--out", default=str(REPO / "results" / "benchmark_mujoco.json"))
    args = ap.parse_args()
    if args.cmd == "episode":
        cell = Cell(seed=args.seed, objects=(args.name,), render=bool(args.video), cam=args.cam)
        r, ep = run_episode(cell, args.name, args.mode, args.actuator, args.p_fixed)
        print(json.dumps(r))
        out = REPO / "results" / f"mj_episode_{args.name}_{args.mode}.json"
        json.dump({"summary": r, "log": ep.log}, open(out, "w"))
        if args.video:
            save_video(cell.frames, args.video)
            print("video", args.video, len(cell.frames), "frames")
    elif args.cmd == "bench":
        res = bench(range(args.seeds), args.out)
        print(json.dumps(res["summary"], indent=1))
