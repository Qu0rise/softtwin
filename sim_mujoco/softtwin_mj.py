"""SoftTwin cell in MuJoCo: the physics engine behind sim_mujoco/twin_core.py (controller, sensors, script, labels).

  python sim_mujoco/softtwin_mj.py episode tomato --video media/mj_tomato.mp4
  python sim_mujoco/softtwin_mj.py bench --seeds 30          # Q1 -> results/benchmark_mujoco.json

Contact softness: MuJoCo's soft contact is a spring whose stiffness scales with the effective mass of the contact,
K = m_eff / (d (1-d) tc^2)  (solimp d constant, critical damping).  We choose the time constant tc per object instance so
that K equals the literature stiffness k [N/m] of the produce (m_eff from a one-off identification, see
scripts/calibrate_contacts.py and sim_mujoco/contact_calibration.json); PhysX gets the same k directly.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "osmesa" if sys.platform.startswith("linux") else "glfw")
import mujoco  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import twin_core as C  # noqa: E402
import visual_assets as VA  # noqa: E402
from twin_core import (ALL_OBJECTS, BASE_OBJECTS, CFG, CLASSES, HANDLING, LOOKALIKES, OBJECTS, TARGET_FORCE,  # noqa: E402,F401
                       VISUAL_CLASS, VISUAL_ID, Actuator, Supply, half_height, sample_instance)

REPO = C.REPO
_CAL = Path(__file__).resolve().parent / "contact_calibration.json"
CONTACT = json.load(open(_CAL)) if _CAL.exists() else {"m_finger": 0.005, "d": 0.9}


def base_name(name):
    """size variants of an object (scripts/size_shift.py, 'tomato@1.2') share the base object's contact calibration"""
    return name.split("@")[0]


def contact_params(inst, obj):
    """solref / solimp of the object geom so that one finger contact has stiffness inst['k'] [N/m]"""
    if inst["k"] is None:                                   # egg shell, steel: effectively rigid (> 1e5 N/m)
        return "0.002 1", "0.99 0.99 0.001"
    d = CONTACT["d"]
    m_eff = 1.0 / (1.0 / CONTACT["m_finger"] + 1.0 / inst["mass"])
    tc = math.sqrt(m_eff / (d * (1 - d) * inst["k"]) * CONTACT.get("correction", {}).get(base_name(inst["name"]), 1.0))
    return f"{max(tc, 0.002):.5f} 1", f"{d} {d} 0.001"


PLUNGER = dict(x=0.0, r=0.004, mass=0.005, kp=2000.0, kv=20.0, stroke=0.14, k_ratio=2.0)


def plunger_params(inst):
    """explicit contact pair plunger tip <-> object: stiffness k_ratio * k.  The literature k was measured with a 4 mm flat
    probe; a flat punch on an elastic half-space has K = 2 a E*, so the 8 mm tip is twice as stiff (k_ratio = 2).
    m_eff identified like the finger contacts (scripts/calibrate_contacts.py --plunger-only)"""
    if inst["k"] is None:
        return "0.002 1", "0.99 0.99 0.001"
    d = CONTACT["d"]
    m_eff = 1.0 / (1.0 / CONTACT.get("m_plunger", 0.005) + 1.0 / inst["mass"])
    tc = math.sqrt(m_eff / (d * (1 - d) * PLUNGER["k_ratio"] * inst["k"]) * CONTACT.get("plunger_correction", {}).get(base_name(inst["name"]), 1.0))
    return f"{max(tc, 0.002):.5f} 1", f"{d} {d} 0.001"


def quat_z(deg):
    h = math.radians(deg) / 2
    return f"{math.cos(h):.6f} 0 0 {math.sin(h):.6f}"


PRETTY = False   # media scripts set this: render-only produce meshes/sepals (no mass, no contacts; physics unchanged)


def build_xml(seed=0, objects=BASE_OBJECTS, kb=None, spots=None):
    top = CFG["table_top"]
    hx, hy, hz = CFG["home"]
    kb = CFG["k_b"] if kb is None else kb
    fingers = []
    for f in range(CFG["n_fingers"]):
        th = 90 + f * 360 / CFG["n_fingers"]
        r = CFG["finger_r"]
        L, w, t = CFG["seg_len"], CFG["seg_w"], CFG["seg_t"]
        x, y, z = r * math.cos(math.radians(th)), r * math.sin(math.radians(th)), -CFG["palm"][2]
        body = ""
        for k in reversed(range(CFG["n_seg"])):
            col = (f"{0.95:.2f} {0.45 + 0.08 * k:.2f} {0.20:.2f} 1" if not PRETTY
                   else ("0.93 0.93 0.91 1" if k % 2 == 0 else "0.84 0.85 0.84 1"))   # silicone, alternate chambers
            inner = body
            body = (f'<body name="f{f}s{k}" pos="0 0 {-L if k else 0}">'
                    f'<joint name="f{f}j{k}" type="hinge" axis="0 1 0" range="-10 80" damping="{CFG["d_b"]}" armature="0.0002"/>'
                    f'<geom type="box" size="{t/2} {w/2} {L*0.48}" pos="0 0 {-L/2}" rgba="{col}" mass="{CFG["seg_mass"]}" friction="1.0 0.02 0.002" condim="4"/>'
                    + "".join(f'<site name="tx_f{f}s{k}_{m}" type="box" size="0.008 {w/2*0.98:.4f} {L/4*0.99:.4f}" pos="{-t/2-0.002:.4f} 0 {-L*(0.25+0.5*m):.4f}" rgba="0.1 0.8 0.9 0.0"/>'
                              f'<geom name="pad_f{f}s{k}_{m}" type="box" size="0.0007 {w/2*0.92:.4f} {L/4*0.92:.4f}" pos="{-t/2-0.0007:.4f} 0 {-L*(0.25+0.5*m):.4f}" rgba="0.20 0.22 0.30 1" contype="0" conaffinity="0" mass="0"/>' for m in range(2))
                    + (f'<geom type="capsule" size="0.004 {w/2*0.8}" pos="{-t/2-0.002} 0 {-L+0.004}" euler="90 0 0" rgba="{col}" mass="0.001" friction="1.0 0.02 0.002" condim="4"/>' if k == CFG["n_seg"] - 1 else "")
                    + f'{inner}</body>')
        fingers.append(f'<body name="f{f}" pos="{x:.4f} {y:.4f} {z:.4f}" quat="{quat_z(th)}">{body}</body>')
    acts = "".join(f'<position name="a_f{f}j{k}" joint="f{f}j{k}" kp="{kb}" ctrlrange="-0.2 1.4"/>'
                   for f in range(CFG["n_fingers"]) for k in range(CFG["n_seg"]))
    objs, insts = [], {}
    for i, name in enumerate(objects):
        o = OBJECTS[name]
        inst = sample_instance(seed, name)
        if spots is not None:
            inst["xy"] = spots[i]
        insts[name] = inst
        sref, simp = contact_params(inst, o)
        sx, sy = inst["xy"]
        sz = " ".join(str(v) for v in o.size)
        look = C.VISUAL_ID.get(base_name(name), base_name(name))       # an overripe fruit looks like the ripe one
        vis = VA.object_visuals(look, o.size) if PRETTY else ""
        rgba = ("0 0 0 0" if vis and look == "strawberry" else "0.55 0.045 0.03 1" if PRETTY and look == "tomato"
                else " ".join(map(str, o.rgba)))
        mat = ' material="tomato_mat"' if PRETTY and look == "tomato" else ""
        objs.append(f'<body name="{name}" pos="{sx:.4f} {sy:.4f} {top + half_height(name) + 0.001:.4f}" quat="{quat_z(inst["yaw_deg"])}"><freejoint name="{name}_free"/>'
                    f'<geom name="{name}_g" type="{o.shape}" size="{sz}" mass="{inst["mass"]:.5f}" rgba="{rgba}"{mat} '
                    f'friction="{inst["mu"]:.3f} 0.005 0.0005" condim="4" priority="1" solref="{sref}" solimp="{simp}"/>{vis}</body>')
    tx, ty = CFG["tray"]
    walls = "".join(f'<geom type="box" size="{a}" pos="{b}" rgba="0.15 0.35 0.65 1"/>' for a, b in [
        ("0.09 0.07 0.005", f"{tx} {ty} {top+0.005}"), ("0.005 0.07 0.02", f"{tx-0.09} {ty} {top+0.02}"),
        ("0.005 0.07 0.02", f"{tx+0.09} {ty} {top+0.02}"), ("0.09 0.005 0.02", f"{tx} {ty-0.07} {top+0.02}"),
        ("0.09 0.005 0.02", f"{tx} {ty+0.07} {top+0.02}")])
    xml = f"""<mujoco model="softtwin">
<option timestep="0.001" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic" impratio="10" noslip_iterations="5"/>
<visual><global offwidth="1280" offheight="720"/><quality shadowsize="4096"/><headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5"/></visual>
<asset><texture type="skybox" builtin="gradient" rgb1="0.82 0.86 0.92" rgb2="0.35 0.38 0.45" width="512" height="512"/><texture name="grid" type="2d" builtin="checker" rgb1="0.30 0.31 0.33" rgb2="0.26 0.27 0.29" width="512" height="512"/>
<material name="grid" texture="grid" texrepeat="8 8" reflectance="0.05"/>
<material name="wood" rgba="0.62 0.50 0.38 1"/>{VA.assets_xml() if PRETTY else ""}</asset>
<worldbody>
<light pos="0.6 -0.8 2.2" dir="-0.3 0.4 -1" diffuse="0.8 0.8 0.8" castshadow="true"/>
<geom type="plane" size="3 3 0.1" material="grid"/>
<geom type="box" size="0.5 0.35 0.02" pos="0 0 {top-0.02}" material="wood" friction="0.8 0.005 0.0005"/>
{''.join(f'<geom type="box" size="0.02 0.02 {(top-0.04)/2}" pos="{a} {b} {(top-0.04)/2}" rgba="0.35 0.28 0.22 1"/>' for a,b in [(-0.46,-0.31),(0.46,-0.31),(-0.46,0.31),(0.46,0.31)])}
{walls}
<geom type="cylinder" size="0.01 0.6" pos="-0.55 0 0.6" rgba="0.6 0.6 0.62 1" contype="0" conaffinity="0"/>
<body name="palm" pos="{hx} {hy} {hz}" gravcomp="1">
  <joint name="slide_x" type="slide" axis="1 0 0" range="-0.45 0.45" damping="40"/>
  <joint name="slide_y" type="slide" axis="0 1 0" range="-0.35 0.35" damping="40"/>
  <joint name="slide_z" type="slide" axis="0 0 1" range="-0.3 0.2" damping="40"/>
  <inertial pos="0 0 0" mass="0.5" diaginertia="0.002 0.002 0.002"/>
  <geom type="box" size="{CFG['palm'][0]} {CFG['palm'][1]} {CFG['palm'][2]}" rgba="{'0.62 0.64 0.67 1' if PRETTY else '0.85 0.85 0.88 1'}" contype="0" conaffinity="0" mass="0"/>
  <geom type="cylinder" size="0.012 0.12" pos="0 0 0.13" rgba="0.6 0.6 0.62 1" contype="0" conaffinity="0" mass="0"/>
  <camera name="wrist" pos="-0.018 0 -0.012" xyaxes="1 0 0 0 1 0" fovy="70"/>
  <camera name="track" mode="track" pos="0.03 -0.28 -0.04" xyaxes="0.994 0.107 0 -0.017 0.157 0.987" fovy="44"/>
  <body name="plunger" pos="{PLUNGER['x']} 0 {-CFG['palm'][2]}">
    <joint name="plunger_z" type="slide" axis="0 0 -1" range="0 {PLUNGER['stroke']}" damping="2"/>
    <geom name="plunger_tip" type="cylinder" size="{PLUNGER['r']} 0.003" pos="0 0 -0.003" rgba="0.25 0.25 0.28 1" mass="{PLUNGER['mass']}" contype="0" conaffinity="0"/>
    <geom type="capsule" fromto="0 0 0 0 0 {PLUNGER['stroke']}" size="0.0018" rgba="0.75 0.75 0.78 1" contype="0" conaffinity="0" mass="0"/>
  </body>
  {''.join(fingers)}
</body>
{''.join(objs)}
<camera name="hero" pos="0.75 -0.75 1.22" xyaxes="0.707 0.707 0 -0.28 0.28 0.92"/>
<camera name="close" pos="0.22 -0.30 0.92" xyaxes="0.81 0.59 0 -0.25 0.34 0.91"/>
<camera name="overhead" pos="0 0 1.9" xyaxes="1 0 0 0 1 0"/>
</worldbody>
<contact>{"".join(f'<pair geom1="plunger_tip" geom2="{n}_g" condim="3" friction="0.5 0.5 0.005 0.0005 0.0005" solref="{plunger_params(insts[n])[0]}" solimp="{plunger_params(insts[n])[1]}"/>' for n in objects)}</contact>
<sensor>{"".join(f'<touch name="tx_f{f}s{k}_{m}" site="tx_f{f}s{k}_{m}"/>' for f in range(CFG["n_fingers"]) for k in range(CFG["n_seg"]) for m in range(2))}</sensor>
<actuator>{acts}<position name="a_x" joint="slide_x" kp="4000" kv="200"/><position name="a_y" joint="slide_y" kp="4000" kv="200"/><position name="a_z" joint="slide_z" kp="6000" kv="250"/><position name="a_probe" joint="plunger_z" kp="{PLUNGER['kp']}" kv="{PLUNGER['kv']}" ctrlrange="0 {PLUNGER['stroke']}"/></actuator>
</mujoco>"""
    return xml, insts


class Cell:
    """MuJoCo engine for twin_core (see the interface there).  One cell = one seed and its object instance(s)."""

    def __init__(self, seed=0, objects=BASE_OBJECTS, render=None, cam="hero", size=(960, 540), kb=None, spots=None):
        xml, self.insts = build_xml(seed, objects, kb, spots)
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)
        mujoco.mj_forward(self.m, self.d)
        self.seed, self.objects = seed, objects
        self.inst = self.insts[objects[0]]
        self.dt_ctrl = CFG["dt_ctrl"]
        self.sub = int(round(self.dt_ctrl / self.m.opt.timestep))
        self.home = np.array(CFG["home"])
        self.palm_target = self.home.copy()
        self.nj = CFG["n_fingers"] * CFG["n_seg"]
        self.jadr = np.array([self.m.joint(f"f{f}j{k}").qposadr[0] for f in range(CFG["n_fingers"]) for k in range(CFG["n_seg"])])
        self.finger_bodies = {self.m.body(f"f{f}s{k}").id for f in range(CFG["n_fingers"]) for k in range(CFG["n_seg"])}
        self.frames, self.frame_t, self.renderer, self.cam = [], [], None, cam
        if render:
            self.renderer = mujoco.Renderer(self.m, size[1], size[0])
            self.render_every = int(round(1 / 30 / self.m.opt.timestep))
        self.t_step = 0
        self.bend_cmd = 0.0
        for _ in range(300):  # settle objects
            self._step()

    # ---- engine interface (twin_core)
    @property
    def t(self):
        return float(self.d.time)

    def _step(self):
        mujoco.mj_step(self.m, self.d)
        self.t_step += 1
        if self.renderer is not None and self.t_step % self.render_every == 0:
            self.renderer.update_scene(self.d, camera=self.cam)
            self.frames.append(self.renderer.render().copy())
            self.frame_t.append(float(self.d.time))

    def set_palm(self, xyz):
        self.palm_target = np.asarray(xyz, float)
        self.d.ctrl[self.nj:self.nj + 3] = self.palm_target - self.home

    def apply(self, bend_deg):
        self.bend_cmd = float(bend_deg)
        self.d.ctrl[: self.nj] = math.radians(self.bend_cmd / CFG["n_seg"])

    def advance(self, n_ctrl=1):
        for _ in range(n_ctrl * self.sub):
            self._step()

    def obj_pos(self, name):
        return self.d.body(name).xpos.copy()

    def q(self):
        return self.d.qpos[self.jadr].reshape(CFG["n_fingers"], CFG["n_seg"])

    def bend_true(self):
        return np.degrees(self.q().sum(1))

    def taxels_true(self):
        return self.d.sensordata[: self.nj * 2].copy()

    def force_true(self, name):
        gid = self.m.geom(f"{name}_g").id
        tot, f6 = 0.0, np.zeros(6)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if gid in (c.geom1, c.geom2) and {self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]} & self.finger_bodies:
                mujoco.mj_contactForce(self.m, self.d, i, f6)
                tot += abs(f6[0])
        return tot

    # ---- firmness plunger (MuJoCo only; see twin_core.probe_plunger)
    def plunger_set(self, ext):
        self.d.ctrl[self.m.actuator("a_probe").id] = float(np.clip(ext, 0.0, PLUNGER["stroke"]))

    def plunger_pos(self):
        return float(self.d.qpos[self.m.joint("plunger_z").qposadr[0]])

    def plunger_force(self):
        gid, tot, f6 = self.m.geom("plunger_tip").id, 0.0, np.zeros(6)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if gid in (c.geom1, c.geom2):
                mujoco.mj_contactForce(self.m, self.d, i, f6)
                tot += abs(f6[0])
        return tot

    def on_frame(self, tag, ep):
        cb = getattr(self, "frame_cb", None)
        if cb is not None:
            cb(self, tag, ep)

    # ---- helpers for scripts
    def move_palm(self, target, seconds, bend=0.0):
        """open-loop palm move with a fixed bend (scripts / tests only)"""
        start = self.palm_target.copy()
        n = max(1, int(round(seconds / self.dt_ctrl)))
        for i in range(n):
            a = (i + 1) / n
            self.set_palm(start + a * a * (3 - 2 * a) * (np.asarray(target) - start))
            self.apply(bend)
            self.advance(1)


def run_episode(cell: Cell, name: str, **kw):
    """twin_core.run_episode on this MuJoCo cell (same arguments)"""
    return C.run_episode(cell, name, seed=cell.seed, **kw)


def save_video(frames, path, fps=30):
    import imageio.v2 as imageio

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=8) as w:
        for f in frames:
            w.append_data(f)


Q1_CONFIGS = {   # name: run_episode arguments
    "open_loop_recipe": dict(mode="open_loop", actuator="learned"),
    "closed_affine": dict(actuator="affine"),
    "closed_static": dict(actuator="static"),
    "closed_greybox": dict(actuator="greybox"),
    "closed_greybox_play": dict(actuator="greybox_play"),
    "closed_learned": dict(actuator="learned"),
    "closed_learned_tip": dict(actuator="learned", lever="tip"),
    "closed_learned_nocontactskin": dict(actuator="learned", contact="est"),
    "closed_ideal_sensor": dict(actuator="learned", feedback="ideal"),
}
RECIPE = REPO / "results" / "open_loop_recipe.json"


def steady_force(name, p, seed=10_000):
    """true grip force after holding pressure p for 2 s on a nominal-ish instance (for the open-loop recipe)"""
    cell = Cell(seed=seed, objects=(name,))
    r, ep = run_episode(cell, name, mode="open_loop", p_fixed=p, max_retries=0)
    g = [e["f_true"] for e in ep.log if e["phase"] == "grip"]
    return float(np.mean(g[-10:])) if g else 0.0


def open_loop_recipe():
    """pressure per handling class, as a line would set it: bisection so that the mean steady grip force over the class's
    base objects equals the class target (seed 10000, outside the evaluation seeds)"""
    tab = {}
    for cls in CLASSES:
        objs = [n for n in BASE_OBJECTS if OBJECTS[n].cls == cls] or [n for n in ALL_OBJECTS if OBJECTS[n].cls == cls]
        lo, hi = 5.0, 60.0
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            f = np.mean([steady_force(n, mid) for n in objs])
            lo, hi = (mid, hi) if f < HANDLING[cls]["target"] else (lo, mid)
        tab[cls] = round(0.5 * (lo + hi), 2)
    json.dump(tab, open(RECIPE, "w"), indent=1)
    return tab


def _q1_job(a):
    seed, name, key = a
    kw = dict(Q1_CONFIGS[key])
    if kw.get("mode") == "open_loop":
        kw["p_fixed"] = json.load(open(RECIPE))[OBJECTS[name].cls]
    cell = Cell(seed=seed, objects=(name,))
    r, ep = run_episode(cell, name, **kw)
    r["config"] = key
    return r


def bench(seeds, out, configs=tuple(Q1_CONFIGS), workers=2):
    from multiprocessing import Pool
    if not RECIPE.exists():
        open_loop_recipe()
    jobs = [(s, n, c) for s in seeds for c in configs for n in BASE_OBJECTS]
    with Pool(workers) as pool:
        rows = pool.map(_q1_job, jobs, chunksize=4)
    summary = {}
    for c in configs:
        rs = [r for r in rows if r["config"] == c]
        summary[c] = {"n": len(rs), "success": sum(r["success"] for r in rs), "crushed": sum(r["crushed"] for r in rs),
                      "failure": {f: sum(r["failure"] == f for r in rs) for f in ("no_contact", "slip", "dropped", "crushed")},
                      "mean_attempts": round(float(np.mean([r["attempts"] for r in rs])), 2),
                      "est_mae_contact_N": round(float(np.nanmean([r["est_mae_contact"] if r.get("est_mae_contact") is not None else np.nan for r in rs])), 3),
                      "by_object": {o: [sum(r["success"] for r in rs if r["object"] == o), sum(r["crushed"] for r in rs if r["object"] == o)] for o in BASE_OBJECTS}}
    res = {"engine": f"MuJoCo {mujoco.__version__}", "seeds": list(seeds), "recipe_kPa": json.load(open(RECIPE)), "summary": summary, "rows": rows}
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
    b.add_argument("--seed0", type=int, default=0, help="first seed (30: the held-out confirmation run)")
    b.add_argument("--configs", default=",".join(Q1_CONFIGS))
    b.add_argument("--out", default=str(REPO / "results" / "benchmark_mujoco.json"))
    b.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    sp.add_parser("recipe")
    args = ap.parse_args()
    if args.cmd == "episode":
        cell = Cell(seed=args.seed, objects=(args.name,), render=bool(args.video), cam=args.cam)
        r, ep = run_episode(cell, args.name, mode=args.mode, actuator=args.actuator, p_fixed=args.p_fixed)
        print(json.dumps(r))
        out = REPO / "results" / f"mj_episode_{args.name}_{args.mode}.json"
        json.dump({"summary": r, "log": ep.log}, open(out, "w"))
        if args.video:
            save_video(cell.frames, args.video)
            print("video", args.video, len(cell.frames), "frames")
    elif args.cmd == "bench":
        res = bench(range(args.seed0, args.seed0 + args.seeds), args.out, configs=tuple(args.configs.split(",")), workers=args.workers)
        print(json.dumps(res["summary"], indent=1))
    elif args.cmd == "recipe":
        print(open_loop_recipe())
