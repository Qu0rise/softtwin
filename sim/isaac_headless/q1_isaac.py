"""Q1 re-run in Isaac Sim 5.0's PhysX (headless, no RTX renderer), with the SAME twin_core as MuJoCo.

sim_mujoco/twin_core.py (objects and their random instance per seed, muscle, actuator models, sensors, estimator,
controller, pick-and-place script, outcome labels) is imported unchanged.  This file only provides the engine:
the cell is built in USD by cell_usd.py with the geometry of sim_mujoco/softtwin_mj.py (palm, 3 fingers x 4 segments, joint
stiffness k_b, segment mass, fingertip capsule, table, tray), and what differs is the physics:
  MuJoCo : soft constraints (solref / solimp), elliptic friction cones, contact stiffness set through the time constant
  PhysX  : compliant contact with an explicit stiffness [N/m], patch friction, TGS solver
Contact stiffness is calibrated here the same way as in MuJoCo (grip at 1 N with an ideal force loop, F / penetration
per finger contact, corrected until it matches the instance's k) and written to isaac_io/v2_contact_calibration.json.

Plan (isaac_io/v2_plan.json, optional): {"seeds": [...], "objects": [...], "configs": [...], "calibrate": true,
"cal_seeds": [0, 1, 2], "replay": [[seed, object, config], ...]}
Outputs (isaac_io/): v2_rows.jsonl (one line per episode, resumable), v2_status.json, v2_summary.json,
v2_replay_<seed>_<object>_<config>.json (palm, finger joints and object pose per control step, for a MuJoCo render).
"""
import asyncio
import json
import math
import os
import sys
import time
import traceback

import numpy as np
import omni.kit.app
import omni.usd

REPO = os.environ["PAI_REPO"]
sys.path.insert(0, os.path.join(REPO, "sim_mujoco"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import twin_core as C  # noqa: E402
from cell_usd import DT_PHYS, HOME, L, NF, NS, build  # noqa: E402

IO = os.path.join(REPO, "isaac_io")
ROWS = os.path.join(IO, "v2_rows.jsonl")
STATUS = os.path.join(IO, "v2_status.json")
CAL = os.path.join(IO, "v2_contact_calibration.json")
APP = omni.kit.app.get_app()

CFG = C.CFG
SUB = int(round(CFG["dt_ctrl"] / DT_PHYS))

CONFIGS = {   # same names and arguments as softtwin_mj.Q1_CONFIGS
    "open_loop_recipe": dict(mode="open_loop", actuator="learned"),
    "closed_affine": dict(actuator="affine"),
    "closed_static": dict(actuator="static"),
    "closed_greybox": dict(actuator="greybox"),
    "closed_learned": dict(actuator="learned"),
    "closed_learned_tip": dict(actuator="learned", lever="tip"),
    "closed_ideal_sensor": dict(actuator="learned", feedback="ideal"),
}
PLAN = dict(seeds=list(range(20)), objects=list(C.BASE_OBJECTS),
            configs=["open_loop_recipe", "closed_affine", "closed_static", "closed_greybox", "closed_learned", "closed_learned_tip",
                     "closed_ideal_sensor"],
            calibrate=True, cal_seeds=[0, 1, 2], replay=[[0, n, "closed_learned"] for n in C.BASE_OBJECTS])
_pf = os.path.join(IO, "v2_plan.json")
if os.path.exists(_pf):
    PLAN.update(json.load(open(_pf)))
RECIPE = json.load(open(os.path.join(REPO, "results", "open_loop_recipe.json")))


class Cell:
    """PhysX engine for twin_core (same interface as softtwin_mj.Cell)"""

    def __init__(self, seed, name, k_scale=1.0, record=False):
        import omni.physics.tensors as T
        from omni.physx import get_physx_simulation_interface
        self.ps = get_physx_simulation_interface()
        self.seed, self.name = seed, name
        self.inst = build(seed, name, k_scale)
        self.ps.attach_stage(omni.usd.get_context().get_stage_id())
        self.t = 0.0
        self.dt_ctrl = CFG["dt_ctrl"]
        self._raw(5)
        self.v = T.create_simulation_view("numpy")
        self.v.set_subspace_roots("/")
        self.art = self.v.create_articulation_view("/World/Gripper/root")
        names = list(self.art.shared_metatype.dof_names)
        self.ix = {n: i for i, n in enumerate(names)}
        self.fj = np.array([self.ix[f"f{f}_j{k}"] for f in range(NF) for k in range(NS)])
        self.sl = np.array([self.ix[n] for n in ("slide_x", "slide_y", "slide_z")])
        self.obj = self.v.create_rigid_body_view(f"/World/Objects/{name}")
        self.palm = self.v.create_rigid_body_view("/World/Gripper/palm")
        self.segs = [self.v.create_rigid_body_view(f"/World/Gripper/f{f}_s{k}") for f in range(NF) for k in range(NS)]
        self.cv = self.v.create_rigid_contact_view(f"/World/Objects/{name}",
                                                   filter_patterns=[f"/World/Gripper/f{f}_s{k}" for f in range(NF) for k in range(NS)],
                                                   max_contact_data_count=1024)
        self.tgt = self.art.get_dof_position_targets()[0].copy()
        self.palm_target = HOME.copy()
        self.bend_cmd = 0.0
        self.record = record
        self.trace = []
        self.cal = None              # (F_i, penetration_i) samples, filled when calibrating
        self._raw(300)               # settle, as in MuJoCo

    def close(self):
        self.ps.detach_stage()

    def _raw(self, n):
        for _ in range(n):
            self.ps.simulate(DT_PHYS, self.t)
            self.ps.fetch_results()
            self.t += DT_PHYS

    # ---- engine interface (twin_core)
    def set_palm(self, xyz):
        self.palm_target = np.asarray(xyz, float)

    def apply(self, bend_deg):
        self.bend_cmd = float(bend_deg)

    def advance(self, n_ctrl=1):
        for _ in range(n_ctrl):
            self.tgt[self.sl] = self.palm_target - HOME
            self.tgt[self.fj] = math.radians(self.bend_cmd / NS)
            self.art.set_dof_position_targets(self.tgt[None, :], np.array([0], dtype=np.int32))
            self._raw(SUB)
            if self.record:
                po = self.obj.get_transforms()[0]
                self.trace.append([round(self.t, 3)] + [round(float(v), 5) for v in self.palm.get_transforms()[0][:3]]
                                  + [round(float(v), 5) for v in self.art.get_dof_positions()[0][self.fj]]
                                  + [round(float(v), 5) for v in po[:3]] + [round(float(v), 5) for v in (po[6], po[3], po[4], po[5])])
            if self.cal is not None:
                self._cal_sample()

    def obj_pos(self, name=None):
        return self.obj.get_transforms()[0][:3].copy()

    def bend_true(self):
        return np.degrees(self.art.get_dof_positions()[0][self.fj].reshape(NF, NS).sum(1))

    def force_true(self, name=None):
        m = self.cv.get_contact_force_matrix(DT_PHYS)       # (1, 12, 3): object <- each finger segment
        return float(np.linalg.norm(m[0], axis=1).sum())

    def _contacts(self):
        """per finger segment: list of (normal force, point in the segment frame, separation)"""
        forces, points, _normals, seps, counts, starts = self.cv.get_contact_data(DT_PHYS)
        out = []
        for j, sv in enumerate(self.segs):
            n, s0 = int(counts[0, j]), int(starts[0, j])
            if n == 0:
                out.append([])
                continue
            tr = sv.get_transforms()[0]
            p, (qx, qy, qz, qw) = tr[:3], tr[3:7]
            R = np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                          [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                          [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])
            out.append([(abs(float(forces[i].ravel()[0])), R.T @ (points[i] - p), float(seps[i].ravel()[0])) for i in range(s0, s0 + n)])
        return out

    def taxels_true(self):
        """2 taxels per segment on the inner face (upper / lower half), object contacts only"""
        tax = np.zeros(NF * NS * 2)
        for j, cs in enumerate(self._contacts()):
            for F, pl, _ in cs:
                if pl[0] <= 0.0005:                           # inner side of the segment
                    tax[2 * j + (0 if pl[2] > -L / 2 else 1)] += F
        return tax

    def on_frame(self, tag, ep):
        pass

    # ---- contact stiffness calibration
    def _cal_sample(self):
        if self.force_true() < 0.5:
            return
        for cs in self._contacts():
            F = sum(c[0] for c in cs)
            pen = [-c[2] for c in cs if c[2] < 0]
            if F > 0.1 and pen and np.mean(pen) > 1e-5:
                self.cal.append(F / float(np.mean(pen)) / self.inst["k"])


def status(**kw):
    s = json.load(open(STATUS)) if os.path.exists(STATUS) else {}
    s.update(kw, updated=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(s, open(STATUS, "w"), indent=1)


async def new_stage():
    await omni.usd.get_context().new_stage_async()
    await APP.next_update_async()


async def calibrate():
    """PhysX compliant contact acts per contact point, so the stiffness of one finger contact differs from the material
    value (several points on a flat face).  Same procedure as scripts/calibrate_contacts.py: hold at 1 N (ideal force
    loop), median of F / penetration per finger contact, rescale, repeat."""
    soft = [n for n in PLAN["objects"] if C.OBJECTS[n].k is not None]
    cal = json.load(open(CAL)) if os.path.exists(CAL) else {"k_scale": {}, "measured_ratio": {}, "history": []}
    for it in range(3):
        todo = [n for n in soft if abs(cal["measured_ratio"].get(n, 0.0) - 1.0) > 0.05]
        if not todo:
            break
        for n in todo:
            ks = cal["k_scale"].get(n, 1.0)
            ratios = []
            for s in PLAN["cal_seeds"]:
                await new_stage()
                cell = Cell(s, n, k_scale=ks)
                cell.cal = ratios
                try:
                    C.run_episode(cell, n, F_target=1.0, feedback="ideal", max_retries=0)
                finally:
                    cell.close()
                await APP.next_update_async()
            r = float(np.median(ratios)) if ratios else float("nan")
            cal["history"].append(dict(iter=it, object=n, k_scale=round(ks, 4), measured_ratio=round(r, 3), n=len(ratios)))
            if np.isfinite(r) and r > 0:
                cal["measured_ratio"][n] = round(r, 3)
                cal["k_scale"][n] = round(ks / r, 4) if abs(r - 1) > 0.05 else ks
            json.dump(cal, open(CAL, "w"), indent=1)
            status(phase="calibrate", last=cal["history"][-1])
    return cal


async def main():
    t0 = time.time()
    status(phase="start", plan=PLAN, errors=[])
    errors = []
    try:
        cal = await calibrate() if PLAN.get("calibrate") else (json.load(open(CAL)) if os.path.exists(CAL) else {"k_scale": {}})
    except Exception:
        errors.append({"where": "calibrate", "tb": traceback.format_exc()})
        status(errors=errors)
        cal = {"k_scale": {}}
    done = set()
    if os.path.exists(ROWS):
        for line in open(ROWS):
            r = json.loads(line)
            done.add((r["seed"], r["object"], r["config"]))
    replay = {tuple(x) for x in PLAN.get("replay", [])}
    jobs = [(s, n, c) for s in PLAN["seeds"] for c in PLAN["configs"] for n in PLAN["objects"]]
    for i, (s, n, c) in enumerate(jobs):
        if (s, n, c) in done:
            continue
        try:
            await new_stage()
            kw = dict(CONFIGS[c])
            if kw.get("mode") == "open_loop":
                kw["p_fixed"] = RECIPE[C.OBJECTS[n].cls]          # the recipe tuned in MuJoCo, used unchanged
            cell = Cell(s, n, k_scale=cal["k_scale"].get(n, 1.0), record=(s, n, c) in replay)
            try:
                r, ep = C.run_episode(cell, n, seed=s, **kw)
            finally:
                cell.close()
            r.update(config=c, engine="Isaac Sim 5.0 PhysX 5 (headless, 1 kHz)")
            with open(ROWS, "a") as fp:
                fp.write(json.dumps(r) + "\n")
            if cell.record:
                json.dump(dict(seed=s, object=n, config=c, columns="t palm_xyz q12 obj_xyz obj_quat_wxyz", rows=cell.trace,
                               log=[dict(t=e["t"], phase=e["phase"], f_true=e.get("f_true"), f_est=e.get("f_est")) for e in ep.log]),
                          open(os.path.join(IO, f"v2_replay_{s}_{n}_{c}.json"), "w"))
        except Exception:
            errors.append({"key": [s, n, c], "tb": traceback.format_exc()[-1500:]})
        status(phase="run", done=i + 1, total=len(jobs), elapsed_s=round(time.time() - t0, 1), errors=errors[-5:], n_errors=len(errors))
        await APP.next_update_async()
    rows = [json.loads(x) for x in open(ROWS)] if os.path.exists(ROWS) else []
    summ = {}
    for c in PLAN["configs"]:
        rs = [r for r in rows if r["config"] == c]
        if rs:
            summ[c] = {"n": len(rs), "success": sum(r["success"] for r in rs), "crushed": sum(r["crushed"] for r in rs),
                       "failure": {f: sum(r["failure"] == f for r in rs) for f in ("no_contact", "slip", "dropped", "crushed")},
                       "by_object": {o: [sum(r["success"] for r in rs if r["object"] == o), sum(r["crushed"] for r in rs if r["object"] == o),
                                         sum(1 for r in rs if r["object"] == o)] for o in PLAN["objects"]}}
    json.dump({"engine": "Isaac Sim 5.0.0, PhysX 5 (omni.physx), headless, 1 kHz, TGS", "plan": PLAN, "contact_calibration": cal,
               "summary": summ, "n_errors": len(errors), "elapsed_s": round(time.time() - t0, 1)},
              open(os.path.join(IO, "v2_summary.json"), "w"), indent=1)
    status(phase="finished", elapsed_s=round(time.time() - t0, 1))
    APP.post_quit()


asyncio.ensure_future(main())
