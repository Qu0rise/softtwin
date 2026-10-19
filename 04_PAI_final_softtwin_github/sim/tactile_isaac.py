"""Tactile skin for the Isaac Sim cell (run inside Kit after sim/softtwin_scene.py; same 24-taxel layout as MuJoCo).

Every finger segment carries PhysxContactReportAPI (set in softtwin_scene.py).  Each physics step we read the PhysX
contact report, convert impulses to forces, bin each contact point into the proximal / distal half of its segment
(2 taxels per segment, 3 fingers x 4 segments) and pass the result through the same fabric-skin sensor model as
sim_mujoco/tactile.py (crosstalk, gain error, lag, noise, dead zone, saturation, 12-bit ADC).

  isaac_run_file("sim/tactile_isaac.py")          # defines SKIN and starts the per-step update
  isaac_exec("_result = SKIN.read().tolist()")    # 24 values [N]

Status: written for Isaac Sim 5.x / 6.x APIs; not yet executed on the target PC (MuJoCo results are the reference).
"""
from __future__ import annotations

import numpy as np
import omni.physx
import omni.usd
from pxr import PhysicsSchemaTools, UsdGeom

N_F, N_S = 3, 4
N_TAX = N_F * N_S * 2
G = "/World/SoftGripper"


class IsaacFabricSkin:
    def __init__(self, seed=0, tau=0.015, crosstalk=0.10, gain_sd=0.08, noise=0.006, dead=0.02, sat=8.0, bits=12):
        self.rng = np.random.default_rng(seed + 7)
        self.tau, self.noise, self.dead, self.sat = tau, noise, dead, sat
        self.gain = 1.0 + self.rng.normal(0, gain_sd, N_TAX)
        self.lsb = sat / (2 ** bits)
        self.y = np.zeros(N_TAX)
        self.raw = np.zeros(N_TAX)
        self.A = np.eye(N_TAX)
        for f in range(N_F):
            for i in range(N_S * 2):
                for d in (-1, 1):
                    if 0 <= i + d < N_S * 2:
                        self.A[f * 8 + i, f * 8 + i + d] = crosstalk
        self.stage = omni.usd.get_context().get_stage()
        self.seg_index = {f"{G}/f{f}_s{k}": (f, k) for f in range(N_F) for k in range(N_S)}
        self.sim = omni.physx.get_physx_simulation_interface()
        self.sub = omni.physx.get_physx_interface().subscribe_physics_step_events(self._on_step)

    def _bin(self, seg_path, pos_world):
        f, k = self.seg_index[seg_path]
        xf = UsdGeom.Xformable(self.stage.GetPrimAtPath(seg_path)).ComputeLocalToWorldTransform(0).GetInverse()
        z_local = xf.Transform(pos_world)[2]
        m = 0 if z_local > 0 else 1            # proximal half -> taxel 0, distal half -> taxel 1
        return f * 8 + k * 2 + m

    def _on_step(self, dt):
        headers, data = self.sim.get_contact_report()
        raw = np.zeros(N_TAX)
        for h in headers:
            a0 = str(PhysicsSchemaTools.intToSdfPath(h.actor0))
            a1 = str(PhysicsSchemaTools.intToSdfPath(h.actor1))
            seg = a0 if a0 in self.seg_index else a1 if a1 in self.seg_index else None
            if seg is None:
                continue
            for i in range(h.contact_data_offset, h.contact_data_offset + h.num_contact_data):
                c = data[i]
                imp = np.array([c.impulse[0], c.impulse[1], c.impulse[2]])
                n = np.array([c.normal[0], c.normal[1], c.normal[2]])
                fn = abs(float(imp @ n)) / max(dt, 1e-6)      # normal force [N]
                raw[self._bin(seg, c.position)] += fn
        self.raw = raw
        a = 1 - np.exp(-dt / self.tau)
        self.y += a * (self.A @ raw * self.gain - self.y)

    def read(self):
        v = self.y + self.rng.normal(0, self.noise, N_TAX)
        v = np.where(v < self.dead, 0.0, v)
        return np.round(np.clip(v, 0, self.sat) / self.lsb) * self.lsb

    def close(self):
        self.sub = None


try:
    SKIN.close()  # noqa: F821  (re-run safe)
except NameError:
    pass
SKIN = IsaacFabricSkin()
if __name__ in ("__main__", "__mcp__"):
    _result = {"taxels": N_TAX, "status": "subscribed to physics steps"}
