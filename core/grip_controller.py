"""Grip-force controller for the real gripper, without ROS (ros2/softtwin_grasp_node.py only moves topics in and out).

It is not a re-implementation: the actuator model, the force estimator and the control law are the ones in
sim_mujoco/twin_core.py, the same code the MuJoCo and PhysX evaluations ran.  Inputs per control period:
  p      measured chamber pressure [kPa]
  bend   measured total bend of each finger [deg], shape (3,)
  skin   fabric-skin reading [N], shape (24,), finger-major (finger, segment, upper/lower half)
Output: the pressure command [kPa].
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim_mujoco"))
import twin_core as C  # noqa: E402


class GripController:
    def __init__(self, actuator="learned", k_b=C.CFG["k_b"], lever="skin", p_max=80.0):
        self.act = C.Actuator(actuator)
        self.k_b, self.lever, self.p_max = k_b, lever, p_max
        self.ctrl = None
        self.F_hat = 0.0
        self.p_cmd = 0.0

    @property
    def state(self):
        if self.ctrl is None:
            return "idle"
        return "holding" if self.ctrl.contact else "closing"

    @property
    def reached(self):
        return self.ctrl is not None and self.ctrl.hold > 20

    @property
    def contact(self):
        return self.ctrl is not None and self.ctrl.contact

    def start(self, F_star):
        self.act.reset()
        self.ctrl = C.Controller("closed_loop", float(F_star))
        self.p_cmd = 0.0

    def release(self):
        self.ctrl, self.p_cmd = None, 0.0

    def step(self, p, bend, skin, dt=0.02):
        free = self.act(float(p), dt)
        skin = np.asarray(skin, float)
        self.F_hat = C.estimate_force(free, bend, self.k_b, skin=skin if self.lever == "skin" else None)
        if self.ctrl is None:
            return 0.0
        self.p_cmd = min(self.p_max, self.ctrl.step(self.F_hat, float(p), float(skin.sum()), dt))
        return self.p_cmd
