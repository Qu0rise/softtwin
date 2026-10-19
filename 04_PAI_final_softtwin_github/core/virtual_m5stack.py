"""Virtual M5Stack: stands in for the real gripper controller until the hardware is wired.

Same ROS 2 interface the real M5Stack (micro-ROS, firmware/m5stack_softgripper) will expose:
  sub  /softtwin/pressure_cmd   std_msgs/Float32   target chamber pressure [kPa]
  pub  /real/pressure           std_msgs/Float32   measured pressure [kPa]   (50 Hz)
  pub  /real/bend               std_msgs/Float32   measured finger bend [deg] (flex sensor / IMU)
It also logs t,pressure_kpa,bend_deg to data/real_log_<ts>.csv (input format of learning/actuator_net.py).

Run:  pixi run --manifest-path C:\\pai_ros\\pixi.toml virtual_hw
Offline (no ROS):  python core/virtual_m5stack.py --offline --seconds 60
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "learning"))
from actuator_net import SyntheticMuscle, pressure_profile  # noqa: E402

import numpy as np  # noqa: E402


class PressureLoop:
    """First-order pneumatic supply (pump + valve) tracking the commanded pressure."""

    def __init__(self, tau=0.25):
        self.p, self.tau = 0.0, tau

    def step(self, cmd, dt):
        self.p += (cmd - self.p) * (1 - math.exp(-dt / self.tau))
        return self.p


def offline(seconds, out):
    rng = np.random.default_rng(1)
    m, loop = SyntheticMuscle(seed=99), PressureLoop()
    t, cmd = pressure_profile(rng, T=seconds)
    dt = t[1] - t[0]
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "pressure_kpa", "bend_deg"])
        for ti, ci in zip(t, cmd):
            p = loop.step(ci, dt)
            w.writerow([round(ti, 3), round(p, 3), round(m.step(p, dt), 3)])
    print("wrote", out)


def ros(rate_hz=50.0):
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32

    class VirtualM5(Node):
        def __init__(self):
            super().__init__("virtual_m5stack")
            self.cmd = 0.0
            self.m, self.loop = SyntheticMuscle(seed=7), PressureLoop()
            self.pub_p = self.create_publisher(Float32, "/real/pressure", 10)
            self.pub_b = self.create_publisher(Float32, "/real/bend", 10)
            self.create_subscription(Float32, "/softtwin/pressure_cmd", lambda msg: setattr(self, "cmd", float(msg.data)), 10)
            (REPO / "data").mkdir(exist_ok=True)
            self.f = open(REPO / "data" / f"real_log_{time.strftime('%Y%m%d_%H%M%S')}.csv", "w", newline="")
            self.w = csv.writer(self.f)
            self.w.writerow(["t", "pressure_kpa", "bend_deg"])
            self.t0 = time.time()
            self.dt = 1.0 / rate_hz
            self.create_timer(self.dt, self.tick)

        def tick(self):
            p = self.loop.step(self.cmd, self.dt)
            b = self.m.step(p, self.dt)
            self.pub_p.publish(Float32(data=float(p)))
            self.pub_b.publish(Float32(data=float(b)))
            self.w.writerow([round(time.time() - self.t0, 3), round(p, 3), round(b, 3)])

    rclpy.init()
    node = VirtualM5()
    try:
        rclpy.spin(node)
    finally:
        node.f.close()
        rclpy.shutdown()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--out", default=str(REPO / "data" / "virtual_log.csv"))
    a = ap.parse_args()
    if a.offline:
        Path(a.out).parent.mkdir(exist_ok=True)
        offline(a.seconds, a.out)
    else:
        ros()
