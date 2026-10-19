"""Core engine: keeps the real gripper and the Isaac Sim twin in sync and measures how far apart they are.

ROS 2 in  : /real/pressure, /real/bend (M5Stack or virtual_m5stack), /softtwin/joint_states (Isaac Sim)
ROS 2 out : /softtwin/joint_command (twin follows the real pressure through the actuator model)
            /twin/metrics (std_msgs/String, JSON, 10 Hz)
App bus   : metrics + events are also published to RabbitMQ (exchange 'softtwin', topic keys) when RMQ_URL
            is set, so app/UI services never talk to ROS directly (design: robot domain = ROS 2,
            business/app domain = message broker).
Log       : results/engine_<ts>.jsonl
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sim"))


class ActuatorPredictor:
    """Same models as sim/softtwin_runtime.py, evaluated outside Kit (numpy)."""

    def __init__(self):
        d = json.load(open(REPO / "learning" / "actuator_net.json", encoding="utf-8"))
        self.W = [np.array(w) for w in d["W"]]
        self.b = [np.array(b) for b in d["b"]]
        self.xm, self.xs = np.array(d["x_mean"]), np.array(d["x_std"])
        self.ym, self.ys, self.c = d["y_mean"], d["y_std"], d["linear_c"]
        self.prev_p, self.prev_a, self.prev_t = 0.0, 0.0, None

    def learned(self, p, t):
        dt = 0.02 if self.prev_t is None else max(t - self.prev_t, 1e-3)
        h = (np.array([p, (p - self.prev_p) / dt, self.prev_a]) - self.xm) / self.xs
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < len(self.W) - 1:
                h = np.tanh(h)
        a = float(h[0] * self.ys + self.ym)
        self.prev_p, self.prev_a, self.prev_t = p, a, t
        return a

    def linear(self, p):
        return self.c * p


class Bus:
    def __init__(self):
        self.ch = None
        url = os.environ.get("RMQ_URL")
        if url:
            try:
                import pika

                conn = pika.BlockingConnection(pika.URLParameters(url))
                self.ch = conn.channel()
                self.ch.exchange_declare("softtwin", exchange_type="topic", durable=False)
            except Exception as e:
                print("RabbitMQ unavailable, continuing without it:", e)

    def publish(self, key, obj):
        if self.ch:
            self.ch.basic_publish("softtwin", key, json.dumps(obj).encode())


def main():
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float32, String

    class Engine(Node):
        def __init__(self):
            super().__init__("softtwin_core_engine")
            self.model = ActuatorPredictor()
            self.bus = Bus()
            self.real_p = self.real_b = None
            self.real_stamp = None
            self.sim_bend = None
            self.err_lin, self.err_net = deque(maxlen=500), deque(maxlen=500)
            self.lat = deque(maxlen=200)
            self.create_subscription(Float32, "/real/pressure", self.on_p, 10)
            self.create_subscription(Float32, "/real/bend", self.on_b, 10)
            self.create_subscription(JointState, "/softtwin/joint_states", self.on_js, 10)
            self.cmd = self.create_publisher(JointState, "/softtwin/joint_command", 10)
            self.pub = self.create_publisher(String, "/twin/metrics", 10)
            (REPO / "results").mkdir(exist_ok=True)
            self.log = open(REPO / "results" / f"engine_{time.strftime('%Y%m%d_%H%M%S')}.jsonl", "w")
            self.t0 = time.time()
            self.create_timer(0.1, self.report)

        def on_p(self, m):
            self.real_p = float(m.data)
            self.real_stamp = time.time()
            t = time.time() - self.t0
            pred_net = self.model.learned(self.real_p, t)
            pred_lin = self.model.linear(self.real_p)
            if self.real_b is not None:
                self.err_net.append(pred_net - self.real_b)
                self.err_lin.append(pred_lin - self.real_b)
            # drive the twin: same pressure -> same bend target on every finger joint (radians, per joint)
            js = JointState()
            js.name = [f"f{f}_j{k}" for f in range(3) for k in range(4)]
            js.position = [math.radians(pred_net / 4.0)] * len(js.name)
            self.cmd.publish(js)

        def on_b(self, m):
            self.real_b = float(m.data)

        def on_js(self, m):
            fingers = [p for n, p in zip(m.name, m.position) if n.startswith("f")]
            if fingers:
                self.sim_bend = math.degrees(sum(fingers)) / 3.0  # total bend per finger
            if self.real_stamp:
                self.lat.append(time.time() - self.real_stamp)

        def report(self):
            rms = lambda d: round(float(np.sqrt(np.mean(np.square(d)))), 2) if d else None  # noqa: E731
            met = {
                "t": round(time.time() - self.t0, 2),
                "real_pressure_kpa": self.real_p, "real_bend_deg": self.real_b, "sim_bend_deg": self.sim_bend,
                "twin_gap_deg": None if (self.sim_bend is None or self.real_b is None) else round(self.sim_bend - self.real_b, 2),
                "rmse_linear_deg": rms(self.err_lin), "rmse_learned_deg": rms(self.err_net),
                "loop_latency_ms": round(1000 * float(np.median(self.lat)), 1) if self.lat else None,
            }
            s = json.dumps(met)
            self.pub.publish(String(data=s))
            self.bus.publish("metrics.twin", met)
            self.log.write(s + "\n")

    rclpy.init()
    n = Engine()
    try:
        rclpy.spin(n)
    finally:
        n.log.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
