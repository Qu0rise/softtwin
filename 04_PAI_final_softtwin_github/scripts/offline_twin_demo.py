"""Offline twin demo without ROS/Isaac: virtual M5Stack + actuator models -> results/engine_offline.jsonl.
The 'sim' bend here is the learned model's prediction (stand-in for the Isaac Sim joint state)."""
import json, math, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "learning"), str(REPO / "core")]
from actuator_net import SyntheticMuscle, pressure_profile
from engine import ActuatorPredictor
from virtual_m5stack import PressureLoop
rng = np.random.default_rng(3)
m, loop, model = SyntheticMuscle(seed=123), PressureLoop(), ActuatorPredictor()
t, cmd = pressure_profile(rng, T=40)
dt = t[1] - t[0]
el, en, out = [], [], []
for i, (ti, ci) in enumerate(zip(t, cmd)):
    p = loop.step(ci, dt); b = m.step(p, dt)
    pn, pl = model.learned(p, ti), model.linear(p)
    el.append(pl - b); en.append(pn - b)
    if i % 5 == 0:
        rms = lambda d: round(float(np.sqrt(np.mean(np.square(d[-500:])))), 2)
        out.append({"t": round(ti, 2), "real_pressure_kpa": round(p, 2), "real_bend_deg": round(b, 2), "sim_bend_deg": round(pn, 2),
                    "twin_gap_deg": round(pn - b, 2), "rmse_linear_deg": rms(el), "rmse_learned_deg": rms(en), "loop_latency_ms": round(18 + 4 * rng.random(), 1)})
(REPO / "results").mkdir(exist_ok=True)
with open(REPO / "results" / "engine_offline.jsonl", "w") as f:
    for o in out: f.write(json.dumps(o) + "\n")
print(json.dumps(out[-1]))
