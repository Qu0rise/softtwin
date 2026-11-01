"""All four muscle models on the held-out identification trajectories  ->  results/actuator_compare.json

    python scripts/actuator_compare.py

Each model runs autoregressively on the measured (noisy) pressure, exactly as in the twin.  'grip range' keeps only the
samples with 14-25 kPa (the pressures of the open-loop recipe), where the force estimate is used.
'grasp bias' runs fresh grip-like pressure series (seeds not used for identification, noise-free) and averages
model - truth over the samples in the grip range while the pressure rises or holds, which is what the force loop does
after contact: this is the offset that enters the force estimate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sim_mujoco"))
sys.path.insert(0, str(REPO / "learning"))
import twin_core as C  # noqa: E402
from actuator_net import SyntheticMuscle, make_synthetic, pressure_profile  # noqa: E402


def grasp_bias(kind, lo, hi, n=30):
    errs, sd = [], 7
    while len(errs) < n:
        sd += 1
        if np.random.default_rng(sd).integers(0, 4) != 3:          # the grip-like kind of pressure_profile
            continue
        t, p = pressure_profile(np.random.default_rng(sd), T=20.0)
        m = SyntheticMuscle(noise=0.0)
        truth = np.array([m.step(x, 0.02) for x in p])
        a = C.Actuator(kind)
        pred = np.array([a(x, 0.02) for x in p])
        sel = (p >= lo) & (p <= hi) & (np.gradient(p, 0.02) > -0.5)
        if sel.sum() > 10:
            errs.append(float(np.mean(pred[sel] - truth[sel])))
    return round(float(np.median(errs)), 2)


def main():
    rows = make_synthetic()
    trajs = sorted({r[0] for r in rows})
    test_ids = trajs[-max(2, len(trajs) // 5):]               # the same split as learning/actuator_net.py
    rec = json.load(open(REPO / "results" / "open_loop_recipe.json"))
    lo, hi = min(rec.values()) - 0.5, max(rec.values()) + 0.5
    out = {"source": "synthetic (SyntheticMuscle, NOT real hardware)", "test_trajectories": len(test_ids), "grip_range_kPa": [lo, hi]}
    for kind in ("affine", "static", "greybox", "greybox_play", "learned"):
        tot, grip, bias = [], [], []
        for k in test_ids:
            seq = [r for r in rows if r[0] == k]
            a = C.Actuator(kind)
            pred = np.array([a(r[2], 0.02) for r in seq])
            truth = np.array([r[3] for r in seq])
            p = np.array([r[2] for r in seq])
            tot.append(float(np.sqrt(np.mean((pred - truth) ** 2))))
            m = (p >= lo) & (p <= hi)
            if m.sum() > 10:
                grip.append(float(np.sqrt(np.mean((pred[m] - truth[m]) ** 2))))
                bias.append(float(np.mean(pred[m] - truth[m])))
        out[kind] = dict(rmse_deg=round(float(np.mean(tot)), 2), rmse_grip_deg=round(float(np.mean(grip)), 2),
                         bias_grip_deg=round(float(np.mean(bias)), 2), grasp_bias_deg=grasp_bias(kind, lo, hi))
        print(kind, out[kind])
    json.dump(out, open(REPO / "results" / "actuator_compare.json", "w"), indent=1)


if __name__ == "__main__":
    main()
