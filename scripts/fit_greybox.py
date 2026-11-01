"""Grey-box actuator baseline: identify a physically structured model on the same data as the MLP.

    bend_free(p) = A (1 - exp(-max(0, p - p0) / P)) + B max(0, p - p0),   first-order lag tau   (5 parameters)

No hysteresis term on purpose: it is the model an engineer would write down from a datasheet-style step test.
Same train/test split (by trajectory) and the same autoregressive rollout as learning/actuator_net.py.

    python scripts/fit_greybox.py   ->  learning/actuator_greybox.json, results/actuator_greybox.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "learning"))
import actuator_net as AN  # noqa: E402

NAMES = ["p0", "A", "P", "B", "tau"]


def simulate(theta, seq):
    p0, A, P, B, tau = theta
    a, out, prev_t = 0.0, [], None
    for _, t, p, _ in seq:
        dt = 0.02 if prev_t is None else max(t - prev_t, 1e-3)
        x = max(0.0, p - p0)
        a += (A * (1 - math.exp(-x / P)) + B * x - a) * (1 - math.exp(-dt / tau))
        out.append(a)
        prev_t = t
    return np.array(out)


def simulate_play(theta, seq):
    """grey box + one play operator (the hysteresis an engineer adds once a ramp-up / ramp-down test shows a loop):
    the static curve is the middle of a band of width w; the state stays put inside the band"""
    p0, A, P, B, tau, w = theta
    a, st, out, prev_t = 0.0, None, [], None
    for _, t, p, _ in seq:
        dt = 0.02 if prev_t is None else max(t - prev_t, 1e-3)
        x = max(0.0, p - p0)
        c = A * (1 - math.exp(-x / P)) + B * x
        st = c if st is None else min(max(st, c - w / 2), c + w / 2)
        a += (st - a) * (1 - math.exp(-dt / tau))
        out.append(a)
        prev_t = t
    return np.array(out)


def split(rows):
    by = {}
    for r in rows:
        by.setdefault(r[0], []).append(r)
    ids = sorted(by)
    test_ids = set(ids[-max(2, len(ids) // 5):])
    return [by[k] for k in ids if k not in test_ids], [by[k] for k in ids if k in test_ids]


def main():
    rows = AN.make_synthetic()                      # identical data to the MLP (same seed)
    train, test = split(rows)

    def resid(theta):
        return np.concatenate([simulate(theta, s) - np.array([r[3] for r in s]) for s in train])

    fit = least_squares(resid, x0=[3.0, 80.0, 20.0, 0.2, 0.2], bounds=([0, 1, 1, 0, 0.01], [30, 200, 100, 2, 2]))
    theta = fit.x
    rmse = [float(np.sqrt(np.mean((simulate(theta, s) - np.array([r[3] for r in s])) ** 2))) for s in test]
    params = {k: round(float(v), 4) for k, v in zip(NAMES, theta)}
    json.dump({"params": params, "source": "synthetic (SyntheticMuscle, NOT real hardware)", "model": __doc__.split("\n")[3].strip()},
              open(REPO / "learning" / "actuator_greybox.json", "w"), indent=1)
    # variant with the hysteresis term (6 parameters)
    def resid_p(theta):
        return np.concatenate([simulate_play(theta, s) - np.array([r[3] for r in s]) for s in train])
    fit_p = least_squares(resid_p, x0=list(theta) + [4.0], bounds=([0, 1, 1, 0, 0.01, 0], [30, 200, 100, 2, 2, 20]))
    params_p = {k: round(float(v), 4) for k, v in zip(NAMES + ["w"], fit_p.x)}
    rmse_p = [float(np.sqrt(np.mean((simulate_play(fit_p.x, s) - np.array([r[3] for r in s])) ** 2))) for s in test]
    json.dump({"params": params_p, "source": "synthetic (SyntheticMuscle, NOT real hardware)",
               "model": "grey box + play operator of width w (6 parameters)"},
              open(REPO / "learning" / "actuator_greybox_play.json", "w"), indent=1)
    res = {"params": params, "params_play": params_p, "rmse_greybox_play_deg": round(float(np.mean(rmse_p)), 2), "true_params_of_stand_in": {"p0": 6.0, "A": 95.0, "P": 28.0, "B": 0.35, "tau": 0.12, "hyst_not_modelled": 6.0},
           "rmse_greybox_deg": round(float(np.mean(rmse)), 2), "per_traj_rmse_deg": [round(x, 2) for x in rmse]}
    json.dump(res, open(REPO / "results" / "actuator_greybox.json", "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
