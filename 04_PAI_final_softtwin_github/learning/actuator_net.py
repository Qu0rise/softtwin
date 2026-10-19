"""Identify a soft pneumatic actuator model (pressure -> bend) from logged data and export it for the twin.

Input CSV columns (same format the M5Stack logger writes, see core/virtual_m5stack.py):
    t [s], pressure_kpa, bend_deg            (bend = total bend of one finger)
Without real data, `--synthetic` generates a stand-in dataset from a nonlinear, hysteretic,
lagged muscle model.  ALWAYS report results on synthetic data as such.

Model: tiny MLP  [p, dp/dt, bend_prev] -> bend   (numpy only, exported to learning/actuator_net.json)
Evaluation: autoregressive rollout on held-out trajectories, compared with the best linear fit bend = c*p.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


# ------------------------------------------------------------------ synthetic "real" muscle
class SyntheticMuscle:
    """Stand-in for the silicone bending actuator until real logs exist.
    deadband, saturating static curve, first-order lag, direction-dependent hysteresis, sensor noise."""

    def __init__(self, seed=0, p0=6.0, A=95.0, P=28.0, B=0.35, tau=0.12, hyst=6.0, noise=0.6):
        self.rng = np.random.default_rng(seed)
        self.p0, self.A, self.P, self.B, self.tau, self.hyst, self.noise = p0, A, P, B, tau, hyst, noise
        self.a, self.prev_p = 0.0, 0.0

    def static(self, p, rising):
        x = max(0.0, p - self.p0)
        a = self.A * (1 - math.exp(-x / self.P)) + self.B * x
        return a if rising else a + self.hyst * min(1.0, x / 20.0)

    def step(self, p, dt):
        target = self.static(p, p >= self.prev_p)
        self.a += (target - self.a) * (1 - math.exp(-dt / self.tau))
        self.prev_p = p
        return self.a + self.rng.normal(0, self.noise)


def pressure_profile(rng, T=20.0, dt=0.02):
    t = np.arange(0, T, dt)
    kind = rng.integers(0, 3)
    if kind == 0:  # steps
        p = np.repeat(rng.uniform(0, 80, size=int(T)), int(1 / dt))[: len(t)]
    elif kind == 1:  # ramps
        knots = rng.uniform(0, 80, size=6)
        p = np.interp(t, np.linspace(0, T, 6), knots)
    else:  # sines
        f = rng.uniform(0.05, 0.6)
        p = 40 + 38 * np.sin(2 * math.pi * f * t + rng.uniform(0, 6.28))
    return t, np.clip(p, 0, 80)


def make_synthetic(n_traj=80, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_traj):
        m = SyntheticMuscle(seed=seed * 1000 + k)
        t, p = pressure_profile(rng)
        dt = t[1] - t[0]
        for ti, pi in zip(t, p):
            rows.append((k, float(ti), float(pi), m.step(float(pi), dt)))
    return rows


def load_csv(path):
    rows, traj = [], 0
    last_t = None
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            t = float(r["t"])
            if last_t is not None and t < last_t:
                traj += 1
            last_t = t
            rows.append((traj, t, float(r["pressure_kpa"]), float(r["bend_deg"])))
    return rows


# ------------------------------------------------------------------ dataset / model
def features(rows):
    X, Y, G = [], [], []
    by = {}
    for k, t, p, a in rows:
        by.setdefault(k, []).append((t, p, a))
    for k, seq in by.items():
        for i in range(1, len(seq)):
            t0, p0, a0 = seq[i - 1]
            t1, p1, a1 = seq[i]
            dt = max(t1 - t0, 1e-3)
            X.append([p1, (p1 - p0) / dt, a0])
            Y.append(a1)
            G.append(k)
    return np.array(X), np.array(Y), np.array(G)


class MLP:
    def __init__(self, sizes=(3, 48, 48, 1), seed=0):
        rng = np.random.default_rng(seed)
        self.W = [rng.normal(0, 1 / math.sqrt(a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]

    def forward(self, x):
        hs = [x]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = x @ W + b
            if i < len(self.W) - 1:
                x = np.tanh(x)
            hs.append(x)
        return x, hs

    def train(self, X, Y, epochs=60, lr=3e-3, batch=256, seed=0):
        rng = np.random.default_rng(seed)
        m = [np.zeros_like(w) for w in self.W] + [np.zeros_like(b) for b in self.b]
        v = [np.zeros_like(w) for w in self.W] + [np.zeros_like(b) for b in self.b]
        step = 0
        for ep in range(epochs):
            idx = rng.permutation(len(X))
            for s in range(0, len(X), batch):
                j = idx[s : s + batch]
                out, hs = self.forward(X[j])
                g = 2 * (out[:, 0] - Y[j])[:, None] / len(j)
                grads_W, grads_b = [], []
                for i in reversed(range(len(self.W))):
                    grads_W.insert(0, hs[i].T @ g)
                    grads_b.insert(0, g.sum(0))
                    if i > 0:
                        g = (g @ self.W[i].T) * (1 - hs[i] ** 2)
                step += 1
                params = self.W + self.b
                grads = grads_W + grads_b
                for k, (pp, gg) in enumerate(zip(params, grads)):
                    m[k] = 0.9 * m[k] + 0.1 * gg
                    v[k] = 0.999 * v[k] + 0.001 * gg**2
                    mh, vh = m[k] / (1 - 0.9**step), v[k] / (1 - 0.999**step)
                    pp -= lr * mh / (np.sqrt(vh) + 1e-8)
        return self


def rollout(rows_traj, predict):
    """Autoregressive: the model only sees its own previous output (as in the twin)."""
    preds, prev_p, prev_a, prev_t = [], 0.0, 0.0, None
    for _, t, p, a in rows_traj:
        dt = 0.02 if prev_t is None else max(t - prev_t, 1e-3)
        a_hat = predict(p, (p - prev_p) / dt, prev_a)
        preds.append(a_hat)
        prev_p, prev_a, prev_t = p, a_hat, t
    return np.array(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="logged data from the real gripper (t,pressure_kpa,bend_deg)")
    ap.add_argument("--synthetic", action="store_true", help="use the synthetic stand-in muscle")
    ap.add_argument("--out", default=str(HERE / "actuator_net.json"))
    ap.add_argument("--epochs", type=int, default=150)
    a = ap.parse_args()
    rows = load_csv(a.csv) if a.csv else make_synthetic()
    source = a.csv or "synthetic (SyntheticMuscle, NOT real hardware)"
    trajs = sorted({r[0] for r in rows})
    test_ids = set(trajs[-max(2, len(trajs) // 5) :])  # split by trajectory (lecture 2: episode-wise split)
    train = [r for r in rows if r[0] not in test_ids]
    test = [r for r in rows if r[0] in test_ids]

    X, Y, _ = features(train)
    xm, xs = X.mean(0), X.std(0) + 1e-6
    ym, ys = float(Y.mean()), float(Y.std() + 1e-6)
    net = MLP().train((X - xm) / xs, (Y - ym) / ys, epochs=a.epochs)

    def pred_net(p, dp, ap_):
        out, _ = net.forward((np.array([p, dp, ap_]) - xm) / xs)
        return float(out[0] * ys + ym)

    P = np.array([r[2] for r in train])
    A = np.array([r[3] for r in train])
    c = float((P @ A) / (P @ P))  # best linear fit through the origin

    res = {"source": source, "linear_c_deg_per_kpa": round(c, 4), "test_trajectories": len(test_ids), "per_traj": []}
    err_lin, err_net = [], []
    by = {}
    for r in test:
        by.setdefault(r[0], []).append(r)
    for k, seq in by.items():
        truth = np.array([r[3] for r in seq])
        yl = np.array([c * r[2] for r in seq])
        yn = rollout(seq, pred_net)
        el, en = float(np.sqrt(np.mean((yl - truth) ** 2))), float(np.sqrt(np.mean((yn - truth) ** 2)))
        err_lin.append(el)
        err_net.append(en)
        res["per_traj"].append({"traj": int(k), "rmse_linear_deg": round(el, 2), "rmse_learned_deg": round(en, 2)})
    res["rmse_linear_deg"] = round(float(np.mean(err_lin)), 2)
    res["rmse_learned_deg"] = round(float(np.mean(err_net)), 2)
    res["improvement_pct"] = round(100 * (1 - res["rmse_learned_deg"] / res["rmse_linear_deg"]), 1)

    json.dump({"W": [w.tolist() for w in net.W], "b": [b.tolist() for b in net.b], "x_mean": xm.tolist(), "x_std": xs.tolist(),
               "y_mean": ym, "y_std": ys, "n_joints": 4, "linear_c": c, "source": source}, open(a.out, "w"))
    out_dir = REPO / "results"
    out_dir.mkdir(exist_ok=True)
    json.dump(res, open(out_dir / "actuator_identification.json", "w"), indent=1)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        k, seq = next(iter(by.items()))
        t = [r[1] for r in seq]
        fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
        ax[0].plot(t, [r[2] for r in seq], color="#555")
        ax[0].set_ylabel("pressure [kPa]")
        ax[1].plot(t, [r[3] for r in seq], label="measured", color="#222", lw=1.2)
        ax[1].plot(t, [c * r[2] for r in seq], label=f"linear (RMSE {res['per_traj'][0]['rmse_linear_deg']}°)", color="#c44", ls="--")
        ax[1].plot(t, rollout(seq, pred_net), label=f"learned (RMSE {res['per_traj'][0]['rmse_learned_deg']}°)", color="#27c")
        ax[1].set_ylabel("bend [deg]")
        ax[1].set_xlabel("time [s]")
        ax[1].legend(loc="upper right", fontsize=8)
        fig.suptitle("Actuator identification (" + ("synthetic stand-in" if not a.csv else "real gripper") + ")")
        fig.tight_layout()
        fig.savefig(out_dir / "actuator_fit.png", dpi=130)
    except Exception as e:  # plotting is optional
        print("plot skipped:", e)
    print(json.dumps({k: v for k, v in res.items() if k != "per_traj"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
