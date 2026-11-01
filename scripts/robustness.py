"""Mismatch study: does the grip still hold when the hand is not the one the models were identified on?

Two kinds of mismatch, each swept on its own (10 seeds x 5 base objects per cell):
  muscle  - every episode gets its own muscle: p0, A, P, B, tau and the hysteresis are each scaled by an independent
            factor ~ U(1 - delta, 1 + delta).  The actuator model (and the open-loop recipe) stay nominal.
  k_b     - the finger joints are stiffer / softer than the k_b the force estimator assumes (ratio 0.8 or 1.25).
"+cal": before the first grasp, a dry stroke in the air (pressure staircase from 8 kPa until the fingers reach 24 deg,
nothing in the hand) tabulates measured - model bend vs pressure, and that residual is added to the model
(twin_core.calibrate_dry_stroke).  It can only correct the muscle side: a wrong k_b is invisible in the air.

    python scripts/robustness.py                 # ~30 min on 2 cores  ->  results/robustness.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sim_mujoco"))
sys.path.insert(0, str(REPO / "learning"))
import softtwin_mj as S  # noqa: E402
import twin_core as C  # noqa: E402
from actuator_net import SyntheticMuscle  # noqa: E402

NOMINAL = dict(p0=6.0, A=95.0, P=28.0, B=0.35, tau=0.12, hyst=6.0)
CONTROLLERS = {  # name: (run_episode arguments)
    "open_loop_recipe": dict(mode="open_loop", actuator="learned"),
    "closed_static": dict(actuator="static"),
    "closed_static+cal": dict(actuator="static", calibrate=True),
    "closed_greybox": dict(actuator="greybox"),
    "closed_greybox+cal": dict(actuator="greybox", calibrate=True),
    "closed_learned": dict(actuator="learned"),
    "closed_learned+cal": dict(actuator="learned", calibrate=True),
}
CONDITIONS = [("muscle", 0.0), ("muscle", 0.1), ("muscle", 0.2), ("muscle", 0.3), ("k_b", 0.8), ("k_b", 1.25)]


def muscle(delta, seed, name):
    rng = np.random.default_rng([seed, zlib.crc32(name.encode()), int(round(delta * 100))])
    kw = {k: v * float(rng.uniform(1 - delta, 1 + delta)) for k, v in NOMINAL.items()}
    return SyntheticMuscle(seed=0, noise=0.0, **kw), kw


def _job(a):
    kind, level, seed, ctrl, name = a
    kw = dict(CONTROLLERS[ctrl])
    if kw.get("mode") == "open_loop":
        kw["p_fixed"] = json.load(open(S.RECIPE))[C.OBJECTS[name].cls]
    delta = level if kind == "muscle" else 0.0
    kb = C.CFG["k_b"] * (level if kind == "k_b" else 1.0)
    plant, params = muscle(delta, seed, name)
    cell = S.Cell(seed=seed, objects=(name,), kb=kb)
    r, ep = S.run_episode(cell, name, plant=plant, **kw)
    return dict(kind=kind, level=level, seed=seed, controller=ctrl, object=name, success=r["success"], crushed=r["crushed"],
                failure=r["failure"], attempts=r["attempts"], peak_force=r["peak_force_true"], max_safe=r["max_safe"],
                muscle={k: round(v, 3) for k, v in params.items()}, calibration=getattr(ep, "calibration", None))


def summarize(rows):
    summ = {}
    for kind, level in CONDITIONS:
        for c in CONTROLLERS:
            v = [r for r in rows if r["kind"] == kind and r["level"] == level and r["controller"] == c]
            if not v:
                continue
            k = sum(r["success"] for r in v)
            summ.setdefault(f"{kind}={level}", {})[c] = {
                "n": len(v), "success": k, "crushed": sum(r["crushed"] for r in v),
                "failure": {f: sum(r["failure"] == f for r in v) for f in ("no_contact", "slip", "dropped", "crushed")},
                "peak_over_target_median": round(float(np.median([r["peak_force"] / C.HANDLING[C.OBJECTS[r["object"]].cls]["target"] for r in v])), 2)}
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=str(REPO / "results" / "robustness.json"))
    a = ap.parse_args()
    t0 = time.time()
    jobs = [(k, lv, s, c, n) for k, lv in CONDITIONS for s in range(a.seeds) for c in CONTROLLERS for n in C.BASE_OBJECTS]
    with Pool(a.workers) as pool:
        rows = pool.map(_job, jobs, chunksize=4)
    summ = summarize(rows)
    json.dump({"design": __doc__.strip().split("\n\n")[1], "nominal_muscle": NOMINAL, "conditions": CONDITIONS, "seeds": list(range(a.seeds)),
               "summary": summ, "rows": rows}, open(a.out, "w"), indent=1)
    for cond, d in summ.items():
        print(cond, {c: (x["success"], x["crushed"]) for c, x in d.items()})
    print(f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
