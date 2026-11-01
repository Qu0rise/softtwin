"""Where does the force-estimate error come from?  ->  results/estimator_error_split.json

    python scripts/estimator_error_split.py [--seeds 10]

Closed-loop grasps (the Q1 setting: 5 objects, no dry-stroke calibration) are re-run, and on every logged sample while
the object is really held (true force > 0.5 N, phases grip / lift / transport) the estimate of Eq. (2) is recomputed
with one input at a time replaced by its truth:

    F_hat                          the estimate the controller used
    F_hat[free = truth]            the muscle model's free bend replaced by the plant's
    F_hat[free, bend = truth]      ... and the bend sensor (noise, 20 ms lag) by the true joint angles
    F_hat[free, bend, skin = truth] ... and the skin readings (dead zone, gain, cross-talk, noise) by the true taxel loads

so that   F_hat / F = (muscle) x (bend sensor) x (skin) x (lever),  each factor the ratio of two neighbours; the last one,
F_hat[all inputs true] / F, is what Eq. (1) itself leaves: the taxel grid (contact beyond the last taxel), the finger
not being straight, and force components that are not normal to the finger.  The factor of each episode is the median
over its held samples; the table gives the median over episodes.
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sim_mujoco"))
import softtwin_mj as S  # noqa: E402
import twin_core as C  # noqa: E402

MODELS = ("static", "greybox", "greybox_play", "learned")
FACTORS = ("total", "muscle", "bend_sensor", "skin", "lever")


def _job(a):
    seed, name, actuator = a
    cell = S.Cell(seed=seed, objects=(name,))
    r, ep = S.run_episode(cell, name, actuator=actuator, max_retries=0)
    kb = C.CFG["k_b"]
    rows = []
    for e in ep.log:
        if e["phase"] not in ("grip", "lift", "transport") or e["f_true"] <= 0.5:
            continue
        nf = len(e["bend_meas"])
        free_m, free_t = [e["bend_model"]] * nf, [e["bend_free"]] * nf
        meas, true = e["bend_meas"], e["bend_true"]
        f0 = C.estimate_force(free_m, meas, kb, skin=e["tax"])
        f1 = C.estimate_force(free_t, meas, kb, skin=e["tax"])
        f2 = C.estimate_force(free_t, true, kb, skin=e["tax"])
        f3 = C.estimate_force(free_t, true, kb, skin=e["tax_true"])
        if min(f1, f2, f3) <= 0.05:
            continue
        F = e["f_true"]
        rows.append((f0 / F, f0 / f1, f1 / f2, f2 / f3, f3 / F))
    if not rows:
        return None
    med = np.median(np.array(rows), axis=0)
    return dict(seed=seed, object=name, actuator=actuator, n=len(rows), **{k: round(float(v), 4) for k, v in zip(FACTORS, med)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    jobs = [(s, n, m) for s in range(a.seeds) for n in S.BASE_OBJECTS for m in MODELS]
    with Pool(a.workers) as pool:
        rows = [r for r in pool.map(_job, jobs, chunksize=2) if r]
    summ = {}
    for m in MODELS:
        rs = [r for r in rows if r["actuator"] == m]
        summ[m] = {k: round(float(np.median([r[k] for r in rs])), 3) for k in FACTORS}
        summ[m]["episodes"] = len(rs)
        print(m, summ[m], flush=True)
    json.dump(dict(design=__doc__.strip().splitlines()[0], seeds=a.seeds, summary=summ, rows=rows),
              open(REPO / "results" / "estimator_error_split.json", "w"), indent=1)


if __name__ == "__main__":
    main()
