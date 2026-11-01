"""Objects of a size the open-loop recipe was not tuned on  ->  results/size_shift.json

    python scripts/size_shift.py [--seeds 20]

The recipe (results/open_loop_recipe.json) sets one pressure per handling class from the nominal objects.  A line also
receives the same food in other sizes.  Here the fragile objects and the overripe tomato are scaled by s in {0.8, 1.2} (all lengths; mass by s^3;
contact stiffness, friction and allowable force unchanged) and the recipe is compared with the force loop on the same
instances.  The camera knows the size, so the grasp height follows it in both controllers.
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

SCALES = (0.8, 1.0, 1.2)
BASE = ("strawberry", "tofu", "tomato", "tomato_overripe")
CONTROLLERS = {"open_loop_recipe": dict(mode="open_loop", actuator="learned"), "closed_learned": dict(actuator="learned", calibrate=True),
               "closed_mass": dict(actuator="learned", calibrate=True)}
# weight-aware target (closed_mass): raise the class target to a friction floor computed from what the camera sees
#   F_min = SAFETY * m_hat * (g + a_max) / mu_hat,  m_hat = nominal mass of the type * s^3,  mu_hat = nominal friction of the type
# and clip it at CAP * allowable force.  SAFETY is read off the nominal-size objects only (no scaled object is used):
# friction margins mu F / m(g+a) of 2.9 and above dropped nothing, 2.3 dropped 1 in 10, 2.0 dropped 3 in 10.
SAFETY, A_MAX, CAP = C.FLOOR_SAFETY, C.A_MAX, C.FLOOR_CAP


def variant(name, s):
    """register name@s in the object table; the individual (mass, friction, stiffness, place) is the base object's.
    Both controllers are given the true handling class (the class decision is Q2, not this test)."""
    o = C.OBJECTS[name]
    key = f"{name}@{s:g}"
    C.OBJECTS[key] = o._replace(size=tuple(v * s for v in o.size), mass=o.mass * s ** 3, label=f"{o.label} ×{s:g}")
    return key


def mass_target(name, s):
    return C.weight_floor_target(name, C.OBJECTS[name].cls, s)


def _job(a):
    seed, name, s, ctrl = a
    key = variant(name, s)
    kw = dict(CONTROLLERS[ctrl])
    if kw.get("mode") == "open_loop":
        kw["p_fixed"] = json.load(open(S.RECIPE))[C.OBJECTS[name].cls]
    if ctrl == "closed_mass":
        kw["F_target"] = mass_target(name, s)
    inst = C.sample_instance(seed, name)          # the base object's individual, then scaled
    orig = C.sample_instance
    C.sample_instance = S.sample_instance = lambda sd, nm: dict(inst, name=nm, mass=inst["mass"] * s ** 3) if nm == key else orig(sd, nm)
    try:
        cell = S.Cell(seed=seed, objects=(key,))
        r, _ = S.run_episode(cell, key, **kw)
    finally:
        C.sample_instance = S.sample_instance = orig
    return dict(seed=seed, object=name, scale=s, controller=ctrl, success=bool(r["success"]), crushed=bool(r["crushed"]),
                failure=r["failure"], peak_force=round(float(r["peak_force_true"]), 3), target=kw.get("F_target") or C.TARGET_FORCE[C.OBJECTS[name].cls],
                max_safe=float(r["max_safe"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default=str(REPO / "results" / "size_shift.json"))
    a = ap.parse_args()
    jobs = [(sd, n, s, c) for sd in range(a.seeds) for n in BASE for s in SCALES for c in CONTROLLERS]
    with Pool(2) as pool:
        rows = pool.map(_job, jobs, chunksize=2)
    summ = {}
    for c in CONTROLLERS:
        for s in SCALES:
            rs = [r for r in rows if r["controller"] == c and r["scale"] == s]
            summ[f"{c}@{s:g}"] = dict(n=len(rs), success=sum(r["success"] for r in rs), crushed=sum(r["crushed"] for r in rs),
                                     slip_or_drop=sum(r["failure"] in ("slip", "dropped") for r in rs),
                                     peak_over_target_median=round(float(np.median([r["peak_force"] / r["target"] for r in rs])), 3),
                                     by_object={o: [sum(r["success"] for r in rs if r["object"] == o), sum(r["crushed"] for r in rs if r["object"] == o),
                                                    round(float(np.median([r["peak_force"] for r in rs if r["object"] == o])), 2)] for o in BASE})
            print(c, s, summ[f"{c}@{s:g}"], flush=True)
    json.dump(dict(design=__doc__.strip().splitlines()[0], scales=SCALES, objects=BASE, seeds=a.seeds,
                   recipe_kPa=json.load(open(S.RECIPE)), summary=summ, rows=rows), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
