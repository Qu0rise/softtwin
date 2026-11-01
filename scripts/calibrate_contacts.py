"""Identify MuJoCo's effective contact mass so that every produce contact gets its literature stiffness [N/m].

MuJoCo's soft contact behaves (quasi-statically) like a spring K = m_eff / (d (1-d) tc^2), where m_eff is the
effective mass of the contact (here mostly the finger segment).  softtwin_mj.contact_params sets tc from the target
k and an assumed finger mass; this script grips each soft object with an ideal force loop, measures K = F / penetration
on every finger contact, corrects the finger mass and repeats until K matches k (within 5 %).

    python scripts/calibrate_contacts.py      # -> sim_mujoco/contact_calibration.json
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sim_mujoco"))
import mujoco  # noqa: E402

import softtwin_mj as S  # noqa: E402

OUT = REPO / "sim_mujoco" / "contact_calibration.json"
SOFT = [n for n in S.ALL_OBJECTS if S.OBJECTS[n].k is not None]


def body_pairs(cell, gid, bodies):
    """contacts of geom gid grouped by the other body: (sum of normal forces, mean penetration) per body"""
    acc, f6 = {}, np.zeros(6)
    for i in range(cell.d.ncon):
        c = cell.d.contact[i]
        if gid not in (c.geom1, c.geom2):
            continue
        other = cell.m.geom_bodyid[c.geom2 if c.geom1 == gid else c.geom1]
        if bodies is not None and other not in bodies:
            continue
        if bodies is None and cell.m.body(other).name not in S.ALL_OBJECTS:
            continue
        mujoco.mj_contactForce(cell.m, cell.d, i, f6)
        a = acc.setdefault(other, [0.0, []])
        a[0] += abs(f6[0])
        a[1].append(-c.dist)
    return [(F, float(np.mean(p))) for F, p in acc.values()]


def measure(name, seeds=(0, 1, 2), F=1.0):
    """median single-contact stiffness F_i / penetration_i [N/m] while the object is held at F (sum of the fingers)"""
    ks, ratios = [], []
    for seed in seeds:
        cell = S.Cell(seed=seed, objects=(name,))
        gid = cell.m.geom(f"{name}_g").id
        orig = cell.advance

        def adv(n=1, cell=cell, orig=orig, gid=gid):
            orig(n)
            if cell.force_true(name) < 0.5 * F:
                return
            for F_i, pen in body_pairs(cell, gid, cell.finger_bodies):
                if F_i > 0.1 and pen > 1e-5:
                    ks.append(F_i / pen)
                    ratios.append(F_i / pen / cell.inst["k"])
        cell.advance = adv
        S.run_episode(cell, name, F_target=F, feedback="ideal", max_retries=0)
    return float(np.median(ks)), float(np.median(ratios)), len(ks)


def measure_plunger(name, seeds=(0, 1, 2)):
    """median plunger-tip contact stiffness F / penetration [N/m] during the firmness probe, as a ratio to k_ratio * k"""
    import twin_core as C
    ratios = []
    for seed in seeds:
        cell = S.Cell(seed=seed, objects=(name,))
        gid = cell.m.geom("plunger_tip").id
        orig = cell.advance

        def adv(n=1, cell=cell, orig=orig, gid=gid):
            orig(n)
            for F_i, pen in body_pairs(cell, gid, None):
                if F_i > 0.1 and pen > 1e-6:
                    ratios.append(F_i / pen / (S.PLUNGER["k_ratio"] * cell.inst["k"]))
        cell.advance = adv
        ep = C.Episode(name, "closed_loop", "learned")
        o = cell.obj_pos(name)
        cell.move_palm([o[0], o[1], C.CFG["table_top"] + 0.22], 1.0)
        cell.move_palm([o[0], o[1], C.grasp_height(name, o[2])], 0.8)
        C.probe_plunger(cell, ep, top_expected=o[2] + C.half_height(name))
    return float(np.median(ratios)), len(ratios)


def calibrate_plunger(cal):
    cal.pop("plunger_correction", None)
    S.CONTACT.pop("plunger_correction", None)
    cal["m_plunger"] = 0.005
    for it in range(8):
        S.CONTACT.update(cal)
        res = {n: measure_plunger(n) for n in SOFT}
        r = np.nanmedian([v[0] for v in res.values()])
        print("plunger", it, round(cal["m_plunger"], 5), {n: round(v[0], 3) for n, v in res.items()}, flush=True)
        if abs(r - 1) < 0.05:
            break
        cal["m_plunger"] *= r
    cal["plunger_correction"] = {n: round(v[0], 4) for n, v in res.items()}
    S.CONTACT.update(cal)
    res = {n: measure_plunger(n) for n in SOFT}
    print("plunger corrected", {n: round(v[0], 3) for n, v in res.items()}, flush=True)
    cal["plunger_measured_ratio"] = {n: round(v[0], 3) for n, v in res.items()}
    return cal


def main():
    cal = json.load(open(OUT)) if OUT.exists() else {"m_finger": 0.005, "d": 0.9}
    if "--plunger-only" in sys.argv:                 # keep the finger contacts, redo the plunger tip only
        S.CONTACT.update(cal)
        cal = calibrate_plunger(cal)
        json.dump(cal, open(OUT, "w"), indent=1)
        print(json.dumps({k: v for k, v in cal.items() if "plunger" in k}, indent=1))
        return
    cal.pop("correction", None)
    cal.pop("measured", None)
    S.CONTACT.pop("correction", None)
    for it in range(6):
        json.dump(cal, open(OUT, "w"), indent=1)
        S.CONTACT.update(cal)
        res = {n: measure(n) for n in SOFT}
        r = np.median([v[1] for v in res.values()])
        print(it, round(cal["m_finger"], 5), {n: (round(v[0]), round(v[1], 3)) for n, v in res.items()}, flush=True)
        if abs(r - 1) < 0.05:
            break
        # K ~ m_eff: scale the finger's share of the effective mass by the median ratio
        cal["m_finger"] = cal["m_finger"] * r
    # per-object residual (contact geometry differs a little between shapes): one more pass with a per-object factor
    cal["correction"] = {n: round(v[1], 4) for n, v in res.items()}
    json.dump(cal, open(OUT, "w"), indent=1)
    S.CONTACT.update(cal)
    res = {n: measure(n) for n in SOFT}
    print("corrected", {n: (round(v[0]), round(v[1], 3)) for n, v in res.items()}, flush=True)
    cal["measured"] = {n: dict(k_target=S.OBJECTS[n].k, k_measured_median=round(v[0], 1), ratio_to_instance=round(v[1], 3), n_samples=v[2])
                       for n, v in res.items()}
    cal = calibrate_plunger(cal)
    json.dump(cal, open(OUT, "w"), indent=1)
    print(json.dumps(cal, indent=1))


if __name__ == "__main__":
    main()
