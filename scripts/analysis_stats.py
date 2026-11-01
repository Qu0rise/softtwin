"""Statistics for the report -> results/metrics.json

    python scripts/analysis_stats.py

What is computed (all from the committed result files, nothing re-simulated):
  Q1  force control without a force sensor (5 base objects x 30 seeds, MuJoCo; 20 seeds, PhysX):
      success / crush / drop with Wilson 95 % CIs; paired comparisons on the same (object, seed) episodes:
      exact McNemar + a 95 % CI of the paired difference (Newcombe's method 10), Holm over the pre-declared family;
      force-estimate error while in contact; per-object table; MuJoCo vs PhysX for the same configuration.
  Q2  look-alikes (7 objects x 30 seeds): the same for the perception policies, look-alike subset, cycle time,
      failure taxonomy (what the policy believed vs what the object was).
  Q3  stratified 5-fold CV of the class models (vision / +finger / +plunger / all) on the dataset episodes.
  probe map and mismatch study: copied from their result files with the operating point marked.
  requirements: the targets of the report's Sec. 2 with pass / fail.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binom

REPO = Path(__file__).resolve().parents[1]
R = REPO / "results"
sys.path[:0] = [str(REPO / "sim_mujoco")]
Z = 1.959964


def wilson(k, n, z=Z):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def _wilson_raw(k, n, z=Z):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def rate(rows, key):
    k = sum(bool(r[key]) for r in rows)
    return {"k": k, "n": len(rows), "rate": round(k / max(1, len(rows)), 3), "ci95": wilson(k, len(rows))}


def _sig(p):
    """p-values keep 3 significant digits (rounding to 4 decimals would print 1e-10 as 0)"""
    return float(f"{float(p):.3g}")


def mcnemar_exact(pairs):
    """pairs: list of (a_ok, b_ok) for the same (object, seed). Two-sided exact McNemar on the discordant pairs."""
    b = sum(1 for a, c in pairs if a and not c)
    c = sum(1 for a, c2 in pairs if (not a) and c2)
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * binom.cdf(min(b, c), n, 0.5))
    return b, c, p


def newcombe_paired(pairs):
    """95 % CI of p_a - p_b for paired binary outcomes (Newcombe 1998, method 10: Wilson score intervals + phi)"""
    n = len(pairs)
    a = sum(1 for x, y in pairs if x and y)
    b = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if (not x) and y)
    d = n - a - b - c
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = _wilson_raw(a + b, n)
    l2, u2 = _wilson_raw(a + c, n)
    den = (a + b) * (c + d) * (a + c) * (b + d)
    if den == 0:
        phi = 0.0
    else:
        num = a * d - b * c
        num = num - n / 2 if num > n / 2 else (num if num < 0 else 0.0)     # continuity correction as in Newcombe (1998)
        phi = num / math.sqrt(den)
    dl = math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    du = math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    diff = p1 - p2
    return [round(diff - dl, 3), round(diff + du, 3)]


def holm(ps):
    m, adj, run = len(ps), [0.0] * len(ps), 0.0
    for rank, i in enumerate(sorted(range(m), key=lambda i: ps[i])):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        adj[i] = run
    return adj


def sha(p):
    p = Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else None


def outcome_table(rows):
    return {"success": rate(rows, "success"), "crushed": rate(rows, "crushed"),
            "dropped": rate([dict(x, d=x["failure"] == "dropped") for x in rows], "d"),
            "slip": rate([dict(x, d=x["failure"] == "slip") for x in rows], "d"),
            "mean_attempts": round(float(np.mean([x["attempts"] for x in rows])), 2),
            "cycle_time_s": round(float(np.mean([x["cycle_time_s"] for x in rows])), 2),
            "est_mae_contact_N": _nanmean([x.get("est_mae_contact") for x in rows]),
            "peak_over_allowable_max": round(max(x["peak_force_true"] / x["max_safe"] for x in rows), 3),
            "est_ratio_held_median": _nanmedian([x.get("est_ratio_held") for x in rows]),
            "hold_over_target_median": _nanmedian([x.get("hold_over_target") for x in rows]),
            "free_bend_err_held_median": _nanmedian([x.get("free_bend_err_held") for x in rows])}


def _nanmedian(v):
    v = [x for x in v if x is not None]
    return round(float(np.median(v)), 3) if v else None


def _nanmean(v):
    v = [x for x in v if x is not None]
    return round(float(np.mean(v)), 3) if v else None


def paired(groups, a, b, key, subset=None, label=None):
    """exact McNemar + Newcombe CI on episodes present in both groups (key: 'success' or 'crushed' (fewer is better))"""
    A = {(x["object"], x["seed"]): x for x in groups[a] if subset is None or x["object"] in subset}
    B = {(x["object"], x["seed"]): x for x in groups[b] if subset is None or x["object"] in subset}
    ks = sorted(set(A) & set(B))
    good = (lambda x: not x[key]) if key == "crushed" else (lambda x: bool(x[key]))
    pairs = [(good(A[k]), good(B[k])) for k in ks]
    n10, n01, p = mcnemar_exact(pairs)
    raw = [(bool(A[k][key]), bool(B[k][key])) for k in ks]
    return {"a": a, "b": b, "metric": key, "subset": label or "all", "n_pairs": len(ks),
            "a_rate": round(sum(x for x, _ in raw) / len(ks), 3), "b_rate": round(sum(y for _, y in raw) / len(ks), 3),
            "diff_ci95": newcombe_paired(raw), "a_better_only": n10, "b_better_only": n01, "p_two_sided": _sig(p)}


def per_object(groups):
    out = {}
    for g, v in groups.items():
        for o in sorted({x["object"] for x in v}):
            w = [x for x in v if x["object"] == o]
            out.setdefault(o, {})[g] = {"n": len(w), "success": sum(bool(x["success"]) for x in w), "crushed": sum(bool(x["crushed"]) for x in w),
                                        "dropped": sum(x["failure"] == "dropped" for x in w), "slip": sum(x["failure"] == "slip" for x in w),
                                        "peak_force_median": round(float(np.median([x["peak_force_true"] for x in w])), 3)}
    return out


def main():
    import mujoco
    import twin_core as C

    out = {"provenance": {}, "q1": {}, "q1_physx": {}, "sim2sim": {}, "q2": {}, "q3_cv": {}, "tests": {}, "failures": {},
           "probe_map": {}, "mismatch": {}, "requirements": []}
    out["provenance"] = {
        "python": platform.python_version(), "numpy": np.__version__, "mujoco": mujoco.__version__, "platform": platform.platform(),
        "dataset_index_sha256_16": sha(REPO / "data/tactile_dataset/index.jsonl"),
        "actuator_net_sha256_16": sha(REPO / "learning/actuator_net.json"),
        "classifier_sha256_16": sha(REPO / "learning/tactile_classifier.json"),
        "force_window_sha256_16": sha(REPO / "learning/force_window.json"),
        "contact_calibration_sha256_16": sha(REPO / "sim_mujoco/contact_calibration.json"),
    }
    tests = []
    # ---------------- Q1 MuJoCo
    b1 = json.load(open(R / "benchmark_mujoco.json"))
    out["provenance"]["seeds_q1"] = [min(b1["seeds"]), max(b1["seeds"])]
    g1 = {}
    for r in b1["rows"]:
        g1.setdefault(r["config"], []).append(r)
    out["q1"] = {k: outcome_table(v) for k, v in g1.items()}
    out["q1_per_object"] = per_object(g1)
    fam1 = [("closed_learned", "closed_affine", "success"), ("closed_learned", "closed_learned_tip", "success"), ("closed_greybox", "closed_affine", "success"),
            ("closed_learned", "closed_greybox", "success"), ("closed_learned", "closed_static", "success"),
            ("closed_learned", "open_loop_recipe", "success"), ("closed_learned", "closed_ideal_sensor", "success")]
    tests += [dict(paired(g1, a, b, k), family="Q1 MuJoCo") for a, b, k in fam1 if a in g1 and b in g1]
    # ---------------- Q1 PhysX (Isaac Sim) and the same configs side by side
    if (R / "benchmark_isaac.json").exists():
        bi = json.load(open(R / "benchmark_isaac.json"))
        gi = {}
        for r in bi["rows"]:
            gi.setdefault(r["config"], []).append(r)
        out["q1_physx"] = {k: outcome_table(v) for k, v in gi.items()}
        out["q1_physx_per_object"] = per_object(gi)
        out["q1_physx_contact_calibration"] = bi.get("contact_calibration")
        seeds_i = {r["seed"] for r in bi["rows"]}
        for c in gi:
            if c in g1:
                mj = [r for r in g1[c] if r["seed"] in seeds_i]
                out["sim2sim"][c] = {"mujoco": rate(mj, "success"), "physx": rate(gi[c], "success"),
                                     "mujoco_crushed": rate(mj, "crushed"), "physx_crushed": rate(gi[c], "crushed"),
                                     "peak_force_median_mujoco": round(float(np.median([r["peak_force_true"] for r in mj])), 3),
                                     "peak_force_median_physx": round(float(np.median([r["peak_force_true"] for r in gi[c]])), 3)}
                # the same individual in both engines: does it end the same way, and how far apart are the peak forces?
                mjk = {(r["seed"], r["object"]): r for r in mj}
                pairs = [(mjk[(r["seed"], r["object"])], r) for r in gi[c] if (r["seed"], r["object"]) in mjk]
                same = sum((a["success"], a["crushed"]) == (b["success"], b["crushed"]) for a, b in pairs)
                def rel(objs):
                    v = [abs(b["peak_force_true"] / a["peak_force_true"] - 1) for a, b in pairs
                         if a["peak_force_true"] > 0.05 and a["object"] in objs]
                    return round(float(np.median(v)), 4) if v else None
                def lab(r):
                    return "crushed" if r["crushed"] else ("ok" if r["success"] else "lost")
                cats = ("ok", "crushed", "lost")
                po = same / len(pairs) if pairs else float("nan")
                pe = sum((sum(lab(a) == k for a, _ in pairs) / len(pairs)) * (sum(lab(b) == k for _, b in pairs) / len(pairs))
                         for k in cats) if pairs else float("nan")
                kappa = (po - pe) / (1 - pe) if pairs and pe < 1 else None
                crush_same = sum(a["crushed"] == b["crushed"] for a, b in pairs)
                out["sim2sim"][c].update(kappa=round(kappa, 3) if kappa is not None else None, crush_same=crush_same)
                out["sim2sim"][c].update(pairs=len(pairs), same_outcome=same,
                                         peak_rel_diff_median_soft=rel({"strawberry", "tofu", "tomato"}),
                                         peak_rel_diff_median_stiff=rel({"egg", "bolt"}))
        fam_i = [("closed_learned", "closed_affine", "success"), ("closed_learned", "open_loop_recipe", "success"),
                 ("closed_learned", "open_loop_recipe", "crushed")]
        tests += [dict(paired(gi, a, b, k, label="physx"), family="Q1 PhysX") for a, b, k in fam_i if a in gi and b in gi]
    # ---------------- Q2 look-alikes
    b2 = json.load(open(R / "benchmark_lookalike.json"))
    LA = set(C.LOOKALIKES)
    g2 = {}
    for r in b2["rows"]:
        g2.setdefault(r["policy"], []).append(r)
    out["q2"] = {p: dict(outcome_table(v), lookalike_success=rate([x for x in v if x["object"] in LA], "success"),
                         lookalike_crushed=rate([x for x in v if x["object"] in LA], "crushed"))
                 for p, v in g2.items()}
    out["q2_per_object"] = per_object(g2)
    fam2 = [("plunger_clf", "visual_prior", "success", None), ("plunger_clf", "visual_prior", "crushed", None),
            ("plunger_clf", "finger_clf", "success", LA), ("plunger_clf", "finger_clf", "crushed", LA),
            ("plunger_clf", "soft_start", "success", None), ("plunger_clf", "plunger_clf_tip", "crushed", LA), ("force_window", "plunger_clf", "success", None),
            ("force_window_loo", "force_window", "success", None)]
    tests += [dict(paired(g2, a, b, k, s, "lookalike" if s else "all"), family="Q2") for a, b, k, s in fam2 if a in g2 and b in g2]
    for t, q in zip(tests, holm([t["p_two_sided"] for t in tests])):
        t["p_holm"] = _sig(q)
    out["tests"] = {"paired": tests,
                    "note": "exact McNemar on (object, seed) pairs, Newcombe (1998) method-10 CI of the paired difference a - b; "
                            "Holm over all comparisons listed (declared before the final runs)"}
    # failure taxonomy (Q2): what the policy believed vs what the object was
    order = {c: i for i, c in enumerate(C.CLASSES)}       # very_fragile < soft < fragile < sturdy (firmness)
    for p, v in g2.items():
        tax = {"crush_after_believing_firmer": 0, "crush_after_regrasp": 0, "crush_other": 0, "drop_or_slip_after_believing_softer": 0,
               "drop_or_slip_other": 0, "n_fail": 0}
        for x in v:
            if x["success"]:
                continue
            tax["n_fail"] += 1
            believed = x["cls"] if p == "oracle" else (x.get("pred") or x["prior"])
            firmer, softer = order[believed] > order[x["cls"]], order[believed] < order[x["cls"]]
            if x["crushed"]:
                key = "crush_after_believing_firmer" if firmer else "crush_after_regrasp" if x["attempts"] > 1 else "crush_other"
            else:
                key = "drop_or_slip_after_believing_softer" if softer else "drop_or_slip_other"
            tax[key] += 1
        out["failures"][p] = tax
    # ---------------- Q3: class models, stratified 5-fold CV over dataset episodes
    import tactile as T
    import tactile_dataset as TD

    metas = TD.load_index(REPO / "data/tactile_dataset")
    X, y = TD.feature_sets(metas)
    rng = np.random.default_rng(0)
    folds = [[] for _ in range(5)]
    for c in np.unique(y):
        for i, j in enumerate(rng.permutation(np.where(y == c)[0])):
            folds[i % 5].append(j)
    la = np.array([m["object"] in LA or m["object"] in C.VISUAL_ID.values() for m in metas])
    for k, Xk in X.items():
        accs, acc_la = [], []
        for f in range(5):
            te = np.array(folds[f])
            tr = np.array([j for g in range(5) if g != f for j in folds[g]])
            pred = T.Softmax().fit(Xk[tr], y[tr]).predict(Xk[te])
            accs.append(float(np.mean(pred == y[te])))
            m = la[te]
            acc_la.append(float(np.mean(pred[m] == y[te][m])))
        out["q3_cv"][k] = {"mean": round(float(np.mean(accs)), 3), "sd": round(float(np.std(accs, ddof=1)), 3),
                           "folds": [round(a, 3) for a in accs], "tomato_strawberry_pairs_mean": round(float(np.mean(acc_la)), 3)}
    out["q3_dataset"] = {"episodes": len(metas), "objects": sorted({m["object"] for m in metas})}
    # ---------------- probe map, mismatch
    if (R / "probe_sensitivity.json").exists():
        pm = json.load(open(R / "probe_sensitivity.json"))
        i25 = list(pm["ratios"]).index(2.5)
        out["probe_map"] = {pair: {"finger_0.3deg_at_ratio_2.5": d["finger"]["0.3"][i25], "plunger_2um_at_ratio_2.5": d["plunger"]["2e-06"][i25],
                                   "finger_noise_free_at_ratio_2.5": d["finger"]["0.0"][i25]} for pair, d in pm["pairs"].items()}
    if (R / "robustness.json").exists():
        out["mismatch"] = json.load(open(R / "robustness.json"))["summary"]
    # ---------------- requirements (report Sec. 2)
    ac = json.load(open(R / "actuator_compare.json"))       # the deployed models, rolled out as in the twin
    act = {"rmse_learned_deg": ac["learned"]["rmse_deg"], "rmse_linear_deg": ac["affine"]["rmse_deg"]}
    q1l, q1g = out["q1"]["closed_learned"], out["q1"]["closed_greybox"]
    pl = out["q2"].get("plunger_clf")
    mm = out["mismatch"].get("muscle=0.2", {}).get("closed_learned+cal")
    req = [
        ("R1 actuator twin: free-bend RMSE <= 3 deg on held-out trajectories", act["rmse_learned_deg"], act["rmse_learned_deg"] <= 3.0,
         f"best straight line {act['rmse_linear_deg']} deg"),
        ("R2 grip-force estimate MAE <= 0.2 N while in contact (learned model)", q1l["est_mae_contact_N"], q1l["est_mae_contact_N"] <= 0.2,
         f"grey-box model {q1g['est_mae_contact_N']} N"),
        ("R3 base objects: success >= 90 %", q1l["success"]["rate"], q1l["success"]["rate"] >= 0.9, f"CI95 {q1l['success']['ci95']}"),
        ("R4 base objects: crush <= 5 % (upper CI95)", q1l["crushed"]["rate"], (q1l["crushed"]["ci95"][1] or 0) <= 0.05,
         f"CI95 {q1l['crushed']['ci95']}"),
    ]
    if pl:
        req += [("R5 look-alikes: success >= 85 %", pl["lookalike_success"]["rate"], pl["lookalike_success"]["rate"] >= 0.85,
                 f"CI95 {pl['lookalike_success']['ci95']}"),
                ("R6 all 7 objects: crush <= 5 % (upper CI95)", pl["crushed"]["rate"], (pl["crushed"]["ci95"][1] or 0) <= 0.05,
                 f"CI95 {pl['crushed']['ci95']}")]
    if "plunger" in out["q3_cv"]:
        req += [("R7 handling class from vision + plunger: accuracy >= 95 % (5-fold CV)", out["q3_cv"]["plunger"]["mean"],
                 out["q3_cv"]["plunger"]["mean"] >= 0.95, f"sd {out['q3_cv']['plunger']['sd']}")]
    if mm:
        req += [("R8 muscle off by up to +-20 %, after the dry-stroke calibration: success >= 90 %", round(mm["success"] / mm["n"], 3),
                 mm["success"] / mm["n"] >= 0.9, f"n = {mm['n']}")]
    out["requirements"] = [{"requirement": a, "measured": b, "pass": bool(c), "note": d} for a, b, c, d in req]
    json.dump(out, open(R / "metrics.json", "w"), indent=1)
    for r in out["requirements"]:
        print(("PASS " if r["pass"] else "FAIL ") + r["requirement"], r["measured"], r["note"])
    return out


if __name__ == "__main__":
    main()
