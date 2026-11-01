"""Every number quoted in the report, from the result files  ->  docs/paper/numbers.tex

    python scripts/make_numbers.py

The report never types a result by hand: it writes \\V{key} and this script defines the keys.  A key that is missing
prints as a red '??key' in the PDF, so a stale number cannot slip through.
"""
from __future__ import annotations

import json
import math
import re
import sys

import numpy as np
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
R = REPO / "results"
OUT = REPO / "docs" / "paper" / "numbers.tex"


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def pval(p):
    if p is None:
        return "--"
    if p < 1e-15:
        return r"$<10^{-15}$"
    if p < 1e-3:
        m, e = f"{p:.1e}".split("e")
        return rf"$={m}\times10^{{{int(e)}}}$"
    return f"$={p:.2f}$" if p >= 0.01 else f"$={p:.3f}$"


def main():
    V = {}
    m = json.load(open(R / "metrics.json"))
    import mujoco
    V["env.mujoco"] = mujoco.__version__
    q1 = m["q1"]
    names = {"recipe": "open_loop_recipe", "affine": "closed_affine", "static": "closed_static", "greybox": "closed_greybox",
             "learned": "closed_learned", "tip": "closed_learned_tip", "ideal": "closed_ideal_sensor", "gbplay": "closed_greybox_play"}
    for k, c in names.items():
        if c not in q1:
            continue
        x = q1[c]
        V[f"q1.{k}.succ"] = pct(x["success"]["rate"])
        V[f"q1.{k}.cr"] = str(x["crushed"]["k"])
        V[f"q1.{k}.mae"] = f"{x['est_mae_contact_N']:.2f}" if x.get("est_mae_contact_N") is not None else "--"
        if x.get("est_ratio_held_median") is not None:
            V[f"q1.{k}.bias"] = f"{x['est_ratio_held_median']:.2f}"
            V[f"q1.{k}.hold"] = f"{x['hold_over_target_median']:.2f}"
            if x.get("free_bend_err_held_median") is not None:
                V[f"q1.{k}.fberr"] = f"{x['free_bend_err_held_median']:+.1f}"
        lo, hi = x["success"]["ci95"]
        V[f"q1.{k}.ci"] = f"[{pct(lo)}, {pct(hi)}]"
    V["q1.n"] = str(q1["closed_learned"]["success"]["n"])
    V["q1.affine.bolt.slip"] = str(m["q1_per_object"]["bolt"]["closed_affine"]["slip"])
    for k, c in names.items():
        po = m["q1_per_object"]["tofu"].get(c)
        if po:
            V[f"q1.{k}.tofudrop"] = str(po["dropped"])
        pe = m["q1_per_object"]["egg"].get(c)
        if pe:
            V[f"q1.{k}.eggdrop"] = str(pe["dropped"] + pe.get("slip", 0))
    # tests
    tests = m["tests"]["paired"]
    V["t.n"] = str(len(tests))

    def tkey(t):
        a = t["a"].replace("closed_", "").replace("_clf", "")
        b = t["b"].replace("closed_", "").replace("_clf", "")
        return f"t.{'q2' if t['family'] == 'Q2' else ('px' if t['family'] == 'Q1 PhysX' else 'q1')}.{a}-{b}" + \
            (".la" if t["subset"] == "lookalike" else "") + (".cr" if t["metric"] == "crushed" else "")
    for t in tests:
        k = tkey(t)
        V[k + ".p"] = pval(t["p_holm"])
        V[k + ".ab"] = f"{t['a_better_only']} 対 {t['b_better_only']}"
        lo, hi = t["diff_ci95"]
        sign = 1 if t["metric"] == "success" else -1           # unrounded paired difference of the raw rates, a - b
        d = sign * (t["a_better_only"] - t["b_better_only"]) / t["n_pairs"]
        V[k + ".diff"] = f"{100 * d:.1f} ポイント（95\\,\\%CI {100 * lo:.1f}〜{100 * hi:.1f}）"
    # requirements
    for i, r in enumerate(m["requirements"], 1):
        mark = r"\checkmark" if r["pass"] else r"\textcolor{crit}{未達}"
        v = r["measured"]
        if "RMSE" in r["requirement"]:
            s = f"{v:.2f}°"
        elif "MAE" in r["requirement"]:
            s = f"{v:.2f}\\,N"
        else:
            s = f"{100 * v:.1f}"
            ci = re.search(r"CI95 \[([\d.]+), ([\d.]+)\]", r["note"])
            if ci:
                s += f" [{100 * float(ci.group(1)):.1f}, {100 * float(ci.group(2)):.1f}]"
        V[f"req.R{i}"] = f"{s} {mark}"
    # actuator
    ac = json.load(open(R / "actuator_compare.json"))
    for k, short in (("affine", "linear"), ("static", "static"), ("greybox", "greybox"), ("greybox_play", "gbplay"), ("learned", "learned")):
        if k not in ac:
            continue
        V[f"act.{short}"] = f"{ac[k]['rmse_deg']:.1f}"
        V[f"act.{short}.grip"] = f"{ac[k]['rmse_grip_deg']:.1f}"
        V[f"act.{short}.bias"] = f"{ac[k]['bias_grip_deg']:+.1f}"
        if "grasp_bias_deg" in ac[k]:
            V[f"act.{short}.gbias"] = f"{ac[k]['grasp_bias_deg']:+.1f}"
    rec = json.load(open(R / "open_loop_recipe.json"))
    V["rec.lo"], V["rec.hi"] = f"{min(rec.values()):.0f}", f"{max(rec.values()):.0f}"
    # contact calibration
    cal = json.load(open(REPO / "sim_mujoco" / "contact_calibration.json"))
    dev = max(abs(v["ratio_to_instance"] - 1) for v in cal["measured"].values())
    V["cal.mj.maxdev"] = f"{max(math.ceil(dev * 100), 1)}\\,\\%"   # ratios are stored to 3 decimals
    pxc = m.get("q1_physx_contact_calibration") or {}
    if pxc.get("measured_ratio"):
        r_ = list(pxc["measured_ratio"].values())
        V["cal.px.range"] = f"{min(r_):.2f}〜{max(r_):.2f}"
    # control
    if (R / "control_analysis.json").exists():
        ca = json.load(open(R / "control_analysis.json"))
        V["ctl.dFdp"] = f"{ca['dF_dp_mean']:.2f}"
        V["ctl.taucl"] = f"{ca['tau_closed_loop_s']:.1f}"
        r22 = [v["rise_to_80pct_s"] for v in ca["per_object"].values() if abs(v["target_N"] - 2.2) < 1e-6]
        V["ctl.rise22"] = f"{min(r22):.1f}〜{max(r22):.1f}"
    # probe map
    pm = json.load(open(R / "probe_sensitivity.json"))
    i25 = list(pm["ratios"]).index(2.5)
    i2 = list(pm["ratios"]).index(2.0)
    for pair, d in pm["pairs"].items():
        o = pair.split("/")[0]
        V[f"pm.{o}.finger"] = pct(d["finger"]["0.3"][i25], 0)
        V[f"pm.{o}.finger0"] = pct(d["finger"]["0.0"][i25], 0)
        V[f"pm.{o}.plunger"] = pct(min(d["plunger"]["2e-06"][i2], d["plunger"]["2e-06"][i25]), 0)
        i15 = list(pm["ratios"]).index(1.5)
        V[f"pm.{o}.plunger15"] = pct(d["plunger"]["2e-06"][i15], 0)
        V[f"pm.{o}.plunger15n50"] = pct(d["plunger"]["5e-05"][i15], 0)
        V[f"pm.{o}.plunger2n50"] = pct(d["plunger"]["5e-05"][i2], 0)
    # Q2
    q2 = m["q2"]
    pol = {"prior": "visual_prior", "soft": "soft_start", "finger": "finger_clf", "plunger": "plunger_clf", "tip": "plunger_clf_tip", "mass": "plunger_clf_mass",
           "fw": "force_window", "fwloo": "force_window_loo", "oracle": "oracle"}
    for k, p in pol.items():
        if p not in q2:
            continue
        x = q2[p]
        V[f"q2.{k}.succ"] = pct(x["success"]["rate"])
        V[f"q2.{k}.cr"] = str(x["crushed"]["k"])
        V[f"q2.{k}.la"] = pct(x["lookalike_success"]["rate"])
        V[f"q2.{k}.la.cr"] = str(x["lookalike_crushed"]["k"])
        V[f"q2.{k}.ct"] = f"{x['cycle_time_s']:.1f}"
    V["q2.ct.delta"] = f"{q2['plunger_clf']['cycle_time_s'] - q2['visual_prior']['cycle_time_s']:.1f}"
    # Q3
    cv = m["q3_cv"]
    for k in ("vision", "finger", "plunger", "all"):
        if k in cv:
            V[f"cv.{k}"] = f"{100 * cv[k]['mean']:.1f}$\\pm${100 * cv[k]['sd']:.1f}"
    card = (REPO / "data" / "tactile_dataset" / "DATASET_CARD.md").read_text()
    V["ds.n"] = re.search(r"episodes: (\d+)", card).group(1)
    V["ds.min"] = re.search(r"generated in ([\d.]+) min", card).group(1)
    V["ds.samples"] = f"{int(re.search(r'grip to lower\): (\d+)', card).group(1)):,}"
    oc = re.search(r"success (\d+)%, crushed (\d+)%,\s+slipped (\d+)%, dropped (\d+)%", card)
    V["ds.outcomes"] = f"成功 {oc.group(1)}\\,\\%・破損 {oc.group(2)}\\,\\%・滑り {oc.group(3)}\\,\\%・落下 {oc.group(4)}\\,\\%" if oc else "--"
    # sim2sim
    s2 = m.get("sim2sim", {})
    for k, c in names.items():
        if c in s2:
            V[f"s2s.{k}.mj"] = pct(s2[c]["mujoco"]["rate"])
            V[f"s2s.{k}.px"] = pct(s2[c]["physx"]["rate"])
            V[f"s2s.{k}.pk.mj"] = f"{s2[c]['peak_force_median_mujoco']:.2f}"
            V[f"s2s.{k}.pk.px"] = f"{s2[c]['peak_force_median_physx']:.2f}"
            if s2[c].get("pairs"):
                V[f"s2s.{k}.same"] = f"{s2[c]['same_outcome']}/{s2[c]['pairs']}"
                V[f"s2s.{k}.rel.soft"] = f"{100 * s2[c]['peak_rel_diff_median_soft']:.1f}"
                V[f"s2s.{k}.rel.stiff"] = f"{100 * s2[c]['peak_rel_diff_median_stiff']:.0f}"
                if s2[c].get("kappa") is not None:
                    V[f"s2s.{k}.kappa"] = f"{s2[c]['kappa']:.2f}"
                V[f"s2s.{k}.crsame"] = f"{s2[c].get('crush_same', 0)}/{s2[c]['pairs']}"
    if (R / "benchmark_isaac.json").exists():                  # where the two engines disagree, over all methods
        mjr = {(r["seed"], r["object"], r["config"]): r for r in json.load(open(R / "benchmark_mujoco.json"))["rows"]}
        pairs = [(mjr[(r["seed"], r["object"], r["config"])], r) for r in json.load(open(R / "benchmark_isaac.json"))["rows"]
                 if (r["seed"], r["object"], r["config"]) in mjr]
        diff = [(a, b) for a, b in pairs if (a["success"], a["crushed"]) != (b["success"], b["crushed"])]
        V["s2s.all.same"] = f"{len(pairs) - len(diff)}/{len(pairs)}"
        V["s2s.diff.n"] = str(len(diff))
        V["s2s.diff.tofu"] = str(sum(a["object"] == "tofu" and a["failure"] == "dropped" and b["success"] for a, b in diff))
    for k, c in names.items():
        if c in m.get("q1_physx", {}):
            V[f"px.{k}.succ"] = pct(m["q1_physx"][c]["success"]["rate"])
            V[f"px.{k}.cr"] = str(m["q1_physx"][c]["crushed"]["k"])
            V[f"px.{k}.mae"] = f"{m['q1_physx'][c]['est_mae_contact_N']:.2f}" if m["q1_physx"][c].get("est_mae_contact_N") else "--"
    if "q1_physx" in m and "closed_learned" in m["q1_physx"]:
        V["px.n"] = str(m["q1_physx"]["closed_learned"]["success"]["n"])
    # size shift (objects of a size the recipe was not tuned on)
    if (R / "size_shift.json").exists():
        ss = json.load(open(R / "size_shift.json"))
        short = {"open_loop_recipe": "recipe", "closed_learned": "learned", "closed_mass": "mass"}
        for key, v in ss["summary"].items():
            c, sc = key.split("@")
            t = f"ss.{short[c]}.{sc}"
            V[f"{t}.succ"] = f"{v['success']}/{v['n']}"
            V[f"{t}.cr"] = str(v["crushed"])
            V[f"{t}.pk"] = f"{v['peak_over_target_median']:.2f}"
            for o, (k, cr, pk) in v["by_object"].items():
                V[f"{t}.{o}.succ"] = f"{k}/{ss['seeds']}"
                V[f"{t}.{o}.cr"] = str(cr)
                V[f"{t}.{o}.pk"] = f"{pk:.2f}"
        V["ss.n"] = str(ss["seeds"])
        # weight-aware target: the floor it computed per object and scale
        sys.path.insert(0, str(REPO / "scripts"))
        import size_shift as _ss
        for o in ss["objects"]:
            for sc in ss["scales"]:
                V[f"ss.mass.F.{o}.{sc:g}"] = f"{_ss.mass_target(o, sc):.2f}"
        V["ss.mass.safety"], V["ss.mass.cap"] = f"{_ss.SAFETY:g}", f"{_ss.CAP:g}"
        for o in ss["objects"]:          # the floor before clipping, at 1.2x
            ob = _ss.C.OBJECTS[o]
            V[f"ss.mass.Fmin.{o}.1.2"] = f"{_ss.SAFETY * ob.mass * 1.2 ** 3 * (9.81 + _ss.A_MAX) / ob.mu:.1f}"
    # robustness
    mm = m["mismatch"]
    rate = lambda d: pct(d["success"] / d["n"], 0)  # noqa: E731
    V["rob.learned.30"] = rate(mm["muscle=0.3"]["closed_learned"])
    V["rob.learnedcal.30"] = rate(mm["muscle=0.3"]["closed_learned+cal"])
    V["rob.recipe.30"] = rate(mm["muscle=0.3"]["open_loop_recipe"])
    V["rob.kb125.cr"] = str(mm["k_b=1.25"]["closed_learned"]["crushed"])
    V["rob.kb125cal.cr"] = str(mm["k_b=1.25"]["closed_learned+cal"]["crushed"])
    V["rob.kb125.pk"] = f'{mm["k_b=1.25"]["closed_learned"]["peak_over_target_median"]:.2f}'
    V["rob.kb100.pk"] = f'{mm["muscle=0.0"]["closed_learned"]["peak_over_target_median"]:.2f}'
    V["rob.learnedcal.20"] = rate(mm["muscle=0.2"]["closed_learned+cal"])
    for c, short in (("closed_static", "static"), ("closed_greybox", "greybox"), ("open_loop_recipe", "recipe")):
        V[f"rob.kb125.{short}.pk"] = f'{mm["k_b=1.25"][c]["peak_over_target_median"]:.2f}'
        V[f"rob.kb125.{short}.cr"] = str(mm["k_b=1.25"][c]["crushed"])
    V["rob.nominal.learned"] = rate(mm["muscle=0.0"]["closed_learned"])
    V["rob.nominal.learned.pk"] = f'{mm["muscle=0.0"]["closed_learned"]["peak_over_target_median"]:.2f}'
    V["rob.nominalcal.learned.pk"] = f'{mm["muscle=0.0"]["closed_learned+cal"]["peak_over_target_median"]:.2f}'
    V["rob.kb125.learned.pk"] = f'{mm["k_b=1.25"]["closed_learned"]["peak_over_target_median"]:.2f}'
    v08 = [100 * d["success"] / d["n"] for d in mm["k_b=0.8"].values() if isinstance(d, dict) and "success" in d]
    V["rob.kb080.min"], V["rob.kb080.max"] = f"{min(v08):.0f}", f"{max(v08):.0f}"
    V["rob.learnedcal.30.cr"] = str(mm["muscle=0.3"]["closed_learned+cal"]["crushed"])
    V["rob.learned.30.cr"] = str(mm["muscle=0.3"]["closed_learned"]["crushed"])
    V["rob.recipe.30.cr"] = str(mm["muscle=0.3"]["open_loop_recipe"]["crushed"])
    V["rob.kb125cal.pk"] = f'{mm["k_b=1.25"]["closed_learned+cal"]["peak_over_target_median"]:.2f}'
    V["rob.learned.20"] = rate(mm["muscle=0.2"]["closed_learned"])
    # failures
    f = m["failures"]
    V["fail.prior.firmer"] = str(f["visual_prior"]["crush_after_believing_firmer"])
    V["fail.prior.n"] = str(f["visual_prior"]["n_fail"])
    V["fail.plunger.firmer"] = str(f["plunger_clf"]["crush_after_believing_firmer"])
    V["fail.plunger.n"] = str(f["plunger_clf"]["n_fail"])
    # per-object failure counts (Q2 bench), to keep the failure narrative generated
    la_rows = json.loads((R / "benchmark_lookalike.json").read_text())["rows"]
    for pol, short in (("plunger_clf", "plunger"), ("oracle", "oracle")):
        for obj in sorted({r["object"] for r in la_rows}):
            rr = [r for r in la_rows if r["policy"] == pol and r["object"] == obj]
            V[f"fail.{short}.{obj}"] = str(sum(not r["success"] for r in rr))
            V[f"fail.{short}.{obj}.cr"] = str(sum(bool(r["crushed"]) for r in rr))
        both = {(r["seed"], r["object"]) for r in la_rows if r["policy"] == pol and not r["success"]}
        V[f"fail.{short}.set"] = both  # temp, used below
    for obj in ("strawberry_overripe", "tofu", "tomato_overripe"):     # what separates the drops from the successes
        rr = [r for r in la_rows if r["policy"] == "plunger_clf" and r["object"] == obj]
        for lab, sub in (("fail", [r for r in rr if not r["success"]]), ("ok", [r for r in rr if r["success"]])):
            if sub:
                V[f"fail.margin.{obj}.{lab}"] = f'{np.median([r["mu"] * r["F_star_final"] / (r["mass"] * (9.81 + 0.9)) for r in sub]):.1f}'
                V[f"fail.mu.{obj}.{lab}"] = f'{np.median([r["mu"] for r in sub]):.2f}'
                V[f"fail.mass.{obj}.{lab}"] = f'{1000 * np.median([r["mass"] for r in sub]):.0f}'
    for obj in sorted({r["object"] for r in la_rows}):
        rr = [r for r in la_rows if r["policy"] == "plunger_clf_mass" and r["object"] == obj]
        if rr:
            V[f"fail.massv.{obj}"] = str(sum(not r["success"] for r in rr))
    common = V.pop("fail.plunger.set") & V.pop("fail.oracle.set")
    V["fail.common"] = str(len(common))
    # held-out confirmation on seeds 30-59 (never looked at during development)
    if (R / "benchmark_mujoco_holdout.json").exists():
        ho = json.load(open(R / "benchmark_mujoco_holdout.json"))
        V["ho.seeds"] = f"{ho['seeds'][0]}〜{ho['seeds'][-1]}"
        for k, c in names.items():
            if c in ho["summary"]:
                x = ho["summary"][c]
                V[f"ho.q1.{k}.succ"] = pct(x["success"] / x["n"])
                V[f"ho.q1.{k}.cr"] = str(x["crushed"])
        V["ho.q1.n"] = str(ho["summary"]["closed_learned"]["n"])
    if (R / "benchmark_lookalike_holdout.json").exists():
        ho2 = json.load(open(R / "benchmark_lookalike_holdout.json"))
        q2map = {"prior": "visual_prior", "soft": "soft_start", "finger": "finger_clf", "plunger": "plunger_clf",
                 "tip": "plunger_clf_tip", "mass": "plunger_clf_mass", "fw": "force_window", "fwloo": "force_window_loo", "oracle": "oracle"}
        for k, pcy in q2map.items():
            if pcy in ho2["summary"]:
                x = ho2["summary"][pcy]
                V[f"ho.q2.{k}.succ"] = pct(x["success"] / x["n"])
                V[f"ho.q2.{k}.cr"] = str(x["crushed"])
                V[f"ho.q2.{k}.la"] = pct(x["lookalike_success"] / x["lookalike_n"])
                V[f"ho.q2.{k}.la.cr"] = str(x["lookalike_crushed"])
                ks, nn, z = x["lookalike_success"], x["lookalike_n"], 1.959964
                ph = ks / nn
                cen = (ph + z * z / (2 * nn)) / (1 + z * z / nn)
                half = z * math.sqrt(ph * (1 - ph) / nn + z * z / (4 * nn * nn)) / (1 + z * z / nn)
                V[f"ho.q2.{k}.la.ci"] = f"[{pct(cen - half)}, {pct(cen + half)}]"
    # where the force-estimate error comes from (scripts/estimator_error_split.py)
    if (R / "estimator_error_split.json").exists():
        sp = json.load(open(R / "estimator_error_split.json"))["summary"]
        for mdl, d in sp.items():
            for fct in ("total", "muscle", "bend_sensor", "skin", "lever"):
                V[f"split.{mdl}.{fct}"] = f"{d[fct]:.3f}"
            V[f"split.{mdl}.n"] = str(d["episodes"])
    lines = ["% generated by scripts/make_numbers.py from results/*.json -- do not edit"]
    for k in sorted(V):
        lines.append(f"\\expandafter\\def\\csname V@{k}\\endcsname{{{V[k]}}}")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(V)} numbers -> {OUT}")
    render_readme(V)
    return V


def md(v):
    """the LaTeX-formatted value as plain text for Markdown"""
    v = v.replace("\\,\\%", " %").replace("\\,", " ").replace("$\\pm$", "±").replace("\\checkmark", "✓")
    v = re.sub(r"\$=?([\d.]+)\\times10\^\{(-?\d+)\}\$", r"\1×10^\2", v)
    return v.replace("$", "")


def render_readme(V):
    """README.md = docs/templates/README.template.md with {{key}} replaced; an unknown key stops the build"""
    tpl = (REPO / "docs" / "templates" / "README.template.md").read_text(encoding="utf-8")
    missing = sorted(set(re.findall(r"\{\{([\w.\-]+)\}\}", tpl)) - set(V))
    if missing:
        raise SystemExit(f"README not written: the template uses keys with no result yet: {missing}")
    out = re.sub(r"\{\{([\w.\-]+)\}\}", lambda m: md(V[m.group(1)]), tpl)
    (REPO / "README.md").write_text("<!-- generated from docs/templates/README.template.md by scripts/make_numbers.py -->\n" + out,
                                    encoding="utf-8")


if __name__ == "__main__":
    main()
