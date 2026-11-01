"""When can touch tell a ripe fruit from an overripe look-alike?  A map over firmness contrast x sensor noise.

For each pair (tomato / overripe tomato, strawberry / overripe strawberry) the overripe one's contact stiffness is set to
k_ripe / ratio, ratio in RATIOS (real produce: about 1.5-3, docs/physical_parameters.md).  Each seed runs
  (a) the finger probe: squeeze to 0.8 N with the fabric skin and the bend sensors, and
  (b) the firmness plunger: 8 mm flat tip, load cell + encoder, up to 0.3 N,
with noise-free sensors, and the true signals are stored.  Sensor noise is then added offline (Monte Carlo, the same
sensor models as online) for every noise level, the stiffness feature is computed, and a one-dimensional threshold
classifier (ripe vs overripe) is scored by 5-fold cross-validation over seeds.

    python scripts/probe_sensitivity.py            # -> results/probe_sensitivity.json, docs/figures/fig_probe_map.png
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sim_mujoco"))
import softtwin_mj as S  # noqa: E402
import twin_core as C  # noqa: E402

PAIRS = [("tomato", "tomato_overripe"), ("strawberry", "strawberry_overripe")]
RATIOS = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)
SEEDS = range(30)
BEND_NOISE = (0.0, 0.1, 0.3, 1.0)          # deg, 1 sigma (finger probe); 0.3 = the online assumption
ENC_NOISE = (1e-6, 2e-6, 10e-6, 50e-6)     # m, 1 sigma (plunger encoder); 2 um = the online assumption
OUT = REPO / "results" / "probe_sensitivity.json"
CLEAN = C.SensorSpec(p_noise=0.0, bend_noise=0.0, bend_res=1e-6, bend_tau=1e-4, skin_dead=0.0, skin_noise=0.0, skin_gain_sd=0.0,
                     skin_xt=0.0, skin_tau=1e-4, skin_bits=24)
PCLEAN = C.PlungerSpec(enc_noise=0.0, enc_res=1e-9, load_noise=0.0, load_res=1e-6)


def _job(a):
    ripe, name, ratio, seed = a
    if name != ripe:
        C.OBJECTS[name] = C.OBJECTS[name]._replace(k=C.OBJECTS[ripe].k / ratio)
    # finger probe (noise-free sensors, raw truths kept)
    cell = S.Cell(seed=seed, objects=(name,))
    keep = {}

    def stop(c, e):
        keep["log"] = [dict(bend=x["bend_true"], tax=x["tax_true"]) for x in e.log if x["phase"] == "grip" and x.get("probing")]
        return 1.0
    S.run_episode(cell, name, probe="finger", probe_fn=stop, spec=CLEAN, max_retries=0)
    # plunger (noise-free)
    cell = S.Cell(seed=seed, objects=(name,))
    ep = C.Episode(name, "closed_loop", "learned")
    o = cell.obj_pos(name)
    cell.move_palm([o[0], o[1], C.CFG["table_top"] + 0.22], 1.0)
    cell.move_palm([o[0], o[1], C.grasp_height(name, o[2])], 0.8)
    pl = C.probe_plunger(cell, ep, PCLEAN, top_expected=o[2] + C.half_height(name))
    return dict(ripe=ripe, name=name, ratio=ratio, seed=seed, finger=keep.get("log", []), plunger=pl.get("samples", []))


def finger_feature(log, bend_noise, rng):
    """median over fingers of the slope of measured bend vs skin force (deg/N), sensors as online but with bend_noise"""
    if not log:
        return np.nan
    sens = C.Sensors(int(rng.integers(2**31)), C.SensorSpec(bend_noise=bend_noise))
    bend = np.array([sens.bend(x["bend"]) for x in log])
    tax = np.array([sens.skin(x["tax"]) for x in log]).reshape(len(log), C.NF, -1).sum(2)
    on = np.nonzero(tax.sum(1) > C.SensorSpec.contact_thr)[0]
    if not len(on):
        return np.nan
    sl = []
    for f in range(C.NF):
        idx = np.arange(on[0], len(log))
        idx = idx[tax[idx, f] > 0]
        if len(idx) >= 5 and np.ptp(tax[idx, f]) > 0.05:
            sl.append(np.polyfit(tax[idx, f], bend[idx, f], 1)[0])
    return float(np.median(sl)) if sl else np.nan


def plunger_feature(samples, enc_noise, rng, sp=C.PlungerSpec()):
    """slope of load vs indentation [N/mm] with encoder noise enc_noise and the online load cell model"""
    if not samples:
        return np.nan
    a = np.array(samples)
    x = np.round((a[:, 2] + rng.normal(0, enc_noise, len(a))) / sp.enc_res) * sp.enc_res
    f = np.maximum(0, np.round((a[:, 3] + rng.normal(0, sp.load_noise, len(a))) / sp.load_res) * sp.load_res)
    x = x - x[0]
    m = (f >= 0.05) & (f <= sp.F_probe)
    return float(np.polyfit(x[m] * 1e3, f[m], 1)[0]) if m.sum() >= 3 else np.nan


def population_bound(ratio, a=C.RANDOMISE["k"]):
    """the best accuracy any 1-D threshold on the TRUE stiffness can reach when each individual's stiffness is the nominal
    one times U(1-a, 1+a) (ripe: k0 * u, overripe: k0 / ratio * u'): the limit set by the spread of the population,
    not by the sensor.  Any monotone measurement of k (the plunger's k_p, with the hand in series) has the same bound."""
    t = np.linspace(0.2, 1.6, 28001)
    p_ripe_above = np.clip(((1 + a) - t) / (2 * a), 0, 1)
    p_over_below = np.clip((t * ratio - (1 - a)) / (2 * a), 0, 1)
    return float(np.max(0.5 * (p_ripe_above + p_over_below)))


def cv_accuracy(xr, xo, rng, folds=5):
    """5-fold CV over seeds of a 1-D threshold (midpoint rule fitted on the training folds, direction learned)"""
    n = min(len(xr), len(xo))
    idx = rng.permutation(n)
    acc = []
    for k in range(folds):
        te = idx[k::folds]
        tr = np.setdiff1d(idx, te)
        a, b = np.nanmean(xr[tr]), np.nanmean(xo[tr])
        thr, sign = 0.5 * (a + b), (1 if a > b else -1)
        pr = sign * (xr[te] - thr) > 0          # predicted ripe
        po = sign * (xo[te] - thr) <= 0         # predicted overripe
        acc.append((np.nansum(pr) + np.nansum(po)) / (2 * len(te)))
    return float(np.mean(acc))


def main(workers=2):
    jobs = [(r, n, ratio, s) for r, o in PAIRS for ratio in RATIOS for n in (r, o) if not (n == r and ratio != RATIOS[0]) for s in SEEDS]
    with Pool(workers) as pool:
        raw = pool.map(_job, jobs, chunksize=2)
    rng = np.random.default_rng(0)
    res = {"ratios": RATIOS, "bend_noise_deg": BEND_NOISE, "enc_noise_m": ENC_NOISE, "pairs": {}}
    for r, o in PAIRS:
        ripe = [x for x in raw if x["name"] == r]
        out = {"finger": {}, "plunger": {}}
        for ratio in RATIOS:
            over = [x for x in raw if x["name"] == o and x["ratio"] == ratio]
            for bn in BEND_NOISE:
                accs = []
                for _ in range(5):                       # Monte Carlo over the noise draws
                    xr = np.array([finger_feature(x["finger"], bn, rng) for x in ripe])
                    xo = np.array([finger_feature(x["finger"], bn, rng) for x in over])
                    accs.append(cv_accuracy(xr, xo, rng))
                out["finger"].setdefault(str(bn), []).append(round(float(np.mean(accs)), 3))
            for en in ENC_NOISE:
                accs = []
                for _ in range(5):
                    xr = np.array([plunger_feature(x["plunger"], en, rng) for x in ripe])
                    xo = np.array([plunger_feature(x["plunger"], en, rng) for x in over])
                    accs.append(cv_accuracy(xr, xo, rng))
                out["plunger"].setdefault(str(en), []).append(round(float(np.mean(accs)), 3))
        res["pairs"][f"{r}/{o}"] = out
        print(r, json.dumps(out), flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
    plot(res)


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams.update({"font.size": 11, "xtick.labelsize": 10, "ytick.labelsize": 10})
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), sharey=True)
    styles = {"finger": ("指プローブ（曲げ雑音）", ["#9ecae1", "#4292c6", "#08519c", "#08306b"]),
              "plunger": ("プランジャ（変位雑音）", ["#fdd0a2", "#fd8d3c", "#d94801", "#7f2704"])}
    for ax, (pair, d) in zip(axes, res["pairs"].items()):
        ax.axvspan(1.33, 3.1, color="#eeeeee", zorder=0)
        if pair.startswith("tomato"):
            ax.text(2.2, 0.375, "熟度段階のあいだの硬さ比（Vursavuş ら 2017）", ha="center", fontsize=9.5, color="#555")
        ax.axhline(0.5, color="#999", lw=0.8, ls=":")
        rr = np.linspace(1.0, 4.0, 301)
        ax.plot(rr, [population_bound(x) for x in rr], color="#b8b6ae", lw=5.0, alpha=0.6, zorder=1,
                label="個体差（剛性 ±20 %）だけで決まる上限")
        ax.text(3.98, 0.505, "偶然", ha="right", va="bottom", fontsize=9, color="#888")
        for probe, (lab, cols) in styles.items():
            for (lv, acc), c in zip(d[probe].items(), cols):
                unit = f"{float(lv):.1f}°" if probe == "finger" else f"{float(lv)*1e6:.0f} µm"
                online = (probe == "finger" and float(lv) == 0.3) or (probe == "plunger" and abs(float(lv) - 2e-6) < 1e-9)
                ax.plot(res["ratios"], acc, "-o" if probe == "plunger" else "--s", color=c, ms=3.5, lw=2.2 if online else 1.1,
                        label=f"{lab.split('（')[0]} {unit}" + "（想定）" * online)
        ax.set_title({"tomato/tomato_overripe": "中玉トマト vs 過熟", "strawberry/strawberry_overripe": "いちご vs 過熟"}[pair], fontsize=12)
        ax.set_xlabel("剛性比（熟した方 / 過熟）")
        ax.set_ylim(0.35, 1.02)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("見分ける正解率（5分割交差検証）")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, fontsize=9.5, loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    fig.savefig(REPO / "docs" / "figures" / "fig_probe_map.png", dpi=170)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot(json.load(open(OUT)))
    else:
        main()
