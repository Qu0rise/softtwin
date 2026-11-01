"""Figures of the report from the committed result files  ->  docs/figures/fig_*.png

    python scripts/make_figures.py            # all figures whose inputs exist
    python scripts/make_figures.py q1 q2      # a subset

Colours: categorical slots in a fixed order for model identity; outcome colours (success / drop / crush) are the
status colours and always come with a label.  One y-axis per panel, recessive grid, direct labels where few series.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
R, FIG = REPO / "results", REPO / "docs" / "figures"
sys.path[:0] = [str(REPO / "sim_mujoco"), str(REPO / "learning"), str(REPO / "scripts")]

INK, INK2, MUTE, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
SER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OK, WARN, SERIOUS, CRIT = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"
plt.rcParams.update({
    "font.family": ["Noto Sans CJK JP", "DejaVu Sans"], "font.size": 9, "axes.edgecolor": MUTE, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 2.0,
    "savefig.dpi": 200, "savefig.bbox": "tight", "mathtext.fontset": "dejavusans"})

CFG_LABEL = {"open_loop_recipe": "内圧表（開ループ）", "closed_affine": "閉ループ・直線",
             "closed_static": "閉ループ・静特性の表", "closed_greybox": "閉ループ・グレーボックス", "closed_greybox_play": "閉ループ・グレーボックス＋遊び",
             "closed_learned": "閉ループ・学習（提案）", "closed_learned_tip": "同・作用点を指先に固定", "closed_learned_nocontactskin": "同・接触判定も推定値",
             "closed_ideal_sensor": "同・真の力で帰還"}
POL_LABEL = {"visual_prior": "見た目だけ", "soft_start": "弱く握り始める", "finger_clf": "指プローブで決める",
             "plunger_clf": "プランジャで決める（提案）", "plunger_clf_tip": "同・作用点を指先に固定", "plunger_clf_mass": "同・重さの下限を加える", "force_window": "力の窓",
             "force_window_loo": "力の窓（その物体を学習に含めない）", "oracle": "正解の区分を与える（判断の上限）"}
OBJ_LABEL = {"strawberry": "いちご", "egg": "卵", "tofu": "豆腐", "tomato": "トマト", "bolt": "ボルト",
             "tomato_overripe": "過熟トマト", "strawberry_overripe": "過熟いちご"}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def outcome_bars(ax, rows_by, order, labels, title=None, crush_x=114.5):
    """horizontal 100 % bars: success / slip+drop / crush, with the success-rate CI as a whisker"""
    y = np.arange(len(order))[::-1]
    for yi, key in zip(y, order):
        if key not in rows_by:                                   # keeps the rows aligned with a neighbouring panel
            ax.text(2, yi, "（このエンジンでは評価していない）", va="center", fontsize=8, color=MUTE)
            continue
        rs = rows_by[key]
        n = len(rs)
        s = sum(r["success"] for r in rs)
        cr = sum(r["crushed"] for r in rs)
        dr = n - s - cr
        left = 0
        for v, col in ((s, OK), (dr, WARN), (cr, CRIT)):
            if v:
                ax.barh(yi, 100 * v / n, left=left, height=0.62, color=col, edgecolor="white", linewidth=1.2)
            left += 100 * v / n
        lo, hi = wilson(s, n)
        ax.plot([100 * lo, 100 * hi], [yi + 0.42] * 2, color=INK, lw=1.0)
        for xx in (100 * lo, 100 * hi):
            ax.plot([xx, xx], [yi + 0.36, yi + 0.48], color=INK, lw=1.0)
        ax.text(101.5, yi, f"{100 * s / n:.1f} %", va="center", ha="left", fontsize=8.5, color=INK)
        ax.text(crush_x, yi, f"破損 {cr}", va="center", ha="left", fontsize=8, color=CRIT if cr else MUTE)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[k] for k in order])
    for tl in ax.get_yticklabels():                     # the configuration the report proposes
        if "提案" in tl.get_text():
            tl.set_color(SER[0]), tl.set_fontweight("bold")
    ax.set_xlim(0, 100)
    ax.set_xlabel("試行の割合 [%]（ひげ：成功率の 95 % 信頼区間）")
    ax.grid(axis="y", visible=False)
    if title:
        ax.set_title(title, loc="left", fontsize=9.5, color=INK)


def outcome_legend(fig, y=0.0):
    from matplotlib.patches import Patch
    h = [Patch(color=OK, label="成功"), Patch(color=WARN, label="滑り・落下"), Patch(color=CRIT, label="破損（許容力を超えた）")]
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, y), fontsize=8.5)


# ---------------------------------------------------------------- actuator models
def fig_actuator():
    import twin_core as C
    from actuator_net import SyntheticMuscle

    P = np.linspace(0, 45, 91)
    plant = SyntheticMuscle(noise=0.0)
    rise = [plant.static(p, True) for p in P]
    fall = [plant.static(p, False) for p in P]
    models = {}
    for kind in ("affine", "static", "greybox", "learned"):
        a = C.Actuator(kind)
        out = []
        for p in P:                                 # staircase: hold each level 3 s, read the settled value
            for _ in range(150):
                v = a(p, 0.02)
            out.append(v)
        models[kind] = out
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.4, 3.6), gridspec_kw=dict(width_ratios=[1, 1.35]))
    ax.fill_between(P, rise, fall, color=GRID, label="模擬筋肉（上昇と下降の枝の間）")
    ax.plot(P, rise, color=INK, lw=1.2, label="模擬筋肉・上昇の枝")
    for k, c, lab in (("affine", SER[1], "直線モデル"), ("static", SER[4] if len(SER) > 4 else "#8a8984", "静特性の表"),
                      ("greybox", SER[2], "グレーボックス"), ("learned", SER[0], "学習モデル（MLP）")):
        ax.plot(P, models[k], color=c, lw=1.6, label=lab, ls="--" if k == "affine" else "-")
    rec = json.load(open(R / "open_loop_recipe.json"))
    ax.axvspan(min(rec.values()), max(rec.values()), color=SER[3], alpha=0.12, lw=0)
    ax.text(0.5 * (min(rec.values()) + max(rec.values())), 6, "把持で使う\n内圧の範囲", ha="center", fontsize=8.6, color=INK2)
    ax.set_xlabel("内圧 p [kPa]")
    ax.set_ylabel("自由曲げ角（定常）[°]")
    ax.set_xlim(0, 45)
    ax.set_ylim(0, 85)
    ax.legend(fontsize=8.0, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False, handlelength=1.6)
    ax.set_title("(a) 定常特性", loc="left", fontsize=10.5)
    # dynamic test: grip-like pressure profile, models run autoregressively on the measured pressure
    from actuator_net import pressure_profile
    for sd in range(7, 500):                         # the first held-out seed that draws the grip-like kind (kind 3)
        if np.random.default_rng(sd).integers(0, 4) != 3:
            continue
        t, p = pressure_profile(np.random.default_rng(sd), T=12.0)
        if 15 < p.max() < 30 and (p > 1).mean() > 0.6:
            break
    plant = SyntheticMuscle(noise=0.0)
    truth = np.array([plant.step(x, 0.02) for x in p])
    ac = json.load(open(R / "actuator_compare.json"))
    rec_lo, rec_hi = ac["grip_range_kPa"]
    grip = (p >= rec_lo) & (p <= rec_hi) & (np.gradient(p, 0.02) > -0.5)
    edges = np.flatnonzero(np.diff(np.r_[0, grip.astype(int), 0]))
    for a0, a1 in zip(edges[::2], edges[1::2]):
        if t[min(a1, len(t) - 1)] - t[a0] >= 0.5:             # mark the longer stretches only
            bx.axvspan(t[a0], t[min(a1, len(t) - 1)], color=SER[3], alpha=0.14, lw=0)
    bx.axhline(0, color=INK, lw=1.0)
    for k, c, lab in (("static", SER[4], "静特性の表"), ("greybox", SER[2], "グレーボックス"), ("greybox_play", SER[2], "グレーボックス＋遊び"),
                      ("learned", SER[0], "学習モデル")):
        a = C.Actuator(k)
        y = np.array([a(x, 0.02) for x in p])
        bx.plot(t, y - truth, color=c, lw=1.4, ls=":" if k == "greybox_play" else "-",
                label=f"{lab}（{ac[k]['grasp_bias_deg']:+.1f}°）".replace("-", "\u2212"))
    bx.plot([], [], color=SER[1], lw=1.4, ls="--", label=f"直線モデル（{ac['affine']['grasp_bias_deg']:+.1f}°，枠外）".replace("-", "\u2212"))
    bx.text(0.7, -5.4, "網掛け：把持の内圧域で増圧・保持中", fontsize=8, color=INK2, va="center")
    bx.set_xlabel("時間 [s]")
    bx.set_ylabel("自由曲げの誤差（モデル − 真値）[°]")
    bx.set_ylim(-6, 6)
    bx2 = bx.twinx()
    bx2.plot(t, p, color="#8a8984", lw=0.9, ls=":")
    bx2.set_ylabel("内圧 [kPa]（点線）", color="#8a8984")
    bx2.set_ylim(0, 60)
    bx2.tick_params(axis="y", colors="#8a8984")
    bx.legend(fontsize=8.0, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False, handlelength=1.6)
    bx.set_title("(b) 把持に似た内圧での自由曲げの誤差（学習に使っていない系列）", loc="left", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig_actuator.png")
    plt.close(fig)


# ---------------------------------------------------------------- Q1
def fig_q1():
    d = json.load(open(R / "benchmark_mujoco.json"))
    by = {}
    for r in d["rows"]:
        by.setdefault(r["config"], []).append(r)
    order = ["open_loop_recipe", "closed_affine", "closed_static", "closed_greybox", "closed_greybox_play", "closed_learned", "closed_learned_tip",
             "closed_ideal_sensor"]                                     # same order as Table 5
    phys = None
    if (R / "benchmark_isaac.json").exists():
        b = json.load(open(R / "benchmark_isaac.json"))
        phys = {}
        for r in b["rows"]:
            phys.setdefault(r["config"], []).append(r)
    ncol = 2 if phys else 1
    fig, axes = plt.subplots(1, ncol, figsize=(4.9 * ncol + 1.6, 3.3), sharey=True)
    axes = np.atleast_1d(axes)
    import mujoco
    outcome_bars(axes[0], by, [k for k in order if k in by], CFG_LABEL,
                 f"MuJoCo {mujoco.__version__}（5 物体 × 30 シード，各 n = {len(by['closed_learned'])}）", crush_x=119)
    if phys:
        outcome_bars(axes[1], phys, [k for k in order if k in by], CFG_LABEL,
                     f"Isaac Sim 5.0 / PhysX（5 物体 × 20 シード，各 n = {len(phys['closed_learned'])}）", crush_x=119)
    for ax in axes:
        ax.set_xlim(0, 136)
        ax.set_xticks([0, 25, 50, 75, 100])
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    outcome_legend(fig)
    fig.savefig(FIG / "fig_q1.png")
    plt.close(fig)


def fig_force_trace(seed=0, name="strawberry"):
    """same strawberry, same seed: straight-line actuator model vs learned model; true and estimated grip force"""
    import softtwin_mj as S
    import twin_core as C
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.0), sharey=True)
    for ax, act, lab in zip(axes, ("affine", "learned"), ("直線モデル", "学習モデル")):
        cell = S.Cell(seed=seed, objects=(name,))
        r, ep = S.run_episode(cell, name, actuator=act, max_retries=0)
        L = [e for e in ep.log if e["phase"] in C.HELD]
        t = np.array([e["t"] for e in L])
        t0 = t[0]
        ft, fe = np.array([e["f_true"] for e in L]), np.array([e["f_est"] for e in L])
        ph = [e["phase"] for e in L]
        for p_, c in (("lift", "#f4f3ef"), ("transport", "#ecebe6"), ("lower", "#f4f3ef")):
            idx = [i for i, x in enumerate(ph) if x == p_]
            if idx:
                ax.axvspan(t[idx[0]] - t0, t[idx[-1]] - t0, color=c, lw=0)
                ax.text(0.5 * (t[idx[0]] + t[idx[-1]]) - t0, 3.88, {"lift": "持ち上げ", "transport": "搬送", "lower": "載置"}[p_],
                        ha="center", fontsize=8.5, color=INK2)
        ax.axhline(C.OBJECTS[name].fmax, color=CRIT, lw=1.0, ls=(0, (4, 2)))
        ax.text(0.2, C.OBJECTS[name].fmax + 0.06, f"許容力 {C.OBJECTS[name].fmax:.1f} N", color=CRIT, fontsize=7.5)
        ax.axhline(C.HANDLING[C.OBJECTS[name].cls]["target"], color=MUTE, lw=0.9, ls=":")
        ax.text(0.2, C.HANDLING[C.OBJECTS[name].cls]["target"] - 0.2, "目標 2.2 N", color=INK2, fontsize=7.5)
        ax.plot(t - t0, ft, color=INK, lw=1.8, label="真の把持力（シミュレータ）")
        ax.plot(t - t0, fe, color=SER[0], lw=1.4, label="推定値（力センサなし）")
        verdict = "破損" if r["crushed"] else ("成功" if r["success"] else r["failure"])
        ax.set_title(f"{lab}：{verdict}（最大 {r['peak_force_true']:.2f} N）", loc="left", fontsize=9.5,
                     color=CRIT if r["crushed"] else (OK if r["success"] else INK))
        ax.set_xlabel("把持開始からの時間 [s]")
        ax.set_ylim(0, 4.05)
    xm = max(ax.get_xlim()[1] for ax in axes)
    for ax in axes:
        ax.set_xlim(0, xm)
        ax.set_xticks(np.arange(0, xm + 0.01, 2))
    axes[0].set_ylabel("把持力 [N]")
    axes[1].legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig_force_trace.png")
    plt.close(fig)


# ---------------------------------------------------------------- Q2
def fig_q2():
    d = json.load(open(R / "benchmark_lookalike.json"))
    import twin_core as C
    by, la = {}, {}
    for r in d["rows"]:
        by.setdefault(r["policy"], []).append(r)
        if r["object"] in C.LOOKALIKES:
            la.setdefault(r["policy"], []).append(r)
    order = [p for p in ("oracle", "plunger_clf", "plunger_clf_tip", "plunger_clf_mass", "force_window", "force_window_loo", "finger_clf", "soft_start", "visual_prior") if p in by]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.2), sharey=True)
    outcome_bars(axes[0], by, order, POL_LABEL, f"7 物体すべて（各 n = {len(by[order[0]])}）")
    outcome_bars(axes[1], la, order, POL_LABEL, f"見た目が同じ 2 物体だけ（各 n = {len(la[order[0]])}）", crush_x=117)
    axes[0].set_xlim(0, 128)
    axes[1].set_xlim(0, 150)
    for ax in axes:
        ax.set_xticks([0, 25, 50, 75, 100])
    for yi, p in zip(np.arange(len(order))[::-1], order):
        ct = np.mean([r["cycle_time_s"] for r in by[p]])
        axes[1].text(149, yi, f"{ct:.1f} s", va="center", ha="right", fontsize=8, color=INK2)
    axes[1].text(149, len(order) - 0.4, "1 回の時間", ha="right", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    outcome_legend(fig)
    fig.savefig(FIG / "fig_q2.png")
    plt.close(fig)


# ---------------------------------------------------------------- mismatch
def fig_robustness():
    d = json.load(open(R / "robustness.json"))
    S = d["summary"]
    show = [("open_loop_recipe", SER[3], "内圧表（開ループ）", "--"), ("closed_greybox", SER[2], "グレーボックス", ":"),
            ("closed_greybox+cal", SER[2], "グレーボックス＋空中較正", "-"), ("closed_learned", SER[0], "学習モデル", ":"),
            ("closed_learned+cal", SER[0], "学習モデル＋空中較正", "-")]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.4, 3.1), sharey=True, gridspec_kw=dict(width_ratios=[1.5, 1]))
    xs = [0.0, 0.1, 0.2, 0.3]
    for k, c, lab, ls in show:
        v = [100 * S[f"muscle={x}"][k]["success"] / S[f"muscle={x}"][k]["n"] for x in xs]
        ax.plot([100 * x for x in xs], v, color=c, ls=ls, marker="o", ms=4, lw=1.6, label=lab)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_xticklabels(["0", "±10", "±20", "±30"])
    ax.set_xlabel("筋肉の特性のずれ（6 つの定数を独立に）[%]")
    ax.set_ylabel("成功率 [%]（各点 n = 50）")
    ax.set_ylim(60, 101)
    ax.legend(fontsize=7.5, loc="lower left")
    ax.set_title("(a) 人工筋肉の定数のずれ", loc="left", fontsize=9.5)
    ks = ["k_b=0.8", "muscle=0.0", "k_b=1.25"]
    for k, c, lab, ls in show:
        v = [100 * S[x][k]["success"] / S[x][k]["n"] for x in ks]
        bx.plot([0.8, 1.0, 1.25], v, color=c, ls=ls, marker="o", ms=4, lw=1.6)
    for k, name in (("closed_learned", "学習"), ("closed_learned+cal", "学習＋空中較正")):
        for xx, x in zip([0.8, 1.0, 1.25], ks):
            c_ = S[x][k]["crushed"]
            if c_:
                bx.text(xx + 0.025, 100 * S[x][k]["success"] / S[x][k]["n"] - (1.6 if "cal" in k else 0), f"破損 {c_}（{name}）",
                        fontsize=7.5, color=CRIT, ha="left", va="center")
    bx.set_xlim(0.76, 1.52)
    bx.set_xticks([0.8, 1.0, 1.25])
    bx.set_xlabel("指の曲げ剛性の比　$k_b^{\\mathrm{true}}/k_b^{\\mathrm{model}}$")
    bx.set_title("(b) 指の曲げ剛性 $k_b$ のずれ", loc="left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig_robustness.png")
    plt.close(fig)


# ---------------------------------------------------------------- Q3 dataset sample
def fig_dataset():
    """one auto-annotated episode per look-alike pair: wrist RGB + mask, plunger curves, skin over time"""
    D = REPO / "data" / "tactile_dataset"
    metas = [json.loads(x) for x in open(D / "index.jsonl")]
    pick = {}
    for m in metas:
        if m["object"] in ("tomato", "tomato_overripe") and m["object"] not in pick and m["outcome"]["failure"] in ("ok", "crushed"):
            pick[m["object"]] = m
    a, b = pick["tomato"], pick["tomato_overripe"]
    za, zb = np.load(D / a["file"]), np.load(D / b["file"])
    fig = plt.figure(figsize=(10.5, 3.3))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.25, 1.25, 1.1, 1.4], wspace=0.42)
    ax = fig.add_subplot(gs[0])
    ax.imshow(za["pregrasp_rgb"])
    ax.set_title("(a) 手先カメラ（把持の直前）", loc="left", fontsize=8.5)
    ax.axis("off")
    ax = fig.add_subplot(gs[1])
    over = za["pregrasp_rgb"].astype(float) / 255
    msk = za["pregrasp_mask"].astype(bool)
    over[msk] = 0.45 * over[msk] + 0.55 * np.array([0.11, 0.69, 0.48])
    ax.imshow(over)
    ax.set_title("(b) 自動で作ったマスク", loc="left", fontsize=8.5)
    ax.axis("off")
    ax = fig.add_subplot(gs[2])
    for z, c, lab in ((za, SER[0], "中玉トマト"), (zb, SER[1], "過熟")):
        i0 = int(np.argmax(z["plunger_load_true"] > 0.01))
        x = (z["plunger_ext"][i0:] - z["plunger_ext"][i0]) * 1e3
        ax.plot(x, z["plunger_load_true"][i0:], color=c, lw=1.8, label=lab)
        ax.plot(x, z["plunger_load"][i0:], color=c, lw=0.7, alpha=0.55)
    ax.set_xlabel("接触からの押し込み [mm]")
    ax.set_ylabel("荷重 [N]")
    ax.set_title("(c) 硬さプランジャ", loc="left", fontsize=8.5)
    ax.legend(fontsize=7.5, loc="lower right")
    ax = fig.add_subplot(gs[3])
    t = za["t"] - za["t"][0]
    ax.imshow(za["skin"].T, aspect="auto", cmap="Blues", extent=(t[0], t[-1], 23.5, -0.5), vmin=0, vmax=max(0.3, float(za["skin"].max())))
    ax.set_yticks([3.5, 11.5, 19.5])
    ax.set_yticklabels(["指 1", "指 2", "指 3"])
    for y in (7.5, 15.5):
        ax.axhline(y, color="white", lw=1)
    ax.set_xlabel("把持開始からの時間 [s]")
    ax.set_title("(d) 布スキン 24 点（センサの値）", loc="left", fontsize=8.5)
    ax.grid(False)
    fig.savefig(FIG / "dataset_sample.png")
    plt.close(fig)


# ---------------------------------------------------------------- one-page summary: few rows, large type
def fig_poster():
    import twin_core as C
    w = 3.45                                                   # inches: one column of the A4 landscape summary
    with plt.rc_context({"font.size": 8, "axes.titlesize": 8.5}):
        # Q1 (MuJoCo): the four controllers that carry the argument
        d = json.load(open(R / "benchmark_mujoco.json"))
        by = {}
        for r in d["rows"]:
            by.setdefault(r["config"], []).append(r)
        order = ["open_loop_recipe", "closed_affine", "closed_greybox", "closed_learned"]
        fig, ax = plt.subplots(figsize=(w, 1.55))
        outcome_bars(ax, by, order, CFG_LABEL, None, crush_x=141)
        ax.set_xlim(0, 172)
        ax.set_xticks([0, 50, 100])
        ax.set_xlabel(f"試行の割合 [%]（各 n = {len(by['closed_learned'])}）")
        fig.tight_layout()
        fig.savefig(FIG / "poster_q1.png", dpi=300)
        plt.close(fig)
        # Q2: look-alikes only
        d = json.load(open(R / "benchmark_lookalike.json"))
        la = {}
        for r in d["rows"]:
            if r["object"] in C.LOOKALIKES:
                la.setdefault(r["policy"], []).append(r)
        order = ["visual_prior", "finger_clf", "plunger_clf", "oracle"]
        fig, ax = plt.subplots(figsize=(w, 1.45))
        outcome_bars(ax, la, order, POL_LABEL, None, crush_x=162)
        ax.set_xlim(0, 178)
        ax.set_xticks([0, 50, 100])
        ax.set_xlabel(f"見た目が同じ 2 物体の試行の割合 [%]（各 n = {len(la['plunger_clf'])}）")
        fig.tight_layout()
        fig.savefig(FIG / "poster_q2.png", dpi=300)
        plt.close(fig)
        # map: tomato pair, the assumed noise only
        pm = json.load(open(R / "probe_sensitivity.json"))
        d = pm["pairs"]["tomato/tomato_overripe"]
        fig, ax = plt.subplots(figsize=(w, 2.0))
        ax.axvspan(1.33, 3.1, color="#eeeeee", zorder=0)
        ax.text(2.2, 0.885, "熟度段階のあいだの比", ha="center", fontsize=7, color=INK2)
        fk = min(d["finger"], key=lambda k: abs(float(k) - 0.3))
        pk = min(d["plunger"], key=lambda k: abs(float(k) - 2e-6))
        ax.plot(pm["ratios"], d["finger"][fk], "--s", color="#08519c", ms=3, lw=1.6, label="指プローブ（曲げ雑音 0.3°）")
        ax.plot(pm["ratios"], d["plunger"][pk], "-o", color="#d94801", ms=3, lw=1.6, label="プランジャ（変位雑音 2 µm）")
        ax.axhline(0.5, color="#999", lw=0.7, ls=":")
        ax.set_ylim(0.45, 1.02)
        ax.set_xlabel("剛性比（熟した中玉トマト / 過熟）")
        ax.set_ylabel("正解率")
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.45, -0.32), ncol=2, frameon=False, handlelength=2.2)
        fig.tight_layout()
        fig.savefig(FIG / "poster_map.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def fig_force_window():
    """the force window of each object vs its size: below the friction floor it slips, above the allowable force it is
    crushed.  Floor: S m(g+a)/mu with the nominal mass scaled by s^3 (S read off the nominal sizes, size_shift.py);
    the band shows the individual spread of mass (+-15 %) and friction (+-20 %).  Dots: closed loop at the class target."""
    import twin_core as C
    ss = json.load(open(R / "size_shift.json"))
    objs = list(ss["objects"])
    fig, axes = plt.subplots(1, len(objs), figsize=(10.4, 3.1), sharey=True)
    sc = np.linspace(0.7, 1.3, 61)
    g = 9.81 + C.A_MAX
    for ax, o in zip(axes, objs):
        ob, h = C.OBJECTS[o], C.HANDLING[C.OBJECTS[o].cls]
        mid = C.FLOOR_SAFETY * ob.mass * sc ** 3 * g / ob.mu
        lo = C.FLOOR_SAFETY * ob.mass * 0.85 * sc ** 3 * g / (ob.mu * 1.2)
        hi = C.FLOOR_SAFETY * ob.mass * 1.15 * sc ** 3 * g / (ob.mu * 0.8)
        ax.fill_between(sc, lo, hi, color=SER[3], alpha=0.25, lw=0)
        ax.plot(sc, mid, color="#b26b00", lw=1.6, label=f"落とさない下限 {C.FLOOR_SAFETY:g}·m(g+a)/μ")
        ax.axhline(h["limit"], color=CRIT, lw=1.4, label="区分の許容力")
        if abs(ob.fmax - h["limit"]) > 1e-6:
            ax.axhline(ob.fmax, color=CRIT, lw=1.0, ls=(0, (4, 2)), label="真の許容力")
        ax.axhline(h["target"], color=INK2, lw=0.9, ls=":", label="区分の目標")
        for s_ in ss["scales"]:
            k_ok, k_cr, _ = ss["summary"][f"closed_learned@{s_:g}"]["by_object"][o]
            frac = k_ok / ss["seeds"]
            ax.scatter([s_], [h["target"]], s=70, color=plt.cm.RdYlGn(frac), edgecolor=INK, lw=0.6, zorder=5)
            ax.annotate(f"{k_ok}/{ss['seeds']}", (s_, h["target"]), xytext=(0, -15), textcoords="offset points", ha="center", fontsize=8,
                        bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.9), zorder=6)
            r_ok, r_cr, _ = ss["summary"][f"open_loop_recipe@{s_:g}"]["by_object"][o]
            if r_cr:
                ax.annotate(f"内圧表：破損 {r_cr}", (s_, h["limit"]), xytext=(-4, 6), textcoords="offset points", ha="right",
                            fontsize=8, color=CRIT, bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.9), zorder=6)
        ax.set_title(OBJ_LABEL.get(o, o), fontsize=10)
        ax.set_xlabel("寸法の倍率 s")
        ax.set_xlim(0.7, 1.3)
        ax.set_ylim(0, 5.4)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("把持力 [N]")
    h_, l_ = axes[2].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=4, fontsize=8.5, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(FIG / "fig_force_window.png")
    plt.close(fig)


ALL = {"dataset": fig_dataset, "actuator": fig_actuator, "q1": fig_q1, "trace": fig_force_trace, "q2": fig_q2, "robustness": fig_robustness, "poster": fig_poster, "window": fig_force_window}

if __name__ == "__main__":
    want = sys.argv[1:] or list(ALL)
    for k in want:
        try:
            ALL[k]()
            print("ok", k)
        except FileNotFoundError as e:
            print("skip", k, e)
