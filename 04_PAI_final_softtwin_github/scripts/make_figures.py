"""Paper / video figures from the MuJoCo results (python scripts/make_figures.py)."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sim_mujoco"))
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

INK, MUTE = "#1d2129", "#6b7280"
C_OK, C_CRUSH, C_DROP = "#2f6fdf", "#d64545", "#e0a106"


def fig_benchmark():
    d = json.load(open(REPO / "results" / "benchmark_mujoco.json"))
    order = [("open_loop_20kPa", "開ループ\n20 kPa固定"), ("open_loop_60kPa", "開ループ\n60 kPa固定"),
             ("closed_loop_linear", "閉ループ\n線形モデル"), ("closed_loop_learned", "閉ループ\n学習モデル（提案）")]
    S = d["summary"]
    x = np.arange(len(order))
    w = 0.26
    fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=200)
    for i, (key, lab, col) in enumerate([("success_rate", "成功", C_OK), ("crush_rate", "破損", C_CRUSH), ("drop_rate", "落下", C_DROP)]):
        v = [100 * S[k][key] for k, _ in order]
        b = ax.bar(x + (i - 1) * w, v, w, label=lab, color=col)
        for xi, vi in zip(x + (i - 1) * w, v):
            ax.text(xi, vi + 1.5, f"{vi:.0f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, [l for _, l in order])
    ax.set_ylabel("割合 [%]")
    ax.set_ylim(0, 112)
    ax.legend(ncol=3, frameon=False, loc="upper left", fontsize=9)
    n = S[order[0][0]]["n"]
    ax.set_title(f"把持・搬送の結果（各手法 n={n}，MuJoCo，合成筋肉モデル）", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(REPO / "results" / "fig_benchmark.png")
    plt.close(fig)


def fig_force_tracking():
    import softtwin_mj as S

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), dpi=200, sharey=True)
    for ax, act, title in [(axes[0], "learned", "学習モデル（提案）"), (axes[1], "linear", "線形モデル")]:
        cell = S.Cell(seed=0, objects=("tofu",))
        r, ep = S.run_episode(cell, "tofu", mode="closed_loop", actuator=act, max_retries=0)
        log = [e for e in ep.log if e["phase"] in ("inflate", "lift", "transport")]
        t = np.array([e["t"] for e in log])
        t -= t[0]
        ax.plot(t, [e["f_true"] for e in log], color=INK, lw=1.4, label="接触力の真値")
        ax.plot(t, [e["f_est"] for e in log], color=C_OK, lw=1.4, ls="--", label="推定値 $\\hat F$")
        ax.axhline(r["F_star"], color="#2a9d6a", lw=1, ls=":", label="目標 $F^*$")
        ax.axhline(S.OBJECTS["tofu"][4], color=C_CRUSH, lw=1, ls="-.", label="許容力")
        res = "成功" if r["success"] else ("落下" if not r["lifted"] else "失敗")
        ax.set_title(f"{title}：{res}", fontsize=10, color=INK)
        ax.set_xlabel("時間 [s]")
    axes[0].set_ylabel("把持力 [N]（豆腐）")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(REPO / "results" / "fig_force_tracking.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_benchmark()
    fig_force_tracking()
    print("figures written")
