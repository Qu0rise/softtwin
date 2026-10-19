"""Figures for the tactile extension -> results/fig_tactile.png, results/fig_lookalike.png, docs/figures/dataset_sample.png"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "tactile_dataset"
for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
    if Path(f).exists():
        font_manager.fontManager.addfont(f)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

JP = {"strawberry": "いちご", "egg": "卵", "tofu": "豆腐", "tomato": "トマト", "bolt": "ボルト",
      "tomato_overripe": "完熟トマト*", "sample_strawberry": "食品サンプル*"}
OBJ = list(JP)
CLS_JP = {"very_fragile": "極脆", "soft": "軟", "fragile": "脆", "rigid": "剛"}
C = {"vision": "#9aa3b2", "touch": "#4ea1ff", "fusion": "#1f6feb", "vlm_prior": "#9aa3b2", "vision_clf": "#c7ccd6",
     "touch_clf": "#4ea1ff", "fusion_clf": "#1f6feb", "oracle": "#3ecf8e"}


def load():
    metas = [json.loads(line) for line in open(DATA / "index.jsonl")]
    return metas


def pick(metas, obj, ok=True, F=None):
    for m in metas:
        if m["object"] == obj and m["boxes"].get("pregrasp") and (F is None or abs(m["F_target"] - F) < 0.35):
            return m
    return next(m for m in metas if m["object"] == obj)


def overlay(rgb, mask):
    """draw the auto-generated mask as a green outline"""
    img = rgb.copy()
    m = mask.astype(bool)
    edge = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
    img[edge] = [40, 230, 120]
    return img


def main():
    metas = load()
    clf = json.load(open(REPO / "results" / "tactile_classifier.json"))

    # ---------------- fig_tactile: 2 x 3
    fig = plt.figure(figsize=(10.5, 6.0))
    gs = fig.add_gridspec(2, 3, wspace=0.32, hspace=0.45)
    # (a) look-alikes in the wrist camera
    sub = gs[0, 0].subgridspec(2, 2, wspace=0.05, hspace=0.25)
    for i, o in enumerate(["tomato", "tomato_overripe", "strawberry", "sample_strawberry"]):
        ax = fig.add_subplot(sub[i // 2, i % 2])
        m = pick(metas, o)
        z = np.load(DATA / m["file"])
        bx = m["boxes"]["pregrasp"]
        cx, cy = bx[0] + bx[2] // 2, bx[1] + bx[3] // 2
        r = max(bx[2], bx[3])
        crop = overlay(z["pregrasp_rgb"], z["pregrasp_mask"])[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
        ax.imshow(crop)
        ax.set_title(JP[o], fontsize=8)
        ax.axis("off")
    fig.text(0.08, 0.93, "(a) 手先カメラ＋自動マスク：見た目は同じ", fontsize=9)
    # (b) skin heat maps: tomato vs overripe
    sub = gs[0, 1].subgridspec(2, 1, hspace=0.75)
    for i, o in enumerate(["tomato", "tomato_overripe"]):
        ax = fig.add_subplot(sub[i])
        m = pick(metas, o, F=1.8)
        z = np.load(DATA / m["file"])
        sk = z["skin"]
        pr = z["probing"]
        ax.imshow(sk.T, aspect="auto", cmap="magma", vmin=0, vmax=0.45, extent=[0, len(sk) * 0.02, 24, 0])
        end = pr.sum() * 0.02
        ax.axvline(end, color="#3ecf8e", lw=1)
        ax.set_ylabel("タクセル", fontsize=8)
        out = "破損" if m["outcome"]["crushed"] else "成功" if m["outcome"]["success"] else "落下"
        ax.set_title(f"{JP[o]}：F* = {m['F_target']:.1f} N → {out}", fontsize=8)
        if i == 1:
            ax.set_xlabel("把持開始からの時間 [s]（緑線：プローブ終了）", fontsize=8)
    fig.text(0.39, 0.93, "(b) 布状触覚スキン 24 点の時系列", fontsize=9)
    # (c) stiffness proxy per object
    ax = fig.add_subplot(gs[0, 2])
    for i, o in enumerate(OBJ):
        v = [m["probe_features"]["stiff"] for m in metas if m["object"] == o and m["probe_features"]["stiff"] > 0]
        ax.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-0.18, 0.18, len(v)), v, s=6,
                   color="#e5534b" if o in ("tomato_overripe", "sample_strawberry") else "#4ea1ff", alpha=0.7)
    ax.set_xticks(range(len(OBJ)), [JP[o] for o in OBJ], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("押し込み剛性 ΔF/Δθ [N/deg]")
    ax.set_title("(c) 0.4 N プローブで測る硬さ", fontsize=9, loc="left")
    # (d) classifier accuracy per object
    ax = fig.add_subplot(gs[1, 0:2])
    w = 0.26
    for k, key in enumerate(["vision", "touch", "fusion"]):
        mm = clf["models"][key]
        acc = [mm["per_object"][o] or 0 for o in OBJ]
        ax.bar(np.arange(len(OBJ)) + (k - 1) * w, acc, w, color=C[key],
               label={"vision": "画像のみ", "touch": "触覚のみ", "fusion": "画像＋触覚"}[key] + f"（全体 {mm['accuracy']:.0%}）")
    ax.set_xticks(range(len(OBJ)), [JP[o] for o in OBJ], fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("扱い方クラスの正解率（テスト）")
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(fontsize=8, ncol=3, loc="upper left", frameon=False)
    ax.set_title(f"(d) 扱い方クラスの分類（学習 {clf['n_train']}／評価 {clf['n_test']} 試行，*：見た目が同じで中身が違う）", fontsize=9, loc="left")
    # (e) force window from outcome labels
    ax = fig.add_subplot(gs[1, 2])
    for c, col in zip(["very_fragile", "soft", "fragile", "rigid"], ["#e5534b", "#f5a524", "#4ea1ff", "#3ecf8e"]):
        rows = clf["force_window"][c]
        ax.plot([r["F"] for r in rows], [r["success"] for r in rows], "-o", ms=3, color=col, label=CLS_JP[c])
    ax.set_xlabel("目標把持力 F* [N]")
    ax.set_ylabel("成功率（自動ラベル）")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=8, frameon=False, title="真のクラス", title_fontsize=8)
    ax.set_title("(e) 結果ラベルから見る安全な力の窓", fontsize=9, loc="left")
    fig.savefig(REPO / "results" / "fig_tactile.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------------- fig_lookalike
    b = json.load(open(REPO / "results" / "benchmark_lookalike.json"))
    S = b["summary"]
    pol = ["vlm_prior", "vision_clf", "touch_clf", "fusion_clf", "oracle"]
    lab = {"vlm_prior": "VLM事前\n知識のみ", "vision_clf": "画像\n分類器", "touch_clf": "触覚\n分類器", "fusion_clf": "画像＋触覚\n（提案）", "oracle": "真値クラス\n（上限）"}
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.0), gridspec_kw={"width_ratios": [1.0, 1.6]})
    x = np.arange(len(pol))
    axs[0].bar(x - 0.2, [S[p]["success"] for p in pol], 0.4, color="#1f6feb", label="成功")
    axs[0].bar(x + 0.2, [S[p]["crushed"] for p in pol], 0.4, color="#e5534b", label="破損")
    axs[0].set_xticks(x, [lab[p] for p in pol], fontsize=7.5)
    axs[0].set_ylim(0, 1.05)
    axs[0].legend(fontsize=8, frameon=False)
    axs[0].set_title(f"(a) 7 物体 × {len(b['seeds'])} シード（見た目が同じ物体を含む）", fontsize=9, loc="left")
    w = 0.16
    for k, p in enumerate(pol):
        axs[1].bar(np.arange(len(OBJ)) + (k - 2) * w, [S[p]["by_object"][o] for o in OBJ], w, color=C[p], label=lab[p].replace("\n", ""))
    axs[1].set_xticks(range(len(OBJ)), [JP[o] for o in OBJ], fontsize=8, rotation=20)
    axs[1].set_ylim(0, 1.05)
    axs[1].set_ylabel("成功率")
    axs[1].legend(fontsize=7, ncol=5, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0))
    axs[1].set_title("(b) 物体別", fontsize=9, loc="left", pad=18)
    fig.savefig(REPO / "results" / "fig_lookalike.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------------- dataset sample sheet (one row per object: wrist RGB / mask / depth / scene)
    fig, axs = plt.subplots(len(OBJ), 4, figsize=(6.4, 1.25 * len(OBJ)))
    for i, o in enumerate(OBJ):
        m = pick(metas, o)
        z = np.load(DATA / m["file"])
        d = z["pregrasp_depth"].astype(float)
        for j, (img, kw) in enumerate([(z["pregrasp_rgb"], {}), (z["pregrasp_mask"], {"cmap": "gray"}), (d, {"cmap": "viridis"}),
                                       (overlay(z["grasp_rgb"], z["grasp_mask"]), {})]):
            axs[i, j].imshow(img, **kw)
            axs[i, j].axis("off")
        axs[i, 0].text(-8, 72, JP[o], fontsize=7, ha="right", va="center")
    for j, t in enumerate(["手先 RGB", "マスク", "深度", "把持中（マスク）"]):
        axs[0, j].set_title(t, fontsize=8)
    fig.savefig(REPO / "docs" / "figures" / "dataset_sample.png", dpi=180, bbox_inches="tight")
    print("figures written")


if __name__ == "__main__":
    main()
