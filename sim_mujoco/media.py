"""Short clips for the README and the video: rendered grasp + the grip force under it, two runs side by side.

    python sim_mujoco/media.py            -> media/*.gif, media/*.mp4

Every clip is simulation and says so on the frame.  A PhysX run (recorded in Isaac Sim) is drawn with MuJoCo's
renderer from the recorded poses (sim_mujoco/replay.py) and is labelled that way.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import os  # noqa: E402

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import softtwin_mj as S  # noqa: E402
import twin_core as C  # noqa: E402

REPO = C.REPO
MEDIA = REPO / "media"
FONT_R = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_B = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
INK, MUTE, OK, CRIT, BLUE = (11, 11, 11), (110, 110, 105), (12, 163, 12), (208, 59, 59), (42, 120, 214)
BG = (252, 252, 251)


def font(size, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT_R, size, index=0)


def mujoco_run(seed, name, cam="track", size=(640, 360), chooser=None, **kw):
    """chooser='plunger': the deployed policy of Q2 (vision features + firmness plunger -> class -> F*)"""
    cell = S.Cell(seed=seed, objects=(name,), render=True, cam=cam, size=size)
    if chooser == "plunger":
        import tactile as T
        import tactile_dataset as TD
        cams = TD.Cameras(cell, name)
        clf = T.Softmax.from_json(json.load(open(REPO / "learning" / "tactile_classifier.json"))["plunger"])

        def choose(c, e):
            pr = clf.predict_proba(np.r_[TD.vision_feat(cams), T.plunger_features(e)])[0]
            return C.TARGET_FORCE[T.CLASSES[int(pr.argmax())]]
        kw.update(probe="plunger", probe_fn=choose)
    r, ep = S.run_episode(cell, name, **kw)
    log = [dict(t=e["t"], phase=e["phase"], f_true=e.get("f_true", 0.0), f_est=e.get("f_est", 0.0),
                **({"load": e["load"]} if "load" in e else {})) for e in ep.log]
    return dict(frames=cell.frames, t=cell.frame_t, log=log, r=r, name=name)


def physx_run(path, cam="track", size=(640, 360)):
    import replay
    fr, ts, d = replay.frames(path, cam, size)
    return dict(frames=fr, t=ts, log=d["log"], r=None, name=d["object"])


class ForcePanel:
    """grip force vs time drawn once; a cursor and the current value are added per frame"""

    def __init__(self, run, w, h, fmax, target, t0, t1, ymax):
        self.w, self.h, self.t0, self.t1, self.ymax = w, h, t0, t1, ymax
        self.ml, self.mr, self.mt, self.mb = 46, 12, 10, 26
        img = Image.new("RGB", (w, h), BG)
        d = ImageDraw.Draw(img)
        f = font(13)
        for v in np.arange(0, ymax + 1e-9, 1.0):
            y = self.y(v)
            d.line([(self.ml, y), (w - self.mr, y)], fill=(230, 229, 224))
            d.text((self.ml - 8, y), f"{v:.0f}", font=f, fill=MUTE, anchor="rm")
        d.text((6, self.mt), "N", font=f, fill=MUTE)
        if fmax <= ymax:                                       # an allowable force above the axis is not drawn
            y = self.y(fmax)
            for x in range(self.ml, w - self.mr, 10):
                d.line([(x, y), (x + 5, y)], fill=CRIT, width=2)
            d.text((w - self.mr - 4, y - 3), f"許容 {fmax:.1f} N", font=font(13), fill=CRIT, anchor="rb")
        if target:
            y = self.y(target)
            for x in range(self.ml, w - self.mr, 6):
                d.line([(x, y), (x + 2, y)], fill=MUTE, width=1)
            d.text((self.ml + 4, y - 2), f"目標 {target:g} N", font=font(12), fill=MUTE, anchor="lb")
        self.base = img
        self.log = [e for e in run["log"] if t0 <= e["t"] <= t1]

    def x(self, t):
        return self.ml + (t - self.t0) / (self.t1 - self.t0) * (self.w - self.ml - self.mr)

    def y(self, v):
        return self.h - self.mb - min(v, self.ymax) / self.ymax * (self.h - self.mt - self.mb)

    def at(self, t):
        img = self.base.copy()
        d = ImageDraw.Draw(img)
        pts = [e for e in self.log if e["t"] <= t]
        if len(pts) > 1:
            d.line([(self.x(e["t"]), self.y(e["f_est"])) for e in pts], fill=BLUE, width=2)
            d.line([(self.x(e["t"]), self.y(e["f_true"])) for e in pts], fill=INK, width=3)
            v = pts[-1]["f_true"]
            if v > 0.05:
                d.text((min(self.x(pts[-1]["t"]) + 6, self.w - self.mr - 58), self.y(v) - 2), f"{v:.2f} N", font=font(13, True), fill=INK, anchor="lb")
        d.text((self.ml, self.h - 4), "黒：真の把持力　青：推定値（力センサなし）", font=font(12), fill=MUTE, anchor="lb")
        return img


def side_by_side(a, b, title, lab_a, lab_b, verdict_a, verdict_b, out, fps=15, t_span=None, badge="シミュレーション（MuJoCo）",
                 badge_b=None, ymax=4.0, target=None, fmax=None, gif=True):
    fw, fh = a["frames"][0].shape[1], a["frames"][0].shape[0]
    ph = 150
    W, H = 2 * fw + 12, 54 + fh + ph + 8
    t0 = min(e["t"] for e in a["log"] if e["phase"] == "grip") - 1.0 if t_span is None else t_span[0]
    t1 = max(a["t"][-1], b["t"][-1]) if t_span is None else t_span[1]
    fmax = C.OBJECTS[a["name"]].fmax if fmax is None else fmax
    pa = ForcePanel(a, fw, ph, fmax, target, t0, t1, ymax)
    pb = ForcePanel(b, fw, ph, C.OBJECTS[b["name"]].fmax if fmax is None else fmax, target, t0, t1, ymax)
    out = Path(out)
    tmp = out.with_suffix("")
    tmp.mkdir(parents=True, exist_ok=True)
    n = 0
    for t in np.arange(t0, t1, 1.0 / fps):
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        d.text((W // 2, 12), title, font=font(20, True), fill=INK, anchor="mt")
        for k, (run, lab, verdict, panel, bdg) in enumerate(((a, lab_a, verdict_a, pa, badge), (b, lab_b, verdict_b, pb, badge_b or badge))):
            x0 = k * (fw + 12)
            i = int(np.clip(np.searchsorted(run["t"], t), 0, len(run["frames"]) - 1))
            img.paste(Image.fromarray(run["frames"][i]), (x0, 44))
            dd = ImageDraw.Draw(img, "RGBA")
            dd.rounded_rectangle((x0 + 8, 52, x0 + 16 + dd.textlength(lab, font=font(16, True)), 78), 6, fill=(255, 255, 255, 215))
            dd.text((x0 + 12, 54), lab, font=font(16, True), fill=INK)
            dd.rounded_rectangle((x0 + 8, 44 + fh - 30, x0 + 16 + dd.textlength(bdg, font=font(12)), 44 + fh - 8), 5, fill=(10, 12, 18, 160))
            dd.text((x0 + 12, 44 + fh - 28), bdg, font=font(12), fill=(235, 235, 235))
            if t >= t1 - 2.5 and verdict:
                col = CRIT if verdict.startswith("破損") or verdict.startswith("落下") else OK
                tw = dd.textlength(verdict, font=font(22, True))
                dd.rounded_rectangle((x0 + fw - tw - 26, 52, x0 + fw - 8, 86), 7, fill=col + (230,))
                dd.text((x0 + fw - tw - 17, 54), verdict, font=font(22, True), fill=(255, 255, 255))
            img.paste(panel.at(t), (x0, 44 + fh + 6))
        img.save(tmp / f"f_{n:04d}.png")
        n += 1
    mp4 = out.with_suffix(".mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(tmp / "f_%04d.png"), "-pix_fmt", "yuv420p",
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-crf", "20", str(mp4)], check=True)
    if gif:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(tmp / "f_%04d.png"), "-vf",
                        "split[a][b];[a]palettegen=max_colors=160:stats_mode=full[p];[b][p]paletteuse=dither=sierra2_4a",
                        str(out.with_suffix(".gif"))], check=True)
    for f in tmp.glob("*.png"):
        f.unlink()
    tmp.rmdir()
    return n


def verdict(r):
    if r is None:
        return ""
    return {"ok": "成功", "crushed": "破損", "dropped": "落下", "slip": "滑り", "no_contact": "接触なし"}[r["failure"]] + \
        (f"  最大 {r['peak_force_true']:.2f} N" if r["failure"] in ("ok", "crushed") else "")


def lookalike_clip(seed=1):
    """the same-looking overripe tomato: the tomato recipe from vision vs the class chosen after the plunger"""
    name = "tomato_overripe"
    a = mujoco_run(seed, name, F_target=C.TARGET_FORCE["fragile"], max_retries=0, calibrate=True)
    b = mujoco_run(seed, name, chooser="plunger", F_target=C.TARGET_FORCE["fragile"], max_retries=0, calibrate=True)
    return side_by_side(a, b, "見た目は普通の中玉トマト。中身は過熟（許容 2.0 N）", "見た目だけで決める（2.2 N）",
                        "押して硬さを測ってから決める（1.55 N）", verdict(a["r"]), verdict(b["r"]), MEDIA / "lookalike_tomato.gif", ymax=3.0)


def q1_clip(seed=0):
    name = "strawberry"
    a = mujoco_run(seed, name, actuator="affine", max_retries=0)
    b = mujoco_run(seed, name, actuator="learned", max_retries=0)
    return side_by_side(a, b, "いちご（許容 3.0 N）を目標 2.2 N で：筋肉のモデルだけが違う", "直線モデル", "学習モデル",
                        verdict(a["r"]), verdict(b["r"]), MEDIA / "q1_strawberry.gif", ymax=4.0, target=2.2)


def physx_clip(seed=0, name="tomato"):
    p = REPO / "results" / "isaac_replay" / f"v2_replay_{seed}_{name}_closed_learned.json"
    if not p.exists():
        return 0
    a = mujoco_run(seed, name, actuator="learned")
    b = physx_run(p)
    rows = json.load(open(REPO / "results" / "benchmark_isaac.json"))["rows"]
    rb = next((r for r in rows if r["seed"] == seed and r["object"] == name and r["config"] == "closed_learned"), None)
    fmax = C.OBJECTS[name].fmax
    return side_by_side(a, b, f"普通の中玉トマト（許容 {fmax:g} N）を目標 2.2 N で：同じ制御コア・同じ個体", "MuJoCo",
                        "Isaac Sim 5.0 / PhysX", verdict(a["r"]), verdict(rb), MEDIA / "sim2sim_tomato.gif",
                        badge="シミュレーション（MuJoCo）", badge_b="PhysX の計算結果を MuJoCo の描画器で表示", ymax=5.5, target=2.2)


def teaser(seed=1, out=REPO / "docs" / "figures" / "teaser.png"):
    """Fig. 1: the same-looking overripe tomato, vision-only recipe vs plunger, three moments each + the grip force"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": ["Noto Sans CJK JP", "DejaVu Sans"], "font.size": 9})
    name = "tomato_overripe"
    runs = [mujoco_run(seed, name, size=(480, 300), F_target=C.TARGET_FORCE["fragile"], max_retries=0, calibrate=True),
            mujoco_run(seed, name, size=(480, 300), chooser="plunger", F_target=C.TARGET_FORCE["fragile"], max_retries=0, calibrate=True)]
    titles = ["見た目だけで決める：「トマト」→ 2.2 N", "硬さプランジャで決める：「軟」→ 1.55 N"]
    targets = [C.TARGET_FORCE["fragile"], C.TARGET_FORCE["soft"]]
    tmax = max(run["log"][-1]["t"] - run["log"][0]["t"] for run in runs)
    fig = plt.figure(figsize=(13.2, 4.3))
    gs = fig.add_gridspec(2, 6, height_ratios=[0.62, 0.75], hspace=0.22, wspace=0.04)
    fmax = C.OBJECTS[name].fmax
    for col, (run, title) in enumerate(zip(runs, titles)):
        log = run["log"]
        tg = [e["t"] for e in log if e["phase"] == "grip"]
        tt = [e["t"] for e in log if e["phase"] == "transport"]
        tp = [e["t"] for e in log if e["phase"] == "plunger"]
        moments = [(tp[len(tp) // 2] if tp else tg[0] - 0.5, "押す" if tp else "下りる"), (tg[-1], "握る"), (tt[len(tt) // 2], "運ぶ")]
        for j, (tm, lab) in enumerate(moments):
            ax = fig.add_subplot(gs[0, 3 * col + j])
            i = int(np.clip(np.searchsorted(run["t"], tm), 0, len(run["frames"]) - 1))
            ax.imshow(run["frames"][i][30:270, 60:420])
            ax.set_xticks([]), ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#bbbbbb")
            ax.text(8, 22, lab, fontsize=9, color="black", bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5))
            if col == 0 and j == 2 and run["r"]["crushed"]:          # the crush is not visible on a rigid shape: say it
                for sp in ax.spines.values():
                    sp.set_color("#d03b3b"), sp.set_linewidth(2.0)
                ax.text(352, 228, f"{run['r']['peak_force_true']:.2f} N > {fmax:.1f} N", fontsize=8.5, color="white", ha="right",
                        va="bottom", fontweight="bold", bbox=dict(facecolor="#d03b3b", edgecolor="none", pad=1.5))
            if j == 1:
                ax.set_title(title, fontsize=10.5, fontweight="bold", color="#d03b3b" if col == 0 else "#0a8a0a")
        ax = fig.add_subplot(gs[1, 3 * col:3 * col + 3])
        t0 = log[0]["t"]
        t = np.array([e["t"] - t0 for e in log])
        ax.axhline(fmax, color="#d03b3b", lw=1.0, ls=(0, (4, 2)))
        ax.axhline(targets[col], color="#8a8984", lw=0.9, ls=":")
        up = targets[col] > fmax          # keep the two labels apart: the upper line's label above it, the lower one's below
        ax.text(0.3, fmax + (-0.06 if up else 0.07), f"許容力 {fmax:.1f} N", color="#d03b3b", fontsize=8, ha="left",
                va="top" if up else "bottom")
        ax.text(0.3, targets[col] + (0.07 if up else -0.06), f"目標 {targets[col]:g} N", color="#52514e", fontsize=8, ha="left",
                va="bottom" if up else "top")
        ax.plot(t, [e["f_true"] for e in log], color="#0b0b0b", lw=1.8, label="真の把持力")
        ax.plot(t, [e["f_est"] for e in log], color="#2a78d6", lw=1.2, label="推定値（力センサなし）")
        if tp:
            ax.axvspan(tp[0] - t0, tp[-1] - t0, color="#eda100", alpha=0.15, lw=0)
            pl = [e for e in log if e["phase"] == "plunger" and "load" in e]
            ax.plot([e["t"] - t0 for e in pl], [e["load"] for e in pl], color="#b26b00", lw=1.4, label="プランジャの荷重")
            ax.text(0.5 * (tp[0] + tp[-1]) - t0, 0.55, "0.3 N だけ\n押す", fontsize=8, ha="center", color="#7a4a00")
        r = run["r"]
        v = {"ok": "成功", "crushed": "破損", "dropped": "落下", "slip": "滑り", "no_contact": "接触なし"}[r["failure"]]
        ax.text(0.99, 0.97, f"{v}（最大 {r['peak_force_true']:.2f} N）", transform=ax.transAxes, fontsize=10, fontweight="bold",
                color="#0ca30c" if r["success"] else "#d03b3b", va="top", ha="right")
        ax.set_ylim(0, 2.9)
        ax.set_xlim(0, tmax)
        ax.set_xlabel("時間 [s]")
        if col == 0:
            ax.set_ylabel("力 [N]")
        else:
            ax.tick_params(labelleft=False)
            handles = [ln for ln in ax.get_lines() if not ln.get_label().startswith("_")]
            fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.07))
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(alpha=0.3)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    # the image row alone, for the one-page summary
    fig, axes = plt.subplots(1, 6, figsize=(7.2, 1.25))
    for col, run in enumerate(runs):
        log = run["log"]
        tg = [e["t"] for e in log if e["phase"] == "grip"]
        tt = [e["t"] for e in log if e["phase"] == "transport"]
        tp = [e["t"] for e in log if e["phase"] == "plunger"]
        moments = [(tp[len(tp) // 2] if tp else tg[0] - 0.5, "押す" if tp else "下りる"), (tg[-1], "握る"), (tt[len(tt) // 2], "運ぶ")]
        for j, (tm, lab) in enumerate(moments):
            ax = axes[3 * col + j]
            i = int(np.clip(np.searchsorted(run["t"], tm), 0, len(run["frames"]) - 1))
            ax.imshow(run["frames"][i][30:270, 60:420])
            ax.set_xticks([]), ax.set_yticks([])
            edge = "#d03b3b" if (col == 0 and j == 2) else ("#0a8a0a" if (col == 1 and j == 2) else "#bbbbbb")
            for sp in ax.spines.values():
                sp.set_color(edge), sp.set_linewidth(1.8 if j == 2 else 0.8)
            ax.text(8, 30, lab, fontsize=7, color="black", bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01, wspace=0.04)
    fig.savefig(REPO / "docs" / "figures" / "teaser_strip.png", dpi=300)
    plt.close(fig)
    return str(out)


if __name__ == "__main__":
    want = sys.argv[1:] or ["q1", "lookalike", "physx", "teaser"]
    for k in want:
        print(k, {"q1": q1_clip, "lookalike": lookalike_clip, "physx": physx_clip, "teaser": teaser}[k]())
