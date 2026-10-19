"""Build the submission demo video (MuJoCo render + Japanese overlays) -> media/softtwin_demo.mp4

Sequence: title -> system diagram -> scene overview -> 5 language-instructed picks (closed loop, learned model)
-> baseline failure (fixed 60 kPa crushes a strawberry) -> results -> closing.
All numbers shown are produced by this repository (MuJoCo, synthetic stand-in muscle) and are labelled as such.
"""
from __future__ import annotations

import json
import math
import os
import sys

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "sim_mujoco"), str(REPO / "app")]
import mujoco  # noqa: E402
import softtwin_mj as S  # noqa: E402
import tactile as T  # noqa: E402
import tactile_dataset as D  # noqa: E402
from instruction import keyword_plan  # noqa: E402

W, H = 1280, 720
FPS = 30
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_B = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def font(size, bold=False):
    try:
        return ImageFont.truetype(FONT_B if bold else FONT, size, index=0)
    except Exception:
        return ImageFont.load_default()


def card(lines, sub=None, bg=(18, 20, 26)):
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    y = H // 2 - 40 * len(lines)
    for i, (txt, size, bold) in enumerate(lines):
        f = font(size, bold)
        w = d.textlength(txt, font=f)
        d.text(((W - w) / 2, y), txt, font=f, fill=(240, 242, 246))
        y += size + 28
    if sub:
        f = font(22)
        w = d.textlength(sub, font=f)
        d.text(((W - w) / 2, H - 70), sub, font=f, fill=(150, 156, 168))
    return np.array(img)


def image_card(path, title, note=None, crop=None):
    img = Image.new("RGB", (W, H), (250, 250, 252))
    d = ImageDraw.Draw(img)
    d.text((40, 24), title, font=font(34, True), fill=(20, 22, 28))
    pic = Image.open(path).convert("RGB")
    if crop:
        pic = pic.crop(crop)
    scale = min((W - 80) / pic.width, (H - 140) / pic.height)
    pic = pic.resize((int(pic.width * scale), int(pic.height * scale)), Image.LANCZOS)
    img.paste(pic, ((W - pic.width) // 2, 90))
    if note:
        d.text((40, H - 40), note, font=font(20), fill=(90, 96, 108))
    return np.array(img)


def overlay(frame, hud):
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img, "RGBA")
    if hud.get("title"):
        d.rounded_rectangle((20, 20, 640, 168), 12, fill=(10, 12, 18, 190))
        d.text((40, 30), hud.get("title", ""), font=font(26, True), fill=(255, 255, 255))
    for i, t in enumerate(hud.get("lines", [])):
        d.text((40, 72 + 30 * i), t, font=font(21), fill=(220, 226, 236))
    if "F" in hud:
        F, Fs, Fmax, p = hud["F"], hud["F_star"], hud["F_max"], hud["p"]
        x0, y0, x1 = 700, 40, 1240
        d.rounded_rectangle((680, 20, 1260, 168), 12, fill=(10, 12, 18, 190))
        scale = (x1 - x0) / max(Fmax * 1.25, Fs * 2, 1.0)
        d.text((x0, y0 - 4), "推定把持力（固有感覚）", font=font(19), fill=(220, 226, 236))
        d.rectangle((x0, y0 + 30, x1, y0 + 56), fill=(40, 44, 54, 255))
        col = (230, 70, 70) if F > Fmax else (78, 161, 255)
        d.rectangle((x0, y0 + 30, x0 + min(F * scale, x1 - x0), y0 + 56), fill=col + (255,))
        xs = x0 + Fs * scale
        d.line((xs, y0 + 24, xs, y0 + 62), fill=(62, 207, 142), width=4)
        if Fmax < 1e6:
            xm = x0 + min(Fmax * scale, x1 - x0)
            d.line((xm, y0 + 24, xm, y0 + 62), fill=(245, 165, 36), width=4)
        d.text((x0, y0 + 66), f"推定 {F:4.2f} N   目標 F* = {Fs:.2f} N   許容 {('∞' if Fmax > 1e6 else f'{Fmax:.1f}')} N", font=font(19), fill=(220, 226, 236))
        d.text((x0, y0 + 94), f"内圧 p = {p:4.1f} kPa", font=font(19), fill=(170, 176, 188))
    if hud.get("skin") is not None:
        sk = np.asarray(hud["skin"]).reshape(3, 8)
        x0, y0, cw, ch = 40, H - 190, 34, 30
        d.rounded_rectangle((20, y0 - 44, 20 + 8 * cw + 70, y0 + 3 * ch + 18), 12, fill=(10, 12, 18, 200))
        d.text((x0, y0 - 38), "布状触覚スキン 24 点 [N]", font=font(19), fill=(220, 226, 236))
        for fi in range(3):
            d.text((x0 - 2, y0 + fi * ch + 4), f"指{fi + 1}", font=font(15), fill=(170, 176, 188))
            for j in range(8):
                v = min(1.0, sk[fi, j] / 0.6)
                col = (int(40 + 215 * v), int(44 + 120 * v * (1 - v) * 4 * 0.5), int(54 + 30 * (1 - v)))
                d.rectangle((x0 + 34 + j * cw, y0 + fi * ch, x0 + 34 + j * cw + cw - 4, y0 + fi * ch + ch - 4), fill=col + (255,))
        d.text((x0 + 34, y0 + 3 * ch - 2), "根元 → 指先", font=font(14), fill=(150, 156, 168))
    if hud.get("speed"):
        d.text((W - 110, H - 50), hud["speed"], font=font(26, True), fill=(255, 255, 255))
    if hud.get("crushed"):
        d.rounded_rectangle((W // 2 - 170, H - 110, W // 2 + 170, H - 50), 12, fill=(200, 40, 40, 220))
        d.text((W // 2 - 140, H - 102), "許容力超過（破損）", font=font(30, True), fill=(255, 255, 255))
    return np.array(img)


CURRENT = {}


class Director:
    """renders frames while the episode runs; smooth free camera that follows the palm"""

    def __init__(self, cell: S.Cell, speed=1):
        self.c = cell
        self.r = mujoco.Renderer(cell.m, H, W)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 0.62, 135, -22
        self.cam.lookat[:] = [0, 0, S.CFG["table_top"] + 0.08]
        self.frames, self.hud, self.speed = [], {}, speed
        self.every = int(round(speed / FPS / cell.m.opt.timestep))
        self.n = 0
        self.crush_marked = set()
        self.over = 0
        orig = cell._step

        def step():
            orig()
            self.n += 1
            if self.n % self.every == 0:
                self.grab()

        cell._step = step

    def grab(self):
        follow = getattr(self, "follow", True)
        if getattr(self.c, "skin", None) is not None:
            sk = self.c.skin_read()
            for i, v in enumerate(sk):
                f, k, mm = i // 8, (i % 8) // 2, i % 2
                a = min(1.0, v / 0.6)
                self.c.m.geom_rgba[self.c.m.geom(f"pad_f{f}s{k}_{mm}").id] = [0.2 + 0.8 * a, 0.22 + 0.5 * a * (1 - a), 0.30 * (1 - a), 1]
        palm = self.c.d.body("palm").xpos
        tgt = np.array([palm[0], palm[1], max(S.CFG["table_top"] + 0.05, palm[2] - 0.12)])
        if follow:
            self.cam.lookat[:] = 0.92 * self.cam.lookat + 0.08 * tgt
        self.r.update_scene(self.c.d, camera=self.cam)
        self.frames.append(overlay(self.r.render(), self.hud))


def mark_crushed(cell, name):
    gid = cell.m.geom(f"{name}_g").id
    cell.m.geom_rgba[gid] = [0.30, 0.06, 0.06, 1]


def pick(cell, director, name, text, mode="closed_loop", p_fixed=60.0, slot=(0.0, 0.0)):
    plan = keyword_plan(text)
    cls = plan["class"]
    Fs = S.TARGET_FORCE[cls]
    Fmax = S.OBJECTS[name][4]
    base = {
        "title": f"指示：「{text}」",
        "lines": [f"計画：対象 = {name}，脆さ = {cls}（{plan['engine']}）",
                  ("制御：閉ループ（学習アクチュエータモデル＋固有感覚）" if mode == "closed_loop" else f"比較：開ループ（内圧 {p_fixed:.0f} kPa 固定）"),
                  "MuJoCo / 合成筋肉モデルで実機を模擬"],
        "F_star": Fs, "F_max": Fmax, "speed": f"x{director.speed}",
    }
    director.hud = dict(base, F=0.0, p=0.0)
    # live HUD update via the estimator
    orig_grab = director.grab

    def grab():
        f = cell.force_est()
        director.hud["F"] = f
        director.hud["p"] = getattr(cell, "_p_now", 0.0)
        if getattr(cell, "skin", None) is not None:
            director.hud["skin"] = cell.skin_read()
        ft = cell.force_true(name)
        director.over = director.over + 1 if ft > Fmax else 0
        if mode == "open_loop" and director.over >= 3 and name not in director.crush_marked:
            director.crush_marked.add(name)
            mark_crushed(cell, name)
        director.hud["crushed"] = name in director.crush_marked
        orig_grab()

    director.grab = grab
    r, _ = S.run_episode(cell, name, mode=mode, p_fixed=p_fixed, slot=slot)
    director.grab = orig_grab
    director.hud = {}
    return r


def pick_lookalike(name, policy, seed=2):
    """overripe tomato: VLM prior (looks like a tomato -> 1.8 N) vs touch probe + fusion classifier"""
    cell = S.Cell(seed=seed, objects=(name,))
    T.FabricSkin(cell, seed=seed)
    CURRENT["cell"] = cell
    director = Director(cell, speed=1)
    cams = D.Cameras(cell, name, renderer=director.r, tags=("pregrasp",))
    # fixed camera between the object and the tray so pick, transport and place stay in view
    o = cell.obj_pos(name)
    director.follow = False
    director.cam.lookat[:] = [(o[0] + S.CFG["tray"][0]) / 2, (o[1] + S.CFG["tray"][1]) / 2, S.CFG["table_top"] + 0.08]
    director.cam.distance, director.cam.azimuth, director.cam.elevation = 0.95, 120, -24
    Fmax = S.OBJECTS[name][4]
    prior = S.VISUAL_CLASS[S.VISUAL_ID.get(name, name)]
    head = "見た目：トマト → VLM の事前知識「脆い」F* = 1.8 N"
    title = "触覚なし（VLM の事前知識だけ）" if policy == "prior" else "触覚あり（0.4 N で触って確かめる）"
    director.hud = {"title": title, "lines": [head, "実は：完熟で柔らかい（許容 1.5 N）", ""], "F_star": S.TARGET_FORCE[prior], "F_max": Fmax,
                    "F": 0.0, "p": 0.0, "speed": "x1"}
    models = {k: T.Softmax.from_json(v) for k, v in json.load(open(REPO / "learning" / "tactile_classifier.json")).items()}

    def choose(c, e):
        fv, ft = D.vision_feat(cams), T.probe_features(e)
        pr = models["fusion"].predict_proba(np.r_[fv, ft])[0]
        cls = T.CLASSES[int(pr.argmax())]
        jp = {"very_fragile": "極脆", "soft": "軟", "fragile": "脆", "rigid": "剛"}[cls]
        director.hud["lines"][2] = f"触覚判定：{jp}（確率 {pr.max():.0%}，剛性 {ft[2]:.3f} N/deg）→ F* = {S.TARGET_FORCE[cls]:.1f} N"
        director.hud["F_star"] = S.TARGET_FORCE[cls]
        return S.TARGET_FORCE[cls]

    orig = director.grab

    def grab():
        director.hud["F"] = cell.force_est()
        director.hud["p"] = getattr(cell, "_p_now", 0.0)
        director.hud["skin"] = cell.skin_read()
        ft = cell.force_true(name)
        director.over = director.over + 1 if ft > Fmax else 0
        if director.over >= 3 and name not in director.crush_marked:
            director.crush_marked.add(name)
            mark_crushed(cell, name)
        director.hud["crushed"] = name in director.crush_marked
        orig()

    director.grab = grab
    if policy == "prior":
        r, _ = S.run_episode(cell, name, F_target=S.TARGET_FORCE[prior], max_retries=0)
    else:
        r, _ = S.run_episode(cell, name, F_target=S.TARGET_FORCE[prior], probe_fn=choose, max_retries=0)
    director.r.close()
    last = Image.fromarray(director.frames[-1])
    d = ImageDraw.Draw(last, "RGBA")
    msg, col = (("成功：潰さずにトレーへ", (30, 150, 90, 230)) if r["success"] else
                ("破損：許容力を超えた", (200, 40, 40, 230)) if r["crushed"] else ("失敗：落下", (200, 120, 30, 230)))
    d.rounded_rectangle((W // 2 - 220, H // 2 - 50, W // 2 + 220, H // 2 + 30), 14, fill=col)
    d.text((W // 2 - 190, H // 2 - 40), msg, font=font(40, True), fill=(255, 255, 255))
    return director.frames + [np.array(last)] * int(1.5 * FPS), r


def main():
    out = REPO / "media" / "softtwin_demo.mp4"
    frames = []

    def hold(img, sec):
        frames.extend([img] * int(sec * FPS))

    hold(card([("SoftTwin", 64, True), ("柔らかい食品を潰さずにつかむ", 40, False), ("学習アクチュエータモデル × 固有感覚による力推定 × VLM 指示", 28, False)],
              sub="Physical AI 講座 応用編1 最終課題　isobe_97"), 4)
    bd = REPO / "docs" / "figures" / "block_diagram.png"
    hold(image_card(bd, "システム構成（ブロック線図）：左から入り右へ出る", "A アプリ層 → B 制御層 → C プラント（実機／ツイン）→ D 推定・認識，E 監視・学習"), 4)
    hold(image_card(bd, "A アプリ層・B 制御層：指示 → 目標把持力 → 内圧指令", "力の誤差 e ＝ 目標 F* − 推定把持力 を，比例＋飽和＋積分で内圧指令に変換（50 Hz）", crop=(0, 120, 1960, 1000)), 6)
    hold(image_card(bd, "C プラント・D 推定：人工筋肉 → センサ → 把持力推定・触覚判定", "青線：推定把持力の帰還（力の閉ループ）と，触って決め直した目標力 F*touch の帰還", crop=(1880, 120, 4075, 1000)), 6)
    hold(image_card(bd, "E 監視・学習（非実時間）", "自動アノテーションデータで学習した重みを推定・認識ブロックへ配備", crop=(200, 1020, 2500, 1493)), 3)

    # ----- main demo: 5 instructed picks in one scene (closed loop)
    cell = S.Cell(seed=4)
    T.FabricSkin(cell, seed=4)
    CURRENT["cell"] = cell
    director = Director(cell, speed=2)
    sup_hook = S.Supply.step

    def step_with_record(self, cmd, dt):
        p = sup_hook(self, cmd, dt)
        CURRENT["cell"]._p_now = p
        return p

    S.Supply.step = step_with_record
    director.cam.distance, director.cam.azimuth, director.cam.elevation = 1.25, 130, -30
    for _ in range(int(3 * FPS)):
        director.grab()
    director.cam.distance, director.cam.azimuth, director.cam.elevation = 0.62, 135, -22
    results = []
    slots = [(-0.045, -0.032), (0.045, -0.032), (-0.045, 0.032), (0.045, 0.032), (0.0, 0.0)]
    for (name, text), slot in zip([("tofu", "豆腐を崩さないようにトレーへ"), ("strawberry", "いちごをそっとトレーに移して"), ("egg", "卵を割らずに運んで"),
                                   ("tomato", "トマトをトレーへ"), ("bolt", "ボルトをしっかりつかんでトレーへ")], slots):
        results.append(pick(cell, director, name, text, slot=slot))
    frames += director.frames
    director.r.close()  # one live OSMesa context at a time (several confuse each other)

    # ----- baseline failure
    hold(card([("比較：内圧を固定した開ループ制御", 40, True), ("60 kPa 固定 → いちごに許容力を超える力がかかる", 30, False)]), 3)
    cell2 = S.Cell(seed=4, objects=("strawberry",))
    CURRENT["cell"] = cell2
    d2 = Director(cell2, speed=2)
    pick(cell2, d2, "strawberry", "いちごをトレーへ", mode="open_loop", p_fixed=60.0)
    frames += d2.frames
    d2.r.close()

    # ----- tactile skin: look-alike object
    hold(card([("見た目が同じでも，中身は違う", 40, True), ("完熟トマト：カメラにはふつうのトマトに見える", 30, False),
               ("→ 布状触覚スキンで 0.4 N だけ触って硬さを測り，把持力を決め直す", 28, False)]), 4)
    for pol in ("prior", "touch"):
        fr, rr = pick_lookalike("tomato_overripe", pol)
        frames += fr
        results.append(dict(rr, video_policy=pol))
    S.Supply.step = sup_hook

    # ----- results
    for fig, title, note in [
        ("results/fig_benchmark.png", "固定シード 10 × 5 物体（MuJoCo，質量・摩擦をランダム化）", "提案法：成功率 96 %・破損 0 %（数値は results/benchmark_mujoco.json）"),
        ("results/fig_force_tracking.png", "固有感覚による把持力推定と接触力の真値", "MuJoCo の接触力を真値として比較"),
        ("results/actuator_fit.png", "アクチュエータ同定（合成データ）", "学習モデルで曲げ角の誤差を約 85 % 低減"),
        ("docs/figures/dataset_sample.png", "自動アノテーション付きマルチモーダルデータ（420 試行を 4 分で生成）", "画像・深度・マスク・触覚 24 点・内圧・関節角・把持力の真値・成否ラベル"),
        ("results/fig_tactile.png", "触覚データで学習した分類器", "画像だけでは見分けられない物体を触覚で判別"),
        ("results/fig_lookalike.png", "見た目が同じ物体を含むベンチマーク", "数値は results/benchmark_lookalike.json"),
    ]:
        if (REPO / fig).exists():
            hold(image_card(REPO / fig, title, note), 5)
    hold(card([("コード・レポートは GitHub で公開（Apache-2.0）", 32, True), ("次の段階：シリコン人工筋肉＋M5Stack の実機で同じ評価", 28, False)],
              sub="数値はシミュレーション（合成筋肉モデル）による結果です"), 4)

    import imageio.v2 as imageio

    out.parent.mkdir(exist_ok=True)
    with imageio.get_writer(out, fps=FPS, codec="libx264", quality=8, macro_block_size=8) as w:
        for f in frames:
            w.append_data(f)
    json.dump(results, open(REPO / "results" / "demo_results.json", "w"), indent=1, default=str)
    print("video", out, len(frames) / FPS, "s", [(r["object"], r["success"]) for r in results])


if __name__ == "__main__":
    main()
