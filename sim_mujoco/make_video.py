"""Submission video (about 3 min, silent, Japanese captions, 1920x1080)  ->  media/softtwin_demo.mp4

    python sim_mujoco/make_video.py        (run sim_mujoco/media.py and scripts/make_figures.py first)

Every simulated frame carries a badge saying which engine produced it; nothing is filmed on hardware.
Editing: the problem shot opens the video, then the title; figures are shown whole and then the camera moves to the
part the caption talks about, with a highlight drawn on the figure; bullet lists build up line by line; segments are
joined with short cross-fades; a chapter bar at the bottom shows where you are.
Layout values are written for a 1280x720 frame and scaled by K.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from fontTools.ttLib import TTCollection
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
FIG, MEDIA = REPO / "docs" / "figures", REPO / "media"
K = 1.5
W, H, FPS = int(1280 * K), int(720 * K), 30
BG, INK, SUB, BLUE, RED, GREEN = (252, 252, 251), (11, 11, 11), (82, 81, 78), (42, 120, 214), (208, 59, 59), (10, 138, 10)
HL = (235, 104, 52)                                                   # highlight colour on figures
FR = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FB = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
FL = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"               # Latin letters the CJK font lacks (ş, ř)
FLB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
CJK_CMAP = set(TTCollection(FR).fonts[0].getBestCmap())
CHAPTERS = ["問題", "仕組み", "Q1", "Q2", "Q3", "検証", "まとめ"]
XFADE = 0.45                                                          # s


def P(v):
    return int(round(v * K))


def font(size, bold=False):
    return ImageFont.truetype(FB if bold else FR, P(size), index=0)


def text(d, xy, s, size, bold=False, fill=INK):
    """left-aligned text with a per-character fallback font; returns the advance in pixels"""
    x, y = xy
    runs, cur, cur_fb = [], "", None
    for ch in s:
        fb = ord(ch) not in CJK_CMAP
        if cur and fb != cur_fb:
            runs.append((cur, cur_fb))
            cur = ""
        cur, cur_fb = cur + ch, fb
    if cur:
        runs.append((cur, cur_fb))
    x0 = x
    for seg, fb in runs:
        f = ImageFont.truetype(FLB if bold else FL, P(size * 0.92)) if fb else font(size, bold)
        d.text((x, y + (P(size * 0.08) if fb else 0)), seg, font=f, fill=fill)
        x += d.textlength(seg, font=f)
    return x - x0


def text_width(d, s, size, bold=False):
    img = Image.new("RGB", (8, 8))
    return text(ImageDraw.Draw(img), (0, 0), s, size, bold)


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def numbers():
    """the same values the report uses (docs/paper/numbers.tex)"""
    import re
    txt = (REPO / "docs" / "paper" / "numbers.tex").read_text(encoding="utf-8")
    V = dict(re.findall(r"\\csname V@([^\\]+)\\endcsname\{(.*)\}", txt))
    return {k: v.replace("\\,\\%", " %").replace("\\,", " ").replace("$\\pm$", "±").replace("$", "") for k, v in V.items()}


class Writer:
    """frames go to ffmpeg; the first frames of each segment are blended with the last frame of the one before"""

    def __init__(self, path):
        self.p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                                   "-r", str(FPS), "-i", "-", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium",
                                   "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
        self.n, self.last, self.blend_left = 0, None, 0
        self.chapters = []                                        # (frame index, chapter) for the description

    def segment(self, chapter=None):
        if chapter and (not self.chapters or self.chapters[-1][1] != chapter):
            self.chapters.append((self.n, chapter))
        self.blend_left = int(XFADE * FPS) if self.last is not None else 0
        self.n_blend = self.blend_left

    def add(self, img, seconds=None):
        a = np.asarray(img.convert("RGB"), dtype=np.uint8)
        for _ in range(int(round((seconds or 1 / FPS) * FPS))):
            out = a
            if self.blend_left > 0:
                w = ease(1 - self.blend_left / self.n_blend)
                out = (self.last.astype(np.float32) * (1 - w) + a.astype(np.float32) * w).astype(np.uint8)
                self.blend_left -= 1
            self.p.stdin.write(out.tobytes())
            self.n += 1
        self.last = a

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def canvas():
    return Image.new("RGB", (W, H), BG)


def chapter_bar(img, chapter):
    """a thin bar at the bottom: where we are in the story"""
    d = ImageDraw.Draw(img)
    y, x0, x1 = H - P(16), P(48), W - P(48)
    seg = (x1 - x0) / len(CHAPTERS)
    i = CHAPTERS.index(chapter) if chapter in CHAPTERS else -1
    for k, c in enumerate(CHAPTERS):
        a, b = x0 + k * seg + P(3), x0 + (k + 1) * seg - P(3)
        d.rounded_rectangle((a, y, b, y + P(4)), P(2), fill=BLUE if k == i else ((198, 214, 236) if k < i else (226, 225, 220)))
        if k == i:
            w = d.textlength(c, font=font(12, True))
            d.text(((a + b) / 2 - w / 2, y - P(20)), c, font=font(12, True), fill=BLUE)
    return img


def header(img, title, chapter=None, alpha=1.0):
    d = ImageDraw.Draw(img)
    col = tuple(int(BG[k] + (INK[k] - BG[k]) * alpha) for k in range(3))
    text(d, (P(48), P(26)), title, 30, True, col)
    d.line((P(48), P(78), W - P(48), P(78)), fill=(226, 225, 220), width=P(1.5))
    if chapter:
        chapter_bar(img, chapter)
    return img


def lower_third(img, s, t, sub=None):
    """caption band that slides up and fades in during the first 0.5 s"""
    if not s:
        return img
    a = ease(t / 0.5)
    lines = s.split("\n")
    d0 = ImageDraw.Draw(img)
    wmax = max(text_width(d0, ln, 23) for ln in lines)
    hh = P(36) * len(lines) + P(18)
    y = H - P(52) - hh + int((1 - a) * P(14))
    x = (W - wmax) / 2 - P(22)
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    d.rounded_rectangle((x, y, x + wmax + P(44), y + hh), P(10), fill=(255, 255, 255, int(236 * a)), outline=(*BLUE, int(255 * a)),
                        width=P(1.5))
    d.rectangle((x, y + P(10), x + P(5), y + hh - P(10)), fill=(*BLUE, int(255 * a)))
    img.paste(Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB"))
    d = ImageDraw.Draw(img)
    col = tuple(int(255 + (INK[k] - 255) * a) for k in range(3))
    for i, ln in enumerate(lines):
        text(d, (x + P(24), y + P(8) + P(36) * i), ln, 23, False, col)
    return img


def view_of(fig, rect, box):
    """crop fig to rect (fractions) keeping the aspect of box (x, y, w, h); returns image and the pixel transform"""
    bx, by, bw, bh = box
    x0, y0, x1, y1 = rect
    fw, fh = fig.size
    cx, cy = (x0 + x1) / 2 * fw, (y0 + y1) / 2 * fh
    w, h = (x1 - x0) * fw, (y1 - y0) * fh
    if w / h > bw / bh:
        h = w * bh / bw
    else:
        w = h * bw / bh
    crop = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    sc = bw / w
    big = Image.new("RGB", (int(np.ceil(w)), int(np.ceil(h))), BG)    # outside the figure: page colour, not black
    x0c, y0c = int(round(crop[0])), int(round(crop[1]))
    ix0, iy0 = max(0, x0c), max(0, y0c)
    ix1, iy1 = min(fw, x0c + big.width), min(fh, y0c + big.height)
    if ix1 > ix0 and iy1 > iy0:
        big.paste(fig.crop((ix0, iy0, ix1, iy1)), (ix0 - x0c, iy0 - y0c))
    im = big.resize((bw, bh), Image.LANCZOS)

    def tf(fx, fy):
        return bx + (fx * fw - crop[0]) * sc, by + (fy * fh - crop[1]) * sc
    return im, tf


def figure_seg(wr, fig, title, chapter, seconds, cap=None, keys=((0.0, (0, 0, 1, 1)),), marks=(), badge=None):
    """keys: (time, rect) camera keyframes in figure fractions; marks: (t_on, t_off, rect, label) highlight boxes"""
    fig = fig.convert("RGB") if isinstance(fig, Image.Image) else Image.open(fig).convert("RGB")
    box = (P(40), P(92), W - P(80), H - P(92) - P(cap.count("\n") * 36 + 100 if cap else 40))
    wr.segment(chapter)
    n = int(seconds * FPS)
    for i in range(n):
        t = i / FPS
        k = max(j for j, (tk, _) in enumerate(keys) if tk <= t)
        if k + 1 < len(keys):
            (ta, ra), (tb, rb) = keys[k], keys[k + 1]
            u = ease((t - ta) / max(tb - ta, 1e-6))
            rect = tuple(ra[m] + (rb[m] - ra[m]) * u for m in range(4))
        else:
            rect = keys[k][1]
        img = header(canvas(), title, chapter)
        im, tf = view_of(fig, rect, box)
        img.paste(im, (box[0], box[1]))
        d = ImageDraw.Draw(img)
        for t_on, t_off, r, lab in marks:
            if t_on <= t < t_off:
                a = ease((t - t_on) / 0.35)
                p0, p1 = tf(r[0], r[1]), tf(r[2], r[3])
                grow = (1 - a) * P(14)
                d.rounded_rectangle((p0[0] - grow, p0[1] - grow, p1[0] + grow, p1[1] + grow), P(8), outline=HL, width=P(3.5))
                if lab:
                    tw = text_width(d, lab, 18, True)
                    lx, ly = p1[0] - tw - P(16), p0[1] - P(34) - grow
                    if ly < box[1]:
                        ly = p1[1] + P(6) + grow
                    d.rounded_rectangle((lx, ly, lx + tw + P(16), ly + P(28)), P(6), fill=HL)
                    text(d, (lx + P(8), ly + P(1)), lab, 18, True, (255, 255, 255))
        if badge:
            text(d, (box[0] + P(6), box[1] + box[3] - P(24)), badge, 14, False, SUB)
        lower_third(img, cap, t)
        wr.add(img)


def bullets_seg(wr, title, chapter, items, seconds, note=None, start=0.4, step=1.1):
    """lines appear one after another; '**' marks a heading line"""
    wr.segment(chapter)
    n = int(seconds * FPS)
    for i in range(n):
        t = i / FPS
        img = header(canvas(), title, chapter)
        d = ImageDraw.Draw(img)
        y = P(128)
        for j, ln in enumerate(items):
            a = ease((t - start - j * step) / 0.45)
            if a <= 0:
                break
            bold = ln.startswith("**")
            s = ln.strip("*")
            base = INK if bold else SUB
            col = tuple(int(BG[k] + (base[k] - BG[k]) * a) for k in range(3))
            x = P(96) + (0 if bold else P(28)) - int((1 - a) * P(18))
            if bold:
                d.rounded_rectangle((P(72), y + P(6), P(78), y + P(32)), P(2), fill=tuple(int(BG[k] + (BLUE[k] - BG[k]) * a) for k in range(3)))
            text(d, (x, y), s, 29 if bold else 25, bold, col)
            y += P(60 if bold else 50)
        if note:
            text(d, (P(96), H - P(78)), note, 17, False, SUB)
        wr.add(img)


def clip_seg(wr, path, title, chapter, cap=None, speed=1.0, crop_top=44, hold=1.2):
    """a side-by-side clip from media.py (its own title band is cut so the title is not shown twice)"""
    rd = imageio.get_reader(str(path))
    fps = rd.get_meta_data().get("fps", 15)
    frames = [Image.fromarray(f).crop((0, crop_top, Image.fromarray(f).width, Image.fromarray(f).height)) for f in rd]
    wr.segment(chapter)
    n_out = int(len(frames) / fps / speed * FPS)
    top, bottom = P(88), H - P(52) - P((cap.count("\n") + 1) * 36 + 30 if cap else 20)
    img = None
    for i in range(n_out + int(hold * FPS)):
        f = frames[min(len(frames) - 1, int(i * speed * fps / FPS))]
        img = header(canvas(), title, chapter)
        s = min((W - P(40)) / f.width, (bottom - top) / f.height)
        g = f.resize((int(f.width * s), int(f.height * s)), Image.LANCZOS)
        img.paste(g, ((W - g.width) // 2, top))
        lower_third(img, cap, i / FPS - 0.6)
        wr.add(img)


def title_seg(wr, seconds=4.0):
    wr.segment()
    for i in range(int(seconds * FPS)):
        t = i / FPS
        img = canvas()
        d = ImageDraw.Draw(img)
        for j, (s, size, bold, col, y) in enumerate([
                ("SoftTwin", 84, True, INK, 196),
                ("見た目が同じでも中身が違う食品を、", 36, False, INK, 318),
                ("力センサのない柔らかい指で壊さずにつかむ", 36, False, INK, 370),
                ("Physical AI 講座 2026 応用編1 最終課題　github.com/Qu0rise/softtwin", 19, False, SUB, 470),
                ("映像はすべてシミュレーション（MuJoCo、Isaac Sim 5.0 / PhysX）", 19, False, SUB, 502)]):
            a = ease((t - 0.15 - 0.25 * j) / 0.5)
            c = tuple(int(BG[k] + (col[k] - BG[k]) * a) for k in range(3))
            text(d, (P(84) - int((1 - a) * P(20)), P(y)), s, size, bold, c)
        a = ease((t - 0.1) / 0.6)
        d.rectangle((P(84), P(300), P(84) + int(a * P(420)), P(304)), fill=BLUE)
        wr.add(img)


def counter_seg(wr, V, seconds=5.0):
    """the headline number, counted up"""
    wr.segment("Q2")
    target = float(V.get("q2.plunger.la", "0"))
    for i in range(int(seconds * FPS)):
        t = i / FPS
        img = header(canvas(), "見た目が同じ過熟トマト・過熟いちご（各 30 試行）", "Q2")
        d = ImageDraw.Draw(img)
        a1, a2 = ease((t - 0.2) / 0.5), ease((t - 1.0) / 0.5)
        c1 = tuple(int(BG[k] + (RED[k] - BG[k]) * a1) for k in range(3))
        text(d, (P(150), P(220)), "見た目で決める", 30, True, tuple(int(BG[k] + (SUB[k] - BG[k]) * a1) for k in range(3)))
        text(d, (P(150), P(270)), "0 %", 110, True, c1)
        text(d, (P(150), P(420)), "全数破損（60/60）", 26, False, c1)
        val = target * ease((t - 1.3) / 1.6)
        c2 = tuple(int(BG[k] + (GREEN[k] - BG[k]) * a2) for k in range(3))
        text(d, (P(700), P(220)), "押して硬さを測ってから決める", 30, True, tuple(int(BG[k] + (SUB[k] - BG[k]) * a2) for k in range(3)))
        text(d, (P(700), P(270)), f"{val:.1f} %", 110, True, c2)
        text(d, (P(700), P(420)), f"破損 {V.get('q2.plunger.la.cr', '0')}", 26, False, c2)
        text(d, (P(700), P(462)), "正解の区分を与えた場合と、全試行で成否が一致", 22, False, c2)
        a3 = ease((t - 0.6) / 0.4)
        d.polygon([(P(560), P(330)), (P(620), P(330) - P(18)), (P(620), P(330) + P(18))], fill=None)
        d.line((P(560), P(330), P(640), P(330)), fill=tuple(int(BG[k] + (SUB[k] - BG[k]) * a3) for k in range(3)), width=P(4))
        d.polygon([(P(660), P(330)), (P(636), P(316)), (P(636), P(344))], fill=tuple(int(BG[k] + (SUB[k] - BG[k]) * a3) for k in range(3)))
        wr.add(img)


def pdf_png(pdf, dpi=400):
    out = Path(tempfile.mkdtemp()) / "x"
    subprocess.run(["pdftoppm", "-r", str(dpi), "-png", "-singlefile", str(pdf), str(out)], check=True)
    return Image.open(str(out) + ".png").convert("RGB")


def main():
    V = numbers()
    out = MEDIA / "softtwin_demo.mp4"
    wr = Writer(out)
    # --- cold open: the failure first, then the title
    clip_seg(wr, MEDIA / "lookalike_tomato.mp4", "見た目は普通の中玉トマト。中身は過熟（許容 2.0 N）", "問題",
             "左：見た目から「トマト＝2.2 N」で握り、許容力を超えて潰す\n右：0.3 N だけ押して硬さを測り、1.55 N で運ぶ")
    title_seg(wr)
    bullets_seg(wr, "硬さは画像に映らない。指には力センサがない", "問題", [
        "**Q1　指の曲げと内圧だけで、潰さず落とさない力で握れるか**",
        "シリコーンの空圧人工筋肉は、不感帯・飽和・遅れ・履歴を持つ",
        "**Q2　見た目が同じ物体を、握る前に見分けられるか**",
        "過熟の硬さは 1/2.5。トマトの熟度段階のあいだの比は 1.3〜3.1 倍",
        "**Q3　人手のラベル付けなしで、判断の学習データを作れるか**",
        "画像・触覚・内圧・真の接触力・成否を自動で同時に記録",
    ], 9.0)
    # --- system
    top = pdf_png(FIG / "fig_block_top.pdf")
    A, B, C_, D_, E = (.10, .115, .21, .325), (.27, .235, .38, .44), (.44, .35, .55, .56), (.61, .47, .72, .68), (.78, .59, .89, .795)
    figure_seg(wr, top, "一つの制御コアを、二つの物理エンジンで", "仕組み", 11.0,
               "前向きの信号は左上から右下へ流れる。制御器に戻るのは推定値の束だけ",
               marks=[(1.2, 3.0, A, "握る前に 1 回だけ判断"), (3.0, 4.8, B, "力の積分則"), (4.8, 6.6, C_, "MuJoCo / PhysX"),
                      (6.6, 8.4, D_, "仮定した仕様"), (8.4, 11.0, E, "力センサなしで推定")])
    det = pdf_png(FIG / "fig_block_detail.pdf")
    figure_seg(wr, det, "推定器：指の曲げ不足を、作用点のてこで割る", "仕組み", 9.0,
               "把持力の推定 ＝ k_b・Δθ ／ Λ。Λ はスキンの出力で重み付けたてこの和（スキンが触れていなければ指先）",
               keys=((0.0, (0, 0, 1, 1)), (1.6, (0, 0.66, 1, 1.0))),
               marks=[(2.6, 5.6, (.25, .728, .48, .803), "作用点"), (5.6, 9.0, (.185, .852, .495, .921), "曲げ不足 Δθ")])
    bullets_seg(wr, "物理を文献値に合わせる", "仕組み", [
        "**接触剛性を N/mm で与え、エンジンごとに測って合わせる**",
        f"MuJoCo：実効質量を同定し、目標の {V.get('cal.mj.maxdev', '?')} 以内",
        f"PhysX：コンプライアント接触の剛性を同じ手順で確認（目標との比 {V.get('cal.px.range', '?')}）",
        "**トマトの硬さ：緑熟 3.58 → レッド 1.16 N/mm（Vursavuş ら 2017）**",
        "熟したトマト 1.5、過熟 0.6 N/mm。卵の破断力 38〜45 N → 3 指の和で 25 N",
        "質量・摩擦・剛性・置き場所・向きは試行ごとに乱数で振る",
    ], 9.5)
    # --- Q1
    figure_seg(wr, FIG / "fig_actuator.png", "筋肉のモデル：把持の内圧域での曲げの偏り", "Q1", 10.0,
               f"増圧中の偏り：静特性の表 {V.get('act.static.gbias', '?')}°、グレーボックス {V.get('act.greybox.gbias', '?')}°、学習 {V.get('act.learned.gbias', '?')}°\n"
               "1° は力の約 5 %。正の偏りは力を大きく見積もり、弱く握る",
               keys=((0.0, (0, 0, 1, 1)), (2.2, (0.0, 0.0, 0.42, 1.0)), (5.2, (0.0, 0.0, 0.42, 1.0)), (6.6, (0.4, 0.0, 1.0, 1.0))),
               marks=[(2.6, 5.2, (.15, .08, .23, .80), "把持の内圧域"), (7.0, 10.0, (.46, .07, .96, .80), None)])
    clip_seg(wr, MEDIA / "q1_strawberry.mp4", "同じいちご・同じシードで、筋肉のモデルだけを替える", "Q1",
             f"直線モデルは力を小さく見積もり続け、いちごを全数潰す。学習モデルは成功 {V.get('q1.learned.succ', '?')} %・破損 0")
    figure_seg(wr, FIG / "fig_q1.png", "Q1 の結果：MuJoCo と PhysX", "Q1", 10.0,
               f"力の推定誤差 {V.get('q1.learned.mae', '?')} N（作用点を指先と仮定すると {V.get('q1.tip.mae', '?')} N）。PhysX でも破損 0",
               keys=((0.0, (0, 0, 1, 1)), (2.0, (0.0, 0.0, 0.56, 1.0)), (5.5, (0.0, 0.0, 0.56, 1.0)), (7.0, (0, 0, 1, 1))),
               marks=[(2.3, 5.5, (.01, .20, .52, .27), "直線：いちごを全数破損"), (2.3, 5.5, (.01, .51, .52, .58), "学習（提案）"),
                      (7.2, 10.0, (.53, .51, .99, .58), "PhysX")])
    # --- Q2
    figure_seg(wr, FIG / "fig_probe_map.png", "指で触るだけでは、文献の熟度差を見分けきれない", "Q2", 10.0,
               f"指は果実の 16〜61 倍柔らかく、果実のへこみが指のたわみに埋もれる\n剛性比 2.5 で指は {V.get('pm.tomato.finger', '?')} %（トマト）。掌のプランジャなら {V.get('pm.tomato.plunger', '?')} %",
               keys=((0.0, (0, 0, 1, 1)), (2.0, (0.0, 0.0, 0.52, 0.9)), (10.0, (0.0, 0.0, 0.52, 0.9))),
               marks=[(3.0, 10.0, (.12, .09, .37, .71), "熟度段階のあいだの比")])
    counter_seg(wr, V)
    figure_seg(wr, FIG / "fig_q2.png", "Q2 の結果：目標把持力の決め方", "Q2", 10.0,
               f"失敗 {V.get('fail.plunger.n', '?')} 件は、正解の区分を与えても同じ試行で起きる落下。判断の誤りは 0\n"
               f"重さから目標の下限を上げると {V.get('q2.mass.la', '?')} %。残る過熟いちごは握り方の問題",
               keys=((0.0, (0, 0, 1, 1)), (2.0, (0.5, 0.0, 1.0, 1.0)), (10.0, (0.5, 0.0, 1.0, 1.0))),
               marks=[(2.4, 6.0, (.60, .12, .995, .26), "プランジャ＝判断の上限"), (6.0, 10.0, (.60, .325, .995, .395), "重さの下限")])
    # --- Q3
    figure_seg(wr, FIG / "dataset_sample.png", "Q3：人手なしで作った学習データ", "Q3", 10.5,
               f"{V.get('ds.n', '?')} 試行を 2 CPU で {V.get('ds.min', '?')} 分。画像＋プランジャで区分の正解率 {V.get('cv.plunger', '?')} %")
    # --- checks
    if (MEDIA / "sim2sim_tomato.mp4").exists():
        clip_seg(wr, MEDIA / "sim2sim_tomato.mp4", "同じ制御コア・同じ個体を、別の物理エンジンで", "検証",
                 f"右は PhysX が計算した姿勢を MuJoCo の描画器で表示\n直線モデルの成否は同じ個体で {V.get('s2s.affine.same', '?')} 一致（一致係数 κ = {V.get('s2s.affine.kappa', '?')}）")
    figure_seg(wr, FIG / "fig_robustness.png", "筋肉と指が想定とずれたら", "検証", 9.0,
               f"筋肉のずれは空中較正で {V.get('rob.learnedcal.30', '?')} % まで戻る\n指の曲げ剛性のずれは残る → 実機ではロードセルで一度校正",
               keys=((0.0, (0, 0, 1, 1)), (4.5, (0, 0, 1, 1)), (5.8, (0.6, 0.0, 1.0, 1.0)), (9.0, (0.6, 0.0, 1.0, 1.0))))
    figure_seg(wr, FIG / "fig_force_window.png", "物体の大きさが変わったら：力の窓", "検証", 11.0,
               f"寸法 1.2 倍で内圧表は過熟トマトを {V.get('ss.recipe.1.2.tomato_overripe.cr', '?')} 回潰す。閉ループは潰さないが、重くなった物を落とす\n"
               "落とさない下限（茶）は寸法の 3 乗で上がり、大きい過熟トマトでは許容力（赤）を超える",
               keys=((0.0, (0, 0, 1, 1)), (5.0, (0, 0, 1, 1)), (6.5, (0.74, 0.0, 1.0, 0.9)), (11.0, (0.74, 0.0, 1.0, 0.9))),
               marks=[(7.0, 11.0, (.86, .08, .99, .62), "力の窓がない")])
    bullets_seg(wr, "限界と次の一歩", "まとめ", [
        "**評価はすべてシミュレーション**",
        "人工筋肉とセンサは仮定した特性のモデル。許容力は卵以外は仮定",
        f"見た目が同じ物体の成功は {V.get('q2.plunger.la', '?')} %（要求 85 % に未達）",
        f"重さの下限を加えると、開発に使っていないシードで {V.get('ho.q2.mass.la', '?')} %（破損 {V.get('ho.q2.mass.la.cr', '?')}）",
        f"プランジャで 1 回の把持が {V.get('q2.ct.delta', '?')} s 長くなる",
        "**次：人工筋肉 1 本の内圧と曲げを記録してモデルを学び直し、同じ評価を実機で**",
    ], 9.0)
    wr.segment("まとめ")
    for i in range(int(4.5 * FPS)):
        t = i / FPS
        e = canvas()
        d = ImageDraw.Draw(e)
        a = ease((t - 0.2) / 0.6)
        text(d, (P(80), P(250)), "github.com/Qu0rise/softtwin", 46, True, tuple(int(BG[k] + (INK[k] - BG[k]) * a) for k in range(3)))
        text(d, (P(84), P(330)), "コード Apache-2.0　データ CC BY 4.0　数字はすべて results/metrics.json から自動で書き出し", 21, False,
             tuple(int(BG[k] + (SUB[k] - BG[k]) * a) for k in range(3)))
        wr.add(e)
    wr.close()
    secs = wr.n / FPS
    chap = [f"{int(f / FPS) // 60}:{int(f / FPS) % 60:02d} {c}" for f, c in wr.chapters]
    (MEDIA / "softtwin_demo_chapters.txt").write_text("\n".join(chap) + "\n", encoding="utf-8")
    print(out, round(secs, 1), "s")
    print("\n".join(chap))


if __name__ == "__main__":
    main()
