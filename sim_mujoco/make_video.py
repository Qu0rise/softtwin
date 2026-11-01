"""Submission video (under 3 min, Japanese narration and subtitles, 1920x1080)  ->  media/softtwin_demo.mp4

    python sim_mujoco/narration.py         (VOICEVOX engine on 127.0.0.1:50021; writes media/narration/)
    python sim_mujoco/make_video.py        (run sim_mujoco/media.py and scripts/make_figures.py first)

Every simulated frame names the engine that produced it; the closing "next step" shot is real hardware (media/hardware/, not in the repository) and is skipped when absent.
Style: a talk, not a promo -- slides are static, every slide change is the same short cross-fade, the subtitle is the
sentence being spoken, and nothing is boxed or highlighted on top of the figures.  Each slide lasts as long as its
narration.  The background music is a light plucked arrangement synthesised here (numpy, no samples), kept
quiet and ducked under the voice, with one beat of silence before the headline number.
Layout values are written for a 1280x720 frame and scaled by K.
"""
from __future__ import annotations

import json
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
XFADE = 0.5                                                           # s, every slide change


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
    """frames go to ffmpeg; the first frames of every segment are cross-faded from the last frame of the one before"""

    def __init__(self, path):
        self.p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                                   "-r", str(FPS), "-i", "-", "-pix_fmt", "yuv420p", "-crf", "19", "-preset", "medium",
                                   "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
        self.n, self.last, self.blend_left = 0, None, 0
        self.chapters = []                                        # (frame index, chapter) for the description
        self.speech = []                                          # (start s, wav path)

    def segment(self, chapter=None):
        if chapter and (not self.chapters or self.chapters[-1][1] != chapter):
            self.chapters.append((self.n, chapter))
        self.blend_left = int(XFADE * FPS) if self.last is not None else int(XFADE * FPS)
        self.n_blend = self.blend_left
        if self.last is None:
            self.last = np.full((H, W, 3), BG, np.uint8)
        return self.n / FPS

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

    def fade_out(self, seconds=0.8):
        a = self.last.astype(np.float32)
        n = int(seconds * FPS)
        for i in range(n):
            w = ease((i + 1) / n)
            self.p.stdin.write((a * (1 - w) + np.array(BG, np.float32) * w).astype(np.uint8).tobytes())
            self.n += 1

    def close(self):
        self.p.stdin.close()
        self.p.wait()


NAR = json.loads((MEDIA / "narration" / "index.json").read_text(encoding="utf-8"))
LEAD, GAP, TAIL = 0.5, 0.3, 0.55                                      # s: before the first sentence, between, after


class Speech:
    """timing of one segment's narration; subtitle(t) is the sentence being spoken at t (segment time)"""

    def __init__(self, sid):
        self.items, t = [], LEAD
        for x in NAR.get(sid, []):
            self.items.append((t, t + x["sec"], x["text"], MEDIA / "narration" / x["file"]))
            t += x["sec"] + GAP
        self.length = (t - GAP + TAIL) if self.items else 0.0

    def register(self, wr, t0):
        wr.speech += [(t0 + a, f) for a, b, s, f in self.items]

    def subtitle(self, t):
        for a, b, s, f in self.items:
            if a - 0.05 <= t <= b + 0.25:
                return s
        return None


def canvas():
    return Image.new("RGB", (W, H), BG)


def wrap(d, s, size, maxw):
    """one line if it fits, else two lines broken at the 、/。 nearest the middle"""
    if text_width(d, s, size) <= maxw:
        return [s]
    cuts = [i + 1 for i, ch in enumerate(s[:-1]) if ch in "、。"]
    if not cuts:
        cuts = [len(s) // 2]
    k = min(cuts, key=lambda i: abs(i - len(s) / 2))
    return [s[:k], s[k:]]


def header(img, title, chapter=None):
    d = ImageDraw.Draw(img)
    text(d, (P(48), P(26)), title, 30, True, INK)
    if chapter:
        cw = text_width(d, chapter, 15)
        text(d, (W - P(48) - cw, P(38)), chapter, 15, False, SUB)
    d.line((P(48), P(78), W - P(48), P(78)), fill=(214, 213, 208), width=P(1.2))
    return img


def subtitle(img, s):
    if not s:
        return img
    d = ImageDraw.Draw(img)
    lines = wrap(d, s, 23, W - P(160))
    y = H - P(40) - P(36) * len(lines)
    for i, ln in enumerate(lines):
        w = text_width(d, ln, 23)
        text(d, ((W - w) / 2, y + P(36) * i), ln, 23, False, INK)
    return img


CONTENT = (P(40), P(92), W - P(80), H - P(92) - P(118))               # x, y, w, h of the figure area


def view_of(fig, rect, box):
    """crop fig to rect (fractions) keeping the aspect of box (x, y, w, h)"""
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
    big = Image.new("RGB", (int(np.ceil(w)), int(np.ceil(h))), BG)
    x0c, y0c = int(round(crop[0])), int(round(crop[1]))
    ix0, iy0 = max(0, x0c), max(0, y0c)
    ix1, iy1 = min(fw, x0c + big.width), min(fh, y0c + big.height)
    if ix1 > ix0 and iy1 > iy0:
        big.paste(fig.crop((ix0, iy0, ix1, iy1)), (ix0 - x0c, iy0 - y0c))
    return big.resize((bw, bh), Image.LANCZOS)


def figure_seg(wr, fig, title, chapter, sid, keys=((0.0, (0, 0, 1, 1)),), note=None, min_s=4.0):
    """a figure on a slide; keys: (time, rect) camera keyframes in figure fractions (whole panels only)"""
    fig = fig.convert("RGB") if isinstance(fig, Image.Image) else Image.open(fig).convert("RGB")
    sp = Speech(sid)
    sp.register(wr, wr.segment(chapter))
    n = int(max(sp.length, min_s) * FPS)
    cache = {}
    for i in range(n):
        t = i / FPS
        k = max(j for j, (tk, _) in enumerate(keys) if tk <= t)
        if k + 1 < len(keys):
            (ta, ra), (tb, rb) = keys[k], keys[k + 1]
            u = ease((t - ta) / max(tb - ta, 1e-6))
            rect = tuple(ra[m] + (rb[m] - ra[m]) * u for m in range(4))
        else:
            rect = keys[k][1]
        key = (tuple(round(v, 4) for v in rect), sp.subtitle(t))
        if key not in cache:
            img = header(canvas(), title, chapter)
            img.paste(view_of(fig, rect, CONTENT), (CONTENT[0], CONTENT[1]))
            if note:
                text(ImageDraw.Draw(img), (CONTENT[0] + P(8), CONTENT[1] + CONTENT[3] - P(20)), note, 13, False, SUB)
            cache = {key: subtitle(img, key[1])}
        wr.add(cache[key])


def bullets_seg(wr, title, chapter, items, sid, note=None, min_s=4.0):
    """the whole list at once; '**' marks a heading line"""
    sp = Speech(sid)
    sp.register(wr, wr.segment(chapter))
    base = header(canvas(), title, chapter)
    d = ImageDraw.Draw(base)
    y = P(132)
    for ln in items:
        bold = ln.startswith("**")
        s = ln.strip("*")
        text(d, (P(96) + (0 if bold else P(30)), y), s, 28 if bold else 24, bold, INK if bold else SUB)
        y += P(58 if bold else 48)
    if note:
        text(d, (P(96), H - P(150)), note, 16, False, SUB)
    last = None
    for i in range(int(max(sp.length, min_s) * FPS)):
        s = sp.subtitle(i / FPS)
        if last is None or s != last[0]:
            last = (s, subtitle(base.copy(), s))
        wr.add(last[1])


def clip_seg(wr, path, title, chapter, sid, speed=1.0, crop_top=44, hold=1.0):
    """a side-by-side clip from media.py (its own title row is cut so the title is not shown twice)"""
    rd = imageio.get_reader(str(path))
    fps = rd.get_meta_data().get("fps", 15)
    frames = [Image.fromarray(f) for f in rd]
    frames = [f.crop((0, crop_top, f.width, f.height)) for f in frames]
    sp = Speech(sid)
    sp.register(wr, wr.segment(chapter))
    clip_s = len(frames) / fps / speed
    n_out = int(max(clip_s + hold, sp.length) * FPS)
    top, bottom = P(90), H - P(128)
    f0 = frames[0]
    s = min((W - P(40)) / f0.width, (bottom - top) / f0.height)
    size = (int(f0.width * s), int(f0.height * s))
    for i in range(n_out):
        f = frames[min(len(frames) - 1, int(i * speed * fps / FPS))]
        img = header(canvas(), title, chapter)
        img.paste(f.resize(size, Image.LANCZOS), ((W - size[0]) // 2, top))
        wr.add(subtitle(img, sp.subtitle(i / FPS)))


def hardware_seg(wr, seconds=4.0):
    """the next step, shown without narration: a silicone bending actuator on real hardware and its pneumatic circuit"""
    clip_p, diag_p = MEDIA / "hardware" / "actuator_hang.mp4", MEDIA / "hardware" / "pneumatic_circuit.png"
    if not (clip_p.exists() and diag_p.exists()):
        return
    clip = [Image.fromarray(f) for f in imageio.get_reader(str(clip_p))]
    fps = 30000 / 1001
    wr.segment("まとめ")
    top, bottom = P(100), H - P(110)
    hh = bottom - top
    cw = int(clip[0].width * hh / clip[0].height)
    diag = Image.open(diag_p).convert("RGB")
    gap = P(40)
    dw = min(int(W * 0.56), int(diag.width * hh / diag.height))
    dh = int(diag.height * dw / diag.width)
    diag = diag.resize((dw, dh), Image.LANCZOS)
    x0 = (W - cw - gap - dw) // 2
    base = header(canvas(), "次のステップ：実機の空圧アクチュエータと空圧回路", "まとめ")
    xd, yd = x0 + cw + gap, top + (hh - dh) // 2
    base.paste(diag, (xd, yd))
    d = ImageDraw.Draw(base)
    text(d, (x0, bottom + P(8)), "加圧で曲がる", 16, False, SUB)
    text(d, (xd, bottom + P(8)), "空圧回路と M5Stack の配線図", 16, False, SUB)
    n = int(seconds * FPS)
    for i in range(n):
        img = base.copy()
        f = clip[min(len(clip) - 1, int(i / FPS * fps))]
        img.paste(f.resize((cw, hh), Image.LANCZOS), (x0, top))
        wr.add(img)


def title_seg(wr, seconds=4.0):
    wr.segment()
    img = canvas()
    d = ImageDraw.Draw(img)
    text(d, (P(96), P(210)), "SoftTwin", 76, True, INK)
    text(d, (P(98), P(318)), "見た目が同じでも中身が違う食品を、力センサのない柔らかい指で壊さずにつかむ", 30, False, INK)
    d.line((P(98), P(380), P(98) + P(560), P(380)), fill=(160, 159, 154), width=P(1.2))
    text(d, (P(98), P(400)), "Physical AI 講座 2026 応用編1 最終課題", 19, False, SUB)
    text(d, (P(98), P(432)), "映像は最後の「次のステップ」を除き、すべてシミュレーション（MuJoCo、Isaac Sim 5.0 / PhysX）", 19, False, SUB)
    wr.add(img, seconds)


def counter_seg(wr, V, sid):
    """the headline numbers, side by side"""
    sp = Speech(sid)
    sp.register(wr, wr.segment("Q2"))
    base = header(canvas(), "見た目が同じ過熟トマト・過熟いちご（各 30 試行）", "Q2")
    d = ImageDraw.Draw(base)
    text(d, (P(170), P(200)), "見た目で決める", 28, True, SUB)
    text(d, (P(170), P(250)), "0 %", 104, True, RED)
    text(d, (P(170), P(400)), "全数破損（60/60）", 25, False, RED)
    d.line((P(560), P(318), P(650), P(318)), fill=SUB, width=P(3))
    d.polygon([(P(668), P(318)), (P(646), P(305)), (P(646), P(331))], fill=SUB)
    text(d, (P(720), P(200)), "押して硬さを測ってから決める", 28, True, SUB)
    text(d, (P(720), P(250)), f"{V.get('q2.plunger.la', '?')} %", 104, True, GREEN)
    text(d, (P(720), P(400)), f"破損 {V.get('q2.plunger.la.cr', '0')}", 25, False, GREEN)
    text(d, (P(720), P(440)), "正解の区分を与えた場合と、全試行で成否が一致", 21, False, SUB)
    last = None
    for i in range(int(max(sp.length, 4.0) * FPS)):
        s = sp.subtitle(i / FPS)
        if last is None or s != last[0]:
            last = (s, subtitle(base.copy(), s))
        wr.add(last[1])


def _pluck(f, dur, sr, decay=3.2, bright=0.35):
    """a soft plucked note (marimba-like): sine + a little 4th harmonic, exponential decay, 4 ms attack"""
    t = np.arange(int(dur * sr)) / sr
    env = np.minimum(1, t / 0.004) * np.exp(-decay * t)
    return ((np.sin(2 * np.pi * f * t) + bright * np.sin(2 * np.pi * 4 * f * t) * np.exp(-6 * t)) * env).astype(np.float32)


def bgm(seconds, sr=48000, seed=11, bpm=104):
    """a light, quiet arrangement written for this video (no samples): plucked melody on a pentatonic scale over
    C - Am - F - G, a soft bass on the beat; returns float32 stereo normalised to 1"""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    out = np.zeros((n, 2), np.float32)
    beat = 60 / bpm
    chords = [(48, (60, 64, 67)), (45, (57, 60, 64)), (41, (53, 57, 60)), (43, (55, 59, 62))]
    penta = [60, 62, 64, 67, 69, 72, 74, 76]
    hz = lambda m: 440 * 2 ** ((m - 69) / 12)

    def put(x, t0, gain, pan):
        a = int(t0 * sr)
        if a >= n:
            return
        x = x[: n - a] * gain
        out[a:a + len(x), 0] += x * (1 - pan)
        out[a:a + len(x), 1] += x * pan

    bar = 0
    motif = [0, None, 2, 3, None, 2, 1, None]                         # 8th-note motif, varied per bar
    while bar * 4 * beat < seconds:
        root, triad = chords[bar % 4]
        t_bar = bar * 4 * beat
        for k in range(4):                                           # bass: root on 1 and 3, fifth on 2 and 4
            put(_pluck(hz(root + (0 if k % 2 == 0 else 7)), beat * 0.9, sr, decay=4.0, bright=0.1), t_bar + k * beat, 0.55, 0.5)
        for k in (1, 3):                                             # off-beat chord stabs
            for m in triad:
                put(_pluck(hz(m), beat * 0.6, sr, decay=7.0, bright=0.2), t_bar + k * beat + beat / 2, 0.12, 0.35)
        if bar % 8 >= 2:                                             # melody enters after two bars
            shift = int(rng.integers(0, 3))
            for k, step in enumerate(motif):
                if step is None or (bar % 4 == 3 and k > 5):
                    continue
                m = penta[min(len(penta) - 1, step + shift)]
                put(_pluck(hz(m + 12), beat * 0.8, sr, decay=5.0, bright=0.3), t_bar + k * beat / 2, 0.30, 0.62)
        bar += 1
    return out / max(1e-6, np.abs(out).max())


def mux_audio(video, speech, seconds, out):
    """narration at its start times; music at -29 dBFS peak, ducked by 7 dB while a sentence is spoken"""
    import soundfile as sf
    sr = 48000
    n = int(seconds * sr)
    voice = np.zeros(n, np.float32)
    for t0, f in speech:
        x, r = sf.read(str(f), dtype="float32")
        assert r == sr
        a = int(t0 * sr)
        voice[a:a + len(x)] += x[: max(0, n - a)]
    voice *= 0.89 / max(1e-6, np.abs(voice).max())
    music = bgm(seconds, sr) * 10 ** (-29 / 20)
    gate = np.ones(n, np.float32)                                     # a beat of silence before the headline number
    t_cnt = [t0 for t0, f in speech if Path(f).stem == "counter_0"]
    if t_cnt:
        a0, a1 = int((t_cnt[0] - 0.4) * sr), int((t_cnt[0] + 3.2) * sr)
        ramp = int(0.25 * sr)
        gate[a0:a1] = 0
        gate[a0 - ramp:a0] = np.linspace(1, 0, ramp)
        gate[a1:a1 + ramp] = np.linspace(0, 1, ramp)
    active = np.convolve((np.abs(voice) > 0.01).astype(np.float32), np.ones(int(0.3 * sr)) / int(0.3 * sr), "same")
    duck = 10 ** (-7 * np.clip(active * 4, 0, 1) / 20)
    fade = np.clip(np.minimum(np.arange(n) / (2 * sr), (n - np.arange(n)) / (2.5 * sr)), 0, 1)
    mix = voice[:, None] + music * (duck * fade * gate)[:, None]
    wav = Path(tempfile.mkdtemp()) / "mix.wav"
    sf.write(str(wav), mix, sr)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-i", str(wav), "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-shortest", "-movflags", "+faststart", str(out)], check=True)


def pdf_png(pdf, dpi=400):
    out = Path(tempfile.mkdtemp()) / "x"
    subprocess.run(["pdftoppm", "-r", str(dpi), "-png", "-singlefile", str(pdf), str(out)], check=True)
    return Image.open(str(out) + ".png").convert("RGB")


def main():
    V = numbers()
    out = MEDIA / "softtwin_demo.mp4"
    silent = MEDIA / "softtwin_demo_silent.mp4"
    wr = Writer(silent)
    # --- cold open: the failure first, then the title
    clip_seg(wr, MEDIA / "lookalike_tomato.mp4", "見た目は普通の中玉トマト。中身は過熟（許容 2.0 N）", "問題", "open")
    title_seg(wr, 1.8)
    bullets_seg(wr, "硬さは画像に映らない。指には力センサがない", "問題", [
        "**Q1　力センサなしで、潰さず落とさない力で握れるか**",
        "シリコーンの空圧人工筋肉は、不感帯・飽和・遅れ・履歴を持つ",
        "**Q2　見た目が同じ物体を、握る前に見分けられるか**",
        "過熟の硬さは 1/2.5。トマトの熟度段階のあいだの比は 1.3〜3.1 倍",
        "**Q3　人手のラベル付けなしで、判断の学習データを作れるか**",
        "画像・触覚・内圧・真の接触力・成否を自動で同時に記録",
    ], "questions")
    # --- system
    figure_seg(wr, pdf_png(FIG / "fig_block_top.pdf"), "一つの制御コアを、二つの物理エンジンで", "仕組み", "block")
    det = pdf_png(FIG / "fig_block_detail.pdf")
    det = det.crop((0, int(det.height * 0.612), det.width, det.height))          # panel (c) only
    figure_seg(wr, det, "推定器：曲げ不足×曲げ剛性を、作用点のてこの和で割る", "仕組み", "estimator")
    bullets_seg(wr, "物理を文献値に合わせる", "仕組み", [
        "**接触剛性を N/mm で与え、エンジンごとに測って合わせる**",
        f"MuJoCo：実効質量を同定し、接触剛性を目標の {V.get('cal.mj.maxdev', '?')} 以内に",
        f"PhysX：コンプライアント接触の剛性を同じ手順で確認（目標との比 {V.get('cal.px.range', '?')}）",
        "**トマトの硬さ：緑熟 3.58 → レッド 1.16 N/mm（Vursavuş ら 2017）**",
        "熟したトマト 1.5、過熟 0.6 N/mm。卵の許容力は 3 指の和で 25 N（破断力 38〜45 N より低い）",
        "質量・摩擦・剛性・置き場所・向きは試行ごとに乱数で振る",
    ], "physics", min_s=6.5)
    # --- Q1
    figure_seg(wr, FIG / "fig_actuator.png", "筋肉のモデル：把持の内圧域での曲げの偏り", "Q1", "actuator", min_s=7.5)
    clip_seg(wr, MEDIA / "q1_strawberry.mp4", "同じいちご・同じシードで、筋肉のモデルだけを替える", "Q1", "q1clip")
    figure_seg(wr, FIG / "fig_q1.png", "Q1 の結果：MuJoCo と PhysX", "Q1", "q1res", min_s=8.0)
    # --- Q2
    figure_seg(wr, FIG / "fig_probe_map.png", "指で触るだけでは、文献の熟度差を見分けきれない", "Q2", "probe")
    counter_seg(wr, V, "counter")
    figure_seg(wr, FIG / "fig_q2.png", "Q2 の結果：目標把持力の決め方", "Q2", "q2res", min_s=7.0)
    # --- Q3
    figure_seg(wr, FIG / "dataset_sample.png", "Q3：人手なしで作った学習データ", "Q3", "q3")
    # --- checks
    if (MEDIA / "sim2sim_tomato.mp4").exists():
        clip_seg(wr, MEDIA / "sim2sim_tomato.mp4", "同じ制御コア・同じ個体を、別の物理エンジンで", "検証", "sim2sim", hold=0.6)
    figure_seg(wr, FIG / "fig_robustness.png", "筋肉と指が想定とずれたら", "検証", "robust", min_s=5.0)
    figure_seg(wr, FIG / "fig_force_window.png", "物体の大きさが変わったら：力の窓", "検証", "window", min_s=6.5)
    bullets_seg(wr, "限界と次の一歩", "まとめ", [
        "**評価はすべてシミュレーション**",
        "人工筋肉とセンサは仮定した特性のモデル。許容力は卵以外は仮定",
        f"見た目が同じ物体の成功は {V.get('q2.plunger.la', '?')} %（要求 85 % に未達）",
        f"重さの下限を加えると、開発に使っていないシードで {V.get('ho.q2.mass.la', '?')} %（破損 {V.get('ho.q2.mass.la.cr', '?')}）",
        f"見た目だけで決める場合より、1 回の把持が {V.get('q2.ct.delta', '?')} s 長くなる",
        "**次：実機の人工筋肉 1 本で内圧と曲げを測り、筋肉モデルを学び直す**",
    ], "limits", min_s=5.0)
    hardware_seg(wr)
    wr.segment("まとめ")
    e = canvas()
    d = ImageDraw.Draw(e)
    text(d, (P(96), P(250)), "github.com/Qu0rise/softtwin", 44, True, INK)
    text(d, (P(98), P(330)), "コード Apache-2.0　データ CC BY 4.0　図と数値は results/ から自動で作成", 20, False, SUB)
    text(d, (P(98), P(372)), "音声　VOICEVOX:No.7　　音楽　本動画のために作成", 18, False, SUB)
    wr.add(e, 2.0)
    wr.fade_out(0.9)
    wr.close()
    secs = wr.n / FPS
    mux_audio(silent, wr.speech, secs, out)
    silent.unlink()
    chap = [f"{int(f / FPS) // 60}:{int(f / FPS) % 60:02d} {c}" for f, c in wr.chapters]
    (MEDIA / "softtwin_demo_chapters.txt").write_text("\n".join(chap) + "\n", encoding="utf-8")
    print(out, round(secs, 1), "s")
    print("\n".join(chap))


if __name__ == "__main__":
    main()
