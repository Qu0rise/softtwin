"""Generate the SoftTwin system block diagram as a draw.io file (Simulink style).

Every block takes its inputs on the LEFT edge and gives its outputs on the RIGHT edge.
The signal flow runs from top-left to bottom-right; feedback lines run back along the bottom.
Large modules (containers) hold medium modules (sub-containers) that hold small blocks.

  python docs/figures/make_block_diagram_drawio.py        # -> docs/figures/block_diagram.drawio
  xvfb-run -a drawio -x -f pdf --crop -o docs/figures/block_diagram.pdf docs/figures/block_diagram.drawio
"""
from __future__ import annotations

import html
from pathlib import Path

OUT = Path(__file__).resolve().parent / "block_diagram.drawio"

FONT = "fontFamily=Noto Sans CJK JP;"
C = {
    "design": "fillColor=#DCE8FF;strokeColor=#1F4E9C;",      # designed / learned in this work
    "hw": "fillColor=#FFEFD9;strokeColor=#B86B00;",          # hardware (real device) / its twin
    "env": "fillColor=#EEEEEE;strokeColor=#555555;",         # environment / physics
    "plain": "fillColor=#FFFFFF;strokeColor=#333333;",
    "io": "fillColor=#FFFFFF;strokeColor=#333333;rounded=1;arcSize=40;",
}

cells: list[str] = []
geo: dict[str, tuple[float, float, float, float, str]] = {}   # id -> absolute x, y, w, h, parent
_n = [0]


def nid(p="c"):
    _n[0] += 1
    return f"{p}{_n[0]}"


def esc(s):
    return html.escape(s, quote=True)


def _rel(parent, x, y):
    if parent in ("1", None):
        return x, y
    px, py, _, _, _ = geo[parent]
    return x - px, y - py


def container(cid, x, y, w, h, title, style_extra="", parent="1", size=12, dashed=False):
    rx, ry = _rel(parent, x, y)
    st = (f"rounded=1;arcSize=2;whiteSpace=wrap;html=1;container=1;collapsible=0;recursiveResize=0;{FONT}"
          f"verticalAlign=top;align=left;spacingLeft=8;spacingTop=4;fontStyle=1;fontSize={size};{style_extra}"
          + ("dashed=1;dashPattern=6 4;" if dashed else ""))
    cells.append(f'<mxCell id="{cid}" value="{esc(title)}" style="{st}" vertex="1" parent="{parent}">'
                 f'<mxGeometry x="{rx}" y="{ry}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    geo[cid] = (x, y, w, h, parent)
    return cid


def block(bid, x, y, w, h, label, kind="plain", parent="1", shape="", size=11):
    rx, ry = _rel(parent, x, y)
    st = f"rounded=0;whiteSpace=wrap;html=1;{FONT}fontSize={size};{C[kind]}{shape}"
    cells.append(f'<mxCell id="{bid}" value="{esc(label)}" style="{st}" vertex="1" parent="{parent}">'
                 f'<mxGeometry x="{rx}" y="{ry}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    geo[bid] = (x, y, w, h, parent)
    return bid


def text(x, y, w, h, label, size=11, parent="1", align="left", style=""):
    rx, ry = _rel(parent, x, y)
    tid = nid("t")
    st = f"text;html=1;whiteSpace=wrap;{FONT}fontSize={size};align={align};verticalAlign=middle;{style}"
    cells.append(f'<mxCell id="{tid}" value="{esc(label)}" style="{st}" vertex="1" parent="{parent}">'
                 f'<mxGeometry x="{rx}" y="{ry}" width="{w}" height="{h}" as="geometry"/></mxCell>')


def wire(src, sy, dst, dy, label="", pts=(), dashed=False, lx=-0.55, ly=9, color="#222222", lsize=10):
    """src output port at absolute y=sy on its right edge -> dst input port at absolute y=dy on its left edge"""
    sx0, sy0, _, sh, _ = geo[src]
    dx0, dy0, _, dh, _ = geo[dst]
    ey = (sy - sy0) / sh
    ny = (dy - dy0) / dh
    assert 0 < ey < 1 and 0 < ny < 1, (src, dst, ey, ny)
    st = (f"edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;{FONT}fontSize={lsize};endArrow=block;endFill=1;endSize=6;"
          f"strokeColor={color};strokeWidth=1.3;jumpStyle=arc;jumpSize=7;labelBackgroundColor=none;"
          f"exitX=1;exitY={ey:.4f};exitDx=0;exitDy=0;entryX=0;entryY={ny:.4f};entryDx=0;entryDy=0;"
          + ("dashed=1;dashPattern=6 4;" if dashed else ""))
    p = "".join(f'<mxPoint x="{a}" y="{b}"/>' for a, b in pts)
    arr = f'<Array as="points">{p}</Array>' if pts else ""
    cells.append(f'<mxCell id="{nid("e")}" value="{esc(label)}" style="{st}" edge="1" parent="1" source="{src}" target="{dst}">'
                 f'<mxGeometry x="{lx}" y="{ly}" relative="1" as="geometry">{arr}</mxGeometry></mxCell>')


def dot(x, y):
    """Simulink branch point"""
    block(nid("d"), x - 3.5, y - 3.5, 7, 7, "", "plain", shape="ellipse;fillColor=#222222;strokeColor=#222222;")


# ============================================================ title
text(30, 20, 1500, 30, "<b>SoftTwin システム構成（ブロック線図）</b>　信号はすべて各ブロックの左辺から入り，右辺から出る．帰還は下側を通って左へ戻る．", 15)

# ============================================================ external inputs (left edge)
block("in_s", 30, 133, 92, 24, "指示文 s", "io")
block("in_I", 30, 183, 92, 24, "俯瞰画像 I", "io")
block("in_x", 30, 288, 92, 24, "物体位置 x<sub>obj</sub>", "io")

# ============================================================ A: application layer
container("A", 150, 80, 320, 290, "A  アプリ層（指示→目標）", C["design"].replace("DCE8FF", "F4F8FF"))
block("vlm", 180, 120, 135, 100, "VLM タスク計画器<br>Qwen2.5-VL<br>（Ollama）", "design", "A")
block("lut", 352, 175, 82, 40, "目標力表<br>F*(c)", "design", "A")
block("seq", 355, 235, 95, 95, "タスク系列<br>制御<br>（状態機械）", "plain", "A")

# ============================================================ B: controller (M5Stack / ROS 2, 50 Hz)
container("B", 495, 80, 445, 290, "B  制御層（M5Stack / ROS 2，50 Hz）", C["design"].replace("DCE8FF", "F4F8FF"))
block("sw", 515, 180, 80, 66, "目標選択<br>（切替）", "design", "B")
block("sum", 625, 196, 30, 44, "Σ", "plain", "B", shape="ellipse;fontSize=14;fontStyle=1;")
text(612, 198, 14, 14, "+", 12, "1", "center")
text(612, 230, 14, 14, "−", 12, "1", "center")
container("FC", 680, 165, 245, 100, "力制御則", "fillColor=#FFFFFF;strokeColor=#1F4E9C;", parent="B", size=11)
block("kp", 690, 205, 38, 30, "K=4", "design", "FC", shape="triangle;direction=east;fontSize=10;")
block("sat1", 742, 202, 56, 36, "飽和<br>−15…3", "design", "FC", size=10)
block("int", 812, 202, 40, 36, "1/s<br>reset", "design", "FC", size=10)
block("sat2", 866, 202, 50, 36, "飽和<br>0…80", "design", "FC", size=10)
text(690, 240, 230, 20, "接触検知時に積分器を実測内圧へリセット", 9, "FC", style="fontColor=#555555;")
block("servo", 760, 292, 110, 36, "位置サーボ（PD）", "plain", "B")

# ============================================================ C: plant (real device or its digital twin)
container("C", 965, 80, 600, 345, "C  プラント（実機：シリコン人工筋肉＋M5Stack ／ ツイン：MuJoCo・Isaac Sim，同じ入出力）", C["hw"].replace("FFEFD9", "FFF8EE"))
block("pump", 985, 200, 90, 40, "空圧供給<br>1/(0.25s+1)", "hw", "C")
block("musc", 1120, 195, 110, 50, "人工筋肉指<br>3指×4節（PRBM）", "hw", "C")
block("gan", 1120, 290, 110, 40, "ガントリ<br>（3 軸）", "hw", "C")
block("cont", 1268, 190, 105, 145, "接触・物体<br>（物理エンジン<br>／実世界）", "env", "C")
container("SEN", 1395, 130, 160, 230, "センサ", "fillColor=#FFFFFF;strokeColor=#B86B00;", parent="C", size=11)
block("ps", 1405, 155, 140, 30, "圧力センサ", "hw", "SEN")
block("bs", 1405, 205, 140, 30, "曲げセンサ", "hw", "SEN")
block("skin", 1405, 243, 140, 46, "布状触覚スキン 24 点<br><span style=\"font-size:9px\">漏れ・遅れ・雑音・12 bit</span>", "design", "SEN")
block("camw", 1405, 297, 140, 52, "手先カメラ<br><span style=\"font-size:9px\">RGB・深度・マスク</span>", "hw", "SEN")

# ============================================================ D: estimation / perception
container("D", 1585, 80, 300, 290, "D  推定・認識", C["design"].replace("DCE8FF", "F4F8FF"))
block("fphi", 1600, 150, 120, 40, "学習アクチュエータ<br>モデル f<sub>φ</sub>", "design", "D")
block("est", 1772, 185, 104, 60, "固有感覚<br>把持力推定", "design", "D")
block("feat", 1600, 250, 120, 50, "触覚プローブ<br>特徴（剛性 k<sub>t</sub>…）", "design", "D")
block("clf", 1772, 260, 104, 85, "画像＋触覚<br>分類器<br>→ F*(クラス)", "design", "D")

# ============================================================ outputs (right edge)
block("out_obj", 1945, 383, 120, 26, "搬送結果（物体）", "io")
block("out_F", 1945, 130, 120, 26, "推定把持力 F̂", "io")

# ============================================================ wires: forward path
wire("in_s", 145, "vlm", 145, "")
wire("in_I", 195, "vlm", 195, "")
wire("in_x", 300, "seq", 300, "", lx=0.2)
wire("vlm", 145, "seq", 255, "対象 o", pts=[(338, 145), (338, 255)], lx=-0.85)
wire("vlm", 195, "lut", 195, "脆さ c", lx=-0.2)
wire("lut", 195, "sw", 196, "F*<sub>prior</sub>", lx=0)
wire("seq", 250, "sw", 232, "位相 φ", pts=[(482, 250), (482, 232)], lx=-0.6)
wire("seq", 310, "servo", 310, "ガントリ目標 x<sub>ref</sub> [m]", lx=-0.5)
wire("sw", 214, "sum", 214, "F*", lx=-0.3)
wire("sum", 220, "kp", 220, "e", lx=0.1)
wire("kp", 220, "sat1", 220)
wire("sat1", 220, "int", 220)
wire("int", 220, "sat2", 220)
wire("sat2", 220, "pump", 220, "p<sup>cmd</sup> [kPa]", lx=0)
wire("servo", 310, "gan", 310, "u<sub>x</sub>", lx=0)
wire("pump", 220, "musc", 220, "内圧 p", lx=0.1)
wire("pump", 220, "ps", 170, "", pts=[(1098, 220), (1098, 170)])
dot(1098, 220)
wire("musc", 220, "cont", 220, "曲げ θ", lx=0.1)
wire("gan", 310, "cont", 310, "x", lx=0.1)
wire("cont", 220, "bs", 220, "関節角 q", lx=0, lsize=9)
wire("cont", 265, "skin", 265, "接触力分布", lx=0, lsize=9)
wire("cont", 310, "camw", 310, "物体姿勢", lx=0, lsize=9)
# sensors -> estimation
wire("ps", 170, "fphi", 170, "p [kPa]", lx=-0.2)
wire("fphi", 170, "est", 200, "θ̂", pts=[(1745, 170), (1745, 200)], lx=0.75)
wire("bs", 220, "est", 230, "q [rad]", pts=[(1572, 220), (1572, 230)], lx=-0.45)
wire("bs", 220, "feat", 285, "", pts=[(1572, 220), (1572, 285)])
dot(1572, 230)
wire("skin", 265, "feat", 265, "s [N]×24", lx=-0.3, lsize=9)
wire("feat", 275, "clf", 275, "k<sub>t</sub>", lx=0)
wire("camw", 325, "clf", 325, "画像 I<sub>w</sub>", lx=-0.75, lsize=9)
wire("est", 215, "out_F", 143, "", pts=[(1925, 215), (1925, 143)])
# object state -> output (branch below the sensors)
wire("cont", 310, "out_obj", 396, "", pts=[(1385, 310), (1385, 396)])
dot(1385, 310)

# ============================================================ wires: feedback (run back along the bottom)
wire("est", 215, "sum", 232, "F̂ [N]（固有感覚の帰還）", pts=[(1905, 215), (1905, 455), (606, 455), (606, 232)], lx=-0.1, ly=-9,
     color="#1F4E9C")
dot(1905, 215)
wire("clf", 300, "sw", 214, "F*<sub>touch</sub> [N]（触覚で決め直した目標力）", pts=[(1915, 300), (1915, 485), (502, 485), (502, 214)], lx=-0.1,
     ly=-9, color="#1F4E9C")

# ============================================================ E: monitoring and learning (offline / non real-time)
container("E", 150, 540, 1110, 225, "E  監視・学習（非リアルタイム）", "fillColor=#FAFAFA;strokeColor=#666666;", dashed=True)
block("from1", 170, 590, 120, 50, "From<br>[p] [q] [s] [I<sub>w</sub>] [成否]", "plain", "E", shape="rounded=1;arcSize=30;", size=10)
block("rec", 330, 590, 190, 50, "記録＋自動アノテーション<br><span style=\"font-size:9px\">マスク・深度・真の力・成否ラベル</span>", "design", "E")
block("ds", 560, 590, 170, 50, "マルチモーダルデータ<br><span style=\"font-size:9px\">420 試行・12 万サンプル</span>", "design", "E")
block("learn", 770, 590, 190, 50, "学習<br><span style=\"font-size:9px\">アクチュエータ f<sub>φ</sub>／触覚分類器 W</span>", "design", "E")
block("out_w", 1010, 602, 220, 26, "重み φ, W → D へ配備", "io")
block("from2", 170, 680, 120, 50, "From<br>[θ 実機] [q ツイン]", "plain", "E", shape="rounded=1;arcSize=30;", size=10)
block("eng", 330, 680, 190, 50, "コアエンジン<br><span style=\"font-size:9px\">ツイン誤差 e・RMSE・遅延 τ</span>", "plain", "E")
block("rmq", 560, 680, 170, 50, "RabbitMQ<br><span style=\"font-size:9px\">アプリ層との境界</span>", "plain", "E")
block("ui", 770, 680, 190, 50, "監視 UI<br><span style=\"font-size:9px\">HTTP / WebSocket</span>", "plain", "E")
block("out_ui", 1010, 692, 220, 26, "ダッシュボード", "io")
wire("from1", 615, "rec", 615)
wire("rec", 615, "ds", 615)
wire("ds", 615, "learn", 615)
wire("learn", 615, "out_w", 615, dashed=True)
wire("from2", 705, "eng", 705)
wire("eng", 705, "rmq", 705, "e, RMSE, τ", lx=0, lsize=9)
wire("rmq", 705, "ui", 705)
wire("ui", 705, "out_ui", 705)

# ============================================================ legend
lx, ly = 1290, 545
block("lg1", lx, ly + 5, 26, 16, "", "design")
text(lx + 32, ly + 3, 330, 20, "本研究で設計・学習した要素", 11)
block("lg2", lx, ly + 33, 26, 16, "", "hw")
text(lx + 32, ly + 31, 400, 20, "実機（シミュレーションでは合成筋肉モデル・センサモデルで代替）", 11)
block("lg3", lx, ly + 61, 26, 16, "", "env")
text(lx + 32, ly + 59, 330, 20, "環境（物体・接触）", 11)
text(lx, ly + 86, 760, 20, "大モジュール A〜E ＞ 中モジュール（力制御則・センサ）＞ 小ブロック．● は分岐点，[ ] は Goto/From タグ．", 11)
text(lx, ly + 108, 760, 20, "青い線は帰還：F̂ は力の閉ループ，F*<sub>touch</sub> は 0.4 N で触った後に目標力を決め直す経路．", 11)
text(lx, ly + 130, 760, 40, "s：指示文　I：画像　o：対象　c：脆さクラス　F*：目標把持力　e：力の誤差　p：内圧　θ：曲げ角　q：関節角　θ̂：自由曲げ角の予測　F̂：推定把持力　k<sub>t</sub>：押し込み剛性　φ, W：学習済みパラメータ", 10)

xml = ('<mxfile host="SoftTwin-generator"><diagram id="softtwin" name="SoftTwin block diagram">'
       '<mxGraphModel dx="2200" dy="900" grid="1" gridSize="5" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="0" '
       'pageScale="1" pageWidth="2100" pageHeight="800" math="0" shadow="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>'
       + "".join(cells) + "</root></mxGraphModel></diagram></mxfile>")
OUT.write_text(xml, encoding="utf-8")
print("wrote", OUT, len(cells), "cells")
