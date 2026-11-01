"""Narration for the submission video (Japanese), synthesised with VOICEVOX (voice: No.7).

    python sim_mujoco/narration.py          (needs a VOICEVOX engine on 127.0.0.1:50021)

Each segment of make_video.py has an id here.  Words the engine tends to misread are written in kana
(MuJoCo, PhysX, GitHub, CPU, units); `--check` prints the engine's reading of every sentence for proofreading.
Output: media/narration/<id>_<k>.wav and media/narration/index.json (durations).
"""
from __future__ import annotations

import io
import json
import sys
import urllib.parse
import urllib.request
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "media" / "narration"
URL = "http://127.0.0.1:50021"
SPEAKER = 29            # No.7 (ノーマル)
SPEED = 1.15

# (segment id, [sentences]) -- each sentence is shown as the subtitle while it is spoken
SCRIPT = [
    ("open", ["見た目は普通のトマトでも、中は熟しすぎていることがあります。",
              "左は見た目から決めた力で握って潰した例、右は少しだけ押して硬さを測ってから運んだ例です。"]),
    ("title", []),
    ("questions", ["問いは三つです。",
                   "力センサなしで、潰さず落とさずに握れるか。見た目が同じ物体を、握る前に見分けられるか。",
                   "そして、判断の学習データを、人がラベルを付けずに作れるかです。"]),
    ("block", ["制御のコアは一つで、物理は MuJoCo と PhysX の二つのエンジンで計算します。",
               "把持力の帰還に力センサは使わず、制御器に戻るのは推定値だけです。"]),
    ("estimator", ["把持力は、指の曲げの不足分に曲げ剛性を掛け、触覚スキンで求めた作用点までの、各関節のてこの長さの和で割って推定します。"]),
    ("physics", ["果実の硬さを接触剛性として文献値で与え、エンジンごとに押し込んで測って合わせました。"]),
    ("actuator", ["空気圧の人工筋肉には不感帯や飽和、遅れ、履歴があり、直線モデルは把持に使う内圧の範囲で曲げを小さく見積もります。"]),
    ("q1clip", ["筋肉のモデルだけを替えると、直線モデルはいちごを潰し、学習モデルは目標の力で運びます。"]),
    ("q1res", ["五つの物体を30回ずつ、計150回試すと、成功96パーセント、破損ゼロでした。学習モデルは PhysX で計算しても破損ゼロです。"]),
    ("probe", ["指は果実より16倍以上柔らかく、触っても果実のへこみが指のたわみに埋もれます。",
               "そこで、手のひらに小さな硬さプランジャを付けました。"]),
    ("counter", ["見た目で決めると全数を潰しますが、押して硬さを測ってから決めると、成功76.7パーセント、破損ゼロです。"]),
    ("q2res", ["残る失敗は判断の誤りではなく落下で、重さから力の下限を上げると80パーセントになります。"]),
    ("q3", ["画像、触覚、内圧、真の接触力、成否を、人の手を介さずに同時に記録し、420試行を約4分で生成しました。", "このデータだけで学んだ分類器は、プランジャを使うと区分を99.8パーセント当てました。"]),
    ("sim2sim", ["別の物理エンジンでも、学習モデルは成功99パーセント、破損ゼロでした。失敗の多い直線モデルでも、成否は100回中97回一致しました。"]),
    ("robust", ["筋肉の特性が30パーセントずれると成功は80パーセントに下がり、空中での較正で94パーセントまで戻ります。指の曲げ剛性のずれは、この較正では直せません。"]),
    ("window", ["過熟トマトを1.2倍の大きさにすると、落とさないための力が許容力を超え、握る力だけでは解けなくなります。"]),
    ("limits", ["評価はすべてシミュレーションで、このグリッパの実機はまだありません。",
                "次は、人工筋肉一本の内圧と曲げを実機で測り、筋肉のモデルを学び直します。"]),
]


# display text -> what the engine should read (checked with --check)
SPEECH_FIX = [("力センサ", "りきセンサ"), ("指の曲げと", "指のまげと"), ("指の曲げの", "指のまげの"), ("範囲で曲げを", "範囲でまげを"), ("熟すにつれて", "じゅくすにつれて"),
              ("範囲で曲げを", "範囲でまげを"), ("曲げの1度", "まげの1度"), ("内圧と曲げを", "内圧とまげを"),
              ("MuJoCo と PhysX の", "ムジョコとフィジックスの"), ("PhysX で", "フィジックスで"), ("CPU 二つ", "シーピーユー二つ")]


def spoken(text):
    for a, b in SPEECH_FIX:
        text = text.replace(a, b)
    return text


def _post(path, data=None, **q):
    url = f"{URL}{path}?{urllib.parse.urlencode(q)}"
    req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=120).read()


def query(text):
    return json.loads(_post("/audio_query", b"", text=text, speaker=SPEAKER))


def reading(q):
    return "/".join("".join(m["text"] for m in ph["moras"]) for ph in q["accent_phrases"])


def synth(text, path):
    q = query(spoken(text))
    q.update(speedScale=SPEED, pitchScale=0.0, intonationScale=1.05, volumeScale=1.0,
             prePhonemeLength=0.08, postPhonemeLength=0.12, outputSamplingRate=48000)
    wav = _post("/synthesis", json.dumps(q).encode(), speaker=SPEAKER)
    path.write_bytes(wav)
    with wave.open(io.BytesIO(wav)) as w:
        return w.getnframes() / w.getframerate()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if "--check" in sys.argv:
        for sid, ss in SCRIPT:
            for s in ss:
                print(sid, "|", s, "\n   ", reading(query(spoken(s))))
        return
    index = {}
    for sid, ss in SCRIPT:
        index[sid] = [dict(text=s, file=f"{sid}_{k}.wav", sec=round(synth(s, OUT / f"{sid}_{k}.wav"), 3)) for k, s in enumerate(ss)]
    (OUT / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    print(sum(x["sec"] for v in index.values() for x in v), "s of speech")


if __name__ == "__main__":
    main()
