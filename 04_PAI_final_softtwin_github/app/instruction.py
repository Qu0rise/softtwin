"""App layer: natural-language instruction + overhead image -> task plan (target object, target grip force class).

Uses a local vision-language model through Ollama when available (default model: qwen2.5vl:7b,
override with OLLAMA_VLM). Falls back to a keyword parser so the pipeline always runs.
The plan's `class` selects the target grip force in sim/softtwin_runtime.py (TARGET_FORCE).

  python app/instruction.py "いちごを潰さずにトレーへ" --image media/overhead.png
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys

OBJECTS = {
    "strawberry": ("fragile", ["いちご", "苺", "イチゴ", "strawberry"]),
    "egg": ("fragile", ["卵", "たまご", "タマゴ", "egg"]),
    "tofu": ("very_fragile", ["豆腐", "とうふ", "tofu"]),
    "tomato": ("fragile", ["トマト", "tomato"]),
    "bolt": ("rigid", ["ボルト", "ねじ", "ネジ", "bolt", "screw"]),
}

PROMPT = (
    "You control a soft pneumatic gripper above a table. Objects on the table may include: "
    + ", ".join(OBJECTS)
    + ". Read the user's instruction and the overhead image. Reply with JSON only: "
    '{"target": <one object name from the list>, "place": "tray", "class": "very_fragile|fragile|rigid", '
    '"reason": <short Japanese sentence>}. Classify by how easily the object is damaged when squeezed.'
)


def keyword_plan(text: str) -> dict:
    for name, (cls, words) in OBJECTS.items():
        if any(w.lower() in text.lower() for w in words):
            return {"target": name, "place": "tray", "class": cls, "reason": "キーワード一致（VLM未使用）", "engine": "keyword"}
    return {"target": None, "place": "tray", "class": "fragile", "reason": "対象を特定できませんでした", "engine": "keyword"}


def vlm_plan(text: str, image: str | None, model: str, host: str) -> dict:
    import requests

    msg = {"role": "user", "content": PROMPT + "\nInstruction: " + text}
    if image and os.path.exists(image):
        msg["images"] = [base64.b64encode(open(image, "rb").read()).decode()]
    r = requests.post(f"{host}/api/chat", json={"model": model, "messages": [msg], "stream": False, "format": "json", "options": {"temperature": 0}}, timeout=120)
    r.raise_for_status()
    content = r.json()["message"]["content"]
    plan = json.loads(re.search(r"\{.*\}", content, re.S).group(0))
    if plan.get("target") not in OBJECTS:
        raise ValueError(f"VLM returned unknown target: {plan.get('target')}")
    plan.setdefault("class", OBJECTS[plan["target"]][0])
    plan["engine"] = f"ollama:{model}"
    return plan


def plan(text: str, image: str | None = None) -> dict:
    model = os.environ.get("OLLAMA_VLM", "qwen2.5vl:7b")
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    try:
        return vlm_plan(text, image, model, host)
    except Exception as e:
        p = keyword_plan(text)
        p["vlm_error"] = f"{type(e).__name__}: {e}"[:200]
        return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--image")
    a = ap.parse_args()
    print(json.dumps(plan(a.text, a.image), ensure_ascii=False))
    sys.exit(0)
