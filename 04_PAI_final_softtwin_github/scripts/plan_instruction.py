"""Wrapper: python scripts/plan_instruction.py "<指示>" [--image media/overhead.png]"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from instruction import plan
args = sys.argv[1:]
img = args[args.index("--image") + 1] if "--image" in args else None
text = [a for i, a in enumerate(args) if a != "--image" and (i == 0 or args[i - 1] != "--image")][0]
print(json.dumps(plan(text, img), ensure_ascii=False))
