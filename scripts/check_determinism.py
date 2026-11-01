"""Compare a fresh Q1 run with the published rows of the same seeds.

    python sim_mujoco/softtwin_mj.py bench --seeds 2 --out /tmp/q1.json      (configs and seeds 0-1 of the published run)
    python scripts/check_determinism.py /tmp/q1.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def key(r):
    return r["object"], r["config"], r["seed"]


def main(path):
    new = json.load(open(path))["rows"]
    ref = {key(r): r for r in json.load(open(REPO / "results" / "benchmark_mujoco.json"))["rows"]}
    same, diff = 0, []
    for r in new:
        old = ref.get(key(r))
        if old is not None and all(old.get(k) == v for k, v in r.items()):
            same += 1
        else:
            diff.append(key(r))
    print(f"{same}/{len(new)} rows identical to results/benchmark_mujoco.json")
    for k in diff[:10]:
        print("differs:", k)
    return 0 if not diff else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/q1.json"))
