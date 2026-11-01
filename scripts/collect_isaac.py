"""Collect a PhysX run (sim/isaac_headless/q1_isaac.py) into the results folder.

    python scripts/collect_isaac.py <isaac_io folder>   ->  results/benchmark_isaac.json, results/isaac_replay/*.json
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FAIL = ("no_contact", "slip", "dropped", "crushed")


def main(io):
    io = Path(io)
    rows = [json.loads(x) for x in open(io / "v2_rows.jsonl") if x.strip()]
    plan = json.load(open(io / "v2_plan.json"))
    cal = json.load(open(io / "v2_contact_calibration.json"))
    summ = {}
    for c in plan["configs"]:
        rs = [r for r in rows if r["config"] == c]
        if rs:
            summ[c] = {"n": len(rs), "success": sum(r["success"] for r in rs), "crushed": sum(r["crushed"] for r in rs),
                       "failure": {f: sum(r["failure"] == f for r in rs) for f in FAIL},
                       "by_object": {o: [sum(r["success"] for r in rs if r["object"] == o), sum(r["crushed"] for r in rs if r["object"] == o),
                                         sum(1 for r in rs if r["object"] == o)] for o in plan["objects"]}}
    out = dict(engine="Isaac Sim 5.0.0 / PhysX 5 (omni.physx), headless Kit app, 1 kHz, TGS", plan=plan, contact_calibration=cal,
               summary=summ, rows=rows)
    json.dump(out, open(REPO / "results" / "benchmark_isaac.json", "w"), indent=1)
    dst = REPO / "results" / "isaac_replay"
    dst.mkdir(exist_ok=True)
    for s, n, c in plan.get("replay", []):
        f = io / f"v2_replay_{s}_{n}_{c}.json"
        if f.exists():
            shutil.copy(f, dst / f.name)
    for c, v in summ.items():
        print(f"{c:22s} {v['success']:3d}/{v['n']}  crushed {v['crushed']}  {v['failure']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else REPO / "isaac_io")
