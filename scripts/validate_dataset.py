"""Dataset validator (lecture D2: schema + validator before training; kiso D6: "debug at the data stage").
Checks every episode against meta/info.json: keys, dtypes, shapes, finite values, physical ranges, label consistency.

    python scripts/validate_dataset.py [--quick]      -> exit 1 on any violation, writes data/tactile_dataset/meta/validation.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
D = REPO / "data" / "tactile_dataset"


def check(ep_path: Path, row: dict, info: dict) -> list[str]:
    errs = []
    z = np.load(ep_path)
    T = int(row["n_steps"])
    for name, spec in info["features"].items():
        if name not in z.files:
            errs.append(f"missing {name}")
            continue
        a = z[name]
        shape = tuple(T if s == "T" else (a.shape[i] if s == "P" else s) for i, s in enumerate(spec["shape"]))
        if a.shape != shape:
            errs.append(f"{name} shape {a.shape} != {shape}")
        if str(a.dtype) != spec["dtype"] and not (spec["dtype"] == "str" and a.dtype.kind == "U"):
            errs.append(f"{name} dtype {a.dtype} != {spec['dtype']}")
        if a.dtype.kind == "f" and not np.isfinite(a.astype(np.float64)).all():
            errs.append(f"{name} has NaN/inf")
        if "range" in spec and a.size:
            lo, hi = spec["range"]
            if a.min() < lo - 1e-6 or a.max() > hi + 1e-6:
                errs.append(f"{name} out of range [{lo},{hi}]: {a.min():.3g}..{a.max():.3g}")
    if "t" in z.files and np.any(np.diff(z["t"]) <= 0):
        errs.append("t not strictly increasing")
    o = row["outcome"]
    if o["success"] and (o["crushed"] or not o["in_tray"]):
        errs.append("label inconsistency: success with crushed/not in tray")
    if o["crushed"] != (o["peak_force_true"] > row["max_safe_force"] + 1e-9) and row["max_safe_force"] < 1e6:
        errs.append("label inconsistency: crushed flag vs peak force")
    if row["true_class"] not in info["classes"]:
        errs.append(f"unknown class {row['true_class']}")
    return errs


def main(quick=False):
    info = json.load(open(D / "meta" / "info.json"))
    rows = [json.loads(line) for line in open(D / "index.jsonl")]
    if quick:
        rows = rows[:: max(1, len(rows) // 20)]
    bad = {}
    for r in rows:
        e = check(D / r["file"], r, info)
        if e:
            bad[r["id"]] = e
    rep = {"checked": len(rows), "violations": len(bad), "examples": dict(list(bad.items())[:10])}
    (D / "meta" / ("validation_quick.json" if quick else "validation.json")).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    sys.exit(main(ap.parse_args().quick))
