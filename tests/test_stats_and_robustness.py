"""Statistics helpers, the mismatch machinery, and the PhysX result file."""
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "sim_mujoco"), str(ROOT / "learning"), str(ROOT / "scripts")]


def test_mcnemar_holm_newcombe():
    import analysis_stats as A
    b, c, p = A.mcnemar_exact([(True, False)] * 6 + [(True, True)] * 20)
    assert (b, c) == (6, 0) and abs(p - 2 * 0.5 ** 6) < 1e-12          # two-sided exact binomial on discordant pairs
    assert A.mcnemar_exact([(True, True)] * 5)[2] == 1.0               # no discordant pair -> p = 1
    assert A.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    lo, hi = A.newcombe_paired([(1, 1)] * 36 + [(1, 0)] * 12 + [(0, 1)] * 2)   # Newcombe (1998) example, diff 0.20
    assert lo < 0.20 < hi and 0.04 < lo < 0.08 and 0.31 < hi < 0.36
    assert A.wilson(0, 150)[0] == 0.0 and A.wilson(150, 150)[1] == 1.0


def test_mismatched_muscle_is_reproducible():
    import robustness as R
    (_, k1), (_, k2) = R.muscle(0.3, 3, "tofu"), R.muscle(0.3, 3, "tofu")
    assert k1 == k2 and k1["p0"] != 6.0
    assert R.muscle(0.0, 3, "tofu")[1] == R.NOMINAL


@pytest.mark.skipif(os.environ.get("SOFTTWIN_SKIP_SIM") == "1", reason="physics smoke test disabled")
def test_dry_stroke_calibration_stays_in_free_range():
    """the staircase stops before the fingertips meet (about 28 deg) and the table is monotone in pressure"""
    import softtwin_mj as S
    import twin_core as C
    cell = S.Cell(seed=0, objects=("tomato",))
    rig = C.make_rig(cell.inst, "greybox", dt=cell.dt_ctrl)
    cal = C.calibrate_dry_stroke(cell, rig)
    assert len(cal["P"]) >= 2 and max(cal["P"]) < 16 and sorted(cal["P"]) == cal["P"]


def test_physx_summary_matches_its_rows():
    p = ROOT / "results/benchmark_isaac.json"
    if not p.exists():
        pytest.skip("no PhysX run in this checkout")
    b = json.load(open(p))
    for c, s in b["summary"].items():
        rs = [r for r in b["rows"] if r["config"] == c]
        assert s["n"] == len(rs) and s["success"] == sum(r["success"] for r in rs) and s["crushed"] == sum(r["crushed"] for r in rs)
    assert {r["seed"] for r in b["rows"]} == set(b["plan"]["seeds"])
