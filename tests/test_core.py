"""Fast checks (no GPU, no renderer): the physics of the twin, the estimator, the published numbers."""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "sim_mujoco"), str(ROOT / "learning"), str(ROOT / "scripts")]
SKIP_SIM = pytest.mark.skipif(os.environ.get("SOFTTWIN_SKIP_SIM") == "1", reason="physics smoke test disabled")


def test_objects_are_physically_plausible():
    """densities of food and steel, and the look-alikes differ from their twins only in the hidden properties"""
    import twin_core as C
    for n, o in C.OBJECTS.items():
        a = o.size
        vol = {"sphere": lambda: 4 / 3 * math.pi * a[0] ** 3, "ellipsoid": lambda: 4 / 3 * math.pi * a[0] * a[1] * a[2],
               "box": lambda: 8 * a[0] * a[1] * a[2], "cylinder": lambda: math.pi * a[0] ** 2 * 2 * a[1]}[o.shape]()
        rho = o.mass / vol / 1000.0
        assert (0.9 <= rho <= 1.15) if n != "bolt" else (7.5 <= rho <= 8.0), (n, rho)
    for la, twin in C.VISUAL_ID.items():
        a, b = C.OBJECTS[la], C.OBJECTS[twin]
        assert (a.shape, a.size, a.mass, a.rgba, a.mu) == (b.shape, b.size, b.mass, b.rgba, b.mu)
        assert a.k < b.k and a.fmax < b.fmax


def test_estimator_inverts_the_lever_model():
    """a normal force F at distance s along a finger bends it back by F * Lambda(s) / k_b; the estimator must return F
    when the skin reports that location, and underestimate by Lambda(s)/Lambda(tip) when it assumes the tip"""
    import twin_core as C
    F, k_b = 0.7, C.CFG["k_b"]
    for k in range(C.NS):
        for m in range(2):
            s = (k + 0.25 + 0.5 * m) * C.L
            miss = math.degrees(F * C.lever_sum(s) / k_b)
            skin = np.zeros(24)
            skin[2 * k + m] = F
            est = C.estimate_force([miss + 10, 10, 10], [10, 10, 10], k_b, skin=skin)
            assert est == pytest.approx(F, rel=1e-9)
            tip = C.estimate_force([miss + 10, 10, 10], [10, 10, 10], k_b)
            assert tip == pytest.approx(F * C.lever_sum(s) / C.lever_sum(C.NS * C.L), rel=1e-9)


def test_same_instance_in_every_engine():
    import twin_core as C
    a, b = C.sample_instance(7, "tomato"), C.sample_instance(7, "tomato")
    assert a == b and C.sample_instance(8, "tomato") != a


@SKIP_SIM
def test_mujoco_contact_stiffness_matches_target():
    """identified effective mass: one finger contact has the instance's stiffness [N/m] within 5 %"""
    cal = json.load(open(ROOT / "sim_mujoco/contact_calibration.json"))
    for n, v in cal["measured"].items():
        assert abs(v["ratio_to_instance"] - 1) < 0.05, n


@SKIP_SIM
def test_one_closed_loop_episode_and_determinism():
    import softtwin_mj as S
    ref = next(r for r in json.load(open(ROOT / "results/benchmark_mujoco.json"))["rows"]
               if r["seed"] == 0 and r["object"] == "tofu" and r["config"] == "closed_learned")
    r, ep = S.run_episode(S.Cell(seed=0, objects=("tofu",)), "tofu", actuator="learned")
    for k in ("success", "crushed", "attempts", "peak_force_true", "F_star_final", "failure"):
        assert r[k] == ref[k], k
    f = np.array([e["f_est"] for e in ep.log])
    assert np.isfinite(f).all()


def test_dataset_validator_quick():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/validate_dataset.py"), "--quick"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_metrics_recompute_and_numbers():
    """results/metrics.json and the report's numbers are recomputed from the raw result files"""
    r = subprocess.run([sys.executable, str(ROOT / "scripts/analysis_stats.py")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    m = json.load(open(ROOT / "results/metrics.json"))
    b = json.load(open(ROOT / "results/benchmark_mujoco.json"))
    rows = [x for x in b["rows"] if x["config"] == "closed_learned"]
    assert m["q1"]["closed_learned"]["success"]["k"] == sum(x["success"] for x in rows)
    assert all(t["n_pairs"] > 0 for t in m["tests"]["paired"])
    r = subprocess.run([sys.executable, str(ROOT / "scripts/make_numbers.py")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
