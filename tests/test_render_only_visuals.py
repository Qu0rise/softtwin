"""The produce meshes and sepals used for the video are render-only: same trial, same result with or without them."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim_mujoco"))
import softtwin_mj as S  # noqa: E402


def _run(pretty, name):
    S.PRETTY = pretty
    try:
        r, _ = S.run_episode(S.Cell(seed=0, objects=(name,)), name, actuator="learned", max_retries=0)
    finally:
        S.PRETTY = False
    return r["failure"], round(r["peak_force_true"], 6)


def test_visuals_do_not_change_physics():
    for name in ("strawberry", "tomato_overripe"):
        assert _run(False, name) == _run(True, name)
