"""Wrapper: identify the actuator model (python scripts/train_actuator.py [--csv data/real_log_x.csv | --synthetic])."""
import runpy, sys
from pathlib import Path
sys.argv = ["actuator_net.py"] + (sys.argv[1:] or ["--synthetic"])
runpy.run_path(str(Path(__file__).resolve().parents[1] / "learning" / "actuator_net.py"), run_name="__main__")
