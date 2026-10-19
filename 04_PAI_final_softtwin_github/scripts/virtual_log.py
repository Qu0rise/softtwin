"""Wrapper: generate a virtual M5Stack log offline (python scripts/virtual_log.py [seconds])."""
import runpy, sys
from pathlib import Path
sec = sys.argv[1] if len(sys.argv) > 1 else "60"
sys.argv = ["virtual_m5stack.py", "--offline", "--seconds", sec]
runpy.run_path(str(Path(__file__).resolve().parents[1] / "core" / "virtual_m5stack.py"), run_name="__main__")
