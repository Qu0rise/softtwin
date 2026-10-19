"""Environment self-test (run via MCP project_run_script('selftest.py') or `pixi run selftest`)."""
import importlib
import json
import platform
import socket
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
out = {"python": sys.version.split()[0], "platform": platform.platform()}
for mod in ("numpy", "rclpy", "mcp", "pika", "websockets", "matplotlib", "requests", "mcap"):
    try:
        importlib.import_module(mod)
        out[mod] = "ok"
    except Exception as e:
        out[mod] = f"missing ({type(e).__name__})"


def port_open(port):
    s = socket.socket()
    s.settimeout(0.5)
    try:
        return s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        s.close()


out["isaac_bridge_8766"] = port_open(8766)
out["ollama_11434"] = port_open(11434)
out["rabbitmq_5672"] = port_open(5672)
try:
    out["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
except Exception as e:
    out["nvidia_smi"] = f"n/a ({e})"
print(json.dumps(out, ensure_ascii=False, indent=1))
