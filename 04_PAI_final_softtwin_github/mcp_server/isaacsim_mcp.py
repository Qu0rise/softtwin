"""MCP server (stdio) that lets Claude drive a running Isaac Sim through the pai.mcp.bridge extension.

Run (registered in Claude Desktop config):
    pixi run --manifest-path C:\\pai_ros\\pixi.toml python <repo>\\mcp_server\\isaacsim_mcp.py

Tools are thin wrappers over the bridge commands plus a few project helpers:
  isaac_status, isaac_exec, isaac_run_file, isaac_screenshot, isaac_camera, isaac_timeline,
  isaac_step, isaac_stage_tree, isaac_new_stage, isaac_open_stage, isaac_save_stage,
  project_run_script (only scripts inside <repo>/scripts), project_read_file (inside <repo>),
  ros_topics, ros_echo (if rclpy is importable in this environment).
"""
from __future__ import annotations

import base64
import itertools
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP, Image

HOST = os.environ.get("PAI_BRIDGE_HOST", "127.0.0.1")
PORT = int(os.environ.get("PAI_BRIDGE_PORT", "8766"))
TOKEN = os.environ.get("PAI_MCP_TOKEN") or None
REPO = Path(os.environ.get("PAI_REPO", Path(__file__).resolve().parents[1]))

ISAAC_DIR = Path(os.environ.get("ISAAC_SIM_DIR", r"C:\isaac-sim-standalone-5.0.0-windows-x86_64"))
LOGS = REPO / "tools" / "logs"

mcp = FastMCP("isaacsim")
_ids = itertools.count(1)


def _call(cmd: str, args: Optional[dict] = None, timeout: float = 120.0) -> Any:
    req = {"id": next(_ids), "cmd": cmd, "args": args or {}}
    if TOKEN:
        req["token"] = TOKEN
    try:
        with socket.create_connection((HOST, PORT), timeout=10) as s:
            s.settimeout(timeout)
            s.sendall((json.dumps(req) + "\n").encode())
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = s.recv(1 << 20)
                if not chunk:
                    break
                buf += chunk
    except (ConnectionRefusedError, socket.timeout, OSError) as e:
        raise RuntimeError(
            f"Isaac Sim bridge not reachable at {HOST}:{PORT} ({e}). "
            "Start Isaac Sim with tools\\launch_sim.bat (enables pai.mcp.bridge)."
        )
    resp = json.loads(buf)
    if not resp.get("ok"):
        raise RuntimeError(resp.get("error", "unknown error") + ("\n" + resp.get("traceback", "") if resp.get("traceback") else ""))
    return resp["result"]


def _fmt(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, indent=1, default=str)


@mcp.tool()
def isaac_status() -> str:
    """Check that Isaac Sim is running and the bridge answers. Returns app/version/stage/timeline info."""
    return _fmt(_call("info"))


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection((HOST, port), timeout=1):
            return True
    except OSError:
        return False


@mcp.tool()
def isaac_launch(ros2: bool = False, extra_args: list[str] | None = None, wait_s: float = 240.0) -> str:
    """Start Isaac Sim (GUI) with the pai.mcp.bridge extension and wait until the bridge answers.
    ros2=True also enables isaacsim.ros2.bridge. Does nothing if the bridge is already up."""
    import time

    if _port_open(PORT):
        return _fmt({"already_running": True, "info": _call("info")})
    bat = ISAAC_DIR / "isaac-sim.bat"
    if not bat.exists():
        raise RuntimeError(f"Isaac Sim not found at {ISAAC_DIR} (set ISAAC_SIM_DIR)")
    args = [str(bat), "--ext-folder", str(REPO / "isaac_ext"), "--enable", "pai.mcp.bridge"]
    if ros2:
        args += ["--enable", "isaacsim.ros2.bridge"]
    args += list(extra_args or [])
    LOGS.mkdir(parents=True, exist_ok=True)
    log = open(LOGS / "isaac_launch.log", "a", encoding="utf-8")
    flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    subprocess.Popen(["cmd", "/c", *args], cwd=str(ISAAC_DIR), stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=flags)
    t0 = time.time()
    while time.time() - t0 < wait_s:
        if _port_open(PORT):
            time.sleep(3)
            return _fmt({"started": True, "seconds": round(time.time() - t0, 1), "info": _call("info")})
        time.sleep(3)
    return _fmt({"started": False, "note": f"bridge not up after {wait_s}s; Isaac Sim may still be loading (first start compiles shaders). Call isaac_status later.", "log": str(LOGS / "isaac_launch.log")})


@mcp.tool()
def isaac_quit() -> str:
    """Ask Isaac Sim to quit cleanly."""
    try:
        _call("exec", {"code": "import omni.kit.app\nomni.kit.app.get_app().post_quit()"}, timeout=10)
    except Exception as e:
        return f"quit requested ({e})"
    return "quit requested"


@mcp.tool()
def isaac_log_tail(lines: int = 200) -> str:
    """Tail the newest Kit log file of Isaac Sim (useful when the bridge does not come up)."""
    cands = []
    for base in [Path(os.environ.get("LOCALAPPDATA", "")) / "ov" / "data" / "Kit", ISAAC_DIR / "kit" / "logs"]:
        if base.exists():
            cands += list(base.rglob("*.log"))
    cands += list(LOGS.glob("*.log")) if LOGS.exists() else []
    if not cands:
        return "no log files found"
    newest = max(cands, key=lambda p: p.stat().st_mtime)
    txt = newest.read_text(encoding="utf-8", errors="replace").splitlines()
    return f"{newest}\n" + "\n".join(txt[-lines:])


@mcp.tool()
def isaac_exec(code: str, timeout_s: float = 600.0, reset_namespace: bool = False) -> str:
    """Execute Python inside Isaac Sim (Kit main thread). Top-level `await` is allowed
    (e.g. `await frames(10)`). Set `_result = ...` to return a JSON-able value. Namespace persists between calls."""
    r = _call("exec", {"code": code, "timeout": timeout_s, "reset_namespace": reset_namespace}, timeout=timeout_s + 30)
    return _fmt(r)


@mcp.tool()
def isaac_run_file(relative_path: str, timeout_s: float = 600.0) -> str:
    """Execute a Python file from the project repo inside Isaac Sim (path relative to repo root, e.g. 'sim/build_scene.py')."""
    p = (REPO / relative_path).resolve()
    if REPO.resolve() not in p.parents:
        raise ValueError("file must be inside the project repo")
    r = _call("exec", {"file": str(p), "timeout": timeout_s}, timeout=timeout_s + 30)
    return _fmt(r)


@mcp.tool()
def isaac_screenshot(save_to_repo_as: str = "") -> Image:
    """Capture the active Isaac Sim viewport and return it as an image. Optionally also save under <repo>/media/<name>.png."""
    path = str((REPO / "media" / f"{save_to_repo_as}.png").resolve()) if save_to_repo_as else ""
    r = _call("screenshot", {"path": path} if path else {}, timeout=120)
    return Image(data=base64.b64decode(r["png_base64"]), format="png")


@mcp.tool()
def isaac_camera(eye: list[float], target: list[float]) -> str:
    """Move the viewport camera: eye=[x,y,z], target=[x,y,z] (metres, Z-up)."""
    return _fmt(_call("set_camera", {"eye": eye, "target": target}))


@mcp.tool()
def isaac_timeline(action: str = "state") -> str:
    """Control simulation: action = play | pause | stop | state."""
    return _fmt(_call("timeline", {"action": action}))


@mcp.tool()
def isaac_step(frames: int = 60) -> str:
    """Let the app run N frames (simulation advances if playing)."""
    return _fmt(_call("step", {"frames": frames}, timeout=max(60, frames / 10)))


@mcp.tool()
def isaac_stage_tree(root: str = "/", depth: int = 3, limit: int = 300) -> str:
    """List prims under root."""
    return _fmt(_call("stage_tree", {"root": root, "depth": depth, "limit": limit}))


@mcp.tool()
def isaac_new_stage() -> str:
    """Open an empty stage."""
    return _fmt(_call("new_stage"))


@mcp.tool()
def isaac_open_stage(path: str) -> str:
    """Open a USD file (absolute path or omniverse URL)."""
    return _fmt(_call("open_stage", {"path": path}, timeout=600))


@mcp.tool()
def isaac_save_stage(relative_path: str) -> str:
    """Save current stage into the repo (relative path, e.g. 'assets/scene.usd')."""
    p = (REPO / relative_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    return _fmt(_call("save_stage", {"path": str(p)}))


@mcp.tool()
def project_run_script(script: str, args: list[str] | None = None, timeout_s: float = 900.0, interpreter: str = "isaac") -> str:
    """Run a Python script from <repo>/scripts. interpreter='isaac' uses Isaac Sim's python.bat (numpy, torch, matplotlib),
    'mcp' uses this server's Python. Only files directly inside <repo>/scripts are allowed."""
    p = (REPO / "scripts" / script).resolve()
    if p.parent != (REPO / "scripts").resolve() or p.suffix != ".py" or not p.exists():
        raise ValueError("script must be an existing .py directly inside <repo>/scripts")
    exe = ["cmd", "/c", str(ISAAC_DIR / "python.bat")] if interpreter == "isaac" else [sys.executable]
    cp = subprocess.run([*exe, str(p), *(args or [])], cwd=str(REPO), capture_output=True, text=True, timeout=timeout_s, encoding="utf-8", errors="replace")
    return _fmt({"returncode": cp.returncode, "stdout": cp.stdout[-20000:], "stderr": cp.stderr[-20000:]})


@mcp.tool()
def project_read_file(relative_path: str, max_chars: int = 50000) -> str:
    """Read a text file inside the project repo (logs, results, reports)."""
    p = (REPO / relative_path).resolve()
    if REPO.resolve() not in p.parents:
        raise ValueError("file must be inside the project repo")
    return p.read_text(encoding="utf-8", errors="replace")[-max_chars:]


@mcp.tool()
def project_screenshot_file(relative_path: str) -> Image:
    """Return a PNG/JPG stored inside the repo (e.g. media/xxx.png, results/plot.png)."""
    p = (REPO / relative_path).resolve()
    if REPO.resolve() not in p.parents:
        raise ValueError("file must be inside the project repo")
    return Image(data=p.read_bytes(), format=p.suffix.lstrip(".").replace("jpg", "jpeg"))


@mcp.tool()
def ros_topics() -> str:
    """List ROS 2 topics visible from this environment (needs rclpy)."""
    try:
        import rclpy
        from rclpy.node import Node
    except Exception as e:
        return f"rclpy not available: {e}"
    import time

    rclpy.init()
    try:
        n = Node("pai_mcp_probe")
        time.sleep(1.5)
        return _fmt(sorted(n.get_topic_names_and_types()))
    finally:
        rclpy.shutdown()


@mcp.tool()
def ros_echo(topic: str, count: int = 1, timeout_s: float = 5.0) -> str:
    """Receive up to `count` messages from a ROS 2 topic (type auto-detected)."""
    try:
        import rclpy
        from rclpy.node import Node
        from rosidl_runtime_py.utilities import get_message
        from rosidl_runtime_py.convert import message_to_ordereddict
    except Exception as e:
        return f"rclpy not available: {e}"
    import time

    rclpy.init()
    try:
        n = Node("pai_mcp_echo")
        t_end = time.time() + 2
        types = {}
        while time.time() < t_end and topic not in types:
            types = dict((k, v[0]) for k, v in n.get_topic_names_and_types())
            time.sleep(0.2)
        if topic not in types:
            return f"topic {topic} not found"
        msgs = []
        n.create_subscription(get_message(types[topic]), topic, lambda m: msgs.append(message_to_ordereddict(m)), 10)
        t_end = time.time() + timeout_s
        while time.time() < t_end and len(msgs) < count:
            rclpy.spin_once(n, timeout_sec=0.1)
        return _fmt({"type": types[topic], "messages": msgs})
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    mcp.run()
