"""Isaac Sim / Kit extension: exposes Kit to a local MCP server over 127.0.0.1.

Commands: ping, list_commands, info, exec, screenshot, timeline, step, stage_tree,
new_stage, open_stage, save_stage, set_camera.
"""
from __future__ import annotations

import ast
import asyncio
import base64
import contextlib
import io
import os
import tempfile
import time
import traceback

import carb
import omni.ext
import omni.kit.app
import omni.timeline
import omni.usd

from .server import CommandServer

_SETTINGS = "/exts/pai.mcp.bridge"


async def _frames(n: int = 1) -> None:
    app = omni.kit.app.get_app()
    for _ in range(max(1, int(n))):
        await app.next_update_async()


class KitHandlers:
    def __init__(self):
        # persistent namespace so successive exec calls can share state
        self.ns = {"__name__": "__mcp__", "asyncio": asyncio, "omni": omni, "carb": carb, "frames": _frames}

    # ---------- info ----------
    async def info(self, _args):
        s = carb.settings.get_settings()
        stage = omni.usd.get_context().get_stage()
        return {
            "kit_version": omni.kit.app.get_app().get_kit_version(),
            "app_name": s.get("/app/name"),
            "app_version": s.get("/app/version"),
            "stage_url": omni.usd.get_context().get_stage_url(),
            "has_stage": stage is not None,
            "timeline_playing": omni.timeline.get_timeline_interface().is_playing(),
            "ros_bridge": s.get("/isaac/startup/ros_bridge_extension"),
            "pid": os.getpid(),
        }

    # ---------- exec ----------
    async def exec(self, args):
        code = args.get("code", "")
        if args.get("file"):
            with open(args["file"], "r", encoding="utf-8") as f:
                code = f.read()
            self.ns["__file__"] = args["file"]
        if args.get("reset_namespace"):
            self.__init__()
        out = io.StringIO()
        t0 = time.time()
        self.ns.pop("_result", None)
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                compiled = compile(code, args.get("file") or "<mcp>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
                ret = eval(compiled, self.ns)
                if asyncio.iscoroutine(ret):
                    await asyncio.wait_for(ret, timeout=float(args.get("timeout", 600)))
            err = None
        except Exception:
            err = traceback.format_exc()
        res = self.ns.get("_result")
        try:
            import json

            json.dumps(res)
        except Exception:
            res = repr(res)
        return {"stdout": out.getvalue()[-200000:], "result": res, "error": err, "seconds": round(time.time() - t0, 3)}

    # ---------- viewport ----------
    async def screenshot(self, args):
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

        await _frames(int(args.get("settle_frames", 3)))
        path = args.get("path") or os.path.join(tempfile.gettempdir(), f"pai_mcp_{int(time.time()*1000)}.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        vp = get_active_viewport()
        cap = capture_viewport_to_file(vp, file_path=path)
        try:
            await cap.wait_for_result()
        except Exception:
            pass
        for _ in range(120):
            if os.path.exists(path) and os.path.getsize(path) > 0:
                break
            await _frames(1)
        if not os.path.exists(path):
            raise RuntimeError("viewport capture did not produce a file")
        data = open(path, "rb").read()
        return {"path": path, "bytes": len(data), "png_base64": base64.b64encode(data).decode() if args.get("inline", True) else None}

    async def set_camera(self, args):
        eye, target = args["eye"], args["target"]
        try:
            from isaacsim.core.utils.viewports import set_camera_view

            set_camera_view(eye=eye, target=target, camera_prim_path=args.get("camera", "/OmniverseKit_Persp"))
        except Exception:
            from omni.kit.viewport.utility import get_active_viewport
            from pxr import Gf, UsdGeom

            stage = omni.usd.get_context().get_stage()
            cam = stage.GetPrimAtPath(get_active_viewport().camera_path)
            m = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse()
            xf = UsdGeom.Xformable(cam)
            xf.ClearXformOpOrder()
            xf.AddTransformOp().Set(m)
        await _frames(2)
        return {"eye": eye, "target": target}

    # ---------- timeline ----------
    async def timeline(self, args):
        tl = omni.timeline.get_timeline_interface()
        action = args.get("action", "state")
        if action == "play":
            tl.play()
        elif action == "pause":
            tl.pause()
        elif action == "stop":
            tl.stop()
        await _frames(2)
        return {"playing": tl.is_playing(), "stopped": tl.is_stopped(), "time": tl.get_current_time()}

    async def step(self, args):
        n = int(args.get("frames", 1))
        await _frames(n)
        return {"stepped": n, "time": omni.timeline.get_timeline_interface().get_current_time()}

    # ---------- stage ----------
    async def stage_tree(self, args):
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return []
        root = stage.GetPrimAtPath(args.get("root", "/"))
        depth = int(args.get("depth", 3))
        out = []

        def walk(p, d):
            if d > depth:
                return
            out.append({"path": str(p.GetPath()), "type": p.GetTypeName(), "depth": d})
            for c in p.GetChildren():
                walk(c, d + 1)

        walk(root, 0)
        return out[: int(args.get("limit", 500))]

    async def new_stage(self, _args):
        await omni.usd.get_context().new_stage_async()
        await _frames(5)
        return {"ok": True}

    async def open_stage(self, args):
        await omni.usd.get_context().open_stage_async(args["path"])
        await _frames(10)
        return {"stage_url": omni.usd.get_context().get_stage_url()}

    async def save_stage(self, args):
        ok = omni.usd.get_context().save_as_stage(args["path"])
        return {"saved": bool(ok), "path": args["path"]}


class Mailbox:
    """File-based command channel (no sockets, no terminal needed).

    <mailbox>/inbox/NAME.json  {"cmd": "exec", "args": {...}}   or   <mailbox>/inbox/NAME.py (run as exec file=...)
    -> <mailbox>/outbox/NAME.json  {"ok", "result" | "error", "t"}  and the request moves to <mailbox>/done/.
    <mailbox>/status.json is rewritten every few seconds as a heartbeat.
    """

    def __init__(self, root, handlers):
        self.root, self.h = root, handlers
        for d in ("inbox", "outbox", "done", "media"):
            os.makedirs(os.path.join(root, d), exist_ok=True)
        self.t_start = time.time()
        self.n_done = 0
        self.busy = None

    def _write(self, path, obj):
        import json

        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1, default=repr)
        os.replace(tmp, path)

    async def run(self):
        import json

        last_beat = 0.0
        while True:
            try:
                now = time.time()
                if now - last_beat > 3:
                    last_beat = now
                    info = await self.h.info({})
                    info.update(alive=True, t=now, uptime_s=round(now - self.t_start, 1), done=self.n_done, busy=self.busy)
                    self._write(os.path.join(self.root, "status.json"), info)
                inbox = os.path.join(self.root, "inbox")
                for name in sorted(os.listdir(inbox)):
                    src = os.path.join(inbox, name)
                    stem, ext = os.path.splitext(name)
                    if ext not in (".json", ".py"):
                        continue
                    self.busy = name
                    t0 = time.time()
                    try:
                        if ext == ".py":
                            dst_py = os.path.join(self.root, "done", name)
                            os.replace(src, dst_py)
                            req = {"cmd": "exec", "args": {"file": dst_py, "timeout": 3600}}
                        else:
                            with open(src, "r", encoding="utf-8") as f:
                                req = json.load(f)
                            os.replace(src, os.path.join(self.root, "done", name))
                        fn = getattr(self.h, req["cmd"])
                        res = await fn(req.get("args", {}))
                        if isinstance(res, dict):
                            res.pop("png_base64", None)
                        out = {"ok": True, "result": res}
                    except Exception:
                        out = {"ok": False, "error": traceback.format_exc()}
                    out.update(request=name, seconds=round(time.time() - t0, 2), t=time.time())
                    self._write(os.path.join(self.root, "outbox", stem + ".json"), out)
                    self.n_done += 1
                    self.busy = None
            except Exception:
                carb.log_error("[pai.mcp.bridge] mailbox: " + traceback.format_exc())
            await asyncio.sleep(1.0)


class Extension(omni.ext.IExt):
    def on_startup(self, ext_id):
        s = carb.settings.get_settings()
        mailbox = s.get(f"{_SETTINGS}/mailbox") or os.environ.get("PAI_MAILBOX")
        if mailbox:
            self._mailbox = Mailbox(mailbox, KitHandlers())
            self._mtask = asyncio.ensure_future(self._mailbox.run())
            carb.log_warn(f"[pai.mcp.bridge] mailbox at {mailbox}")
        host = s.get(f"{_SETTINGS}/host") or "127.0.0.1"
        port = int(s.get(f"{_SETTINGS}/port") or 8766)
        token = os.environ.get("PAI_MCP_TOKEN") or None
        self._server = CommandServer(host, port, token)
        h = KitHandlers()
        for name in ("info", "exec", "screenshot", "set_camera", "timeline", "step", "stage_tree", "new_stage", "open_stage", "save_stage"):
            self._server.register(name, getattr(h, name))
        self._task = asyncio.ensure_future(self._start(host, port))

    async def _start(self, host, port):
        try:
            await self._server.start()
            carb.log_warn(f"[pai.mcp.bridge] listening on {host}:{port}")
        except Exception as e:
            carb.log_error(f"[pai.mcp.bridge] failed to start: {e}")

    def on_shutdown(self):
        if getattr(self, "_mtask", None):
            self._mtask.cancel()
        if getattr(self, "_server", None):
            asyncio.ensure_future(self._server.stop())
        self._server = None
