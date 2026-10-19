"""Generic JSON-lines command server (no Kit dependency, unit-testable).

Protocol: one JSON object per line.
  request : {"id": <any>, "cmd": "<name>", "args": {...}, "token": "<optional>"}
  response: {"id": <same>, "ok": true, "result": ...} | {"id":..., "ok": false, "error": "..."}
"""
from __future__ import annotations

import asyncio
import json
import traceback
from typing import Any, Awaitable, Callable, Dict, Optional

Handler = Callable[[Dict[str, Any]], Awaitable[Any]]

MAX_LINE = 64 * 1024 * 1024  # screenshots come back base64 encoded


class CommandServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 8766, token: Optional[str] = None):
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("bridge must bind to localhost only")
        self.host, self.port, self.token = host, port, token
        self.handlers: Dict[str, Handler] = {}
        self._server: Optional[asyncio.AbstractServer] = None
        self.register("ping", self._ping)
        self.register("list_commands", self._list)

    def register(self, name: str, fn: Handler) -> None:
        self.handlers[name] = fn

    async def _ping(self, _args):
        return {"pong": True}

    async def _list(self, _args):
        return sorted(self.handlers)

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._client, self.host, self.port, limit=MAX_LINE)

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            try:
                await self._server.wait_closed()
            except Exception:
                pass
            self._server = None

    async def _client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                resp = await self.dispatch(line)
                writer.write((json.dumps(resp, default=str) + "\n").encode())
                await writer.drain()
        except Exception:
            pass
        finally:
            try:
                writer.close()
            except Exception:
                pass

    async def dispatch(self, raw: bytes) -> Dict[str, Any]:
        rid = None
        try:
            req = json.loads(raw)
            rid = req.get("id")
            if self.token and req.get("token") != self.token:
                return {"id": rid, "ok": False, "error": "unauthorized"}
            cmd = req.get("cmd")
            fn = self.handlers.get(cmd)
            if fn is None:
                return {"id": rid, "ok": False, "error": f"unknown command: {cmd}"}
            result = await fn(req.get("args") or {})
            return {"id": rid, "ok": True, "result": result}
        except Exception as e:  # report, never crash the app
            return {"id": rid, "ok": False, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
