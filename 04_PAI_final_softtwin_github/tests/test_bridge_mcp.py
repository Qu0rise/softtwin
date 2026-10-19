"""Offline test: fake Kit bridge (CommandServer + mock handlers) <-> MCP server over stdio."""
import asyncio, base64, importlib.util, os, sys, threading
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("srv", ROOT/"isaac_ext/pai.mcp.bridge/pai/mcp/bridge/server.py")
srv = importlib.util.module_from_spec(spec); spec.loader.exec_module(srv)
PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
ns = {}
async def h_info(a): return {"app_version": "mock", "has_stage": True}
async def h_exec(a):
    import io, contextlib
    out = io.StringIO()
    with contextlib.redirect_stdout(out): exec(a.get("code",""), ns)
    return {"stdout": out.getvalue(), "result": ns.get("_result"), "error": None}
async def h_shot(a): return {"path": "x.png", "png_base64": base64.b64encode(PNG).decode()}
def run_bridge(ready):
    async def main():
        s = srv.CommandServer("127.0.0.1", 18766)
        s.register("info", h_info); s.register("exec", h_exec); s.register("screenshot", h_shot)
        await s.start(); ready.set(); await asyncio.sleep(3600)
    asyncio.run(main())
async def client():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    env = dict(os.environ, PAI_BRIDGE_PORT="18766", PAI_REPO=str(ROOT))
    params = StdioServerParameters(command=sys.executable, args=[str(ROOT/"mcp_server/isaacsim_mcp.py")], env=env)
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = [t.name for t in (await s.list_tools()).tools]
            print("tools:", tools)
            print((await s.call_tool("isaac_status", {})).content[0].text)
            print((await s.call_tool("isaac_exec", {"code": "print('hi'); _result = 1+1"})).content[0].text)
            res = await s.call_tool("isaac_screenshot", {})
            print("screenshot content type:", res.content[0].type, getattr(res.content[0], "mimeType", None))
ready = threading.Event(); threading.Thread(target=run_bridge, args=(ready,), daemon=True).start(); ready.wait(5)
asyncio.run(client())
