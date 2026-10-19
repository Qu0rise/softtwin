"""UI gateway: HTTP (dashboard page) + WebSocket (live metrics) for the React/HTML monitor.

Sources (first available): RabbitMQ (RMQ_URL, exchange 'softtwin', key metrics.#) -> ROS 2 /twin/metrics -> replay of results/engine_*.jsonl
Open http://localhost:8080 after starting:  pixi run --manifest-path C:\\pai_ros\\pixi.toml ui   (or --replay)
"""
from __future__ import annotations

import argparse
import asyncio
import functools
import glob
import http.server
import json
import os
import threading
from pathlib import Path

import websockets

REPO = Path(__file__).resolve().parents[1]
CLIENTS: set = set()
LATEST: dict = {}


async def broadcast(msg: dict):
    LATEST.update(msg)
    data = json.dumps(msg)
    for ws in list(CLIENTS):
        try:
            await ws.send(data)
        except Exception:
            CLIENTS.discard(ws)


async def ws_handler(ws):
    CLIENTS.add(ws)
    if LATEST:
        await ws.send(json.dumps(LATEST))
    try:
        await ws.wait_closed()
    finally:
        CLIENTS.discard(ws)


def serve_http(port):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(REPO))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"dashboard: http://localhost:{port}/ui/dashboard.html")


async def source_replay(loop_forever=True):
    files = sorted(glob.glob(str(REPO / "results" / "engine_*.jsonl")))
    bench = REPO / "results" / "benchmark.json"
    ident = REPO / "results" / "actuator_identification.json"
    static = {}
    if bench.exists():
        static["benchmark"] = json.load(open(bench, encoding="utf-8"))["summary"]
    if ident.exists():
        static["identification"] = json.load(open(ident, encoding="utf-8"))
    while True:
        if static:
            await broadcast(static)
        if not files:
            await asyncio.sleep(2)
        for f in files[-1:]:
            for line in open(f, encoding="utf-8"):
                await broadcast({"metrics": json.loads(line)})
                await asyncio.sleep(0.1)
        if not loop_forever:
            return


async def source_ros(loop):
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    rclpy.init()
    node = Node("softtwin_ui_gateway")
    node.create_subscription(String, "/twin/metrics", lambda m: asyncio.run_coroutine_threadsafe(broadcast({"metrics": json.loads(m.data)}), loop), 10)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    await asyncio.Future()


async def source_rmq(loop, url):
    import pika

    def run():
        ch = pika.BlockingConnection(pika.URLParameters(url)).channel()
        ch.exchange_declare("softtwin", exchange_type="topic", durable=False)
        q = ch.queue_declare("", exclusive=True).method.queue
        ch.queue_bind(q, "softtwin", "metrics.#")
        ch.basic_consume(q, lambda c, m, p, b: asyncio.run_coroutine_threadsafe(broadcast({"metrics": json.loads(b)}), loop), auto_ack=True)
        ch.start_consuming()

    threading.Thread(target=run, daemon=True).start()
    await asyncio.Future()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--http", type=int, default=8080)
    ap.add_argument("--ws", type=int, default=8765)
    ap.add_argument("--replay", action="store_true")
    a = ap.parse_args()
    serve_http(a.http)
    loop = asyncio.get_running_loop()
    async with websockets.serve(ws_handler, "127.0.0.1", a.ws):
        if a.replay:
            await source_replay()
        elif os.environ.get("RMQ_URL"):
            await source_rmq(loop, os.environ["RMQ_URL"])
        else:
            try:
                await source_ros(loop)
            except ImportError:
                await source_replay()


if __name__ == "__main__":
    asyncio.run(main())
