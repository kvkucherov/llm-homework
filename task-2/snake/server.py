"""Battlesnake API v1 server. Run with python3 -m snake.server."""
import argparse
import json
import logging
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from .engine import DEFAULT_WEIGHTS, Engine

LOG = logging.getLogger("viper")


def handler_factory(engine, budget_ms=280, depth=32, policy=None, weights=DEFAULT_WEIGHTS):
    class Handler(BaseHTTPRequestHandler):
        server_version = "Viper/0.2"
        def setup(self):
            super().setup()
            self.connection.settimeout(5)

        def reply(self, status, data):
            raw = json.dumps(data, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            try:
                self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if urlsplit(self.path).path not in ("/", "/health"):
                return self.reply(404, {"error": "not found"})
            self.reply(200, {"apiversion": "1", "author": os.environ.get("SNAKE_AUTHOR", ""),
                             "color": "#36d6a0", "head": "default", "tail": "default",
                             "version": "viper-0.2"})

        def do_POST(self):
            path = urlsplit(self.path).path
            if path not in ("/start", "/move", "/end"):
                return self.reply(404, {"error": "not found"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 1024 * 1024:
                    return self.reply(413, {"error": "invalid body size"})
                payload = json.loads(self.rfile.read(size))
                if path != "/move":
                    return self.reply(200, {})
                started = time.perf_counter()
                if policy:
                    from tools.baselines import choose
                    move, stats = choose(payload, policy), {}
                else:
                    move, stats = engine.choose(payload, budget_ms, depth, weights)
                elapsed = (time.perf_counter() - started) * 1000
                LOG.info(json.dumps({"game": payload.get("game", {}).get("id"),
                                     "turn": payload.get("turn"), "move": move,
                                     "elapsed_ms": round(elapsed, 3), **stats}))
                self.reply(200, {"move": move})
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                LOG.warning("Invalid request: %s", exc)
                self.reply(400, {"error": "invalid Battlesnake request"})

        def log_message(self, fmt, *args):
            LOG.debug(fmt, *args)
    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    p.add_argument("--budget-ms", type=int, default=int(os.environ.get("MOVE_BUDGET_MS", "280")))
    p.add_argument("--depth", type=int, default=int(os.environ.get("MAX_DEPTH", "32")))
    p.add_argument("--policy", choices=("greedy", "cautious", "territory", "hunter"))
    p.add_argument("--weights", help="JSON file with a weights list")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    weights = DEFAULT_WEIGHTS
    if args.weights:
        with open(args.weights) as f:
            weights = json.load(f)["weights"]
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(message)s")
    engine = Engine()
    server = ThreadingHTTPServer((args.host, args.port), handler_factory(engine, args.budget_ms, args.depth, args.policy, weights))
    LOG.warning("Viper listening on http://%s:%s (budget %sms, depth %s)", args.host, server.server_port, args.budget_ms, args.depth)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
