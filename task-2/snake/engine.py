"""Small ctypes bridge; no third-party Python dependencies."""
import ctypes
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MOVES = ("up", "down", "left", "right")
DEFAULT_WEIGHTS = (65.0, 3.0, 42.0, 16.0, 1.0)


def encode(payload):
    board = payload["board"]
    width, height = int(board["width"]), int(board["height"])
    if not (2 <= width <= 25 and 2 <= height <= 25):
        raise ValueError("Supported board sizes: 2..25 on each axis")
    rules = payload.get("game", {}).get("ruleset", {})
    name = rules.get("name", "standard")
    if name not in ("standard", "royale", "wrapped", "constrictor", "solo"):
        raise ValueError("Unsupported ruleset: " + name)
    settings = rules.get("settings", {})
    def point(p):
        x, y = int(p["x"]), int(p["y"])
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError("Position outside board")
        return y * width + x
    food = [point(p) for p in board.get("food", [])]
    hazards = [point(p) for p in board.get("hazards", [])]
    snakes = list(board["snakes"])
    uid = payload["you"]["id"]
    own = [s for s in snakes if s["id"] == uid]
    if len(own) != 1:
        raise ValueError("you.id must identify a living snake")
    snakes = own + [s for s in snakes if s["id"] != uid]
    if len(snakes) > 16:
        raise ValueError("At most 16 snakes are supported")
    data = [width, height, int(settings.get("hazardDamagePerTurn", 14)),
            int(name == "wrapped"), int(name == "constrictor"),
            len(food), *food, len(hazards), *hazards, len(snakes)]
    for s in snakes:
        body = [point(p) for p in s["body"]]
        if not body or len(body) > 753:
            raise ValueError("Invalid snake length")
        data.extend([len(body), int(s["health"]), *body])
    return (ctypes.c_int * len(data))(*data), snakes


class Engine:
    def __init__(self, library=None):
        path = Path(library or os.environ.get("SNAKE_LIBRARY", ROOT / "build/libserpent.so"))
        if not path.exists():
            raise RuntimeError("Build the native engine first: make")
        self.lib = ctypes.CDLL(str(path))
        ip, dp = ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double)
        self.lib.bs_choose.argtypes = (ip, ctypes.c_int, ctypes.c_int, ctypes.c_int, dp, dp)
        self.lib.bs_choose.restype = ctypes.c_int
        self.lib.bs_step.argtypes = (ip, ctypes.c_int, ip, ip, ctypes.c_int)
        self.lib.bs_step.restype = ctypes.c_int

    def choose(self, payload, budget_ms=280, max_depth=32, weights=DEFAULT_WEIGHTS):
        data, _ = encode(payload)
        stats = (ctypes.c_double * 4)()
        w = (ctypes.c_double * 5)(*weights)
        timeout = int(payload.get("game", {}).get("timeout", 500))
        budget = max(1, min(int(budget_ms), int(timeout * 0.65), 350))
        move = self.lib.bs_choose(data, len(data), budget, max_depth, w, stats)
        if move not in range(4):
            raise ValueError("Native engine rejected board")
        return MOVES[move], {"depth": int(stats[0]), "nodes": int(stats[1]),
                             "score": round(stats[2], 2), "engine_ms": round(stats[3], 3)}

    def transition(self, payload, moves):
        data, snakes = encode(payload)
        actions = (ctypes.c_int * len(snakes))(*(MOVES.index(moves[s["id"]]) for s in snakes))
        out = (ctypes.c_int * 16000)()
        size = self.lib.bs_step(data, len(data), actions, out, len(out))
        if size < 0:
            raise ValueError("Transition failed")
        k = 0
        def read():
            nonlocal k
            v = out[k]; k += 1
            return v
        width = payload["board"]["width"]
        def point(p):
            return {"x": p % width, "y": p // width} if p >= 0 else {"x": -1, "y": -1}
        food = [point(read()) for _ in range(read())]
        result = []
        for i in range(read()):
            alive, health, length = read(), read(), read()
            body = [point(read()) for _ in range(length)]
            result.append({**snakes[i], "alive": bool(alive), "health": health,
                           "body": body, "head": body[0], "length": length})
        return food, result
