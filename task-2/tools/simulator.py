"""Independent, readable Standard/Constrictor/Wrapped transition oracle.

This simulator is for fast experiments. The official CLI remains the authority.
Random spawns are seeded, with no access to future food given to the agent.
"""
import copy
import random

DELTAS = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}


def point(x, y):
    return {"x": x, "y": y}


def snake(sid, body, health=100):
    body = [point(*p) if isinstance(p, tuple) else dict(p) for p in body]
    return {"id": sid, "name": sid, "health": health, "body": body,
            "head": body[0], "length": len(body), "shout": "", "latency": "0"}


def request(snakes, food=(), width=11, height=11, hazards=(), mode="standard", turn=0, you=0):
    return {"game": {"id": "local", "timeout": 500,
                     "ruleset": {"name": mode, "version": "v1.2.3", "settings": {
                         "foodSpawnChance": 15, "minimumFood": 1, "hazardDamagePerTurn": 14,
                         "royale": {"shrinkEveryNTurns": 25}}}},
            "turn": turn, "board": {"width": width, "height": height,
                                     "snakes": snakes, "food": [point(*p) if isinstance(p, tuple) else p for p in food],
                                     "hazards": [point(*p) if isinstance(p, tuple) else p for p in hazards]},
            "you": snakes[you]}


def advance(payload, actions):
    state = copy.deepcopy(payload)
    board = state["board"]
    mode = state["game"]["ruleset"]["name"]
    damage = state["game"]["ruleset"]["settings"].get("hazardDamagePerTurn", 14)
    food = {(p["x"], p["y"]) for p in board["food"]}
    eaten = set()
    snakes = board["snakes"]
    for s in snakes:
        dx, dy = DELTAS[actions[s["id"]]]
        x, y = s["body"][0]["x"] + dx, s["body"][0]["y"] + dy
        if mode == "wrapped":
            x, y = x % board["width"], y % board["height"]
        head = point(x, y)
        s["body"] = [head] + s["body"][:-1]
        s["health"] -= 1
        eat = (x, y) in food
        if not eat:
            s["health"] -= damage * board["hazards"].count(head)
        if eat or mode == "constrictor":
            s["health"] = 100
            s["body"].append(dict(s["body"][-1]))
        if eat:
            eaten.add((x, y))
        s["head"], s["length"] = head, len(s["body"])
        s["health"] = max(0, s["health"])
        s["alive"] = s["health"] > 0 and 0 <= x < board["width"] and 0 <= y < board["height"]
    active = [s for s in snakes if s["alive"]]
    killed = set()
    for s in active:
        if any(s["head"] in o["body"][1:] for o in active):
            killed.add(s["id"])
        if any(o["id"] != s["id"] and o["head"] == s["head"] and o["length"] >= s["length"] for o in active):
            killed.add(s["id"])
    for s in snakes:
        s["alive"] &= s["id"] not in killed
    board["food"] = [p for p in board["food"] if (p["x"], p["y"]) not in eaten]
    state["turn"] += 1
    return state


def spawn_food(state, rng):
    board = state["board"]
    if state["game"]["ruleset"]["name"] == "constrictor":
        return
    settings = state["game"]["ruleset"]["settings"]
    count = max(0, settings.get("minimumFood", 1) - len(board["food"]))
    if count == 0 and rng.randrange(100) < settings.get("foodSpawnChance", 15):
        count = 1
    occupied = {(p["x"], p["y"]) for s in board["snakes"] for p in s["body"]}
    occupied.update((p["x"], p["y"]) for p in board["food"])
    # Official standard spawning excludes cells adjacent to any living head.
    for s in board["snakes"]:
        for dx, dy in DELTAS.values():
            occupied.add((s["head"]["x"] + dx, s["head"]["y"] + dy))
    empty = [(x, y) for y in range(board["height"]) for x in range(board["width"]) if (x, y) not in occupied]
    for xy in rng.sample(empty, min(count, len(empty))):
        board["food"].append(point(*xy))


def initial(seed, swap=False, mode="standard"):
    rng = random.Random(seed)
    spots = [(1, 1), (1, 5), (1, 9), (5, 1), (5, 9), (9, 1), (9, 5), (9, 9)]
    a, b = rng.sample(spots, 2)
    food = [(5, 5)]
    for x, y in (a, b):
        options = [(x+dx, y+dy) for dx,dy in ((1,1),(1,-1),(-1,1),(-1,-1))
                   if 0 <= x+dx < 11 and 0 <= y+dy < 11 and (x+dx,y+dy) not in food and (x+dx,y+dy) not in (a,b)]
        food.append(rng.choice(options))
    if swap:
        a, b = b, a
    snakes = [snake("viper", [a] * 3), snake("opponent", [b] * 3)]
    return request(snakes, [] if mode == "constrictor" else food, mode=mode), rng
