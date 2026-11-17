"""Original practice opponents. These are NOT the four course reference snakes."""
from collections import deque
from .simulator import DELTAS


def choose(payload, policy="cautious"):
    b, us = payload["board"], payload["you"]
    w, h = b["width"], b["height"]
    wrapped = payload["game"]["ruleset"]["name"] == "wrapped"
    enemies = [s for s in b["snakes"] if s["id"] != us["id"]]
    blocked = {(p["x"], p["y"]) for s in b["snakes"] for p in s["body"][:-1]}
    food = {(p["x"], p["y"]) for p in b["food"]}
    hazards = {(p["x"], p["y"]) for p in b.get("hazards", [])}
    damage = payload["game"]["ruleset"]["settings"].get("hazardDamagePerTurn", 14)
    def neighbors(p):
        for dx, dy in DELTAS.values():
            q = (p[0] + dx, p[1] + dy)
            if wrapped:
                q = (q[0] % w, q[1] % h)
            if 0 <= q[0] < w and 0 <= q[1] < h:
                yield q
    threats = set()
    for s in enemies:
        if len(s["body"]) >= len(us["body"]):
            threats.update(neighbors((s["head"]["x"], s["head"]["y"])))
    def bfs(start):
        dist = {start: 0}; queue = deque([start])
        while queue:
            p = queue.popleft()
            for q in neighbors(p):
                if q not in blocked and q not in dist:
                    dist[q] = dist[p] + 1; queue.append(q)
        return dist
    enemy_dist = [bfs((s["head"]["x"], s["head"]["y"])) for s in enemies] if policy in ("territory", "hunter") else []
    candidates = []
    for index, (move, (dx, dy)) in enumerate(DELTAS.items()):
        p = (us["head"]["x"] + dx, us["head"]["y"] + dy)
        if wrapped:
            p = (p[0] % w, p[1] % h)
        if p in blocked or not (0 <= p[0] < w and 0 <= p[1] < h):
            continue
        if p not in food and us["health"] <= 1 + damage * (p in hazards):
            continue
        dist = bfs(p)
        nearest = min((dist.get(f, 1000) for f in food), default=30)
        score = -nearest * (3 if us["health"] < 35 else 1)
        if policy != "greedy":
            score += min(len(dist), len(us["body"]) * 2) * 0.4
            score -= max(0, len(us["body"]) + 2 - len(dist)) * 20
            score -= 1000 * (p in threats)
        if policy in ("territory", "hunter"):
            territory = sum(all(d < ed.get(q, 1000) for ed in enemy_dist) for q, d in dist.items())
            score += territory * 0.7
            score -= (abs(p[0]-w//2)+abs(p[1]-h//2))*0.3
        if policy == "hunter":
            for s in enemies:
                if len(s["body"]) < len(us["body"]):
                    score -= 0.7 * (abs(p[0]-s["head"]["x"])+abs(p[1]-s["head"]["y"]))
        score -= 5 * (p in hazards)
        candidates.append((score, -index, move))
    return max(candidates)[2] if candidates else "up"
