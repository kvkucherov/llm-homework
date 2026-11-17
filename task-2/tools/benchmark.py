"""Reproducible local tournaments with paired starting positions and replays."""
import argparse
import copy
import json
import math
import statistics
import time
import hashlib
from pathlib import Path

from snake.engine import DEFAULT_WEIGHTS, Engine
from .baselines import choose as baseline
from .simulator import advance, initial, spawn_food


def wilson(wins, n):
    if not n:
        return [0.0, 1.0]
    z = 1.96
    center = (wins / n + z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt((wins/n)*(1-wins/n)/n+z*z/(4*n*n))/(1+z*z/n)
    return [round(center-half, 4), round(center+half, 4)]


def game(engine, opponent, seed, swap=False, budget=15, depth=32, weights=DEFAULT_WEIGHTS,
         enemy_weights=DEFAULT_WEIGHTS, enemy_budget=None, enemy_depth=2, max_turns=600,
         mode="standard", record=False, enemy_engine=None):
    state, rng = initial(seed, swap, mode)
    state["game"]["id"] = f"local-{opponent}-{seed}-{int(swap)}"
    frames = []
    timings, depths, nodes = [], [], []
    length_margins = []
    for turn in range(max_turns):
        snakes = state["board"]["snakes"]
        state["you"] = next(s for s in snakes if s["id"] == "viper")
        length_margins.append(len(state["you"]["body"])-len(next(s for s in snakes if s["id"]=="opponent")["body"]))
        started = time.perf_counter()
        own, stats = engine.choose(state, budget, depth, weights)
        timings.append((time.perf_counter()-started)*1000)
        depths.append(stats["depth"]); nodes.append(stats["nodes"])
        enemy_state = {**state, "you": next(s for s in snakes if s["id"] == "opponent")}
        if opponent == "search":
            enemy, _ = (enemy_engine or engine).choose(enemy_state, enemy_budget if enemy_budget is not None else budget, enemy_depth, enemy_weights)
        else:
            enemy = baseline(enemy_state, opponent)
        actions = {"viper": own, "opponent": enemy}
        if record:
            frames.append({"state": copy.deepcopy(state), "moves": actions, "stats": stats})
        state = advance(state, actions)
        alive = [s for s in state["board"]["snakes"] if s["alive"]]
        if len(alive) < 2:
            winner = alive[0]["id"] if alive else None
            result = "win" if winner == "viper" else "loss" if winner else "draw"
            break
        state["board"]["snakes"] = alive
        spawn_food(state, rng)
    else:
        result = "turn_limit"
    if record:
        frames.append({"state": copy.deepcopy(state), "moves": {}, "stats": {}})
    final_snakes={s["id"]:s for s in state["board"]["snakes"]}
    return {"result": result, "seed": seed, "swap": swap, "turns": state["turn"],
            "mean_ms": round(statistics.mean(timings), 3), "max_ms": round(max(timings), 3),
            "mean_depth": round(statistics.mean(depths), 2), "mean_nodes": round(statistics.mean(nodes)),
            "mean_length_margin": round(statistics.mean(length_margins),3),
            "behind_fraction": round(sum(v<0 for v in length_margins)/len(length_margins),4),
            "ahead_fraction": round(sum(v>0 for v in length_margins)/len(length_margins),4),
            "final_lengths": {sid:len(s["body"]) for sid,s in final_snakes.items()},
            "frames": frames, "timings": timings}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--games", type=int, default=12, help="games per opponent, paired positions")
    p.add_argument("--opponents", nargs="+", default=["greedy", "cautious", "territory", "hunter"])
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--budget-ms", type=int, default=15)
    p.add_argument("--depth", type=int, default=32)
    p.add_argument("--enemy-depth", type=int, default=2)
    p.add_argument("--enemy-budget-ms", type=int)
    p.add_argument("--library", help="candidate native library for version A/B matches")
    p.add_argument("--enemy-library", help="frozen native library of the incumbent")
    p.add_argument("--weights")
    p.add_argument("--enemy-weights")
    p.add_argument("--mode", default="standard", choices=["standard", "constrictor", "wrapped"])
    p.add_argument("--max-turns", type=int, default=600)
    p.add_argument("--report", default="reports/benchmark.json")
    p.add_argument("--save-replays", action="store_true")
    args = p.parse_args()
    engine = Engine(args.library)
    enemy_engine = Engine(args.enemy_library) if args.enemy_library else engine
    weights = json.loads(Path(args.weights).read_text())["weights"] if args.weights else DEFAULT_WEIGHTS
    enemy_weights = json.loads(Path(args.enemy_weights).read_text())["weights"] if args.enemy_weights else DEFAULT_WEIGHTS
    report = {"configuration": vars(args), "weights": weights,
              "enemy_weights": enemy_weights,
              "engine_sha256": hashlib.sha256(Path(args.library or "build/libserpent.so").read_bytes()).hexdigest(),
              "enemy_sha256": hashlib.sha256(Path(args.enemy_library or args.library or "build/libserpent.so").read_bytes()).hexdigest(),
              "notice": "Practice opponents, NOT the official course reference snakes. Local simulator; validate with official CLI.",
              "opponents": {}}
    for opponent in args.opponents:
        records, timings = [], []
        for i in range(args.games):
            record = game(engine, opponent, args.seed+i//2, bool(i%2), args.budget_ms, args.depth,
                          weights, enemy_weights, args.enemy_budget_ms, args.enemy_depth, args.max_turns,
                          args.mode, args.save_replays, enemy_engine)
            if args.save_replays:
                path = Path("replays") / Path(args.report).stem / f"{opponent}-{args.seed+i//2}-{i%2}.json"
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(json.dumps(record.pop("frames")))
                record["replay"] = str(path)
            else:
                record.pop("frames")
            timings.extend(record.pop("timings"))
            records.append(record)
            print(f"{opponent:10} {i+1:3}/{args.games}: {record['result']:10} turns={record['turns']:3} "
                  f"depth={record['mean_depth']} max={record['max_ms']:.1f}ms", flush=True)
        counts = {r: sum(g["result"] == r for g in records) for r in ("win", "loss", "draw", "turn_limit")}
        summary = {**counts, "win_rate": counts["win"]/args.games,
                   "win_rate_95pct_wilson": wilson(counts["win"], args.games),
                   "p50_ms": round(statistics.median(timings), 3),
                   "p95_ms": round(sorted(timings)[int((len(timings)-1)*0.95)], 3),
                   "max_ms": round(max(timings), 3), "moves": len(timings)}
        report["opponents"][opponent] = {"summary": summary, "games": records}
        path = Path(args.report); path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(report, indent=2))
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
