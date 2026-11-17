"""Run matches with the official Battlesnake CLI and verify native transitions."""
import argparse
import json
import re
import subprocess
from pathlib import Path

from snake.engine import Engine
from .simulator import DELTAS


def verify_replay(path):
    frames = [json.loads(s) for s in Path(path).read_text().splitlines()]
    frames = [f for f in frames if "board" in f]
    engine, checked = Engine(), 0
    for before, after in zip(frames, frames[1:]):
        # Terminal JSON omits dead snakes, so only infer fully observed joint moves.
        old = {s["id"]: s for s in before["board"]["snakes"]}
        new = {s["id"]: s for s in after["board"]["snakes"]}
        if old.keys() != new.keys():
            continue
        actions = {}
        for sid in old:
            a, b = old[sid]["head"], new[sid]["head"]
            delta = (b["x"]-a["x"], b["y"]-a["y"])
            if before["game"]["ruleset"]["name"] == "wrapped":
                w, h = before["board"]["width"], before["board"]["height"]
                delta = ((delta[0]+w//2)%w-w//2,(delta[1]+h//2)%h-h//2)
            actions[sid] = next(k for k,v in DELTAS.items() if v == delta)
        before["you"] = next(iter(old.values()))
        _, predicted = engine.transition(before, actions)
        for s in predicted:
            for key in ("health", "body", "length"):
                if s[key] != new[s["id"]][key]:
                    raise AssertionError(f"Turn {before['turn']}, {s['id']}, {key}: native != official")
        checked += 1
    latency = [int(s.get("latency", 0)) for f in frames[1:] for s in f["board"]["snakes"] if s["name"] == "viper"]
    return {"verified_transitions": checked, "p95_latency_ms": sorted(latency)[int((len(latency)-1)*.95)] if latency else None,
            "max_latency_ms": max(latency, default=0)}


def run_series(cli, ours, opponent, games=11, seed=5000, mode="standard", directory="local/matches"):
    """Play actual HTTP matches; fail on transport/engine errors instead of awarding points."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(games):
        output = directory / f"game-{i + 1:02}.jsonl"
        pairs = [("viper", ours), ("opponent", opponent)]
        if i % 2:
            pairs.reverse()
        cmd = [str(cli), "play", "--seed", str(seed + i), "--sequential", "--timeout", "500",
               "--gametype", mode, "--width", "11", "--height", "11", "--output", str(output)]
        for name, url in pairs:
            cmd += ["--name", name, "--url", url]
        run = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600)
        output.with_suffix(".log").write_text(run.stdout)
        if run.returncode or re.search(r"\b(error|failed|timeout)\b|timed out|connection refused", run.stdout, re.I):
            raise RuntimeError(f"Official engine/API error; inspect {output.with_suffix('.log')}")
        done = next((line for line in run.stdout.splitlines() if "Game completed" in line), "")
        if "viper was the winner" in done:
            result = "win"
        elif "opponent was the winner" in done:
            result = "loss"
        elif "draw" in done.lower():
            result = "draw"
        else:
            raise RuntimeError(f"Unknown game result: {done!r}")
        stats = verify_replay(output)
        if stats["max_latency_ms"] >= 500:
            raise RuntimeError(f"Move reached the 500 ms limit; inspect {output}")
        record = {"seed": seed + i, "result": result, "completion": done,
                  "replay": str(output), **stats}
        records.append(record)
        report = {"ours": ours, "opponent": opponent, "game_count": games, "games": records,
                  "summary": {r: sum(x["result"] == r for x in records) for r in ("win", "loss", "draw")}}
        (directory / "summary.json").write_text(json.dumps(report, indent=2))
        print(f"{i + 1}/{games}: {result}; p95={stats['p95_latency_ms']} ms", flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cli", default="vendor/battlesnake/battlesnake")
    p.add_argument("--ours", default="http://127.0.0.1:8000")
    p.add_argument("--opponent", default="http://127.0.0.1:8001")
    p.add_argument("--games", type=int, default=11)
    p.add_argument("--seed", type=int, default=5000)
    p.add_argument("--mode", default="standard")
    p.add_argument("--report", default="reports/official.json")
    p.add_argument("--verify", help="verify one existing JSONL replay")
    args = p.parse_args()
    if args.verify:
        print(json.dumps(verify_replay(args.verify)));return
    path = Path(args.report)
    report = run_series(args.cli, args.ours, args.opponent, args.games, args.seed,
                        args.mode, Path("replays") / path.stem)
    report["configuration"] = vars(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":main()
