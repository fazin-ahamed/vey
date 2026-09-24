"""Run, record, replay and measure Vey's feature-assisted Snake demo."""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import nullcontext
from datetime import datetime, timezone
from importlib.metadata import version
import io
import json
import os
from pathlib import Path
import platform
import select
import subprocess
import sys
import time

import numpy as np
import torch
from rich.console import Console
from rich.live import Live

from .vey_snake import ORDER, SnakeGame, TorchScorer, Trust, train_demo_head
from .snake_tui import MIN_HEIGHT, MIN_WIDTH, dashboard, layout_size


def snapshot(game: SnakeGame) -> dict:
    return {"width": game.width, "height": game.height, "snake": [list(p) for p in game.snake],
            "food": list(game.food) if game.food is not None else None, "score": game.score,
            "steps": game.steps, "alive": game.alive, "direction": game.direction}


def provenance(args, backend: str) -> dict:
    root = Path(__file__).resolve().parent.parent
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False)
    return {"git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
            "python": platform.python_version(), "rich": version("rich"),
            "numpy": np.__version__, "torch": torch.__version__, "backend": backend,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "platform": platform.platform(),
            "generated_utc": datetime.now(timezone.utc).isoformat(), "model": "demo-specific distilled SuccessHead 9→32→1",
            "interpretation": "feature-assisted demo, not production routing or raw-board perception"}


def prepare(args):
    head = train_demo_head(args.seed, args.train_episodes, args.train_epochs)
    return TorchScorer(head), "torch-cpu"


def analyze_actions(game):
    out = {}
    for d in ORDER:
        a = game.analyze(d)
        reachable_food = False
        if a.legal and game.food is not None:
            body, grows = game._sim_body(d)
            blocked = set(body[1:])
            # Connectivity estimate only; a moving tail can change this later.
            reachable_food = game.food == body[0] or (not grows and game.food not in blocked and
                game.food in _reachable(game, body[0], blocked))
        out[d] = {"legal": a.legal, "area": a.area, "mobility": a.mobility,
                  "food_progress": a.food_progress, "eats": a.eats, "cycle_match": a.cycle_match,
                  "food_reachable_est": reachable_food,
                  "dead_end_est": a.legal and a.area < 0.35}
    return out


def _reachable(game, start, blocked):
    seen = {start}
    pending = [start]
    while pending:
        x, y = pending.pop()
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            p = (x + dx, y + dy)
            if game._inside(p) and p not in blocked and p not in seen:
                seen.add(p)
                pending.append(p)
    return seen


def move(game, trust, scorer, *, shield=True, floor=0.5):
    before = snapshot(game)
    rows = [game.features(d, trust.ema.get(d, 0.8)) for d in ORDER]
    t = time.perf_counter()
    probs = np.asarray(scorer.score_rows(rows), dtype=np.float32)
    infer_ms = (time.perf_counter() - t) * 1000
    if probs.shape != (4,) or not np.isfinite(probs).all():
        raise ValueError("head must return four finite action scores")
    analysis = analyze_actions(game)
    masked = np.where([analysis[d]["legal"] for d in ORDER], probs, -np.inf)
    raw = ORDER[int(np.argmax(probs))]
    if not np.isfinite(masked).any():
        proposal = executed = game.direction
        lcb = trust.lcb(proposal)
        gate = "NO LEGAL MOVE"
        game.alive = False
        game.deaths += 1
    else:
        proposal = ORDER[int(np.argmax(masked))]
        lcb = trust.lcb(proposal)
        if not shield:
            executed, gate = raw, "RAW HEAD · NO SHIELD"
        elif lcb < floor:
            executed, gate = game.planner_choice(), "EMA-LCB ESCALATE"
        elif not game.safety_accepts(proposal):
            executed, gate = game.planner_choice(), "SAFETY SHIELD"
        else:
            executed, gate = proposal, "MODEL"
        safe_before = game.safety_accepts(executed)
        alive = game.step(executed)
        if shield:
            trust.observe(executed, bool(alive and safe_before))
    return {"before": before, "after": snapshot(game), "analysis": analysis,
            "probabilities": {d: float(probs[i]) for i, d in enumerate(ORDER)},
            "raw_proposal": raw, "proposal": proposal, "executed": executed,
            "gate": gate, "trust_lcb": float(lcb), "inference_ms": infer_ms,
            "intervened": bool(shield and executed != proposal), "illegal_raw": not analysis[raw]["legal"]}


class Terminal:
    def __enter__(self):
        self.fd = None
        if sys.stdin.isatty():
            import termios
            import tty
            self.fd = sys.stdin.fileno()
            self.old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def read(self):
        if self.fd is None or not select.select([sys.stdin], [], [], 0)[0]:
            return None
        ch = sys.stdin.read(1)
        if ch == "\x1b" and select.select([sys.stdin], [], [], 0.02)[0]:
            ch += sys.stdin.read(1)
            if ch == "\x1b[" and select.select([sys.stdin], [], [], 0.02)[0]:
                ch += sys.stdin.read(1)
        return ch

    def __exit__(self, *_):
        if self.fd is not None:
            import termios
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


def serialize(frame, floor, width=MIN_WIDTH, height=MIN_HEIGHT):
    stream = io.StringIO()
    console = Console(file=stream, force_terminal=True, color_system="truecolor", width=width, height=height)
    console.print(dashboard(frame, floor, width=width, height=height))
    return stream.getvalue()


def run(args, scorer=None, backend=None):
    if scorer is None:
        scorer, backend = prepare(args)
    game = SnakeGame(args.width, args.height, args.seed)
    trust = Trust(args.trust_floor)
    fps = max(1.0, args.fps)
    paused = False
    last_frame = None
    interventions = total = round_num = 0
    timestamps = deque(maxlen=120)
    started = time.perf_counter()
    record = open(args.record, "w", encoding="utf-8") if args.record else None
    if record:
        record.write(json.dumps({"type": "header", "schema": 1, "config": vars(args),
                                 "provenance": provenance(args, backend)}) + "\n")
    console = Console(color_system="truecolor")
    tty = not args.headless and sys.stdin.isatty() and sys.stdout.isatty()
    live = Live("Starting Vey Snake", console=console, auto_refresh=False, screen=True) if tty else nullcontext()
    try:
        with Terminal() as terminal, live:
            while total < args.steps and (args.duration is None or time.perf_counter() - started < args.duration):
                key = terminal.read() if tty else None
                if key in ("q", "Q"):
                    break
                if key == " ":
                    paused = not paused
                elif key in ("r", "R"):
                    round_num += 1
                    game = SnakeGame(args.width, args.height, args.seed + round_num)
                    trust = Trust(args.trust_floor)
                elif key in ("+", "=", "\x1b[A"):
                    fps += 2
                elif key in ("-", "\x1b[B"):
                    fps = max(1.0, fps - 2)
                if tty:
                    size = os.get_terminal_size(sys.stdout.fileno())
                    required = layout_size(game.width, game.height)
                    if size.columns < required[0] or size.lines < required[1]:
                        live.update(f"Resize terminal to at least {required[0]} columns × {required[1]} rows.\n"
                                    "The game is waiting. Q quits.", refresh=True)
                        time.sleep(.1)
                        continue
                if paused:
                    if tty and last_frame is not None and key is not None:
                        last_frame["paused"] = True
                        size = os.get_terminal_size(sys.stdout.fileno())
                        live.update(dashboard(last_frame, args.trust_floor, width=size.columns, height=size.lines), refresh=True)
                    time.sleep(.03)
                    continue
                t0 = time.perf_counter()
                result = move(game, trust, scorer, shield=not args.unassisted, floor=args.trust_floor)
                total += 1
                interventions += result["intervened"]
                timestamps.append(time.perf_counter())
                rate = (len(timestamps) - 1) / max(1e-9, timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0
                frame = {"type": "move", **result, "step": total, "round": round_num,
                         "seed": args.seed + round_num, "backend": backend, "interventions": interventions,
                         "fps": fps, "rate": rate, "paused": False, "mode": "raw" if args.unassisted else "shielded",
                         "elapsed_s": time.perf_counter() - started, "best_score": game.best, "tick_ms": 0.0}
                last_frame = frame
                if tty:
                    size = os.get_terminal_size(sys.stdout.fileno())
                    frame["tick_ms"] = (time.perf_counter() - t0) * 1000
                    live.update(dashboard(frame, args.trust_floor, width=size.columns, height=size.lines), refresh=True)
                frame["tick_ms"] = (time.perf_counter() - t0) * 1000
                if record:
                    record.write(json.dumps(frame) + "\n")
                if not game.alive:
                    round_num += 1
                    best, deaths = game.best, game.deaths
                    game = SnakeGame(args.width, args.height, args.seed + round_num)
                    game.best, game.deaths = best, deaths
                    trust = Trust(args.trust_floor)
                if not args.max_speed:
                    time.sleep(max(0.0, 1.0 / fps - (time.perf_counter() - t0)))
    finally:
        if record:
            record.close()
    elapsed = time.perf_counter() - started
    summary = {"moves": total, "elapsed_s": round(elapsed, 3),
               "moves_per_s": round(total / max(elapsed, 1e-9), 2), "best_score": game.best,
               "deaths": game.deaths, "interventions": interventions, "backend": backend,
               "mode": "raw" if args.unassisted else "shielded",
               "note": "feature-assisted demo-specific head; not the routing benchmark"}
    if args.headless or not tty:
        print(json.dumps(summary, indent=2))
    return summary


def replay(path):
    """Re-simulate each recorded action; reject any discrepancy or sequence gap."""
    game = None
    trust = None
    count = 0
    with open(path, encoding="utf-8") as stream:
        header = json.loads(next(stream))
        if header.get("type") != "header" or header.get("schema") != 1:
            raise ValueError("unsupported recording header")
        for line in stream:
            row = json.loads(line)
            if row.get("type") != "move" or row["step"] != count + 1:
                raise ValueError(f"record sequence mismatch at row {count + 1}")
            if game is None or game.seed != row["seed"]:
                b = row["before"]
                game = SnakeGame(b["width"], b["height"], row["seed"])
                trust = Trust(header["config"]["trust_floor"])
            if snapshot(game) != row["before"]:
                raise ValueError(f"pre-state mismatch at step {row['step']}")
            d = row["executed"]
            if d not in ORDER:
                raise ValueError(f"invalid action at step {row['step']}")
            probs = np.array([row["probabilities"][direction] for direction in ORDER])
            if not np.isfinite(probs).all() or row["raw_proposal"] != ORDER[int(np.argmax(probs))]:
                raise ValueError(f"raw proposal mismatch at step {row['step']}")
            masked = np.where([game.legal(direction) for direction in ORDER], probs, -np.inf)
            if np.isfinite(masked).any():
                proposal = ORDER[int(np.argmax(masked))]
                if row["proposal"] != proposal or abs(row["trust_lcb"] - trust.lcb(proposal)) > 1e-5:
                    raise ValueError(f"proposal or trust mismatch at step {row['step']}")
                if header["config"]["unassisted"]:
                    expected, gate = row["raw_proposal"], "RAW HEAD · NO SHIELD"
                elif trust.lcb(proposal) < header["config"]["trust_floor"]:
                    expected, gate = game.planner_choice(), "EMA-LCB ESCALATE"
                elif not game.safety_accepts(proposal):
                    expected, gate = game.planner_choice(), "SAFETY SHIELD"
                else:
                    expected, gate = proposal, "MODEL"
                if (d, row["gate"]) != (expected, gate):
                    raise ValueError(f"gate decision mismatch at step {row['step']}")
            elif (d, row["proposal"], row["gate"]) != (game.direction, game.direction, "NO LEGAL MOVE"):
                raise ValueError(f"dead-end decision mismatch at step {row['step']}")
            if row["gate"] == "NO LEGAL MOVE":
                if any(game.legal(direction) for direction in ORDER):
                    raise ValueError(f"false dead end at step {row['step']}")
                game.alive = False
                game.deaths += 1
            else:
                safe_before = game.safety_accepts(d)
                alive = game.step(d)
                if not header["config"]["unassisted"]:
                    trust.observe(d, bool(alive and safe_before))
            if snapshot(game) != row["after"]:
                raise ValueError(f"transition mismatch at step {row['step']}")
            count += 1
    if count == 0:
        raise ValueError("recording has no moves")
    return {"verified_moves": count, "recording": str(path), "provenance": header["provenance"]}


def benchmark(args):
    scorer, backend = prepare(args)
    seeds = [int(s) for s in args.seeds.split(",")]
    rates = [int(s) for s in args.rates.split(",")]
    if not seeds or any(s < 0 for s in seeds) or not rates or any(r <= 0 for r in rates):
        raise ValueError("seeds must be nonnegative and rates positive")
    report = {"provenance": provenance(args, backend), "contract": {
        "active_tick": "features + head + gate + game update + Rich composition + truecolor ANSI serialization; excludes load, warmup, terminal painting and pacing sleep",
        "paced_pass": "at least 99% of active ticks <= 1000/fps ms for every seed",
        "sweep_steps": args.sweep_steps, "soak_steps": args.soak_steps, "uncapped_steps": args.steps,
        "seeds": seeds, "rates": rates, "mode": "raw" if args.unassisted else "shielded",
        "width": args.width, "height": args.height, "trust_floor": args.trust_floor,
        "train_seed": args.seed, "train_episodes": args.train_episodes,
        "train_epochs": args.train_epochs}, "uncapped": [], "sweep": {}, "soak": {}}
    if getattr(args, "resume", False):
        if not args.output or not Path(args.output).exists():
            raise ValueError("--resume requires an existing --output report")
        previous = json.loads(Path(args.output).read_text())
        if previous["contract"] != report["contract"] or previous["provenance"]["git_commit"] != report["provenance"]["git_commit"]:
            raise ValueError("resume contract or git revision differs")
        report = previous
    def persist():
        if args.output:
            target = Path(args.output)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".tmp")
            temporary.write_text(json.dumps(report, indent=2) + "\n")
            os.replace(temporary, target)
    def trial(seed, steps, fps=None):
        game = SnakeGame(args.width, args.height, seed)
        trust = Trust(args.trust_floor)
        intervals, render_ms, inference_ms = [], [], []
        interventions = deaths = best = illegal_raw = 0
        start = time.perf_counter()
        for i in range(steps):
            t0 = time.perf_counter()
            result = move(game, trust, scorer, shield=not args.unassisted, floor=args.trust_floor)
            inference_ms.append(result["inference_ms"])
            illegal_raw += result["illegal_raw"]
            interventions += result["intervened"]
            best = max(best, game.best)
            if not game.alive:
                deaths += 1
            frame = {"type": "move", **result, "step": i + 1, "round": deaths, "seed": seed + deaths,
                     "backend": backend, "interventions": interventions, "fps": fps or 0,
                     "rate": 0.0, "paused": False, "mode": "raw" if args.unassisted else "shielded",
                     "tick_ms": 0.0}
            render_t0 = time.perf_counter()
            composed = dashboard(frame, args.trust_floor)
            sink = io.StringIO()
            Console(file=sink, force_terminal=True, color_system="truecolor", width=MIN_WIDTH, height=MIN_HEIGHT).print(composed)
            sink.getvalue()
            render_ms.append((time.perf_counter() - render_t0) * 1000)
            active_ms = (time.perf_counter() - t0) * 1000
            intervals.append(active_ms)
            if not game.alive:
                game = SnakeGame(args.width, args.height, seed + deaths)
                trust = Trust(args.trust_floor)
            if fps:
                time.sleep(max(0.0, 1 / fps - (time.perf_counter() - t0)))
        elapsed = time.perf_counter() - start
        arr = np.asarray(intervals)
        return {"seed": seed, "steps": steps, "score": best, "deaths": deaths,
                "interventions": interventions, "illegal_raw_proposals": illegal_raw,
                "mean_inference_ms": round(float(np.mean(inference_ms)), 3),
                "mean_rich_render_ms": round(float(np.mean(render_ms)), 3),
                "p50_ms": round(float(np.percentile(arr, 50)), 3),
                "p95_ms": round(float(np.percentile(arr, 95)), 3),
                "p99_ms": round(float(np.percentile(arr, 99)), 3),
                "max_ms": round(float(arr.max()), 3),
                "deadline_fraction": round(float(np.mean(arr <= 1000 / fps)), 5) if fps else None,
                "active_moves_per_s": round(steps / max(1e-9, arr.sum() / 1000), 2),
                "wall_moves_per_s": round(steps / max(elapsed, 1e-9), 2)}
    for seed in seeds:
        if not any(row["seed"] == seed for row in report["uncapped"]):
            report["uncapped"].append(trial(seed, args.steps))
            persist()
    passing = []
    for fps in rates:
        item = report["sweep"].setdefault(str(fps), {"pass": False, "seeds": []})
        for seed in seeds:
            if not any(row["seed"] == seed for row in item["seeds"]):
                item["seeds"].append(trial(seed, args.sweep_steps, fps))
                persist()
        item["pass"] = all(row["deadline_fraction"] >= .99 for row in item["seeds"])
        persist()
        if item["pass"]:
            passing.append(fps)
    # Every passing rate gets a long run; retain failed soaks as measured failures.
    for fps in passing:
        item = report["soak"].setdefault(str(fps), {"pass": False, "seeds": []})
        for seed in seeds:
            if not any(row["seed"] == seed for row in item["seeds"]):
                item["seeds"].append(trial(seed, args.soak_steps, fps))
                persist()
        item["pass"] = all(row["deadline_fraction"] >= .99 for row in item["seeds"])
        persist()
    report["highest_sustained_fps"] = max((fps for fps in passing if report["soak"][str(fps)]["pass"]), default=None)
    report["uncapped_aggregate"] = {"total_steps": sum(r["steps"] for r in report["uncapped"]),
        "total_deaths": sum(r["deaths"] for r in report["uncapped"]),
        "total_interventions": sum(r["interventions"] for r in report["uncapped"]),
        "mean_active_moves_per_s": round(float(np.mean([r["active_moves_per_s"] for r in report["uncapped"]])), 2)}
    persist()
    print(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser(description="Vey Snake · independent feature-assisted decision demo")
    sub = ap.add_subparsers(dest="command")
    bench = sub.add_parser("benchmark", help="uncapped and paced full-tick measurement")
    replay_ap = sub.add_parser("replay", help="verify a recording by deterministic re-simulation")
    replay_ap.add_argument("recording")
    export_ap = sub.add_parser("export", help="render recording to MP4, GIF or PNG poster")
    export_ap.add_argument("recording")
    export_ap.add_argument("output")
    export_ap.add_argument("--fps", type=int, default=30)
    export_ap.add_argument("--start", type=float, default=0)
    export_ap.add_argument("--end", type=float)
    def common(p):
        p.add_argument("--width", type=int, default=24)
        p.add_argument("--height", type=int, default=16)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--fps", type=float, default=12)
        p.add_argument("--steps", type=int, default=600)
        p.add_argument("--trust-floor", type=float, default=.5)
        p.add_argument("--train-episodes", type=int, default=14)
        p.add_argument("--train-epochs", type=int, default=80)
        p.add_argument("--unassisted", action="store_true", help="raw four-way head argmax, no legality mask or trust/safety shield")
    common(ap)
    common(bench)
    ap.add_argument("--max-speed", action="store_true")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--duration", type=float, help="stop after this many wall seconds")
    ap.add_argument("--record")
    bench.add_argument("--seeds", default="101,102,103,104")
    bench.add_argument("--rates", default="10,12,15,18,20,25,30,35,40,45,50,60")
    bench.add_argument("--sweep-steps", type=int, default=120)
    bench.add_argument("--soak-steps", type=int, default=600)
    bench.add_argument("--output")
    bench.add_argument("--resume", action="store_true", help="continue a matching partial --output report")
    args = ap.parse_args()
    if args.command == "replay":
        print(json.dumps(replay(args.recording), indent=2))
    elif args.command == "export":
        from .snake_export import export_recording
        print(json.dumps(export_recording(args.recording, args.output, fps=args.fps, start=args.start, end=args.end), indent=2))
    elif args.command == "benchmark":
        benchmark(args)
    else:
        run(args)

if __name__ == "__main__":
    main()
