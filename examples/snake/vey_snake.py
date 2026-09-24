"""Vey Snake: feature-assisted terminal demo (game rules and demo head).

This is an independently implemented showcase. It is NOT a benchmark of the
production routing task and it is not evidence that the routing SuccessHead
understands raw Snake boards.

A tiny demo-specific SuccessHead (the same 9 -> 32 -> 1 topology used by R0.5)
is distilled at startup from deterministic planner features. Each move then
uses:
    structured board/action features -> SuccessHead -> hard legality mask
    -> EMA-LCB trust gate -> optional safety fallback -> execute

The runtime, recorder, replay verifier and benchmark live in `snake_runtime`;
the reference-style presentation in `snake_tui`; the video/GIF/poster exporter
in `snake_export`. Run it through this entry point:
    python examples/vey_snake.py
    python examples/vey_snake.py --unassisted --headless --steps 1000 --max-speed
    python examples/vey_snake.py --record artifacts/run.jsonl
    python examples/vey_snake.py replay artifacts/run.jsonl
    python examples/vey_snake.py export artifacts/run.jsonl artifacts/run.mp4
    python examples/vey_snake.py benchmark --output artifacts/benchmark.json
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vey.structured import SuccessHead

DIRS = {
    "UP": (0, -1),
    "RIGHT": (1, 0),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
}
ORDER = tuple(DIRS)
OPPOSITE = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}


@dataclass
class Analysis:
    legal: bool
    area: float = 0.0
    mobility: float = 0.0
    food_progress: float = 0.0
    eats: float = 0.0
    cycle_match: float = 0.0
    continuing: float = 0.0
    wall_clearance: float = 0.0

    def row8(self) -> list[float]:
        return [
            float(self.legal),
            self.area,
            self.mobility,
            self.food_progress,
            self.eats,
            self.cycle_match,
            self.continuing,
            self.wall_clearance,
        ]


class SnakeGame:
    def __init__(self, width: int = 24, height: int = 16, seed: int = 0,
                 initial_length: int = 6):
        if width < 6 or height < 6:
            raise ValueError("board must be at least 6x6")
        self.width = width
        self.height = height
        self.initial_length = min(initial_length, width - 2)
        self.rng = random.Random(seed)
        self.seed = seed
        self.best = 0
        self.deaths = 0
        self.reset()

    def reset(self):
        y = self.height // 2
        x = max(self.initial_length + 1, self.width // 3)
        self.snake = [(x - i, y) for i in range(self.initial_length)]
        self.direction = "RIGHT"
        self.score = 0
        self.alive = True
        self.steps = 0
        self.food = self._spawn_food()

    @property
    def head(self):
        return self.snake[0]

    def _spawn_food(self):
        occupied = set(self.snake)
        free = [(x, y) for y in range(self.height) for x in range(self.width)
                if (x, y) not in occupied]
        return self.rng.choice(free) if free else None

    def _inside(self, p):
        return 0 <= p[0] < self.width and 0 <= p[1] < self.height

    def _next(self, direction: str):
        dx, dy = DIRS[direction]
        return self.head[0] + dx, self.head[1] + dy

    def legal(self, direction: str) -> bool:
        nxt = self._next(direction)
        if not self._inside(nxt):
            return False
        grows = nxt == self.food
        blocked = set(self.snake if grows else self.snake[:-1])
        return nxt not in blocked

    def _sim_body(self, direction: str):
        nxt = self._next(direction)
        grows = nxt == self.food
        body = [nxt] + list(self.snake)
        if not grows:
            body.pop()
        return body, grows

    def _flood_area(self, start, blocked) -> int:
        if start in blocked:
            return 0
        q = [start]
        seen = {start}
        while q:
            x, y = q.pop()
            for dx, dy in DIRS.values():
                p = (x + dx, y + dy)
                if self._inside(p) and p not in blocked and p not in seen:
                    seen.add(p)
                    q.append(p)
        return len(seen)

    def _cycle_successor(self):
        """A deterministic reference successor for the default even-height board.

        It is only a planner feature/fallback hint, not a proof of safety after
        arbitrary deviations from the cycle.
        """
        x, y = self.head
        h, w = self.height, self.width
        if h % 2:
            return None

        if y == 0:
            if x < w - 1:
                return "RIGHT"
            return "DOWN"

        if x == 0:
            return "UP"

        if y % 2 == 1:
            if x > 1:
                return "LEFT"
            if y == h - 1:
                return "LEFT"
            return "DOWN"

        if x < w - 1:
            return "RIGHT"
        return "DOWN"

    def analyze(self, direction: str) -> Analysis:
        if not self.legal(direction):
            return Analysis(False)

        body, grows = self._sim_body(direction)
        new_head = body[0]
        blocked = set(body[1:])
        reachable = self._flood_area(new_head, blocked)
        free_total = max(1, self.width * self.height - len(blocked))
        area = reachable / free_total

        mobility = 0
        for dx, dy in DIRS.values():
            p = (new_head[0] + dx, new_head[1] + dy)
            if self._inside(p) and p not in blocked:
                mobility += 1

        if self.food is None:
            progress = 0.0
        else:
            old_d = abs(self.head[0] - self.food[0]) + abs(self.head[1] - self.food[1])
            new_d = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
            progress = (old_d - new_d) / max(self.width, self.height)

        wall = min(
            new_head[0],
            self.width - 1 - new_head[0],
            new_head[1],
            self.height - 1 - new_head[1],
        ) / max(1, min(self.width, self.height) / 2)

        return Analysis(
            True,
            area=float(area),
            mobility=mobility / 4.0,
            food_progress=float(progress),
            eats=float(grows),
            cycle_match=float(direction == self._cycle_successor()),
            continuing=float(direction == self.direction),
            wall_clearance=float(max(0.0, min(1.0, wall))),
        )

    def planner_target(self, direction: str) -> float:
        a = self.analyze(direction)
        if not a.legal:
            return 0.0
        # Soft pseudo-label used only to distill this demo head.
        value = (
            0.04
            + 0.48 * a.area
            + 0.18 * a.mobility
            + 0.12 * max(a.food_progress, 0.0) * max(self.width, self.height)
            + 0.08 * a.eats
            + 0.06 * a.cycle_match
            + 0.02 * a.continuing
            + 0.02 * a.wall_clearance
        )
        return float(np.clip(value, 0.01, 0.99))

    def planner_choice(self) -> str:
        legal = [d for d in ORDER if self.legal(d)]
        if not legal:
            return self.direction
        return max(
            legal,
            key=lambda d: (
                self.analyze(d).area,
                self.analyze(d).eats,
                self.analyze(d).food_progress,
                self.analyze(d).mobility,
                self.analyze(d).cycle_match,
            ),
        )

    def safety_accepts(self, direction: str) -> bool:
        a = self.analyze(direction)
        return a.legal and (a.area >= 0.35 or a.cycle_match > 0.5)

    def features(self, direction: str, ema: float = 0.8) -> list[float]:
        return self.analyze(direction).row8() + [float(ema)]

    def step(self, direction: str) -> bool:
        if not self.legal(direction):
            self.alive = False
            self.deaths += 1
            return False

        nxt = self._next(direction)
        grows = nxt == self.food
        self.snake.insert(0, nxt)
        if grows:
            self.score += 1
            self.best = max(self.best, self.score)
            self.food = self._spawn_food()
        else:
            self.snake.pop()

        self.direction = direction
        self.steps += 1
        return True


def build_demo_training(seed: int, episodes: int = 14, steps: int = 100):
    xs, ys = [], []
    for ep in range(episodes):
        g = SnakeGame(seed=seed + 1000 + ep)
        for _ in range(steps):
            for d in ORDER:
                xs.append(g.features(d, 0.8))
                ys.append(g.planner_target(d))
            d = g.planner_choice()
            if not g.step(d):
                break
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32)


def train_demo_head(seed: int = 0, episodes: int = 14, epochs: int = 80) -> SuccessHead:
    """Distill deterministic planner features into the R0.5 head topology."""
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    x, y = build_demo_training(seed, episodes=episodes)
    head = SuccessHead().cpu()
    opt = torch.optim.AdamW(head.parameters(), lr=5e-3)
    xt = torch.tensor(x)
    yt = torch.tensor(y)
    for _ in range(epochs):
        pred = head(xt)
        loss = F.binary_cross_entropy(pred, yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    head.eval()
    return head


class TorchScorer:
    def __init__(self, head: SuccessHead):
        self.head = head

    def score_rows(self, rows):
        with torch.no_grad():
            return self.head(torch.tensor(np.asarray(rows), dtype=torch.float32)).numpy()


class Trust:
    def __init__(self, floor: float = 0.5):
        self.floor = floor
        self.ema: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def lcb(self, action: str) -> float:
        e = self.ema.get(action, 0.8)
        n = self.counts.get(action, 0)
        return e - 1.28 * math.sqrt(max(e * (1.0 - e), 0.0) / (n + 1))

    def observe(self, action: str, success: bool):
        self.ema[action] = 0.9 * self.ema.get(action, 0.8) + 0.1 * float(success)
        self.counts[action] = self.counts.get(action, 0) + 1


if __name__ == "__main__":
    from .snake import main
    main()
