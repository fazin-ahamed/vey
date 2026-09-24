"""Vey Snake terminal presentation, visually adapted from laya-mlx's Snake demo.

Design reference: https://github.com/mizorewww/laya-mlx (Apache-2.0).
This is a fresh implementation over Vey's own recorded decision fields.
"""
from __future__ import annotations

from rich.text import Text

BG = "#090f13"
FG = "#e3f3ef"
MUTED = "#68868c"
DIM = "#20353c"
GREEN = "#62f5b5"
AMBER = "#ffce73"
RED = "#ff7c8c"
CYAN = "#8ad8e9"
MIN_WIDTH, MIN_HEIGHT = 104, 35

# Explicit 3 × 5 bitmaps. Paired pixel rows pack into half-block terminal
# cells, preserving glyph geometry at three terminal rows of height.
DIGITS = {
    "0": ("███", "█ █", "█ █", "█ █", "███"),
    "1": ("██ ", " █ ", " █ ", " █ ", "███"),
    "2": ("███", "  █", "███", "█  ", "███"),
    "3": ("███", "  █", "███", "  █", "███"),
    "4": ("█ █", "█ █", "███", "  █", "  █"),
    "5": ("███", "█  ", "███", "  █", "███"),
    "6": ("███", "█  ", "███", "█ █", "███"),
    "7": ("███", "  █", "  █", "  █", "  █"),
    "8": ("███", "█ █", "███", "█ █", "███"),
    "9": ("███", "█ █", "███", "  █", "███"),
}
_HALF_BLOCK = {("█", "█"): "█", ("█", " "): "▀",
               (" ", "█"): "▄", (" ", " "): " "}
DISPLAY_DIGITS = {
    digit: tuple(
        "".join(_HALF_BLOCK[top, bottom] for top, bottom in zip(rows[row], rows[row + 1]))
        for row in (0, 2, 4)
    )
    for digit, rows in ((digit, pixels + ("   ",)) for digit, pixels in DIGITS.items())
}


def layout_size(board_width: int, board_height: int):
    return max(MIN_WIDTH, board_width * 2 + 50), max(MIN_HEIGHT, board_height + 19)


class Grid:
    """A bounded, style-per-cell terminal surface reusable by the video exporter."""
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.cells = [[" "] * width for _ in range(height)]
        self.colors = [[FG] * width for _ in range(height)]

    def put(self, y, x, text, color=FG):
        if not 0 <= y < self.height:
            return
        for i, ch in enumerate(str(text)):
            if 0 <= x + i < self.width:
                self.cells[y][x + i] = ch
                self.colors[y][x + i] = color

    def line(self, y, x, amount, color=DIM):
        self.put(y, x, "─" * max(0, amount), color)

    def meter(self, y, x, fraction, length, color=GREEN):
        n = max(0, min(length, round(fraction * length)))
        self.put(y, x, "━" * length, DIM)
        self.put(y, x, "━" * n, color)

    def big_number(self, y, x, value, color):
        for index, digit in enumerate(f"{min(999, max(0, value)):03d}"):
            for row, pixels in enumerate(DISPLAY_DIGITS[digit]):
                self.put(y + row, x + index * 4, pixels, color)

    def rich_text(self):
        text = Text(no_wrap=True, overflow="crop", style=f"{FG} on {BG}")
        for row in range(self.height):
            start = 0
            for x in range(1, self.width + 1):
                if x == self.width or self.colors[row][x] != self.colors[row][start]:
                    text.append("".join(self.cells[row][start:x]), style=self.colors[row][start])
                    start = x
            if row + 1 != self.height:
                text.append("\n")
        return text


def compose(frame: dict, trust_floor: float = .5):
    """Draw the pre-action board, which is the board the four scores describe."""
    b = frame["before"]
    w, h = int(b["width"]), int(b["height"])
    if not (6 <= w <= 70 and 6 <= h <= 26):
        raise ValueError("board outside presentation bounds (6–70 × 6–26)")
    snake = [tuple(p) for p in b["snake"]]
    food = tuple(b["food"]) if b["food"] is not None else None
    if not snake or any(not (0 <= x < w and 0 <= y < h) for x, y in snake + ([food] if food else [])):
        raise ValueError("board contains an out-of-bounds cell")
    width, height = layout_size(w, h)
    g = Grid(width, height)
    left, right, top = 3, max(58, w * 2 + 10), 6
    bottom = top + h + 1
    state = "PAUSED" if frame["paused"] else "RECORDED RUN · 1×" if frame.get("replay") else "LIVE"
    g.put(1, left, "VEY  /  LOCAL DECISIONS", MUTED)
    g.put(1, width - 3 - len(state), state, GREEN)
    g.line(2, left, width - 6)
    g.put(4, left, "S N A K E", FG)
    g.put(4, left + 31, f"ROUND {frame['round'] + 1:02d}", MUTED)
    g.put(top, left, "┌" + "─" * (w * 2) + "┐", DIM)
    g.put(bottom, left, "└" + "─" * (w * 2) + "┘", DIM)
    for y in range(h):
        g.put(top + 1 + y, left, "│", DIM)
        g.put(top + 1 + y, left + w * 2 + 1, "│", DIM)
        g.put(top + 1 + y, left + 1, "· " * w, "#13272e")
    for i, (x, y) in reversed(list(enumerate(snake))):
        shade = 1 - i / max(1, len(snake))
        green = f"#{int(18 + 64 * shade):02x}{int(73 + 150 * shade):02x}{int(57 + 102 * shade):02x}"
        g.put(top + y + 1, left + 1 + 2 * x, "██", "#dcfff0" if i == 0 else green)
    if food is not None:
        g.put(top + food[1] + 1, left + 1 + 2 * food[0], "● ", AMBER)
    for offset, label, number, color in (
        (0, "SCORE", b["score"], GREEN),
        (16, "LENGTH", len(snake), FG),
        (32, "BEST", frame.get("best_score", b["score"]), MUTED),
    ):
        metric_x = left + offset
        g.put(bottom + 2, metric_x + (16 - len(label)) // 2, label, MUTED)
        g.big_number(bottom + 3, metric_x + (16 - 11) // 2, number, color)
    fill = len(snake) / (w * h)
    g.meter(bottom + 7, left, fill, 41)
    g.put(bottom + 7, left + 43, f"{fill * 100:4.1f}%", MUTED)

    g.put(4, right, "Vey R0.5 · demo head", GREEN)
    g.put(5, right, f"{frame['backend']} · local", MUTED)
    g.put(7, right, "NEXT MOVE", FG)
    g.put(7, right + 15, "HEAD PROBABILITIES", MUTED)
    for i, direction in enumerate(("UP", "DOWN", "LEFT", "RIGHT")):
        p = float(frame["probabilities"][direction])
        selected = direction == frame["proposal"]
        color = GREEN if selected else MUTED
        g.put(9 + i, right, f"{'›' if selected else ' '} {direction:<5}", color)
        g.put(9 + i, right + 9, "░" * 18, DIM)
        g.put(9 + i, right + 9, "█" * max(0, min(18, round(p * 18))), color)
        g.put(9 + i, right + 29, f"{p:.2f}", color)
    g.put(14, right, "EXECUTING", MUTED)
    g.put(14, right + 12, frame["executed"], GREEN)
    if frame.get("intervened"):
        g.put(14, right + 23, "SHIELD", AMBER)
    a = frame["analysis"][frame["proposal"]]
    risk = 1 - a["area"] if a["legal"] else 1
    reachable = float(a["food_reachable_est"])
    g.put(16, right, "DEAD-END EST.", MUTED)
    g.meter(17, right, risk, 24, AMBER if risk < .5 else RED)
    g.put(17, right + 29, f"{risk:.2f}", AMBER if risk < .5 else RED)
    g.put(19, right, "FOOD REACH EST.", MUTED)
    g.meter(20, right, reachable, 24, CYAN)
    g.put(20, right + 29, f"{reachable:.2f}", CYAN)
    for y, label, value, color in (
        (22, "INFERENCE", f"{frame['inference_ms']:5.1f} ms", FG),
        (23, "DECISIONS", f"{frame['rate']:5.1f} /s", FG),
        (24, "TRUST / FLOOR", f"{frame['trust_lcb']:.2f} / {trust_floor:.2f}", FG),
        (25, "ENGINE", frame["backend"], MUTED),
        (26, "TARGET", f"{frame['fps']:.0f} fps", MUTED),
    ):
        g.put(y, right, label, MUTED)
        g.put(y, right + 18, value, color)
    g.put(28, right, "Vey + trust / structural shield" if frame["mode"] != "raw" else "Vey · shield OFF", MUTED)
    g.put(29, right, f"Gate  {frame['gate']}", AMBER)
    g.put(30, right, f"Shield interventions  {frame['interventions']:04d}", MUTED)
    g.line(height - 3, left, width - 6)
    g.put(height - 2, left, "SPACE pause   ↑/↓ speed   R reset   Q quit", MUTED)
    clock = f"{int(frame.get('elapsed_s', 0)) // 60:02d}:{int(frame.get('elapsed_s', 0)) % 60:02d}"
    g.put(height - 2, right, f"STRUCTURAL ESTIMATES       {clock}", MUTED)
    return g


def dashboard(frame: dict, trust_floor: float = .5, *, width=104, height=35):
    required_width, required_height = layout_size(frame["before"]["width"], frame["before"]["height"])
    if width < required_width or height < required_height:
        return Text(f"Resize terminal to at least {required_width} columns × {required_height} rows.\n"
                    "The game is waiting. Q quits.", style=f"{AMBER} on {BG}")
    return compose(frame, trust_floor).rich_text()
