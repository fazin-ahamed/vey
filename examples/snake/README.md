# Snake demo

The Snake demo drives Vey's decision loop with the Vey 2 CRUX lane. Each move
is described by its consequences (legality, reachable area, food progress,
mobility, wall clearance) and the trained ordinal comparator ranks the four
moves. It is not evidence that any model perceives raw game frames: the lane
consumes consequence text derived from the board, not pixels.

```bash
pip install -e .
python -m examples.snake.snake
```

## What it shows

The CRUX lane ranks the four moves and the runtime applies the same rules that
drive the rest of Vey:

- a hard legality mask applied before the chosen move executes,
- an online reliability estimate per move,
- a deterministic fallback to the planner when trust is low or the proposed move
  fails the safety check, instead of a random move.

Pass `--legacy-head` to use the older distilled R0.5 SuccessHead instead.

The recorded runs (seed, pacing, and deadline criteria) are reproducible, so a
campaign can be replayed move-for-move.

## Files

| file | purpose |
|---|---|
| `snake.py` | game loop, move generation, recording and replay |
| `snake_tui.py` | terminal rendering |
| `vey_snake.py` | entry point that wires the decision head to the loop |
| `snake_export.py` | export helpers for recorded campaigns |

## Caveats

- The head is not a general game-playing agent; it demonstrates the decision
  loop, not learned perception.
- Latency figures from this demo are host-specific and are not comparable to
  numbers measured on different hardware.
