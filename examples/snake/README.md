# Snake demo

The Snake demo is a feature-assisted demonstration of Vey's structured decision
loop. It is not evidence that the routing head perceives raw game frames. The
head consumes engineered board features, not pixels.

```bash
pip install -e .
python -m examples.snake.snake
```

## What it shows

A small policy head scores legal moves and the runtime applies the same rules
that drive the rest of Vey:

- hard legality masks applied before scoring,
- an online reliability estimate per move,
- exact recent-history predicates,
- a deterministic no-op when trust is low, instead of a random move.

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
