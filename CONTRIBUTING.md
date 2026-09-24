# Contributing to Vey

Contributions are welcome. Bug fixes, documentation, benchmark reproductions,
runtime optimizations, new exact mechanisms, and well-tested integrations are
especially useful.

## Development setup

```bash
git clone https://github.com/fazin-ahamed/vey.git
cd vey
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Supported Python versions: **3.10, 3.11, 3.12**. CI runs all three.

Run the tests:

```bash
pytest -q
```

Smoke-test the Snake demo (records moves and replays them deterministically):

```bash
python -m examples.snake.snake --headless --steps 40 --record /tmp/snake.jsonl
python -m examples.snake.snake replay /tmp/snake.jsonl
```

## Style

Follow the surrounding code. Keep docstrings concrete: state what the function
guarantees, not how clever it is. Avoid unexplained jargon and internal
project vocabulary. Prefer plain punctuation (commas, colons, parentheses) over
decorative dashes.

## What a pull request should include

- Tests for observable behavior. A test that asserts an implementation detail
  rather than a contract will be asked to change.
- A short note on *why*, when the reason is not obvious from the diff.

## Benchmark discipline

This is the part contributors most often get wrong.

- **Raw evidence is required.** Every benchmark number in the docs must be
  reproducible from a committed script and an environment description. Attach
  the command, the hardware, and the result.
- **Do not change a benchmark claim without a new measurement.** If a refactor
  changes a number, say so explicitly and show the before/after on the same
  hardware.
- **Latency is hardware-specific.** Never compare a number measured on one host
  to one measured on another. Report the CPU, thread count, and batch size with
  every timing.
- **New mechanisms need a simpler control.** Before proposing a new component,
  show that the simplest approach does not already do the job. Several plausible
  mechanisms have already been measured and rejected for exactly this reason;
  see [docs/LIMITATIONS.md](docs/LIMITATIONS.md).
- **Ablations must isolate one variable.** If a result depends on a change to
  data, report the data change separately from the code change.

## Adding mechanisms

New capability should enter through a lane interface (`structured`, `semantic`,
`retrieval`, `act`, `trust`), not as a special case in the central dispatcher.
The public API is stable; lanes are the extension point. See
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

## Repository hygiene

- **No downloaded datasets or model weights in Git.** Benchmarks fetch upstream
  data at run time under its own license.
- **No secrets, tokens, credentials, or local filesystem paths.** Not in code,
  comments, tests, fixtures, or committed result files.
- **Apache-2.0 compatible contributions only.** By submitting, you certify that
  you have the right to submit the code under that license.

## Reporting

Bugs and features go to GitHub Issues. Security-sensitive reports go through
[SECURITY.md](SECURITY.md), not a public issue.
