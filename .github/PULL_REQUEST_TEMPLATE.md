## What this changes

<!-- One or two sentences. What is different after this PR? -->

## Why

<!-- The reason. If a benchmark number changes, say so and include the before/after
     on the same hardware, with the exact command. -->

## Testing

- [ ] `pytest -q` passes
- [ ] New behavior is covered by a test that asserts an observable contract
- [ ] `python -m examples.snake.snake` still runs (if runtime was touched)

## Checklist

- [ ] No secrets, tokens, local paths, or private data anywhere in the diff
- [ ] No downloaded datasets or model weights committed
- [ ] Contribution is Apache-2.0 compatible
- [ ] New mechanisms are compared against a simpler control, with evidence
- [ ] Public API stayed stable, or the break is intentional and documented
