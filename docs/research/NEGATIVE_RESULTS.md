# Negative results

Rejections are the record's most valuable entries. Every mechanism below was
measured, not guessed, and each closed a line of inquiry. Benchmark numbers
for each entry live in docs/BENCHMARKS.md on the same commit unless linked below.

## System One shared-state head (2026-11-07)

The frozen 354.5M d1 text trunk plus a re-trained 26.2M head is **not a safe
drop-in replacement** for joint state/question encoding. The text-only experiment
does not build a new multimodal foundation model.

| Mechanism | Result | Decision |
|---|---|---|
| Independent state/question features + adapted head (A) | Best exploratory agreement 0.642; checkpoint selection used evaluation rows | Reject for serving; exploratory result only |
| Per-layer prefix cache + adapted head (B) | SST-5 agreement 0.780 vs gate 0.950; Emotion accuracy 0.745 vs teacher 0.810 | Reject for serving despite speed gain |
| Prefix cache, six questions, 79 state tokens | 1,191 to 641 ms, 1.86x; one question 255 to 318 ms | Speed gate passes at this length, not enough to offset failed quality gates |

Variant B used 2,000 training and 400 validation rows; 200 evaluation rows per
task were not used for that run's optimization or checkpoint selection. Earlier
explorations already exposed the task suite. Calibration passed its gate, but
no reliability or 1 GiB claim follows. Full controls, environment, commands and
results: [SHARED_STATE_HEAD.md](SHARED_STATE_HEAD.md).

Two deployment limits of ordinary T2 were also measured: a 1 GiB no-swap cap
OOM-killed the server on its first image warmup, and a solid-red image was
classified as blue by int8, fp32-vision and full-fp32 controls (2/3 colours correct
in every path). Neither can be advertised as solved. See
[SYSTEMONE.md](../SYSTEMONE.md#memory-and-latency-recheck-2026-11-07).

## Semantic field architecture

| Mechanism | Result | Why it closed the line |
|---|---|---|
| Executor-shaped losses on the scalar head | -0.04 tight-band | The loss damages the competent field it tunes |
| Deeper encoder (8, 12 layers) | Non-inferior at best | Boundary information is not a depth problem |
| Per-layer probes on the frozen encoder | No boundary signal anywhere | Not linearly readable from stock states |
| Set context + gauge projection (Flux) | Loses 0.02-0.08 to scalar | Multi-term objective oscillates; harder to optimize |
| Candidate cross-attention residual | +0.013, spread 0.05 | Indistinguishable from seed noise |
| Slot bottlenecks (R=4, R=8) | Match cross-attention within 0.002 | Compression adds nothing over K x K |
| Sequential schedule (residual, then executor) | -0.0133 | The promoted gain belonged to the joint schedule |
| Residual deleted at inference (DROP) | +0.029 vs +0.073 | Residual is load-bearing, not scaffolding |
| The ambiguity gate | 0.002-0.005 difference | Inert within noise; ships ungated |
| Learned residual representations (MLP) | Ties a frozen random projection | The pathway matters, not the capacity |
| Local frontier warping | near < far movement on every joint arm | The successful arms move the whole field |

## Decision transfer (D1/D2 program, prior line)

| Mechanism | Result |
|---|---|
| D1-A frozen encoder + head | 38.8% vs 25.3% chance: representation lacks decision info |
| D1-B full fine-tune | 86% generic, 0 food / 14.3% planner agreement on Snake |
| D1-C 1.18M decision layer | Worse on every metric, negative ablation |
| Battery v1 as validation | Failed: Laya 28.4% / D1-B 26.2%, no separation |

The battery failure produced the strategic finding: Laya's transfer is
phrasing-locked and asymmetric (maximize 52%, minimize 4%), so operator-level
gates are a higher bar than agreement alone.

## Training and evaluation discipline

These cost GPU runs and produced retractions, so they are recorded as
findings too:

- A frozen-encoder training run completed with plausible losses and measured
  a frozen readout while claiming end-to-end training. Two runs were
  invalidated before the cause was found: activations under no_grad give the
  optimizer None gradients, which AdamW silently skips.
- Unseeded retrains produced baselines ranging 0.6885 to 0.8142 in one
  session, comparable to the effect sizes under interpretation. Every run
  since pins its seed.
- An agreement metric that returned 0.0 for every arm while the other
  metrics moved was missing its numerator line, not its band. The band had
  183 qualifying pairs.
- A zero-init residual's equality check at initialization passes vacuously
  under a shape broadcast; shape assertions belong in the architecture
  contract.
