# STEF: how the technique was found

The full commit-by-commit experimental record lives on the
`research-history` branch and in the pre-rewrite bundle. This document is
the readable version: what was tried, what failed, and what survived.

## The problem

The scalar decision field orders candidates almost perfectly (rank
correlation with the teacher above 0.99 everywhere) but places them badly
around the executor's decision boundary, the band within 0.02 of the
executor's resolution epsilon of 0.15. Every disagreement that matters lives
in that band. A sequence of experiments attacked it.

## What did not work

Each of these was measured and rejected, in order:

- **Better losses.** Executor-shaped losses applied directly to the scalar
  head cost about 0.04 of tight-band accuracy. Applied jointly with a
  residual they helped, but the effect turned out to belong to the
  optimization schedule, not the loss (see below).
- **Better readout.** Multi-layer probes on every layer of the frozen
  encoder found boundary information nowhere in the stock states.
- **More depth.** Eight and six layers were non-inferior to twelve on every
  axis; four lost food progress.
- **Relational candidate reasoning.** Set-context and gauge-projection
  machinery lost to the plain scalar head, and the losses oscillated where
  the baseline converged.
- **A boundary residual with slots or cross-attention.** The residual alone
  gained +0.013 with a seed spread of about 0.05, indistinguishable from
  noise. Slot bottlenecks added nothing over candidate cross-attention.
- **The sequential training schedule.** Training the residual to convergence
  first and applying the executor loss second reversed the gain to -0.013.
  This retraction withdrew the first promotion.
- **Deleting the residual at inference.** The gain collapsed from +0.073 to
  +0.029, below what the bare executor loss achieves. The residual must
  ship.
- **The ambiguity gate.** Gated and ungated residuals differ by 0.002 to
  0.005, inside seed noise. The gate is inert; the simpler ungated form is
  what ships.
- **Residual function class.** A linear residual, a frozen random
  projection, and a full MLP all gained exactly +0.0733. The residual
  contributes an optimization pathway, not representational capacity.

## What survived

**Split-Train Exact-Fold (STEF).** Train a zero-initialized linear
auxiliary path alongside the head under the residual-plus-executor loss,
with the head moving at a small learning rate. Then fold the auxiliary
exactly into the head.

The fold is algebra: GELU(z) - GELU(-z) = z, so a linear residual
a-transpose-h plus beta is reproduced by two hidden units with antipodal
weights (a, beta) and (-a, -beta) and output weights +1 and -1. The
deployed model is a stock head two neurons wider. No distillation, no
approximation, no runtime residual branch.

The mechanism is an optimization-pathway effect, not capacity: the residual
carries 11 to 20 percent of the learned functional change while
contributing about 3 percent of the output magnitude, and a frozen random
projection works as well as a learned MLP.

## Validation chain

1. Pair level, food progress: +0.039, 95% CI [0.0163, 0.0619] on 1,386
   untouched tight-band pairs.
2. Axis transfer, frozen recipe: open space +0.053 CI-positive; headroom
   +0.028, positive but underpowered at 434 pairs.
3. Shared three-axis field: all five preregistered gates pass; the three
   auxiliary vectors are near-orthogonal, so each axis folds into its own
   386-wide head.
4. Program level, real executor: full trace agreement 0.8359 to 0.8744,
   delta +0.0385, CI [0.0226, 0.0549]. Of 320 baseline trace failures, 168
   repaired (52.5 percent); of 1,630 correct traces, 93 broken (5.7
   percent).
5. Deployment: head cost about 2 microseconds at p50 and 772 parameters;
   full-field latency non-inferior; calibration radius halved, 0.1814 to
   0.0913.

## Known open items

- The operational program-certificate rate (best-versus-dropped plus final
  tie-break) is not yet computed; the recorded 0.5405 to 0.6200 rates use a
  survivor-pair definition and are labeled as such.
- Seed 7 is the only trained seed; seeds 11 and 13 are pending.
- Snake integration with trace agreement as the primary metric is pending.
