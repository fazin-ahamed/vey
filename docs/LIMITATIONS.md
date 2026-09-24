# Limitations

## What Vey does not do

- **Not a language model.** Vey produces typed decisions, not text. It is not a
  drop-in replacement for an LLM on open-ended tasks.
- **English only in v1.**
- **Does not execute tools.** It compiles a validated decision object; a caller
  dispatches it.
- **Not calibrated to a probability distribution** in the semantic lane. The
  matcher returns similarity scores; only the pooled head and the trust layer
  produce probabilities, and the trust layer's calibration is imperfect.

## Measured but not retained

Each of these was implemented, measured, and removed because the measurement did
not justify keeping it. They are listed so the design decisions are auditable.

| mechanism | what was measured | outcome |
|---|---|---|
| **Evidence Pages** | a learned per-event evidence representation vs a simpler global one | decision quality was matched; removed as unjustified complexity |
| **Cross reranker** | neural reranking over the top-k lexical candidates | did not improve enough at the tested budget to justify its cost |
| **Frozen span head** | start/end logits over frozen 24M token states, trained only on genuinely extractable values | failed the extraction gate (precision-when-acted 0.000); the exact/regex control was kept |
| **ModernBERT runtime** | a 149M encoder as the fast path | 4–8× slower than the 24M encoder on the target CPU with no accuracy gain |
| **Learned token pruning** | learned selection of encoder input | unnecessary once BM25 sentence pre-selection proved sufficient |
| **DeepSets / set transformer / candidate self-attention** | set-aware architectures for candidate scoring | unnecessary: independent cached scoring already gives exactly 0.0 score drift under permutation/insertion/deletion |
| **Generative tool decoder** | emitting tool-call JSON | unnecessary: the exact compiler cannot emit an invalid or unsupported call |
| **Distillation / elastic supernet / Value-of-Compute / continual learning** | larger or adaptive training regimes | no measured need at the v1 scale; deferred |

## Known open weaknesses

- **Low-budget OOD detection.** OOD recall at a 10% false-UNKNOWN budget is
  ~0.065–0.12. Risk *ranking* is good (AUROC 0.72–0.81); *detection* at a tight
  abstention budget is not.
- **Conservative tool completion.** Free-text slots resolve only on an explicit
  cue, so end-to-end exact tool calls are ~0.09–0.11. The system prefers to ask.
- **The structured lane is not a language model.** It consumes typed input.

## Reproducing the "not retained" results

The full development history, including every implementation and raw result
behind the table above, is kept in a separate private development repository.
This public repository ships the supported runtime and its benchmarks only.
