# Benchmarks

Every number below was measured on the hardware stated with it. No number is
carried across incompatible hardware or incomparable workloads. Raw result
artifacts live under `benchmarks/results/`.

## Definitions used throughout

- **End-to-end semantic decision**: tokenize the state, run the 24M encoder on
  the query, and score against **cached** candidate embeddings. Caching the
  candidate side is the deployed configuration: candidate representations are
  built once at install time.
- **Scorer-only**: dot products against an already-encoded query vector. A
  separate characterization, not the headline, and not a parity claim.
- **Memory**: reported three ways because they answer different questions:
  model weight payload (comparable across systems), peak process RSS (dominated
  by the language runtime), and RSS delta after model load.

## Closed-set intent (Banking77 / CLINC150 / MASSIVE en-US)

Measured on a deterministically stratified export so the training split covers
every class. Frozen 24M encoder; a trained pooled head for the fixed label set.

| task | pooled head accuracy |
|---|---|
| Banking77 | 0.884 |
| CLINC150 | 0.832 |
| MASSIVE | 0.780 |

## Open-candidate matching (same rows, full candidate set)

The frozen matcher, scored by cosine over cached candidate embeddings.

| task | accuracy | chance |
|---|---|---|
| Banking77 (77 candidates) | 0.658 | ~0.24 |
| CLINC150 (151) | 0.675 | ~0.29 |
| MASSIVE (60) | 0.471 | ~0.24 |

The matcher is more accurate than the pooled head on unseen *classes* (it is not
trained on any of them) and less accurate on trained ones, hence two lanes.

## CPU latency (Intel i5-12400F, 6 torch threads, batch=1)

| path | p50 | p95 | p99 |
|---|---:|---:|---:|
| semantic decision, end-to-end | 2.6 – 2.9 ms | 3.2 – 3.9 ms | 3.4 – 4.2 ms |
| semantic decision, scorer-only | 0.030 to 0.070 ms | n/a | n/a |
| long-context decision (after BM25 sentence selection) | n/a | 9 to 11 ms | n/a |

Memory: 24.1M parameters, **92.0 MB FP32** weight payload, ~825 MB peak process
RSS.

## Candidate structural invariance

Permuting, inserting, or removing candidates leaves every existing candidate's
score **exactly** unchanged (drift 0.0) and the winner unchanged (agreement
1.000). This is why no set-architecture is required.

## Tool selection (ToolACE, Apache-2.0)

| split | semantic matcher accuracy | chance |
|---|---|---|
| known tools | 0.669 | 0.239 |
| novel tools (held-out identities) | 0.828 | 0.291 |

The candidate-compiler path: unsupported-argument rate is exactly **0.000** (no
emitted argument lacks an evidence span), schema validity and type validity are
**1.000**, and end-to-end exact tool calls are ~0.09–0.11 (the system prefers to
ask over guessing). See [LIMITATIONS.md](LIMITATIONS.md).

## Reproducing

Benchmarks retrieve their upstream datasets at run time; no dataset is bundled.
See `benchmarks/reproduce/` for the scripts and their expected environment.

## Vey 2 / CRUX (immutable, preregistered, run-once)

Preregistered zero-shot Snake control benchmarks. Numbers are **immutable**:
the values measured at the time of each run, not retro-edited. Thresholds were
frozen before scoring; each seed block was consumed exactly once.

### Planner-assisted variant (options carry planner verdicts), seeds 711–714

| system | params | food | deaths | planner agreement | permutation |
|---|---:|---:|---:|---:|---:|
| **CRUX** | ~150M | **136** | **0** | **1.000** | **1.000** |
| Laya | 421M | 81 | 1 | 0.910 | 0.880 |

### Raw-consequence variant (NO planner verdicts; model must derive utility), seeds 611–614

| system | params | food | deaths | agreement | permutation (at measurement) |
|---|---:|---:|---:|---:|---:|
| **CRUX** | ~150M | **155** | **0** | **0.971** | 0.965 |
| Laya | 421M | 17 | 0 | 0.404 | 0.433 |

The raw-consequence gap (155 vs 17 food) is the load-bearing result: CRUX
reconstructs the decision from grounded consequences rather than reading verdict
labels, at a ~150M frozen backbone versus a 421M decision model.

### Permutation invariance

The raw-consequence run measured **0.965** permutation agreement at the time.
A later change removed the candidate-order/name dependence **architecturally**
(identity-free grounding + content-deterministic tie-break + value
quantization), reaching **1.000 order- and rename-invariance on 1,500 generic
held-out decisions**. Snake was **not** rerun after this change; the historical
0.965 number above is left unchanged. Composition accuracy on those held-out
decisions moved 0.951 → **0.914**, a disclosed tradeoff: removing the
illegitimate candidate-name signal costs ~4 points of composition accuracy in
exchange for exact invariance, still above the ≥0.90 preregistered gate.

Not claimed: a universal "Vey > Laya". These are specific preregistered control
benchmarks. Snake is a closed benchmark family (all seed blocks consumed);
future capability selection uses generic decision suites, not Snake.

## CRUX runtime characterization (CPU, i5-12400F, 6 torch threads)

Measured on the frozen `v2.0.0-rc1` comparator with no changes to the model.
Warm numbers are the median of repeated calls after warmup on one loaded
`Runtime`, so model load is excluded. The pairwise ordinal comparison encodes
`K*(K-1)` sequences per decision, so latency grows roughly with `K²`.

| | |
|---|---|
| model load | 6.1 s (first call only) |
| artifact on disk | 598 MB (`comparator.safetensors`, fp32) |
| RSS before / after load | 375 MB / 1,289 MB (delta 914 MB) |
| peak process RSS | 1,980 MB |
| warm decision, K=4, p50 / p95 / p99 | 262 ms / 434 ms / 435 ms |

### Candidate scaling (one ordinal axis)

| candidates K | pairwise comparisons | warm p50 | warm p95 |
|---:|---:|---:|---:|
| 2 | 2 | 59 ms | 88 ms |
| 4 | 12 | 260 ms | 433 ms |
| 8 | 56 | 1,228 ms | 1,454 ms |
| 16 | 240 | 5,494 ms | 6,133 ms |
| 32 | 992 | 23,514 ms | 26,202 ms |

The growth is close to quadratic: K=32 costs about 19× K=8 while doing 18× the
comparisons. For a large candidate set (a router over many models, a tool
library) the cost is dominated by the pairwise stage, which is the point at
which a cheap preselection step should narrow the set before CRUX ranks it.

### Predicate scaling (K=8)

Categorical filters are linear in the number of predicates, and cheaper than an
ordinal stage because each predicate is one entailment pass over the candidates
rather than a pairwise comparison.

| predicates | warm p50 |
|---:|---:|
| 1 | 464 ms |
| 2 | 1,022 ms |
| 4 | 1,323 ms |
| one ordinal stage (for comparison) | 1,263 ms |

### Other factors (K=8 unless noted)

| factor | warm p50 |
|---|---:|
| candidate text 16 / 32 / 64 / 128 tokens | 1,174 / 1,277 / 1,322 / 1,583 ms |
| CPU threads 1 / 2 / 4 / 6 | 5,810 / 3,121 / 1,737 / 1,386 ms |
| batched grounding vs one sequence at a time (K=16) | 5,424 ms vs 7,427 ms |
| structured lane dispatch (no model) | 0.12 ms |

The structured lane is four orders of magnitude cheaper than the CRUX lane on
the same host, which is why the router uses it whenever the candidates expose
numeric or enum facts.

## Replacing the pairwise comparator with a pointwise field (research)

The pairwise comparator's relation matrix decomposes almost entirely into a
scalar potential. On 200 held-out generic decisions the Hodge integrability was
**1.0000** (10th percentile 0.9999) and the triangle curl RMS was **0.005**, so
the `K*(K-1)` cross-encodings recompute a value that one encoding per candidate
can express. A student that predicts that potential directly, trained against
the frozen comparator as teacher, changes the cost from quadratic to linear.

Same host and protocol as above (i5-12400F, 6 threads, warm, median of 20).

| K | pairwise 150M | pointwise 150M | pointwise field student |
|---:|---:|---:|---:|
| 2 | 59 ms | 38 ms | 17 ms |
| 4 | 262 ms | 64 ms | **23 ms** |
| 8 | 1,228 ms | 108 ms | 33 ms |
| 16 | 5,494 ms | 198 ms | 84 ms |
| 32 | 23,514 ms | 419 ms | 140 ms |

Laya's 421M English model, measured on this same CPU with the same protocol,
scores every option in one forward pass: **77 ms / 161 ms / 170 ms** p50 at
K = 2 / 4 / 8. The pairwise comparator is slower than Laya at K=4 (262 vs 161
ms) and far slower beyond it. The pointwise field student is faster than Laya
at every measured K.

Quality against the pairwise teacher on 400 held-out decisions: the 150M
student agrees on the winner **84.3%** of the time, the smaller student
**93.8%**. The smaller model tracked the teacher better, which is consistent
with the relation being a simple field rather than something that needs model
capacity.

Parameter counts are total, including embeddings. The smaller student is
DeBERTa-v3-xsmall at **70.8M total parameters**: 49.4M of that is the
vocabulary embedding table and 21.3M is the transformer body, plus a 0.15M
potential head, serialized to 283 MB. Calling it a 22M model would count only
the body and is not how the 150M or the 421M Laya figures are counted.

These are research measurements, not a released model: the students cover one
ordinal axis, and the pairwise comparator remains the shipped artifact.

## Semantic Field Atlas (frozen teacher, per axis)

Integrability is not special to one axis. The same decomposition on 150
held-out generic decisions per axis, with the teacher's own ranking checked
against the latent value that generated the text:

| axis | integrability | integrability, 10th pct | curl RMS | latent agreement |
|---|---:|---:|---:|---:|
| room to maneuver | 1.0000 | 0.9999 | 0.0051 | 0.900 |
| advance toward the objective | 1.0000 | 0.9999 | 0.0047 | 0.920 |
| remaining options | 1.0000 | 1.0000 | 0.0033 | 0.933 |
| safety margin | 0.9951 | 0.9917 | 0.0113 | 0.527 |

All four land in the field regime (integrability at or above 0.99), so a
pointwise student is the right architecture for each of them, not just the one
it was trained on. Safety margin is the outlier on accuracy: the field is
integrable but the teacher's ranking agrees with the latent value only about
half the time, which means the comparator reads that axis weakly. That is a
grounding gap, not a geometry problem, and a student distilled from this
teacher would inherit it.

## Axis-conditioned field student (research)

One model takes both the axis and the candidate and emits a scalar potential,
trained on all four graded axes at once. Supervision is the latent grade that
generated the text, not the pairwise teacher, because the teacher ranks safety
margin correctly only about half the time and distilling it would copy that
error. The loss is on field differences plus a ranking term, so the arbitrary
zero point of the potential carries no weight. Evaluated on 400 held-out
decisions per axis:

| axis | winner agreement | pairwise sign accuracy |
|---|---:|---:|
| room to maneuver | 0.938 | 1.000 |
| advance toward the objective | 0.927 | 1.000 |
| remaining options | 0.917 | 1.000 |
| safety margin | 0.630 | 0.774 |

The first three axes are learned to the point where no held-out pair is ordered
wrong. Safety margin is the exception, and since the target here is the latent
value rather than the teacher, the gap is in reading that axis from prose, not
in inheriting the comparator's weakness.

Warm K=4 latency for this run was 54 ms, slower than the 23 ms single-axis
student. Both use the same encoder, and the difference is the tokenizer: this
measurement used DeBERTa's slow sentencepiece tokenizer, the earlier one a fast
tokenizer. The model cost is unchanged; the tokenization path is not a result.

## Unseen-axis generalization (research, negative)

The same architecture trained on four semantic families (capacity, progress,
remaining options, reversibility), each with several axis wordings, and tested
on a fifth family it never saw (risk: safety margin, likelihood of failure,
hazard, exposure to risk).

| family | winner agreement | pairwise sign accuracy |
|---|---:|---:|
| capacity (trained) | 0.907 | 1.000 |
| progress (trained) | 0.937 | 1.000 |
| remaining options (trained) | 0.917 | 1.000 |
| reversibility (trained) | 0.910 | 1.000 |
| risk (held out) | 0.245 | 0.458 |

Trained families are learned completely. The held-out family scores below
chance, so the model is not reading the axis and applying it to new wording.

A counterfactual confirms the mechanism. One option keeps unbounded room but
moves away from the objective; another leaves no room but completes it. Asked
for spare capacity the model should prefer the first, and asked for progress
the second. It preferred the second under both axes. The axis text is not
controlling the score, so the current field student does not generalize to a
semantic criterion it was not trained on. That is the open problem, and it is
not solved by this architecture as it stands.

## Axis-bound field (research)

The previous student ignored the axis because ignoring it was still optimal:
each option mentioned only one fact. Here every option states two facts from
two families, and the score is the support for the queried axis minus the
support for its opposite, so the axis has to select which fact counts.
Reversibility was absent from training.

| family | winner agreement | pairwise sign accuracy |
|---|---:|---:|
| capacity (trained) | 0.917 | 1.000 |
| progress (trained) | 0.937 | 1.000 |
| remaining options (trained) | 0.927 | 1.000 |
| reversibility (held out) | 0.388 | 0.621 |

The trained families are again perfect at the pair level. The axis now controls
the score: one option keeps unbounded room but abandons the objective, the
other leaves no room but completes it, and the model prefers the first under
spare capacity and the second under progress. The earlier student preferred the
same option under both.

The held-out family is only partly there. Reversibility, never seen in
training, reaches 0.621 sign accuracy, above the 0.458 the unbound student
managed on held-out risk but well short of the trained families. Binding the
axis fixes axis control on known criteria. It does not by itself produce a
correct field for a criterion the model has not been trained on.

## Compiling a new axis into the fast field (research)

The base field is trained on capacity, progress and remaining options, then
fine-tuned on reversibility alone at increasing budgets. Each budget repeats
the fine-tune set until roughly 1,600 passes, so the variable is how many
distinct examples the new axis gets, not how long it trains. Sign accuracy on
held-out data:

| reversibility examples | reversibility | capacity | progress | options |
|---:|---:|---:|---:|---:|
| 50 | 1.000 | 1.000 | 1.000 | 1.000 |
| 200 | 0.994 | 0.992 | 1.000 | 0.999 |
| 800 | 0.991 | 0.996 | 1.000 | 1.000 |
| 2,000 | 0.996 | 0.994 | 1.000 | 0.999 |

Fifty examples are enough, and fine-tuning the new axis does not disturb the
ones already compiled. That supports the two-tier runtime: the pairwise model
handles a criterion until it is compiled, and compilation is cheap.

The caveat is the data. The base training loss fell to 0.0001, which means
these templated sentences are far easier than the prose the field will see in
practice. Treat 50 as a floor on the number of examples a new axis needs, not
as the number to budget for real criterion text.

## Compiling a criterion from varied prose (research)

The templated result above used sentences the model could memorize. This one
uses a criterion neither model was trained on, vendor lock-in, written as
varied prose across several sentence frames, and scored against the latent
grade that generated the text.

The pairwise comparator cannot label it: 0.508 sign accuracy, chance. The
frozen NLI backbone can, at 0.752, using entailment of "easy to switch vendors
later" minus entailment of "locks you in." So the labeler for a new criterion
is the NLI backbone, not the comparator.

A field student trained on those NLI labels, evaluated on prose it did not
train on:

| labeled examples | sign accuracy vs latent |
|---:|---:|
| 50 | 0.766 |
| 200 | 0.749 |
| 800 | 0.790 |

The field reaches the labeler and stops there. Fifty examples already match
the NLI backbone's own 0.752, and eight hundred only reach 0.790. More labels
from the same source do not help, because the limit is how well the criterion
can be read from the prose, and the student cannot outgrow its teacher. For a
genuinely new criterion the compilation buys speed, not accuracy: the fast
field reproduces the NLI judgment at field cost, and the residual error is the
grounding, which neither model resolves.

## Richer hypotheses do not ground a new criterion better (research)

The 0.752 above came from one pair of short hypotheses. Replacing it with a
semantic contract, four anchors on each pole aggregated by median, was tested
on the same prose and the same latent grades:

| grounding | sign accuracy |
|---|---:|
| one short hypothesis pair | 0.752 |
| four anchors per pole, median | 0.220 |
| four anchors, hypothesis prior subtracted | 0.158 |
| one verbose hypothesis pair | 0.733 |

The contract made grounding worse, and subtracting the hypothesis-only prior
made it worse again. A single longer hypothesis still scores 0.733, so the
damage is the aggregation, not the wording: the median over several paraphrases
washes out a signal that each hypothesis carries on its own. For a criterion
the model was not trained on, the best cold path found here is the simplest
one, a single direct hypothesis pair, and elaborating it does not buy accuracy.

## An example defines a new criterion better than its name (research)

The same vendor lock-in prose, scored by entailment against a concrete
example of each pole instead of against an abstract hypothesis. The examples
are written separately from the probe text, so nothing is copied across.

| grounding | sign accuracy | winner |
|---|---:|---:|
| abstract hypothesis pair | 0.752 | 0.483 |
| one low example and one high example | 0.825 | 0.600 |
| comparative against one example of each pole | 0.712 | 0.575 |
| two example pairs, averaged | 0.820 | 0.575 |
| four example pairs, averaged | 0.804 | 0.633 |

One pair of examples beats the best hypothesis by seven points on ordering and
twelve on the winner. That is the first thing in this line that moved the
zero-shot ceiling rather than lowering it. More examples do not add anything:
two and four pairs score no better than one, so the gain is in seeing the
criterion instantiated, not in averaging over instances. Asking which option
has more lock-in than a reference is worse than asking which option resembles
it.

So a criterion the model has never seen is grounded best by one example of what
low looks like and one of what high looks like. The name of the criterion, and
any number of paraphrases of it, are strictly worse.

## That gain depends on which example is chosen (research)

The 0.825 above used one hand-picked pair. Twenty distinct phrasings of each
pole, all stating the same grade, were then paired at random and rerun on the
identical probe:

| over 24 prototype pairs | sign accuracy |
|---|---:|
| median | 0.567 |
| 10th percentile | 0.390 |
| 90th percentile | 0.790 |
| worst | 0.352 |
| best | 0.827 |

The best pair reproduces the earlier number and the median falls to 0.567,
below the 0.752 of a plain hypothesis. Most valid examples of the pole are
worse than no example at all, and the worst are close to noise. So the gain is
real but it belongs to the specific example, not to the method of using
examples. Wiring one-low-one-high into the cold path would make accuracy depend
on a choice the caller cannot currently verify.

## True labels compile a new criterion past the zero-shot ceiling (research)

The same vendor lock-in prose, but the training target is the grade that
generated the text rather than another model's judgment. Scored two ways: on
prose built from the clauses the model trained on, and on a second set written
with entirely different clause wording so memorizing phrases cannot help.

| true-labeled examples | seen wording | unseen wording |
|---:|---:|---:|
| 25 | 1.000 | 0.961 |
| 100 | 1.000 | 0.989 |
| 400 | 1.000 | 0.991 |

Twenty-five verified examples clear 0.95 on wording the model never saw, where
the best zero-shot method on this prose, a single hypothesis pair, scores
0.752. The field was capped at its labeler's accuracy in every earlier run
because the labels were noisy. With exact labels the cap disappears and the
criterion transfers to new phrasing immediately.

The cost of compiling a new criterion is therefore the cost of getting correct
labels, not of training. The model's own zero-shot judgments are not correct
enough to be those labels, which is why self-compilation stays closed.

## That sample efficiency does not survive a new criterion (research)

The same procedure on three criteria unrelated to vendor lock-in, each trained
on true latent grades and scored on clause wording and sentence frames held
out of training:

| criterion | 25 labels | 100 | 400 |
|---|---:|---:|---:|
| implementation burden | 0.705 | 0.689 | 0.719 |
| urgency | 0.763 | 0.787 | 0.828 |
| reversibility | 0.848 | 0.853 | 0.832 |

None reaches 0.95, and none improves meaningfully past 100 labels. Vendor
lock-in's 0.961 at 25 examples was a property of that criterion's wording, not
of the method. Where the held-out sentences stay close to the training ones,
transfer looks free. Where they do not, the field saturates well below the
zero-shot hypothesis baseline's complement and more labels do not close it.

So cheap compilation is not yet a capability. It is a result on one criterion,
and the budget a new criterion needs remains something to measure per
criterion rather than a number to quote.

## Wording diversity transfers where more labels do not (research)

Implementation burden again, but the variable is how many ways each grade is
phrased rather than how many rows there are. Every condition trains on the
same 160 examples. The two scoring families, "lift" and "disruption," appear
in no training condition.

| wording families in training | sign accuracy on unseen families |
|---:|---:|
| 1 | 0.712 |
| 2 | 0.826 |
| 4 | 0.852 |
| 8 | 0.781 |

Going from one phrasing to four raises transfer by fourteen points at a fixed
example count, which is more than quadrupling the labels achieved in the table
above. The eighth family gives it back: 0.781, below the four-family result.
So diversity is the lever, and it saturates. Past a handful of distinct ways
to express the criterion, adding more phrasings at a fixed budget thins out
each one and transfer falls.

## The diversity gain is not dilution, and an invariance loss does not help (research)

The drop at eight families above could have been each family getting too few
examples. Separating the two, on the same held-out wording:

| families | examples each | total | unseen-family accuracy |
|---:|---:|---:|---:|
| 4 | 20 | 80 | 0.772 |
| 4 | 40 | 160 | 0.827 |
| 8 | 20 | 160 | 0.740 |
| 8 | 40 | 320 | 0.819 |

Eight families given the same 40 examples as four score 0.819 against 0.827.
The extra diversity does not help once each phrasing is adequately learned, so
the earlier drop was not an artifact of splitting the budget. Four ways of
expressing the criterion transfers as well as eight.

An explicit invariance loss, penalizing disagreement between two phrasings of
the same grade, was added at the best cell (4 families, 40 each):

| loss weight | unseen-family accuracy |
|---:|---:|
| 0 | 0.827 |
| 0.1 | 0.827 |
| 0.3 | 0.810 |
| 1.0 | 0.722 |

It does nothing at a small weight and costs ten points at a large one. The
ranking loss already pulls same-grade sentences together, and forcing it
harder collapses the differences the ranking depends on.

## A factor bottleneck on templated sentences does not test the ceiling (research)

The next attempt scored a sentence through a handful of semantic factors and
trained on controlled interventions: two sentences differing in exactly one
factor, with the score required to move only when the axis depends on that
factor. On wording held out of training it scored 1.000, and the loss was
flat from the first epoch.

That number is not comparable to the 0.827 above. The sentences were three
clauses joined by "and", and an intervention replaced one clause in place, so
the model only had to notice which clause's words changed. That is lexical
differencing on a fixed template, not reading a rewritten sentence. The 0.827
was measured on sentences whose whole surface changed. This run does not move
that ceiling, and a factor model only becomes evidence once the intervention
rewrites the sentence instead of editing a slot in it.


