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

The shipped field, measured through `vey.decide` on this CPU rather than the
research harness: one validated axis, K=4, warm, median of 8 calls, **51 ms**.
It picked the candidate that advances toward food. The Snake demo grounds two
axes per move on CPU, so its on-screen inference time is about twice this.


Scoring every ordinal axis of one decision in a single encoder invocation,
instead of one invocation per axis, on the same CPU and the same weights.
Quantized potentials are identical to the per-axis path and the winner is
unchanged under candidate permutation.

| axes | per-axis invocations | one invocation |
|---:|---:|---:|
| 1 | 41 ms | 51 ms |
| 2 | 104 ms | 79 ms |
| 3 | 155 ms | 110 ms |

One axis is slightly slower batched, because the batch machinery has nothing to
amortize. Two and three axes are faster. The transformer still encodes one
sequence per axis-candidate pair, so the gain is the removed per-axis overhead,
not shared computation.


A single axis does not go through the batched path. Routing it there made one
axis slower, 41 ms to 51 ms, for no gain. With the direct path restored the
same measurement reads 38 ms, 79 ms and 103 ms for one, two and three axes.


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

## One encoding can carry all three compiled axes (research)

The shipped field encodes the axis and the candidate together and emits one
number, so three axes cost three passes. A model with the same backbone that
encodes only the candidate and emits all three coordinates at once, trained on
the same teacher labels and scored on decisions held out of that training:

| axis | shipped field | shared encoding |
|---|---:|---:|
| food progress | 0.958 | 0.954 |
| open space | 0.961 | 0.966 |
| headroom | 0.956 | 0.981 |

Food progress is four thousandths under the shipped field. The other two are
above it. The axis is no longer read from text at runtime; it is a coordinate
of a vector the candidate encoding already contains. That is what makes the
encoder cost independent of how many of these axes a decision names.

A confirmation run on a fresh held-out split of 250 decisions, untouched by
both models, scores both on the same pairs:

| axis | shipped field | shared encoding |
|---|---:|---:|
| food progress | 0.9519 | 0.9470 |
| open space | 0.9756 | 0.9756 |
| headroom | 0.9860 | 0.9854 |

Food progress reads 0.005 lower. A paired bootstrap of that difference over
10,000 resamples of the same pairs gives a 95% interval of [-0.0146, +0.0049],
which crosses zero, so the gap is within what this sample can resolve. Open
space and headroom match or exceed the shipped field outright.

## Six of twelve layers carry the compiled field (research)

Depth was ablated two ways on the development split. A linear head on a
frozen encoder scores 0.70 to 0.77 on every layer probed, last layer included,
so the coordinates are not linearly readable from the stock representation;
the encoder has to be trained for them. Retraining truncated models is the
real measurement:

| depth | food progress | open space | headroom |
|---:|---:|---:|---:|
| 12 | 0.9470 | 0.9756 | 0.9854 |
| 8 | 0.9439 | 0.9762 | 0.9866 |
| 6 | 0.9464 | 0.9775 | 0.9799 |
| 4 | 0.9330 | 0.9738 | 0.9817 |

Eight and six layers are non-inferior to twelve on every axis separately;
six reads above twelve on food progress. Four layers loses food progress. The
shallowest depth that holds all three coordinates is six, which halves the
transformer compute. The frozen-encoder probes also rule out reading these
coordinates off the stock model without training it.



## The six-layer shared field fails the promotion gates (research)

The speed gate passed first: K=4, three coordinates, CPU, warm, p50 18.9 ms
against 103 ms for the shipped three-axis path, 5.45 times. The frozen
protocol then opened one untouched split of 250 decisions, seed 11, and scored
only the six-layer model against the shipped field.

| gate | required | measured |
|---|---:|---:|
| food progress paired delta | >= -0.010 | -0.0126 |
| open space paired delta | >= -0.010 | -0.0013 |
| headroom paired delta | >= -0.010 | -0.0025 |
| winner agreement | >= 0.995 | 0.800 |
| survivor-trace agreement | >= 0.990 | 0.476 |
| permutation invariance | 1.000 | 1.000 |

Food progress, the winner, and the survivor trace all fail. The pairwise
coordinate scores stay close, but after lexicographic composition the decision
itself changes on one case in five and the survivor set on about half. The
shipped field stays the public runtime. This split is closed; the next attempt
needs a new one.


## The shared field only disagrees at the resolution boundary (research)

A fresh diagnostic split of 250 decisions, seed 13, compares the six-layer
shared field to the shipped field pair by pair. Each pair is classed as
strictly better, tied, or strictly worse at the Semantic Resolution band of
0.15.

| axis | ternary agreement | crossings |
|---|---:|---:|
| food progress | 0.9486 | 84 of 1635 |
| open space | 0.9664 | 55 of 1635 |
| headroom | 0.9829 | 28 of 1635 |

Every crossing falls in the two bins touching the band, teacher gaps of 0.15
to 0.20. No pair whose teacher gap exceeds 0.30 crosses, and no pair inside
0.10 does either. A per-axis scale fit, forcing the student difference to match
the teacher's, gives multipliers of 0.954, 0.993 and 0.998 and moves ternary
agreement by about a thousandth. The student is not mis-scaled. It is wrong
exactly where the teacher's own gap sits on the boundary the executor uses,
which a ranking loss never sees.


## An epsilon-relation loss does not move the boundary (research)

The six-layer shared field again, with the loss changed and nothing else. Each
pair is hinged into the interior of the class the shipped field assigns it at
the 0.15 band, with a 0.02 margin, and pairs near the band are weighted up to
four times. Ranking and potential losses stay in at 0.1 and 0.05. Scored on a
fresh development split, seed 17:

| axis | ternary agreement | tie to strict | strict to tie |
|---|---:|---:|---:|
| food progress | 0.9496 | 35 | 47 |
| open space | 0.9717 | 24 | 22 |
| headroom | 0.9828 | 22 | 6 |

No pair flipped from better to worse. The agreement is the same as the
ranking-trained model measured on the diagnostic split, 0.9486, 0.9664 and
0.9829. The prerequisite for opening another final split was 0.99 on every
axis. This does not approach it, so that split stays undrawn and the shipped
field stays the runtime.


## Trace stability tracks distance to the resolution boundary (research)

The shipped field's own scores, perturbed, on 120 decisions. For each decision
the distance from the nearest candidate to the 0.15 Semantic Resolution band is
recorded, then uniform noise is added to the scores until the executor's trace
changes.

| distance to nearest boundary | decisions | trace flipped by noise of 0.04 or less |
|---|---:|---:|
| under 0.02 | 41 | 0.76 |
| 0.02 to 0.05 | 34 | 0.53 |
| 0.05 to 0.10 | 25 | 0.12 |
| 0.10 to 0.20 | 20 | 0.00 |

The median decision sits 0.031 from a boundary, and 103 of 120 traces flip
under noise of 0.16. Nothing beyond 0.10 flips at 0.04. Of 354 pairs within
0.05 of the band, 204 lie just inside it and 150 just outside, so both error
directions are common. A score error only changes the decision when it crosses
the band, and how close the decision already is tells you whether it will.

## A certified scout is cheap on two axes and expensive on the hard one (research)

A six-layer student of the shipped field, scored on 250 held-out decisions. A
pair is certified when the student's distance from the 0.15 band exceeds a
calibrated error radius, meaning its ternary relation cannot reach the band.
The smallest radius that catches every wrong relation, and the fraction of
pairs that radius still leaves uncertified:

| axis | shared scout, bare text | control, axis-conditioned |
|---|---|---|
| food progress | radius 0.0618, uncertified 0.300 | radius 0.0487, uncertified 0.238 |
| open space | radius 0.0545, uncertified 0.152 | radius 0.0484, uncertified 0.133 |
| headroom | radius 0.0359, uncertified 0.074 | radius 0.0108, uncertified 0.023 |

The control changes only the input: the shipped field's "Axis: … Candidate: …"
format with a one-output head, instead of bare text with three outputs. Same
depth, loss, epochs, split and batch size. Axis conditioning helps every
coordinate and helps headroom most, collapsing its radius from 0.0359 to
0.0108. It does not rescue food progress, which improves from 0.300 to 0.238
uncertified and remains the axis where a certified cascade would have to pay
for deep computation on nearly a quarter of pairs. The gap on food progress is
therefore not the missing axis name. The two candidate explanations left were
a weak readout or missing depth; the probe below rules out the first but does
not prove the second.

## A stronger readout does not recover near-boundary accuracy (research)

The six-layer control's encoder was frozen completely and an overpowered head,
four times wider with two hidden layers, was trained on its mean-pooled token
states against the same food-progress pair differences. If the boundary
information survived six layers, this head should recover it.

| teacher gap | overpowered probe | deployed head |
|---|---:|---:|
| all pairs | 0.9659 | 0.9665 |
| within 0.05 of the band | 0.8637 | 0.8662 |
| within 0.02 of the band | 0.7268 | 0.7268 |

The probe matches the deployed head everywhere, including exactly 0.7268 in
the tightest band. A much larger head on the same states recovers no additional
near-boundary accuracy, so the limit is the representation rather than the
readout, and there is currently no evidence that a cheaper readout can solve
the boundary. Two caveats bound the claim. The probe pools mean tokens while
the deployed head reads the CLS state, so the comparison varies pooling
alongside head capacity. And its training loss averages Huber uniformly over
all pairs, which easy pairs dominate, so it was never optimized for the
near-band region it is measured in; training loss reached 0.0002 in one epoch
while held-out near-band agreement stayed 0.7268, a gap consistent with
boundary under-fitting as well as absence. A boundary-focused probe is the
discriminating follow-up. An earlier probe that scored 0.6885 in the tight band
is invalid: it froze a freshly initialized encoder and never loaded the
trained weights, so its number measures random layers, not this
representation.

## The layer-depth curve is flat, and the diagnostic is the bottleneck (research)

The same deployed-sized CLS probe was trained independently on every layer's
hidden state of the shipped 12-layer field encoder, food progress only. If
boundary information emerged at some depth, agreement in the tight band would
rise with layer. It does not:

| layer | agreement within 0.02 | uncertified fraction |
|---:|---:|---:|
| 1 | 0.5410 | 0.967 |
| 3 | 0.5738 | 0.803 |
| 6 | 0.6066 | 0.804 |
| 8 | 0.6120 | 0.844 |
| 10 | 0.6284 | 0.875 |
| 12 | 0.5519 | 0.925 |

Layer 12 scores no better than layer 1. This is the uninterpretable outcome.
The teacher labels were verified to be the shipped field's own output, 312
score comparisons with a worst absolute difference of 0.000000, so the shipped
head can recover its own field exactly and the missing distinction is
recoverable from the final layer; yet the probe cannot get it at any depth.
The full 12-layer curve is in the run log; the table above elides the
intervening layers. The curve measures the diagnostic's ceiling, not where
semantic resolution emerges.
Two caveats bound the read. This probes the stock
pretrained encoder, so it is architecture evidence only and says nothing
about appending layers to the independently trained six-layer scout; that
lineage difference is real and a continuation from the exact scout checkpoint
must be trained separately. And the probe is a small CLS head, which the
boundary-focused runs show is the weakest of the readouts tried. The result
closes nothing about depth; it says the next diagnostic must be the
continuation itself, trained end to end from the six-layer checkpoint and
measured on the tight band.

## Continuation does not recover the boundary, and the frontier version is worse (research)

The continuation experiment, corrected after an earlier version was retracted
for training on the wrong target: the deep branch now learns only the
correction, the pair difference of its delta against the teacher gap minus the
scout gap, and the new blocks initialize from the stock pretrained encoder's
layers 7 through 6+N rather than copies of the scout's last layer. Seed pinned,
stratified to the deployment distribution, the exact frozen scout as prefix,
and the frontier-only ablation included. Held-out food progress:

| depth | agreement within 0.02 | uncertified | frontier-only, uncertified |
|---|---:|---:|---:|
| 6 | 0.7377 | 0.2686 | - |
| 6+1 | 0.6011 | 0.3410 | 0.5177 |
| 6+2 | 0.7596 | 0.2765 | 0.4019 |
| 6+3 | 0.7213 | 0.3039 | 0.3465 |
| 6+4 | 0.7158 | 0.2533 | 0.7607 |
| 6+6 | 0.7268 | 0.2613 | 0.7229 |

No depth clears the scout meaningfully. The best, 6+2, gains 0.022 in the
tight band while its uncertified fraction rises. The frontier-only ablation is
the harder finding: when only unresolved candidates pass through the deep
blocks, the mixed scout-deep edges become less certifiable than the scout
alone, 0.40 to 0.76 uncertified against 0.27 for the plain scout. Localizing
the extra computation to the frontier, the mechanism the cascade depends on,
degrades the certificate rather than tightening it. A second run of the same
configuration before the seed was pinned reproduced the same direction with the
baseline at 0.7541, so the pattern is not seed noise.

Two earlier versions of this experiment are retracted on the record: one
trained the correction head on the full gap instead of the residual, and one
initialized the new blocks as copies of the scout's own last layer. The result
above is the one under the corrected protocol.

## Flux-0: relational, gauge-fixed training does not beat the scalar head (research)

The first program-conditioned field experiment. Three arms, each with its own
freshly initialized 12-layer encoder trained end to end, the same per-arm seed,
the same deployment-stratified sampling, one decision per gradient step.
Baseline A is the existing architecture's shape, a per-candidate scalar head
with plain gap Huber. Flux-B adds a pooled set context, gauge projection onto a
zero-sum potential, and executor-aware losses (gap, margin-to-epsilon, ternary
relation hinge, survivor BCE). Flux-C adds a raw pair-difference branch with
Laplacian projection and a cycle-consistency loss.

| arm | agreement within 0.02 | within 0.05 | all | radius | uncertified |
|---|---:|---:|---:|---:|---:|
| baseline A | 0.6284 | 0.7956 | 0.9488 | 0.0790 | 0.349 |
| Flux-B | 0.5792 | 0.6569 | 0.8940 | 0.3001 | 0.789 |
| Flux-C | 0.5519 | 0.7129 | 0.9153 | 0.2072 | 0.721 |

Both Flux arms score below the plain scalar head in the tight band, the
scale-invariant and therefore comparable metric. Their residual radii are not
comparable across arms in this run: the baseline's only loss directly
minimizes the gap residual, while the Flux arms split their objective across
four terms in which magnitude is one quarter of the mean, so a scale mismatch
inflates their radius mechanically. The arms also differ in loss, not only
architecture: the earlier epsilon-relation experiment (422934b) already showed
executor-shaped losses do not help the same architecture, so the deficit
cannot be attributed to set context or gauge projection alone. The set
context implemented here was a mean-pooled context through a tanh bottleneck,
the floor of the design space, not a slotted bottleneck. The Flux arms'
training losses oscillate while the baseline converges smoothly, so the
multi-term objective was harder to optimize, not easier. An earlier version
of this run is invalid and is not counted: it trained all arms on a frozen
stock encoder whose activations were computed under no_grad, so no arm ever
trained its encoder. The comparison above is the corrected, end-to-end run,
with every arm's encoder trained and the gradient verified by a smoke test
before launch.

The honest negative is narrow: under this training regime, one decision per
step, K candidates jointly encoded, the specific set-context and gauge
machinery tested here did not help. The absolute numbers are not comparable to
the ~0.74 fine-tuned baseline in earlier sections, which trained on multi-row
batches with a different loop; the controlled comparison is arm against arm.
Whether relational training helps under a batched regime closer to the shipped
field's recipe remains untested.

## BRF-0: a zero-init boundary residual helps; executor losses hurt (research)

The factorial over architecture and loss, two-phase per protocol: the scalar
field trains to convergence first; the residual attaches to it frozen. The
residual is per-decision candidate-to-candidate cross-attention, not learned
relation slots: there is no slot parameter and the module-level slot count was
unused. It is zero-initialized, so it starts as the exact base field. The
boundary gate reads the live base scores; the base stayed fixed because the
encoder and head were excluded from the Phase-2 optimizer, which the smoke
test verified, and not because the gate detached them. Arm D's fine-tune put
the head back into the optimizer at a small learning rate. Calibration rows
(50) are excluded from training; the scale alpha is fit on them only. All four
arms trained on 600 with-replacement rows per epoch, a lighter recipe than the
full passes behind the ~0.74 baselines, so absolute levels are arm-against-arm
only.

| arm | agreement within 0.02 | scaled | radius (scaled) | uncertified (scaled) | rho |
|---|---:|---:|---:|---:|---:|
| A scalar, gap | 0.5733 | 0.6200 | 0.1010 | 0.447 | 0.9929 |
| B scalar, executor losses | 0.5333 | 0.5467 | 0.1128 | 0.523 | 0.9901 |
| C A + zero-init residual | 0.6533 | 0.6400 | 0.1021 | 0.461 | 0.9931 |
| D C + executor fine-tune | 0.6267 | 0.6467 | 0.0847 | 0.395 | 0.9945 |

Architecture and loss separate cleanly. The residual branch improves the
scalar field, +0.08 raw and +0.02 after scale correction, and arm D carries the
best residual radius and uncertified fraction. The executor losses hurt the
same scalar field (A to B, -0.04 both ways), reconfirming the earlier
epsilon-relation result on a 12-layer end-to-end base. The scale correction
matters: the gap-trained baseline came out 21 percent mis-scaled (alpha
0.79), and the raw numbers underrate it accordingly; alpha near 0.93 to 0.99
for the residual arms shows the residual target itself regularizes scale.
Correlation with the teacher is at least 0.99 on every arm, so all models
order pairs almost perfectly and every difference is boundary placement.

The promotion gate asked for a scaled gain of at least 0.05 over A. C and D
deliver +0.02 to +0.03 with D's certificate the strongest of the four. That is
a real, reproducible improvement from the residual architecture and the first
positive architectural result in this line, but it is below the gate and does
not reach the 0.80 target. The direction is validated: preserve the scalar
field, spend relational capacity only on its near-boundary residual, and do
not add executor-shaped losses.






## BRF-1: the residual plus executor fine-tune clears the gate (research)

Full-recipe confirmation of the boundary residual, three seeds. Phase 1 now
matches the strong scalar recipe exactly: three-axis conditioning, six layers,
batch 16, single learning rate 3e-5, three full passes. The residual attaches
to the frozen converged field; each arm restores the base snapshot before its
own training and scoring, so no arm sees another's mutated weights. Alpha is
fit on a 50-row calibration split excluded from training; the gate uses the
scale-corrected column.

| arm | mean A_0.02 (scaled) | delta vs A | spread | positive every seed | best radius | uncertified |
|---|---:|---:|---:|---|---:|---:|
| A scalar | 0.6400 | - | - | - | 0.0804 | 0.371 |
| C0 candidate cross-attention | 0.6533 | +0.0133 | 0.0499 | no | 0.0953 | 0.424 |
| C4 four learned slots | 0.6555 | +0.0155 | 0.0515 | no | 0.0953 | 0.424 |
| C8 eight learned slots | 0.6533 | +0.0133 | 0.0484 | no | 0.0953 | 0.424 |
| D8 C8 + executor fine-tune | 0.7067 | **+0.0667** | 0.0340 | yes | **0.0710** | **0.339** |
| M monotone odd map | 0.6378 | -0.0022 | 0.0062 | no | 0.0800 | 0.370 |

The residual alone does not clear the gate. C0, C4 and C8 improve the scalar
field by about 0.013 on average, but with a seed spread of roughly 0.05, so
the effect is not separable from noise. Their raw agreement looked higher
(0.66 to 0.75) because the residual arms also fixed the field's scale
contracture, alpha 0.91 to 0.97 against the baseline's 0.83 to 0.91; after
correction the gain shrinks to the noise floor. The slot bottleneck adds
nothing over plain candidate cross-attention within 0.002 across both slot
counts.

The decisive term is the short Phase-3 executor fine-tune, and it only works
once the residual exists. D8 gains +0.0667 mean over the same-seed baseline,
positive on all three seeds (+0.033, +0.053, +0.113), with the tightest
spread (0.034) and the strongest certificate of any arm in either run:
radius 0.071 to 0.078, uncertified 0.339 to 0.358, correlation 0.996. It also
lands at alpha 1.007 to 1.020, scale-correct within two percent, so its
advantage is not a scale artifact. The monotone map control is a clean null,
-0.002 with a 0.006 spread, which rules out the entire gain being a
calibration trick.

The interaction is the finding. Executor-shaped losses cost the scalar field
about 0.04 when applied directly, in BRF-0 arm B and in the earlier
epsilon-relation experiment, but the same losses gain 0.05 to 0.06 when the
residual geometry is already present. The residual supplies a region in
parameter space where the executor objective can be optimized without
damaging the competent scalar field; without it, the same objective
degrades that field.

Withdrawn. BRF-2 reruns this arm under the sequential schedule the protocol
specified, base frozen and the executor loss applied only after the residual
converges, and the gain reverses to -0.0133 with no seed-positive arm. The
+0.0667 belongs to the joint schedule in which the residual and the executor
loss trained together with the head free to move, not to the architecture;
see the BRF-2 section. The numbers below remain a valid record of what that
run measured, but the promotion does not hold, and neither does the
interaction claim: under the sequential schedule the executor loss helps the
bare scalar and hurts every residual arm.

## BRF-2: the BRF-1 gain does not survive the sequential schedule (research)

Mechanism dissection. Same strong base, same seeds, same arms as BRF-1 plus a
decisive control: DP, a per-candidate residual with no candidate interactions
at all. Phase 2 (residual only) and Phase 3 (executor) are now strictly
sequential, with the base frozen throughout -- the schedule as written, where
BRF-1's D8 trained residual and executor jointly with the head free to move.

| arm | mean A_0.02 (scaled) | delta vs A | per-seed deltas | positive every seed |
|---|---:|---:|---|---|
| A scalar | 0.6400 | - | - | - |
| B executor on bare scalar | 0.6600 | +0.0200 | +0.047, -0.013, +0.027 | no |
| D0 cross-attention + executor | 0.6266 | -0.0134 | -0.007, -0.073, +0.040 | no |
| D4 four slots + executor | 0.6356 | -0.0044 | -0.013, -0.067, +0.067 | no |
| D8 eight slots + executor | 0.6267 | -0.0133 | -0.013, -0.067, +0.040 | no |
| DP per-candidate + executor | 0.6267 | -0.0133 | -0.020, -0.080, +0.060 | no |

The result is negative and the ordering is unambiguous: arm B, the executor
loss applied directly to the bare scalar with no residual at all, is the best
arm in the run at +0.0200, above every residual arm, all of which are negative
against the baseline. The executor loss does not need a residual to help under
this schedule, and the residual does not help with or without it.

DP matches every relational variant within 0.01, so candidate interactions
contribute nothing measurable, and the residual arms' raw advantage (0.65 to
0.71 against 0.55 to 0.64) is scale repair: their alpha sits near 0.92 against
the baseline's 0.88, and after correction the gain is gone. The relational
machinery is not the mechanism, and under this schedule the residual itself is
not one either.

BRF-2 does not reconcile with BRF-1, and the difference is a schedule confound
the two runs cannot separate. BRF-1's D8 gained +0.0667 with the residual and
the executor loss trained jointly from the start and the head free to move at
3e-6; BRF-2's identical arm loses 0.0133 with the base frozen and the executor
loss applied only after the residual converged. The frozen-versus-joint
distinction is exactly what the protocol asked to be tested, and BRF-2 does not
test it, because every D arm in this run freezes the base. The missing
comparison is a D8 variant that trains jointly as BRF-1 did, on this run's
code and metrics. Until that exists, BRF-1's +0.0667 cannot be attributed to
the architecture rather than to the joint schedule, and the boundary-residual
line has no demonstrated mechanism under the protocol as written. BRF-1's
promotion is withdrawn on that basis, not on the strength of this run alone.

## BRF-3: the gain is the moving head, not the residual structure (research)

The comparison BRF-2 left open, run with one variable. D8J and D8F see the same
loss, the same two epochs, and the same residual learning rate; the only
difference is that D8J puts the head in the optimizer at 3e-6 and D8F keeps it
out. The encoder is trained in Phase 1 and excluded from the Phase 2 and Phase
3 optimizer in every arm; the joint phase encodes under no_grad, so no gradient
reaches it there. Guards assert the head changed for every moving arm and did not change for
D8F, and the recorded head movement confirms it: exactly 0.0 for D8F, 0.0049 to
0.0063 for every other arm.

| arm | mean A_0.02 (scaled) | delta vs A | spread | per-seed deltas | positive every seed |
|---|---:|---:|---:|---|---|
| A scalar | 0.6400 | - | - | - | - |
| B executor on bare scalar | 0.6822 | +0.0422 | 0.031 | +0.053, 0.000, +0.073 | no |
| DPJ per-candidate, head moving | 0.7133 | +0.0733 | 0.028 | +0.053, +0.053, +0.113 | yes |
| D0J cross-attention, head moving | 0.7156 | +0.0756 | 0.037 | +0.040, +0.060, +0.127 | yes |
| D8F eight slots, head frozen | 0.6533 | +0.0133 | 0.048 | -0.007, -0.033, +0.080 | no |
| D8J eight slots, head moving | 0.7200 | +0.0800 | 0.036 | +0.040, +0.073, +0.127 | yes |

D8J beats D8F by 0.067 on identical code, and its +0.080 reproduces BRF-1's
+0.067 under the matched procedure, so the earlier number was not an artifact.
The gain is the moving head.

But the structure of the residual does not matter. DPJ, a per-candidate
residual with no candidate interactions at all, gains +0.073, and D0J, plain
candidate cross-attention, gains +0.076. Both are positive on every seed and
within 0.007 of D8J. The eight-slot bottleneck adds nothing over the simplest
residual that exists. What the joint schedule needs is some residual
parameterization present while the head moves, not relational machinery.

The frontier diagnostic rejects the local-warping interpretation. Mean score
movement near the band is smaller than movement far from it for every joint
arm, with near-to-far ratios of 0.41 to 0.65. The training moves the metric
more where the teacher is far from the threshold, not where it is close. D8F,
the frozen arm, is the only one where near-band movement exceeds far-band
movement, and it is also the arm that does not work.

The mechanism, then, is narrower than BRF-1 claimed and cheaper than the
architecture it proposed: joint executor fine-tuning of the scalar head in the
presence of a residual branch. The residual's job is to exist, not to reason
over candidates. Arm B, the executor loss on the bare scalar, gains only
+0.042 and is not positive on every seed, so the residual's presence still
matters, but its form does not.

## BRF-4: the residual's function class does not matter either (research)

A function-class ladder on the same base and the same joint schedule. Every
residual is zero-initialized and trained jointly with the head.

| arm | mean A_0.02 (scaled) | delta vs A | spread | per-seed deltas | positive every seed | residual share |
|---|---:|---:|---:|---|---|---:|
| A scalar | 0.6400 | - | - | - | - | - |
| B executor on bare scalar | 0.6822 | +0.0422 | 0.031 | +0.053, 0.000, +0.073 | no | - |
| R-linear | 0.7133 | +0.0733 | 0.043 | +0.053, +0.033, +0.133 | yes | 0.031 |
| R-fixed | 0.7133 | +0.0733 | 0.039 | +0.060, +0.033, +0.127 | yes | 0.031 |
| R-mlp | 0.7133 | +0.0733 | 0.038 | +0.053, +0.040, +0.127 | yes | 0.031 |

The three residuals tie to four decimals. R-linear is a single linear layer on
the same features the head already sees. R-fixed is a linear layer on a frozen
random projection, so it cannot learn a representation at all; it only supplies
fixed basis functions. R-mlp is the per-candidate MLP that led BRF-3. All three
gain +0.0733, all positive on every seed, all within each other's spread.

The residual carries about 3 percent of the final output magnitude in every
arm: its root-mean-square is 0.007 against 0.20 to 0.24 for the head. So the
branch that accounts for the gain contributes almost none of the score. That is
the signature of an optimization catalyst rather than a second field: the
residual changes the directions the executor objective can move in, and the
head absorbs the result.

What this does not show. The base head is a two-layer network with a GELU, not
a linear map, and the residual is multiplied by a data-dependent gate, so the
two branches do not sum to a single linear function and cannot be folded
together algebraically. R-linear is a cheap linear residual on the same
features; it is not a proof that the gain survives merging the weights and
dropping the gate. That measurement is still missing, and it is the one that
would decide whether the residual can be deleted at inference.

## BRF-4b: the residual must ship, the gate does not matter (research)

The same joint training as BRF-4, scored three ways on the post-training head:
the trained model, the residual deleted, and the residual kept with its gate
removed.

| scoring | mean A_0.02 (scaled) | delta vs A | spread | per-seed deltas | positive every seed |
|---|---:|---:|---:|---|---|
| linear, full | 0.7133 | +0.0733 | 0.043 | +0.053, +0.033, +0.133 | yes |
| linear, residual deleted | 0.6689 | +0.0289 | 0.017 | +0.047, +0.007, +0.033 | yes |
| linear, gate removed | 0.7111 | +0.0711 | 0.038 | +0.067, +0.027, +0.120 | yes |
| mlp, full | 0.7200 | +0.0800 | 0.039 | +0.040, +0.067, +0.133 | yes |
| mlp, residual deleted | 0.6644 | +0.0244 | 0.013 | +0.033, +0.007, +0.033 | yes |
| mlp, gate removed | 0.7155 | +0.0755 | 0.036 | +0.053, +0.047, +0.127 | yes |

Deleting the residual collapses the gain from +0.073 to +0.029 for the linear
branch and from +0.080 to +0.024 for the mlp, both below the +0.042 that the
executor loss achieves on the bare scalar head. The residual is load-bearing at
inference. It is not training-only scaffolding, and it cannot be deleted.

Removing the gate does not either. Gated and ungated differ by 0.002 to 0.005
on every arm and seed, inside the seed spread, so the gate is functionally
irrelevant at inference rather than harmful. The simpler ungated form is the
one to ship.

Of the functional change learned during the joint phase, the residual carries
0.11 to 0.20 and the head the rest, several times its 3 percent static output
share. The head absorbs most of the change, but the slice the residual carries
is the slice the score depends on, which is why deleting it costs more than its
magnitude suggests.

So the deployed model is the scalar head plus a small ungated residual, and the
residual is a linear layer on the same features: nothing relational, nothing
gated, and about 3 percent of the output magnitude. Whether that residual can
be absorbed rather than shipped is answered in the next section: it folds
exactly into the head.

## BRF-5: the ungated residual folds exactly into the head (research)

GELU(z) - GELU(-z) = z, so a linear residual r(h) = a^T h + beta is reproduced
exactly by two extra hidden units with weights (a, beta) and (-a, -beta) and
output weights +1 and -1. The training-time decomposition becomes an ordinary
GELU head two neurons wider, with no residual branch and no gate.

Verified on one seed (7) with the trained weights: the folded head and the two
separate branches differ by at most 1.19e-07 across 807 held candidates and
2.38e-07 across 4096 random feature vectors, both FP32 rounding. Those two
errors are the evidence. The two forms also score identically, but that is an
identity rather than a measurement: the metric uses only pairwise differences,
and the folded head differs from the separate branches by a per-decision
constant that cancels in every difference, so the two cannot disagree on it.
The reported 0.6867 is raw and single-seed, since this run fits no scale
factor, and it is not comparable to the scaled three-seed tables above.

One seed is enough here because the claim is algebraic, not statistical. The
identity does not depend on the seed; the check is against floating-point noise.

The deployed model is the original scalar field head at width 386 rather than
384, which is 0.52 percent wider: no relational machinery, no gate, no residual
kernel. The two-path training parameterization and the deployed one-path model
compute the same function. This does not retract BRF-4b. The residual is still
load-bearing at inference; the fold shows that the load-bearing function was
always inside the class of a head two neurons wider, and that training the head
directly does not reach it. Both hold: the residual must ship, and the form it
ships in is a stock head.

## Enlarged held evaluation: the gain replicates at +0.039 (research)

Frozen models, no retraining, on a 2,000-decision pool drawn from the same
corpus outside the 2,250 training rows and never used in any BRF run. Alpha is
fit on a 50-decision calibration split excluded from the test; the remaining
1,950 decisions yield 1,386 tight-band pairs, against about 150 per seed in the
earlier evaluations. The paired difference is bootstrapped over decisions, not
pairs, so comparisons sharing a decision stay together.

| metric | scalar baseline | folded head |
|---|---:|---:|
| A_0.02 | 0.6378 | 0.6768 |
| alpha | 0.8874 | 1.009 |
| radius q | 0.0880 | 0.0735 |
| uncertified | 0.3689 | 0.3199 |
| rho | 0.9951 | 0.9965 |

The paired delta is +0.039 with a 95 percent confidence interval of
[0.0163, 0.0619], entirely above zero, so the effect is not small-sample noise.
It is smaller than the +0.07 of the three-seed runs. That narrowing is
observed, not explained. The interval's upper bound is
consistent with the earlier estimate. The certificate improvements move with
it: the folded head's radius is 16 percent smaller and its uncertified
fraction drops 13 percent relative to the baseline.

This is one seed (7), matching the fold run, so it confirms exactness of the
gain rather than its magnitude across seeds. The final split remains untouched.

## STEF axis transfer: same recipe, same result on two of three axes (research)

The frozen food-progress recipe applied unchanged to the other two axes, each
independently from the same Phase 1 snapshot. No retuning of any kind: same
band, same learning rates, same schedule, same fold construction.

| axis | delta A_0.02 | 95% CI | clears zero | tight pairs | A_0.05 baseline to folded |
|---|---:|---|---|---:|---|
| food progress | +0.039 | [0.0163, 0.0619] | yes | 1,386 | 0.775 to 0.811 |
| open space | +0.0529 | [0.0188, 0.0855] | yes | 699 | 0.766 to 0.811 |
| headroom | +0.0276 | [-0.0069, 0.0633] | no | 434 | 0.828 to 0.874 |

Food progress reproduces the promoted numbers exactly, confirming the
pipeline is deterministic end to end. Open space gains more than food did.
Headroom points the same direction but its interval touches zero, so the
effect is unproven there rather than refuted; with 434 tight pairs, roughly a
third of the food sample, the interval is wide enough that a real +0.03 gain
could hide inside it. The A_0.05 columns, absent from the earlier
evaluation, are now filled: the wider band moves the same direction on all
three axes.

The tight-band density differs by axis by construction: the band sits at
plus-or-minus 0.02 around a gap of 0.15, but the median teacher gap is 0.157
on food, 0.364 on open space, and 0.505 on headroom, so the band samples a
thinner slice of each successive axis. That is the price of not retuning, and
the confidence intervals carry it honestly.

The in-run exactness probe, a single-decision check, disagreed by 0.004 to
0.033 and the cause is unexplained: both sides of the comparison were ungated,
so it is not a gate mismatch. It is not used as evidence. The binding
verification remains the BRF-5 section: 1.19e-07 worst error across 807 held
candidates and 2.38e-07 across 4096 random features.

## STEF shared field: one field, three axes, all gates pass (research)

The deployment question: can one STEF-trained field improve the hard axis
without damaging the easy ones? One encoder, one shared head, three
axis-conditioned linear auxiliaries trained jointly in a single phase, then
folded. Phase 1 is byte-identical to every prior run, pooled scale. Phase 2
normalizes by per-axis scale (0.261, 0.507, 0.675) and weights the three
axis losses equally, so headroom's thinner band does not shrink its gradient.
Gates were fixed before the run: food delta above zero with CI low above
zero, open space and headroom non-inferior at -0.01, A_0.05 non-inferior at
-0.005 on all three.

| axis | delta A_0.02 | 95% CI | tight pairs | A_0.05 baseline to folded |
|---|---:|---|---:|---|
| food progress | +0.0397 | [0.0171, 0.0625] | 1,386 | 0.775 to 0.811 |
| open space | +0.0486 | [0.0142, 0.0824] | 699 | 0.766 to 0.809 |
| headroom | +0.0300 | [-0.0047, 0.0653] | 434 | 0.828 to 0.874 |

All five gates pass. Food and open space gain as much under the shared head
as they did when trained alone, so the axes do not interfere destructively;
headroom's point estimate holds at +0.03, its interval still touching zero at
a third of the food sample, and nothing degrades on any axis. Headroom has
now landed near +0.03 in two independent runs, which accumulates confidence
without settling it.

The auxiliary vectors are near-orthogonal (cosines 0.17, -0.10, -0.15), so
the axes need distinct correction directions and no shared low-rank basis
compresses them. The deployed form for a three-axis field is three per-axis
folded heads, each a 386-wide stock head with its own antipodal pair; a
single 390-wide head would emit one scalar, not three axis scores.

This is the last pair-level result needed: STEF is a candidate replacement
for the shipped field. What remains is program-level evidence, not more
pair-level architecture work.

## STEF at program level: the executor traces agree, and the right ones are fixed (research)

The real Vey executor on the 1,950 untouched test decisions, program
"Prefer more food progress, then more open space, then more headroom" —
compiling to MAX food progress / MAX open space / MAX headroom at the ordinal
resolution 0.15, the same epsilon every experiment in this line used. Three
scorers: the teacher's FieldScorer potentials, the Phase-1 head, and the
folded STEF heads. Trace agreement compares every stage's survivor set.

| metric | baseline | STEF | delta | 95% CI |
|---|---:|---:|---:|---|
| full trace agreement | 0.8359 | 0.8744 | +0.0385 | [0.0226, 0.0549] |
| winner agreement | 0.9308 | 0.9513 | +0.0205 | [0.0103, 0.0308] |
| first-stage survivor set | 0.8518 | 0.8882 | +0.0365 | [0.0210, 0.0518] |

The trace delta's interval is entirely above zero: the promotion condition.
The decomposition shows the mechanism working as intended. Of 320 decisions
where the baseline trace disagreed with the teacher, STEF repairs 168, a
52.5 percent fix rate. Of 1,630 decisions the baseline got right, STEF breaks
93, a 5.7 percent regression rate. The fix rate is nine times the regression
rate, so the gain is concentrated in exactly the boundary mistakes the
training targeted, not spread as noise.

The trace delta, +0.0385, is nearly identical to the pair-level food gain,
+0.0397, and the survivor-set agreement moves most of the three metrics. The
repaired boundary pairs happened to move the trace by almost the same amount
as the pair metric, in this one program. That is an observation, not a rule:
the program has three axes and lexicographic reachability, so there is no
reason the two gains must coincide in general. They do land where the executor's epsilon
resolution actually decides survivor sets. STEF survives composition through
the real program executor.

## STEF deployment: latency non-inferior, calibration radius halved (research)

Two deployment measurements on the frozen seed-7 models.

Latency, head-only, local CPU (torch 2.5.1, six threads), identical hidden
states, the full three-axis scoring operation, 4,000 iterations each:

| model | mean | p50 | p95 | p99 | parameters |
|---|---:|---:|---:|---:|---:|
| baseline, three 384-wide calls | 98.8 us | 93.4 us | 143.7 us | 155.6 us | 148,225 |
| STEF, three 386-wide calls | 100.9 us | 98.2 us | 112.1 us | 153.1 us | 148,997 |

The two-neuron widening costs about two microseconds at p50 on this machine
and 772 parameters per head, 0.52 percent, and it is faster in the tail. Not
zero, but noise-level.

Latency, full field, the same Modal T4 container for both models: encode
three axis prompts plus compose, per decision, over the 1,950 test decisions.

| model | mean | p50 | p95 | p99 |
|---|---:|---:|---:|---:|
| baseline | 33,182 us | 32,736 us | 35,484 us | 44,631 us |
| STEF | 33,127 us | 32,691 us | 35,311 us | 42,423 us |

STEF is non-inferior within measurement noise and faster at p99. The two
levels were measured on different machines, so the head and field columns are
not comparable to each other, only within each level.

Calibration radius, fit on the 50-decision calibration split, all three axes
pooled: the baseline's 0.999-quantile error is 0.1814 and STEF's is 0.0913.
STEF's calibration error is half the baseline's. This number is
definition-independent and is the strongest certification evidence in the
run: the executor can certify the same relations with roughly half the
fallback radius.

Certificate rate, with an explicit definitional caveat. The run computed
certification under a survivor-pair definition: pairs among each stage's
surviving candidates, skipping stages that keep everyone. That is not the
program's operational definition, which is every relation the executed trace
actually relied on (best versus each dropped candidate, plus the final
tie-break). Under the survivor-pair definition the rates are 0.5405 baseline
and 0.6200 STEF, directionally consistent with the radius halving, but the
operational C_program is not yet computed. Per-decision rows, 1,950 with
trace and certificate flags for both models, are persisted at
vey-data/decisionmix/d2/stef/cert_per_decision.json so the recompute and the
failure-mode breakdown are local analyses, not retrains.

## Operational C_program: STEF certifies 39 percent more programs, and the trade is visible (research)

The operational definition, preregistered before the run: a decision is
certified when every comparison the executed trace relied on is certified.
Per ordinal stage, the stage-best versus every candidate the stage dropped;
a drop is certified when the model's gap exceeds the executor band plus
twice the per-axis radius, gap greater than 0.15 + 2*q_a. Survivor-vs-
survivor comparisons are not required; a multi-member final survivor set is
recorded uncertified, because the stage band caps those margins below the
certification threshold by construction. Per-axis radii fit on the
50-decision calibration split only: baseline q = 0.116, 0.181, 0.159 and
STEF q = 0.081, 0.091, 0.067 on food, open space, and headroom.

| metric | baseline | STEF |
|---|---:|---:|
| C_program | 0.2077 | 0.2882 |
| certified decisions | 405 | 562 |
| certified and trace-correct | 403 | 536 |
| certified and trace-wrong | 2 | 26 |
| certified precision | 0.9951 | 0.9537 |

Delta C is +0.0805 with a decision-level paired 95 percent interval of
[0.0687, 0.0928], entirely above zero. The absolute rate is low by
construction, as preregistered: the drop threshold needs a gap above
roughly 0.33 (STEF) or 0.51 (baseline), and multi-survivor finals are
uncertifiable by definition. The earlier survivor-pair 0.62 measured a
different, weaker question.

The precision trade is real and is the honest finding. STEF certifies 39
percent more programs, but 26 of its certified decisions disagree with the
teacher trace against 2 for the baseline: certified precision falls from
0.9951 to 0.9537. All 26 have singleton final sets and certified relations
concentrated on open space (49 relations) and food progress (28), with the
smallest certified margin at 0.001, sitting essentially on the threshold.
The per-relation radius is a 99.9 percent quantile, so roughly one relation
in a thousand can breach it; STEF certifies more relations per decision and
therefore takes more tail risk. The certificate is not weakened silently;
the added coverage costs about four points of certified precision.

The gate passes on its preregistered form, C_STEF greater than C_base with
the paired interval above zero, which the halved radius delivers. The ideal
form, precision holding while coverage grows, does not hold at this radius.
Whether a tighter quantile (99.99 percent) restores precision while keeping
most of the coverage gain is a one-line follow-up now that the per-decision
scores are persisted at
vey-data/decisionmix/d2/stef/cprogram_per_decision.json: the recompute is a
local analysis, not a retrain.

## Certificate decomposition: the failures are not threshold failures (research)

All numbers below are local analyses of the persisted per-decision rows; no
GPU, no retraining.

Per-relation, STEF's calibration claim is honest and the baseline's is not:
certified relations whose true error exceeds their per-axis radius number
4 of 3,606 for STEF (0.11 percent) and 51 of 2,986 for the baseline (1.71
percent). The baseline's food-progress relations breach their own radius at
2.07 percent. So the earlier certified-precision gap is not explained by
per-relation miscalibration.

The slack hypothesis also fails. The baseline's wrong-certified programs
carry slack 0.041 over threshold, not near zero, and STEF's spread from
0.001 to 0.071. And the sharpest finding: zero of the 26 wrong-certified
programs contain an erroneous required relation. In every one, every
certified best-vs-dropped relation is genuinely true under the teacher.

The failure channel is survivor-set divergence. Twenty-five of the 26
diverge at stage 0: the model keeps candidates the teacher dropped and drops
candidates the teacher kept, while every drop the model made is certifiably
correct. The certification rule validates drops against the model's own band
but never checks that the model's kept-set matches the band membership of the
teacher's scores. The rule is incomplete, not the field.

The risk-coverage sweep confirms the failures are threshold-invariant. With
the radius scaled by lambda from 1.0 to 4.0, STEF's certified risk plateaus
near 0.03 to 0.05 and never approaches the baseline's 0.008; no lambda
recovers baseline precision while retaining coverage. The 157 decisions STEF
certifies and the baseline does not are 84.7 percent trace-correct, so the
newly acquired coverage is genuinely good. The trust layer needs a
survivor-set relation, not a larger radius.

What this sets up: the certificate must be calibrated at the same
compositional level as the executor. A rule that certifies drops but not
kept-set membership can underwrite a program whose trace still diverges.
Program Risk Calibration belongs on the queue ahead of Snake, because Snake
fallback behavior would otherwise integration-test a trust policy with a
known blind spot.

## Survivor-Set Certificate: the two-sided rule closes the blind spot (research)

The runtime-computable two-sided rule, no teacher required. At each stage
with model scores s_i, best M, and gap g_i = M - s_i, a kept candidate is
certified when g_i + q_a <= 0.15 and a dropped candidate when
g_i - q_a > 0.15, with the per-axis radius q_a. A program is certified when
every reached stage passes both sides and the final set is a singleton.
Equivalently: no candidate may sit in the uncertainty shell
(0.15 - q_a, 0.15 + q_a) around the boundary. For STEF's radii the keep
threshold is a gap at or below 0.069 food, 0.059 open, 0.083 headroom.

The rule is exact in the sense that it proves both assertions the trace
makes: dropped candidates truly deserve dropping, and kept candidates truly
belong. Recomputed locally on the same 1,950 persisted rows:

| rule | model | coverage | certified precision | correct-certified | UCB95 |
|---|---|---:|---:|---:|---:|
| drop-only (old) | base | 0.2077 | 0.9951 | 0.2067 | 0.0178 |
| drop-only (old) | STEF | 0.2882 | 0.9537 | 0.2749 | 0.0669 |
| SSC, lambda 1 | base | 0.0000 | - | 0.0000 | - |
| SSC, lambda 1 | STEF | 0.0903 | 1.0000 | 0.0903 | 0.0214 |

All 26 wrong-certified STEF programs, and both of the baseline's, fail the
keep-side: the survivor-set diagnosis was complete, and the one-sided rule
was the entire defect. SSC's precision is perfect on these rows, 176 of 176.

The honest trade: the drop-only rule was coverage-rich and precision-weak;
SSC is precision-perfect and coverage-poor. The strict two-sided guarantee
is what it says. The baseline certifies nothing at all under the correct
rule, because its per-axis radii exceed the keep-side headroom everywhere,
so certification under SSC exists only because STEF halved the radii. The
lambda sweep (diagnostic only on these heavily examined rows) shows
coverage falling with lambda while precision holds.

The queue follows: a fresh calibration and evaluation pool, selection and
validation halves for the program-level lambda, and the preregistered PRC
gate at a 1 percent trace-error budget with a Clopper-Pearson bound. These
1,950 rows are retired from future final claims; they are design and
debugging data now.

## PRC arms: strict SSC is the certificate that ships (research)

Fresh untouched pools, drawn and labeled per a manifest recorded before any
model touched them: 1,500 selection, 5,500 validation, 500 final, all
disjoint, all outside the retired 1,950 design rows. The corpus supports only
7,500 untouched rows after the design pool, so the attachment's
2000+5500+2000 allocation was adjusted to 1500+5500+500 to protect the
validation gate. Per-axis radii refit on the design pool's calibration split.
Lambda* was chosen on selection (0.55, the smallest lambda with zero
certified failures and enough certified programs to rank), frozen, and
tested once on validation. Exact one-sided Clopper-Pearson bounds; the
budget is 1 percent.

| arm | validation coverage | certified | wrong | risk | CP-UCB95 | final coverage | final wrong |
|---|---:|---:|---:|---:|---:|---:|---:|
| A drop-only | 0.3222 | 1,772 | 104 | 0.0587 | 0.0752 | 0.3420 | 15 |
| B strict SSC | 0.0835 | 459 | 0 | 0.0000 | **0.0065** | 0.0820 | 0 |
| C PRC, lambda 0.55 | 0.1369 | 753 | 1 | 0.0013 | 0.0129 | 0.1480 | 1 |

The gate has a clean verdict. Arm B passes: zero failures in 459 certified
programs, upper bound 0.0065, well under the 1 percent budget, and it
reproduces on the untouched final pool, 41 certified with zero wrong. Arm C
misses: one failure among 753 certified puts its upper bound at 0.0129.
The single failure is exactly the shape the pool arithmetic predicted: at
this sample size the gate tolerates zero failures, and one lands at the
boundary. Per the preregistered protocol, lambda was frozen before
validation and the miss stands; no retuning.

Arm A's numbers confirm the blind spot scales rather than shrinks on fresh
data: 104 wrong-certified at 5.9 percent risk on validation, worse than the
design pool's 26 at 4.6 percent, because the one-sided rule certifies more
as it reaches farther.

The trust layer is frozen on strict SSC. Certificate mode "strict" is the
shipping rule: lambda 1, two-sided, singleton final. The PRC relaxation
demonstrably buys coverage, 64 percent more than strict SSC, but its risk
control is not tight enough to certify at the 1 percent budget on this
data; if a future pool supports a looser budget, mode "risk" can return,
but it earns nothing at 1 percent today.

Per-decision rows for all three arms and pools are persisted under
vey-data/decisionmix/d2/stef/prc/ with the split manifest. Trust-layer
architecture is closed: relation calibration, then the survivor-set
certificate, both two-sided and runtime-computable.

## Seed replication: seed 11 passes both gates (research)

The frozen shared recipe, seed 11, zero tuning. Evaluation on the retired
1,950 design rows, whose role is robustness replication.

| gate | seed 7 | seed 11 |
|---|---|---|
| trace delta above zero | +0.0385 [0.0226, 0.0549] | +0.0718 [0.0554, 0.0887] |
| fix rate above regression rate | 52.5% / 5.7% | 56.1% / 4.8% |
| trace agreement, base to STEF | 0.8359 to 0.8744 | 0.8036 to 0.8754 |

Both gates pass with a larger margin than seed 7. STEF lands at essentially
the same trace agreement from a weaker starting point, which is itself
informative: the mechanism converges to the same place from a different
initialization rather than riding one lucky basin.

Strict-SSC coverage is recorded, not gated, and it is zero for this seed
against seed 7's 0.0903. The cause is arithmetic, not a defect: seed 11's
per-axis radii are wider (0.0999/0.0977/0.0546 vs 0.0811/0.0913/0.0671), the
uncertainty windows around the executor band are wider with them (0.200 and
0.195 on food and open space against 0.162 and 0.183), and a 5-candidate
decision almost always puts at least one candidate gap inside a window that
wide. SSC coverage is radius-sensitive by construction; a decision that
clears the window on one seed may not clear it on another at equal accuracy.
That variance is the certificate's honest operating characteristic, not a
regression, and it is the number to track when the field's calibration
improves further.

## Seed replication: seed 13 passes; three seeds, three passes (research)

Seed 13, same frozen recipe, same retired evaluation rows.

| gate | seed 7 | seed 11 | seed 13 |
|---|---|---|---|
| trace delta | +0.0385 [0.0226, 0.0549] | +0.0718 [0.0554, 0.0887] | +0.0369 [0.0241, 0.0497] |
| fix vs regression | 52.5% / 5.7% | 56.1% / 4.8% | 40.6% / 2.8% |
| base to STEF trace | 0.8359 to 0.8744 | 0.8036 to 0.8754 | 0.8497 to 0.8867 |

All three seeds pass both gates. The fix-to-regression ratio is at least
nine to one on every seed and reaches 14.3 to 1 on seed 13, whose baseline
is the strongest of the three. STEF's final trace agreement lands in a
narrow band, 0.874 to 0.887, from baselines spanning 0.80 to 0.85, so the
mechanism converges to the same operating point from different
initializations. The trace delta varies with baseline strength, as expected:
the strongest baseline leaves the least to repair and still gains 3.7
points. Seed 7 was not a lucky basin.

Strict-SSC coverage is zero for seeds 11 and 13, matching the radius-window
arithmetic recorded for seed 11; seed 13's radii (0.102/0.099/0.053) put its
food window at 0.204. Certificate coverage at the current field
calibration remains seed-sensitive; the accuracy mechanism does not.

Replication is complete. What remains for Vey 2 is Snake integration with
the frozen system: shared STEF, strict SSC, the real executor, trace
agreement primary.

## Snake integration protocol, preregistered before measurement (research)

The final Vey 2 gate. Recorded before any Snake measurement, so the numbers
below this section cannot influence the protocol.

Model: the seed-7 shipping model, the one whose strict SSC was actually
validated on the fresh DecisionMix pools. Seeds 11 and 13 were replication
runs; selecting one of them after seeing its trace score would be post-hoc
model selection, and their certificate geometry differs. Seed 7 only.

Dataset: 800 recorded Snake states, 200 each from seeds 101-104, generated
by driving the deterministic planner through the demo game. The trajectory
depends on the planner alone, so baseline and STEF are scored on identical
rows. Candidate texts are the demo's own descriptions; the instruction is
the demo's own question, "Choose the move with the most food progress, then
the most open space", which compiles to two stages, MAX food progress then
MAX open space. Headroom is not queried.

Teacher: the shipped field scorer on those candidate texts, the same teacher
every prior trace measurement used. The planner's move is a separate
secondary agreement metric, not the trace teacher.

Fixed and not touched after results: compiler, executor, epsilon 0.15,
candidate order, fallback logic, per-axis radii refit on the retired design
pool's calibration split exactly as in the replication runs. Compose traces
are normalized at the boundary, once, before anything consumes them.

Primary gate, integration rather than efficacy: trace delta at or above
zero, and no meaningful latency regression. A confidence interval whose
lower bound clears zero is the strong close but is not required; this run
was preregistered as non-regression, and raising the bar after the fact
would be the same post-hoc move the protocol exists to prevent.

Reported alongside, not gated: move agreement with the planner, baseline
fixes versus regressions, strict-SSC coverage, precision among certified
decisions, the uncertified rate, and p50/p95/p99 latency for both fields on
the same machine.

The existing 1 percent SSC risk bound does not transfer. It was measured on
the DecisionMix validation distribution, and Snake was never part of that
exchangeability assumption. SSC numbers on Snake are an integration
measurement. Zero certified cases would mean the certificate declined to
transfer, not that Snake failed.

Live games are a secondary behavioral demonstration only, run after the
replay gate. Once the two fields choose differently their future states
diverge, so food, deaths and score cannot be the scientific gate.

## Snake integration: winner transfer survives, band reproduction collapses (research)

Run as preregistered above, then corrected once. The first recorded version
of this section concluded that the heads do not track the field on Snake
text at all. That conclusion is false, and it is false on the numbers that
version itself committed. Inclusion-exclusion on its own table gives
P(baseline winner = teacher winner) at least 0.7462 + 0.7588 - 1 = 0.505,
and the same for STEF at least 0.478, against a 0.25 winner chance. The
protocol asked for move agreement with the teacher and the first run
substituted planner agreement, which cannot adjudicate tracking. This
section replaces that one with the direct measurement.

Seed 7, radii frozen at the validated refit rather than recomputed:
0.0811/0.0913/0.0671 for STEF and 0.1155/0.1814/0.1588 for the baseline,
each reproduced exactly on the two earlier runs. 800 planner-driven states,
seeds 101-104, identical rows. The question compiled to the two expected
stages. The winner used here is compose's winner, the extreme on the first
ordinal stage with the content tie-break, not the first id of the final
survivor band. Per-decision rows are persisted.

| metric | baseline | STEF | teacher |
|---|---:|---:|---:|
| trace agreement | 0.0000 | 0.0000 | — |
| trace delta | 0.0000 [0.0000, 0.0000] | | |
| winner agreement with teacher | **0.8375** | **0.7725** | — |
| planner move agreement | 0.7462 | 0.7188 | 0.7588 |
| strict SSC coverage | 0.0000 | 0.0000 | — |
| head-only latency p50 / p95 / p99, ms | 0.355 / 0.385 / 0.435 | 0.255 / 0.278 / 0.339 | — |

Winner agreement is more than triple the 0.25 chance floor. The heads do
track the teacher's top candidate. The latency bound, stated before this
rerun as STEF p95 at or below 1.2 times the baseline's, holds: 0.278 against
a ceiling of 0.462. The latency is head-only on the two queried axes, over
precomputed embeddings; the encoder pass is shared and excluded.

The trace agreement stays exactly zero, and the full-ordering diagnostic
explains the gap between a correct winner and a wrong trace. Full
candidate-ordering agreement with the teacher, reproduced exactly from the
first run:

| axis | baseline | STEF | chance (4 candidates) |
|---|---:|---:|---:|
| food progress | 0.0300 | 0.0300 | 0.0417 |
| open space | 0.0925 | 0.0312 | 0.0417 |

Conditional on the winner being correct, the full food-progress ordering is
still right only 0.0358 of the time for the baseline and 0.0388 for STEF.
The top candidate transfers and the ranks below it do not. The open-space
baseline cell at 0.0925 is above its chance floor, which the first version
of this section reported and then contradicted.

The corrected reading is band sensitivity, not register blindness. Snake's
candidate texts carry small field gaps: a sampled state shows a 0.001 gap
between two near-identical candidates and a 0.135 food-progress gap sitting
against the 0.15 band edge. Exact full-order reproduction and epsilon-band
membership are both brittle at that scale, while the identity of the top
candidate is not. The text-register difference is real, median 174 characters
of prose against median 85 of terse consequence strings, but it does not
explain a student that finds the teacher's winner 84 percent of the time.

What transfers and what does not, stated at the resolution the data
supports. Winner-level transfer survives the demo's text. Full-order
reproduction and epsilon-band reproduction collapse, which is why every
trace disagrees and why the strict certificate certifies nothing. The
certificate's refusal is still read as non-transfer of the 1 percent bound,
per the protocol, and not as a Snake failure.

Two findings, not one, and only one of them has a candidate cause. The trace
and certificate failure is a band-membership failure: both fields score the
teacher's top candidate and neither reproduces the epsilon survivor sets.
Calibration at the band boundary could address that and nothing else.

The other finding is a reversal, and band calibration cannot explain it.
Winner agreement is 0.8375 for the baseline and 0.7725 for STEF, so STEF is
6.5 points worse at the decision itself, before any band is consulted. The
marginals alone put the exact McNemar upper bound near 0.004, so this is not
sampling noise on 800 rows. The fold is not the suspect: the exact fold is an
algebraic identity, and unless that identity fails numerically on these
hidden states the statement is that the STEF-trained function generalizes
worse than the baseline function on Snake top-1. That is a property of the
trained function, not of the deployment conversion.

The reversal is undiagnosed. The per-decision rows this section previously
called persisted were written to /tmp and are gone; the commit that claimed
persistence touched only this file. Reproducing them needs the training rows,
and those are also gone. The recipe recorded for them, a seed-0 shuffle cut
at 2,500 and a seed-17 90/10 group split, rebuilds to the right count but
trains to a scalar epoch-0 loss of 0.0267, which none of the recorded runs
produced, and a model trained on it scores Snake at 0.8712/0.8250 rather than
0.8375/0.7725. Those rows were discarded rather than analyzed as if they were
the recorded model. Until the training rows are recovered or the run is
redone from a pinned split, the discordant-pair counts, the tie-robust
top-set agreement, and the margin split of the 52 net losses are not
computable, and no fix should be chosen.

What is durable: the 800 recorded states, regenerated from the planner and
seeds 101-104, at
vey-data/decisionmix/d2/snake/states.json, sha256 289a4694. Of them, 117 have
byte-identical candidate texts, 33 because two or more moves are illegal and
share one fixed description, 84 because legal moves quantize to the same
percentages. The full-order chance floor of 0.0417 and the exact-winner
metric are both tie-fragile on those rows, and the tie-robust top-set metric
is the one to lead with once scores exist again.

The record as it stands, and no wider: on DecisionMix, STEF beats the
baseline on all three seeds. On Snake top-1, both transfer and the baseline
wins. On Snake survivor topology, neither transfers. On Snake, the
certificate abstains entirely. The open question is why the repair that helps
on DecisionMix costs top-1 robustness here. It is unanswered, and it blocks
locking Vey 2.

## Pinned Snake transfer replication, preregistered before training (research)

The 0.8375 to 0.7725 reversal came from a training split that no longer
exists, and a reconstruction that matched its row count trained a different
model. Matching those two numbers is not the goal and would be accidental if
it happened. This study replaces that run. It is written down before the
split is drawn and before any model trains.

Split, defined here and then hashed. Source is the d6 generic train corpus,
12,000 rows. Shuffle with seed 0. Take the first 2,500 rows. Group by the
option-tuple key; a group is complete only when all three axes are labeled.
Shuffle the complete groups with seed 17. Cut at 90 percent: the first part
is train, the remainder is dev. The dev rows are for the in-distribution
trace check only and are not used to choose anything. Teacher labels come
from the shipped field scorer, revision 3eb1460a82c98fc99c350032b9cf29a74075a4a6.
Encoder is microsoft/deberta-v3-xsmall at the revision pinned by
transformers 4.48.0. Persisted before training, with sha256 of the canonical
JSON: train rows, dev rows, the id lists, the manifest, and the Snake states
already at vey-data/decisionmix/d2/snake/states.json (sha256 289a4694, full
hash 289a4694f756246955c784c7f8a6b0dadb1c35c343a628cec6c3a4d48176c44e). The
manifest records this commit, the seeds, and the algorithm above. If a
rebuild does not reproduce the train hash, the rebuild is wrong.

Models. The frozen recipe, unchanged: six layers, three epochs of pooled
scalar training, two epochs of the shared residual, the exact antipodal
fold, no tuning. Seeds 7, 11, and 13, because the question is now whether
the reversal is a property of the recipe or of one realization. Each seed
persists its checkpoint and its per-decision Snake outputs under
vey-data/decisionmix/d2/snake/. Nothing is deleted after the run.

Primary metric. 117 of the 800 states have byte-identical candidate texts,
so the exact-winner comparison is tie-fragile. The teacher top set is
T(x) = {i : t_i = max_j t_j} on the deciding axis, food progress. A model is
correct when its compose winner is in T(x). Per seed, report both rates and
their difference, STEF minus baseline, with a paired bootstrap interval and
the exact discordant counts: baseline correct and STEF wrong, against
baseline wrong and STEF correct, with the exact McNemar probability. The
chance rate is the mean of |T(x)|/K over states, not 0.25.

Reported, not gated: exact compose-winner agreement, trace agreement, strict
SSC coverage, full-order agreement as a diagnostic only, head-only latency
on the two queried axes, and the teacher top margin. The margin bins are
[0, 0.01), [0.01, 0.05), [0.05, 0.10), [0.10, infinity), and the per-bin
difference says whether any loss sits at microscopic margins or takes
large-margin winners too.

Fold parity, amended before the seed-7 rerun. The first seed-7 run measured
a maximum absolute score gap of 5.8e-2 between the split head and the folded
head and aborted. That gap equals the mean of the residual over the state's
candidates: the training centers the residual per candidate set and the fold
does not, so the fold shifts every candidate in a state by the same constant.
A constant shift changes no pairwise difference, so it cannot move a winner,
a top set, a trace, or a margin. The decision-relevant quantity is the
maximum change in any pairwise candidate difference, and the gate applies to
that. It must be floating-point noise. The absolute gap is reported and does
not stop the run. This correction is algebraic rather than empirical: it
holds at any measured magnitude, which is why it is being made after seeing
the 5.8e-2 and not tuned against it.

Lock rule, fixed now. If the difference is at or above zero on most seeds,
or the pooled interval contains zero with no consistent negative direction,
the historical reversal stands as a result from an unrecoverable split and
is not reproduced, and Vey 2 locks with Snake topology transfer left open.
If the difference is negative on the seeds and the pooled interval lies
below zero, Vey 2 stays open and the problem is named as stated: the recipe
improves in-distribution topology and harms out-of-distribution top-1. No
fix is designed before that result exists.

## Pinned Snake transfer: the reversal does not reproduce, Vey 2 locks (research)

Run as preregistered above. Train split sha256
257a9489858db9b83c590d7df674d0c725c05f575466235abbd5073d42cc5309, dev
sha256 1621dd6b6cc95fa3baa4f656e163afb2e0b18e941cfb5165c0c212ce901f54a4,
states sha256 289a4694f756246955c784c7f8a6b0dadb1c35c343a628cec6c3a4d48176c44e.
Row files persisted and hashed under vey-data/decisionmix/d2/snake/:
seed 7 3f9fb982a5730656736cd4e1858aec4ace9b7005297df02a1db0c1bad7ba2a29,
seed 11 f4117303db41b42e8b33dd83d300a20838d59d0b29bbe292230f38d2a77632c7,
seed 13 a58acb9a63bb5756f5d8306e62e991863d4589b1155ce050065127764ef30aa4.
Checkpoints alongside.

Fold parity held on all three seeds as pairwise gaps 4.8e-8, 5.2e-8, 6.0e-8,
floating-point noise. The absolute gaps, 5.8e-2 to 7.3e-2, are the per-state
residual mean and move no decision.

Tie-robust top-set agreement on the deciding axis, chance 0.2553:

| seed | baseline | STEF | delta | CI | n10 / n01 | p |
|---|---:|---:|---:|---|---|---:|
| 7 | 0.8588 | 0.8712 | +0.0125 | [+0.0050, +0.0212] | 1 / 11 | 0.0063 |
| 11 | 0.1200 | 0.8387 | +0.7188 | — | 0 / 575 | 0.0000 |
| 13 | 0.9038 | 0.8912 | −0.0125 | [−0.0224, −0.0037] | 10 / 0 | 0.0020 |

Seed 11's baseline scored 0.1200, below the 0.2553 chance rate, with a normal
loss curve across all three training epochs. That is a failed baseline
generalization on this text, not a STEF gain, so it is excluded from the pool
and reported as itself. Its 575-to-0 discordant split is one seed's baseline
failing on nearly every decision, not a mechanism effect.

Over the two healthy seeds, 1600 pairs: baseline 0.8812, STEF 0.8812, delta
+0.0000, CI [−0.0056, +0.0056], n10 11, n01 11, exact McNemar p 1.0000.
Perfectly balanced.

Margin behavior on the healthy seeds: every disagreement sits below teacher
margin 0.01, both models score 1.0000 at margin 0.01 and above, and seed 7's
small-margin losses exactly offset seed 13's. The only place STEF and the
baseline differ on Snake is at microscopic teacher margins, and there they
cancel.

The lock rule, fixed before these numbers existed, is met on its first
branch: the pooled interval contains zero, symmetric, with no consistent
negative direction. Vey 2 locks. The historical 0.8375 to 0.7725 reversal is a
result from an unrecoverable training split, not reproduced, and not evidence
against the mechanism.

What stays open, named precisely. Trace agreement on Snake is 0.0000 on seeds
7 and 13 and 0.1238 on seed 11's STEF: the students find the teacher's winner
as well as the baseline does and do not reproduce its epsilon bands, so the
survivor topology does not travel out of distribution and the strict
certificate correctly refuses everywhere (coverage 0.0000 on all three
seeds). Restricted-candidate evals keep the trace-agreement requirement;
unrestricted production runs deploy behind the fallback chain. This is an
out-of-distribution fidelity limitation at the band level, recorded as the
open limitation of Vey 2, and the signal by which future Laya versus
frozen-Vey decisions are judged.

## BSC: band-scale calibration, preregistered before fitting (research)

First post-Vey-2 experiment. Vey 2 stays frozen at vey-2-final; this branch
is post-v2/bsc. Nothing here modifies the locked reference. Recorded before
any alpha is fitted or any new Snake episode is scored.

Form: s'_{i,a} = alpha_a s_{i,a}, alpha_a > 0, one scale per axis. A positive
scale preserves argmax exactly, so winner behavior is invariant by
construction; the gates assert it in code anyway.

Data: new Snake episodes only, generated by the same deterministic planner
loop as the locked study. Locked seeds 101-104 are never used. BSC fits on
calibration seeds 201-202, verifies on validation seeds 211-212, and reports
on final seeds 221-222, 200 states per seed.

Fit: alpha_a = argmin over alpha > 0 of sum_r w_r (alpha d^S_r - d^T_r)^2,
with r over decision candidate pairs from the calibration pool and
w_r = exp(-((|d^T_r| - 0.15)^2 / (2 * 0.05^2))). An unweighted least-squares
control runs alongside; if weighting does not beat it on calibration trace
agreement, the weighting is rejected in favor of unweighted.

Gates, all measured on validation; the final pool is reported untouched:

1. Winner invariance: identical argmax per state, asserted in code.
2. Trace agreement strictly above the frozen baseline.
3. Strict SSC coverage strictly above frozen, radius refit on calibration.
4. The strict risk rule holds on validation at the fitted radius.
5. End-to-end production p95, full chain including the fallback path, is
   below the frozen chain's p95 on the same machine. Head-only latency does
   not count.
6. Persisted: fraction certified, fraction falling back, fallback latency,
   total expected latency, certified-versus-fallback trace correctness.

Hard rule: if topology improves but full-chain latency does not drop, it is
a fidelity result, not a speed result, and does not earn the speed claim.

## BSC result: fidelity improves, the certificate still refuses (research)

Fit and gates as preregistered above, on the persisted frozen seed-7 scores
of the three new pools (400 states each, seeds 201-202/211-212/221-222).

The weighting earned nothing: weighted and unweighted fits gave identical
calibration and validation trace agreement, so the unweighted fit ships.
Alpha 3.7517 on food progress, 1.5803 on open space, applied to every number
below.

| pool | frozen trace | BSC trace | frozen winner | BSC winner |
|---|---:|---:|---:|---:|
| validation | 0.0000 | 0.1200 | — | — |
| final, untouched | 0.0000 | 0.0650 | 0.8575 | 0.8575 |

Gate verdicts. Winner invariance holds exactly, asserted per state on both
pools. Trace agreement improves on both, from exactly zero to 0.1200 and
0.0650. Strict SSC coverage is 0/400 on validation for frozen and 0/400 for
BSC: gate 3 fails. The cause is structural. With the calibration-refit
radius 0.083 on food, the keep-side window is 0.067 wide, and 178 of 308
clean validation states carry two stage-1 survivors whose second sits
outside the window while remaining inside the band. The student's Snake-text
residual at the 0.999 quantile is 0.083, over half the band itself, and a
positive per-axis scale cannot fix that: scaling widens gaps and windows
symmetrically. Duplicate-text states are unfixable by any calibration since
the inputs are byte-identical; they are 100/400 of calibration, 92/400 of
validation, 81/400 of final. A clean-state-only refit (alpha 3.6963 and
1.5215, radii 0.0834 and 0.1681) certifies 0/308 clean states at stage 1.

Because coverage is zero under both, the production chains are identical,
so the latency gate cannot improve and no speed claim is earned. BSC is
recorded as a fidelity result only, per the hard rule.

The finding at its resolution: the Snake epsilon-topology deficit is two
problems, and BSC addresses exactly one. The geometry problem, student gaps
systematically too small relative to epsilon, is calibratable, and
calibration moves trace agreement off zero on fresh episodes without moving
a single winner. The certificate problem is structural: the two-sided strict
rule needs residuals far smaller than half the band, and this student does
not have them on Snake text. What could change that is a better student
(smaller radius), a per-pair or per-state radius, or relaxed strictness on
this domain. Each is a new preregistered experiment, none is BSC, and BSC's
alphas are recorded here so any successor starts from them rather than from
scratch.

## CBF-0: criterion basis field, preregistered before any corpus (research)

Second post-Vey-2 experiment, branch post-v2/cbf, Vey 2 frozen at
vey-2-final. Recorded before the criterion corpus exists.

Model: s(x, q) = c(q)^T z(x), z and c both in R^m, m in {32, 64}. No
cross-attention, no candidate interaction, no slots, no residual branches,
no per-criterion heads. z(x) and c(q) come from a shared frozen text
encoder followed by one linear probe each.

Question: can one shared field score candidates for criteria it never
trained on.

Split, the critical part. Every paraphrase of a semantic criterion is one
group. Three levels: seen criterion with unseen wording at test time
(sanity), unseen criterion within a family seen in training (primary
development), entire families held out of training (hard transfer).
Families: preference, risk, urgency, relevance, suitability, quality, cost,
safety, compliance, plausibility, similarity, sentiment, intent-routing.
If "low financial risk" is test-only, "safer financially" and
"minimize financial exposure" are absent from training.

Gates, all on held-out-criterion levels, paired 95% CI. G1: CBF top-1 beats
the frozen-embedding cosine baseline between criterion and candidate text.
G2: positive transfer across the held-out families, per family. G3: exact
candidate-permutation invariance, asserted in code. G4: m=64 versus m=32;
a tie ships 32, and 64 must earn its cost.

Persisted per split: top-1 accuracy, pairwise agreement, Kendall/Spearman
rank agreement, rho(s, t) where the teacher is continuous, binned by
K in {2, 4, 8, 16}. Criterion-direction cosine cos(c(q1), c(q2)) is a
diagnostic only; no regularizer without evidence pointing there.

Failure reading fixed in advance: clearing unseen wording but not unseen
criteria is recorded as "learned criterion interpolation, not arbitrary
criterion generalization," and the next step is diagnosis, not capacity.
B-STEF waits until CBF passes on genuinely unseen criteria.

### CBF-0 validity correction, before the corrected fit

The first procedural corpus and timed-out ALS run are invalid, not CBF
evidence. Its in-family queries were training paraphrases with an
"unseen in-family" prefix. Its gold rank broke ties by candidate position,
and several criteria shared targets despite opposite meanings. The reported
m=32 gains cannot establish unseen-criterion transfer. Those artifacts are
retained separately as invalid-v0; they are not used below.

The corrected experiment is a synthetic numeric multiattribute test, not a
Jev/Laya parity claim. Each family has four distinct attributes/criterion
groups with four paraphrases. Ten families train on groups 0/1 and hold out
groups 2/3 completely. Compliance, similarity, and intent-routing are entirely
held out. Sanity evaluation uses forms 2/3 of seen groups, while training uses
forms 0/1. Every candidate describes all four independent attributes. A
criterion scores only its named attribute, with a fixed positive or negative
direction. Scores are normalized attribute values; neither gold nor text
depends on candidate position. Identical scenarios are queried under several
criteria, so ignoring the criterion cannot recover all winners.

K is 2/4/8/16. Train uses eight scenarios per K/family; each evaluation level
uses four, with disjoint deterministic scenario seeds. IDs, criteria, rows,
encoder revision, and source hashes are persisted before fitting.
The frozen full DeBERTa-v3-xsmall uses safetensors revision
eb2d654bf0a5b628c8be6c4be7d29118fbef95b8, normalized mean pooling, and
independent FP32 text encoding. The first cache inherited half precision
from the upstream weights and the head's first forward rejected its dtype;
that cache is retained but excluded. No head update occurred before the
explicit FP32 correction. Only two bias-free linear maps are trained:
m=32/64, seed 7, AdamW, learning rate 0.001, weight decay 0.0001,
200 epochs, batch size 128, squared error against scores in [0,1].
This replaces the impractical dense ALS solver; no outcome-driven tuning.

G1 requires a criterion-group-paired 95% bootstrap lower bound above zero
against frozen cosine similarity on each genuinely unseen level. Bootstrap
uses 10,000 draws, seed 0, resampling whole criterion IDs equally.
G2 additionally requires positive accuracy deltas on at least two of the
three held-out families and a positive hard-transfer macro delta.
G3 requires bitwise-equal scores mapped back through reversed and seeded
candidate permutations at every K, not approximate sorted-score equality.
G4 selects 64 only if its unseen-criterion advantage over 32 is at least
one percentage point with paired CI above zero; otherwise 32 remains the
next-experiment candidate. Hard-transfer regression vetoes promotion of 64.
No dimension earns promotion if G1/G2 fail.

Reports retain per-row scores, tie-aware ranking/correlation metrics,
family/K bins, criterion-direction cosines, and a criterion-blind scoring
control. Undefined correlations are counted and represented as null.
B-STEF and replication remain blocked until this corrected CBF-0 passes.

### CBF-0 corrected result: neither dimension passes

Recipe commit `6c721ee`, seed 7, both fixed 200-epoch fits completed.
This is a synthetic numeric-attribute screen, not a natural-domain arbitrary
criterion demonstration. Frozen Vey 2 was not changed.

| evaluation | decisions | cosine top-1 | CBF-32 top-1 | CBF-64 top-1 | 32 minus cosine, paired 95% CI |
|---|---:|---:|---:|---:|---:|
| seen criteria, unseen wording | 640 | 0.271875 | 0.281250 | 0.296875 | +0.009375 [-0.046875, +0.065625] |
| unseen criteria, known families | 1280 | 0.262500 | 0.292188 | 0.290625 | +0.029688 [-0.039844, +0.103145] |
| wholly held-out families | 768 | 0.279948 | 0.273438 | 0.269531 | -0.006510 [-0.091146, +0.078125] |

CBF-64 versus cosine also fails the unseen gates: known-family criterion
delta +0.028125, CI [-0.041406, +0.104707]; held-family delta -0.010417,
CI [-0.084635, +0.061198]. Even wording-only improvement is not established:
the CBF-64 delta +0.025000 has CI [-0.031250, +0.084375].
Thus neither arbitrary-criterion generalization nor a reliable
wording-interpolation improvement is demonstrated.

CBF-32 ranking and continuous-score diagnostics:

| evaluation | pairwise | Spearman | Kendall tau-b | Pearson |
|---|---:|---:|---:|---:|
| unseen wording | 0.576977 | 0.206414 | 0.153955 | 0.232386 |
| unseen criterion | 0.551800 | 0.134717 | 0.103599 | 0.145861 |
| held family | 0.534621 | 0.083059 | 0.069243 | 0.086203 |

CBF-32 top-1 by candidate count:

| evaluation | K=2 | K=4 | K=8 | K=16 |
|---|---:|---:|---:|---:|
| unseen wording | 0.506250 | 0.287500 | 0.218750 | 0.112500 |
| unseen criterion | 0.521875 | 0.362500 | 0.178125 | 0.106250 |
| held family | 0.536458 | 0.296875 | 0.156250 | 0.104167 |

Both dimensions improve on two held families, but compliance regresses
enough that aggregate hard transfer is negative:

| held family | cosine | CBF-32 | CBF-64 |
|---|---:|---:|---:|
| compliance | 0.367188 | 0.207031 | 0.226562 |
| intent-routing | 0.265625 | 0.316406 | 0.316406 |
| similarity | 0.207031 | 0.296875 | 0.265625 |

The criterion-blind control averages the training criterion directions
and otherwise uses the same learned candidate map. Its held-family top-1
is 0.307292 at 32 and 0.343750 at 64, above the corresponding conditioned
models. On unseen criteria in known families it is worse (0.246875 and
0.218750). Query conditioning has mixed effects; robust transfer is not
established. No causal account of this failure is claimed.

Mean criterion-direction cosine within/across families is 0.873416/0.723193
at 32 and 0.785976/0.555222 at 64. These are diagnostics of the averaged
four-wording directions, not a gate or evidence of useful decisions.

G1 and G2 fail at both dimensions. G3 passes: reloaded checkpoint scores
exactly match all 2688 saved decisions, with 10,752 mapped-winner
permutation checks, zero predicted top ties, and 8064 candidate-subset
checks. G4 rejects 64: its unseen-criterion delta versus 32 is -0.001563,
CI [-0.025000, +0.019531], and hard-transfer accuracy also decreases.
The projection parameter counts are 24,576 versus 49,152. This is a
parameter-cost comparison, not a latency or production-speed claim.
32 is only the smaller failed experimental configuration; nothing ships.
No replication or B-STEF was launched.

All per-decision scores and metrics, both dimensions and controls, every
family/K bin, split/source/cache hashes, checkpoints, and direction cosines
are retained under
`/home/fazinahamed/Documents/vey-data/decisionmix/d3/cbf/corrected-v1/`.
Git records the full metric and hash inventory in
`research/cbf0/result_manifest.json`; weights and datasets stay outside Git.
The invalid original corpus/scripts and rejected FP16 cache remain
separately marked and are not evidence for this result.

Key SHA-256:

- rows: `0fecbf151bdf44507c97f7a359eb713ebdf208f6e17c4bf5c7ba6ce1e626769e`
- predictions: `a3ff7c1f5a088d25962c316fe505c9fc9a258824c67c6ae6673f1f8f8363a0a5`
- results: `97723ece8d7e0c36975c3ca362719d8ca6560a885b404787e3ae7575e6be96af`

### CBF-1: oracle factorization and criterion-swap audit, preregistration

CBF-0 failed; B-STEF remains prohibited. CBF-1 reuses its exact rows,
splits, frozen FP32 embeddings, and CBF-32 checkpoint. No new examples,
encoder pass, widths, layers, attention, heads per family, or extra epochs.

The actual generator assigns every criterion a signed preference on one
numeric attribute. Compliance has no thresholds; similarity has no
reference-vector distance; intent routing has no categorical label.
Those names must not be mistaken for richer tasks the corpus does not
contain. Oracle numeric features are the family's four values divided by
100, in fixed catalog order, followed by constant 1, padded to width 32.
Oracle criterion features are a signed one-hot on the selected attribute,
with constant coordinate 1 for minimizing criteria and 0 otherwise.
Their dot product is exactly value or 1 minus value. This basis is valid
for the generated same-family candidate sets, not a universal semantics
claim for mixed-family sets or natural-language criteria.

The four cells are oracle/oracle, text candidate/oracle criterion,
oracle candidate/text criterion, and the unchanged full CBF-32.
The hybrids each fit one bias-free linear map from the frozen text
embedding, using the original flattened train pairs, seed 7, AdamW
0.001/0.0001, batch 128, 200 epochs. The loss is vector MSE on the five
active coordinates. Other padded coordinates stay zero. Candidate targets
expose all four numeric values on existing training examples; criterion
targets expose the signed selection vector only for existing train queries.
Privileged oracle supervision makes these diagnostic probes, not deployable
models or objective-parity competitors to CBF-0.

Gate 0 requires perfect top-1/pairwise on every source row and maximum
score error at most 1e-12. A hybrid is considered adequate only if top-1
and pairwise are at least 0.95 on both genuinely unseen splits and each
held-out family. Baseline-paired CIs, seen-wording metrics, family/K bins,
training losses, and per-coordinate reconstruction errors are also retained.
Failure means the tested fixed-budget map is inadequate, not that
information is absent from the frozen embedding.

An identification caveat is pinned before fitting: train criteria select
only attribute slots 0/1. The criterion probe has no target support for
slots 2/3, unlike the densely supervised candidate probe. Report the
training target span, out-of-span criterion directions, and held-family
supported/unsupported-direction subsets. An unsupported-direction failure
cannot uniquely diagnose a neural criterion encoder defect.

For causal swaps, retain every ordered pair of distinct semantic criteria
on the same evaluation candidate set, including all wording combinations.
CRA is the probability that student winner-change equals teacher
winner-change. Separately report teacher-change counts, student change
given teacher change, change to the new teacher winner, correctness of
both endpoints, and student change on teacher-stable pairs. The old
criterion-blind controls must have zero student changes. Persist per-swap
rows plus split/family/K denominators; no model fitting uses these swaps.

Decision tree: oracle failure invalidates this factorization; candidate
hybrid failure identifies an inadequate tested candidate map; candidate
success with criterion failure points to criterion mapping only subject to
the basis-support caveat; two successful hybrids with full CBF failure are
consistent with joint alignment/optimization failure, not proof of a unique
cause. If both hybrids fail, neither side is adequate and there is no
uniquely identified criterion bottleneck. The machine protocol is
`research/cbf0/audit_protocol.json`; evidence stays in
`vey-data/decisionmix/d3/cbf/audit-v1/`.

### CBF-1 result: exact factorization, two inadequate text mappings

Recipe `ca6a8ef`; checkpoint-only analysis `e2a9aef`. Both 200-epoch
fits completed once. The first scoring pass failed when a NumPy boolean
reached JSON serialization. Native-boolean conversion and a scoring-only
resume fixed delivery; neither checkpoint was retrained. Only the five
printed epoch-loss observations per arm survived that failure, not the
complete per-epoch history. Exact frozen-model coordinate errors are
persisted separately.

Gate 0 passes on all 3968 source rows: maximum score error **0.0**,
top-1 and pairwise **1.0** throughout. The task does admit the pinned
dot-product factorization. Compliance's CBF-0 collapse is not caused by
an unrepresented threshold function: this corpus has no such function.
This does not validate bilinear representation of real compliance,
similarity, or routing tasks.

| evaluation | oracle/oracle | text candidate, true criterion | exact candidate, text criterion | full CBF-32 |
|---|---:|---:|---:|---:|
| training rows | 1.000000 | 0.421875 | 1.000000 | 0.480469 |
| seen criteria, unseen wording | 1.000000 | 0.421875 | 0.460938 | 0.281250 |
| unseen criteria, known families | 1.000000 | 0.434375 | 0.239844 | 0.292188 |
| wholly held-out families | 1.000000 | 0.380208 | 0.363281 | 0.273438 |

Neither hybrid meets the preregistered 0.95 adequacy gate. The candidate
probe is already poor on its training rows despite receiving all four
numeric targets: attribute-coordinate training MSEs are
0.058326/0.057576/0.061540/0.058492. It preserves some usable structure:
versus cosine, unseen-criterion top-1 delta is +0.171875,
paired CI [+0.110938, +0.239844]; held-family delta is +0.100260,
CI [+0.007813, +0.200521]. These relative gains do not make the
candidate representation adequate.

The criterion probe fits train queries (top-1 1.0; final frozen vector
MSE about 0.00000452), but does not transfer reliably even to unseen
wordings of those criteria. Its held-family delta against cosine is
+0.083333, CI [-0.020866, +0.194010].

Training criterion targets span only **3** of the five active dimensions:
attribute slots 0/1 and the constant. Every known-family unseen criterion
requires an unsupported slot 2/3. Within held families, the criterion
hybrid reaches 0.500000 on supported directions and 0.226562 on
unsupported directions. Thus its wholly unseen-coordinate failure does
not isolate a defect in the neural encoder. However, the in-span
unseen-wording failure remains real.

Held-family top-1 by diagnostic arm:

| family | text candidate, true criterion | exact candidate, text criterion | full CBF-32 |
|---|---:|---:|---:|
| compliance | 0.375000 | 0.351562 | 0.207031 |
| intent-routing | 0.406250 | 0.371094 | 0.316406 |
| similarity | 0.359375 | 0.367188 | 0.296875 |

Criterion swaps are direct fixed-candidate interventions, not cosine
diagnostics. Denominators below are ordered wording-expanded pairs;
semantic criterion IDs must differ.

| evaluation | swaps | teacher changes | CRA | student changes given teacher change | changes to new teacher winner | both endpoints correct |
|---|---:|---:|---:|---:|---:|---:|
| unseen wording | 1280 | 968 | 0.504688 | 0.431818 | 0.092975 | 0.033058 |
| unseen criterion | 5120 | 3840 | 0.588672 | 0.603646 | 0.127083 | 0.041146 |
| held family | 9216 | 7072 | 0.556858 | 0.544118 | 0.103224 | 0.046380 |

The last three rates condition on teacher change. On teacher-stable
pairs, full CBF-32 changes anyway at rates 0.269231/0.456250/0.401119.
Oracle/oracle has CRA 1, correct-new-winner rate 1, and no teacher-stable
changes. Both criterion-blind controls have exactly zero student changes.
Criterion use is therefore present, but often does not implement the
requested direction; it is not simply ignored. On held-family
teacher-changing pairs, only 730 of 3848 student changes reach the new
teacher winner, and both endpoints are correct in 328 of 7072 pairs.

Decision-tree outcome: **both tested text mappings are inadequate**.
The oracle hypothesis is valid for this corpus, but neither a unique
criterion bottleneck nor a pure joint-alignment failure has been isolated.
This fixed-budget linear probe does not prove information is absent from
the encoder or rule out optimizer limitations. No larger model, more
epochs, new data, anchored-basis method, or B-STEF is justified by this
audit alone. Vey 2 remains unchanged.

Evidence includes all four arms plus cosine/blind controls, train/eval
ranking and continuous correlations, every family/K bin, basis support,
coordinate MSEs, 3968 per-decision rows, and 15,616 causal swap rows.
Reloaded checkpoints reproduced every score; 7936 hybrid permutation
checks and all fixed-candidate swap/control checks passed.
The complete metric/hash inventory is
`research/cbf0/audit_result_manifest.json`; data/checkpoints remain under
`vey-data/decisionmix/d3/cbf/audit-v1/`.

SHA-256:

- predictions: `eb6abbb4add4cbcbaa091f15a575a263210a206b87cd5b1078065477860c4344`
- swaps: `9c9359c7c0678ae0f0ad63437039422c7afdc19a981e9051e54c86cdd5597d2e`
- results: `bb6a3cd674dc40bc1e00ded4b4fa002bf84ea1b019913295c8e33701f4c3cfc6`

### CBF-2: linear ceilings and supported-span audit, preregistration

CBF-2A-C use only cached representations and saved CBF-1 scores. No neural
refitting, new encoder pass, data, capacity, or B-STEF. Candidate OLS is
the FP64 economy-SVD minimum-norm solution without a bias or centering.
The cutoff is machine epsilon times max(N,d) times the largest singular
value. Uniformly repeated training candidates are deduplicated only after
checking their weights and targets agree. All five original probe outputs
are solved to preserve intercept-score parity; the first four are exactly
the requested numeric-attribute least-squares solution.

Ridge uses lambdas 1e-8/1e-6/1e-4/0.01/1/100, selected by four-attribute
MSE on an inner validation split of original train scenarios. Within each
family/K group, seed 7 selects two of eight scenarios for validation.
The selected lambda then refits original training candidates only.
Persist the spectrum, rank, condition number, entropy effective rank,
normal-equation residual, per-slot and named-attribute MSE/R2/correlations/
within-scenario ordering, and oracle-criterion decision metrics.
This is an attribute least-squares ceiling, not a maximum-ranking theorem.

For CBF-2B, only the 20 supported training criterion IDs are eligible.
Nearest-centroid retrieval uses their original two training wordings and
tests their 40 unseen wording strings. Semantic-ID accuracy and oracle
direction accuracy are distinct: a wrong semantic ID may share the same
numeric direction. OLS first repeats that exact 40-string train/40-string
test setup for solver parity with SGD, then runs four leave-template-out
folds with three wordings per ID training and one testing. Report direction
cosine, vector MSE, sign accuracy, exact-candidate decision accuracy, fold
IDs, and equal-fold macro metrics. The three-wording folds change coverage,
so their gains alone cannot identify optimizer effects.

CBF-2C reaggregates the saved swaps for every arm. Report all swaps and
the subset where both criterion directions are supported, with oracle and
blind controls on identical subsets and split/family/K denominators.

CBF-2D may run only if candidate OLS train top-1 and all four slot R2s
reach 0.95, selected-ridge supported transfer reaches 0.95, centroid
semantic-ID accuracy reaches 0.95, and leave-wording-out OLS reaches
top-1 0.95 with mean cosine 0.99. Otherwise no neural training occurs.
An existing-row resplit is audited regardless: reserve one quarter of
scenarios per known-family/source-split/K stratum, train only wording
indices 0/1, and hold out the existing negative-slot-3 direction. Held
families remain excluded. Require rank-5 training targets, all evaluation
directions in their span, and disjoint scenarios/candidate strings.
No mixed-coordinate criteria exist, so no novel-combination claim is made.
If gated in, full CBF-32 retains the original 200-epoch recipe and must
pass paired unseen-direction and held-family gains plus multi-family
transfer before B-STEF is unblocked. This audit never launches B-STEF.
The exact protocol is `research/cbf0/ceiling_protocol.json`; artifacts
remain under `vey-data/decisionmix/d3/cbf/ceiling-v1/`.

### CBF-2 result: partial linear recovery, wording failures, no resplit fit

Recipe `2109906`. CBF-2A-C completed with zero neural retraining and no
new encoder pass or examples. All four preregistered supported-span gates
fail. CBF-2D training was not run; B-STEF remains prohibited.

#### Candidate least-squares ceiling

The deduplicated training matrix has shape **2400 x 384**, numerical rank
**384**, condition number **44,678.914095**, and singular-value entropy
effective rank **25.028211**. Each candidate had uniform multiplicity 4
in the original 9600 flattened training pairs. OLS normal-equation
residual infinity norm is **2.440226e-11**; an independent
`numpy.linalg.lstsq` reproduces its training MSE within 1e-12.
These statistics concern the cached pooled vectors, not encoder tokens.

OLS training recovery, pooled over unique candidates; slots refer to the
fixed family-local attribute order:

| slot | minimum MSE | R2 | Pearson | Spearman | macro scenario pairwise |
|---|---:|---:|---:|---:|---:|
| 0 | 0.031408 | 0.619435 | 0.787042 | 0.798742 | 0.798240 |
| 1 | 0.031697 | 0.612799 | 0.782814 | 0.793609 | 0.786001 |
| 2 | 0.031065 | 0.628054 | 0.792499 | 0.804134 | 0.811536 |
| 3 | 0.031961 | 0.600621 | 0.774997 | 0.786245 | 0.803806 |

The comparable pooled SGD slot R2s are 0.298749/0.303946/0.264437/0.279788.
SGD therefore left substantial linear recovery unused, but the exact
least-squares solution still does not approach perfect attribute recovery.
This is the training MSE optimum in the pinned bias-free function class,
not an optimal ranking bound or proof that every readout must fail.

Ridge selects **lambda=0.0001** using 1800 inner-train and 600
inner-validation candidates, grouped by whole scenarios. Validation numeric
MSE is 0.046518. The selected solution refits only the original training
pool, and its normal equations were independently checked.

Text candidate / oracle criterion top-1:

| evaluation | old SGD | OLS | selected ridge |
|---|---:|---:|---:|
| train | 0.421875 | 0.567188 | 0.548438 |
| unseen wording | 0.421875 | 0.471875 | 0.478125 |
| unseen criterion | 0.434375 | 0.534375 | 0.512500 |
| held family | 0.380208 | 0.302083 | 0.328125 |

Training improvement is real (+0.145313 top-1 for OLS) but neither exact
decoding nor held-family transfer is solved. OLS held-family slot R2s
are -7.790504/-0.904533/-8.078671/-8.019249; ridge also remains negative
on every slot. Among supported held-family directions, selected-ridge
top-1 is 0.291667. Optimization alone does not close this interface's gap.
Per-family named-attribute metrics are retained, not just these slot means.

#### Supported criterion wording ceiling

Nearest-centroid semantic-ID recovery is **14/40 = 0.35**. Exact oracle
direction recovery is **26/40 = 0.65**, and exact-candidate decision
accuracy is **0.712500**. These are different metrics: wrong semantic IDs
can share the correct signed slot, and wrong directions can accidentally
choose the same winner.

Same-data OLS, using exactly the two original training wordings per
criterion, reaches **0.345313** exact-candidate top-1 versus SGD's
**0.460938**. Mean direction cosine is 0.274544; vector MSE is 1.689852.
Replacing SGD with the minimum-norm solution does not repair wording
generalization.

Leave-one-wording-out OLS, training on the other three wordings per ID:

| held template index | top-1 | mean direction cosine | vector MSE | nonzero-coordinate sign accuracy |
|---|---:|---:|---:|---:|
| 0 | 0.629688 | 0.736919 | 0.213448 | 0.950000 |
| 1 | 0.542188 | 0.686015 | 0.225085 | 0.875000 |
| 2 | 0.331250 | 0.184544 | 1.289919 | 0.625000 |
| 3 | 0.471875 | 0.471459 | 0.416085 | 0.750000 |

Equal-fold macro top-1 is **0.493750**, mean cosine **0.519734**.
Micro top-1 is 0.524479 because the original corpus has more decisions
for templates 0/1. The macro result is the preregistered gate quantity.
All queries here use supported directions; slots 2/3 are excluded.
The centroid failure demonstrates inadequate invariance for this retrieval
interface, not that semantic information is absent from the entire encoder.
Three-wording folds change supervision coverage and cannot alone establish
a solver-only diagnosis.

#### Supported causal decomposition

Held-family swaps restricted to two supported directions contain **1536**
ordered pairs, with **1216** teacher changes. All arms below use that
identical subset and the original saved CBF-1 checkpoints:

| arm | CRA | student changes given teacher change | changes to new teacher winner | both endpoints correct |
|---|---:|---:|---:|---:|
| oracle/oracle | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| text candidate / oracle criterion | 0.479167 | 0.368421 | 0.065789 | 0.026316 |
| oracle candidate / text criterion | 0.651042 | 0.641447 | 0.270559 | 0.187500 |
| full CBF-32 | 0.571615 | 0.544408 | 0.097862 | 0.044408 |

The last three rates condition on teacher change. Teacher-stable unwanted
change rates are 0/0.100000/0.312500/0.325000 in the same row order.
Both blind controls have zero changes. Bad candidate geometry already
corrupts responses with perfect criteria; the supported criterion mapper
also fails with exact candidates. These errors do not add linearly, and
the full model's 0.103224 rate on all held swaps has a different denominator.
All-swap and supported-swap family/K reports remain available.

#### Decision and durable evidence

All supported-span gates fail. The same-corpus support-complete plan is
feasible and audited: **1344 train / 288 unseen-wording / 64 unseen-direction
/ 768 held-family rows**; target span rank 5; maximum held-direction
span residual 7.327682e-16. Scenarios and candidate strings are disjoint.
The held direction is negative slot 3, expressible from supported targets.
No mixed-coordinate questions were generated or claimed. This plan was
not fitted because the prerequisite gates failed.

The justified conclusion is narrower than abandoning every frozen encoder:
**this pooled frozen representation plus a bias-free linear numeric
interface is inadequate**, with both optimization underuse and a substantial
least-squares recovery ceiling. Its supported criterion wording interface
also fails. No larger basis, extra epochs, B-STEF, or new architecture was
run. Representation-interface investigation is the next justified branch,
not a capacity escalation.

Saved solver weights reproduce all 3968 candidate decision rows and 120
criterion query vectors. Independent least-squares/ridge checks and
supported causal reaggregation pass. Spectra, per-attribute metrics,
family/K bins, query folds, grouped ridge IDs, the conditional split plan,
and artifact hashes are recorded in
`research/cbf0/ceiling_result_manifest.json`. Data and numerical weights
stay under `vey-data/decisionmix/d3/cbf/ceiling-v1/`. Vey 2 is unchanged.

SHA-256:

- candidate predictions: `a2d8ac74b27c43b2dbad6e47103b95767d4a5958b387124b6c097611e376c647`
- criterion predictions: `0487a691b421ff66a9a222d6b3035dfcba6d41b83e72487e681fc99d8766217f`
- results: `8eef26577b3414953156f1d5d5607f84e14ce0be7366c74e1882fdaa72d49cb9`

### CBF-3: representation-interface tomography, preregistration

CBF-3 first tests cached geometry, without neural retraining or capacity
increase. Difference OLS fits the original training scenarios' unordered
pairs, equivalent to scenario-centered rows weighted by sqrt(K).
Both it and the absolute comparator map 384 to four numeric attributes;
the latter reuses CBF-2's first four columns. True minimizing-criterion
constants cancel in ranking, so no estimated fifth intercept is compared.
Adequacy remains top-1/pairwise at least 0.95 on unseen-criterion and
held-family splits and each held family.

Family-local OLS uses all existing rows as diagnostic data, including
previously held families. Its 13x13 transfer matrix must distinguish
resubstitution diagonals from four-fold scenario-grouped CV inside every
family, stratified by source split/K, seed 7. Training interpolation on
a small family is not evidence of transferable numeric coordinates.

Template projection uses only the 20 supported criterion IDs. Two
crossfit groups split ten families into five donor and five recipient
families. Donors' four template means define a rank-at-most-three contrast
subspace; evaluated recipient strings never fit that projection.
All templates are observed through donors, so this is known-template
nuisance analysis, not unseen-template deployment. One global projection
per crossfit group is applied without OLS renormalization; no
criterion-specific parameters. Rerun 20-class centroid retrieval,
original-wording OLS, and three-wording-to-one-wording OLS.
Repair requires retrieval 0.95, wording-out top-1 0.95, and cosine 0.99.

If global difference adequacy and template repair do not both pass, make
one frozen FP32 pass over the exact original sorted texts using the same
stock DeBERTa revision as CBF-0, not Vey-2 weights. Persist every layer's
valid token states, offsets, IDs, masks, and fixed pooled features.
Final shipped pooling must reproduce the original cache within 1e-6.
Candidate readouts are shipped mean, CLS, content mean, numeric-token
union mean, and attribute-name/value-token union mean; criterion readouts
are shipped mean, CLS, content mean, and attribute-keyword mean.
Every readout stays 384-wide, normalized, with no layer/attribute
concatenation. Span selection uses visible benchmark grammar, not latent
facts or numeric targets; it is a grammar-assisted diagnostic interface.
Embedding output and every transformer layer are swept with the same
closed-form probes, without selecting a smaller posthoc suite.

Same-layer token-readout success against weak mean pooling supports a
pooling bottleneck; earlier-layer success supports a depth/readout choice.
Failure of this finite suite does not prove that all frozen readouts lack
information. Supported-wording or oracle-hybrid success does not establish
full unseen-criterion transfer. Best layer results are exploratory, not
promotion evidence. B-STEF remains blocked throughout this audit.
Exact protocol: `research/cbf0/tomography_protocol.json`; artifacts remain
under `vey-data/decisionmix/d3/cbf/tomography-v1/`.

### CBF-3 result: criterion cues recover; candidate suite remains inadequate

Recipe `397d7cd`; manifest-only recovery `287a429`. No neural head was
trained, no width increased, and B-STEF was not run.

#### Cached objective and family transport

Difference OLS uses 12,400 unordered training pairs. Its weighted-centered
objective was independently verified against the explicit pair sum.
Both comparisons below use four output attributes and true criterion
directions; the minimizing constant cancels, unlike CBF-2's estimated
fifth-output score convention.

| evaluation | absolute top-1 | difference top-1 | absolute pairwise | difference pairwise |
|---|---:|---:|---:|---:|
| train | 0.565625 | 0.528125 | 0.792121 | 0.763194 |
| unseen wording | 0.471875 | 0.471875 | 0.757768 | 0.741272 |
| unseen criterion | 0.534375 | 0.503125 | 0.763270 | 0.751659 |
| held family | 0.302083 | 0.307292 | 0.570933 | 0.598810 |

The difference objective does not repair the interface. Absolute
reconstruction R2 is not an adequacy criterion for a difference-trained
map: scenario offsets are unconstrained and cancel in decisions.

Family-local four-fold scenario CV ranges from 0.362500 to 0.506250
top-1; **zero of 13 families** meets the high-CV gate. Previously held
families illustrate the resubstitution trap:

| family | fit-on-all-family top-1 | out-of-scenario CV top-1 | CV slot R2s |
|---|---:|---:|---|
| compliance | 1.000000 | 0.421875 | -0.0352 / 0.2017 / 0.1345 / 0.1822 |
| intent-routing | 1.000000 | 0.406250 | -0.2827 / -0.3256 / -0.8413 / 0.0199 |
| similarity | 1.000000 | 0.468750 | -0.0161 / 0.3991 / -0.1343 / 0.1737 |

The 156 off-diagonal family-transfer cells average 0.273658 top-1,
range 0.131250-0.375000. All local fits and the 13x13 matrix are purely
diagnostic. High training interpolation with poor CV does not establish
an adequate family-dependent coordinate system. It also does not prove
the encoder has lost the information.

#### Template nuisance

Cross-fitted rank-three template removal changes semantic-ID retrieval
from 0.350000 to **0.475000**, original-wording OLS top-1 from
0.345313 to **0.548438**, and wording-out macro top-1 from
0.493750 to **0.627734**. Mean wording-out direction cosine increases
from 0.519734 to **0.731171**.
Oracle-direction retrieval stays 0.650000; centroid decision accuracy
changes only from 0.712500 to 0.718750. Template nuisance matters, but
the repair gate still fails. Donors expose all templates; evaluated
recipient strings do not estimate their own projection.

#### One frozen pass, complete token evidence

All **5368** original strings were encoded once in **84** batches:
**131,964 valid tokens**, **13 states** (embedding output plus 12 layers),
width **384**, FP32. Every valid token state, token ID, offset, special
mask, and span mask is persisted. Final shipped features match the
original cache with maximum error **0.0**. No encoder keys were missing
or mismatched; the stock checkpoint SHA-256 is
`964ceb3612da6cfdb45997d380fdb95f92c7499ffcabb50cbeea55e04756cafd`.
This is the CBF stock encoder, not trained Vey-2 weights.

Capture completed and saved all states/features before metadata
serialization rejected Hugging Face's loading-info sets. Recovery converted
those sets to lists and reloaded weights on CPU solely for metadata;
**zero additional encoder forwards** occurred.

The complete suite contains **65 candidate** and **52 criterion**
layer/readout combinations. Every candidate interface fails adequacy.
Exploratory maxima, not promoted choices:

- Best absolute training top-1: layer-10 content mean, 0.584375.
- Best absolute unseen-criterion top-1: 0.534375 (including layer-9
  numeric mean and the original final shipped mean).
- Best held-family top-1: embedding-output numeric mean, 0.401042
  absolute / 0.421875 difference.
- Best minimum training slot R2 across all absolute readouts remains
  0.600621 at final shipped pooling. No readout approaches exact numeric
  recovery on all four coordinates.

Final-layer numeric pooling has training slot R2s
0.555520/0.537270/0.581860/0.545860 and held-family top-1
0.328125 absolute / 0.317708 difference. Numeric-token selection is not
a candidate-side rescue. Union pooling does not separately expose fields,
so this is not a proof against every possible structured token interface.

The criterion side has a positive interface result:

| criterion interface | semantic-ID retrieval | wording-out top-1 | mean direction cosine |
|---|---:|---:|---:|
| final shipped mean | 0.350000 | 0.493750 | 0.519734 |
| layer-1 shipped mean | 1.000000 | 0.820313 | 0.963705 |
| embedding-output keyword mean | 1.000000 | 1.000000 | 1.000000 |
| layer-1 keyword mean | 1.000000 | 0.977344 | 0.998629 |
| layer-2 keyword mean | 1.000000 | 0.957422 | 0.996450 |
| final keyword mean | 0.650000 | 0.651172 | 0.745562 |

Keyword readouts at embedding output and layers 1/2 pass supported-wording
gates. Same-layer keyword versus whole-mean differences support an
interface bottleneck for these criterion cues; earlier-layer results also
show that later contextualization reduces their linear accessibility.

The perfect embedding-output result is **lexical identity recovery**:
grammar-assisted spans retain the same literal attribute name across
templates. This corpus fixes polarity per attribute, so removing
preference words does not challenge reversal or new-combination semantics.
It does not demonstrate arbitrary-question understanding or unseen
semantic-criterion transfer. No interface is promoted from this scan.

#### Decision and verification

The objective-mismatch branch fails. The high-family-local-CV branch
fails. Template nuisance has a measured partial effect, not a complete
repair. A criterion readout/depth bottleneck is supported; no candidate
readout in the fixed suite is adequate. Thus the whole frozen encoder
line is **not** jointly falsified: stable criterion cues exist outside
the final pooled vector, while candidate decoding remains unresolved.
The factorization itself remains valid; full unseen-criterion transfer
and B-STEF remain blocked. Vey 2 is unchanged.

Verification reconstructed all fixed features from saved raw token states
with maximum error **2.384186e-7**, checked visible-text span masks, and
reproduced **670,800** candidate attribute vectors plus **6240** criterion
query vectors from saved decoders. Cached verification also reproduced
all **5160** out-of-scenario family predictions.
Full metrics, spectra, masks, CV IDs, transfer matrices, decoder weights,
raw states, and hashes are inventoried in
`research/cbf0/tomography_result_manifest.json`; data remains under
`vey-data/decisionmix/d3/cbf/tomography-v1/`.

SHA-256:

- cached results: `aa30699db4fef7cd8ecad6ee52eb364542dd44107f42aae6cd0c0b46bab2d18d`
- layer results: `0d29f6918b3308dc5f7052af43fe5efe5dbb2a6463248457cb71efb7fd8be953`
- raw-state verification: `48062482cbaefd2bb61de6c873f9af42aeef0794777b7166f441a6b2ae6e7e86`






## CBF-4: Exact-State / Semantic-Criterion Field (preregistered research)

CBF-3's local keyword result recovered literal axis cues under fixed polarity;
it did not establish semantic criterion mapping. CBF-4 removes candidate
reconstruction from the experiment. A deterministic parser owns four exact
state coordinates: reliability, purchase expense, operating expense, and
convenience. Only the criterion map is learned.

The original corpus contains separate quality, cost, and preference scenarios.
This screen joins their existing candidate facts by source split, K, repetition,
and candidate index. It resamples no numeric value, but the joined four-field
state is a **new benchmark input**, not the original single-family input.
Training diagnostics use original train states; alias/polarity tests use
unseen-wording states; composition tests use unseen-criterion states.

`research/cbf0/exact_state_protocol.json` freezes the complete recipe before
corpus construction or encoding. `exact_state_build.py` fixes all authored
aliases and question renderers. Training supports every axis, both polarities,
and magnitudes one/two. Tests cover:

- aliases containing none of the literal axis tokens, absent from training;
- unseen same-attribute positive/negative question pairs on fixed candidates;
- weighted compositions with held positive-ray directions;
- entire held axis-pair supports, not just held question strings.

Four interfaces: literal keyword lookup with generic polarity/weight cues
(UNKNOWN on absent names), frozen final mean, layer-1 mean, and layer-1 mean
over the entire semantic expression after generic task-prefix removal.
The span selector never searches for attribute names; it retains polarity,
weights, aliases, and every clause. One stock DeBERTa-v3-xsmall FP32 pass
encodes **criterion strings only**. Three identical bias-free 384-to-4 maps
use six training-template-out folds to select a fixed-grid ridge coefficient.
No test criterion participates in fitting. Four active coordinates remain
within the previous width32 budget; no candidate parameters remain.

Gates, fixed before measurement:

1. Alias top1 at least 0.80, with paired semantic-direction-cluster significance
   above lexical lookup. Report nominal 95% and three-comparison simultaneous
   intervals; the simultaneous lower bound must exceed zero.
2. Polarity top1 at least 0.90, every held literal reverse-question direction
   cosine negative, and a student winner change for every teacher-changing
   polarity reversal.
3. Composition and held-support top1 each at least 0.80, with disjoint training
   direction rays and held supports.
4. On teacher-changing criterion swaps, the student chooses the **new teacher
   winner** at least 0.80 of the time. This rate does not additionally require
   the student to change; changed-to-new and both-endpoint rates are separate.
5. Exactly identical restored scores and content-tie-broken winner after
   reversed and seed7-shuffled candidate orders.

Report every interface and stratum, not an observed best layer. Any passing
learned interface only opens a separately preregistered CBF-4 replication;
B-STEF stays blocked until replication passes. This controlled four-axis
screen is not arbitrary-criterion or Jev parity, and it does not validate
semantic candidate extraction. Frozen Vey-2 is untouched.


### CBF-4 result: tested criterion maps fail with exact candidate state

Preregistration: `739b567`. Measurement implementation: `6a33275`.
Correctness amendment pinned before its six new encodings: `3f3db51`.
The corpus contains 288 training questions, 32 alias questions, 32 polarity
questions, 48 held-weight compositions, and 72 held-support compositions.
Training activates all four outputs under both signs and magnitudes one/two
(target rank4). Composition test rays are absent from training, including
positive scalar equivalents; the two held axis-pair supports are absent too.
All alias strings omit every literal axis token. The test contains 2,944
question/scenario rows, balanced over K=2/4/8/16 within each stratum.

The initial frozen FP32 stock DeBERTa-v3-xsmall pass encoded 472 unique
questions in eight batches (7,812 valid tokens; maximum28 tokens).
A grammar-only amendment encoded six new strings in one additional batch,
without repeating any original encoding or refitting. Final evidence uses
472 questions and7,800 tokens; total capture is478 strings and7,876 tokens
over nine batches, with **zero candidate forwards**. Missing/mismatched
encoder keys are empty. The final/layer1 means include valid special tokens;
the layer1 span preserves the entire expression, removes only the frozen
generic task-prefix patterns, excludes special tokens, and never looks up
an attribute name. Unrecognized prefixes remain in the span. Each trained
map is 384-to-4 (1,536 active parameters), within the previous width32 budget.

Six training-template-out folds selected ridge0.1 for final mean, ridge0.1
for layer1 mean, and ridge1 for layer1 span. Training coefficient MSE is
0.603607 / 0.591313 / 0.673146 respectively; these are regularized selected
models, not interpolation ceilings. No test query or candidate decision
participated in selection. All models and gates were frozen before scoring.

#### Held-question decision accuracy

Teacher-top-set membership; UNKNOWN counts incorrect. Alias/polarity each
have512 rows, composition768, held support1,152.

| interface | A: aliases | B: polarity | C: compositions | D: held supports |
|---|---:|---:|---:|---:|
| lexical schema lookup + surface polarity/weights | 0.000000 | 1.000000 | 1.000000 | 1.000000 |
| frozen final mean | 0.267578 | 0.302734 | 0.311198 | 0.328125 |
| layer1 mean (predeclared headline) | 0.318359 | 0.347656 | 0.337240 | 0.329861 |
| layer1 semantic-expression span | 0.320313 | 0.373047 | 0.308594 | 0.315104 |

The lexical control returns UNKNOWN on all aliases, not a positional
candidate fallback; it recovers the exact signed coefficients on all440
literal questions, including all held composition directions. This confirms
that the candidate parser, criterion coefficient basis, and teacher are
compatible. It is not an alias-aware semantic control.

All learned interfaces are statistically above an abstaining lexical
baseline on aliases, but none approaches the required0.80 accuracy:

| interface | alias improvement | paired95% CI | simultaneous CI, three comparisons |
|---|---:|---|---|
| final mean | +0.267578 | [0.197266,0.347656] | [0.181641,0.369141] |
| layer1 mean | +0.318359 | [0.257813,0.386719] | [0.244141,0.400391] |
| layer1 span | +0.320313 | [0.250000,0.398438] | [0.238281,0.417969] |

Bootstrap: 10,000 paired draws, seed0, eight signed semantic-direction
clusters. This interval describes this four-axis alias screen; beating
UNKNOWN is not sufficient evidence of useful semantic criterion mapping.

#### Polarity and causal criterion swaps

Each literal positive/negative pair reverses the teacher winner on all16
fixed candidate scenarios: 16 question pairs, 256 reversal cases.

| interface | negative reverse-pair cosines /16 | mean reverse cosine | student winner-change rate |
|---|---:|---:|---:|
| lexical | 16 | -1.000000 | 1.000000 |
| final mean | 1 | +0.598741 | 0.500000 |
| layer1 mean | 9 | -0.061979 | 0.769531 |
| layer1 span | 14 | -0.624856 | 0.886719 |

Span pooling improves direction reversal but does not ground the correct
axis sufficiently: polarity decision accuracy remains0.373047. On alias
reversals, negative cosines occur in only2/16,3/16,7/16 pairs for final,
layer1 mean, and layer1 span; the lexical control abstains.

The causal screen evaluates every ordered semantically different criterion
pair within each test stratum/template on the same candidate set:
45,184 eligible pairs, 34,622 teacher-changing pairs.

| interface | new teacher winner (G4) | changed to new teacher winner | both endpoints correct |
|---|---:|---:|---:|
| lexical | 0.916354 | 0.916354 | 0.916354 |
| final mean | 0.278233 | 0.131015 | 0.069147 |
| layer1 mean | 0.299318 | 0.172780 | 0.081105 |
| layer1 span | 0.287043 | 0.157588 | 0.075732 |

G4 is exactly the requested conditional new-winner accuracy, **without**
additionally requiring a student change. The verification found5,097 /
4,381 /4,482 correct-new cases with an unchanged student winner in the
three learned interfaces; these count toward G4, not changed-to-new.
Per-stratum and tie-aware rates are persisted. Lexical lookup is perfect
on all teacher-changing literal pairs and abstains on alias pairs.

#### Gates, arithmetic correction, and evidence

All three learned interfaces fail G1,G2,G3,G4 and pass G5. No learned
interface qualifies for replication or B-STEF. The lexical control passes
G2-G5 but fails alias transfer; it is not a learned semantic field.

G5 checks reversed and seed7-shuffled orders on every test row/arm:
23,552 comparisons, including1,024 lexical abstention comparisons.
All22,528 numeric comparisons have exactly identical restored scores
and content-tie-broken winners; maximum score error0.

The initial score implementation divided each percent before summation.
Although the lexical coefficient vectors were exact, floating-point
roundoff split mathematical ties: two teacher scores of5 became
0.04999999999999993 and0.05000000000000002. This changed three lexical
concrete winners and some pairwise relations. The correction sums
coefficient times integer percent first, then divides once by100.
It changes no data, coefficient, model, hyperparameter, or gate.
The initial artifacts remain at the study root; the arithmetic-only replay
is in `score-corrected/`, using saved vectors with **no refitting or
additional encoder forward**. That replay left all learned winners and
quality summaries unchanged. Corrected lexical literal top1, concrete
winner, and pairwise agreement are all1.000000.

Corpus review also found three alias pairs with malformed wrappers, such
as “Choose the option with the least likely to fail.” The amendment
`exact_state_amendment.json` pins the six sentence-only repairs before
encoding them. It changes no alias meaning, coefficient, training query,
state, literal test question, head, ridge choice, interface, or gate.
Every training feature and fitted map is byte-identical. Even perfect
predictions on all six repairs could raise the original alias top1 to
at most0.455078 /0.501953 /0.501953, so the amendment cannot manufacture
a passing verdict. G2-G4 had already failed independently.

Final tables above use `grammar-corrected/score-corrected/`; all original
and arithmetic-only evidence remains preserved. The grammar repair changes
only alias predictions: alias top1 becomes0.267578 /0.318359 /0.320313
from0.267578 /0.314453 /0.314453. Every literal-stratum summary is unchanged.
The semantic span policy is unchanged, including retaining unrecognized
generic prefixes; no attribute knowledge was added to span selection.

Persisted-evidence verification reconstructed the three pooled interfaces
from every saved layer1/final token state (maximum error1.79e-7), checked
1,920 parsed source fields against the original donor strings, recovered
every query vector from saved features/maps exactly, and verified all2,944
decision rows,84,480 scalar scores, and34,622 teacher-changing swap records.
There are27 composed teacher-top-set tie rows; tie handling is content-based,
never position-based. Verification performs zero encoder forwards/refits.

This fails the **tested frozen linear criterion interfaces** under the
preregistered training/selection recipe. It does not establish that the
encoder lacks criterion semantics or that every nonlinear criterion map
would fail. Candidate reconstruction cannot explain this result: state
is exact. No candidate tomography, larger model, B-STEF, replication, or
semantic candidate extraction was launched. Frozen product source,
tests, examples, and `vey-2-final` remain unchanged.

Artifacts: `vey-data/decisionmix/d3/cbf/exact-state-v1/`; inventory and full
final corrected metrics: `research/cbf0/exact_state_result_manifest.json`.
Final measurement/verification commands:

```sh
ROOT=/home/fazinahamed/Documents/vey-data/decisionmix/d3/cbf/exact-state-v1/grammar-corrected
python research/cbf0/exact_state_run.py --root "$ROOT" --replay
python research/cbf0/exact_state_verify.py --root "$ROOT" --replay
```

The replay command refuses to overwrite existing measured evidence.


## CBF-5: schema-grounded criterion compiler (preregistered research)

CBF-4 leaves axis grounding versus orientation unresolved: opposite question
vectors do not prove either correct schema axis or correct sign. CBF-5
separates semantic atom resolution from exact operators and composition.
No continuous coefficient regressor, candidate neural path, model widening,
encoder fine-tuning, or B-STEF.

`research/cbf0/schema_grounding_protocol.json` fixes the study before new
composition corpus construction or scoring. Eight positive/negative schema
prototypes are normalized means of the48 weight-one atomic training questions
(four axes, both signs, six templates). Unknown aliases alone enter the
semantic resolver; literal names, polarity operators, weights, aggregation,
and integer-percent field scoring stay in code.

Three requested comparisons: raw positive-prototype axis cosine with sign
resolved within that axis's prototype pair; signed antipodal grounding using
the maximum absolute positive-minus-negative cosine difference; and the
same antipodal score after template-nuisance projection. Strong eight-signed
nearest-prototype retrieval and projected ordinary retrieval controls prevent
crediting a weak polarity baseline or projection itself as an antipodal gain.
Every method shares the same eight base prototypes.

Projection reapplies the CBF-3 training-template-mean/SVD recipe in the
chosen layer1 semantic-span space: six centered balanced template means,
all nonzero contrasts, rank at most5. It does not transplant a final-layer
projector, use held queries, or refit projected prototypes.

All32 corrected CBF-4 alias questions remain unchanged for paired comparison.
New hard test:50 alias compositions covering all six axis pairs, both signs,
exact unit/double weights, and the supplied natural fail-likelihood /
regular-running-bill question plus its reverse. The parser isolates atomic
contexts; the neural resolver never receives a full composition or outer
weight frame. Cached old atomic features are reused; only new atomic contexts
are encoded. Exact candidate states remain unchanged.

Report alias axis accuracy, sign accuracy conditional on correct axis
(including its numerator/denominator), and joint signed-atom accuracy
separately. Task gates: joint atoms and alias decisions at least0.80,
paired/clustered improvement over the frozen CBF-4 best0.3203125, exact
literal compilation at1.0 (no semantic credit), alias-composition decisions
at least0.80, correct new teacher winner at least0.80 on alias-containing
teacher-changing swaps, and exact permutation invariance.

An antipodal variant earns its mechanism only if it also beats both ordinary
retrieval controls in the same raw/projected space on joint atoms and alias
decisions with simultaneous paired lower bounds above zero. Otherwise retain
simpler retrieval if it meets the task gates. Bootstrap keeps semantic
direction clusters intact; reused CBF-4 aliases are a development screen,
not independent confirmation. Passing opens many-axis expansion before
B-STEF; failing suggests an evidence-scoped axis or orientation follow-up,
not that semantic information is absent from every encoder readout.

### CBF-5 result: exact compiler passes; antipodal grounding does not

**Verdict: `SEMANTIC_SCHEMA_GROUNDING_NOT_EARNED`.** No method passes
the semantic task gates; ASG also fails its mechanism comparison. Ordinary
signed-prototype retrieval is materially better in point estimates, but its
best joint alias accuracy is only0.50 and its best alias decision accuracy
is0.5859375. Neither retrieval nor ASG is promoted. No many-axis expansion,
learned metric, stronger encoder run, or B-STEF was started.

The corpus/compiler/prototype rules were committed at `cc0e8a0` before new
corpus construction and encoder capture. Measurement was committed at
`2f8b31a` before resolver scoring. The encoder is the same pinned FP32 stock
DeBERTa-v3-xsmall used by CBF-4, not the trained Vey-2 checkpoint. Prototype
and nuisance fitting use only48 atomic training questions; projection rank
is5. No alias labels or evaluation outcomes enter inference or fitting.

#### Alias atoms, decisions, composition, and causal response

Alias diagnostics use32 unique corrected CBF-4 questions. Conditional sign
is measured **only on correctly selected axes**, with its denominator shown.
Joint accuracy requires both axis and sign. Alias decisions use512 paired
rows on the16 original unseen-wording states. Composition decisions use800
rows from50 fixed alias-containing questions on16 unseen-criterion states.

| method | alias axis | sign given correct axis | joint alias atoms | alias decisions | alias-composition decisions | correct new causal winner |
|---|---:|---:|---:|---:|---:|---:|
| positive-first cosine |22/32 =0.6875 |15/22 =0.6818 |15/32 =0.46875 |0.56640625 |0.43000 |0.399491 |
| ASG |5/32 =0.15625 |2/5 =0.4000 |2/32 =0.06250 |0.275390625 |0.21000 |0.174131 |
| ASG + nuisance projection |3/32 =0.09375 |0/3 =0.0000 |0/32 =0.00000 |0.240234375 |0.23875 |0.203484 |
| eight-signed cosine |23/32 =0.71875 |16/23 =0.6957 |16/32 =0.50000 |0.58593750 |0.46000 |0.443160 |
| positive-first cosine + projection |24/32 =0.7500 |16/24 =0.6667 |16/32 =0.50000 |0.57812500 |0.53125 |0.513430 |
| eight-signed cosine + projection |22/32 =0.6875 |14/22 =0.6364 |14/32 =0.43750 |0.537109375 |0.46250 |0.441116 |

All methods fail G1 joint atoms, G2 alias decisions, G4 alias-composition
decisions, and G5 causal correct-new response. Each requires at least0.80.
All pass G3 exact literal compilation and G6 permutation invariance.

The composition component diagnostics count100 **occurrences**, not100
independent new semantic terms: the generator deliberately reuses old alias
atoms under exact conjunction/weight frames.

| method | component axis | sign given correct axis | joint components | exact compiled vectors |
|---|---:|---:|---:|---:|
| positive-first cosine |52/100 |38/52 =0.7308 |38/100 |5/50 |
| ASG |12/100 |0/12 =0 |0/100 |0/50 |
| ASG + projection |12/100 |0/12 =0 |0/100 |1/50 |
| eight-signed cosine |56/100 |43/56 =0.7679 |43/100 |7/50 |
| positive-first cosine + projection |64/100 |50/64 =0.78125 |50/100 |10/50 |
| eight-signed cosine + projection |58/100 |44/58 =0.7586 |44/100 |7/50 |

A correct aggregate vector does not imply correct component grounding:
equal-weight terms can exchange axes and sum to the right vector. That
explains projected ASG's one exact composition despite no correct components.
For the supplied “least likely to fail / regular running bill small” question,
the teacher vector is `[1,0,-1,0]`; both ASG variants instead compile
`[0,0,0,-2]`. Both also produce that same vector for the opposite supplied
question. The semantic failure is visible before candidate scoring.

Causal evaluation selects the lexicographically first visible question per
positive coefficient ray, without looking at predictions:56 representatives,
98,560 eligible ordered swaps,73,416 teacher-changing swaps. The gate is
`P(student_after == teacher_after | teacher_before != teacher_after)`;
**student change is not an additional requirement**. Stronger changed-to-new
rates are0.060069 /0.072627 for raw/projected ASG versus0.174131 /0.203484
correct-new rates. All four ordinary controls also fail the0.80 correct-new
gate. Swap rows are dependent, not73,416 independent experimental units.

#### Paired uncertainty and the antipodal mechanism

The frozen bootstrap uses10,000 seed0 draws over eight signed-direction
clusters, keeping the four surface forms and paired states together.
Bonferroni simultaneous intervals cover the20 planned contrasts/absolute
accuracies. Reused CBF-4 aliases are a development screen, not fresh transfer
confirmation.

| method | alias decision delta vs CBF-4 best0.3203125 | nominal95% CI | simultaneous CI |
|---|---:|---:|---:|
| positive-first cosine |+0.246094 |[+0.029297,+0.466846] |[-0.078125,+0.580078] |
| ASG |-0.044922 |[-0.103516,+0.015625] |[-0.130859,+0.042969] |
| ASG + projection |-0.080078 |[-0.142578,-0.013672] |[-0.169922,+0.023438] |
| eight-signed cosine |+0.265625 |[+0.039063,+0.496094] |[-0.069338,+0.599609] |
| positive-first cosine + projection |+0.257813 |[+0.044922,+0.480469] |[-0.037109,+0.589849] |
| eight-signed cosine + projection |+0.216797 |[-0.003906,+0.449219] |[-0.099609,+0.553718] |

No method's simultaneous improvement lower bound exceeds zero, nor does
its simultaneous absolute-accuracy lower bound exceed0.3203125. Nominal
retrieval improvements do not satisfy the preregistered promotion rule.

Raw ASG minus eight-signed retrieval: joint-atom delta−0.4375,
simultaneous CI[-0.8125,-0.09375]; decision delta−0.310547,
CI[-0.609375,-0.021484]. Projected ASG minus projected eight-signed
retrieval: joint delta−0.4375, CI[-0.78125,-0.09375]; decision
delta−0.296875, CI[-0.643557,+0.007813]. Neither ASG variant beats
both ordinary controls in the same space. Projection is not an ASG gain.

The motivating CBF-4 opposite-question result needs a representation-scope
correction: its cosines were computed **after the learned four-coordinate
map**, not on raw384-dimensional layer1 spans. CBF-5's training prototype
pair cosines are **+0.957946 reliability, +0.980656 purchase expense,
+0.977509 operating expense, +0.961070 convenience**. These measured
prototypes are near-parallel, not antipodal.

By the frozen diagnostic thresholds (axis high0.80, conditional sign
high0.90), every method has both low axis and low orientation accuracy.
ASG's conditional samples of5 and3 are especially small. This rules out
promotion of this frozen span/prototype-retrieval interface, not all possible
readouts of the frozen encoder. An axis-only tiny metric is not justified by
the low-axis/high-sign branch here; a stronger semantic criterion
encoder/readout is a next candidate under a new protocol. Do not infer that
semantic information is absent, or start B-STEF.

#### Exact compiler and persisted proof

All155 literal polarity/composition/held-combination criteria compile to
the exact integer teacher vectors. All4,960 literal decision rows have
top-set accuracy, concrete-winner accuracy, and pairwise agreement1.0
under all six methods. Literal terms never invoke neural resolution; this
is compiler correctness, not semantic progress.

The candidate-state artifact is byte-identical to corrected CBF-4. Of84
isolated atomic inputs,80 cached features are reused exactly; four new
inputs require one encoder forward. Candidate neural forwards and
full-composition forwards are both zero. Independent reconstruction from
saved token offsets/masks/states reproduces all four new span features
within2.015e-8 maximum coordinate error.

Independent verification reconstructs training-only prototypes/projector,
792 semantic component resolutions, all7,584 decision rows,91,008 reversed/
shuffled permutation checks,73,416 teacher-changing swaps, and all20 paired
intervals. Maximum cosine discrepancy is3.331e-16. UNKNOWN propagation,
exact cancellation, unsupported syntax rejection, and synthetic antipodal
resolution were also exercised before measurement.

Evidence remains under
`/home/fazinahamed/Documents/vey-data/decisionmix/d3/cbf/schema-grounding-v1`.
Git records the protocol, code, hashes, counts, license class, and machine
inventory in `research/cbf0/schema_grounding_result_manifest.json`;
features, prototype arrays, parsed terms, compiled predictions, decision/
swap rows, intervals, and independent verification stay in the data tree.
No frozen Vey-2 runtime, examples, or tests changed.

```sh
python research/cbf0/schema_grounding_capture.py
python research/cbf0/schema_grounding_run.py
python research/cbf0/schema_grounding_verify.py
```

Capture/measurement refuse existing evidence. Run them in order only on
a fresh protocol output root; verification consumes persisted evidence and
refuses to overwrite its own proof.


## CBF-6: schema relation encoder (preregistered research)

**HYPOTHESIS:** joint criterion/field self-attention may support the ternary
relation `lower / unrelated / higher` that independent prototype geometry
failed to ground. ASG remains a closed negative result; this is a different
interface, not an ASG rerun. No CBF-6 quality result is recorded here yet.

Preregistration `cf4fb73` fixes the linear frozen-pair head, one 64-wide GELU
control if it fails development, and final-transformer-layer-only adaptation
if both frozen heads fail. Amendment `a473500` replaces two uses of
`equally many uses`, rejected by the unchanged compiler's weighting grammar,
with `the same number of uses` before any model outcome. The original partial
corpus remains preserved. Encoder adaptation starts from a fresh stock load.
The protocol is `research/cbf0/schema_relation_protocol.json`.

**MEASURED, pretraining audit:** the new final contains128 atomic aliases:
16 per signed direction across the four fields, organized as64 reversal
pairs. A blinded independent reviewer read shuffled questions without authored
labels or pair IDs and agreed with128/128 axis/sign labels, with zero ambiguous
items. Normalized duplicate, literal-field-name, complete old-alias inclusion,
and training-text duplicate checks have zero hits. Content-word and shared
ngram overlap is retained rather than silently filtering ordinary shared words.
The128 new compositions form64 reversal pairs and use every final atom.
Their teacher coefficients are checked under the frozen compiler. Candidate
states are byte-identical to CBF-4/5; none are encoded neurally.

Literal-only training, validation and G0 each contain32 unique atoms, paired
with the four bare canonical fields. Validation templates4/5 are held out
with whole-text grouping; numerical weights never enter model input. The old
32 aliases and50 compositions are development only. Checkpoints minimize
unweighted literal-validation cross-entropy, never alias/G0/final metrics.
Head A/B share training and preprocessing; adaptation changes initialization
and optimization as declared, so it is not an isolated encoder-only causal
comparison. Only a preregistered near-pass weak-sign trigger permits one
literal reversal-equivariance control.

The selected architecture opens final once, after immutable development
selection. G0–G7 require literal relation0.95, axis0.85, conditional sign0.90,
joint atom0.80, alias decisions0.80, composition decisions0.80, causal
correct-new0.80, and exact/permutation1.0. Conditional sign retains its
correct-axis denominator. Causal correctness does not require the student to
change its answer; changed-to-new and both-endpoints are separate diagnostics.
Intervals use10,000 seed0 bootstrap draws over the64 reversal pairs, not
derived candidate/swapping rows as independent examples.

Every4-by-3 softmax matrix is uncalibrated class mass, not a certificate or
validated correctness probability. A seed7 pass triggers fixed-architecture
seeds11/13 replication before promotion. B-STEF remains blocked until fresh
transfer, replication, OOD utility and architecture-specific exact foldability.
A negative result activates the declared axis/orientation/stronger-encoder
branch; it does not terminate the autonomous endgame.

Raw evidence stays under
`vey-data/decisionmix/d3/cbf/schema-relation-v1/pretraining-amended`.
Git stores counts, revisions, hashes and license class in
`research/cbf0/schema_relation_corpus_manifest.json`, not corpus/checkpoint
payloads. `schema_relation_prepare.py`, `schema_relation_run.py` and
`schema_relation_verify.py` separate preparation, measurement and independent
head/decoder/integer-decision reconstruction. Output creation refuses overwrite.

### CBF-6 result: literal-trained pair heads do not earn semantic transfer

**Hypothesis and controls.** The joint stock-pair interface was tested with
a1155-parameter linear head, a24,835-parameter GELU head, then the same linear
head with only the final transformer layer trainable. All arms used literal-only
data, unchanged exact state/compiler, and literal-validation checkpoint selection.
Measurement commit `789a18f` follows preregistration `cf4fb73`, pre-outcome
grammar amendment `a473500`, and blinded corpus freeze `c8db098`.

**MEASURED, development progression:**

| Arm | Selected epoch | Literal validation CE | G0 | Old-alias axis | Joint | Sign given correct axis |
|---|---:|---:|---:|---:|---:|---:|
| A: frozen linear | 10 | 0.644220 | 0.890625 | 2/32 | 2/32 | 2/2 |
| B: frozen GELU | 3 | 0.413298 | 0.890625 | 2/32 | 2/32 | 2/2 |
| C: final-layer adaptation | 1 | 0.453979 | 0.859375 | 0/32 | 0/32 | undefined |

All three fail development. The preregistered joint/decision/earlier-arm
tie-break selects A before final opens. The2/2 conditional signs do not establish
general orientation robustness. No reversal trigger is eligible; no reversal
arm or seed replication is run.

**MEASURED, fresh final, selected A:**

| Gate / metric | Result | Required | Outcome |
|---|---:|---:|---|
| G0 held literal ternary relation | 114/128 =0.890625 | 0.95 | FAIL |
| G1 atomic axis | 10/128 =0.078125 | 0.85 | FAIL |
| G2 sign given correct axis | 3/10 =0.300000 | 0.90 | FAIL |
| G3 joint signed atom, primary | 3/128 =0.0234375 | 0.80 | FAIL |
| G4 alias top-set decision | 66/2048 =0.0322266 | 0.80 | FAIL |
| G5 composition top-set decision | 13/2048 =0.0063477 | 0.80 | FAIL |
| G6 teacher-changing correct-new | 0.0086691 | 0.80 | FAIL |
| G7 exact literal and permutation | 1.000000 | 1.00 | PASS |

A returns UNKNOWN for113/128 final atoms. G0's96/96 unrelated predictions hide
weak matched-field behavior:18/32 matched predictions are correct. Runtime
literal execution is separate and remains perfect on155 compiled criteria and
4,960 decisions, with zero neural literal callbacks. No learned primitive,
probability-calibration, certification or performance promotion follows.

**Uncertainty.** The64 atomic reversal-pair bootstrap gives nominal95% intervals:
axis[0.0234375,0.140625], joint[0,0.0546875], conditional sign[0,0.4545455],
alias decisions[0.0078125,0.0625]. The64 composition reversal pairs give
[0,0.0170898]. These are descriptive intervals; the frozen gates use point
thresholds. The150,188 derived swaps are not independent semantic samples.

**Mechanism evidence.** C trains1,774,464 encoder parameters and1155 head
parameters; its first encoder/head gradient L1 sums are1668.22/7.76567, with
zero frozen-prefix gradients and identical before/after frozen-prefix hashes.
A/B encoder hashes remain identical. A separate actual-encoder smoke alternates
backward and inference twice with nonzero final-layer/head gradients.
The generic Mistral-regex warning does not justify rewriting this tokenizer:
524 criterion/field inputs match the pinned native SentencePiece token IDs
exactly, including accent/currency fixtures. Cache-only loading avoids an online
startup timeout without changing any model/tokenizer artifact.

**MEASURED, independent reconstruction:** all four stage/final G0–G7 sets,
1,440 pair inputs, observed matrices, exact coefficients and integer decisions
are reproduced. Final proof covers13,152 decisions,26,304 reversed/shuffled
permutation checks with zero mismatches, and150,188 causal rows. Maximum
head-softmax discrepancy is2.214e-7 across the reconstructed stages.
The first verifier run failed because its independent regex retained leading
field-key whitespace; that failure is preserved, only the verifier key stripping
was repaired, and the original measurement was not rerun or altered.

**INFERENCE:** this literal-only stock-pair interface, including the declared
one-layer adaptation, did not earn alias transfer. This does not prove the
semantic information is absent, that more head width solves it, or that
orientation is robust on the two development successes.

**HYPOTHESIS / next branch:** task-pretrained small relation/NLI encoders may
provide a stronger interface. A controlled open-encoder bake-off is selected
from the development failures, with license/provenance, size, class labels,
context and multilingual scope checked before preregistration. It requires a
new sealed final; CBF-6 final outcomes do not tune later arms. B-STEF remains
blocked. The autonomous endgame continues.

Artifacts and complete hashes are in
`research/cbf0/schema_relation_result_manifest.json`; raw features, tokens,
matrices, training histories/checkpoints, decisions, swaps, intervals and
verification stay in the amended data root. No frozen Vey-2 runtime changed.

### CBF-7 preregistration: relation-pretrained pair interfaces (before outcomes)

CBF-6 selected A from development because its two correct axes (2/32) matched
B while C resolved zero; final A earned10/128 axes and3/128 signed atoms with
113 UNKNOWN. That failure justifies one controlled stronger-interface screen.

Smallest three-label NLI cross-encoder is pinned:
`cross-encoder/nli-deberta-v3-xsmall` revision
`a150876415327c80daeff35ca6f68f5ed8cf5c24`, model.safetensors SHA256
`4e4fc4977f8d29d2a164255c8f69b9d6c158deeb309bb5e70445b94666ccd9e9`,
70,831,107 floating metadata parameters,384-wide/12-layer,512-token context,
SNLI+MNLI tuned, declared Apache-2.0 weights with class order
contradiction/entailment/neutral verified against config. Optional
`cross-encoder/nli-deberta-v3-small` revision
`fa2804872c3b4bd748f38c0185cc85775361e735`
(`ebc79588dd73ccfb6a3f6078519cfbf512c5305384c5ea1845bc71cd32216e86`,
141,897,219 floating elements) triggers only when no small arm passes.

**License class:** the NLI checkpoints are `conditional/review`: SNLI is
CC-BY-SA4.0 and MNLI mixes share-alike/US public-domain sources, so an
Apache-2.0 weight tag alone is not shipping clearance. A research pass does
not clear shipping release. Moritz xsmall is excluded as a binary
entailment classifier; ModernBERT/mmBERT/mDeBERTa NLI variants are declared
NC-derived fine-tunes (research-eval-only, never shipping training), and a
`2048` ModernBERT NLI config or `128` mmBERT training limit must never be
silently substituted with stock8192.

**Arms, all before final:** stock bare reuses the exact CBF6 A checkpoint
as a baseline; new controls are stock generic-hypothesis CLS, NLI bare CLS,
NLI generic CLS, NLI generic native frozen `ContextPooler`, and the native
classifier with name-reordered labels (contradiction→lower, neutral→unrelated,
entailment→higher). The mapping is a falsifiable empirical bridge for authored
unconditional monotone one-field preferences, not a universal logical
reduction. Learned heads keep one width→3 linear R head (no width tuning).
All encoders, native poolers and native classifiers stay frozen/eval with
before/after parameter hashes, native-forward versus pooler/classifier parity,
and native SentencePiece token parity.

Development data remain the old32 aliases and50 compositions plus held literal
templates; CBF-6 final texts serve only as lexical-exclusion reference and
never as development. A fresh128-atom/128-composition final (64 opposite-meaning
pairs, exact-compiler decomposition verified) freezes before any scoring, with
an independent blinded meaning audit. Literal training/validation/G0 cohorts,
states, exact compiler and executor are byte-identical to CBF-6.

Selection: first eligible all-gate pass in arm order among the small arms;
otherwise highest development joint atom, then alias decisions, then lower
parameter count. CBF6 stock bare cannot promote, but a generic-hypothesis stock
pass is a materially new interface result and may. Selection persists before
any final text is encoded; final opens exactly once.

Same G0–G7 gates and64 reversal-pair bootstrap apply. Replication for a
learned head retrains seeds11/13; a native frozen classifier has no training
seeds, so promotion requires two independently authored blinded confirmation
pools — inference repeats are not replication. NC-derived variants never enter
shipping training; B-STEF stays blocked. Candidate roster, provenance and
license register are in `research/endgame/relation_encoder_candidates.json`;
the frozen protocol is `research/cbf0/relation_pretrained_protocol.json`.

**Pre-outcome execution amendment.** The first corpus preparation ran before
the generator was committed, contrary to the intended commit-before-build
order. Its original files are retained. The generator and scoring code are
committed before deterministic regeneration proof or quality measurement;
no original preparation revision is invented. The independent blinded review
accepted128/128 atomic meanings with zero ambiguities.

The pipeline worker ran a literal/live-pair native-library parity probe.
Its accompanying “no model output” statement was incorrect; native forwards
did occur. No CBF7 development/final quality evaluation was performed.
Source review found dict/tuple cache mismatches, missing NumPy and stock-count
metadata assumptions, classifier-versus-encoder output misuse, native
double-pooling/row/device errors, a reversed parameter-count tie-break,
duplicated final-cache tags, and an undefined final interface. Repairs precede
quality outcomes; gates, data, training and pair formats remain unchanged.
The protocol's alias-input wording is clarified: aliases are criterion inputs,
never inserted into schema hypotheses. Native SentencePiece is the tokenizer
comparator, not a Transformers5 “slow” alias backed by the same fast tokenizer.
Model residency uses the shared `/tmp/vey-gpu.lock`; mock-wiring smoke fixtures
are removed in favor of an actual frozen-pair preflight and saved-artifact
reconstruction. These are lineage/correctness changes, not a research result.

The first parent preflight failed at the immutable evidence writer before any
model load: an in-place audit-metadata amendment was correctly rejected.
That failure and original corpus remain intact. Preparation now regenerates
byte-identical final rows into `relation-pretrained-v1/preexecution-amended`,
copies the original blinded labels, and writes corrected lineage once. The
literal execution probe uses distinct attempt directories; no evidence writer
is bypassed to overwrite an earlier attempt.

**MEASURED preflight:** commit `393a5a6` regenerated the final atom/composition
files byte-identically and preserved128/128 accepted blinded meanings. Actual
stock/NLI-xsmall execution checked16 literal pairs total: native SentencePiece
IDs matched, cache reloads matched, both encoders retained their full parameter
hashes with zero gradients, and the R heads had positive gradients. The NLI
native classifier replay was bitwise exact on two pairs; cached-pooler/classifier
probabilities differed from full native inference by at most `5.9605e-8` on
eight pairs. Actual loaded parameter counts are70,682,112 stock and70,831,107
NLI. This proves execution/parity only, not semantic transfer or calibration.
Hashes and the retained failed preflight are recorded in
`research/cbf0/relation_pretrained_preflight_manifest.json`.

### CBF-7 measured result: relation pretraining helps, grounding remains unearned

**MEASURED: negative fresh-transfer screen.** Measurement code/preflight was
committed at `2872d54` before outcomes. All eight development arms failed the
joint gate set, so the preregistered larger-model trigger fired and the frozen
fallback selection chose `nli_xsmall_native`. Its development joint score tied
the learned higher-hypothesis CLS head at20/32; alias decision accuracy
0.6973 versus0.6895 broke the tie. Selection was persisted before final
tokenization; no final result selected a different arm.

| Development arm | G0 relation | Joint signed atoms | Sign given correct axis | Alias decisions | Compositions |
|---|---:|---:|---:|---:|---:|
| Stock bare, exact CBF6 reuse | 0.8906 | 2/32 | 2/2 | 0.0625 | 0.0138 |
| Stock higher CLS | 0.7813 | 2/32 | 2/2 | 0.0625 | 0.0000 |
| NLI xsmall bare CLS | 0.9688 | 17/32 | 17/25 | 0.5313 | 0.3875 |
| NLI xsmall higher CLS | 0.9688 | 20/32 | 20/20 | 0.6895 | 0.5863 |
| NLI xsmall higher pooler | 0.9922 | 19/32 | 19/19 | 0.6309 | 0.4388 |
| NLI xsmall native | 0.5000 | 20/32 | 20/21 | 0.6973 | 0.5813 |
| NLI small higher pooler | 0.9531 | 18/32 | 18/18 | 0.6191 | 0.5150 |
| NLI small native | 0.5859 | 11/32 | 11/11 | 0.4395 | 0.2900 |

The selected native interface's fresh results:

| Gate | Result | Required | Outcome |
|---|---:|---:|---|
| G0 literal relation | 64/128 =0.5000 | >=0.95 | FAIL |
| G1 axis | 51/128 =0.3984 | >=0.85 | FAIL |
| G2 sign given correct axis | 49/51 =0.9608 | >=0.90 | PASS, point gate |
| G3 joint signed atom, primary | 49/128 =0.3828 | >=0.80 | FAIL |
| G4 alias decisions | 1066/2048 =0.5205 | >=0.80 | FAIL |
| G5 composition decisions | 735/2048 =0.3589 | >=0.80 | FAIL |
| G6 teacher-changing correct-new | 0.3445 of150,188 | >=0.80 | FAIL |
| G7 exact literal/permutation path | 1.0000 | 1.0000 | PASS |

G0 matched-field classes were32/32 correct, but unrelated classes only32/96:
the native NLI bridge does not reliably express this preference-relatedness
interface. UNKNOWN occurred in11/128 atoms. Conditional sign excludes
wrong-axis/UNKNOWN cases and is not overall polarity robustness. Causal
correct-new does not require the student to change; changed-to-new was0.2195,
both-endpoints-correct0.1236 and student-changed0.7598.

Nominal95% descriptive64-reversal-pair bootstrap intervals (10,000 draws,
seed0): axis[0.2891,0.5078], joint[0.2734,0.4922],
sign[0.8980,1.0000], alias[0.4355,0.6084],
composition[0.3018,0.4180]. The G2 point pass does not exclude a conditional
sign rate below0.90. These are a fixed four-field wording assay, not confidence
bounds on arbitrary-family transfer or neutral competitor non-inferiority.

All155 literal cases and4960 literal decisions were exact with zero neural
literal callbacks. The final contains13,152 decisions and26,304 permutation
comparisons (24,256 numeric,2048 UNKNOWN), with zero score/winner mismatches.
The independent CPU verifier reconstructed every arm's matrices, head/native
parameters, original-order normalization, epoch selection, compiled programs,
decisions, causal rows, gates, selection and fresh intervals; all checks passed.
It also re-tokenized inputs, checked model/source hashes and confirmed
byte-identical baseline reuse. Verification required correcting its annotated
tag lookup and safetensors parameter accounting: integer position-ID buffers
and inactive/alias/task tensors are not live floating model parameters.
Original verification failures remain retained; no quality measurement reran.
Native normalizer metadata strings `zeros`/`ones` are descriptive sentinels,
not SHA-256 digests; the verifier checked the actual arrays and archive bytes.

**INFERENCE:** the NLI checkpoint package materially improves shared development
grounding relative to stock, but this is not an isolated pretraining-algorithm
causal result. Generic higher hypotheses improve conditional orientation while
field recovery remains limiting. More model capacity did not help development.
CBF6 and CBF7 fresh pools differ; their final scores are not a paired delta.
No calibration, certificate, speed, broad OOD or competitor win is established.

Artifacts: `relation-pretrained-v1/preexecution-amended` under the decisionmix
data root; immutable results SHA-256
`75be4cf4345a33af142580a1b69008cfef56a618ec507686d61e72f418b3418e`,
independent verification
`2bcf7fb4907f2a194ce02a869be7a5572de9f7b49721f9354a2c31d68aac005e`.
`research/cbf0/relation_pretrained_result_manifest.json` records lineage.
Verdict: `C7_NOT_EARNED_ON_FRESH_CORPUS`; no replication trigger, no B-STEF.

### CBF-8 preregistration: targeted field-domain support

**HYPOTHESIS, before outcomes:** commit `b651301` freezes a two-by-two control:
literal-only versus literal-plus64 training-only semantic atoms, and generic
higher-field hypotheses versus the same hypotheses with fixed field
definitions. The NLI-xsmall encoder, raw CLS interface, linear R head, loss,
seed7 and400-epoch recipe remain unchanged. The literal/generic arm reuses the
CBF7 learned checkpoint exactly; three eligible arms isolate data, definition
and interaction effects. Thirty-two additional semantic validation atoms are
diagnostic only; checkpoint selection still uses unchanged literal validation.

Different authors supply training and fresh128-atom/128-composition final
corpora; blinded meanings and lexical exclusions freeze before model execution.
Definitions contain no evaluation aliases. Old development is selection-only;
old finals are exclusion-only. All controls run; one development-selected arm
opens final, with unchanged G0–G7 gates and descriptive pair-cluster intervals.
A clean support failure retires literal-only fixed-registry grounding as a
general solution and opens criterion-conditioned Ephemeral Atoms, not another
arbitrary head-width tweak. A pass requires seeds11/13 and fresh-family/OOD
utility before B-STEF. This screen does not complete the endgame mandate.

**Pre-outcome corpus correctness check:** the authored training generator is
preserved at `ed6431d`. Parent review replaced ambiguous delivery-fee and
energy-unit-tariff proxies with complete initial-ownership and total
continued-use payments, and made mechanism faults explicitly task-stopping.
The first pure build then rejected `once` as a reserved exact-weighting token;
the failure is retained under `schema-support-preparation-attempts`.
Commit `65db021` changes that atom's wording without changing its meaning or
the compiler. The corrected builder produced64 training atoms/32 reversal
pairs and32 diagnostic atoms/16 pairs, with disjoint quantity/templates and
byte-identical literal source cohorts. Training bytes hash
`e58ca21f82c805b103741077ebc96a7227e16a3d6f724ab1abd851e1190d28b1`;
diagnostic validation
`77510b5a16001dbb4faf03755efa13c7a7b551eb5ef5d741ef8b7a69f617b69e`.
No CBF8 model output or final quality result was obtained.

**MEASURED source-custody correction, before corpus freeze/models:** the
committed `bf4d050` source-only smoke rejected a nonexistent historical byte
hash for CBF7's learned-head training metadata. CBF7 did independently
reconstruct its selected epoch116, unweighted literal-validation CE and frozen
parameter hash, while cryptographically checking the checkpoint, normalizer,
features, inputs and lineage. CBF8 now separately labels its new metadata byte
pin and checks those reconstructed fields; it does not rewrite CBF7 evidence.
Corrected execution passed all6 baseline references and16 historical exclusion
files/1178 rows. Failure and source-hashed passing proof are retained under
`schema-support-preparation-attempts`; no fresh final was generated or model
loaded in either source-only attempt.

**MEASURED pre-execution audit, original root retained:** preparation at
`59daa6e` passed64 semantic training/32 diagnostic atoms and128 fresh
atoms/128 compositions, then two opaque automated reviewers judged352 phrases.
They accepted350 and flagged both directions of one training shutdown pair:
an internal shutdown need not stop the intended task. A raw composition
judgment also invented factor2 for two unweighted clauses, despite correct
semantic components. The actual execution guard refused the ambiguous audit
before model load. No encoder/head outcomes were obtained.

The training pair now explicitly states shutdown before assigned-task
completion. A prospective amended root preserves all original files and
judgments. Exact audit arithmetic now follows the frozen syntax parser plus
independently judged axis/sign; raw rater vectors and discrepancies remain
recorded. Its actual352-row smoke retained both semantic blockers and rendered
the unweighted vector `[0,0,1,-1]`, without deriving weights from sealed gold.
Gates, model, recipe and final wording are unchanged; new blind review is
required before execution. Original definition-guard failure is also retained:
parent `b5a0d68` had copied an incorrect preparation string; `59daa6e` restored
both guards to the authoritative protocol's `maintaining it`, not a new
definition.

**MEASURED amended blind audit and tokenizer correction:** the new opaque
reviews accepted all352 meanings with no vector discrepancy. One reviewer
transcribed an opaque ID incorrectly; the original output remains intact
beside the reviewer's complete ID-corrected artifact. No semantic judgment
changed.

The first real CUDA preflight then failed before any encoder forward or head
training: Transformers5.17's `DebertaV2Tokenizer` lacks the legacy special-token
builder methods used by the parity checker. Inspection of the actual pinned
tokenizer established `[CLS]:0 A:0 [SEP]:0 B:1 [SEP]:1`; native SentencePiece
special IDs are1/2. Independent rendering of that template passed actual
generic/defined pair-ID, segment-ID and attention-mask checks. The failed
root and proof remain retained. A new tokenizer-amended root freezes corrected
code; blind judgments may be reused only against byte-identical packets.
No corpus wording, neural input, learning recipe, gate or selection rule
changes. This tokenizer check is not a capability result or a completed
forward-and-gradient preflight.

**MEASURED tokenizer-amended audit failure, still before model outcomes:** the
resumed blind half flagged 17 compositions solely because its packet omitted
the frozen unit-coefficient convention for `A; also B`. Its reviewer confirmed
both terms' meanings were clear. Three accepted rows were also transcribed as
atomic despite containing two terms; the assembler stopped on its kind/factor
assertion. These raw artifacts and the failed command remain preserved.

A prospective packet correction states the existing exact grammar and requires
whole-question kind checks. It does not change criterion text, semantic labels,
compiler arithmetic, model inputs, learning recipe or any gate. All genuine
meaning ambiguity still blocks execution. The new root requires fresh complete
opaque review; no parent-forced acceptance or old-ID remapping is allowed.

### CBF-8 result: domain support improves points, not all-gate transfer

**MEASURED:** preparation `6f9dc32`, measurement `f436741`; the frozen
two-by-two protocol and352/352 freshly accepted blind judgments precede scoring.
Frozen FP32 NLI-xsmall, unchanged linear relation head and exact state/compiler:

| Development arm | Joint atoms | Alias decisions |
|---|---:|---:|
| literal/generic, exact CBF7 baseline reuse | 20/32 (.625000) | .689453 |
| literal/defined | 25/32 (.781250) | .822266 |
| semantic/generic | 25/32 (.781250) | .818359 |
| semantic/defined | 27/32 (.843750) | .867188 |

All eligible development arms fail at least one gate. Frozen joint/alias/order
selection chooses `semantic_defined`; exactly that arm opens the fresh final.

| Fresh final endpoint | Result | Gate |
|---|---:|---|
| G0 held literal relation, not exact compilation | 77/128 (.601563) | FAIL |
| G1 axis | 114/128 (.890625) | PASS |
| G2 sign given correct axis | 105/114 (.921053) | PASS |
| G3 joint signed atom | 105/128 (.820313) | PASS |
| G4 alias decisions | 1725/2048 (.842285) | PASS |
| G5 alias compositions | 1553/2048 (.758301) | FAIL |
| G6 teacher-changing correct-new winner | .765148 | FAIL |
| G7 exact literals/permutations | 1.000000 | PASS |

Five atoms resolve UNKNOWN. G0 matched fields are31/32 but unrelated fields
are46/96. Composition compiler accuracy is86/128. Causal correct-new uses
150188 teacher-changing rows, without requiring the student to change;
changed-to-new is .683763 and both endpoints correct .584108. These derived
swaps are not independent semantic directions.

Nominal95% reversal-cluster bootstrap,10000 draws seed0: final joint
[.742188,.890625], alias decisions [.772449,.905273], composition
[.688477,.824707]. Gates were preregistered point screens. Development uses16
matched reversal clusters: mean semantic-data joint effect .109375
[0,.234375], definition effect .109375 [-.046875,.281250], interaction -.093750
[-.312500,.125000]. The generic-format semantic-data simple effect is .156250
[.062500,.281250]; these descriptive intervals do not establish broad semantic
or competitor superiority.

**MEASURED independent reconstruction PASS:** the actual CPU CLI recovered
all four development arms and the selected final, all semantic audit weights,
160 final relation matrices,13152 final decisions,150188 teacher-changing rows
and26304 permutation comparisons. Final head-mass maximum error is6.98e-7.
All155 exact literals/4960 literal decisions pass with zero neural callbacks.
Independent SentencePiece reproduces every persisted token/segment/mask.

The never-run verifier required corrections to actual schemas, hash aliases,
shared old G0 isolation, batch dimensions and missing reconstruction helpers;
incidental caption/prose assertions were removed. Preparation and measurement
commits differ, but all model-affecting source bytes match at both commits and
the current source. Reencoded FP32 features are not byte reuse: preflight maximum
drift4.83e-6; the528 matching new-generic/old pairs differ by at most2.30e-5.
Only the baseline arm claims exact old-cache reuse. No corpus, model, recipe,
prediction, gate or final-selection change occurred.

Proof SHA `10aa4a1822fabf16f413c0d2f930ca77645398f52ef04c77a6c42901c11a2752`;
`research/cbf0/schema_support_result_manifest.json` records108 artifact hashes
and17 preserved verifier failures. **INFERENCE:** this supported frozen
relation interface remains inadequate for the complete transfer gate; this is
not evidence that semantic information is absent or every encoder fails.
No replication or B-STEF is earned. The frozen failure branch opens ECA-1
criterion-conditioned evidence pages, not another registry/head-width retry.




### Neutral endgame suite: source-first prospective freeze

**HYPOTHESIS, no rows or quality outcomes acquired:** the original neutral
draft required tens of thousands of new human-authored sessions/documents,
millions of candidate reviews and51 native-review teams. No such workforce or
budget is authorized. That unexecuted draft is preserved under
`decisionmix/protocol-drafts/neutral-human-only-v0.json`; it is not a scientific
or access hard block.

The revised protocol pins public source candidates and uses workflow-sealed
public tests with explicit exposure disclosures, plus bounded synthetic
objective controls reported separately. Public familiarity is not universal
freshness; source labels do not establish capabilities outside their task.
Main HelpSteer2 aggregates do not identify annotator distributions; the pinned
release separately supplies filtered individual ratings. All21 dimensions,
51-language scope, K1000, natural8192-token evidence, shared exact workflow
code and strongest appropriate competitor routes remain required.

Source grouping, allocation seed/caps and input-only dedup are fixed before
acquisition; realized hashes/counts, legal grants and full endpoint registry
must freeze before study-specific model choices. DEV-only precision screening
precedes final: insufficient independent families or multiplicity resolution
remains inconclusive, never a zero-variance equivalence shortcut. Missing
Jev competitive-research authorization remains explicit and cannot end other
reachable work. Protocol: `research/endgame/neutral_benchmark_protocol.json`.

**MEASURED metadata correction, before split rows or model outcomes:** the
pinned HelpSteer2 card's Disagreements section and Hub file listing establish
retained individual0–4 arrays after outlier filtering, contrary to the earlier
means-only inference. They permit a filtered-rater probability target, not
unfiltered population calibration; actual coverage/joins and prompt rights
remain unresolved. Its preference resource also releases unprocessed signed
human preference strengths. Human justifications remain label provenance,
never model-visible evidence. Source:
https://huggingface.co/datasets/nvidia/HelpSteer2/blob/990b2711a36180dd19d9c94b8627844866f8982a/README.md

The MASSIVE card documents individual grammar0–4/spelling0–2 judgments,
while intent/slot judgment categories are not ordinal scales. Per-locale
coverage and English-seed availability remain to be censused. This offers a
translation-quality Score source, not general arbitrary-criterion gold.
Public card examples were encountered as documentation, not used for
training or choices; all source exposure remains disclosed. Source:
https://huggingface.co/datasets/AmazonScience/massive/blob/ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740/README.md

**MEASURED source-custody failure, before model execution:** the first
Banking77/MASSIVE acquisition froze all eight raw-file hashes and every
archive-member hash, then failed while parsing MASSIVE. The SQLite storage
guard unpacked the one-value `PRAGMA page_count` result into two variables.
The correction reads `page_count` and `page_size` separately. The original
root, input snapshots, raw files and failure record remain intact; the
amended run uses a new root and byte-verified immutable raw-file reuse.
Source scope, grouping, allocation and all quality gates are unchanged.
The acquired MASSIVE1.1 archive SHA-256 is
`4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577`.
No source grouping, multilingual quality or competitor parity is established
by this failed attempt.

**MEASURED completed acquisition and independent reconstruction:** the
amended pipeline acquired and grouped 855,654 rows: 13,083 Banking77 rows and
842,571 MASSIVE rows across exactly 51 locales. Each locale contains
11,514 train, 2,033 validation and 2,974 test rows. All 51 locale descendants
remain joined by original utterance ID. No model or quality evaluation ran.

The reporting smoke caught a unit-label defect: `independent_group_count`
counted constituent source-lineage groups, while caps apply to connected
components. Corrected reporting on the actual retained groups preserved every
membership hash, split and row count. The 29,604 source groups form 22,169
components. Source-specific connected-component counts:

| source | train | dev | calibration | confirmation | unused |
|---|---:|---:|---:|---:|---:|
| Banking77 |4096|1024|1024|3078|3819|
| MASSIVE |3652|1024|1024|1856|1572|

Original manifests remain intact beside a derived count-unit correction.
Neither lineage-member counts nor locale descendants increase the allocation
unit count; unresolved author/template lineage still prevents a claim of
statistical independence. The actual English MASSIVE rows contain no
`judgments` field or individual rating records. Non-English judgment
availability is retained as source metadata, not fabricated 51-language
Boolean/Score gold or probability-calibration evidence.

The retained independent verifier completed on 2026-10-04 with all 14 checks
passing. Its report SHA-256 is
`e16563bb35658b9d94a054ec6cb987754c800d85d2fe8b29f0824caff378d3a7`;
exercised and current verifier bytes both hash to
`d224e9a7435348f0c00a1220897698b691ca0ca850fd59663076b1157de43f40`.
It reconstructs raw/native payload identity, all 38,393 grouping edges,
complete prefix-search coverage, connected-component allocation and absence
of cross-split component leakage. This record corrects the stale pending
status by pinning the existing proof; no acquisition or verifier rerun was
needed. No model quality, general Boolean/Score target mapping, natural
long-context eligibility, statistical independence or competitor parity
follows from source custody. Source-to-DecisionIR/workflow and metric-family
freezes, plus tokenizer pinning, remain required before neutral evaluation.

**MEASURED QASPER acquisition attempts:** the first stopped on anonymous
metadata transport. The second acquired and SHA-sealed both archives and all
three selected members before parsing 888 train, 281 validation and 416 test
papers; their question counts are 2593, 1005 and 1451. It then stopped before
completed grouping/allocation because the QASPER schema and shared prefix
builder both declared the same gram-frequency index.

The shared builder is now the sole index owner. A prospective amendment pins
the failed attempt's pre-parse receipt and permits only byte-verified reuse;
the actual four-object hardlink smoke preserved 14,716,506 bytes and every
SHA without a network call. The failed databases and manifests remain intact.
At this acquisition stage, completed grouping/allocation and independent
reconstruction had not yet been established. Nullable annotations and gold
evidence remained separate from model state; primitive quality and natural
token eligibility were unmeasured.

**MEASURED QASPER custody completed:** the corrected replay sealed 1585 papers,
5049 questions and 7993 individual answer-annotation records. Input-only
grouping found 1585 components and no accepted cross-paper edges. Allocation
contains 532 train, 319 development, 318 calibration and 416 confirmation
components, with no unused papers. Full-set integer Jaccard rejected all
122,984 prefix candidates. The acquisition report alone did not establish
independence, natural 8192/16K eligibility, primitive quality or competitor
non-inferiority.

**MEASURED independent QASPER custody: PASS, 16/16 checks.**
`neutral_paper_verify.py` independently reconstructed every native paper,
question and annotation payload, field presence/type census, document identity,
complete integer-Jaccard prefix join, component closure, allocation reasons,
caps and descendant/membership hashes. It verified both archives, all three
selected members, four raw hardlinks and historical snapshots without running
the source loader or acquisition pipeline. The private database stayed
byte-identical. The proof SHA is
`7a1120b239e222ada0e6db33ec0f5e28633ed68267dccea95c92b9dff3ea3abc`;
its exact path and exercised verifier hash are in
`research/endgame/neutral_source_custody_result_manifest.json`.

Two failed verifier receipts remain preserved: a branding-text check mistaken
for a license invariant, and a scratch gram-count comparison before index
construction. Both were verifier defects, not source repairs. Counts and
allocations above were independently recovered; no model-visible gold,
Boolean/Score quality, statistical independence, natural-token eligibility or
shipping/underlying-paper republication authorization follows from custody.

### ECA-1 preregistration: criterion-conditioned Evidence Pages

**HYPOTHESIS, no model outcomes:** commit `4697017` freezes
`research/endgame/ephemeral_pages_protocol.json`. This branch replaces the fixed
four-field registry with question-conditioned reads of cached semantic pages.
The exact lane retains facts, factors, composition, ties and UNKNOWN propagation.
Frozen NLI-xsmall page/query features feed a rank64 conditional reader; paired
controls are a real joint cross-encoder, frozen-cosine attention, training-only
lexical models and a query-blind page regressor. No adapter or head-width search.

The prospective authored assay spans20 families with distinct A/B/C properties,
held wording, fields and four whole families. Whole worlds are split units;
320 final worlds balance64 property exposures per family. Independent opaque
meaning review must accept every unique page/question before encoder capture.
Missing and contradictory evidence, polarity, composition, candidate counts,
page changes/erasures, zeroed state/question features and uniform attention are
paired interventions, not extra independent observations.
World-cluster intervals condition on these fixed authored rubrics/wordings and
random grade assignments; they do not bound new authors, templates or future
families. This prospective clarification changes no assay count, recipe or gate.
Prospective implementation freeze precedes review: `785c25b` commits the
rank64 reader/controls, hashed frozen-encoder feature capture with the
`vey.eca.selection-calibration.v1` final-opening receipt, and the authored
builder. `--export-authorship-seed`/`--prepare-source` wrote60 properties,
300 rubrics,4500 page wordings and1800 questions with zero cross-split text,
template or normalized-text overlap. The SHA-bound opaque packet (6300 items,
zero grade/family/split/orientation/target leakage) and separately sealed
target key exist under the private root. At this pre-review checkpoint, no
encoder forward, head training, calibration or model outcome had occurred;
the independent two-reviewer meaning audit was still pending.

**MEASURED Banking77+MASSIVE custody reconstruction PASS:** after two
verifier-only join corrections (`c230d95` — `raw_rows.final_split`/
`component_id` were never populated, so both the split row-count and the
source-row comparisons must join through `source_groups`), the independent
CPU CLI passed all15 checks over the full frozen custody tree:
29,604 source-lineage groups (13,083 Banking77 rows; 16,521 MASSIVE ids),
855,654 raw rows across the exact51 frozen locales, 22,169 connected
components under the proved-complete exact/char5gram-Jaccard graph, and the
seeded train/dev/calibration/confirmation/unused allocation replaying the
historical report exactly. Verification receipt
`e16563bb35658b9d94a054ec6cb987754c800d85d2fe8b29f0824caff378d3a7`. No model
output, gold mapping, Boolean/Score quality, statistical independence or
shipping clearance is earned by custody; native MASSIVE judgment metadata
stays byte-checked and unmapped.

**MEASURED ECA-1 corpus built after accepted independent review:** two gateway
reviewers (earlier attempts died on provider usage limits and never opened the
packet) each judged all6300 opaque items; agreement6300/6300 with zero
ambiguous/reject, residual judgment notes retained. The accepted receipt binds
source`1fb1f660`, packet`adddbb2e`, and both raw review files. The builder
then streamed75,264 canonical DecisionIR rows over768 worlds:
train12,544/validation6,272/calibration12,544/development12,544/final31,360.
Structure verified against the frozen protocol: field codes A(train/val/cal),
B(development), C(final) only; final covers all20 families x64 property
exposures with the last4 families held to final only; K probes2/8/16/32/64;
two candidate permutations, rename, page-reorder, two-term question-reorder,
and erase/contradiction/grade-change paired interventions per main query;
22,272 rows are authored UNKNOWN gold (missing/contradiction); generated page/
question hash overlap across splits is zero; exact fact blocks never enter
semantic pages. Construction fixes (`4e755d1`) were outcome-blind: eager-IR
validation on intermediate rows (k=2 gold/evidence, erasure evidence), a
dict-vs-Term type error in question reorder, and missing per-intervention id
distinctions; each fix re-prepared byte-identical source/packet (packet SHA
unchanged), so the completed reviews stayed bound. At this build checkpoint,
no encoder forward, head training, calibration or model outcome had occurred.

**Measured ECA-1 prelaunch checks:** feature capture exposed a candidate-renaming
provenance error: page-owner values must be remapped, whereas candidate-indexed
maps remap their keys. The old corpus was preserved and rebuilt before any
encoder forward. Authored source, inventory, opaque packet and sealed-key
SHA-256 values remained byte-identical. The feature module's stale protocol pin
was synchronized to the committed optimizer-example clarification
(`785c25b`); the protocol itself did not change.

Training/validation/calibration/development capture persisted respectively
75,520/37,760/75,520/75,520 term-candidate records. Actual unique question/page/
cross-pair counts were 72/180/4,824; 72/180/4,461; 72/180/4,819; and
96/240/6,393, with 160/149/160/211 encoder forward calls. Each phase retained
input IDs/masks, mmap features, hashes, and identical before/after encoder
parameter hashes. Final features remain receipt-locked.

The masked-loss smoke reproduced finite forward losses but nonfinite gradients
when undefined NaN targets were subtracted before masking. Selecting defined
targets before arithmetic fixes the backward path without changing the
objective. Structural no-page UNKNOWN now contributes exactly zero BCE rather
than evaluating the undefined floating operation BCE(-infinity, 0). All four
neural readers passed finite-loss, finite/nonzero-gradient and no-page checks;
full-batch and uneven 17-record accumulation losses/gradients matched. The
streaming normalizer matched the existing shared question/page policy. A
two-epoch reduced run exercised all five controls in explicitly marked smoke
artifacts, not experiment outcomes. A real cached-runtime smoke observed one
state encoding, question-only new-query work, zero forwards for a repeated
question, and packed/live score parity. It earns no capability, latency,
calibration or certification claim. The full frozen 400-epoch run has not yet
started at this implementation checkpoint.

The evaluation implementation uses exact equality of predicted and gold
maximal-winner sets for the frozen primary metric. Concrete winner-in-gold
accuracy is reported separately, including for causal correct-new outcomes;
it does not replace the stricter primary. A synthetic boundary smoke exercised
this distinction, required-child UNKNOWN, integer exact arithmetic, and the
sparse-discordance NI guard. Eight identical worlds correctly remained
inconclusive under that guard. No final examples were opened.

Promotion requires all frozen quality, UNKNOWN, exact/invariance and causal gates,
an internal cross-control NI bound at a5-point margin, and corrected evidence
of benefit over lexical/zeroing controls. This is a mechanism screen, not the
endgame1-point competitor NI test. Passing requires seeds11/13 and fresh neutral
transfer before primitive/product promotion; failure routes to a conditional
interface audit or one preregistered final-layer adaptation, never final tuning.
The checkpoint remains conditional/review. Canonical Vey-U IR dependency hashes,
JSON serialization, audited corpus construction and actual cache counters were
exercised. Quality, useful calibration, certification, end-to-end performance
and competitor superiority remain unmeasured at this implementation checkpoint.

**MEASURED — first prefinal execution, retained negative evidence.** All five
controls completed the frozen recipe. Development primary exact-set accuracy:
pages0.131836, cross0.334961, cosine0.117188, lexical0.633789 and
query-blind0.332031. Pages ordinal MAE0.314524, attribution0.248779 and supported
coverage0.471875 fail their gates. This is opened development, not fresh
confirmation. The original selection/calibration receipt SHA-256 is
`5d6b4718a75e111bf4e281a6bf488c245b16aff96133f27c4f5cca5eaded8f03`.

The grade-change diagnostic is invalid: every prefinal replacement preserves
the teacher maximal set (train1536, validation768, calibration1536,
development1536 cases). Improving the worst candidate by one grade supplies
zero teacher-changing cases. `ephemeral_pages_grade_amendment.json` prospectively
repairs this construction before any final opening: enumerate exact
teacher-changing single-page grade replacements, prefer disjoint winner sets,
then select by a fixed content hash without using model outputs. Original
corpus/captures/predictions/receipts stay retained; all non-grade rows and every
fitted checkpoint must remain identical. A400-case prefinal construction smoke
passed deterministic teacher changes and unchanged exact facts.

The unchanged calibration rule includes all calibration variants, so corrected
calibration is refit on calibration worlds only and resealed before final.
Existing encoder vectors are reused exactly. This repair cannot rescue the
already-failed non-grade development gates; no optimizer, checkpoint,
architecture, gate or final-driven selection changes are authorized.

The corrected prefinal assembly retained all non-grade rows and all34560
optimizer/validation-objective record tensors exactly. All5376 prefinal grade
variants now change the teacher maximal set. Existing encoder inputs were not
recomputed: only21 previously uncaptured question/page pairs were encoded
(train1, validation17, calibration1, development2), one forward per phase.
Four actual selected-checkpoint forward/backward smokes remained finite after
removing the obsolete masking helper. The old unamended receipt was rejected
before final data access.

A separate prospective four-definition adapter replays existing CBF8 exact
states and closed aliases/compositions. It is diagnostic only, not a new
field-free registry or selection input. Its literal preflight reproduced9920
original exact-state rows with zero neural resolver/encoder calls. Learned
closed replay remains pending corrected calibration at this checkpoint.

**MEASURED — corrected prefinal execution.** Independent reconstruction passed
all raw control/intervention predictions, calibration, typed composition,
clustered statistics and fitting-tensor byte checks. Corrected development
primary metrics are unchanged; pages/cross grade-swap correct-new rates are
0.114583/0.318359. The corrected immutable receipt SHA-256 is
`14933f3283b75ecea66d6048606162b35782477c225777fc4c80e8a0689df7cc`.
The real calibrated cache path and the closed replay both ran. Closed literals
remain100%; closed-final alias/composition concrete accuracies0.201904/0.117188
show severe semantic regression, not field-free success. No product cutover is
earned.

The winner comparator now uses exact score equality, not a1e-12 near-tie
tolerance. This changed zero of121338 opened non-UNKNOWN baseline choices;
near-zero unequal-score and genuine stable-tie boundary smokes passed before
any ECA final opening. Original code/artifacts remain hash-addressable.

### ECA-1 frozen-interface audit (prospective)

`ephemeral_pages_audit_protocol.json` activates the predeclared both-fail branch.
It separates learned relevance, ordinal extent, orientation and knownness with
privileged component controls; measures a finite FP64 linear grade ceiling; and
tests one canonical128-hidden conditional readout over cached question/page
features. Training/validation/calibration rules remain fixed. Only opened
development is scored; ECA final is excluded from this audit and cannot select
its head or recipe. No stronger encoder, arbitrary width search, neutral
capability, probability certificate, product promotion or B-STEF is authorized.
Any successful diagnostic requires fresh audited confirmation before promotion.

### ECA-1 final measurement and atomic-target invalidation

**MEASURED, retained under confounded supervision:** the sealed final was opened
once after corrected calibration. Independent reconstruction reproduced all
31360 DecisionIR outcomes, eight control/intervention outputs, calibration,
10000 world-cluster bootstrap draws and exact composition; no model forwards
were needed for verification. Evaluation SHA-256:
`c48acad896340915b7e906c9db70f22200f61dc186e2b27f37807863cc765d6d`.
The receipt's `verified_persisted_reconstruction` status establishes numerical
reproduction, not independent validity of the training target function.

| Control | Final exact-set atomic macro accuracy | 95% world-cluster CI |
|---|---:|---:|
| Pages | 0.216797 | [0.201917, 0.232235] |
| Cross | 0.258984 | [0.243164, 0.274788] |
| Cosine | 0.249609 | [0.232052, 0.267587] |
| Lexical | 0.396094 | [0.371704, 0.420311] |
| Query-blind | 0.357813 | [0.336741, 0.378281] |

Pages held-four-family accuracy0.228516, composition0.173438,
attribution0.286230, orientation0.606250, UNKNOWN recall0.472000,
precision0.761905 and grade-swap correct-new0.151823 do not pass.
Internal cross NI also fails: one-sided95% lower difference−0.060276
against the fixed−0.05 margin. The joint learned-benefit gate fails.
Supported coverage0.63125; K-probe choice accuracy remains
2:0.434375,4:0.236648,8:0.271875,16:0.256250,32:0.256250,64:0.256250.
Typed exact replay verified439040 integer cases and206080 identity-invariant
cases; the evaluator literal gate itself remains a parent-owned closed
regression, not an earned endgame result. No promotion, replication credit,
certificate or endgame capability credit is earned.

**MEASURED correctness defect:** the component truth audit found that the
projector copied whole-parent `metadata.known` into every child record.
Erasing or contradicting one required property therefore labels a separately
supported sibling UNKNOWN and masks its grade/orientation supervision. The
other parent requirement is not present in the atomic reader's question.
Independent owned-page truth reconstruction found2560 affected training,
1280 validation,2560 calibration and2560 development child records. The audit
opened no final records and made zero encoder/model calls. Audit SHA-256:
`3f6137500b2be51260195de0c9a76ed17086d09e2d9ac23d88a1e3c79278cff4`.

These results cannot retire the intended child-local interface or establish
its semantic ceiling. The component verifier's original parent-wide label
reconstruction and OLS selector inherit the same defect. Further conditional
readout fitting is held; its reduced smoke and gradient checks are not quality
evidence. Original checkpoints, projections, scores and receipts remain
unchanged and auditable.

**Prospective correction:** `ephemeral_pages_atomic_protocol.json` freezes
ECA-2 before any corrected fitting. A child's knownness and supervision masks
depend only on its own required property; whole-parent UNKNOWN remains the
exact conjunction of required child truth. All five controls refit under the
unchanged encoder, architecture, recipe, checkpoint/calibration rules and
gates. Prefinal IR/input strings and cached encoder outputs are retained.
The ECA-1 final is closed for selection. New independently meaning-audited
properties, phrases and worlds form a separate320-world final, sealed before
corrected evaluation. This remains an authored English mechanism assay,
not a neutral benchmark or completion of the many-axis endgame requirements.

### ECA-2 target-only custody repair

MEASURED: the first corrected repack used legacy cache lineage, unnecessarily
re-encoded existing cross-pair extensions, and recorded total known labels
instead of changed labels. Its prematurely launched full fit was terminated
before custody validation. The feature directory and partial run remain under
`features.superseded-reencoded-repack` and
`runs/seed7.superseded-unverified-custody`; the retention receipt has SHA-256
`717ca6bcce4d3a15bb2470ef339293cfda02b2bef3c8050205bb07afe417b4be`.
These artifacts earn no corrected quality evidence.

The replacement reuses the exact grade-corrected packed query, page, cross-pair,
page-mask, raw-grade and relevance files. Only child-local knownness,
grade/orientation masks, directed grades and orientation targets change.
Each of these five targets changes in 2,560 training, 1,280 validation,
2,560 calibration and 2,560 development records. Encoder forwards are zero.
Byte-identical prefinal IR copies have explicit parent-build provenance;
missing files never silently redirect to an old corpus.

MEASURED: a current-source projection scenario checked 264,320 child records
over 43,904 prefinal IRs. Each child agrees with its independent single-term
projection; parent knownness equals the conjunction of required children.
The proof retains its predecessor and has SHA-256
`bef8f43ece58d8b387d645fde8cb21b72ddd0d2c046623dd8b690e1070255432`.
The independent prefit verifier additionally reconstructed owned-page truth,
checked packed target layouts/padding, inherited token caches and retained
cross extensions, and compared frozen input bytes. It passed 9,874,072
assertions and 26,360,471 compared values. Its receipt has SHA-256
`cfaa475dd6c65d4a572387e9a8555806a848ae857e14c2babf1e4d70ac68459b`.
The initial verification failure from a mismatched projection-receipt field
remains retained separately.

Inference: these checks establish corrected supervision and artifact custody,
not semantic transfer or a quality ceiling. Corrected fitting must also bind
the independently reviewed fresh final pool, all capture hashes, environment,
encoder/tokenizer identity and source revision before optimization. Final
neural access still requires all five fitted controls and an immutable
selection/calibration receipt. Corrected quality results remain unmeasured.

MEASURED source gate: the two actual opaque reviews cover all 420 new-final
items. The first accepts all; the second accepts 412 and preserves eight
ambiguities about crispness, a lifting boundary, compensation-component count,
receipt-relative currency, reminder frequency and actual price revisions.
The merged 840 judgments produce a rejected receipt, SHA-256
`22a696fe7a92d097d457b096a87b127d97a226f6e2d58076e84343aea7518d51`.
The final builder rejects it with exit code 2 before creating any final IR,
decision/world ledger or build manifest. Corrected optimization remains closed.

`ephemeral_pages_source_clarification_protocol.json` preregisters eight
text-only clarifications before installation or new review. The proposed
source SHA-256 is
`0e986104f659a833a510863492aa367375d0e2850181f22f27af3440fb8b1d15`;
the outside-Git amendment SHA-256 is
`3abff15ee05cf99a612d1d01a95b59b5be3795f529fd6fd02b1aac2175ff0e25`.
Exact byte substitution preserves every other source byte, all 412 other
texts, property/source IDs, rubrics, grade positions and orientations.
The rejected source and all judgments remain retained. Two new independent
opaque reviews must accept the complete clarified packet before building the
fresh final pool; no semantic-quality outcome selected these edits.

MEASURED component-verifier correction: archived `a275466` rejects a valid
authored-but-locally-absent query because its family metadata intentionally
uses the source-world families. The corrected rule follows the frozen
generator, while ordinary queries still use every required authored property
even when evidence is erased. Before-failure/after-pass replay covers all
43,904 prefinal IRs and 264,320 children, including 2,240 absent-query rows,
with zero family-provenance exclusions and zero neural calls. Receipt SHA-256:
`d4c85e4bb44be97f54fd81f418c8dcc9d28327ffd9579432ef4abdee146a477e`.

MEASURED second source review: fresh independent reviewers accept 416/420
and 411/420 items, leaving an eleven-item ambiguity union. The rejected
receipt SHA-256 is
`efb6a8c8db714aa8a07a1edff6db06489361255e04f2837d35c2b24f0a09f2b2`.
The exercised builder again exits 2 without authorizing the final pool.
These are source-validity failures, not neural-quality measurements.

After two related failures, targeted rubric-condition audits distinguish
actual events from possible events, denomination from funding, precise
boundary/role definitions from relative descriptions, and unprompted
collection from automatic execution. They identify three additional texts
with the same missing-condition problem. The prospective
`ephemeral_pages_rubric_entailment_protocol.json` pins fourteen text-only
clarifications and preserves the other 406 texts, all targets and every
model/split/statistical rule. Every rejected source, preparation and raw
judgment remains retained; two complete fresh opaque reviews remain required.

MEASURED third source review: independent reviewers accept 417/420 and
418/420 items, leaving three ambiguous pages. The actual rejected receipt
SHA-256 is
`03de2831e3e599ec2599c8d9a7b527f87a4f4d53311a72985e31e4215884f432`.
The builder again exits 2. Formal-report reminders do not establish
scheduled-review reminders; public summaries do not exclude annex access;
public main text does not establish title and summary access.

INFERENCE: implicit-prefix source authoring is insufficient. That assumption
is retired, not interpreted as a neural-interface failure. The prospective
`ephemeral_pages_explicit_predicates_protocol.json` pins three text-only
corrections stating every included and excluded rubric component. Proposed
source SHA-256:
`0b687edbff46295cdf612e016c004a61ea146add9150ab1b2f64d7349ef175c2`.
The other 417 texts, all source IDs, labels, questions, model controls and
statistical gates remain unchanged. Both complete reviews must be repeated;
all three rejected source stages and actual judgments remain retained.

MEASURED fourth source review: one reviewer accepts all 420 items; the other
accepts 419 and marks one printing page ambiguous. “Smaller lettering” does
not distinguish ordinary body text from fine print. Neither judgment is
overridden. Rejected receipt SHA-256:
`eab58d301971983b5ed14540cfc96c75df3481d106d5cd5e8268d2c422e36528`.
The prospective `ephemeral_pages_text_size_protocol.json` substitutes named
text-size categories in that sentence only, preserving all other 419 texts
and the frozen experiment. Proposed source SHA-256:
`b9e45b89a2ec380e3c5831744c6a7f454077c381dfbf9bda165824554e92074c`.
Complete independent source review remains required before optimization.

MEASURED fifth source review: one reviewer accepts 420/420; the other accepts
417/420 and identifies three unstated between-opportunity reminder conditions.
Rejected receipt SHA-256:
`68c3d76d8ec9e74e476bf45dae626ace6db6e5b22e484567e6363f9613d598df`.
The builder remains closed. A finite predicate-frame audit now checks all
fifteen reminder wordings against scheduled review, formal report, informal
update and between-opportunity events. Seven sentences need explicit scope;
the private audit frame is not model-visible.
`ephemeral_pages_reminder_scope_protocol.json` preregisters those substitutions,
preserving all other 413 texts and every target/model/statistical rule.
Proposed source SHA-256:
`715f51b55fc07b61bb7a291ccaf8f8e0ebe78592d967a359638628944b9b720f`.
This is source-predicate completeness, not a neural architecture experiment.

MEASURED accepted source: both new independent reviewers accept all 420
items. The 840 actual judgments bind accepted receipt SHA-256
`92697d72fc6edd33be9d6b5f5beb8cd46a382abba9c9d3a4a9018e75df04f700`.
The exercised builder produces 320 fresh worlds and 31,360 final IRs:
final-build manifest SHA-256
`afb863875223e4f9abcbe6a5d8599d24369a1d59dacc6b72e2be9181219389e5`;
final IR SHA-256
`7e2b2c1ae4fb90be10f06ae983ce15fda8920e6077d7c8d35ed38993a78e6b25`.
Old world IDs, row IDs and normalized source-inventory text have zero overlap.
All five rejected source stages and raw judgments remain retained.

MEASURED prelaunch smoke: pages, cross, cosine and query-blind controls
have finite losses, finite nonzero gradients and an actual AdamW update on
72 records including eight corrected labels. Whole-batch versus 17-record
chunks agree within the frozen tolerance; maximum gradient difference is
`8.940696716308594e-08`. Gradient receipt SHA-256:
`4dd3eed28893565790bf93682a52b4cad4a0030d79410cd42a6e077b9331cd11`.
The real two-epoch CLI smoke then completes all five controls, including
lexical, under the atomic context. Smoke artifacts are retained separately.
Final neural capture/scoring remains sealed; corrected quality is unmeasured.

`ephemeral_pages_atomic_custody_manifest.json` pins the source, preservation
proof, accepted reviews, final build, independent prefit proof and actual
gradient/five-control smoke artifacts. Current correction manifests now
store text/rubric hashes instead of source payloads. Originally registered
bytes are retained unchanged outside Git and in historical commits; custody
pointers resolve the retained originals. Two copied secondary count phrases
are corrected to one and seven substitutions. Primary metrics, gates, model
inputs, labels and code are unchanged; no model outcome selected this update.

### Pinned Laya runtime/artifact custody, without model execution

MEASURED: the isolated current-source checkout imports Laya 0.3.25 at
`859b8ee595cc04f84dd2af476d6d1d90ec1fea46`. Three real route calls choose
English, typed-decisions and multilingual under their explicit recommended
selectors. The training environment and frozen Vey product are not changed.

All fifteen required files match upstream Git/LFS content identities from
bundle revision `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`; all three
safetensors headers parse. English/typed/multilingual model-file sizes are
842,609,210 / 842,609,220 / 643,835,514 bytes. These are payload sizes,
not RAM/VRAM measurements. Stored scalar counts include buffers and are not
live parameter counts.

The first inventory run failed on valid shared Xet cache storage. Its
implementation is retained. Corrected code verifies pinned upstream identities
instead of assuming repository-local blobs or SHA-256 cache filenames.
An intentionally wrong upstream Git-blob identity is rejected before model
execution, with no successful inventory receipt written. Its input and
rejection receipt are retained and pinned by the custody manifest.
The exercised receipt SHA-256 is
`639de7150c1fec59c7bd00972d903c75a262cf4db4b8ec7153613c2656cfd4e1`;
`research/competitors/laya_runtime_custody_manifest.json` pins every artifact
and correction. This is a custody probe, not a quality experiment:
zero model loads/forwards, no Torch import, no final benchmark access.
Quality, calibration, inference compatibility, performance and residency remain
unmeasured. No statistical test, promotion gate or neutral superiority claim
applies. Next: serialized offline prediction compatibility, then the frozen
neutral workflows and metric family before any quality comparison.

### Current Laya offline CPU typed compatibility

Hypothesis: all three pinned current routes load on the existing CPU runtime
and return valid mixed Choice/Noul/Score payloads. Preregistration `421f44d`;
source-derived serialization correction `4e4d724`. One synthetic state and
identical questions across English, typed-decisions and multilingual; no
quality/generalization sample, fitting or final-pool access.

MEASURED result: all three complete offline CPU eager FP32 prediction without
AMP. The compatibility gate passes. Actual loaded parameter counts are
421,293,827 / 421,293,827 / 321,908,995, distinct from stored scalar counts.
All three tokenizer config hashes remain unchanged in isolated smoke copies.
Only one route is resident at a time; no GPU execution or training-environment
upgrade occurs.

The original smoke failed after English prediction because its unsupported
`1e-5` sum tolerance ignored four-decimal served probability rounding.
Original source and failure receipt are retained; the failed prediction itself
was not persisted. Corrected capture precedes assertions, and the normalization
bound `K*0.00005 + 1e-6` comes from pinned decoder source, not fitted outcomes.
Checkpoint, input, questions, backend and calibration remain unchanged.

Successful receipt SHA-256:
`b339c32fd4f7b0caf4711aa5c6a202011b3e26dfde7a63fcd159a471b62416ac`.
`research/competitors/laya_cpu_compatibility_result_manifest.json` pins complete
predictions and both lineages. No statistical quality test applies.
Unknown: GPU compatibility, quality, calibration, high-K, multilingual quality,
long-context reliability and same-host performance/residency. No neutral
superiority or capability-promotion gate is earned.
Next: freeze neutral workflows, metric family and pinned-tokenizer length
census before quality selection.

### QASPER substantive input-length census

Hypothesis: the verified natural papers may supply an 8,192/16,384-token
stratum under the pinned current Laya tokenizers. Preregistration `cf3099c`;
tokenizer bundle `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`,
native tokenizers 0.23.2. English and typed-decisions share tokenizer bytes.

Controls: project only title, abstract and full-text sections from the existing
read-only model-state column; reuse the existing deterministic document
serializer. Disable padding, truncation and special-token insertion. A SQLite
authorizer permits only paper input/identity and existing split/component
columns. No question, answer, evidence target, worker ID, Torch import or
model forward occurs. Counts exclude question/option overhead and figure
captions outside this declared projection.

MEASURED: all 1,585 papers/components retain their original source database and
split identities. The census gate passes. Source-native wording/order is
unchanged; there is no padding, repetition or unrelated concatenation.

| Tokenizer | All papers >=8,192 | Confirmation >=8,192 | All >=16,384 | Confirmation >=16,384 | Median all |
|---|---:|---:|---:|---:|---:|
| English | 143 | 32 | 17 | 2 | 5,009 |
| Multilingual | 138 | 29 | 17 | 2 | 4,931 |

Report SHA-256:
`8d384df6c15bdeebf1b345160537289e300c1cd39ea1a98d44e1474746266db3`.
Private per-paper IDs, split/component IDs, text/state hashes and exact counts
are pinned by `research/endgame/neutral_paper_length_result_manifest.json`.
The full database hash is verified before and after projection.

INFERENCE: under hypothetical zero harmful paired discordances, even an
unadjusted two-sided 95% exact upper bound is .108881 for 32 papers or .119445
for 29, above the frozen .01 accuracy margin. These are precision scenarios,
not observed errors or predictions. Native label/evidence eligibility can
only reduce these input counts. Neither long-context capability nor a neutral
non-inferiority gate is earned. Quality, evidence-position reliability and
author/template independence remain unknown.

Next branch: retain QASPER for descriptive natural evidence and audit another
lawful natural long-context source before acquisition. Do not pad the papers,
lower 8,192, substitute synthetic evidence or relax statistical gates.

### Natural long-source rights, exposure audit and input-only preregistration

HYPOTHESIS: native Natural Questions full documents may provide more
8,192/16,384-token confirmation inputs than QASPER. Input counts, label/evidence
eligibility and effective independent units remain unknown.
Preregistration: `research/endgame/natural_questions_length_protocol.json`;
implementation: `natural_questions_length_census.py`.

MEASURED: the rendered [official Google download page](https://ai.google.com/research/NaturalQuestions/download)
expressly releases Natural Questions under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).
Pinned HF revision `e8103d566bef4154c2c12b17c6095ec5275840cc` lists seven
validation shards totaling exactly 1,337,126,358 compressed bytes and reports
7,830 validation rows. Row counts and source payload integrity are not yet
independently reconstructed. Review authorizes bounded anonymous local
evaluation preparation, not shipping training, weights, redistribution or
blanket clearance of unrelated upstream material. Preserve original page
title/URL, publisher/Wikipedia attribution, license links and modifications.

Controls: decode only ID and document title/URL/native tokens. Exclude HTML,
questions, answer candidates and annotations. Retain native visible token
strings/order with one-space joining; no padding, clipping, generated summaries,
title duplication or unrelated concatenation. Both pinned Laya tokenizers run
without truncation, padding or special tokens. All official validation stays
confirmation-only; prior Vey/Laya/backbone exposure remains UNKNOWN.
Title/URL/text-identity components are overlap controls, not independent
annotator/user/template sampling units. No model runs or quality final occurs.

MEASURED guard smoke: real synthetic Parquet files cover both native token
layouts; document text survives projection while HTML/question/gold sentinels
are excluded. Eight illegal projection requests fail before decoding; unequal
token-list lengths fail. No released dataset record or model is used in this
smoke. Receipt SHA-256:
`cbadf1225f4ac694622bc3854e3275763b9b917fe701015c6ac2ff39357987ca`.
This earns decoder-boundary evidence, not source-census/capability credit.

Two metadata scouts exceeded their read scopes. The NQ scout opened an
illustrative documentation example; its values are excluded from gold.
The LongBench-v2 scout opened forbidden `data.json`, exposing identified row
`66fcffd9bb02136c067c94c5` and reporting a partial second row. Exact remaining
exposure is unresolved; the claim that 502 rows are unopened is rejected.
Existing tool text and correction reports are retained outside Git.
Do not call this source fresh or use its 503 publisher items as independent
units. The exact pinned `data.json` is 465,490,535 bytes; HF `usedStorage`
647,930,799 is aggregate storage. Apache-2.0 packaging and MIT code do not
establish every underlying document grant. LongBench-v2 remains on hold.

`research/endgame/long_source_metadata_review_manifest.json` pins primary
metadata, the official rendered grant, correction reports and the exposure
archive. Next: execute the frozen bounded NQ acquisition/census, preserve
failures and verify every source object. Native gold mapping, cross-source
grouping, workflow/metric freeze, DEV precision and evidence-position tests
remain mandatory before model comparison. No statistical quality test or
neutral non-inferiority/superiority gate is earned.

The first NQ attempt (`23ea2ee`) verified the first shard's 185,560,593 bytes
but failed before a complete census: native Wikipedia URLs use
`//w/index.php`, while the identity parser dispatched only `/w/index.php`.
A URL-only projection confirmed this shape without questions, annotations
or document text. Preserve the incomplete attempt and original code/protocol.
`natural_questions_length_url_protocol.json` preregisters collapsing duplicate
leading slashes for route dispatch only; original URL, text, labels, all rows,
allocation, tokenizers and gates remain unchanged. The corrected attempt uses
a new output root. Both decoder-boundary layouts and single/double-slash route
equivalence pass; receipt SHA-256:
`0750cba548e14a546b55034ed82be06de04dc077f46fcaa850ce70368c52a6a6`.
This is an input-parser correctness amendment, not outcome-driven selection.

### ECA-2 corrected-context runtime and closed literal replay custody

MEASURED: the live cached driver initially failed with `NameError: records`.
`load_phase` returns a dictionary; the corrected driver indexes its `records`
member and passes the atomic protocol into `CachedRuntime`. Original code,
failure receipt and corrected live receipt remain outside Git. No fitted
parameter, input, target, threshold or decoder changed.

The corrected CUDA invocation exercised two distinct questions, a repeated
question, one changed page and actual joint-cross encoding. Initial state
encoding used 16 pages in one forward; a repeated question added zero page
or query forwards; the changed page added one encoded page and no new query.
Packed/live score maximum errors were at most `5.90086e-6`. The first
decision was correct, but the second question and page-change decisions
were wrong. This proves cache execution/parity, not semantic quality,
certification or end-to-end performance.

Live receipt SHA-256:
`e919c7169c80065c998bc89bf89d136de1776c2cda548a661e031673aee9a739`.
Original failure:
`68fd66f0c7b5f8a109ce57822ecfe73718653eafd73b690789df6bad38076b2d`.

Before replaying the unchanged, already closed CBF8 literal/alias cohorts,
`ephemeral_pages_atomic_closed_protocol.json` pins the corrected pages
checkpoint, sealed selection, calibration and explicit atomic context.
Its SHA-256 is
`5c70f25c7c5592b5582aa8c6d9528a223a1c96a8003a3b7993a2a6334790fde3`.
The adapter now propagates that context through receipt validation,
checkpoint loading and encoding; historical ECA-1 replay remains explicit.
The exact compiler, integer facts and closed cohort recipe are unchanged.
This replay is a literal regression and closed diagnostic, never a fresh
transfer evaluation or a source of checkpoint/threshold selection.

### Natural Questions native input-length census (completed)

MEASURED: all seven revision-pinned official validation Parquet objects,
1,337,126,358 compressed bytes and 7,830 rows, passed byte custody before
and after input-only projection. Decode only ID, document title, original
URL and native document tokens; join non-HTML tokens with one ASCII space.
Question text, candidates, HTML and annotation values were not decoded.
Both pinned tokenizers ran without padding, truncation or special tokens.

| Tokenizer | Rows >=8,192 | Source components >=8,192 | Rows >=16,384 | Source components >=16,384 |
|---|---:|---:|---:|---:|
| Laya English | 2,810 | 2,288 | 1,052 | 815 |
| Laya multilingual | 3,045 | 2,483 | 1,242 | 956 |

There are 6,930 title/URL/exact-text overlap components and 7,378 distinct
substantive-text hashes. English median/max content tokens are 5,740/85,897;
multilingual median/max are 6,219.5/93,286. These count document content
only, not future question/candidate/workflow framing.

An independent seven-shard projection, token-count and graph/DFS replay
passed 133,300,885 assertions, including complete row/group/threshold
coverage and unchanged raw bytes. The producer groups with union-find;
the verifier reconstructs graph connected components separately.
Producer report SHA-256:
`c5fa50cf576ce7c42699ed230be47a3e2daa1356cf4df274663006f31d922607`.
Independent receipt:
`094e2c117a4e35fc38e33d110bcd732c17555c6ec1a52aad00a9023da1325337`.
`research/endgame/natural_questions_length_result_manifest.json` records
source, tokenizer, code, parser-failure and custody lineage.

Gate PASS for the frozen input-only census and no-gold decoder boundary.
No model forwards, semantic outcomes, power calculation or quality test
occurred. Source components are overlap controls, not proof of independent
authors, users or templates. All official validation remains
confirmation-only; prior training/selection exposure remains UNKNOWN.
The conditional/review license scope permits this bounded local evaluation
preparation, not shipping training or redistribution.

INFERENCE: this source has substantial natural 8K/16K content under the
pinned tokenizers. Native-task eligibility, gold mapping, cross-source
overlap, evidence positions, DEV precision and actual long-context model
behavior remain unmeasured. No neutral non-inferiority or capability
credit is earned. Next: freeze native task/workflow/metric mappings and
source-group separation before any model comparison.

### ECA-2 child-local corrected experiment: final quality failure

HYPOTHESIS: correcting parent-wide UNKNOWN leakage into independently
supported children may recover the intended atomic Evidence Pages
interface. Preregistration `f47434c`, atomic protocol SHA-256
`21e295c362da058b4b5d2247b61caf247449e79132ed7520a6f637926816b0fb`,
inherits the original five architectures, optimizer, 400 epochs, seed7,
normalization, validation selection, calibration grid and gates.

Data: cached prefinal encoder inputs/features remain byte-identical;
child targets/masks were independently rebuilt from owned active-property
evidence. All five heads were refitted. The final has 31,360 DecisionIRs
over 320 disjoint worlds, newly authored property meanings/wordings across
the fixed20 families and two opaque reviews of all420 new source items.
Calibration and the pages-only eligible-arm receipt were sealed before
final feature capture. Eligibility is not a quality pass or final winner
selection. Final outcomes were scored once for the five fitted controls
and three frozen ablations; no final fitting or threshold changes occurred.

Primary metric: supported K4 atomic exact-maximal-set family macro.
MEASURED final results:

| Control | Atomic macro [world-cluster 95% CI] | Held4 macro | Composition | Ordinal MAE | Attribution | Supported coverage | Teacher-changing correct-new |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pages | .085547 [.075212,.096362] | .132813 | .045313 | .361047 | .247754 | .280000 | .080327 |
| Joint cross | .281641 [.265691,.297274] | .357422 | .178125 | .360000 | .883203 | .520938 | .248935 |
| Cosine | .030078 [.021771,.039272] | .054688 | .012500 | .352883 | .292188 | .083438 | .027202 |
| Lexical | .269531 [.249565,.289259] | .322266 | .364063 | .318965 | .383984 | 1.000000 | .283026 |
| Query blind | .425000 [.405991,.443405] | .468750 | .425000 | .340777 | .250000 | 1.000000 | .346449 |

Statistics: 10,000 paired world-cluster bootstrap draws, seed0,
conditioned on the fixed authored family/property/wording inventory.
These are not independent human authors, future fields or native task
samples. Pages minus cross is `-.196094`, CI `[-.213915,-.178217]`;
the one-sided95 lower bound is `-.210855`, failing the inherited internal
.05 non-inferiority margin. This is not the endgame .01 comparison.
There are 280 harmed worlds among292 discordant/320; CP upper95 is
.904282.

Gate FAIL: every fitted control fails overall semantic quality.
Pages fails atomic/held-family/composition >=.80, ordinal MAE <=.10,
orientation >=.90, attribution >=.85, UNKNOWN precision/recall >=.90,
supported coverage >=.90 and causal correct-new >=.80. Its UNKNOWN
precision/recall are .729863/.778125; orientation is .541309.
Cross attribution alone passes its point screen; its ordinal and decision
quality do not. Lexical/query-blind UNKNOWN precision is undefined
(zero predicted UNKNOWN), not zero-error evidence.
Permutation/rename/page/question reorder invariance is1 structurally.
No promotion, B-STEF, certificate, speed or competitor credit is earned.

Mechanism evidence: pages exceeds zero-question and zero-page ablations,
each Bonferroni lower bound .074356 across the three declared benefit
claims. It does not exceed lexical (lower bound -.208029), and causal
quality fails. Nonzero input use does not establish adequate semantic
decisions. Cross matching .883203 with ordinal MAE .360000 makes matching
versus grade/orientation a diagnostic hypothesis, not a proven cause.

Independent NumPy/typed replay passed 75,391,388 assertions and compared
293,516,622 values, including all31,360 decisions per control, causal
populations, exact child truth, 439,040 typed integer cases, 206,080
identity cases and four large-integer/missing-child boundaries. Numerical
reproduction of the superseded ECA-1 does not repair its confounded targets.
Do not interpret its old final versus this new final as an isolated
target-fix effect: both the targets and final property/wording inventory
changed.

Final evaluation SHA-256:
`5af3ef025e98243885c68ed145b2b190a1e0fcbeb48b53c9112485d4fd6a2e35`.
Independent final replay:
`c056e26eb68c33e5e2cc94f55114a0c499df7813bf694ef00fad6b0f938cb618`.
Sealed selection:
`3555e4cbd83b1aa4c9b27beeb602b1633c3c804c00dacf08bc8e91529ac57b7c`.
The immutable final report leaves its separately owned live-runtime and
closed-literal prerequisites unfilled; their subsequent receipts must be
reported separately, not retroactively inserted into that report.

INFERENCE: these tested corrected frozen final-mean/readout interfaces
remain inadequate. This does not prove semantic information is absent
from every encoder state or retire all evidence-caching architectures.
Unknown: whether conditional readout, relation/grade recovery or an
explicit caller-supplied rubric resolves the remaining interface.

Next branch: corrected DEV-only truth/component substitutions, one finite
FP64 grade least-squares ceiling and the previously declared conditional
768->128->1 heads. Corrected DEV pages/cross atomic macro are
.058594/.313477, sufficient to activate the original failure branch
without choosing an architecture from final outcomes.
`ephemeral_pages_atomic_audit_protocol.json` prospectively pins this
child-local execution; SHA-256
`1d41ea1c2afb81b82472500859119fa5b48bb37065076a6b15afa96e72e2f1fa`.
No new final features, labels or outcomes enter these diagnostics or
their fitting/selection. Old DEV success can justify a fresh experiment,
but cannot earn promotion.

#### Separately completed exact regression and retained execution failures

MEASURED: corrected-context closed CBF8 replay preserves all4,960 literal
integer decisions exactly in each of its two historical cohorts.
Independent replay passed 2,004,676 assertions and compared443,070 values;
rank64 NumPy/reader maximum absolute error was `7.62939e-6`.
It reconstructs 7,584 development decisions/73,416 teacher-changing swaps
and 13,152 closed-final decisions/150,188 swaps. Literal maximal-set and
concrete accuracy are1 with zero original-teacher differences.

Closed semantic results remain negative: development alias/composition
exact-set accuracy .056641/.008750, causal correct-new .007274;
historical closed-final alias/composition .228760/.142822, causal
correct-new .125782. Correct-new conditions on teacher change without
additionally requiring student change. These cohorts were already closed
and are diagnostic only; they are not fresh ECA-2 transfer evidence.
The real encoder processed four field pages in one forward and164 distinct
query spans in six forwards; candidate integers bypassed the encoder.

Closed result SHA-256:
`a85f4e575880d636f4a3cbb0e9a321bfcf2018608b36c0e82e9c8de82d231ae2`.
Independent replay:
`4eb334350e43a2f11107af36ff08765c462b38d209c7630ff9df6f4495b40fae`.
`ephemeral_pages_atomic_result_manifest.json` pins the complete final,
supplemental live/closed proofs, immutable original gate fields and failures.
The exact and live-runtime prerequisites now pass separately; overall
semantic quality remains FAIL.

An outer launch `flock` initially blocked the encoder's own exclusive
lock, timing out with an empty output directory. Removing only the
redundant launcher lock completed the unchanged replay; the encoder still
serializes CUDA use. Retain the failed root and timeout receipt.
The independent checker also initially assumed that the validator returned
the entire raw receipt and that the atomic amendment directly contained
its inherited encoder specification. Both incorrect checker assumptions
were repaired against the existing contracts, preserving source snapshots
and failed receipts; no model result, gate, threshold or corpus changed.

Natural Questions custody/exposure closure: the corrected rights and
exposure audit (`long_source_metadata_review_manifest.json`, code revision
`fb26a307b1fe636c97302890a27b491d6530130`, correction receipt
`3b3455d146a1b065b5f710619cd4d2c1c7104cb2a9788e5f1ecd8c11efdd76aa`)
is persisted, together with the input-only census above. Dataset-specific
publisher grant is CC BY-SA 3.0 with attribution to Google Research,
Kwiatkowski et al. 2019 and the underlying Wikipedia contributors.
Per-page upstream grants are not independently audited; share-alike
obligations on transformed content and trained artifacts remain open.
License class stays conditional/review and limits use to bounded anonymous
local acquisition and input-only evaluation preparation.
An earlier scout violated its metadata-only scope by opening an
illustrative documentation record; no released benchmark row was opened
in that audit, and prior exposure remains UNKNOWN rather than "none".
Native annotation payloads, including NULL/NONE/unanswerable and
multiplicity, must be preserved verbatim and independently verified
before any DecisionIR conversion. Toy labels and generic Boolean
interpretations are forbidden.

### Corrected interface diagnostics: extent, orientation and knownness are all binding

Preregistered in `ephemeral_pages_atomic_audit_protocol.json` (`5d5edb69…`)
with the documented import-scope amendment
`ephemeral_pages_atomic_audit_amendment.json` (`e10ad1fb…`). Data:
corrected child-local train/validation/calibration/development only. Zero
encoder forwards, zero reader forwards, no fitting in the factorial part.

MEASURED truth recheck on development independently reproduced all75,520
child-local records, 248,320 page ownership/grade checks, 15,616 question
orientation checks, 12,544 exact-teacher rows, 8,704 missing required
children, 1,536 contradictory children, and 2,560 supported siblings inside
UNKNOWN parents with zero unsupported siblings in known parents. Parent
knownness equals the conjunction of required child truth.

MEASURED learned-baseline reconstruction is FP32 identity: atomic macro
.058594, attribution .260498, orientation .711914, ordinal MAE .321672,
supported coverage .225781, causal correct-new .059304.

Privileged 16-cell pages substitution (atomic macro / attribution /
coverage / correct-new):

| Substituted components | Atomic macro | Attribution | Coverage | Correct-new |
|---|---:|---:|---:|---:|
| none (learned) | .058594 | .260498 | .225781 | .059304 |
| knownness | .260742 | .260498 | 1.000000 | .269709 |
| orientation | .114258 | .260498 | .225781 | .100675 |
| raw extent | .104492 | .260498 | .258594 | .092152 |
| orientation + knownness | .444336 | .260498 | 1.000000 | .414240 |
| raw extent + knownness | .413086 | .260498 | 1.000000 | .388317 |
| raw extent + orientation | .300781 | .260498 | .258594 | .242543 |
| raw extent + orientation + knownness | 1.000000 | .260498 | 1.000000 | .824574 |
| relevance | .000000 | 1.000000 | .000000 | .000000 |
| relevance + knownness | .345703 | 1.000000 | 1.000000 | .357422 |
| relevance + knownness + raw extent + orientation | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

MEASURED: attribution alone adds nothing (it is fixed at .260498 in every
non-relevance cell) and is the only substitution that drives coverage to
zero when relevance is supplied alone, because an oracle-relevant page that
the learned knownness head rejects produces no supported candidate.
Correcting orientation and knownness alone reaches .444336; adding oracle
extent reaches exact1. Each single learned component is therefore
individually limiting, and the exact compiler plus exact teacher can solve
the task when all three semantic components are correct. This identifies
three separable learned interfaces, not one broken interface.

Joint cross substitution is sharper still (learned baseline atomic macro
.313477, attribution .833984, orientation .813721, ordinal MAE .257670,
coverage .652344, correct-new .313388):

| Substituted components | Atomic macro | Attribution | Coverage | Correct-new |
|---|---:|---:|---:|---:|
| none (learned) | .313477 | .833984 | .652344 | .313388 |
| knownness | .470703 | .833984 | 1.000000 | .471413 |
| relevance | .472656 | 1.000000 | 1.000000 | .478871 |
| direct grade | .682617 | .833984 | .653125 | .647550 |
| direct grade + knownness | 1.000000 | .833984 | 1.000000 | .958452 |
| relevance + direct grade | 1.000000 | 1.000000 | .653125 | 1.000000 |

MEASURED: the joint cross reader needs only extent and knownness. Fixing
both reaches exact1.000000 with no relevance substitution, so orientation
and relevance are not binding for that architecture. Extent is the dominant
single bottleneck (.313477 to .682617), knownness is second, and their
combination is sufficient and necessary for exactness.

Together with the linear ceiling (MAE .383230 from an exactly fitted finite
linear map), this retires the joint final-mean reader plus linear repair
interface for extent: the label is not linearly recoverable from the cached
page feature at any training size. Extent identifiability needs a different
model-visible input, which is what an explicit rubric anchor would supply.

The device-mask fix is verified by byte comparison rather than by a
completed rerun. A rerun under the corrected code reached a strict 174-file
prefix of the first run's 724 files before its 1800 s deadline, matching 167
of them byte-for-byte. The 7 differing files are one substitution cell whose
`provenance.json` the rerun never wrote, so it was interrupted mid-write
rather than computed differently. The completed first run remains the
result of record, and the partial rerun root is retained and explicitly not
cited as a result. Rerun evidence:
`fec4f16b0b07a606b2d6328443e639a0f03c7e987928e33e35a8006e889a189b`.

Validation is not a substitute for development evidence here. The page-only
linear ceiling scores family-macro unique-text MAE .21770556 on validation
over 180 unique page texts, which is exactly the train unique-text count,
while development holds 240 unique page texts of which 60 are unseen in the
fit. The low validation figure measures reuse of fit-population texts, not
held-out generalization. Only the development figure (.38322990) supports
the claim that the frozen linear extent map fails on new text, and the
rubric-anchor screen is correspondingly specified on development with
validation explicitly prohibited as a pass/fail basis.

MEASURED: after the import-scope fix the conditional reader ran on CUDA
for the first time. A 2-epoch, 64-record smoke fit 295,686 parameters in
46.67 s under the shared immutable normalizer. Train objective moved
3.15357 to 4.55334 and validation objective 7.64320 to 3.67201, which on
64 records carries no signal and earns nothing. The value of the smoke is
that the mechanism executes end to end with finite objectives, a selected
checkpoint and a hash-verified parameter set. Smoke checkpoint:
`042d2b2784ef68d1e99d681212c4756779030cf958a6b654a90a505b31d0951a`.
Smoke checkpoints are permanently ineligible for calibration or
development evaluation.

MEASURED finite linear ceiling: a bias-free FP64 least-squares map from the
cached 384-d page feature to raw grade fits 180 unique supervised train page
texts (14,848 supervised occurrences, 36 per grade, 15 per property) with
rank180/384 design, retained condition221.81, train MSE 1.82e-29, 204
unidentified coefficient directions, normal-equation gradient inf1.24e-13
and an independent SciPy GELSS solver agreeing to 2.18e-14 relative. On
opened development it reaches family-macro unique-text MAE .383230
(occurrence MAE .361975) over 240 unique page texts, with unclipped
predictions spanning [-1.095, 1.471] against a [0,1] target. It therefore
does not recover grade from the cached page feature in this finite pinned
function class. This bounds the tested linear map on this finite corpus; it
is not a bound on all encoders or nonlinear readouts.

INFERENCE: the trained pages reader fails because relevance, orientation,
extent and knownness are all learned jointly from the same final-mean
feature. Giving the reader an explicit per-property rubric anchor is the
single targeted change the frozen decision tree authorizes next, and it
targets extent identifiability rather than capacity.

Execution defect: `ephemeral_pages_components.py` set
`CUDA_VISIBLE_DEVICES=""` at module import, so importing its truth helpers
from the conditional driver hid a live GPU from the whole process and both
smoke attempts aborted with "CUDA requested but unavailable" before any
optimizer step. The mask is now scoped to direct CLI execution; pre-fix
source snapshots and the failed roots are retained outside Git, and the
component parts rerun in a new exclusive root before any conditional fit.

### Conditional readout diagnostic: real gains, no gate passage

HYPOTHESIS: a frozen cached q/page conditional reader with independent
768->128->1 relevance, grade and direction heads, plus the unchanged
three-parameter knownness expression, distinguishes readout restrictions
from an unresolved frozen representation interface. Preregistered in
`ephemeral_pages_atomic_audit_protocol.json` (`5d5edb69…`) with
import-scope amendment (`e10ad1fb…`). Corrected child-local
train/validation/calibration/development only; the ECA-2 final was never
opened. Baseline predictions were reconstructed from saved artifacts, so
zero baseline model runs were needed.

MEASURED fit: seed7, 400 epochs, AdamW lr .01, weight decay .0001, one
full-batch step per epoch, rank64, 295,686 parameters, identical shared
immutable train normalizer. Earliest minimum held-world validation
objective selected epoch 5 at 2.840483; the epoch-400 validation total was
4.060195, so later training did not improve held-world validation.

MEASURED development, conditional against the frozen pages reader on the
same pool (10,000 paired world-cluster bootstrap draws, seed0):

| Metric | Conditional | Frozen pages | Paired delta [95% CI] |
|---|---:|---:|---:|
| Atomic choice macro | .205078 | .058594 | +.146484 [.121058,.171793] |
| Concrete winner macro | .205078 | .058594 | +.146484 [.121058,.171793] |
| Ordinal MAE | .303060 | .321672 | -.018612 [-.028632,-.008502] |
| Orientation | .830078 | .711914 | +.118164 [.101563,.134766] |
| Page attribution | .260742 | .260498 | +.000244 [-.015869,.015869] |
| Supported coverage | .479688 | .225781 | +.253906 [.214063,.293750] |
| Causal correct-new | .189098 | .059304 | +.129794 [.109197,.150923] |

Permutation/rename/reorder invariance is 1. Calibrated distribution ECE is
.058694. UNKNOWN precision .748014 and recall .617813 both fail.

MEASURED against joint cross, the conditional reader is worse on every
quality metric, consistent with the substitution finding that the joint
reader already recovers relevance and needs only extent and knownness.

Gate FAIL against every parent screen: atomic .205078 < .80, ordinal MAE
.303060 > .10, orientation .830078 < .90, attribution .260742 < .85,
UNKNOWN precision/recall .748014/.617813 < .90, coverage .479688 < .90,
composition .097656 < .80, correct-new .189098 < .80. The preregistered
branch `conditional_control_passes_parent_development_screens` is not met,
so this control earns no capability credit and no promotion despite its
consistent improvement over the frozen pages reader.

MEASURED conclusion: the conditional readout materially improves every
metric it touches and still lands far below the gates. Readout capacity
and joint pooling were not the binding constraint. Combined with the
substitution factorial (extent and knownness binding, relevance not) and
the failed linear extent ceiling (development MAE .383230), the tested
frozen final-mean representation with any readout of it is retired for
this assay. Extent identifiability needs model-visible information the
current inputs do not contain.

INFERENCE: the rubric-anchor branch is the justified next experiment, and
its own CPU-only augmented linear ceiling screen can retire it before GPU
spend. Unknown: whether the frozen encoder's rubric-text mean carries
property-scale information at all, and whether prefinal properties satisfy
the four anchor audit conditions.

Result manifest `ephemeral_pages_conditional_result_manifest.json`
(`6741cdcbf7bf6e7dcab243e55a3d00c8ffe82d28e1e4b3b473e0dfd11bdda295`).
Development evaluation:
`ba7901b5c1156527defa74a1b8809fef8c6bb02db189d6d08251a6f54d220eb3`.

### Rubric-anchor stage S0: mechanical screen failure, no gate edit

HYPOTHESIS: making the caller's ordered per-property rubric explicit as a
model-visible anchor restores extent identifiability. Preregistered in
`ephemeral_pages_grade_interface_preregistration.json` (`668759487d…`),
with the design rationale corrected to cite only the development ceiling
(the .21770556 validation figure measures reuse of fit-population texts
and is explicitly prohibited as a pass/fail basis).

MEASURED S0, CPU only, zero encoder forwards, zero optimizer steps, final
pool never opened. The authored prefinal source (60 properties, 300 ordered
rubric stage strings, 1,800 questions, 4,500 page wordings) yields 60
anchors with 60 distinct SHA-256 digests, each exactly 5 stages.

| Screen | Result |
|---|---|
| one anchor per property, count equality | 60/60 pass |
| anchor nonempty | 60/60 pass |
| no Unicode digit | 60/60 pass |
| no grade_order label occurrence | 60/60 pass |
| no stage heading or ordinal marker | 20/60 FAIL |
| no equality or containment vs same-property question/page text | 60/60 pass |
| token Jaccard below .60 | 60/60 pass, max .538462 |

Gate FAIL. The 20 failures are triggered by ordinary English vocabulary:
"most" (10 properties), "step" (6), "steps" (5), "stage" (1), "stages"
(1). Failing and passing properties have identical stage counts (5) and
near-identical mean anchor length (306.1 vs 309.4 characters), so this is
not a length artifact.

INFERENCE: preregistration condition (b) forbids numeric or literal grade
tokens and stage heading strings. The implementation instead applied a
36-token closed vocabulary that also contains comparatives and process
nouns. That vocabulary was an unpinned implementation choice rather than a
preregistered list, and narrowing it after seeing these 60 properties
would be an outcome-driven gate change, which is forbidden. The original
screen and its negative result are retained verbatim.

The preregistration also requires a blocking two-reviewer opaque audit
that has not run. That audit, not a vocabulary edit, is the correct
adjudicator of whether these tokens constitute stage-ordinal leakage. If
both reviewers accept all 60 anchors, record a documented amendment while
keeping this negative screen. If any reviewer rejects, the design is
unrunnable and rubric grounding retires in favor of minimal final-layer
encoder adaptation. Using the 40 passing anchors as a subset fit is
forbidden in both outcomes.

Mechanical receipt:
`7c2a0ffd8e0514c07bea38fec1e4aaf9556c4986338856ca9bae00bdae9aad7a`.
S0 failure receipt:
`025e01ff8c54447535b71ace5c98f44349933e667823e6544b13507134f3e8d2`.
Neither the reviewer stage nor the S1 augmented linear ceiling is
authorized until the screen outcome is resolved.

#### S0 reviewer gate: FAIL on authored scale defects

MEASURED, two genuinely independent opaque reviewers. Both hashed the packet
before reading it (SHA-256 `3966685cf04b…`), both received only `item_id` and
`anchor_text` across 60 items, and neither ever saw the other's file, the
authored source, the withheld key, the S0 mechanical receipt, any target,
family, split label, model output, weight or metric. The mapping from
`item_id` back to property, family, field code and grade order was written
to a separate key file that reviewers were forbidden to open, and neither
did.

| Reviewer | accept | ambiguous | reject | acceptance |
|---|---:|---:|---:|---|
| Reviewer One | 48 | 12 | 0 | false |
| Reviewer Two | 42 | 18 | 0 | false |

16 items drew different verdicts; 7 anchors are jointly ambiguous, spanning
the durability, plausibility, priority, similarity, suitability and urgency
families. Gate FAIL: the preregistration requires both reviewers to accept
all 60 with zero ambiguous and zero rejected.

MEASURED defect content, from the two disclosed rationales only. Two
reviewers independently read one durability ladder as inverted, and both
read the urgency scale as leaving stages 1 and 2 indistinguishable. The
remaining five are unseparated adjacent boundaries, for example coherence
versus consistency in plausibility and "workflow adjustment" versus "narrow
accommodation" in suitability, with no stated threshold between them.

INFERENCE: these are defects in the authored prefinal rubric scales
themselves, not artifacts of the audit packet and not consequences of the
unpinned marker vocabulary. Both the mechanical screen and the reviewer gate
fail for one root cause: several authored properties are not five cleanly
separable ordered qualitative stages.

Reviewer independence was contested and resolved correctly. The agent that
built and ran the mechanical screens was disqualified from the reviewer
role because it knows which 20 anchors failed screen 5; it declined on
independence grounds, and its offer to serve as a third explicitly
non-independent reader was declined. A reviewer who has seen the mechanical
result cannot independently validate it, and a contaminated accept would
make the gate look satisfied while destroying its purpose.

Gate consequence, per the preregistered decision rule: S0 fails, so the
rubric-anchor design is unrunnable under this protocol. No anchor encoding,
S1 augmented linear ceiling or S2 neural arm is authorized. The packet, both
reviews, the merged receipt and the S0 negative screen are retained verbatim.

Forbidden now: accepting anchors by narrowing the marker vocabulary,
editing rubric text to obtain acceptance, or fitting on the 40
mechanically passing or 48/42 reviewer-accepted subsets. All 60 or none.

Packet `3966685cf04b47a9d4ef31df4808ab353a83da5b2f58fd735aa5c25bbff68cd5`;
withheld key `6e1816d274c9b7bd5511da743514c6473a26a412b81866bdcde6e5e4fef67359`;
merged receipt `16df9d4b4a1f8bc0fe035118e02a1d2c45c3d3b07cb5a4393d49b6e63198cbf5`.

Next branch, selected by the frozen tree: failure branch 1, minimal
final-layer encoder task adaptation, under a separate preregistration.
Rubric grounding is retired for this assay. That branch must not repeat
global numeric reconstruction or widen heads, and it inherits the same
closed ECA-2 final.

### Preregistration: minimal final-layer encoder task adaptation

HYPOTHESIS: extent identifiability fails because the frozen final-layer
masked-mean feature cannot represent it, not because the readout is too
narrow. The authorized branch unfreezes exactly one transformer layer
plus a per-element diagonal adapter on that layer, keeping layers 1
through 11 and the embedding bit-identical and keeping the frozen rank-64
reader equations byte-identical.

Trainable surface, from safetensors header metadata only: `deberta.encoder.layer.11`
is 1,774,464 FP32 scalars; layers 1 through 11 total 19,519,104. A
per-element diagonal adapter on that layer's six rank>=2 weight matrices
adds 1,769,472 scalars initialized to 1.0. Encoder-side trainable is
3,543,936, which is 5.0139% of the loaded encoder against a declared 5.5%
bound; combined with the 74,116-parameter reader it is 5.1134% against a
6.0% bound. Both fractions are recorded at launch, not asserted.

Learning rates are frozen now: encoder side 0.0001, reader 0.01, ratio
0.01, with ratio search forbidden. AdamW, weight decay 0.0001, seed7, 400
epochs, one full-batch step per epoch, chunk 32, train-only normalizer,
earliest minimum held-world validation objective, calibration on the
corrected calibration worlds only, evaluation on development only.

Three mandatory fail-closed screens run in order. Screen A is a smoke fit
of at most 64 records and 2 epochs requiring finite loss, finite gradients
on every trainable parameter and a non-NaN gradient norm, with smoke
checkpoints permanently ineligible. Screen B hashes every encoder
parameter outside layer 11 before and after training and fails closed on
any change. Screen C is CPU-only and must run before any full GPU fit: the
same finite bias-free FP64 least-squares procedure on adapted final-layer
features over unique supervised train page texts, with an identity
precondition that adapter-at-init reproduces frozen page features within
1e-5, and the same development unique-text family-macro MAE threshold
0.3332298906765268. Validation is prohibited as a pass or fail basis.

Gates are the unchanged parent set plus four predeclared mechanism claims
(adapted minus frozen baseline, adapter_blind, zero_pages and lexical) at
one-sided Bonferroni familywise 0.05. A development pass earns no
promotion; replication on seeds 7, 11 and 13 plus a separate final protocol
is required first.

Failure branches: if screen C fails before fitting, encoder adaptation is
retired for this assay and both remaining authorized branches are
exhausted, which is a hard-block candidate for this assay only and not for
Vey overall. If development fails, the negative is recorded and capacity is
not widened. All 48 pinned SHA-256 values in the preregistration were
verified against disk before commit.

`research/endgame/ephemeral_pages_adaptation_preregistration.json`,
SHA-256 `5bc31a5034ee737cf7af4c74df5adc9ff3b1a7957ce005ad0e810d1c87fcbecd`.

Implementation review, before any screen was executed. The screens module
independently reproduced the retired page-only ceiling arithmetic exactly
(rank 180, retained condition 221.8066, development family-macro MAE
0.3832298906765268 over 240 unique texts) and reconciled the safetensors
ledger from disk: 203 tensors, 70,831,107 FP32 scalars, layer 11
16 tensors / 1,774,464 scalars, layers 1 through 11 176 tensors /
19,519,104 scalars. Loaded encoder parameters are 70,682,112 with
parameter hash `6176abe0b5f5be…`, matching the pin.

MEASURED specification defect, found by the implementation rather than by an
outcome: screen C is specified to encode features with the adapted final
layer at adapter init identity 1.0 and to require an identity precondition
that this reproduces the frozen page features within 1e-5. At identity
1.0 that precondition holds by construction, so screen C necessarily
reproduces the retired page-only ceiling of 0.3832298906765268 and returns
`fail` against the 0.3332298906765268 threshold. No amount of fitting can
change it, because screen C runs before any optimizer step.

INFERENCE: screen C as written is a gate that cannot pass, so it would
retire encoder adaptation without testing it. Running it as specified
would manufacture a negative result from a specification error rather than
from evidence. The defect is in the screen, not in the mechanism, and no
evaluation outcome was observed before the defect was found.

Two candidate corrections exist and neither is outcome-driven: measure the
ceiling after a bounded, preregistered adaptation of layer 11 (which makes
the screen a genuine test but spends GPU before the screens report), or
drop screen C and rely on screens A and B plus the frozen development gate
with its own mechanism claims. The choice changes what the screen can
conclude, so it must be decided prospectively and recorded as a documented
amendment before execution, retaining this negative specification finding
verbatim.

Corrected prospectively by documented amendment, before any screen ran. The
degenerate screen is retired and retained as a negative specification
finding (`7e494efc3a09e…`). Screen C splits into two. C0 is the identity
precondition only: adapter-at-init must reproduce frozen page features
within 1e-5, and a deliberately perturbed adapter must demonstrably move the
features so an adapter-path failure is distinguishable from FP32 drift. C0
carries no quality claim and no threshold. C1 is the adapted ceiling: one
bounded FP64 closed-form bias-free adaptation of layer 11 on the unique
supervised train page texts only, under the same train-only normalizer
policy and the same `lstsq` procedure, with no selection, no early stopping
and no tuning, followed by the same development unique-text family-macro MAE
against 0.3332298906765268. C1 runs after A and B and before the full
development fit; its failure retires encoder adaptation and its pass permits
the fit. Validation stays prohibited as a pass or fail basis.

This costs one bounded closed-form solve on unique train page texts, which
is the minimum spend that makes a ceiling screen a genuine test instead of
an identity check. It opens no final pool, selects no model and uses no
development or final outcome to choose the adaptation. The trainable
surface, learning rates, optimizer, epochs, seed, chunks, normalizer, parent
gate set, four mechanism claims, promotion rule and every pinned artifact are
unchanged.

Amendment `81a76d9eeed58745a0215106dfb7f6e1cf245607d10f4b9644d9a4b165304e2c`;
corrected preregistration `2aa64bc77c236192681368a8c83751aeac185414d0cd0660578b2ada9cd7731c`.
Screen order is now A, B, C0, C1, full development fit.

### Final-layer adaptation screens A and B pass, original C0 identity check fails

MEASURED screen A, CUDA, 64 records and 2 epochs: verdict pass. Two
optimizer steps, 4 encoder forward batches over 86 texts, 4 backward calls.
All 28 trainable tensors are as preregistered: 16 layer-11 tensors
(1,774,464 scalars), 6 adapters (1,769,472), and the 6-parameter-plus-
weights reader (74,116). Encoder-side fraction 0.0501390790 against the
0.055 bound; combined 0.0511340429 against 0.060. Both within bounds.
Encoder and adapter learning rates 0.0001, reader 0.01, ratio 0.01.
Final pool untouched. Receipt
`0bd928a526a8186acaac775e2b6911ea09618666db0fbc70771a9602adc2991d`.

MEASURED screen B, encoder immutability: verdict pass, zero encoder forwards
and zero optimizer steps. Loaded encoder parameters 70,682,112 matching
the pinned hash. Receipt
`7816b5e75e386a428e540aa7e0adc8a57abefbc8d898aea469715dc0a143b4aa`.

MEASURED screen C0, identity precondition: verdict FAIL against the 1e-5
tolerance. Adapter-at-init reproduces the comparison basis to 1.29342e-05
on 240 development texts and 1.28746e-05 on 180 train texts. The adapter
liveness control passes at 0.0728226 movement and restore returns to
exactly 0.0. Validation was never opened, no MAE or quality claim was
made, and zero optimizer steps ran. Receipt
`7ef3bbccd8d232b8785bd2c517e45f20e91b02ed192dadcc30c146ec076f6dd6`.

Correction to the interpretation recorded in commit `83183ff`: the cause of
the cache-identity discrepancy is unresolved. The asserted comparison-basis
mismatch is retracted. The C0 implementation maps unique features through
`train_slots`/`development_slots` before comparing matching
`record_index`/`page_index` entries. Different storage shapes do not establish
a comparison defect. The claim that no correct implementation could pass is
also withdrawn.

The earlier adapted-versus-canonical probe reused the same model after
adapter attachment, so both paths contained the patched linear forwards.
It did not independently isolate adapter arithmetic. Synthetic CPU repeated
forwards with zero discrepancy do not rule out device or batch-shape drift
on the actual cached inputs. Manifest settings are custody evidence, not
proof of identical token arrays or numerical computation.

The user explicitly selected a narrower C0 contract: adapter liveness and
restoration only. Amendment
`de2d28b3e0128323e41b22cae4ac516f7cfb764eaf70a3425b200240105ab9ba`
records that this removes the cache-identity test rather than repairs a
demonstrated comparison defect. B establishes parameter byte immutability;
it does not establish identical forward outputs. No numerical cache-identity
claim follows from the narrowed screen. Liveness and restoration retain the
original 1e-5 tolerance; the C1 quality threshold remains 0.3332298906765268.
The original failure receipt remains untouched, and the narrowed screen
writes a separately named receipt.

Retained, superseded diagnosis
`7b26188516f87a1d23b98e272828ee482429cbc816aa5393d037bc31abe10be8`;
its causal conclusion is withdrawn above.
Screen B was first blocked by a namespace defect of its own: it compared a
base-namespace digest against a wrapper-namespace digest, which differ by
construction. Hashing both sides in one namespace makes all three digests
equal with 198 shared names and 0 differing, and the guard is unchanged in
strength. Retained receipts and every launch failure are preserved outside
Git; no failed artifact was deleted or cited as a result.

### Narrowed C0 passes; C1 stops before its adaptation solve

Preregistration: user-authorized C0 amendment committed in `6e25c9d`.
MEASURED CPU liveness and restoration over 180 supervised train texts:
perturbation maximum absolute difference 0.07282257080078125; restoration
0.0; tolerance 1e-5; gate PASS. Eighteen encoder batches encoded 540 texts
across baseline, perturbation and restoration. Zero optimizer steps and
no quality claim. Validation and final pools were not opened.
Receipt `2f0c76347d2e5d96d7125c5bba813c7aa5962b17fa96be481689234f1f37df71`.
The original failed cache-identity receipt remains unchanged.

MEASURED C1 execution stops before fitting or quality scoring:
the pooling helper compares six captured activation batches with 180 mask
rows. A separate synthetic smoke exposes scalar-target `[3]` versus
residual `[3,384]` broadcasting failure. These failures do not evaluate
adaptation quality and do not activate the scientific retirement branch.

Corrective smoke: pooling now aligns each captured batch with its mask
row slice, including a short final batch; the residual hook reads the
second input to the output block rather than the dense layer's activation.
A live pinned-encoder three-text smoke yields activation `[3,1536]`,
residual `[3,384]` and pooled output `[3,384]`; row mismatch fails closed.
The bounded adaptation objective remains under mathematical audit:
linearity before LayerNorm does not establish a closed-form fit to the
final normalized pooled representation. No scalar-to-vector target may be
invented and presented as the old preregistered objective.
C1's threshold remains 0.3332298906765268. No full fit or promotion is
authorized by an implementation crash.

### C1 replacement: bounded train-only scalar extent adaptation

HYPOTHESIS: adapting the existing final transformer layer and its six
multiplicative adapters with scalar extent supervision may improve
development extent transfer through the actual pooled representation.
Preregistration:
`research/endgame/ephemeral_pages_adaptation_c1_gradient_preregistration.json`.
This is a new prerequisite experiment, not an implementation repair of
the retired closed-form proposal.

INFERENCE from the audited equations: scalar extent `[N]` cannot specify
a hidden output target `[N,384]`. The proposed output-block path also
resolves to a module without the expected dense weight. Correcting those
paths and mask slices does not define the missing target, and linearity
before tokenwise LayerNorm does not imply linearity after it. The retained
execution failure supplies no adaptation-quality observation.

The replacement uses the existing PageReader value equation
`sigmoid(wv(normalized_pooled_page))` and its raw grade MSE component.
Exactly the existing 180 unique supervised train page texts fit the
final layer, adapters and existing value head. Other encoder parameters
and reader heads remain frozen. Seed7, 400 deterministic accumulated
full-population updates, encoder/adapters learning rate0.0001,
value-head learning rate0.01 and AdamW decay0.0001 are fixed in advance.
Only the epoch400 checkpoint is eligible. No development, validation or
final data selects or trains this prerequisite.

Frozen-prefix caching must replay the actual final block, including its
attention, residuals, bias and LayerNorm, and pass an initialization
parity check against a complete encoder forward. Gradients and frozen
parameter bytes are explicit controls. A bounded train-only smoke is
permanently ineligible for the C1 gate.

The subsequent CPU FP64 bias-free readout and development population of
240 unique texts are unchanged. Primary gate: equal-family macro raw
unclipped MAE <=0.3332298906765268. This finite deterministic inventory
does not supply an IID confidence interval. Raw predictions, targets,
family IDs and features must be retained for independent reconstruction.
No quality result has yet been obtained for this replacement.

A passing C1 permits the existing full four-component development fit;
it does not authorize promotion, final access or B-STEF. A quality
failure retires this finite grade-supervised prerequisite, not every
possible encoder adaptation. An execution or invariant failure remains
an implementation failure. The autonomous endgame continues along an
evidence-justified, prospectively committed branch.

MEASURED replacement smoke at implementation `00e3ba9` stops before
cache replay or optimization. The pinned configuration omits the optional
`conv_kernel_size` attribute, while the installed encoder constructs
`encoder.conv = None` using its default zero. The prerequisite accessed
the missing configuration attribute instead of checking the actual
module. Corrective change checks `encoder.conv is None` directly;
all targets, hyperparameters and quality gates remain unchanged.
The failed smoke receipt remains under
`interface-audit-v1/adaptation-gradient-smoke-v1/c1-gradient-v1/`.
This is an execution failure, not an adaptation-quality result.

MEASURED corrected GPU smoke at `c1e71d1` passes on eight train-only
texts with two optimizer updates. The frozen prefix executes once;
actual final-block cached replay matches the complete encoder at every
token and after masked pooling with maximum absolute difference0.0.
First-backward gradients are finite and nonzero for the final layer,
each of six adapters and the existing value head. Frozen byte hashes
match before and after both updates; strict CPU checkpoint reload
passes. Receipt
`3944fae749f6eb1b75138525407f9351d8ccb836bebfb3fa93aca46e2db43544`.
Smoke is permanently C1-ineligible and supplies no quality verdict.
The unchanged 180-text, 400-update prerequisite follows this smoke.

MEASURED synthetic CPU readout checks pass at the inclusive MAE
boundary and fail at its next larger FP64 value; nonfinite development
features and mismatched family-membership lengths fail closed.
These checks exercise gate arithmetic without opening any corpus.

MEASURED full prerequisite at `c1e71d1` passes: 180 unique train
texts, 400 updates, nonsmoke checkpoint eligible for C1. Receipt
`c9e96f1f18c7ffacb93844830a3eee5106921a6ab7e6bb9016c34e10e0d1fce6`.
The CPU ceiling reaches its final receipt write, then stops because
the NumPy boolean `full_column_rank` is not directly JSON serializable.
Raw features, targets, coefficients and predictions were persisted
before this write. The partial receipt is retained; no quality verdict
is accepted from the failed serialization. A registered metadata-only
recovery must use these cached arrays without rerunning adaptation or
the encoder, and preserve the original source and artifact hashes.

MEASURED independent full-fit execution audit: its shared encoder
feature table was backpropagated through once per reader chunk without
retaining the graph. A synthetic run through the actual PageReader
and objective reproduces failure on the second chunk. Corrective
retention through the last chunk preserves the full-batch objective;
all reader and encoder-interface gradients match an unchunked
reference within rtol1e-5 and atol1e-7. No corpus or model checkpoint
is opened by this test. This is a graph-lifetime repair, not a changed
loss, training recipe or measured full development fit.

MEASURED gradient-report audit also reproduces double counting:
registered adapter parameters were enumerated again under canonical
aliases. The optimizer already deduplicated them; only reported norm
and tensor count were wrong. Removing the alias enumeration makes
the reported norm match an independently concatenated gradient vector
to absolute tolerance1e-12 in a differentiable synthetic control.

### C1 adapted candidate-extent gate passes; decision fit remains unmeasured

Preregistration `25842d4`; fixed training implementation `c1e71d1`;
metadata-only recovery `a1b3fc3`; independent verification `f57738f`.
MEASURED primary development equal-family macro raw unclipped MAE:
0.3065382709259676, below the unchanged <=0.3332298906765268 gate.
The frozen reference is0.3832298906765268. Gate PASS.
Data: 180 unique supervised train texts, 240 development texts;
seed7, fixed400 train-only updates, final epoch400 only.
The frozen prefix executes six batches once over the180 train texts.

The original CPU ceiling encoded420 texts in14 batches. Recovery
uses only retained, SHA-pinned raw features, coefficients and
predictions: zero encoder forwards, zero optimizer updates and zero
readout refits. The failed partial receipt remains byte-identical.
JSON correction converts NumPy types but preserves rejection of
nonfinite values before creating a receipt.

Independent FP64 NumPy/GELSS reconstruction matches saved train and
development predictions exactly. Rank180; 204 unidentified coefficient
directions; retained singular condition209.54714065159507;
relative normal-equation residual5.152610238543211e-17;
relative GELSS coefficient difference1.1245838679473521e-14.
This is a finite minimum-norm readout result, not a bound on all
possible readouts. No IID confidence interval applies to this fixed
authored inventory.

Artifacts and full lineage:
`research/endgame/ephemeral_pages_adaptation_c1_gradient_result.json`.
Recovered ceiling receipt
`851b173bce6ab71a08cdaa82d7b52e49b60139ffbd9b282dd967babfbb97de1a`;
solver verification
`dbac90cc14bdd5ce5a85ee8ef494ca6018b457fa73fba13231efd61c3a598b2c`.

INFERENCE: this bounded adaptation can produce candidate-extent
features that pass the preregistered development assay. Criterion
transfer, full decision quality, calibration and OOD utility remain
unmeasured for this arm. Passing C1 selects the original full
four-component development-fit branch, with its execution repairs
verified before launch. It earns no promotion, final access,
competitor claim or B-STEF authorization.

Prospective full-fit execution correction:
`research/endgame/ephemeral_pages_adaptation_fullfit_execution_correction.json`.
The independent driver audit additionally identifies active encoder
dropout, nonexistent CLI fraction keys, copied rather than observed
tokenizer/encoder pins, and phase counters overwritten by ablations.
These repairs precede any full development fit and change no loss,
capacity, data, learning rate, epoch count, selection rule or gate.

MEASURED actual pinned-model synthetic smoke: repeated train-mode
forwards differ by3.235638380050659; eval-mode forwards match exactly.
The corrected full trainer keeps the encoder in eval mode with
autograd enabled, matching Screen A's deterministic feature contract.
The actual pinned encoder and pinned PageReader complete two objective
backward chunks with finite trainable gradients. Live tokenizer and
encoder digests match their pins; all corrected summary bound keys
exist. Two repeated prediction calls accumulate four encoder forwards
and16 encoded texts, with identical predictions. Zero optimizer steps
and no corpus or final pool are opened by this smoke.

### Full-fit restart uses summed feature adjoints, not repeated encoder backpropagation

MEASURED the `1badd4b` full-fit execution is interrupted before a
fitted checkpoint or evaluation artifact is present. Its retained-graph
repair avoids the second-backward exception but traverses the entire
shared encoder graph for every reader chunk. Synthetic parameter
hooks observe two traversals for two reader chunks. The actual launch
selects23040 train records, or720 chunks per epoch.
INFERENCE: applying that implementation to the fixed inventory repeats
encoder backward work720 times per epoch unnecessarily.

The outcome-independent correction accumulates reader adjoints into
detached feature-table leaves, then backpropagates their sum through
the original encoder graph once. No feature copy, changed component
mean, changed optimizer step or altered model equation is introduced.
Three regression cases cover repeated pages/shared questions,
component masks and partial chunks; every encoder-interface and reader
gradient matches an independent unchunked full-loss reference within
rtol1e-5/atol1e-7. Actual pinned-model two-chunk backward passes with
one encoder backward and finite trainable gradients.

The original launch bytes remain at
`adaptation-v1/fullfit_interrupted_launch-1badd4b.json`, SHA
`2970994fe43c73f2e85688d51dfe489698df9a1c6ad814b2ba276b268073c020`.
`fullfit_execution_interrupt-1badd4b.json` records cancellation and
unknown completed update count. No quality result is read from this
interrupted fit, and it is not a negative mechanism result.
Execution-correction v2 supersedes the retained-graph implementation
before restart; the earlier registration remains in commit `1badd4b`.
The restarted trainer durably appends each completed epoch's existing
train/validation history and retains its final hash, rather than
keeping the entire progress record only in memory.

MEASURED two-epoch native execution smoke completes two optimizer
updates, four reader backward chunks and two encoder backwards.
Both epoch frozen-byte guards pass. Two durable progress rows exactly
match the returned history. Receipt
`09bc674599574deb202c31ce55faf9b6005b20cbbe5a8f1c5c2cb361965120e4`.
This synthetic fixture is explicitly ineligible for C1, development,
final evaluation or quality credit.

## Competitor metadata refresh, 2026-10-05

MEASURED metadata custody: the refreshed Laya and Jev dossiers retain
56 and 22 raw public HTTP response bodies, respectively. Every body
hash, both current dossier hashes, and both refresh-manifest hashes
were independently checked during matrix integration. No model
payload was downloaded, no terms were accepted, and no authenticated
Jev request or new competitor model evaluation was performed.

[Laya v0.3.27](https://github.com/NandhaKishorM/laya/releases/tag/v0.3.27)
is the current published runtime. Inspected source head
`8a6e1328cce2460a0e5aa348ad465bb1b5821cd2` additionally contains
post-release changes, which are not attributed to that package.
The three updated HF revisions add only a 160-byte root configuration
file. Prior tree entries and advertised LFS identities are unchanged;
that is metadata evidence, not a new payload hash or runtime result.
The recommended task-specific roster, parameter metadata, token budgets,
raw temperatures and high-K controls are unchanged. Published
abstention changes handle installed binning maps, stale low-confidence
flags and whole ties under `coverage_metric_definition2`.

Existing local Laya custody and CPU compatibility remain scoped to
source `859b8ee`, runtime `0.3.25` and bundle `55cf4c4`. They do not
verify local execution of `0.3.27`. Newly reported upstream M1 Pro
measurements remain vendor reports without a recorded HF revision.

[Jev model documentation](https://docs.typesafe.ai/models.md) still
reports `jev-1.13.0`; no documented model/schema/limit/tariff or released
SDK change was found. The [MCA](https://typesafe.ai/legal/mca)
section 2.3(b) authorization gate remains, with an additional
[Site Terms](https://typesafe.ai/legal/terms) automated-access
restriction observed at section 3(b)(vi). Direct Jev measurement
remains unperformed and authorization is unestablished. This does
not block independently developed Vey research.

`research/competitors/COMPETITOR_MATRIX.json` now references the current
dossier and raw-source manifests. All 21 critical cells remain
`UNMEASURED_NEUTRAL_COMPARISON`, with `green=false`. Simultaneous
non-inferiority margins and both historical runtime manifests are
unchanged. Metadata refresh earns no quality, performance,
calibration, promotion or Pareto-victory credit.

## ECA-2 minimal final-layer adaptation: full-fit negative

HYPOTHESIS: adapting only the pinned encoder's final layer and six diagonal
adapters, with the existing rank-64 PageReader equations, repairs the inadequate
frozen final-mean interface. The committed adaptation preregistration fixes
400 epochs, seed 7, reader/encoder learning rates .01/.0001 and earliest
minimum held-world validation-objective selection. All four equal-weight losses,
the corpus, exact compiler, target semantics and calibration grids are unchanged.
The retained correctness amendments fix execution and gradient accumulation,
not the gates or evaluation population.

MEASURED: the full CUDA fit completed 400 epochs and selected epoch 374,
validation objective 2.642213854. Its durable per-epoch journal reproduces the
returned history exactly. All 400 named frozen-parameter checks retain the same
digest; only the final-layer tensor population and six adapters change.
Training recorded 7,200 encoder forwards, 201,600 encoded examples,
288,000 reader-chunk backwards and 400 shared-table encoder backwards.
These counters are execution evidence, not an end-to-end speed result.

Calibration and development each contain 12,544 authored decisions and 75,520
term/candidate outputs. The primary fixed 16-family atomic Choice macro is
0.145508, descriptive 95% CI [0.124388, 0.167030], versus the .80 gate.
Statistics reuse the registered 10,000 whole-world bootstrap draws, seed 0,
128 development worlds; variants remain clustered. The primary denominator
1,024 is derived decision support, not 1,024 independent worlds.

| Development gate | Observed | Required | Verdict |
|---|---:|---:|---|
| Atomic Choice macro | .145508 | >= .80 | FAIL |
| UNKNOWN precision / recall | .730854 / .724688 | both >= .90 | FAIL |
| Composition | .031250 | >= .80 | FAIL |
| Criterion-swap correct-new | .145508 | >= .80 | FAIL |
| Grade-swap correct-new | .091797 | >= .80 | FAIL |
| Ordinal MAE | .337307 | <= .10 | FAIL |
| Orientation | .824219 | >= .90 | FAIL |
| Page attribution | .266602 | >= .85 | FAIL |
| Supported coverage | .332813 | >= .90 | FAIL |
| Teacher-changing correct-new | .130859 | >= .80 | FAIL |
| Permutation / rename / reorder | 1.000000 | 1.00 | PASS |

Exact literals remain an unsupplied closed-regression prerequisite; held-four-family
Choice is unavailable in this development phase. Neither earns a pass.
Binding mechanism claims use one-sided Bonferroni lower bounds, alpha .0125
per contrast, on the same fixed family inventory:

| Adapted atomic macro minus control | Paired delta | Simultaneous lower bound | Verdict |
|---|---:|---:|---|
| Frozen pages | +.086914 | +.059748 | PASS |
| Adapter scalars reset to one | +.004883 | -.003399 | FAIL |
| Zero pages | +.145508 | +.121644 | PASS |
| Lexical | -.492188 | -.552605 | FAIL |

MEASURED mechanism evidence: the joint refit improves over frozen pages and
zero pages, but fails the registered adapter-specific contrast and performs
materially worse than lexical. Resetting adapters does not refit the reader
or encoder. The other recorded interventions restore the frozen final layer,
zero the question and replace attention with uniform weights; all five are
independently reconstructed, without retraining controls.

The independent CLI reconstructs child-local IR projection, exact teacher,
raw page mixtures, knownness, ordinal masses, calibration-grid winners,
composed Choice/Bool/Score decisions, causal pairs, world metric ledgers,
bootstrap indices, gate screens and all four claims. It also reconstructs
saved frozen pages and lexical baselines. Reconstruction PASS is numerical
and provenance verification; the scientific gates still fail.
It runs zero encoder/model forwards and zero optimizer/refit steps.

The result manifest is
`research/endgame/ephemeral_pages_adaptation_fullfit_result_manifest.json`.
The checkpoint SHA-256 is
`3acb2011523d848f7b5af1602c278eb3c0cebea626421dff7918afc88f05b87e`;
the independently exercised v2 replay receipt SHA-256 is
`41abb4f5726210bd01f4d6eeedd89b37b8f8623ff294a34efdac001e5f305944`.
The manifest pins launch, history, reports, original/v2 replay source and
all prerequisite receipts. Raw artifacts remain outside Git.

INFERENCE: this tested representation-change route is inadequate. The results
do not prove semantic information absence, adapter equivalence or exhaustion
of all Vey architectures. Protocol failure branch 3 retires the route for this
authored assay; no wider head, extra layer, enlarged adapter or ratio search
is authorized. Explicit rubric grounding was already retired here.
Next: preserve both negatives, prepare source-native neutral train/dev contracts,
and preregister a separately justified architecture branch. This is not an
endgame hard block. No final pool was accessed in this adaptation protocol.
The earlier ECA-2 final was already evaluated and reported; the preregistration
cites its negative. That old pool is exposed historical evidence, not fresh
confirmation for a later architecture. The inherited final-access prohibition
remains; any future confirmation must be independently fresh and preregistered.
No promotion, B-STEF, certification, neutral comparison or competitor win is earned.

## Published Laya 0.3.27: complete-source CPU compatibility

HYPOTHESIS and preregistration: test the exact current published release and
three current-reference checkpoints on the same synthetic mixed typed request,
offline, sequential CPU eager FP32. The prospective v2 protocol retains the
original wheel failure and changes only source completeness.

MEASURED: the published wheel's registered `backend='eager'` constructor fails
with missing `laya.backends`, before any completed model route. The complete
release tree contains five backend files omitted from the wheel. All 35 shipped
package members match that release; all 40 complete-source members have verified
SHA-256 and Git blob identities. Success uses unmodified complete source at
`b09832bdd3819e375fe8b0d26dbd7a8f75c4f0bd`, not a patched wheel or post-release head.

English, typed specialist and multilingual each complete Choice/Noul/Score.
All nine outputs pass finite range, exact keys and registered serialization
sum-bound checks. This one-state compatibility scenario has no quality metric,
statistical CI, calibration or performance claim. The multilingual isolated
tokenizer copy changes from list to mapping metadata; both hashes are retained,
and the pinned standalone source bytes remain intact. Custody verification uses
the documented current `resolved_path`, not its legacy bundled display path.

The additive result manifest
`research/competitors/laya_published_0_3_27_result_manifest.json`
has SHA-256 `0808f5b0a35d76dbeae85aa9ea00852a34a2c765660dca500b1d896fcf7b8719`;
the successful raw receipt has SHA-256
`3176c04a3b4acd5ed34a2086f99b0ef275573fe6e929fce81247677822218048`.
Original wheel failure, immutable protocols/custody and historical 0.3.25 evidence
remain separate. The current dossier/matrix reference this new compatibility
evidence while retaining all 21 critical cells non-green.
Unknown: neutral quality, same-hardware performance, GPU compatibility,
high-K accuracy and broad multilingual/long-context behavior.
Next: use this exact task-appropriate roster for a prospective neutral development
comparison; no final benchmark access, Jev request or superiority claim occurred.

## Neutral source-native compiler: prospective train/dev contract

MEASURED metadata only: Banking77's complete publisher catalogue has 77
unique IDs; the pinned MASSIVE loader declares 60 unique intent IDs.
Exact source Git blobs, original bytes, derived catalogue hashes, the native
rubric document and current DecisionIR/renderer bytes are recorded in
`research/endgame/neutral_native_catalogue_manifest.json`.
Static literal extraction executes no loader and opens no per-example records.

HYPOTHESIS: the separately registered
`research/endgame/neutral_native_compiler_protocol.json`
can project original train/dev membership into separated serving, DecisionIR,
target and provenance files with exact full-population reconstruction.
The original connected components and role allocation remain unchanged.
Classification retains all K77/K60 publisher IDs on every row. Descriptions
only replace underscores with spaces; no label-selected universe or generated
paraphrases enter the catalogue.

MASSIVE grammar uses its documented naturalness prompt and five native levels;
spelling uses three original error bins. Ordered original ratings, missingness
and native types remain in the target ledger. Distribution targets are empirical
source-retained raters, not workforce/population probabilities. Score DecisionIR
has no fabricated single gold or accepted-label set. Missing ratings remain
unlabeled; spelling is never rescaled to five bins. Native `intent_score` and
`slots_score` are nominal and do not become generic Boolean or ordinal gold.

Selectors may consume only manifest-hashed train/dev exports after code and
projection identities freeze. A nonselecting custodian may stream mixed source
envelopes; parsing them can transiently decode sealed payloads. The contract
therefore prohibits their export, diagnostic use, fitting or selection, rather
than falsely claiming no bytes were decoded. Model text excludes gold,
annotations, worker IDs, source/group identifiers and target metadata.
Reader requests for calibration, confirmation or unused roles must fail before
opening data.

All 21 endgame dimensions remain required. Public legacy intent, source-native
ordinal ratings and 51 locale descendants do not establish arbitrary criteria,
generic Boolean, language usability, natural K1000 or long-context quality.
No model outputs, quality metrics, promotion or new calibration/final exposure
were obtained for this registration. The compiler and its independent verifier
were committed before any selected record was materialized; the materialized
counts and verification receipts are recorded in the following entry.

## Neutral source-native train/dev projection: materialized and reconstructed

MEASURED: the committed compiler materialized the original train and dev
membership into separated serving, DecisionIR, target and provenance streams.
No source row was relabelled, added, dropped or reallocated. Both phases
reproduce the registered connected-component, source-lineage-group and row
counts exactly from the sealed custody manifest:

| Selected denominator | train | dev |
|---|---:|---:|
| Banking77 source rows / decisions | 4,105 | 1,027 |
| MASSIVE source rows | 202,521 | 63,852 |
| MASSIVE decisions (intent + 2 ordinal) | 607,563 | 191,556 |
| Total source rows | 206,626 | 64,879 |
| Total decisions | 611,668 | 192,583 |

The independent verifier re-derives every projected field from the pinned
source envelopes and the canonical DecisionIR classes, recovering both
candidate catalogues from original metadata bytes alone (Banking77 77 unique
IDs at Git blob `cdd2a5c7…`, MASSIVE 60 at `4731be1a…`) and re-parsing the
original MASSIVE rubric document for its five grammar levels and three
spelling levels. It never executes the loader or the compiler's projection
helpers.

MEASURED, dev receipt (64,879 rows / 192,583 decisions): all four streams match
by canonical JSON round trip; coverage and ordering digests match; the
manifest digest binds the receipt; the five registered catalogues, prompt
texts and level descriptions match the recovered originals; all 51 MASSIVE
locales appear; zero invalid present rating types; and requests for
`calibration`, `confirmation`, `unused`, empty and whitespace-padded phases all
fail before any file opens. The phase also fails closed while unverified.

Native missingness is preserved rather than filled. Of 63,852 MASSIVE dev rows,
1,258 carry no valid grammar or spelling rating and stay unlabeled; 187,513
observed native integers are retained in original rater order. Score
DecisionIR carry no fabricated single gold, accepted-label set or rescaled
spelling bins. Rater distributions remain source-retained, not workforce or
population probabilities.

INFERENCE: the source-native path is mechanically faithful, so Choice and
ordinal Score can now be exercised on real public intent data without
inventing labels. This is compiler correctness only. It establishes no model
quality, calibration, language usability, arbitrary-criterion, generic Boolean
or high-K capability, and it does not open any sealed phase.

Five implementation defects surfaced in the verifier before it passed, all
recorded in the retained commit history: a tuple-versus-list comparison across
the JSON boundary, a module passed where a path was required, exercising the
selector gate before writing the receipt it demands, a receipt missing the
manifest digest its own gate requires, and receipt retention happening after
that gate check. Each was fixed in the verifier and committed before rerunning.
The projection compiler was never edited after materialization, because its
digest is frozen into both projection manifests. Each phase retains its earlier
receipt as `verification_receipt_superseded_v1.json` rather than overwriting it.

MEASURED, train receipt (206,626 rows / 611,668 decisions): identical criteria
pass on the larger phase. All four stream digests and sizes match, the manifest
digest binds the receipt, both catalogues and both native rubrics match the
recovered originals, all 51 locales appear, zero invalid present rating types,
and the same five sealed-phase requests are denied. Of 202,521 MASSIVE train
rows, 3,990 carry no valid ordinal rating and stay unlabeled; 594,718 observed
native integers are retained in original rater order.

Both phases are therefore closed and immutable: 271,505 source rows, 804,251
decisions, all reconstructed byte-identically by a verifier that never runs the
compiler's projection code or the MASSIVE loader.

Next: preregister the development Choice and ordinal Score comparison against
this verified projection. All 21 critical cells remain
`UNMEASURED_NEUTRAL_COMPARISON` with `green=false`.

## Neutral development comparison: the frozen Vey Choice arm cannot answer fixed-label intent

Preregistered at `3a45bff` (protocol), amended prospectively at `a4b34dd`
(amendment), arms at `71da28c`, independent verifier at `46de525`.

MEASURED, Vey arm over all 63,852 labeled dev `massive.intent` rows, running
the frozen product `e6b046f` with no source edit and no proxy scorer:

| Quantity | Value |
|---|---|
| rows scored | 63,852 |
| top-1 accuracy | 0.004792 |
| uniform chance (1/60) | 0.016667 |
| `empty_program_fraction` | 1.0 |
| lanes used | `structured` 63,852, `crux` 0, `field` 0 |
| peak RSS | 5.19 GiB |
| wall clock | 107 s, CPU |

Mechanism, from the persisted per-row records rather than inferred: the
source-native question `"Which intent best matches this utterance?"` compiles
to zero stages in every row, so `compose()` has no ordinal stage and its empty
program fallback returns `min(top, key=strip_identity(text))`, which is
alphabetical. Every answer is `alarm_query`, the alphabetically first MASSIVE
intent id, with a degenerate one-hot distribution. The `structured` lane claims
the row before the semantic lanes are consulted, so no grounding ever runs.

INFERENCE: the frozen product's public Choice surface is an ordinal
instruction executor, not a fixed-label classifier. It is not merely
inaccurate here; it is below chance, and it is below chance for a mechanical
reason that no amount of calibration, temperature fitting or threshold choice
can repair. This is consistent with, and independent of, the frozen product's
own documentation, which describes `decide()` as ranking consequence texts
under a compiled priority program.

What remains unknown: whether an added fixed-label head over the frozen
`PooledClassifier` substrate would close this gap. That is a different product
surface and is registered as post-V2 research, not claimed here.

This is the preregistered `vey_empty_program` failure branch executing as
written: the Choice cell is recorded Vey-not-applicable for the frozen
reference, and no proxy scorer was written to fill it. Nothing about promotion,
Pareto credit, B-STEF or endgame completion follows.

The same defect governs the ordinal Score endpoint, and this was confirmed by
direct execution rather than inference. The `massive.grammar_score` source
question, `"Read the sentence out loud. Ignore any spelling, punctuation, or
capitalization errors. Does it sound natural?"`, also compiles to zero stages;
its five rubric levels are candidate *descriptions*, not ordinal axes. Calling
the frozen `decide()` with those five levels as candidates returns level `"0"`,
the alphabetically first id, with `trust.state == "confident"` and
`primary_margin == None`. A confident state on an answer selected by
alphabetical fallback is a trust-labeling defect, independent of the accuracy
defect, and it is recorded as such.

MEASURED scope of `vey-2-final` on this projection: the public `decide()`
surface is an ordinal instruction executor. It cannot express fixed-label
classification or rubric-relative scoring, so on the two endpoints this study
preregistered it has no legal surface at all. Both preregistered arms for the
frozen reference are therefore Vey-not-applicable, by mechanism rather than by
numerical shortfall. The competitor cells remain measurable and are reported
separately.

## Laya pinned English route on MASSIVE dev intent: K=60 imposes a four-token option budget

MEASURED, pinned English bundle `55cf4c4eb`, weights loaded with zero missing
and zero unexpected keys, identical rows to the Vey arm. Measured on a partial
run of 31,228 rows before the completed run was verified:

| Quantity | Value |
|---|---|
| rows scored so far | 31,228 |
| micro accuracy | 0.116799 |
| `en-US` accuracy | 0.4429 |
| non-English mean over 50 locales | 0.1103 |
| best locales | `en-US` 0.443, `fr-FR` 0.296, `pt-PT` 0.282 |
| worst locales | `kn-IN` 0.016, `te-IN` 0.016, `he-IL` 0.021, `ur-PK` 0.021 |
| temperature bucket applied | `choice:11+` (0.10058280825614929) |
| option token spans | `(4,4,4,4,4,4,4,4)` on every row |
| markers vs options | 60 vs 60 on every row |
| sequence length | 259 to 512, hitting the `max_len` 512 cap |

The measured head budget limits each option to `[MASK]` plus at most three
content tokens. That restricts label visibility but does not remove all label text.
Correction to the original anonymous-marker diagnosis: a current pinned-runtime
input-only reconstruction finds 43 distinct token blocks across the 60 options
of one dev row; reversing the catalogue changes the encoded sequence. This
used zero model forwards and is not an option-order quality experiment.
The earlier claim that decisions cannot depend on option text is withdrawn.

The partial locale pattern does not identify the truncation's causal effect.
No controlled budget intervention was run. The vendor's embedding-shortlist
and coarse-to-fine remedies remain different configurations, unmeasured here.

The vendor's `MASSIVE51` macro claim of 0.4008 is advertised on the
multilingual route, which this run does not exercise. Comparing 0.1168
against 0.4008 would be a route mismatch and is refused. The multilingual route
is preregistered separately at `ee2fa01` with identical rows and metrics.

One measurement defect is retained rather than silently corrected: this run
predates the span fix at `ddd8d3c`, so its `min_option_tokens` field reports a
spurious 1 on every row. The unaffected `option_tokens_head` field above is
what the mechanism claim rests on. The completed run's numbers are taken only
after independent verification, and this partial figure is explicitly not a
final number.

## Verified Choice comparison: Vey 2 loses to the pinned Laya English route by 11.2 points

Protocol `3a45bff`, amendment `a4b34dd`, arms `71da28c`, verifier `46de525`
with two later verifier fixes (per-arm absent-field checks, log-space exact
binomial CDF), independent verification receipt written after both runs
completed.

MEASURED on all 63,852 labeled dev `massive.intent` rows, 1,252 lineage
utterance groups in 1,024 input-only connected components, K=60, identical
serving fields to both arms:

| Quantity | Vey 2 (frozen `e6b046f`) | Laya English (pinned `55cf4c4eb`) |
|---|---:|---:|
| top-1 accuracy | 0.004792 | 0.116739 |
| NLL | 27.4986 | 17.0523 |
| Brier | 1.99042 | 1.52798 |
| ECE (15-bin, corrected predicted-label confidence) | 0.995208 | 0.709775 |

Corrected paired ratio bootstrap over 1,024 connected components, 10,000
resamples, seed 0, retaining the row-weighted accuracy estimand:
delta = **-0.111946**, nominal 95% CI **[-0.122288, -0.101730]**. The interval
is wholly below the -0.01 margin. It is not a simultaneous non-inferiority gate.
The former 1,252-lineage interval **[-0.121046, -0.102832]** is superseded:
connected duplicate lineage groups were incorrectly treated as independent.

Exact discordance on shared rows: Vey correct / Laya wrong = 293, Laya correct /
Vey wrong = 7,441, both correct 13, neither 56,105. McNemar n10/n01 = 293/7441.
The historical Clopper-Pearson calculation bounds Vey-only-correct events at
0.005054. That is the wrong discordance direction for a harmful Vey regression:
Laya-only-correct events are the harmful direction when evaluating Vey against
Laya. This historical bound supplies no Vey non-inferiority evidence. The paired
accuracy interval above independently establishes the loss.
Both row-binomial bounds assume independent trials, violated by locale
descendants and connected duplicate lineage groups. They are descriptive
calculations, not cluster-valid risk certificates. The Laya-only event calculation
is 0.118644, without an inferential guarantee for this clustered population.
The reported bootstrap intervals are nominal descriptive 95% intervals, not
the preregistered simultaneous Bonferroni family-wise gate.

Verification controls, all computed from the artifacts rather than asserted:
membership problems 0 of 63,852; shared-exact-workflow mismatched input digests
0 of 63,852 with shared digest `57a1c14f84041e7466360028bd32cb492dff0f3454881aab76c57adc42ff8037`;
accuracy, NLL, Brier and paired accuracy quantities reconstructed without
changing any arm's predictions. Calibration corrections are disclosed below.

Interpretation, stated conservatively. This is a legacy replication of a public
benchmark with prior Vey and Laya exposure disclosed, not fresh transfer. It
establishes that the frozen `vey-2-final` reference loses to the pinned Laya
English route on native multilingual intent by a wide, statistically resolved
margin, for the mechanism recorded above: Vey's public `decide()` surface is an
ordinal instruction executor with no fixed-label or rubric-relative mode, so
both of its preregistered arms here answer by alphabetical fallback. Laya's
K=60 truncation restricts the pinned Laya route's input, but no intervention
here isolates its effect on accuracy. This is not a capability ceiling; the
multilingual route is measured separately below.

What this does NOT establish: nothing about Laya's multilingual route, Jev, any
non-English route selection, or any post-V2 architecture. It earns no promotion,
no Pareto credit, no B-STEF clearance and no endgame completion, and it does not
change any frozen Vey 2 scientific conclusion.

Two verifier defects surfaced before this receipt and are retained in history
rather than corrected silently. The first compared a uniform field list across
both arms using `.get()`, so the Vey arm's absent `question`/`state`/
`candidate_ids` fields read as `None` and manufactured 63,852 false
"differs from projection" errors while silently emptying the shared-digest
control to 0 rows. The second overflowed a float in the naive binomial CDF at
n = 63,852. Both are fixed in the verifier; no arm data was modified.

Next branch, per the preregistered follow-up: the pinned multilingual route on
identical rows (`ee2fa01`), which is also the arm that tests the vendor's
0.4008 MASSIVE-51 claim.

## Laya route selection on identical MASSIVE dev rows: multilingual beats English by 6.1 points

Preregistered at `ee2fa01`; verifier generalized for a second route and
confirmed to reproduce the earlier Choice receipt exactly before use. Both
routes loaded with zero missing and zero unexpected weight keys. Identical
63,852 rows, identical inputs (0 mismatched input digests), identical metrics.

| Quantity | English route | Multilingual route |
|---|---:|---:|
| top-1 accuracy (micro over rows) | 0.116739 | 0.178021 |
| NLL | 17.0523 | 4.48483 |
| Brier | 1.52798 | 1.03461 |
| ECE (15-bin, corrected predicted-label confidence) | 0.709775 | 0.268751 |
| `en-US` accuracy, complete run | 0.434505 | 0.314696 |
| non-English mean over 50 locales, complete run | 0.110383 | 0.175288 |
| Option token budget | At most 4, including `[MASK]` | At most 4, including `[MASK]` |

Corrected paired ratio bootstrap over 1,024 connected components, 10,000
resamples, seed 0: multilingual minus English = **+0.061282**, nominal 95% CI
**[+0.046990, +0.075947]**. The earlier 1,252-lineage interval
**[+0.048957, +0.073341]** is retained but superseded by the component correction.

The route difference is measured; its causal decomposition is unresolved.
`en-US` accuracy falls from 0.434505 to 0.314696, while the non-English mean
rises from 0.110383 to 0.175288. Earlier locale figures used partial runs.
Both routes have a four-token option budget; actual fixed-512-row controls
observe a minimum span of three, while the first eight option spans are four.
Different tokenizers need not retain equivalent content under that budget.
These arms change encoder, tokenizer and weights together, so the difference cannot be
attributed to encoder training alone.

Correction to the original calibration report: the verifier used gold-label
probability rather than predicted-label confidence for ECE, and compared a
probability with an answer string for Vey correctness. The earlier ECE values
0.004792 / 0.007877 / 0.047064 are invalid. Reconstructing from unchanged
predictions gives Vey 0.995208, English 0.709775 and multilingual 0.268751.
Accuracy, NLL and Brier are unchanged; the ECE correction itself did not alter
accuracy intervals. The later connected-component correction above supersedes
both intervals. Multilingual has lower measured ECE, NLL and Brier on these rows.
These likelihood metrics use persisted eight-decimal probabilities with a
`1e-12` NLL floor. They are serialization-defined scores, not reconstruction
of the unrounded logits' likelihood.

Vendor claim: the advertised multilingual MASSIVE-51 macro accuracy is 0.4008.
An input/result census finds exactly 1,252 rows per locale in both runs.
Consequently the measured macro-over-locales accuracy equals micro accuracy:
English 0.116739; multilingual 0.178021. The earlier claim that these statistics
were necessarily incomparable is withdrawn. The advertised number is not
reproduced under this pinned revision and protocol. Different splits, rendering,
route configuration or vendor aggregation remain unresolved; this does not
refute the vendor's separate benchmark.

What this does NOT establish: nothing about Laya's typed-decisions route, Jev,
or any configuration using Laya's documented high-K remedies (embedding
shortlist, coarse-to-fine decomposition), none of which were exercised here.
No same-task comparison with the frozen Vey architectural cells was made.
No promotion, Pareto credit or B-STEF clearance follows.

Next branch, per the registered order: the ordinal Score endpoint on
`massive.grammar_score`, where the frozen Vey reference has no legal surface
and Laya is therefore measured alone against the source-native rubric.

## Ordinal Score on MASSIVE grammar rubric: exploratory multilingual result

MEASURED on 62,594 labeled dev rows with the pinned multilingual bundle
`55cf4c4eb`; 62,338 have at least three observed raters. This historical run
cited the Choice-only multilingual protocol, which did not prospectively cover
Score execution. It is exploratory development evidence, not a preregistered
Score route result. Amendment `33c880c` discloses that defect without rewriting
the original protocol or predictions.

The v2 verifier reconstructs expectations and errors from persisted eight-decimal
probabilities and guarded native targets, verifies complete unique membership
and persisted serving fields, and rejects nonfinite or unnormalized probabilities.
Zero membership problems; 62,594 independently reconstructed rows.

| Quantity | Value |
|---|---:|
| mean absolute error (levels 0-4) | 1.912699 |
| normalized MAE | 0.478175 |
| mean signed error | -1.899193 |
| signed-error slope against rater mean | -1.031111 |
| Spearman rank correlation vs rater mean | -0.034411 |
| normalized RPS | 0.281677 |
| retained-rater NLL, at least three raters | 1.565044 |
| retained-rater Brier, at least three raters | 0.599801 |

The expected-level output has weak negative pooled rank association with the
observed rater mean and a large downward bias on this measured task. These
statistics do not identify a rubric-prior mechanism, prove all ordinal
information absent, or establish which rubric formats were seen in training.
Those earlier causal and training-exposure claims are withdrawn. A constant
prediction has undefined Spearman; an additive offset of a nonconstant perfect
prediction preserves its ordering. Neither observation establishes the model's
failure cause here.

Full target census: exactly 37,666 / 62,594 rows (60.1751%) have mean level four,
with overall rater mean 3.749087. The earlier 29,843 / 36,663 count used a rounded
partial histogram, not exact level-four membership; its claimed 89% was also
arithmetically wrong. Full predicted argmax counts at levels 0/1/2/3/4 are
15,407 / 20,518 / 109 / 19 / 26,541. These are descriptive output statistics,
not evidence of a causal mechanism.

Targets remain finite retained-rater distributions, not population
probabilities. NLL, Brier and RPS are agreement scores against those targets;
no population calibration claim follows. The measured frozen public `decide()`
surface has no rubric-relative mode on this question. No proxy scorer was added.

Earlier verifier failures remain in history: the wrong `level_max` field check
manufactured 62,594 missing-field errors, then v1 trusted arm-derived error
fields and did not enforce complete unique membership. Original receipts are
retained; v2 supersedes their verification status. Tiny numerical differences
from reconstruction reflect saved probability rounding, not changed predictions.
The latest receipts are `verification_choice_routes_v6.json` and
`verification_score_multilingual_v3.json` under the comparison data directory.
Choice replay also checks complete unique membership, native gold, correctness
flags, probability validity and argmax consistency. Throwaway CLI controls
reject missing/duplicate rows, changed input or gold, false correctness flags,
nonfinite or unnormalized probabilities and inconsistent winners.

No result here establishes typed-decisions Score quality, Jev Score, another
rubric, fresh transfer, promotion, Pareto superiority or B-STEF clearance.

## Registered English grammar Score: complete dev replay

MEASURED on all 62,594 labeled grammar dev rows; receipt
`verification_score_english_v2.json` independently reconstructs probabilities,
native targets and errors with zero membership problems.

| Metric | English | Historical exploratory multilingual |
|---|---:|---:|
| MAE, native levels 0-4 | 2.853026 | 1.912699 |
| Normalized MAE | 0.713256 | 0.478175 |
| Mean signed error | -2.851510 | -1.899193 |
| Signed-error slope against rater mean | -0.986646 | -1.031111 |
| Spearman rank correlation | 0.032804 | -0.034411 |
| Normalized RPS | 0.569855 | 0.281677 |
| Retained-rater NLL | 3.832799 | 1.565044 |
| Retained-rater Brier | 1.110896 | 0.599801 |
| Rows with at least three raters | 62,338 | 62,338 |

Both routes replay the same guarded native inputs, digest
`41487b5a717cec9b24c07c799b869c4a5d4da9dbe43c223e5ef828a62d625aa4`.
English grammar was registered under the original comparison protocol.
Multilingual grammar remains exploratory; the amendment cannot retrospectively
change its status. Neither original grammar run persisted rubric description
strings, so these receipts do not prove the exact encoded input bytes.

The study uses historical raw-bundle temperatures, not current 0.3.27 served
calibration. Its English Choice temperature was `.10058280825614929`, whereas
the current served floor is `.5`. An exact replay at that floor cannot recover
original logits from rounded saved probabilities.

Matrix migration accepts only corrected Choice receipt schema v3 and
independently reconstructed Score schema v2. It rejects old schemas without
writing the matrix, separates endpoint/route provenance, and clears a stale
Score scalar if no verified English grammar receipt exists. Grammar receipt
order does not change the English scalar. Actual matrix update retains all
21 critical dimensions and zero green cells.

## Registered spelling Score and bounded determinism controls

Both spelling routes completed all 62,594 eligible native dev rows under
amendment `33c880c`; receipts `verification_spelling_english_v2.json` and
`verification_spelling_multilingual_v2.json` independently reconstruct the
native targets, probabilities and derived metrics with zero membership problems.

| Metric | English | Multilingual |
|---|---:|---:|
| MAE, native levels 0-2 | 1.449158 | 1.006526 |
| Normalized MAE | 0.724579 | 0.503263 |
| Mean signed error | -1.449149 | -1.005592 |
| Signed-error slope against rater mean | -0.992958 | -0.995728 |
| Spearman rank correlation | 0.004382 | 0.003282 |
| Normalized RPS | 0.555603 | 0.310854 |
| Retained-rater NLL | 2.241955 | 1.387245 |
| Retained-rater Brier | 1.187748 | 0.781643 |
| Rows with at least three raters | 62,338 | 62,338 |

Shared native-input digest:
`e6644d5b256834905e6fd192beabf98c59514af0f5d83d020f491f747568eb38`.
Unlike the historical grammar records, both spelling runs persist and validate
the complete native rubric candidate descriptions. These are retained-rater
agreement scores, not population calibration or evidence of a training cause.

Five actual fixed-512-row reruns matched saved answers/expected levels and
eight-decimal probability distributions exactly: frozen Vey Choice, English
Laya Choice, multilingual Laya Choice, English grammar, and multilingual
spelling. Each had zero mismatches; `verification_determinism_v1.json` records
the comparisons; the tracked manifest binds hashes. Determinism is demonstrated only for the
observed subset and saved precision, not unrounded logits or every shape.
The 16-row multilingual spelling smoke is preserved separately.

## Native comparison scope and inference corrections

The input-only component manifests identify 1,024 MASSIVE dev components
containing 1,252 lineage groups and 63,852 locale rows. Schema-v3 Choice replay
reconstructs those component IDs from guarded provenance, validates persisted
IDs when present, and resamples whole components. Component sizes differ, so
each draw divides summed paired correct-count differences by summed row counts;
averaging component means would silently change the estimand. Predictions and
native targets remain unchanged. Old receipts retain the superseded intervals.

The original comparison protocol incorrectly said Banking77 has no dev rows.
The verified native manifest already records 4,105 train and 1,027 dev records;
dev has 1,024 components. That protocol stays immutable and its exclusion
rationale is withdrawn. No Banking77 model predictions or quality measurements
were produced by this MASSIVE-only comparison. Shared readers join all guarded
dev target records before endpoint filtering, so Banking77 target records have
been decoded; they cannot be described as never opened. A Banking77 comparison
requires a separate prospective scope registration, not retroactive expansion.

The matrix retains all four Score endpoint/route results separately. Its scalar
Score cell remains the registered English grammar normalized MAE; the
exploratory multilingual grammar result does not acquire registered status.
All 21 critical dimensions remain non-green. No release, Pareto, certificate,
fresh-transfer, B-STEF, controlled-performance or endgame-completion claim follows.
Custody is recorded in
[`neutral_comparison_result_manifest.json`](../research/endgame/neutral_comparison_result_manifest.json):
27 retained prediction/replay/control artifacts and 11 implementation/protocol
files, each with size and SHA256. Raw records remain outside Git.

## Native English state-field supervision: Banking77 intent earns the screen, MASSIVE intent misses

Preregistered at `9e1b65b` (protocol + guarded data/material/model/train/verify
implementation), run under the committed custody cutover `ba27eac`. Every arm
ran offline on the pinned stock `microsoft/deberta-v3-xsmall`
(`eb2d654bf0a5b628c8be6c4be7d29118fbef95b8`) with no checkpoint substitution.
This is a new source-native study, not a rerun of the retired authored
literal-registry or direct-NLI-head experiments, which CBF-6/7 already failed.

MEASURED on the selected English native projection: 4,105 Banking77 train
records (1,027 dev) and 3,971 MASSIVE en-US train records (1,252 dev). Whole
original components split 70/15/15 into disjoint fit/selection/calibration, so
no component or duplicate lineage crosses a phase; the original dev stays a
development assessment, not a sealed final.

| Field | Arm | real top-1 | masked | swapped vs original | swapped correct-new | status |
|---|---|---:|---:|---:|---:|---|
| banking77.intent | full | 0.8199 | 0.0097 | 0.0019 | 0.8179 | passed |
| banking77.intent | frozen | 0.4859 | 0.0243 | 0.0088 | 0.4830 | failed |
| massive.intent | full | 0.7564 | 0.0759 | 0.0080 | 0.7556 | failed |
| massive.intent | frozen | 0.5120 | 0.0759 | 0.0200 | 0.5096 | failed |

Component ratio bootstrap over 1,024 input-only connected components, 10,000
resamples, seed 0, row-weighted estimand. Full arm Banking77 real-minus-masked
accuracy CI **[+0.7854, +0.8341]** and model-minus-prior CI **[+0.7707, +0.8205]**;
MASSIVE model-minus-prior CI **[+0.6282, +0.6879]**. These are nominal
descriptive intervals, not the endgame simultaneous gate.

The full-encoder recipe earns the Banking77 development screen; the frozen
recipe reaches 0.4859. This supports allowing encoder adaptation in this recipe,
but both heads are trained jointly with their encoders, so it does not isolate
all gain to representation changes. MASSIVE reaches 0.7564 and misses the
0.80 accuracy and 0.80 swapped-correct-new gates. The selected English export
has no grammar or spelling retained-rater targets: both ordinal fields are
data-insufficient, and neither ordinal head was trained.

Controls reconstructed from the artifacts, not asserted: constant-mask states
collapse accuracy to 0.010 / 0.076; donor-swapped states collapse
against-original accuracy to 0.002 / 0.008 while swapped-correct-new tracks the
real accuracy; all 2,279 dev decisions had lawful different-component,
different-target donors (zero unavailable); repeated canonical Choice and
membership reads added no encoder calls while an uncached read advanced the
counter. Candidate-ID reversal is an exact gather, not alias understanding.

The first full-arm smoke stopped before an
optimizer step because the stock-key ledger omitted the disabled absolute-position
table alias `deberta.embeddings.position_embeddings._weight`. Two subsequent
custody failures were caused by parent edits: a verifier edit between arm starts,
then a registry/trainer edit after frozen training. Those guards correctly
detected execution drift. The mutable registry was removed; all three arms were
rerun under committed source state `ba27eac`, with unchanged data and thresholds.
Superseded arm outputs and smoke receipts remain under `retained-pre-cutover/`.
The original dataset manifest retains historical materialization-source hashes;
current arm source hashes are verified separately.
The original failed verification receipt was deleted before the rerun rather than
archived. Its observed error is retained as a labelled transcript extract;
the missing original payload is not reconstructed.

No sealed-final, fresh-transfer, competitor, many-axis, free-form-query,
generic-Boolean, certificate, controlled-performance, shipping or B-STEF credit.
Custody is recorded in
[`native_field_result_manifest.json`](../research/endgame/native_field_result_manifest.json):
36 current artifacts, 39 retained raw artifacts, one failure transcript extract
and seven source/protocol files. Per-class and donor maps remain in the hash-bound
raw verification receipt rather than being duplicated into this manifest.
The manifest initially published at `64da723` accidentally contained the earlier
neutral comparison after a failed construction reused a retained variable.
It is corrected from the native `PASS` receipt; predictions and gates are unchanged.

## NATIVE-2 architecture comparison: pre-training checks

Preregistered at `db12aa1`, corrected before execution and integrated at
`74255c7`. Cross, dual and Evidence Pages use the same NATIVE-1 English data,
component splits and stock encoder. Quality training is a separate pending
measurement; these checks earn no quality, transfer or release credit.

All three actual CUDA optimizer smokes have finite loss, nonzero named
encoder/head gradients and optimizer deltas, changed encoder fingerprints,
and exact save/perturb/restore output equality. Smoke weights are discarded.
The first cross smoke timed out on the resource guard after the optimizer step.
A bounded diagnostic separated live CUDA allocation from recyclable reserved
cache. `b1912cd` releases only idle own cache under low free VRAM, then applies
the unchanged resource thresholds; it does not suppress external pressure.

A four-decision live control probe exposed missing raw readout fields in
candidate-count rows. `0fbd9c4` repairs the producer without changing model,
loss, splits or quality gates. Repaired probes independently reconstruct
target-blind component-seeded subsets, including absent-gold cases.
Subset membership is random input coverage, not learned retrieval recall.

On one state and two public catalogues, the fresh computational probe observes
137 state-containing sequences for cross versus one for dual and Pages.
The foreign catalogue is unlabelled and excluded from semantic metrics.
Later identical-query repeats incur zero additional encoder forwards for
every arm. That is generic memoization, not Pages-specific semantic credit.
Candidate encoding is counted separately; no latency or throughput claim.

[`native_arch_smoke_manifest.json`](../research/endgame/native_arch_smoke_manifest.json)
binds 44 live, interrupted and diagnostic artifacts, including both failed
probes. Counter reconstruction is receipt-backed instrumentation, not an
independent CUDA kernel profiler. Sealed phases and frozen product stay untouched.

The first registered quality fit then fails during cross backpropagation after
reducing microbatch states from four to two to one. Epoch 0 inner-selection
output and all partial artifacts are retained in `cross-fit-oom-v1`; no training
epoch completes and no DEV quality is assessed. The explicit CUDA exception
reports 7.69 GiB live PyTorch allocation, 359.09 MiB unused reserve and only
105.94 MiB device free. This is live activation pressure, distinct from the
earlier recyclable-cache stall.

The prospective execution correction enables non-reentrant encoder activation
checkpointing with RNG preservation for all three arms. It retains FP32,
the full candidate catalogue, 512-token cap, effective batch 32, objective,
optimizer and all selection/calibration/quality gates. Before restarting
from fresh seeds, require stochastic logits/gradient parity and real CUDA
backpropagation through the exact first effective block and longest FIT input.
Training recomputation is not included in inference counters or performance
claims. No completion or quality credit from the interrupted fit.

The corrected CUDA execution checks pass for all three arms. Stock-model
ordinary versus checkpointed logits/gradients match on the bounded parity
probe (maximum gradient difference zero), including post-backward RNG state.
Each arm completes the exact first 32-decision effective block and backpropagates
the input-only longest FIT state (77 joint candidates, padded length 130).
The separate stochastic CPU regression and existing invariants pass 73 tests.
[`native_arch_activation_manifest.json`](../research/endgame/native_arch_activation_manifest.json)
binds the actual proof and retained OOM artifacts. Own allocation high-water
counters are diagnostic shared-host values, not deployment memory or a
controlled performance comparison. Full quality measurement restarts fresh.

User-directed execution change: all subsequent GPU work runs on Modal, never
on the local card. Local job `bg_220` is cancelled and its three partial
artifacts are retained in `cross-fit-interrupted-modal-migration-v1`; persisted
history contains only epoch 0. This interruption is not a neural quality
failure. The remote amendment restarts all arms fresh on one serialized T4,
with exact code, permitted inner-phase data, stock-weight and dependency hashes.
Only 13 explicit allowed input files are transported; no source sealed rows,
mixed raw corpora or credentials. Dedicated private-volume paths preserve the
original dataset manifests. CPU preflight and actual Modal liveness precede
quality fits; timeouts are bounded and automatic retries disabled.
[`native_arch_modal_transport.json`](../research/endgame/native_arch_modal_transport.json)
records the transfer contract. The pooled reference used RTX3060; numerical
identity across hardware and any cross-hardware performance gain are unmeasured.

The actual Modal CPU preflight passes with every transported input and pinned
dependency verified. The first T4 smoke fails before model loading: the trainer
resolves the volume symlink to `/__modal/volumes/<id>`, whereas the immutable
dataset manifest binds the original absolute logical paths. A CPU-only remote
diagnostic proves every phase hash matches and every resolved path spelling
differs. This is a mount-custody execution failure, not a quality result.
The prospective correction preserves the registered logical root and checks
physical root equivalence separately; no source manifest, hash check, neural
recipe or gate is relaxed. Fresh actual Modal liveness remains required.

The subsequent serialized T4 liveness runs pass for cross, dual and pages,
as recorded in `native_arch_modal_smoke_results.json`. This proves numerical
smoke execution only; complete quality artifacts and independent reconstruction
remain required.

MEASURED verifier defect before complete quality reconstruction: supplying no
paired intervals to `screen_arch` returns PASS for all three synthetic arms
when their other controls pass. The implementation conditionally omits each
required comparison gate when its evidence is absent. The reproducer is
retained with SHA-256
`bb5c37f0e4aa44fe1c58e205473a4446cabe5acf2a7de149e85aade2f0235bde`.
The frozen protocol requires those intervals. Implementation remains unchanged
while the registered GPU run is in flight; correction and missing-evidence
regressions must precede any architecture credit. No threshold changes.

The separate published-Laya capture smoke completed three typed outputs while
NATIVE-2 was still running. This violates the peer protocol's explicit sequence;
CPU-only execution and a smoke label do not create an exemption. The original
capture and receipt remain intact. Its three identical answers neither prove
state sensitivity nor rule out degeneracy; historical measurements from another
runtime do not establish either for this capture. No full-DEV or quality credit
follows. `native_execution_audit_manifest.json` binds these corrections and
retained evidence. Further peer execution waits for the registered dependency.

Peer payload-custody correction before the full comparison: the original
capture omitted complete returned diagnostics, including on validation failure.
An exclusive raw sidecar now flushes each returned payload before checking it;
existing raw evidence is refused before runtime setup. Synthetic entrypoint
execution verifies successful diagnostic retention, and 20 peer capture/verifier
tests pass, including malformed question/choice payload retention. No model
loads or forwards were performed for this correction. Published call semantics,
membership, protocol and all 15 Modal transport source files remain unchanged.
The historical three-row smoke cannot earn retrospective full-payload custody.

### NATIVE-2 Modal timeout: retained incomplete training

MEASURED execution failure on the original registered transport: the cross call
hit its 14,400-second deadline. The client reports `FunctionTimeoutError`;
the remote receipt and trainer metadata report generic `InputCancellation`
with “cancelled by user” wording. No user cancellation caused this termination.
The original log footer, remote receipt and every partial artifact are retained.
History contains completed epochs 0 through 6, with partial selected epoch 6.
All persisted partial artifact hashes match their downloaded bytes.

Selected restore, calibration and DEV/control captures were not reached.
Dual and pages training were not called. Thus the registered full comparison is
incomplete: no primary DEV metric, confidence interval, quality PASS/FAIL or
negative neural result follows from the timeout. Liveness smoke remains separate
evidence and cannot replace the missing quality comparison.

`native_arch_modal_timeout_result_manifest.json` binds the preserved lineage.
The timeout is operational, not a terminal scientific block. The chosen recovery
is a fresh isolated volume, the same ten-epoch recipe and gates, and a bounded
43,200-second allowance per arm, with no automatic retries. The selected partial
checkpoint lacks optimizer and continuing RNG state, so it cannot reproduce
an uninterrupted AdamW continuation; recovery must restart freshly seeded.
Whether the larger deadline suffices is unmeasured. Failed volume and artifacts
remain unchanged. Preflight and fresh three-arm numerical smoke precede recovery
training; no local GPU execution is allowed.

Prospective correction after termination: all three architecture screens now
explicitly fail when their required comparison interval is absent or invalid.
The original missing-interval reproducer now returns FAIL for cross, dual and
pages. A failed or incomplete registered study cannot earn architecture
selection, even if an available local screen passes. The original defect
receipt remains unchanged; scientific thresholds and neural recipe are
unchanged. CPU verification passes 192 endgame regressions, including missing
intervals, threshold equality and complete-study selection boundaries.

`native_arch_modal_timeout_recovery_protocol.json` registers fresh v2-volume
preflight and three new smokes before training. Its guard rejects scientific
protocol changes beyond the operational timeout amendment. Independent local
custody checks match all 17 transported sources and 13 unchanged allowed inputs;
model and training implementation bytes match the original transport.
The 13 pinned inputs were uploaded explicitly to v2. CPU preflight verifies
remote bytes and never overwrites inputs. No failed v1 execution receipt or
checkpoint is copied into v2. Original QNATIVE-2 authority remains frozen;
recovered-result use requires a separate lineage/architecture amendment.

MEASURED actual v2 CPU preflight passes, followed by all three serialized T4
numerical smokes. Downloaded metadata hashes match each COMPLETE runtime
receipt; selected smoke restore error is exactly zero, encoder bytes change,
and every recorded active parameter delta is nonzero. The v2 preflight binds
the original stock encoder fingerprint and unchanged input counts. Receipts
are retained in `modal-recovery-v2`; `native_arch_modal_recovery_smoke_results.json`
binds their hashes to preregistration `408f93b`. Fresh full quality execution
has been launched, but no completed architecture result is yet verified.
These are execution proofs, with no quality or performance credit.

## QNATIVE-1: natural-question source projection

Preregistered at `05e5d36`; allocation-order correctness repair at `34f8d73`.
The first train attempt failed before paper/question/annotation payloads:
the original custody hash orders groups by allocation rank, whereas the new
compiler initially used lexical IDs. The refusal and all partial artifacts
remain retained. The correction reproduces the original rank order without
changing any membership, source bytes or native targets.

Both train and DEV independently reconstruct every serving, DecisionIR,
target, provenance and source-census byte and pass the current receipt-bound
reader guards. The source database remains unchanged; 65 behavioral
missingness, type, support, offset and access regressions pass.

| Source population | Train | DEV |
|---|---:|---:|
| Papers / connected components | 532 | 319 |
| Original questions | 1,580 | 1,013 |
| Retained annotations | 1,636 | 1,396 |
| Emitted task decisions | 4,978 | 3,186 |
| Native yes/no questions with an observed Boolean | 238 | 147 |
| Answerability questions with native targets | 1,580 | 1,013 |
| Questions with nonempty native evidence targets | 1,342 | 915 |
| Questions with nonempty native extractive targets | 790 | 597 |

Null native yes/no fields never become false. Exact evidence matching retains
all unmatched targets: 335 of 2,516 train evidence items and 244 of 2,227 DEV
items are not source-block matches. Extractive spans retain every exact
occurrence and unmatched item: 7 of 1,532 train and 6 of 1,440 DEV spans are
unmatched. These targets cannot be removed from future quality denominators.
Free-form answers remain separate provenance, never serving state or inferred
extractive labels.

[`neutral_qasper_native_result_manifest.json`](../research/endgame/neutral_qasper_native_result_manifest.json)
binds all current and retained evidence. Original source-specific CC-BY-4.0
attribution and modification notices remain; underlying full-paper
republication rights remain under review. No raw source redistribution or
shipping-model permission follows from this projection. QASPER is public and
may overlap LongBench/SCROLLS or pretraining; this is not fresh final evidence.
There are zero model forwards and no quality, performance or Pareto credit.
The next separate study must preregister actual natural-variable-question
Boolean/answerability and evidence-conditioned behavior before model work.
Its architecture choice waits for NATIVE-2, not QASPER DEV model outcomes.

The next study is registered prospectively at
`research/endgame/neutral_qasper_question_study_protocol.json` (sha256
`40d66173e91a827725dc107553a644894f39212f4667d4d47c33e45491f73f1f`). It
covers the four natural-variable-question endpoints on this projection, keeps
the inner component split and the sealed phases closed, and defers architecture
selection to the measured NATIVE-2 result through a separate amendment before
any QASPER model forward. Unmatched native evidence and spans stay in every
future quality denominator. No model has loaded against this projection.

MEASURED QNATIVE-2 membership census under preregistration `f49933a`, committed
implementation `1648ff3`: the registered NATIVE-1 component hash rule assigns
QASPER train papers to fit/selection/calibration; outer DEV remains DEV.

| Phase | Papers/components | All endpoint decisions | Native yes/no | Answerability | Evidence retrieval | Extraction |
|---|---:|---:|---:|---:|---:|---:|
| Fit | 354 | 3,363 | 162 | 1,067 | 1,067 | 1,067 |
| Selection | 87 | 785 | 38 | 249 | 249 | 249 |
| Calibration | 91 | 830 | 38 | 264 | 264 | 264 |
| DEV | 319 | 3,186 | 147 | 1,013 | 1,013 | 1,013 |

Independent direct hash reconstruction matches every persisted membership row:
8,164 decisions, 851 components, zero duplicate IDs or cross-phase component
leaks. Every endpoint has nonempty coverage in all four phases. This passes
membership custody, not a semantic gate; no confidence interval or model quality
is measured. In particular, yes/no selection and calibration each contain only
38 questions. Their calibration utility remains unmeasured.

The existing guarded train/DEV provenance reader decodes annotation fields,
but this census uses and persists membership fields only. No targets or
annotation text become model inputs. No model loads, forwards, architecture
selection or sealed-phase access occurred.
`neutral_qasper_question_split_result_manifest.json` binds the census manifest
SHA-256 `42b5dbc6d3398da8bb1193db5216c455aa1beac877a8fd0e77dc8eff518e196a`
and independent receipt. The next branch remains the deferred architecture and
mechanism-gate amendment after NATIVE-2 evidence; this census changes no split.

## HH-CENSUS-1: native human-preference input custody

HYPOTHESIS: helpful-base TRAIN pairs can expose an exact shared dialogue
context and two distinct last-assistant response candidates without using
`chosen`/`rejected` field roles as model inputs. This is a source-feasibility
census, not a learned mechanism or quality experiment.

Prospective contract: `research/endgame/neutral_hh_native_input_protocol.json`,
SHA-256 `44dbcc32b446654aac816403e6d9f6206e2ff3ced7affe35a0f34f14d03b3686`.
The protocol and implementation must be committed before acquiring or opening
TRAIN rows. No model, tokenizer, GPU, fitting, target export or split allocation
is permitted. Every native line stays in the denominator, including mismatched
contexts, absent formatting and blank or identical response candidates.
Role-swapping must leave input features invariant. Independent reconstruction
must match every persisted metadata row and the complete source census.

MEASURED metadata only: the owner [HH-RLHF dataset card](https://huggingface.co/datasets/Anthropic/hh-rlhf/blob/09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa/README.md)
declares MIT and preference/reward-model research as its intended use. It
explicitly warns against supervised dialogue-agent training and documents
potentially offensive content. The pinned revision is
`09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa`; the card SHA-256 is
`f75f40db0268656ba07736ec8e59a9720c1910ce554c85354cec74b1c8bda175`.
The chosen/rejected observations are human judgments over model-generated
responses, not factual truth, human-authored response targets or measured
population probabilities. Rights classification remains `conditional/review`;
local intended research custody does not authorize shipping training or
redistribution of private information.

Only `helpful-base/train.jsonl.gz` is registered for acquisition: 16,200,131
compressed bytes, SHA-256
`518a5bf288456fc9f3b7c980c54116fba0c52f274d3f4d344675d83e4058f6f4`.
The retained publisher metadata reports VirusTotal 0/75 for that file.
Harmless-base TRAIN is **unacquired** because its metadata reports suspicious
1/76; this is a publisher scan report, not an independently reproduced malware
finding. Helpful online/rejection-sampled tranches, red-team transcripts and
all TEST rows are excluded from this census.

Exact metadata/card/license bytes are retained outside Git under
`vey-data/decisionmix/endgame/hh-native-v1/metadata/`; its receipt SHA-256 is
`a3a509b61b1c68f914cfecf5cd72daa7f67034f1a7f406d5bcea939aca388755`.
The normalized first-human-prompt grouping is only an input-derived proxy.
Missing source conversation and annotator IDs prevent a claim of full original
lineage isolation or an independent-example count.

Primary outputs will be deterministic projection coverage, reason counts,
context/root-prompt groups and duplicate input pairs. No confidence interval,
accuracy, calibration, context-support, shipping or Pareto credit follows.
Source row counts and projection feasibility remain unmeasured. A future
natural-question human-preference Choice study requires a separate prospective
architecture and gate contract after NATIVE-2 evidence; this census does not
change the in-flight Modal study or permit B-STEF.
































## Frontier topology and oracle closure on food progress (research)

With the control's certificate radius of 0.0487 on 250 held-out decisions, the
unresolved food-progress edges form small graphs:

| statistic | value |
|---|---:|
| decisions with no frontier | 0.328 |
| mean unresolved edges | 1.596 |
| mean max degree | 1.312 |
| mean connected components | 0.712 |

A third of decisions need no deep computation at all. Substituting the shipped
field's true potential for a refined candidate, which bounds what any
continuation could buy, the mean candidate-deepenings per decision:

| strategy | deepenings |
|---|---:|
| every frontier vertex | 2.212 |
| highest degree first | 1.104 |
| random | 1.664 |
| closest to boundary | 1.752 |

An edge whose two endpoints were both deepened is exact, but an edge with one
deepened and one scout endpoint keeps the scout's error on the unrefined side,
and the simulation re-certified those mixed edges with the full 0.0487
scout-scout radius. That is the conservative choice: a calibrated radius for
mixed edges could only certify more of them and lower the 1.104 further. The
frontier is sparse, and degree-first is the ordering that exploits it.



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


