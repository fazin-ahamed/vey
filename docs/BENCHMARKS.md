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


