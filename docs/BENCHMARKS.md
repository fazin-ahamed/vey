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


