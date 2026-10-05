# Rubric-anchor grade interface: design and preregistration

Evidence class: HYPOTHESIS. Status: prospective design only. No fit, no capture, no new data access, no final-pool access. Companion machine-readable file: `research/endgame/ephemeral_pages_grade_interface_preregistration.json`.

## What the frozen ECA-2 result authorizes

The corrected ECA-2 final (320 fresh worlds, 31,360 DecisionIRs) failed the semantic gates for all five controls. The two numbers that matter for this design:

- cross: `page_attribution` 0.883203125 passes its 0.85 screen, `ordinal_MAE_max` 0.35999951392847995 fails its 0.10 screen.
- pages: `page_attribution` 0.24775390625 fails 0.85, `ordinal_MAE_max` 0.36104662390189335 fails 0.10, `atomic_choice_macro` 0.085546875.

The parent audit decision tree names one branch for this state (`conditional_and_cross_inadequate`): retire only the demonstrated inadequate frozen readouts, not semantic information in all encoder states, and preregister one targeted representation change on fresh material. It offers two mechanisms: minimal final-layer task adaptation if semantic match or polarity is limiting, or explicit caller-supplied ordinal rubric grounding if grade identifiability is limiting. This design takes the second.

The corrected development component audit sharpens the choice. On the pages reader, oracle orientation alone reaches atomic macro 0.114258, oracle raw extent 0.104492, oracle knownness 0.260742, oracle raw extent plus orientation plus knownness 1.0. Extent, orientation and knownness are each individually binding. Oracle relevance alone collapses coverage to 0.0 and is actively harmful without knownness. Page attribution is fixed at 0.260498 in every non-relevance cell. So the failure is not one component: relevance, extent, orientation and knownness all carry part of it.

## The design

The caller's own ordered per-property rubric becomes an explicit model-visible anchor text. One extra frozen-encoder head predicts raw extent from the train-normalized 384-d page feature concatenated with a train-normalized 384-d rubric-anchor feature.

Concretely, on top of the unchanged frozen `PageReader`:

```
raw_value_j = sigmoid( w_v . htilde_page_j + w_r . htilde_anchor + b_v )
```

`htilde_page_j` is the page block under the unchanged checkpoint train-only normalizer. `htilde_anchor` is the anchor block under a new train-only normalizer fit on the distinct train anchors only. `w_r` is one new bias-free 384-to-1 map, 384 new scalars. This is numerically identical to concatenating the two blocks and applying a bias-bearing 768-to-1 linear map; the additive form is normative and a frozen smoke check must reproduce the concatenated form to 1e-6.

Relevance, direction, attention and knownness are unchanged. The rank-64 bias-free projections, the softmax over the frozen page mask, the `tanh` direction, and the three-scalar knownness expression are byte-identical in form. No hidden layer, no activation, no second layer, no new loss term, no reweighting. The change touches the value path only.

The anchor itself: for each active property, concatenate its five `grade_rubrics` strings in `grade_order`, with no ordinal markers, no stage labels, no headings, and no separators that encode stage boundaries. One anchor per property, identical for every page and every question of that property. Because the anchor is property-level, its bytes cannot depend on a page or its authored grade; only the ordered scale content is supplied. The anchor is encoded as a single text through the same pinned frozen encoder with the same FP32 final-layer masked mean, width 384, its own cache key, token limit 128, truncate disabled.

Today no rubric text enters any ECA feature, target, cache key or reader input. The only model-visible strings are `term.question` and `page.text`, and `field_key` is used only for target construction, never encoded. The anchor is genuinely new model-visible text.

## Why this is expected to be weak, and why it must be falsifiable

The finite linear ceiling measured on the corrected DEV audit: bias-free FP64 least squares on the cached 384-d page feature alone, fit on 180 unique supervised train page texts, reaches train MSE 1.816885422626678e-29 at rank 180 of 384 with condition 221.80657657811963 and 204 unidentified coefficient directions, yet generalizes to a DEV unique-text family macro MAE of 0.3832298906765268, with unclipped predictions spanning [-1.0950165169850952, 1.4707872781000932] against a [0,1] target. The cached page feature does not linearly identify extent even when fit exactly on every unique supervised text.

That result bears directly on this design. If the anchor is itself a frozen final-layer mean, and the new map is linear in both blocks, then nothing about the construction guarantees generalization improves. The design's only live mechanism is that joint optimization of the value path against a genuinely different text lets the encoder represent scale information for the page that the page text alone does not carry. That is a real possibility, not a given.

So the preregistration makes the falsifier cheap and reachable before any GPU spend:

- S0 custody. Mechanical checks only: distinct anchors equal active properties, anchors non-degenerate and mutually distinct, no digits, no stage headings, no grade labels, no overlap with any question or page text of the property, and every anchor within the pinned 128-token limit with truncation disabled. An over-length anchor fails the run closed; no truncation and no per-property exclusion, because excluding a property would change the DEV population the gates screen.
- S1 augmented linear ceiling. CPU-only, FP64, zero optimizer steps. Bias-free least squares on unique supervised train page texts in the augmented space [normalized page feature; normalized anchor feature], same solver policy as the measured page-only ceiling. Screen: DEV unique-text family macro MAE at most 0.3332298906765268, the page-only value 0.3832298906765268 minus a predeclared absolute margin of 0.05. If S1 fails, extent identifiability is not the binding constraint at the frozen-representation level, the design is retired without a neural fit, and minimal final-layer encoder adaptation opens under a separate protocol.
- S2 neural DEV run. Only if S0 and S1 pass. One new arm under the frozen recipe, the five existing controls untouched, calibration on the original corrected calibration split only, evaluation on development only.

## Property-definition audit (blocking prerequisite)

Authored rubric texts are private property data. Before any anchor is encoded, two independent opaque reviewers each cover every anchor of the authorized phase and must accept all of them with zero ambiguous and zero rejected items. Reviewers never receive model outputs, weights, DEV performance, targets, family or split labels. The audit must establish:

- (a) fixed before capture: each anchor exists byte-identically in the authored source; the source file sha256 and every per-anchor sha256 are recorded and committed before any encoder call or optimizer step; no anchor may be edited after DEV outcomes are observed.
- (b) no numeric or literal grade tokens: zero Unicode digits, zero `grade_order` labels, zero stage headings, plus reviewer attestation.
- (c) does not reveal the answer for any specific question: the anchor is property-level and identical across all pages and questions; no anchor is a substring of, or shares normalized token Jaccard at least 0.60 with, any question or page text of the same property; reviewer attestation is primary.
- (d) not a paraphrase of the question text: reviewer attestation is primary; the same lexical screen is a necessary mechanical check.

Raw judgments and disagreements are preserved verbatim. Any ambiguity blocks the run. If a condition fails, retain the judgments, do not encode anchors, and treat the design as unrunnable; do not substitute re-authored text.

## Inherited frozen constants

Everything below is inherited byte-for-byte from the parent protocol and is not re-derived here.

- Encoder `cross-encoder/nli-deberta-v3-xsmall`, revision `a150876415327c80daeff35ca6f68f5ed8cf5c24`, weight sha256 `4e4fc4977f8d29d2a164255c8f69b9d6c158deeb309bb5e70445b94666ccd9e9`, frozen, parameters must hash identically before and after.
- Architecture: width 384, rank 64, cross hidden 128.
- Recipe: seed 7, 400 epochs, AdamW lr 0.01, weight decay 0.0001, full cached training set in deterministic chunks, train-only per-coordinate mean/std floor 0.01, earliest minimum held-world validation checkpoint, K4 main/missing/contradiction optimizer membership, unchanged equal-weight four-component loss.
- Calibration: original corrected calibration split, frozen knownness grid 0.05 to 0.95 step 0.05 and frozen sigma grid [0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4]. No DEV tuning.
- Gates: `atomic_choice_macro` 0.80, `held4_family_choice_macro` 0.80, `ordinal_MAE_max` 0.10, `orientation` 0.90, `page_attribution` 0.85, `UNKNOWN_precision` 0.90, `UNKNOWN_recall` 0.90, `supported_coverage` 0.90, `composition` 0.80, `teacher_changing_correct_new` 0.80, `exact_literals` 1.0 (parent-owned, not re-measured), `permutation_rename_reorder` 1.0, `internal_cross_NI_margin` 0.05.
- Statistics: world-cluster bootstrap, 10000 draws, seed 0, whole world as unit, variants clustered, equal family weights, fixed authored inventory.
- Exact compiler, integer facts, exact composition arithmetic and UNKNOWN propagation unchanged; no numeric path enters any encoder.

## Mechanism screen

A predeclared claim set, fixed now and never chosen from outcomes: new arm minus `rubric_blind` (w_r set to zero at evaluation time), minus `zero_pages`, minus `zero_question`, minus `lexical`. One-sided Bonferroni, familywise alpha 0.05 over exactly these four claims, each at 0.05/4, on the same world-cluster bootstrap. All four lower bounds must exceed 0.

The anchor-specific claim is new arm minus `rubric_blind`. It is the falsifier for the design premise. If it does not pass, the observed change is not attributable to the anchor regardless of absolute gate values. This is a development mechanism screen, not the parent promoted mechanism gate, and it earns no final or promotion credit.

## Decision rule and failure branches

Evaluate S0, then S1, then S2. Never skip ahead.

1. S0 fails: design unrunnable; retain audit evidence; stop.
2. S0 passes, S1 fails: extent identifiability is not the binding constraint. Retire rubric grounding, do not fit, do not widen capacity. Open minimal final-layer encoder adaptation under a separate preregistered protocol.
3. S0 and S1 pass, S2 fails any absolute gate: negative evidence for this head route. Record it and do not widen capacity or add layers.
4. S0, S1 and S2 pass every absolute gate and the anchor-specific mechanism claim passes: DEV success. Require replication on seeds 7, 11 and 13 with fresh independently audited wording before any promotion. The untouched ECA-2 final stays closed; promotion additionally requires a separate final protocol.
5. S0 and S1 pass, S2 passes every absolute gate but the anchor-specific claim fails: the premise is falsified even though absolute metrics passed. Retire rubric grounding; do not promote.
6. S0 and S1 pass, S2 improves extent specifically (anchor-specific claim passes) but fails other absolute gates: branch 2 is supported. Extent identifiability improved while the full task still failed. Retire rubric grounding and open minimal final-layer encoder adaptation under a separate protocol.

No arm, threshold, seed, epoch, prompt, pooling or capacity selection may come from DEV outcomes. No B-STEF. No endgame completion claim. A finite DEV negative does not establish absence of semantic information in all encoder states.

## Custody and scope

The ECA-2 final IR, features and labels stay closed; no final outcome selects anything. The old ECA-1 final stays closed for all correction selection and fitting. Every ECA-1 and ECA-2 corrected artifact, receipt and raw result is preserved. The two new files are hashed by the future launch receipt at execution time and must byte-match the committed preregistration before any encoder call.

Scope: authored English mechanism correctness assay on the frozen ECA-2 prefinal inventory, development only. No final-pool access, no promotion, no B-STEF, no many-axis endgame completion, no native Boolean/intent/compliance, no multilingual, no neutral benchmark, no certificate, no speed, no shipping clearance and no competitor superiority credit.

Budget: zero encoder fine-tuning, zero new encoder downloads, one frozen forward per distinct active anchor, S1 CPU-only FP64, S2 one GPU job under the inherited resource guards.

## Measured, inferred, unknown

Measured this turn from disk: the corrected ECA-2 final gate screens and counts, the corrected DEV factorial cells, the finite linear page-feature ceiling diagnostics, the architecture and recipe constants, the encoder revision and weight hash, the feature and corpus byte hashes, and the sealed selection receipt hash.

Inferred: supplying the ordered rubric scale may restore extent identifiability, because the joint reader already recovers page relevance while extent fails. The inference is weak: the linear ceiling shows the frozen final-layer page mean does not linearly identify extent even when fit exactly, so an anchor that is itself a frozen final-layer mean is unlikely to help unless it changes what the encoder represents for the page. The design is expected to fail S1 unless the anchor adds genuine scale information.

Unknown: whether the frozen encoder's rubric-text mean carries property-scale information at all; whether any active anchor exceeds the 128-token limit; whether the prefinal rubrics satisfy audit conditions (a) to (d), since no rubric text was opened here; whether extent is the binding constraint in the deployed reader rather than orientation or knownness, which are also individually binding; and whether the augmented linear ceiling can beat the page-only ceiling on held-out DEV texts.
