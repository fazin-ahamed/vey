# Vey 2.0.0-rc2 release notes

**The public runtime now has three lanes, and the fast one is the default for
the axes it was measured on.**

## What changed since rc1

- **Field lane.** `vey.decide()` routes a program to an O(K) scalar field when
  every stage is a MAX on a validated axis: `food progress`, `open space`,
  `headroom`. One forward pass per candidate. Anything with a filter, a MIN, or
  an axis outside that list stays on the pairwise comparator. Structured facts
  still never load a model.
- **Snake uses `vey.decide()`.** The demo no longer grounds through the
  comparator itself. Recorded frames carry the lane that actually ran.
- **Pinned artifacts.** Field weights are `field.safetensors` on
  `fazinahamed/vey` at `3eb1460a82c98fc99c350032b9cf29a74075a4a6`. The field
  base model `microsoft/deberta-v3-xsmall` is pinned at
  `4b419818330868dff6a60ad3e6b1c730f8b8c0c6`. The CRUX predicate grounder
  `tasksource/ModernBERT-base-nli` is pinned at
  `de4ab7e77845098b7fab7f6ab9d370ddff27b19c`.
- **Held-out agreement with the pairwise teacher**, on decisions the field did
  not train on: food progress 0.958, open space 0.961, headroom 0.956.
- **License.** AGPL-3.0-only from commit `e52bcf2` onward (historical; the tree moved to FSL-1.1-MIT on 2026-09-30). The `v1.0.0` and
  `v2.0.0-rc1` tags were published under Apache-2.0, and anyone who obtained
  those snapshots keeps that grant. See `NOTICE` for the MIT and Apache
  notices on the upstream encoders, and `COMMERCIAL-LICENSE.md` for proprietary
  terms.
- **Routing tests.** Hermetic coverage that a validated all-MAX program takes
  the field, a filter or an unknown axis stays on CRUX, and the field path does
  not construct the comparator.

## What this release does not claim

The field does not cover every qualitative decision. Unvalidated axes and any
program with a predicate filter still use the pairwise comparator. The 23 ms
research figure was a single-axis GPU measurement; the Snake demo runs the
field on CPU and grounds two axes per move, so its on-screen inference time is
higher.
