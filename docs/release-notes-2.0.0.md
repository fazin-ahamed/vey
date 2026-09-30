# Vey 2.0.0 release notes

Vey 2.0.0 is the first stable release on the clean public lineage.

## What it is

A lightweight machine-native decision runtime for fast, typed decisions
without autoregressive generation. Decisions are compiled programs
(FILTER / MAX / MIN stages with epsilon resolution) executed over a frozen
24M-parameter semantic field: deterministic, permutation-exact, with
calibrated trust, abstention, and a certificate on every explained decision.

## Evidence in this release

- The semantic field's tight-band accuracy around the executor boundary,
  improved by Split-Train Exact-Fold (STEF) training: +0.039 tight-band
  agreement with a bootstrap CI entirely above zero, replicating across
  open-space transfer and a shared three-axis field.
- Real executor traces: full trace agreement 0.8359 to 0.8744 on 1,950
  untouched decisions, repairing 52.5 percent of baseline trace failures at
  a 5.7 percent regression rate.
- Deployment: head cost about 2 microseconds and 772 parameters per axis;
  full-field latency non-inferior; calibration radius halved.
- Every number in docs/BENCHMARKS.md is reproducible from committed scripts
  with the environment stated.

## Licensing

FSL-1.1-MIT. Earlier snapshots distributed under Apache-2.0 and AGPL-3.0
keep those grants; see docs/LICENSE_HISTORY.md.

## Earlier release candidates

The release-candidate notes for the pre-restructure lineage are kept in
docs/release-notes-2.0.0-rc1.md and docs/release-notes-2.0.0-rc2.md, and
the full experimental history is on the research-history branch.
