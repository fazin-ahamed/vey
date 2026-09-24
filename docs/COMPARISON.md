# Comparison: Vey vs Laya vs Jev

This compares **interfaces and measured evidence**, not marketing labels, and
never collapses the three systems into a single "winner".

The three overlap in the broad idea of fast typed decisions, but their shipping
surfaces differ:

- **Vey v1**: local, frozen 24M encoder, exact structured lane, deterministic
  action compiler, calibrated trust. English.
- **Laya / Laya-MLX** (Convai Innovations, Apache-2.0): local open-weight typed
  decisions, 322M–421M encoder stack.
- **Jev** (TypeSafe AI): hosted proprietary typed-decision API.

There are two comparison modes and they are never merged:

1. **Strict parity**: identical rows for every system, same host, CPU, batch=1.
2. **Native characterization**: each system on its intended API and hardware,
   with hardware and runtime stated on every row.

## Frozen targets (verified 2026-09-24)

| system | pin |
|---|---|
| Laya package | `laya` 0.3.20 |
| Laya English | `convaiinnovations/laya` @ `55cf4c4e` |
| Jev | `jev-1.13.0` via TypeSafe `/v1/systemone` |

Laya's weights run locally on CPU through its PyTorch runtime; the MLX port is a
separate Apple-Silicon project and is not used for the same-host comparison.

## Track A: strict parity (same host, identical rows)

Intel i5-12400F, 6 threads, CPU, batch=1, 300 rows/task, same question and
candidate set. Vey end-to-end (query encode + cached-candidate scoring) vs Laya
full forward.

| task (options) | Vey acc | Laya acc | Vey p50 | Laya p50 | Vey weights | Laya weights |
|---|---:|---:|---:|---:|---:|---:|
| Banking77 (77) | **0.658** | 0.431 | **2.863 ms** | 478.969 ms | 92 MB | 1607 MB |
| CLINC150 (151) | **0.675** | 0.593 | **2.760 ms** | 1014.777 ms | 92 MB | 1607 MB |
| MASSIVE (60) | **0.471** | 0.403 | **2.598 ms** | 378.022 ms | 92 MB | 1607 MB |

Vey is more accurate on all three, **145–368× faster end-to-end**, with a
**17.5× smaller** weight payload. Laya's cost scales with the candidate count
(it renders every option into the encoder sequence); Vey's is nearly flat.

**Where Laya wins:** it returns a calibrated probability distribution; Vey's
matcher returns similarity scores. And Vey's own pooled head (0.884 / 0.832 /
0.780) is more accurate than the matcher on trained labels. Neither is uniformly
better.

**Jev strict parity: not measured.** No direct TypeSafe credentials were
available and no substitute route was used; Jev's column falls back to the
public benchmark for quality and live latency is reported as not measured.

## Track B: native characterization (own hardware, never one ranking)

| system | latency | quality evidence kind | size |
|---|---|---|---|
| Vey (this host) | 2.6–2.9 ms p50 e2e | gold-labeled accuracy | 92 MB FP32 |
| Laya (this host) | 378–1015 ms p50 | 0.766 = *teacher-agreement*, not gold accuracy | 1607 MB FP32 |
| Laya (MLX, M3 Max) | 10.9–17.8 ms p50 | same | 614–944 MiB FP16 |
| Laya (T4, upstream) | 39.5 ms one-call | base 0.362, below majority | n/a |
| Jev (public bench) | 0.65 s p50 via API | reference *agreement* | remote |

The Laya rows span different hardware and stay separate. Laya's 0.766 and Jev's
first-party numbers are *agreement with a reference distribution*, not
gold-labeled accuracy, and are never placed beside Vey's numbers.

## Capability surface

| capability | Vey | Laya | Jev |
|---|---|---|---|
| local CPU, same host | yes | yes | API only |
| typed choice decisions | yes | yes | yes |
| calibrated probabilities | no (similarity) | yes | yes |
| dynamic / novel candidates | yes (frozen matcher) | call-time options | call-time criteria |
| **UNKNOWN / abstention** | **explicit, validated** | not equivalent (argmax; its act-probability head is documented anti-correlated with correctness) | inspect API |
| exact fact lane | yes | no equivalent | no equivalent |
| lexical / BM25 lane | yes | no equivalent | no equivalent |
| tool compilation | **deterministic, 0.000 unsupported args** | app-side | app-side |

Neither Laya nor Jev calls a tool or generates arguments; both delegate execution
to application code, the same boundary Vey's compiler sits behind. Vey's
deterministic compilation is compared as a **native Vey capability**.

## What Vey can claim

- On identical rows and the same CPU, more accurate than upstream Laya on three
  closed-set tasks, 145–368× faster end-to-end, 17.5× smaller weights.
- Abstention and exact validation are capabilities Laya and Jev do not expose.

## What Vey does not claim

- Overall superiority: Laya returns calibrated probabilities Vey does not, and
  Vey's own pooled lane beats the matcher on trained labels.
- Any cross-hardware latency ranking.
- Any abstention head-to-head, because the others have no equivalent.
