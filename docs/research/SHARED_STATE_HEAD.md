# Shared-state head: preregistered protocol

Written before any measurement. Results are appended below the line when they exist.

## Question

d1-omni-600M builds one sequence per question (`<bos> state <q> question <opt> ...`)
and runs the whole 16-layer bidirectional trunk over it, so a request with K questions
costs K full passes over the state. Measured with the fp32 model on 2 vCPU: 265, 849
and 2,549 ms for 1, 3 and 6 questions over a 73-token state, almost linear in K.

Can the state be encoded once, and each question read against it, without losing
accuracy? The cheapest version of the idea reuses everything d1 already has:

    state    : trunk over  <bos> <state> state-tokens          once per request   -> H_s
    question : trunk over  <q> instruction <opt> <mask> ...    once per question  -> H_q
    head     : d1's own two decision-head layers over [H_s ; H_q] (+ a learned
               segment embedding), scored at the option markers as before

The only trained parameters are d1's head (26.2M, initialized from d1) and two
segment vectors. The trunk is frozen and runs as the deployed int8 model, so the
head is trained on exactly the features it will see in service. Per extra question the
cost falls from a trunk pass over `L_state + L_question` tokens to a trunk pass over
`L_question` tokens plus two head layers over `L_state + L_question`.

## Training

Teacher: the unmodified joint d1 (int8), whose probabilities the student must match.
Loss: KL(teacher || student) on the options' probabilities, using d1's own
per-question-type temperatures, plus 0.3 x cross entropy against the gold label.

Tasks and splits (train splits for fitting, test or validation splits for evaluation):

| Task | Options | Used for training | Used for evaluation |
|---|---:|---|---|
| SST-2 sentiment, as a choice | 2 | yes | held-out rows |
| AG News topic | 4 | yes | held-out rows |
| Yes/no on SST-2 and AG News statements | 2 | yes | held-out rows |
| SST-5 as a score | 5 | yes | held-out rows |
| Emotion | 6 | **no** | held-out rows: the unseen task |

Rows per task: 600 train, 200 evaluation, fixed seeds. This is small on purpose: the
trunk features are the expensive part on a CPU, and the claim being tested is that the
head can be re-pointed cheaply, not that it needs a large corpus.

## Gates

The student ships as an option in the server only if all four hold. Any miss goes to
`NEGATIVE_RESULTS.md` and nothing is wired in.

1. **Agreement:** on held-out rows of the trained tasks, top-1 agreement with the
   joint teacher is at least 0.95 and accuracy against gold is within 2.0 points of the
   teacher's.
2. **Unseen task:** on Emotion, accuracy is within 3.0 points of the teacher's.
3. **Speed:** with six questions over a state of about 100 tokens, latency is at least
   1.8x lower than the joint model on the same host, and one question costs no more than
   1.3x the joint model.
4. **Calibration:** expected calibration error on held-out rows is no worse than the
   teacher's plus 0.03.

Latency is measured with the int8 model, 2 threads, warm, median of 9 calls, on the
host stated with the results.

## Variant B: a prefix-cached trunk (added after run 1, before it was measured)

Run 1 of the design above (below) was under-trained and the head alone has only two layers
to connect state and question, where the joint model connects them at all sixteen. Variant B
keeps every trunk layer in the loop and adds no trunk parameters. Head adaptation
uses the same head and two segment vectors as Variant A:

    state    : trunk over <bos> <state> state-tokens, recording at every attention layer the
               state's keys and values (after normalization and RoPE), and at every
               convolution layer the last state token's gated input   -> cache, H_s
    question : the question tokens only, through the same layers; each attention layer reads
               the cached state keys and values plus the question's own, with RoPE positions
               continuing from the state; each convolution layer reads the cached last state
               token as its left neighbor                              -> H_q
    head     : d1's unchanged head over [H_s ; H_q]

The change from the joint model is that state tokens no longer see the question. d1 was trained
with full bidirectional attention, so this is a distribution shift, and the claim is only that
it is a smaller one than replacing the cross-reading with a re-trained head. It is first
measured with d1's own head, untrained; the same four gates apply, with the same tasks, rows
and teacher.

## Results (2026-11-07)

**Do not enable this head in the server.** Variant B improves multi-question speed,
but the trained head fails both the score-agreement and unseen-task accuracy gates.
The experiment supports further investigation, not a faster drop-in replacement.

Environment: Linux x86-64, 2 vCPU Intel Xeon at 2.6 GHz, AVX-512 VNNI, no GPU,
Python 3.11, PyTorch 2.15.0+cpu, transformers 5.22.1, two Torch threads. We reuse
d1-omni-600M revision `02b55d7076f15129e59ab3f94783f32c4b088674`: its frozen
354.5M text trunk and a 26.2M full-precision trainable head, not a newly trained
foundation model. The shared-state experiment is text-only. The ordinary T2 lane
also loads the separate 94.2M vision tower, but that is not a shared multimodal cache.
No video or audio support is implemented here.

### Variant A, exploratory run

Three epochs at learning rate 2e-5, batch 16, 600 rows per trained task.
Checkpoint agreement rose from 0.469 to 0.515 to 0.642. Student gold accuracy
was 0.695 / 0.460 / 0.855 / 0.290 / 0.295 on SST-2 / AG News / noul / SST-5 /
Emotion, against teacher 0.950 / 0.875 / 0.965 / 0.435 / 0.810. This run used
evaluation rows for checkpoint selection, so these are exploratory, optimistically
selected results, not an independent test. A higher-learning-rate retry did not
complete and is not reported as a finished experiment.

### Variant B, head adaptation

Before this run, checkpoint selection was moved to a separate validation subset:
the last 100 of each task's 600 training rows. Training uses the first 500, for
2,000 training and 400 validation rows total. The 200 evaluation rows per task
are untouched by this run's optimization and checkpoint selection. Seeds: 7 for
training-row sampling, 11 for evaluation-row sampling, 0 for initialization and
shuffle. Earlier exploratory evaluations have already exposed this task suite;
it is not a fresh, blinded benchmark.

Three epochs, batch 8, learning rate 2e-5, AdamW with weight decay 0.01. The
best validation agreement was 0.943 at epoch 1 (epochs 2 and 3: 0.935, 0.938).
Each epoch took 195 / 196 / 221 seconds; this is training time, not inference latency.
The raw-feature cache is fp16, the head is trained and evaluated in fp32, and the
trunk remains frozen int8. The saved checkpoint carries the original license and
a modification notice; neither weights nor downloaded data are committed.

All accuracy/agreement columns here use the option argmax, including SST-5.
The server emits an expected score instead, so do not compare these SST-5 numbers
directly with rounded-score accuracy in the System One report.

| Task (200 rows each) | Joint teacher accuracy | Cached, initial head | Cached, trained head | Trained agreement | Teacher ECE | Trained ECE |
|---|---:|---:|---:|---:|---:|---:|
| SST-2 | 0.950 | 0.930 | 0.930 | 0.950 | 0.020 | 0.030 |
| AG News | 0.875 | 0.860 | 0.860 | 0.965 | 0.067 | 0.089 |
| noul | 0.965 | 0.940 | 0.950 | 0.975 | 0.027 | 0.015 |
| SST-5 score | 0.435 | 0.375 | 0.415 | **0.780** | 0.123 | 0.123 |
| Emotion, held out from head training | 0.810 | 0.745 | **0.745** | 0.865 | 0.104 | 0.112 |

Gate 1 fails on SST-5 agreement. Gate 2 fails on Emotion's 6.5-point loss.
Gate 4 passes. Two hundred examples per task and a single training seed do not
establish generality. The result does not show that more head training cannot help;
it shows that this cheap head-only adaptation is not yet a safe replacement.
"Unseen" here means absent from this head adaptation, not absent from the
upstream model's pretraining; that provenance has not been established.

### Latency

Variant B with the trained fp32 head, actual int8 trunk execution, warm median of
9 calls, two threads, no concurrent training, server or browser. Both paths include
request-end heap trimming. Joint execution uses the normal 256-token batching
budget; cached execution batches the question suffixes and head inputs. The cache
is rebuilt per request, not reused across benchmark iterations.

| State tokens | Questions | Joint ms | Cached ms | Speedup |
|---:|---:|---:|---:|---:|
| 40 | 1 | 189 | 231 | 0.82x |
| 40 | 3 | 416 | 362 | 1.15x |
| 40 | 6 | 843 | 587 | 1.44x |
| 79 | 1 | 255 | 318 | 0.80x |
| 79 | 3 | 661 | 415 | 1.59x |
| 79 | 6 | 1,191 | 641 | 1.86x |
| 391 | 1 | 672 | 794 | 0.85x |
| 391 | 3 | 1,918 | 1,133 | 1.69x |
| 391 | 6 | 3,641 | 1,709 | 2.13x |

Gate 3 passes at the near-100-token fixture (79 tokenizer tokens): 1.86x for six
questions, and 1.25x single-question cost. Single questions are slower at all lengths;
even six questions miss 1.8x on the shortest fixture. Earlier contention-affected
timings and a two-repeat exploratory run are discarded. No 1 GiB memory claim is
made for this prototype or its full-precision head.

### Reproduce

Install `vey[d1]` and `datasets`, then run the local conversion first. These commands
read the same pinned original weights to initialize the trainable head. They need
several GB of RAM for preparation and training; there is no 1 GiB training claim.

```bash
vey-d1-prepare
python scripts/shared_state_head.py extract --out features-a --train-rows 600 --eval-rows 200
python scripts/shared_state_head.py extract-prefix --src features-a --out features-b
python scripts/shared_state_head.py prefix-eval --feat features-a --head init
python scripts/shared_state_head.py train --feat features-b --out features-b/head.pt --epochs 3 --batch 8 --lr 2e-5
python scripts/shared_state_head.py eval --feat features-b --head features-b/head.pt
python scripts/shared_state_head.py speed --head features-b/head.pt --prefix --reps 9
```

Use a fresh output directory for a new feature configuration: extraction resumes by
keeping existing task files. Only load feature/checkpoint files you created yourself;
they are local Torch serialization, not a format for accepting untrusted uploads.
