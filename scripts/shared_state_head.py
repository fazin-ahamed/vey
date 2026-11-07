"""Shared-state head experiment (protocol and gates: docs/research/SHARED_STATE_HEAD.md).

Encode a request's state once and read each question against it, by re-pointing d1's own
decision head at [state features ; question features].

    pip install datasets 'vey[d1]'
    vey-d1-prepare                                   # the int8 model, converted locally
    python scripts/shared_state_head.py extract --out /tmp/ss --train-rows 600 --eval-rows 200
    python scripts/shared_state_head.py train   --feat /tmp/ss --out /tmp/ss/head.pt
    python scripts/shared_state_head.py eval    --feat /tmp/ss --head /tmp/ss/head.pt
    python scripts/shared_state_head.py speed   --head /tmp/ss/head.pt
    python scripts/shared_state_head.py prefix-eval --feat /tmp/ss --head init          # variant B
    python scripts/shared_state_head.py speed --head /tmp/ssb/head.pt --prefix

d1-omni-600M's weights are under the LFM Open License v1.0 (see src/vey/systemone/d1.py).
The head trained here is a Derivative Work of those weights and is not for redistribution.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

TASKS_TRAIN = ("sst2", "ag_news", "noul", "sst5")
TASKS_EVAL = TASKS_TRAIN + ("emotion",)  # held out from this head adaptation, not necessarily upstream pretraining
AG_OPTS = {"world": "international news and politics", "sports": "sports news",
           "business": "business and finance news", "science and technology": "science and technology news"}
EMO_OPTS = {"sadness": "a sad message", "joy": "a joyful message", "love": "a loving message",
            "anger": "an angry message", "fear": "a fearful message", "surprise": "a surprised message"}
SST5_LEVELS = ["very negative review", "negative review", "neutral review", "positive review", "very positive review"]


# ------------------------------------------------------------------- data

def rows_for(task: str, split: str, n: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    rnd = random.Random(seed)
    out = []
    if task == "sst2":
        d = load_dataset("stanfordnlp/sst2", split="train" if split == "train" else "validation")
        q = {"type": "choice", "instructions": "What is the sentiment of this movie review?",
             "criteria": {"negative": "a negative movie review", "positive": "a positive movie review"}}
        for i in rnd.sample(range(len(d)), n):
            out.append({"state": d[i]["sentence"], "question": q, "label": d[i]["label"], "task": task})
    elif task == "ag_news":
        d = load_dataset("fancyzhx/ag_news", split=split)
        q = {"type": "choice", "instructions": "What is the topic of this news article?", "criteria": AG_OPTS}
        for i in rnd.sample(range(len(d)), n):
            out.append({"state": d[i]["text"], "question": q, "label": d[i]["label"], "task": task})
    elif task == "emotion":
        d = load_dataset("dair-ai/emotion", "split", split=split)
        q = {"type": "choice", "instructions": "What emotion does this message express?", "criteria": EMO_OPTS}
        for i in rnd.sample(range(len(d)), n):
            out.append({"state": d[i]["text"], "question": q, "label": d[i]["label"], "task": task})
    elif task == "sst5":
        d = load_dataset("SetFit/sst5", split="train" if split == "train" else "test")
        q = {"type": "score", "instructions": "How positive is this movie review?", "criteria": SST5_LEVELS}
        for i in rnd.sample(range(len(d)), n):
            out.append({"state": d[i]["text"], "question": q, "label": d[i]["label"], "task": task})
    elif task == "noul":
        s = load_dataset("stanfordnlp/sst2", split="train" if split == "train" else "validation")
        a = load_dataset("fancyzhx/ag_news", split=split)
        for i in rnd.sample(range(len(s)), n // 2):
            out.append({"state": s[i]["sentence"], "label": s[i]["label"], "task": task,
                        "question": {"type": "noul", "instructions": "Is the sentiment of this movie review positive?"}})
        for i in rnd.sample(range(len(a)), n - n // 2):
            out.append({"state": a[i]["text"], "label": int(a[i]["label"] == 1), "task": task,
                        "question": {"type": "noul", "instructions": "Is this news article about sports?"}})
    else:
        raise ValueError(task)
    return out


# ---------------------------------------------------------------- d1 pieces

def load_d1(threads: int):
    from vey.systemone.d1 import D1Model

    m = D1Model(threads=threads)
    prompt = sys.modules["vey_d1_code.prompt"]
    enc_mod = sys.modules["vey_d1_code.encoder"]
    return m, prompt, enc_mod


def split_row(m, prompt, row):
    """The joint ids of a row split at the question delimiter, with markers relative to each part."""
    q = prompt.as_question(row["question"])
    tok = m.model.tokenizer
    ids, markers = prompt.encode(tok, row["state"], q, m.model.config.max_length, None, False)
    cut = ids.index(tok.convert_tokens_to_ids(prompt.DELIM["q"]))
    return q, ids, markers, cut


def trunk(m, ids_list):
    """Frozen trunk over one or more id sequences (padded together); one tensor per sequence."""
    import torch

    enc = m.model.encoder
    seqs = [enc.embed_tokens(torch.tensor(i)) for i in ids_list]
    h = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    lens = torch.tensor([len(s) for s in seqs])
    pad = torch.arange(h.shape[1])[None] < lens[:, None]
    out = enc(h, pad, torch.zeros(len(seqs), dtype=torch.long))
    return [out[i, :n] for i, n in enumerate(lens.tolist())]


def temperature_for(m, prompt, q) -> float:
    t = m.model.config.temperatures
    return float(t.get(prompt.temperature_key(q), t.get(q.type, 1.0)))


# --------------------------------------------------------------- extract

def cmd_extract(a) -> None:
    import torch

    m, prompt, _ = load_d1(a.threads)
    os.makedirs(a.out, exist_ok=True)
    plan = [("train", t, a.train_rows) for t in TASKS_TRAIN] + [("eval", t, a.eval_rows) for t in TASKS_EVAL]
    for split, task, n in plan:
        path = os.path.join(a.out, f"{split}_{task}.pt")
        if os.path.exists(path):  # resumable: a finished task is kept
            print(f"kept {path}", flush=True)
            continue
        rows = rows_for(task, "train" if split == "train" else "test", n, seed=7 if split == "train" else 11)
        feats, t0 = [], time.time()
        for r in rows:
            q, ids, markers, cut = split_row(m, prompt, r)
            with torch.no_grad():
                hs, hq = trunk(m, [ids[:cut], ids[cut:]])
                (teacher,) = m.model._forward([(None, ids, markers, q, True)])
            p = list(reversed(teacher)) if q.type == "noul" else list(teacher)  # back to the model's option order
            feats.append({"hs": hs.half(), "hq": hq.half(), "markers": [x - cut for x in markers],
                          "qtype": prompt.QTYPES[q.type], "teacher": p, "label": r["label"], "task": task,
                          "temperature": temperature_for(m, prompt, q), "state_len": cut, "q_len": len(ids) - cut})
        torch.save(feats, path)
        print(f"extracted {split:5s} {task:8s} {len(feats)} rows in {time.time() - t0:.0f}s "
              f"(mean state {np.mean([f['state_len'] for f in feats]):.0f} tokens, "
              f"question {np.mean([f['q_len'] for f in feats]):.0f})", flush=True)


# ----------------------------------------------------------------- student

def build_student(m, enc_mod):
    import torch
    import torch.nn as nn
    from safetensors import safe_open

    d = m.model.config.text_config["hidden_size"]
    head = enc_mod.DecisionHead(d, m.model.config.head_layers)
    if os.environ.get("D1_FP32_DIR"):
        src = os.path.join(os.environ["D1_FP32_DIR"], "model.safetensors")
    else:
        from huggingface_hub import hf_hub_download
        from vey.systemone.d1 import D1_REPO, D1_REVISION

        src = hf_hub_download(D1_REPO, "model.safetensors", revision=D1_REVISION)
    with safe_open(src, "pt") as f:
        head.load_state_dict({k[len("head."):]: f.get_tensor(k).float() for k in f.keys() if k.startswith("head.")})

    class Shared(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = head
            self.seg = nn.Parameter(torch.zeros(2, d))  # state / question segment vectors, start at zero

        def forward(self, batch):
            hs = [b["hs"].float() + self.seg[0] for b in batch]
            hq = [b["hq"].float() + self.seg[1] for b in batch]
            seqs = [torch.cat([s, q]) for s, q in zip(hs, hq)]
            h = nn.utils.rnn.pad_sequence(seqs, batch_first=True)
            lens = torch.tensor([len(s) for s in seqs])
            pad = torch.arange(h.shape[1])[None] < lens[:, None]
            k = max(len(b["markers"]) for b in batch)
            pos = torch.zeros(len(batch), k, dtype=torch.long)
            mask = torch.zeros(len(batch), k, dtype=torch.bool)
            for i, b in enumerate(batch):
                n = len(b["markers"])
                pos[i, :n] = torch.tensor(b["markers"]) + len(hs[i])
                mask[i, :n] = True
            qtype = torch.tensor([b["qtype"] for b in batch])
            return self.head(h, pad, pos, mask, qtype)

    return Shared()


def student_probs(logits, batch):
    import torch

    out = []
    for z, b in zip(logits, batch):
        n = len(b["markers"])
        out.append(torch.softmax(z[:n] / b["temperature"], -1))
    return out


def load_feats(path, split, tasks):
    import torch

    rows = []
    for t in tasks:
        rows += torch.load(os.path.join(path, f"{split}_{t}.pt"))
    return rows


def cmd_train(a) -> None:
    import gc
    import torch
    from vey.systemone.d1 import _trim_heap

    torch.manual_seed(0)
    torch.set_num_threads(a.threads)
    m, prompt, enc_mod = load_d1(a.threads)
    model = build_student(m, enc_mod)
    license_text = (m.cache / "LICENSE").read_text()
    notice = (m.cache / "NOTICE.txt").read_text() + "\nModification: the decision head was fine-tuned on shared-state features.\n"
    del m
    gc.collect()  # the inference model's _run closure owns a bound method back to it
    _trim_heap()
    train, dev = [], []
    for task in TASKS_TRAIN:
        rows = load_feats(a.feat, "train", [task])
        cut = round(len(rows) * 5 / 6)
        train.extend(rows[:cut])
        dev.extend(rows[cut:])
    print(f"train {len(train)} rows, dev {len(dev)} rows from the train split; eval is untouched", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    rnd = random.Random(0)
    best = -1.0
    for epoch in range(a.epochs):
        model.train()
        rnd.shuffle(train)
        t0, losses = time.time(), []
        for i in range(0, len(train), a.batch):
            batch = train[i:i + a.batch]
            ps = student_probs(model(batch), batch)
            loss = 0.0
            for p, b in zip(ps, batch):
                t = torch.tensor(b["teacher"])
                loss = loss + torch.sum(t * (torch.log(t.clamp_min(1e-9)) - torch.log(p.clamp_min(1e-9)))) \
                    + 0.3 * (-torch.log(p[b["label"]].clamp_min(1e-9)))
            loss = loss / len(batch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss))
        model.eval()
        with torch.no_grad():
            agree, n = 0, 0
            for i in range(0, len(dev), 32):
                batch = dev[i:i + 32]
                for p, b in zip(student_probs(model(batch), batch), batch):
                    agree += int(int(p.argmax()) == int(np.argmax(b["teacher"])))
                    n += 1
        print(f"epoch {epoch + 1}: train loss {np.mean(losses):.4f}, dev agreement with teacher {agree / n:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if agree / n > best:
            best = agree / n
            torch.save(model.state_dict(), a.out)
            with open(a.out + ".LICENSE", "w") as f:
                f.write(license_text)
            with open(a.out + ".NOTICE", "w") as f:
                f.write(notice)
    print(f"saved best (dev agreement {best:.3f}) to {a.out}")


# -------------------------------------------------------------------- eval

def ece(conf, ok, bins=10):
    conf, ok = np.asarray(conf, float), np.asarray(ok, float)
    edges = np.linspace(0, 1, bins + 1)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            tot += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(tot)


def cmd_eval(a) -> None:
    import torch

    torch.set_num_threads(a.threads)
    m, prompt, enc_mod = load_d1(a.threads)
    model = build_student(m, enc_mod)
    del m
    if a.head != "init":  # "init" = d1's own head, unchanged, on the shared-state input
        model.load_state_dict(torch.load(a.head))
    model.eval()
    print(f"{'task':9s} {'rows':>5s} {'teacher acc':>12s} {'student acc':>12s} {'diff':>6s} {'agreement':>10s} {'ECE teacher':>12s} {'ECE student':>12s}")
    res = {}
    for task in TASKS_EVAL:
        rows = load_feats(a.feat, "eval", [task])
        s_pred, t_pred, gold, s_conf, t_conf = [], [], [], [], []
        with torch.no_grad():
            for i in range(0, len(rows), 32):
                batch = rows[i:i + 32]
                for p, b in zip(student_probs(model(batch), batch), batch):
                    t = np.asarray(b["teacher"])
                    s_pred.append(int(p.argmax())); t_pred.append(int(t.argmax())); gold.append(b["label"])
                    s_conf.append(float(p.max())); t_conf.append(float(t.max()))
        s_ok = [p == y for p, y in zip(s_pred, gold)]
        t_ok = [p == y for p, y in zip(t_pred, gold)]
        agree = float(np.mean([s == t for s, t in zip(s_pred, t_pred)]))
        res[task] = dict(teacher=float(np.mean(t_ok)), student=float(np.mean(s_ok)), agreement=agree,
                         ece_t=ece(t_conf, t_ok), ece_s=ece(s_conf, s_ok))
        r = res[task]
        print(f"{task:9s} {len(rows):5d} {r['teacher']:12.3f} {r['student']:12.3f} {100 * (r['student'] - r['teacher']):+6.1f} "
              f"{agree:10.3f} {r['ece_t']:12.3f} {r['ece_s']:12.3f}")
    trained = [res[t] for t in TASKS_TRAIN]
    g1 = min(r["agreement"] for r in trained) >= 0.95 and all(r["teacher"] - r["student"] <= 0.02 + 1e-12 for r in trained)
    g2 = res["emotion"]["teacher"] - res["emotion"]["student"] <= 0.03 + 1e-12
    g4 = all(r["ece_s"] <= r["ece_t"] + 0.03 for r in res.values())
    print(f"\ngate 1 (agreement >= 0.95 and within 2.0 points, trained tasks): {'PASS' if g1 else 'FAIL'}")
    print(f"gate 2 (emotion within 3.0 points of the teacher):               {'PASS' if g2 else 'FAIL'}")
    print(f"gate 4 (ECE no worse than teacher + 0.03):                        {'PASS' if g4 else 'FAIL'}")
    json.dump(res, open(os.path.join(a.feat, "eval.json"), "w"), indent=1)


# ------------------------------------------------------------------- speed

def cmd_speed(a) -> None:
    import torch
    from vey.systemone.d1 import _trim_heap

    torch.set_num_threads(a.threads)
    m, prompt, enc_mod = load_d1(a.threads)
    model = build_student(m, enc_mod)
    model.load_state_dict(torch.load(a.head))
    model.eval()
    base = ("Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan. "
            "The invoice number is 4411 and the card ends in 0042. ")
    questions = [{"type": "noul", "instructions": "Is the customer asking for a refund?"},
                 {"type": "choice", "instructions": "Which team should handle this?",
                  "criteria": {"billing": "Charges, refunds, invoices", "shipping": "Delivery and parcels", "technical": "App or site faults"}},
                 {"type": "score", "instructions": "How urgent is this ticket?", "criteria": ["low", "medium", "high"]},
                 {"type": "noul", "instructions": "Does the customer threaten to cancel?"},
                 {"type": "choice", "instructions": "What is the tone?", "criteria": {"angry": "upset", "calm": "neutral", "happy": "pleased"}},
                 {"type": "noul", "instructions": "Is a deadline mentioned?"}]

    def shared(state, qs):
        rows = [split_row(m, prompt, {"state": state, "question": q}) for q in qs]
        with torch.no_grad():
            cut = rows[0][3]
            if a.prefix:
                hs, cache = prefix_state(m, enc_mod, rows[0][1][:cut])
                hqs = prefix_questions(m, enc_mod, [r[1][r[3]:] for r in rows], cache, cut)
            else:
                hs = trunk(m, [rows[0][1][:cut]])[0]
                hqs = trunk(m, [r[1][r[3]:] for r in rows])
            batch = [{"hs": hs.half(), "hq": hq.half(), "markers": [x - r[3] for x in r[2]],
                      "qtype": prompt.QTYPES[r[0].type], "temperature": temperature_for(m, prompt, r[0])}
                     for hq, r in zip(hqs, rows)]
            out = student_probs(model(batch), batch)
        del hs, hqs, batch
        if a.prefix:
            del cache
        _trim_heap()  # the joint model also trims its heap at the end of every request
        return out

    def joint(state, qs):
        return m.probabilities(state, qs)

    print(f"{'state tokens':>12s} {'K':>2s} {'joint ms':>9s} {'shared ms':>10s} {'speedup':>8s}")
    for reps_text in (1, 2, 10):
        state = (base * reps_text).strip()
        n_tok = len(m.model.tokenizer(state)["input_ids"])
        for k in (1, 3, 6):
            qs = questions[:k]
            for f in (joint, shared):
                f(state, qs)
            tj, ts = [], []
            for _ in range(a.reps):
                t = time.perf_counter(); joint(state, qs); tj.append((time.perf_counter() - t) * 1000)
                t = time.perf_counter(); shared(state, qs); ts.append((time.perf_counter() - t) * 1000)
            mj, ms = statistics.median(tj), statistics.median(ts)
            print(f"{n_tok:12d} {k:2d} {mj:9.0f} {ms:10.0f} {mj / ms:7.2f}x", flush=True)


# ----------------------------------------------- variant B: prefix-cached trunk

def prefix_state(m, enc_mod, state_ids):
    """State-only pass that records, per layer, the state's keys and values (attention) or its last
    gated token (convolution). Returns the final state features and the cache."""
    import torch
    import torch.nn.functional as F

    trunk_ = m.model.encoder
    h = trunk_.embed_tokens(torch.tensor(state_ids))[None]
    L = h.shape[1]
    cos, sin = trunk_.rope(L, h)
    ones = torch.ones(1, L)
    cache = []
    for layer in trunk_.layers:
        x = layer.operator_norm(h)
        if layer.is_attention_layer:
            a = layer.self_attn
            shape = (1, L, -1, a.head_dim)
            q = a.q_layernorm(a.q_proj(x).view(shape)).transpose(1, 2)
            k = a.k_layernorm(a.k_proj(x).view(shape)).transpose(1, 2)
            v = a.v_proj(x).view(shape).transpose(1, 2)
            q = q * cos + enc_mod._rotate_half(q) * sin
            k = k * cos + enc_mod._rotate_half(k) * sin
            cache.append((k, v))
            g = a.heads // a.kv_heads
            y = F.scaled_dot_product_attention(q, k.repeat_interleave(g, 1), v.repeat_interleave(g, 1),
                                               scale=a.head_dim ** -0.5)
            x = a.out_proj(y.transpose(1, 2).reshape(1, L, -1))
        else:
            c = layer.conv
            b, _, u = c.in_proj(x).transpose(-1, -2).chunk(3, dim=-2)
            cache.append((b * u)[..., -1:])  # the last state token's gated input, (1, D, 1)
            x = c(x, ones, ones)
        h = h + x
        h = h + layer.feed_forward(layer.ffn_norm(h))
    return trunk_.embedding_norm(h)[0], cache


def prefix_questions(m, enc_mod, q_ids_list, cache, n_state):
    """Question tokens through the trunk, reading the cached state. One tensor per question."""
    import torch
    import torch.nn.functional as F

    trunk_ = m.model.encoder
    seqs = [trunk_.embed_tokens(torch.tensor(i)) for i in q_ids_list]
    h = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    B, Lq, _ = h.shape
    lens = torch.tensor([len(x) for x in seqs])
    pad = torch.arange(Lq)[None] < lens[:, None]
    padf = pad.float()
    cos, sin = trunk_.rope(n_state + Lq, h)
    cos, sin = cos[:, :, n_state:], sin[:, :, n_state:]
    key_mask = torch.zeros(B, 1, 1, n_state + Lq)
    key_mask[:, :, :, n_state:] = ((~pad).float() * -1e9)[:, None, None, :]
    for layer, kv in zip(trunk_.layers, cache):
        x = layer.operator_norm(h)
        if layer.is_attention_layer:
            a = layer.self_attn
            shape = (B, Lq, -1, a.head_dim)
            q = a.q_layernorm(a.q_proj(x).view(shape)).transpose(1, 2)
            k = a.k_layernorm(a.k_proj(x).view(shape)).transpose(1, 2)
            v = a.v_proj(x).view(shape).transpose(1, 2)
            q = q * cos + enc_mod._rotate_half(q) * sin
            k = k * cos + enc_mod._rotate_half(k) * sin
            K = torch.cat([kv[0].expand(B, -1, -1, -1), k], dim=2)
            V = torch.cat([kv[1].expand(B, -1, -1, -1), v], dim=2)
            g = a.heads // a.kv_heads
            y = F.scaled_dot_product_attention(q, K.repeat_interleave(g, 1), V.repeat_interleave(g, 1),
                                               attn_mask=key_mask, scale=a.head_dim ** -0.5)
            x = a.out_proj(y.transpose(1, 2).reshape(B, Lq, -1))
        else:
            c = layer.conv
            b, cc, u = c.in_proj(x * padf[:, :, None]).transpose(-1, -2).chunk(3, dim=-2)
            bx = b * u                                              # (B, D, Lq)
            w = c.conv.weight[:, 0, :]
            left = torch.cat([kv.expand(B, -1, -1), bx[..., :-1]], dim=-1)  # token 0 reads the cached state token
            xp = F.pad(bx, (1, 1))
            y = left * w[:, 0, None] + bx * w[:, 1, None] + xp[..., 2:2 + Lq] * w[:, 2, None]
            x = c.out_proj((cc * y).transpose(-1, -2))
        h = h + x
        h = h + layer.feed_forward(layer.ffn_norm(h))
    out = trunk_.embedding_norm(h)
    return [out[i, :n] for i, n in enumerate(lens.tolist())]


def cmd_prefix_eval(a) -> None:
    import torch

    torch.set_num_threads(a.threads)
    m, prompt, enc_mod = load_d1(a.threads)
    model = build_student(m, enc_mod)
    if a.head != "init":
        model.load_state_dict(torch.load(a.head))
    model.eval()
    print(f"prefix-cached trunk, head: {a.head}")
    print(f"{'task':9s} {'rows':>5s} {'teacher acc':>12s} {'student acc':>12s} {'diff':>6s} {'agreement':>10s} {'ECE teacher':>12s} {'ECE student':>12s}")
    res = {}
    for task in TASKS_EVAL:
        feats = load_feats(a.feat, "eval", [task])
        rows = rows_for(task, "test", len(feats), seed=11)
        s_pred, t_pred, gold, s_conf, t_conf = [], [], [], [], []
        for r, f in zip(rows, feats):
            q, ids, markers, cut = split_row(m, prompt, r)
            with torch.no_grad():
                hs, cache = prefix_state(m, enc_mod, ids[:cut])
                (hq,) = prefix_questions(m, enc_mod, [ids[cut:]], cache, cut)
                b = {"hs": hs, "hq": hq, "markers": [x - cut for x in markers], "qtype": f["qtype"],
                     "temperature": f["temperature"]}
                p = student_probs(model([b]), [b])[0]
            t = np.asarray(f["teacher"])
            s_pred.append(int(p.argmax())); t_pred.append(int(t.argmax())); gold.append(f["label"])
            s_conf.append(float(p.max())); t_conf.append(float(t.max()))
        s_ok = [p == y for p, y in zip(s_pred, gold)]
        t_ok = [p == y for p, y in zip(t_pred, gold)]
        agree = float(np.mean([x == y for x, y in zip(s_pred, t_pred)]))
        res[task] = dict(teacher=float(np.mean(t_ok)), student=float(np.mean(s_ok)), agreement=agree,
                         ece_t=ece(t_conf, t_ok), ece_s=ece(s_conf, s_ok))
        r_ = res[task]
        print(f"{task:9s} {len(feats):5d} {r_['teacher']:12.3f} {r_['student']:12.3f} {100 * (r_['student'] - r_['teacher']):+6.1f} "
              f"{agree:10.3f} {r_['ece_t']:12.3f} {r_['ece_s']:12.3f}", flush=True)
    trained = [res[t] for t in TASKS_TRAIN]
    g1 = min(r["agreement"] for r in trained) >= 0.95 and all(r["teacher"] - r["student"] <= 0.02 + 1e-12 for r in trained)
    g2 = res["emotion"]["teacher"] - res["emotion"]["student"] <= 0.03 + 1e-12
    g4 = all(r["ece_s"] <= r["ece_t"] + 0.03 for r in res.values())
    print(f"\ngate 1: {'PASS' if g1 else 'FAIL'}   gate 2: {'PASS' if g2 else 'FAIL'}   gate 4: {'PASS' if g4 else 'FAIL'}")
def cmd_extract_prefix(a) -> None:
    """Variant B features: reuse the teacher answers of an earlier extraction, recompute hs and hq."""
    import torch

    m, prompt, enc_mod = load_d1(a.threads)
    os.makedirs(a.out, exist_ok=True)
    for split in ("train", "eval"):
        for task in (TASKS_TRAIN if split == "train" else TASKS_EVAL):
            path = os.path.join(a.out, f"{split}_{task}.pt")
            if os.path.exists(path):
                print(f"kept {path}", flush=True)
                continue
            old = torch.load(os.path.join(a.src, f"{split}_{task}.pt"))
            rows = rows_for(task, "train" if split == "train" else "test", len(old), seed=7 if split == "train" else 11)
            feats, t0 = [], time.time()
            for r, f in zip(rows, old):
                q, ids, markers, cut = split_row(m, prompt, r)
                assert cut == f["state_len"], "rows are not aligned with the earlier extraction"
                with torch.no_grad():
                    hs, cache = prefix_state(m, enc_mod, ids[:cut])
                    (hq,) = prefix_questions(m, enc_mod, [ids[cut:]], cache, cut)
                feats.append({**f, "hs": hs.half(), "hq": hq.half()})
            torch.save(feats, path)
            print(f"extracted {split:5s} {task:8s} {len(feats)} rows in {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("extract", cmd_extract), ("train", cmd_train), ("eval", cmd_eval), ("speed", cmd_speed),
                     ("prefix-eval", cmd_prefix_eval), ("extract-prefix", cmd_extract_prefix)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        p.add_argument("--threads", type=int, default=2)
        if name == "extract-prefix":
            p.add_argument("--src", required=True, help="features from `extract` (teacher answers are reused)")
            p.add_argument("--out", required=True)
        if name == "extract":
            p.add_argument("--out", required=True)
            p.add_argument("--train-rows", type=int, default=600)
            p.add_argument("--eval-rows", type=int, default=200)
        if name in ("train", "eval", "prefix-eval"):
            p.add_argument("--feat", required=True)
        if name == "train":
            p.add_argument("--out", required=True)
            p.add_argument("--epochs", type=int, default=3)
            p.add_argument("--batch", type=int, default=16)
            p.add_argument("--lr", type=float, default=2e-5)
        if name in ("eval", "speed", "prefix-eval"):
            p.add_argument("--head", required=True)
        if name == "speed":
            p.add_argument("--reps", type=int, default=9)
            p.add_argument("--prefix", action="store_true", help="measure variant B instead of independent state/question features")
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
