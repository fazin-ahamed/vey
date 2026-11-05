"""Fit the System One calibration and measure the lanes on labelled public sets.

Reproduces the numbers in docs/SYSTEMONE.md. For every task it fits temperatures
on a sample of the *train* split, then reports on a disjoint sample of the *test*
split (SST-2 uses its validation split, the only labelled one):

  * accuracy of the matcher alone and of the cascade (matcher, then NLI rerank
    below ``--rerank-below``), and the share of rows the cascade reranked,
  * expected calibration error (ECE) with the shipped default temperatures and
    with the fitted ones,
  * mean milliseconds per row.

Label names or short descriptions are the options; a deployment's own criteria
will behave differently. Run it on your own data to fit your own temperatures:

    pip install datasets
    python scripts/systemone_eval.py --fit-rows 400 --eval-rows 300 --write calibration.json
    vey-serve --calibration calibration.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vey.semantic.onnx_encoder import OnnxEmbedder  # noqa: E402
from vey.systemone import (Calibration, Engine, EngineConfig, LazyNLI, OnnxNLI, ece,  # noqa: E402
                           fit_noul_temperature, fit_temperature)
from vey.systemone.calibration import _DEFAULTS  # noqa: E402
from vey.systemone.schema import Question, Request  # noqa: E402

CHOICE_TASKS = {
    # name: (loader, bucket key)
    "sst2": "choice:2", "ag_news": "choice:3-5", "emotion": "choice:6-10",
    "banking77": "choice:11+", "clinc150": "choice:11+",
}


def _sample(n, idx, rnd):
    idx = list(idx)
    return rnd.sample(idx, min(n, len(idx)))


def load_task(name: str, split: str, n: int, seed: int):
    """-> (texts, label indices, option descriptions in label order, instruction)."""
    from datasets import load_dataset

    rnd = random.Random(seed)
    if name == "sst2":
        d = load_dataset("stanfordnlp/sst2", split="train" if split == "train" else "validation")
        texts, labels = d["sentence"], d["label"]
        opts = {"negative": "a negative movie review", "positive": "a positive movie review"}
        instr = "What is the sentiment of this movie review?"
    elif name == "ag_news":
        d = load_dataset("fancyzhx/ag_news", split=split)
        texts, labels = d["text"], d["label"]
        opts = {"world": "international news and politics", "sports": "sports news",
                "business": "business and finance news", "science and technology": "science and technology news"}
        instr = "What is the topic of this news article?"
    elif name == "emotion":
        d = load_dataset("dair-ai/emotion", "split", split=split)
        texts, labels = d["text"], d["label"]
        opts = {"sadness": "a sad message", "joy": "a joyful message", "love": "a loving message",
                "anger": "an angry message", "fear": "a fearful message", "surprise": "a surprised message"}
        instr = "What emotion does this message express?"
    elif name == "banking77":
        d = load_dataset("PolyAI/banking77", split=split)
        texts, labels = d["text"], d["label"]
        opts = {n.replace("_", " "): "" for n in d.features["label"].names}
        instr = "What is the customer asking about?"
    elif name == "clinc150":
        d = load_dataset("clinc/clinc_oos", "plus", split=split)
        names = d.features["intent"].names
        keep = [i for i, y in enumerate(d["intent"]) if names[y] != "oos"]
        labs = [x for x in names if x != "oos"]
        remap = {names.index(x): j for j, x in enumerate(labs)}
        idx = _sample(n, keep, rnd)
        return ([d[i]["text"] for i in idx], [remap[d[i]["intent"]] for i in idx],
                {x.replace("_", " "): "" for x in labs}, "What does the user want?")
    else:
        raise ValueError(name)
    idx = _sample(n, range(len(texts)), rnd)
    return [texts[i] for i in idx], [labels[i] for i in idx], opts, instr


def load_score(split: str, n: int, seed: int):
    from datasets import load_dataset

    d = load_dataset("SetFit/sst5", split="train" if split == "train" else "test")
    idx = _sample(n, range(len(d)), random.Random(seed))
    levels = ["very negative review", "negative review", "neutral review", "positive review", "very positive review"]
    return [d[i]["text"] for i in idx], [d[i]["label"] for i in idx], levels


def load_noul(split: str, n: int, seed: int):
    """Balanced yes/no rows: (text, 0/1, question)."""
    from datasets import load_dataset

    rnd = random.Random(seed)
    rows = []
    s = load_dataset("stanfordnlp/sst2", split="train" if split == "train" else "validation")
    for y in (0, 1):
        for i in _sample(n // 4, [i for i, v in enumerate(s["label"]) if v == y], rnd):
            rows.append((s[i]["sentence"], y, "Is the sentiment of this movie review positive?"))
    a = load_dataset("fancyzhx/ag_news", split=split)
    for y in (0, 1):
        for i in _sample(n // 4, [i for i, v in enumerate(a["label"]) if (v == 1) == bool(y)], rnd):
            rows.append((a[i]["text"], y, "Is this news article about sports?"))
    rnd.shuffle(rows)
    return rows


def choice_row(text, opts, instr):
    q = Question("q", "choice", instr, dict(opts))
    return Request("eval", text, [q]), q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-rows", type=int, default=400)
    ap.add_argument("--eval-rows", type=int, default=300)
    ap.add_argument("--rerank-below", type=float, default=0.6)
    ap.add_argument("--write", help="write fitted temperatures to this JSON file")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()

    emb = OnnxEmbedder.from_hub(threads=a.threads)
    nli = LazyNLI(lambda: OnnxNLI.from_hub(threads=a.threads))
    base = Engine(emb, nli, Calibration(dict(_DEFAULTS)), config=EngineConfig(rerank_below=0.0))

    # ------------------------------------------------------------------ fit
    temps = dict(_DEFAULTS)
    pooled_cos: dict[str, tuple[list, list]] = {}
    for name, key in CHOICE_TASKS.items():
        texts, labels, opts, instr = load_task(name, "train", a.fit_rows, seed=1)
        rows = pooled_cos.setdefault(key, ([], []))
        for t, y in zip(texts, labels):
            req, q = choice_row(t, opts, instr)
            rows[0].append(base.cosines(req, q)[0])
            rows[1].append(y)
    for key, (cos, labels) in pooled_cos.items():
        temps[key] = round(fit_temperature(cos, labels), 5)
    texts, labels, levels = load_score("train", a.fit_rows, seed=1)
    cos, ys = [], []
    for t, y in zip(texts, labels):
        q = Question("q", "score", "How positive is this movie review?", levels)
        cos.append(base.cosines(Request("eval", t, [q]), q)[0])
        ys.append(y)
    temps["score"] = round(fit_temperature(cos, ys), 5)
    fitted = Calibration(temps)

    # choice_nli: rows the matcher is unsure about, NLI margins over its top 5
    nli_rows, nli_lab = [], []
    tmp = Engine(emb, nli, fitted, config=EngineConfig(rerank_below=0.0))
    for name, key in CHOICE_TASKS.items():
        texts, labels, opts, instr = load_task(name, "train", a.fit_rows // 2, seed=2)
        for t, y in zip(texts, labels):
            req, q = choice_row(t, opts, instr)
            cos, _ = tmp.cosines(req, q)
            p = np.exp((cos - cos.max()) / fitted.temperature("choice", len(cos)))
            p /= p.sum()
            if p.max() >= a.rerank_below:
                continue
            top = [int(i) for i in np.argsort(-p, kind="stable")[:5]]
            if y not in top:
                continue
            z = tmp.choice_margins(req, q, top)
            nli_rows.append(z)
            nli_lab.append(top.index(y))
    temps["choice_nli"] = round(fit_temperature(nli_rows, nli_lab), 5) if nli_rows else 1.0
    # noul
    zs, ys = [], []
    for t, y, qtext in load_noul("train", a.fit_rows, seed=1):
        q = Question("q", "noul", qtext)
        z, _ = tmp.noul_margin(Request("eval", t, [q]), q)
        zs.append(z)
        ys.append(y)
    temps["noul"] = round(fit_noul_temperature(zs, ys), 5)
    fitted = Calibration(temps)
    print("fitted temperatures:", json.dumps(temps, sort_keys=True))

    # ----------------------------------------------------------------- eval
    default = Engine(emb, nli, Calibration(dict(_DEFAULTS)), config=EngineConfig(rerank_below=0.0))
    matcher = Engine(emb, nli, fitted, config=EngineConfig(rerank_below=0.0))
    cascade = Engine(emb, nli, fitted, config=EngineConfig(rerank_below=a.rerank_below))
    ece_by_key: dict[str, list[tuple[float, float]]] = {}
    hdr = f"{'task (options)':22s} {'matcher':>8s} {'cascade':>8s} {'reranked':>9s} {'ECE def':>8s} {'ECE fit':>8s} {'ECE casc':>9s} {'ms/row':>7s}"
    print(f"\nheld-out rows per task: {a.eval_rows}; rerank below confidence {a.rerank_below}\n{hdr}")

    def run(engine, items):
        out = []
        for req, q, y in items:
            ans, tr, conf = engine._answer_local(req, q)
            pred = ans["choice"] if q.type == "choice" else int(round(ans["score"]))
            out.append((pred, tr, conf, y))
        return out

    for name in CHOICE_TASKS:
        texts, labels, opts, instr = load_task(name, "test" if name != "sst2" else "validation", a.eval_rows, seed=3)
        keys = list(opts)
        items = []
        for t, y in zip(texts, labels):
            req, q = choice_row(t, opts, instr)
            items.append((req, q, keys[y]))
        acc = lambda r: float(np.mean([p == y for p, _, _, y in r]))
        r_def, r_m = run(default, items), run(matcher, items)
        emb._cache.clear()  # time from a cold cache: earlier passes already embedded these texts
        t0 = time.perf_counter()
        r_c = run(cascade, items)
        ms = (time.perf_counter() - t0) / len(items) * 1000
        share = float(np.mean([tr["lane"] == "matcher+nli" for _, tr, _, _ in r_c]))
        e = lambda r: ece([c for _, _, c, _ in r], [p == y for p, _, _, y in r])
        ece_by_key.setdefault(CHOICE_TASKS[name], []).append((e(r_def), e(r_m)))
        print(f"{name + ' (' + str(len(opts)) + ')':22s} {acc(r_m):8.3f} {acc(r_c):8.3f} {share:9.2f} "
              f"{e(r_def):8.3f} {e(r_m):8.3f} {e(r_c):9.3f} {ms:7.1f}")
    texts, labels, levels = load_score("test", a.eval_rows, seed=3)
    items = []
    for t, y in zip(texts, labels):
        q = Question("q", "score", "How positive is this movie review?", levels)
        items.append((Request("eval", t, [q]), q, y))
    r_def, r_m = run(default, items), run(matcher, items)
    exact = lambda r: float(np.mean([p == y for p, _, _, y in r]))
    within1 = lambda r: float(np.mean([abs(p - y) <= 1 for p, _, _, y in r]))
    sd = ece([c for _, _, c, _ in r_def], [p == y for p, _, c, y in r_def])
    sf = ece([c for _, _, c, _ in r_m], [p == y for p, _, _, y in r_m])
    ece_by_key["score"] = [(sd, sf)]
    print(f"{'sst5 score (5)':22s} exact {exact(r_m):.3f}, within one level {within1(r_m):.3f}; "
          f"ECE default {sd:.3f}, fitted {sf:.3f}")

    rows = load_noul("test", a.eval_rows, seed=3)
    noul_ece: dict[str, float] = {}
    for label, cal in (("default T=1", Calibration(dict(_DEFAULTS))), ("fitted T", fitted)):
        eng = Engine(emb, nli, cal, config=EngineConfig())
        probs, ys, t0 = [], [], time.perf_counter()
        for t, y, qtext in rows:
            q = Question("q", "noul", qtext)
            probs.append(eng._noul(Request("eval", t, [q]), q)[0])
            ys.append(y)
        ms = (time.perf_counter() - t0) / len(rows) * 1000
        pred = [p >= 0.5 for p in probs]
        conf = [max(p, 1 - p) for p in probs]
        e_noul = ece(conf, [a_ == bool(y) for a_, y in zip(pred, ys)])
        ece_by_key.setdefault("noul", []).append((e_noul, e_noul))
        noul_ece[label] = e_noul
        print(f"noul yes/no ({label}): accuracy {np.mean([a_ == bool(y) for a_, y in zip(pred, ys)]):.3f}, "
              f"ECE {e_noul:.3f}, {ms:.1f} ms/row ({len(rows)} rows)")

    # Ship a fitted temperature only where it beat the untuned default on held-out rows.
    ece_by_key["noul"] = [(noul_ece["default T=1"], noul_ece["fitted T"])]
    shipped = dict(_DEFAULTS)
    for key, pairs in ece_by_key.items():
        d, f = (float(np.mean([x[i] for x in pairs])) for i in (0, 1))
        keep = f < d
        if keep:
            shipped[key] = temps[key]
        print(f"calibration {key:12s} held-out ECE default {d:.3f} vs fitted {f:.3f} -> ship {'fitted ' + str(temps[key]) if keep else 'default ' + str(_DEFAULTS[key])}")
    shipped["choice_nli"] = temps["choice_nli"]
    if a.write:
        with open(a.write, "w") as fh:
            fh.write(Calibration(shipped).to_json(fit_rows=a.fit_rows, eval_rows=a.eval_rows, source="scripts/systemone_eval.py"))


if __name__ == "__main__":
    main()
