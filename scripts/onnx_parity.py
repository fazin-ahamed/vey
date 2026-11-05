"""Parity and cost of the ONNX encoder against the fp32 reference.

Reproduces the numbers in docs/SYSTEMONE.md. For each backend it reports, on the
same rows, the same candidate descriptions and the same host:

  * open-candidate matcher accuracy on Banking77 and CLINC150 (label names as
    the candidate descriptions, one query encoded per row, candidates cached),
  * mean cosine of query embeddings against the fp32 ONNX export,
  * p50 latency to encode one query,
  * resident memory of a fresh process after load (``--rss`` runs one backend).

    pip install datasets            # only this script needs it
    python scripts/onnx_parity.py --rows 500 [--torch]

Absolute accuracies are not comparable to the README's Laya table: that table
used a different candidate description for each label. Only the differences
between backends in this one run are meaningful.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def rss_mb() -> float:
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS"):
            return int(line.split()[1]) / 1024
    return float("nan")


def load_rows(n: int, seed: int = 0):
    from datasets import load_dataset

    rnd = random.Random(seed)
    sets = {}
    b = load_dataset("PolyAI/banking77", split="test")
    names = b.features["label"].names
    idx = rnd.sample(range(len(b)), min(n, len(b)))
    sets["banking77"] = ([b[i]["text"] for i in idx], [b[i]["label"] for i in idx],
                         [x.replace("_", " ") for x in names])
    c = load_dataset("clinc/clinc_oos", "plus", split="test")
    names = c.features["intent"].names
    keep = [i for i, y in enumerate(c["intent"]) if names[y] != "oos"]
    idx = rnd.sample(keep, min(n, len(keep)))
    labels = [x for x in names if x != "oos"]
    remap = {names.index(x): j for j, x in enumerate(labels)}
    sets["clinc150"] = ([c[i]["text"] for i in idx], [remap[c[i]["intent"]] for i in idx],
                        [x.replace("_", " ") for x in labels])
    return sets


def make_embedder(spec: str):
    """``torch`` or ``onnx:<variant>``."""
    if spec == "torch":
        from vey.semantic import Embedder, load_encoder

        tok, enc = load_encoder(device="cpu")
        return Embedder(tok, enc, "cpu")
    from vey.semantic.onnx_encoder import OnnxEmbedder

    return OnnxEmbedder.from_hub(variant=spec.split(":", 1)[1], threads=int(os.environ.get("THREADS", "2")))


def evaluate(spec: str, sets: dict, ref_vecs: dict | None):
    emb = make_embedder(spec)
    out, vecs = {}, {}
    for name, (texts, labels, cands) in sets.items():
        cv = emb.embed(cands)
        q = emb.embed(texts[:1])  # warm
        correct = 0
        for t, y in zip(texts, labels):
            correct += int(np.argmax(cv @ emb.embed([t])[0]) == y)
        out[f"{name}_acc"] = correct / len(texts)
        vecs[name] = emb.embed(texts)
        if ref_vecs is not None:
            out[f"{name}_cos"] = float((vecs[name] * ref_vecs[name]).sum(1).mean())
    ts = []
    probe = [f"I was charged twice for order {i} and want a refund" for i in range(40)]
    for t in probe:
        s = time.perf_counter()
        emb.embed([t])
        ts.append((time.perf_counter() - s) * 1000)
    out["p50_ms"] = statistics.median(ts)
    return out, vecs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=500)
    ap.add_argument("--torch", action="store_true", help="also run the PyTorch reference (needs vey[torch])")
    ap.add_argument("--variants", default="model,model_int8")
    ap.add_argument("--rss", help="print resident memory after loading one backend, then exit")
    a = ap.parse_args()
    if a.rss:
        base = rss_mb()
        emb = make_embedder(a.rss)
        emb.embed(["warm up"])
        print(json.dumps({"backend": a.rss, "rss_mb": round(rss_mb()), "before_load_mb": round(base)}))
        return
    sets = load_rows(a.rows)
    specs = ["onnx:" + v for v in a.variants.split(",")] + (["torch"] if a.torch else [])
    _, ref = evaluate("onnx:model", sets, None)
    print(f"host: {os.cpu_count()} CPUs, threads={os.environ.get('THREADS', '2')}, rows={a.rows}")
    print(f"{'backend':18s} {'banking77':>10s} {'clinc150':>9s} {'cos b77':>8s} {'cos clinc':>9s} {'p50 ms':>7s} {'RSS MB':>7s}")
    for spec in specs:
        r, _ = evaluate(spec, sets, ref)
        rs = subprocess.run([sys.executable, __file__, "--rss", spec], capture_output=True, text=True)
        try:
            mem = json.loads(rs.stdout.strip().splitlines()[-1])["rss_mb"]
        except Exception:
            mem = float("nan")
        print(f"{spec:18s} {r['banking77_acc']:10.3f} {r['clinc150_acc']:9.3f} {r['banking77_cos']:8.4f} "
              f"{r['clinc150_cos']:9.4f} {r['p50_ms']:7.2f} {mem:7.0f}")


if __name__ == "__main__":
    main()
