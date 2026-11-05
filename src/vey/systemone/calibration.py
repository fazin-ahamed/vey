"""Probability calibration for the System One answers.

Cosine similarities and NLI margins are scores, not probabilities. Each is turned
into a probability with one temperature, chosen by question type and option
count (the same scheme Laya and d1 use), and a deployment re-fits those
temperatures on its own labelled data with :func:`fit_temperature`.

    choice / score   p = softmax(cosine / T)        T by option count
    choice reranked  p = softmax(margin / T)        NLI margins of the matcher's top options
    noul             p = sigmoid(margin / T)        margin = NLI entailment logit gap

The shipped defaults come from ``scripts/fit_calibration.py`` and are fitted on
public intent, topic, emotion and sentiment sets with label-name candidates.
They are a starting point. They are not guaranteed to be calibrated for your
criteria, and the docs say so.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

_DEFAULTS = {"choice:2": 0.05, "choice:3-5": 0.05, "choice:6-10": 0.05, "choice:11+": 0.05,
             "score": 0.05, "noul": 1.0, "choice_nli": 1.0}


def bucket(k: int) -> str:
    return "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"


def key_for(kind: str, k: int = 0) -> str:
    return f"choice:{bucket(k)}" if kind == "choice" else kind


@dataclass
class Calibration:
    temperatures: dict[str, float] = field(default_factory=lambda: dict(_DEFAULTS))

    def temperature(self, kind: str, k: int = 0) -> float:
        return float(self.temperatures.get(key_for(kind, k), _DEFAULTS.get(key_for(kind, k), 1.0)))

    @classmethod
    def load(cls, path: str) -> "Calibration":
        with open(path) as f:
            raw = json.load(f)
        temps = raw.get("temperatures", raw)
        return cls({**_DEFAULTS, **{k: float(v) for k, v in temps.items()}})

    @classmethod
    def default(cls) -> "Calibration":
        path = os.path.join(os.path.dirname(__file__), "calibration_default.json")
        return cls.load(path) if os.path.exists(path) else cls()

    def to_json(self, **meta) -> str:
        return json.dumps({"temperatures": self.temperatures, **meta}, indent=2, sort_keys=True)


def softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    z = np.asarray(scores, dtype=np.float64) / max(temperature, 1e-6)
    z -= z.max()
    e = np.exp(z)
    return e / e.sum()


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -50, 50))))


def _search(objective, lo: float = 0.003, hi: float = 5.0, n: int = 160) -> float:
    grid = np.exp(np.linspace(np.log(lo), np.log(hi), n))
    vals = [objective(t) for t in grid]
    i = int(np.argmin(vals))
    a, b = grid[max(i - 1, 0)], grid[min(i + 1, n - 1)]
    for _ in range(40):  # golden-section refinement around the best grid point
        c, d = b - 0.618 * (b - a), a + 0.618 * (b - a)
        a, b = (a, d) if objective(c) < objective(d) else (c, b)
    return float((a + b) / 2)


def fit_temperature(score_rows: list[np.ndarray], labels: list[int]) -> float:
    """The T minimizing the negative log likelihood of ``softmax(scores / T)``.

    ``score_rows[i]`` holds the raw similarity of row *i* to each of its options;
    ``labels[i]`` is the index of the right option.
    """
    def nll(t: float) -> float:
        return -float(np.mean([np.log(max(softmax(s, t)[y], 1e-12)) for s, y in zip(score_rows, labels)]))

    return _search(nll)


def fit_noul_temperature(margins: list[float], labels: list[int]) -> float:
    """The T minimizing log loss of ``sigmoid(margin / T)`` against 0/1 labels."""
    m = np.asarray(margins, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)

    def nll(t: float) -> float:
        p = np.clip(1.0 / (1.0 + np.exp(-np.clip(m / t, -50, 50))), 1e-12, 1 - 1e-12)
        return -float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    return _search(nll, lo=0.05, hi=20.0)


def ece(confidences: list[float], correct: list[bool], bins: int = 10) -> float:
    """Expected calibration error of top-1 confidences."""
    c = np.asarray(confidences, dtype=np.float64)
    ok = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (c > lo) & (c <= hi) if lo > 0 else (c >= lo) & (c <= hi)
        if m.any():
            total += m.mean() * abs(ok[m].mean() - c[m].mean())
    return float(total)


__all__ = ["Calibration", "softmax", "sigmoid", "fit_temperature", "fit_noul_temperature", "ece",
           "bucket", "key_for"]
