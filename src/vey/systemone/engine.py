"""The System One engine: typed questions in, calibrated typed answers out.

Each question runs through the cheapest lane that can answer it, and escalates
only when it cannot:

    exact     The state is an object with a field named like the question, and
              its value is a valid answer. No model runs. Probability 1 or 0.
    matcher   ``choice`` and ``score``: the 24M encoder (ONNX) embeds the state
              once and each option; cosine becomes a probability by temperature.
    nli       ``noul``: a zero-shot NLI model checks the state against the
              statement. Loaded on first use.
    escalate  Below the confidence floor, or when the request has images or a
              ``box`` question: forwarded to another System One server.

The engine never generates text. A long state is reduced with BM25 sentence
selection before either model reads it, the same long-context lane Vey uses
elsewhere.
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..retrieval import BM25
from .calibration import Calibration, sigmoid, softmax
from .escalate import EscalationError, Escalator
from .nli import LazyNLI, to_statement
from .schema import ApiError, Question, Request, bad_request, readable, to_text

_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class EngineConfig:
    model_id: str = "vey-1"    # the name answers carry; "vey-latest" is an alias for it
    aliases: tuple = ("vey-latest",)
    release_date: str = "2026-11-05"
    description: str = "Vey System One: exact lane, 24M ONNX encoder, zero-shot NLI, no generated text."
    exact_lane: bool = True
    state_budget: int = 92     # tokens of state the 24M encoder reads (its limit is 96)
    nli_budget: int = 220      # tokens of state the NLI model reads
    floor: float = 0.0         # confidence below this escalates (if configured) or is flagged
    rerank_below: float = 0.6  # a choice the matcher is less sure of than this is reranked by the NLI model
    rerank_top: int = 5        # how many of the matcher's top options the NLI model reads
    rerank_max_options: int = 20  # no rerank beyond this many options: measured to hurt at 150, +1 point at 77
    explain: bool = False      # default for the ``explain`` request field


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def select_state(text: str, query: str, budget: int, count) -> str:
    """Keep the sentences most relevant to ``query`` that fit in ``budget`` tokens.

    A state that already fits is returned whole (so every question shares one
    embedding of it). Otherwise sentences are ranked by BM25 against ``query`` and
    taken in rank order until the budget is full, then restored to reading order.
    """
    if count(text) <= budget:
        return text
    sents = [s.strip() for s in _SENTENCE.split(text) if s.strip()]
    if len(sents) <= 1:
        return text  # one long sentence: the model truncates it
    order = np.argsort(-BM25(sents).scores(query), kind="stable")
    chosen, used = [], 0
    for i in order:
        n = count(sents[i])
        if used + n <= budget:
            chosen.append(int(i))
            used += n
    if not chosen:
        chosen = [int(order[0])]
    return " ".join(sents[i] for i in sorted(chosen))


def find_field(state: Any, name: str, depth: int = 3) -> Any:
    """First value, breadth first, of a key that matches ``name`` (case and punctuation blind)."""
    target = _norm(name)
    level = [state] if isinstance(state, dict) else []
    for _ in range(depth):
        nxt = []
        for d in level:
            for k, v in d.items():
                if _norm(k) == target:
                    return v
                if isinstance(v, dict):
                    nxt.append(v)
        level = nxt
    return None


class Engine:
    def __init__(self, embedder, nli: LazyNLI | None = None, calibration: Calibration | None = None,
                 escalator: Escalator | None = None, config: EngineConfig | None = None):
        self.embedder = embedder
        self.nli = nli if nli is not None else LazyNLI()
        self.calibration = calibration or Calibration.default()
        self.escalator = escalator
        self.cfg = config or EngineConfig()
        self.started = time.time()
        self._lock = threading.Lock()
        self.counts = {"requests": 0, "questions": 0, "exact": 0, "matcher": 0, "matcher+nli": 0, "nli": 0, "escalated": 0,
                       "below_floor": 0}

    # --------------------------------------------------------------- helpers

    def _count(self, text: str) -> int:
        f = getattr(self.embedder, "count_tokens", None)
        return f(text) if f else int(len(text.split()) * 1.3) + 2

    def _bump(self, key: str, n: int = 1) -> None:
        with self._lock:
            self.counts[key] += n

    # ----------------------------------------------------------------- lanes

    def _exact(self, state: Any, q: Question) -> dict[str, Any] | None:
        """An answer read straight off the state, or None."""
        if not self.cfg.exact_lane or not isinstance(state, dict):
            return None
        v = find_field(state, q.name)
        if v is None:
            return None
        if q.type == "noul":
            if isinstance(v, bool):
                return {"p": 1.0 if v else 0.0}
            if isinstance(v, str) and v.strip().lower() in ("true", "false", "yes", "no"):
                return {"p": 1.0 if v.strip().lower() in ("true", "yes") else 0.0}
        elif q.type == "choice":
            keys = [k for k, _ in q.options]
            hit = [k for k in keys if _norm(k) == _norm(v)] if isinstance(v, (str, bool)) else []
            if len(hit) == 1:
                return {"p": [1.0 if k == hit[0] else 0.0 for k in keys]}
        elif q.type == "score":
            n = len(q.options)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and float(v).is_integer() and 0 <= v < n:
                return {"p": [1.0 if i == int(v) else 0.0 for i in range(n)]}
        return None

    def cosines(self, req: Request, q: Question) -> tuple[np.ndarray, str]:
        """Raw cosine of the (selected) state to each option of a ``choice`` / ``score`` question."""
        opts = q.options
        if q.type == "score":
            instr = q.instruction_text
            texts = [f"{instr} {q.option_text(k, d)}" for k, d in opts]
        else:
            texts = [q.option_text(k, d) for k, d in opts]
        state = select_state(req.state_text, q.instruction_text + " " + " ".join(texts),
                             self.cfg.state_budget, self._count)
        vecs = self.embedder.embed([state] + texts)
        return vecs[1:] @ vecs[0], state

    def _hypothesis(self, q: Question, key: str, description: Any) -> str:
        d = to_text(description)
        if not d:
            return f"This example is {readable(key)}."
        d = d if d[:2].isupper() else d[0].lower() + d[1:]
        return f"This example is {d.rstrip('.')}."

    def choice_margins(self, req: Request, q: Question, top: list[int]) -> np.ndarray | None:
        """NLI entailment margins of the ``top`` options of a choice, or None without an NLI model."""
        nli = self.nli.get()
        if nli is None:
            return None
        opts = q.options
        hyps = [self._hypothesis(q, opts[i][0], opts[i][1]) for i in top]
        state = select_state(req.state_text, q.instruction_text + " " + " ".join(hyps),
                             self.cfg.nli_budget, nli.count_tokens)
        return nli.margin([(state, h) for h in hyps])

    def noul_margin(self, req: Request, q: Question) -> tuple[float, dict[str, Any]]:
        """The NLI evidence for a yes/no question as one logit (>0 favors yes)."""
        nli = self.nli.get()
        if nli is None:
            raise bad_request("noul questions need the NLI lane, which could not be loaded"
                              + (f" ({self.nli.error})" if self.nli.error else "")
                              + "; answer them with a state field named like the question, or configure escalation")
        crit = q.criteria or {}
        true_h = to_text(crit.get("true")) or (
            to_statement(q.instruction_text) if (q.instructions or not crit.get("false")) else "")
        false_h = to_text(crit.get("false"))
        hyps = [h for h in (true_h, false_h) if h]
        state = select_state(req.state_text, " ".join(hyps), self.cfg.nli_budget, nli.count_tokens)
        m = nli.margin([(state, h) for h in hyps])
        z = float((m[0] - m[1]) / 2) if (true_h and false_h) else float(m[0]) if true_h else float(-m[0])
        return z, {"hypothesis": hyps[0], "state_tokens": nli.count_tokens(state)}

    def _matcher(self, req: Request, q: Question) -> tuple[np.ndarray, dict[str, Any]]:
        """Probabilities over a ``choice`` / ``score`` question's options."""
        cos, state = self.cosines(req, q)
        tau = self.calibration.temperature(q.type, len(cos))
        p = softmax(cos, tau)
        order = np.sort(p)[::-1]
        info = {"temperature": tau, "margin": float(order[0] - order[1]) if len(order) > 1 else 1.0,
                "state_tokens": self._count(state), "matcher_confidence": round(float(order[0]), 6)}
        if (q.type == "choice" and 1 < len(p) <= self.cfg.rerank_max_options
                and order[0] < self.cfg.rerank_below):
            top = [int(i) for i in np.argsort(-p, kind="stable")[: self.cfg.rerank_top]]
            z = self.choice_margins(req, q, top)
            if z is not None:  # keep the matcher's top-vs-rest mass; the NLI model orders within the top
                t = self.calibration.temperature("choice_nli")
                out = p.copy()
                out[top] = float(p[top].sum()) * softmax(z, t)
                info["reranked"] = [q.options[i][0] for i in top]
                info["nli_temperature"] = t
                return out, info
        return p, info

    def _noul(self, req: Request, q: Question) -> tuple[float, dict[str, Any]]:
        z, info = self.noul_margin(req, q)
        t = self.calibration.temperature("noul")
        return sigmoid(z / t), {"temperature": t, "margin_logit": round(z, 4), **info}

    # --------------------------------------------------------------- answers

    @staticmethod
    def _shape(q: Question, p: Any) -> tuple[dict[str, Any], float]:
        """Wire answer and top-1 confidence from probabilities."""
        if q.type == "noul":
            p = float(p)
            return {"type": "noul", "noul": round(p, 6)}, max(p, 1.0 - p)
        probs = [float(x) for x in p]
        best = int(np.argmax(probs))
        keys = [k for k, _ in q.options]
        out = {"type": q.type, "confidence": round(probs[best], 6),
               "probabilities": {k: round(v, 6) for k, v in zip(keys, probs)}}
        if q.type == "choice":
            return {"type": "choice", "choice": keys[best], **{k: out[k] for k in ("confidence", "probabilities")}}, probs[best]
        score = float(sum(i * v for i, v in enumerate(probs)))
        return ({"type": "score", "score": round(score, 6), "confidence": out["confidence"],
                 "legend": {k: d for k, d in q.options}, "probabilities": out["probabilities"]}, probs[best])

    def _answer_local(self, req: Request, q: Question) -> tuple[dict[str, Any], dict[str, Any], float]:
        t0 = time.perf_counter()
        hit = self._exact(req.state, q)
        if hit is not None:
            lane, p, info = "exact", hit["p"], {}
        elif q.type == "noul":
            lane = "nli"
            p, info = self._noul(req, q)
        else:
            p, info = self._matcher(req, q)
            lane = "matcher+nli" if "reranked" in info else "matcher"
        answer, conf = self._shape(q, p)
        trace = {"lane": lane, "confidence": round(conf, 6), **info,
                 "ms": round((time.perf_counter() - t0) * 1000, 2)}
        return answer, trace, conf

    def answer(self, req: Request, raw_body: dict[str, Any] | None = None, headers: dict[str, str] | None = None,
               explain: bool | None = None) -> dict[str, Any]:
        """Answer every question of a validated request. Raises ApiError on a request that cannot be served."""
        t_start = time.perf_counter()
        explain = self.cfg.explain if explain is None else explain
        unreadable = bool(req.assets) or any(q.type == "box" for q in req.questions)
        if unreadable and self.escalator is None:
            raise bad_request(
                "this request has images or a box question, which Vey's local lanes cannot read, and no "
                "escalation endpoint is configured (start the server with --escalate-url)")
        answers: dict[str, Any] = {}
        trace: dict[str, Any] = {}
        remote_model = None
        if unreadable:
            try:
                remote = self._escalate(raw_body or {}, [q.name for q in req.questions], headers)
            except EscalationError as e:  # nothing local to fall back on
                raise ApiError(502, {"detail": str(e)}) from e
            remote_model = remote.get("model")
            for q in req.questions:
                if q.name not in remote["answers"]:
                    raise ApiError(502, {"detail": f"escalation target did not answer {q.name!r}"})
                answers[q.name] = remote["answers"][q.name]
                trace[q.name] = {"lane": "escalated", "target": self.escalator.host}
            usage = remote.get("usage", {"input_tokens": 0, "output_tokens": 0})
            self._bump("escalated", len(req.questions))
        else:
            low: list[str] = []
            for q in req.questions:
                answer, tr, conf = self._answer_local(req, q)
                answers[q.name], trace[q.name] = answer, tr
                if tr["lane"] != "exact" and conf < self.cfg.floor:
                    low.append(q.name)
                self._bump(tr["lane"])
            for name in low:
                trace[name]["below_floor"] = True
            if low:
                self._bump("below_floor", len(low))
                if self.escalator is not None:
                    try:
                        remote = self._escalate(raw_body or {}, low, headers)
                        for name in low:
                            if name in remote["answers"]:
                                answers[name] = remote["answers"][name]
                                trace[name] = {"lane": "escalated", "target": self.escalator.host,
                                               "local": trace[name]}
                                self._bump("escalated")
                    except EscalationError as e:
                        for name in low:
                            trace[name]["escalation_error"] = str(e)
            usage = {"input_tokens": self._usage_tokens(req), "output_tokens": 0}
        self._bump("requests")
        self._bump("questions", len(req.questions))
        out: dict[str, Any] = {"model": remote_model if remote_model and unreadable else self.cfg.model_id,
                               "answers": answers, "usage": usage}
        if explain:
            out["trace"] = {"questions": trace, "total_ms": round((time.perf_counter() - t_start) * 1000, 2),
                            "floor": self.cfg.floor}
        return out

    def _usage_tokens(self, req: Request) -> int:
        n = self._count(req.state_text)
        for q in req.questions:
            n += self._count(q.instruction_text)
            n += sum(self._count(q.option_text(k, d)) for k, d in q.options)
        return n

    def _escalate(self, raw_body: dict[str, Any], names: list[str], headers: dict[str, str] | None) -> dict[str, Any]:
        qs = {n: raw_body["questions"][n] for n in names if n in raw_body.get("questions", {})}
        return self.escalator.post_json({**raw_body, "questions": qs}, headers)  # raises EscalationError

    def info(self) -> dict[str, Any]:
        e = self.embedder
        return {
            "model": self.cfg.model_id,
            "uptime_s": round(time.time() - self.started, 1),
            "lanes": {
                "exact": self.cfg.exact_lane,
                "matcher": {"embedder": type(e).__name__, "dim": getattr(e, "dim", None),
                            "cached_texts": len(getattr(e, "_cache", {}))},
                "nli": {"loaded": self.nli.loaded, "error": self.nli.error},
                "escalation": {"host": self.escalator.host} if self.escalator else None,
            },
            "floor": self.cfg.floor,
            "temperatures": self.calibration.temperatures,
            "counts": dict(self.counts),
        }


__all__ = ["Engine", "EngineConfig", "select_state", "find_field"]
