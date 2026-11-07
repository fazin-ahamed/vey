"""The System One engine: typed questions in, calibrated typed answers out.

Each question runs through the cheapest lane that can answer it, and escalates
only when it cannot:

    exact     The state is an object with a field named like the question, and
              its value is a valid answer. No model runs. Probability 1 or 0.
    matcher   ``choice`` and ``score``: the 24M encoder (ONNX) embeds the state
              once and each option; cosine becomes a probability by temperature.
    nli       ``noul``: a zero-shot NLI model checks the state against the
              statement. Loaded on first use.
    d1        Opt-in int8 d1-omni: images, scores and low-confidence choices.
    escalate  Below the confidence floor, or when the request has images or a
              ``box`` question no local lane can read: forwarded to another server.

The engine never generates text. The light lanes use BM25 sentence selection
for long states; d1 uses its own context and right truncation.
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
    d1_noul: bool = False      # yes/no to d1 instead of the NLI model: 0.953 vs 0.939 at 15x the time, so off
    d1_score: bool = True      # score to d1: exact level 0.22 -> 0.49, within one level 0.69 -> 0.96
    d1_choice_below: float = 0.8   # a choice the matcher is less sure of than this goes to d1 (0.6 left too many confident misses)
    d1_max_options: int = 50   # d1 reads every option in its context; keep large choices on the matcher
    d1_max_images: int = 4
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
                 escalator: Escalator | None = None, config: EngineConfig | None = None, d1=None):
        self.d1 = d1  # Tier 2 (vey.systemone.d1.D1Model) or any object with probabilities(state, questions, images)
        self.embedder = embedder
        self.nli = nli if nli is not None else LazyNLI()
        self.calibration = calibration or Calibration.default()
        self.escalator = escalator
        self.cfg = config or EngineConfig()
        self.started = time.time()
        self._lock = threading.Lock()
        self.counts = {"requests": 0, "questions": 0, "exact": 0, "matcher": 0, "matcher+nli": 0, "nli": 0, "d1": 0, "escalated": 0,
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
        """A forced answer or an answer read straight off the state, or None."""
        if q.type in ("choice", "score") and len(q.options) == 1:
            return {"p": [1.0]}  # the wire schema permits one option; d1 requires two
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

    def _matcher(self, req: Request, q: Question, rerank: bool = True) -> tuple[np.ndarray, dict[str, Any]]:
        """Probabilities over a ``choice`` / ``score`` question's options."""
        cos, state = self.cosines(req, q)
        tau = self.calibration.temperature(q.type, len(cos))
        p = softmax(cos, tau)
        order = np.sort(p)[::-1]
        info = {"temperature": tau, "margin": float(order[0] - order[1]) if len(order) > 1 else 1.0,
                "state_tokens": self._count(state), "matcher_confidence": round(float(order[0]), 6)}
        if (rerank and q.type == "choice" and 1 < len(p) <= self.cfg.rerank_max_options
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

    # ----------------------------------------------------------------- tier 2

    def _d1_inputs(self, req: Request) -> tuple[Any, dict[str, dict[str, Any]], list]:
        """State, question dicts and images for d1, with asset placeholders resolved.

        A placeholder inside a string becomes "the attached image" and its asset joins the
        image list in order of first appearance; a state that is only a placeholder becomes None
        (d1 reads the images alone). Assets no string mentions are ignored.
        """
        from .d1 import decode_image

        assets, order = req.assets or {}, []
        names = sorted((re.escape(k) for k in assets if k), key=len, reverse=True)
        pattern = re.compile(r"(?<!\w)(?:" + "|".join(names) + r")(?!\w)") if names else None

        def replace(match) -> str:
            ph = match.group()
            if ph not in order:
                order.append(ph)
            return "the attached image"

        def sub(v: Any) -> Any:
            if isinstance(v, str):
                return pattern.sub(replace, v) if pattern else v
            if isinstance(v, dict):
                return {k: sub(x) for k, x in v.items()}
            if isinstance(v, list):
                return [sub(x) for x in v]
            return v

        state = req.state
        if isinstance(state, str) and state.strip() in assets:
            order.append(state.strip())
            state = None
        else:
            state = sub(state)
        qs: dict[str, dict[str, Any]] = {}
        for q in req.questions:
            d: dict[str, Any] = {"type": q.type, "instructions": to_text(sub(q.instruction_text))}
            if q.type == "choice":
                d["criteria"] = {k: to_text(sub(v)) for k, v in q.criteria.items()}
            elif q.type == "score":
                d["criteria"] = [to_text(sub(v)) for v in q.criteria]
            elif q.criteria:
                d["criteria"] = {k: to_text(sub(v)) for k, v in q.criteria.items()}
            qs[q.name] = d
        if len(order) > self.cfg.d1_max_images:
            raise bad_request(f"at most {self.cfg.d1_max_images} images per request are read locally, got {len(order)}")
        return state, qs, [decode_image(assets[ph], ph) for ph in order]

    @staticmethod
    def _from_d1(q: Question, probs: list[float]) -> Any:
        return float(probs[0]) if q.type == "noul" else [float(x) for x in probs]  # a noul is [yes, no]

    def _answer_all_local(self, req: Request) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
        """Answer every question locally. Returns answers, traces and names below the floor."""
        d1, has_images = self.d1, bool(req.assets) and self.d1 is not None
        done: dict[str, tuple] = {}
        pending: dict[str, dict[str, Any]] = {}
        for q in req.questions:
            t0 = time.perf_counter()
            hit = self._exact(req.state, q)
            if hit is not None:
                a, conf = self._shape(q, hit["p"])
                done[q.name] = (a, {"lane": "exact", "confidence": round(conf, 6),
                                    "ms": round((time.perf_counter() - t0) * 1000, 2)}, conf)
            elif d1 is not None and (has_images or (q.type == "noul" and self.cfg.d1_noul)
                                     or (q.type == "score" and self.cfg.d1_score)):
                pending[q.name] = {"q": q, "fallback": None}
            elif d1 is not None and q.type == "choice":
                p, info = self._matcher(req, q, rerank=False)
                if len(p) <= self.cfg.d1_max_options and (self.cfg.d1_choice_below >= 1.0
                                                         or float(p.max()) < self.cfg.d1_choice_below):
                    pending[q.name] = {"q": q, "fallback": (p, info, t0)}
                else:
                    a, conf = self._shape(q, p)
                    done[q.name] = (a, {"lane": "matcher", "confidence": round(conf, 6), **info,
                                        "ms": round((time.perf_counter() - t0) * 1000, 2)}, conf)
            else:
                done[q.name] = self._answer_local(req, q)
        if pending:
            names = list(pending)
            state, d1q, images = self._d1_inputs(req)
            t0 = time.perf_counter()
            try:
                probs = d1.probabilities(state, [d1q[n] for n in names], images or None)
                err = None
            except Exception as e:  # out of memory, a bad image, ...
                probs, err = None, f"{type(e).__name__}: {e}"
            ms = (time.perf_counter() - t0) * 1000
            for i, n in enumerate(names):
                q, fb = pending[n]["q"], pending[n]["fallback"]
                if err is None:
                    a, conf = self._shape(q, self._from_d1(q, probs[i]))
                    tr = {"lane": "d1", "confidence": round(conf, 6), "ms": round(ms / len(names), 2),
                          "d1_call_ms": round(ms, 2), "images": len(images)}
                    if fb:
                        tr["matcher_confidence"] = fb[1]["matcher_confidence"]
                    done[n] = (a, tr, conf)
                elif fb:  # keep the matcher's answer
                    p, info, t_q = fb
                    a, conf = self._shape(q, p)
                    done[n] = (a, {"lane": "matcher", "confidence": round(conf, 6), **info, "d1_error": err,
                                   "ms": round((time.perf_counter() - t_q) * 1000, 2)}, conf)
                elif images:
                    raise ApiError(503, {"detail": f"the d1 lane failed on this image request: {err}"})
                else:  # text: fall back to the lane Vey used before d1
                    a, tr, conf = self._answer_local(req, q)
                    tr["d1_error"] = err
                    done[n] = (a, tr, conf)
        answers = {q.name: done[q.name][0] for q in req.questions}
        trace = {q.name: done[q.name][1] for q in req.questions}
        low = [q.name for q in req.questions
               if trace[q.name]["lane"] != "exact" and done[q.name][2] < self.cfg.floor]
        for q in req.questions:
            self._bump(trace[q.name]["lane"])
        return answers, trace, low

    def answer(self, req: Request, raw_body: dict[str, Any] | None = None, headers: dict[str, str] | None = None,
               explain: bool | None = None) -> dict[str, Any]:
        """Answer every question of a validated request. Raises ApiError on a request that cannot be served."""
        t_start = time.perf_counter()
        explain = self.cfg.explain if explain is None else explain
        reason = None
        if any(q.type == "box" for q in req.questions):
            reason = "a box question, which no local lane answers"
        elif req.assets and self.d1 is None:
            reason = "images, which need the d1 lane (start the server with --d1)"
        elif req.assets and any(q.type == "choice" and len(q.options) > self.cfg.d1_max_options for q in req.questions):
            reason = f"an image choice with more than {self.cfg.d1_max_options} options, which exceeds the local d1 limit"
        unreadable = reason is not None
        if unreadable and self.escalator is None:
            raise bad_request(
                "this request has " + reason
                + ", and no escalation endpoint is configured (start the server with --escalate-url)")
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
            answers, trace, low = self._answer_all_local(req)
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
                "d1": self.d1.info() if self.d1 is not None and hasattr(self.d1, "info") else bool(self.d1),
                "escalation": {"host": self.escalator.host} if self.escalator else None,
            },
            "floor": self.cfg.floor,
            "temperatures": self.calibration.temperatures,
            "counts": dict(self.counts),
        }


__all__ = ["Engine", "EngineConfig", "select_state", "find_field"]
