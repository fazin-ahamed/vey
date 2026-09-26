"""Vey 2 lane router and public ``decide`` entrypoint.

Routes a decision to the cheapest lane that can answer it:

    structured  every axis/filter resolves against explicit numeric/enum
                candidate facts -> deterministic DictGrounder, no model.
    crux        otherwise -> frozen NLI predicate grounder + trained ordinal
                comparator (lazy-loaded; fails closed without its artifact).

The learned CRUX backbone runs only when language grounding is required. Both
lanes share the identical permutation-exact executor.
"""
from __future__ import annotations

from .crux.compiler import compile_instruction
from .crux.decider import run_decision
from .decision import DecisionRequest, DecisionResult
from .structured.exact import DictGrounder


class Runtime:
    """Holds lane grounders. The CRUX grounder is created lazily and reused so
    models load at most once per process. Inject ``crux_grounder`` to supply a
    custom/mock backbone (e.g. tests, or a distilled student)."""

    def __init__(self, device: str = "cpu", enable_crux: bool = True, crux_grounder=None):
        self.device = device
        self.enable_crux = enable_crux
        self._dict = DictGrounder()
        self._crux = crux_grounder

    def _crux_grounder(self):
        if self._crux is None:
            from .crux.grounding import SageGrounder
            self._crux = SageGrounder(device=self.device)
        return self._crux

    def decide(self, request: DecisionRequest) -> DecisionResult:
        if not request.candidates:
            return DecisionResult(None, {}, "abstain",
                                  {"state": "abstain", "reason": "no candidates"}, None)
        program = compile_instruction(request.question)
        texts = list(request.candidates.values())
        if self._dict.handles(program, texts):
            return run_decision(self._dict, request, "structured")
        if not self.enable_crux:
            return DecisionResult(None, {i: 0.0 for i in request.candidates}, "abstain",
                                  {"state": "abstain", "reason": "crux lane disabled"}, None)
        return run_decision(self._crux_grounder(), request, "crux")


_DEFAULT: Runtime | None = None


def _default_runtime() -> Runtime:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Runtime()
    return _DEFAULT


def decide(question: str, candidates: dict[str, str], *, state: str = "",
           explain: bool = False, runtime: Runtime | None = None) -> DecisionResult:
    """Vey 2 semantic decision. See ``docs/CRUX.md``.

    (The Vey 1 trust-policy decision remains available as ``vey.trust.decide``.)
    """
    req = DecisionRequest(question=question, candidates=dict(candidates), state=state, explain=explain)
    return (runtime or _default_runtime()).decide(req)
