"""Vey 2 lane router and public ``decide`` entrypoint.

Routes a decision to the cheapest lane that can answer it:

    structured  every axis/filter resolves against explicit numeric/enum
                candidate facts -> deterministic DictGrounder, no model.
    field       every ordinal axis is one the O(K) semantic field was
                validated on, and the program has no predicate filter ->
                one forward pass per candidate.
    crux        anything else that needs language grounding -> frozen NLI
                predicate grounder + trained pairwise ordinal comparator.

The field is preferred over the comparator wherever it applies, because it
answers the same question in O(K) passes instead of O(K^2). Both learned lanes
fail closed without their artifact. All three lanes share the identical
permutation-exact executor.
"""
from __future__ import annotations

from .crux.compiler import compile_instruction
from .crux.decider import run_decision
from .decision import DecisionRequest, DecisionResult
from .structured.exact import DictGrounder


class Runtime:
    """Holds lane grounders. Learned grounders are created lazily and reused so
    models load at most once per process. Inject ``crux_grounder`` or
    ``field_scorer`` to supply a custom backbone (tests, or a student)."""

    def __init__(self, device: str = "cpu", enable_crux: bool = True,
                 crux_grounder=None, field_scorer=None):
        self.device = device
        self.enable_crux = enable_crux
        self._dict = DictGrounder()
        self._crux = crux_grounder
        self._field = field_scorer

    def _field_scorer(self):
        if self._field is None:
            from .crux.field import FieldScorer
            self._field = FieldScorer(device=self.device)
        return self._field

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
        if self._field_handles(program):
            return run_decision(self._field_scorer(), request, "field")
        if not self.enable_crux:
            return DecisionResult(None, {i: 0.0 for i in request.candidates}, "abstain",
                                  {"state": "abstain", "reason": "crux lane disabled"}, None)
        return run_decision(self._crux_grounder(), request, "crux")

    def _field_handles(self, program) -> bool:
        """The field covers ordinal axes only, so a program routes there only
        when every stage is a MAX on a validated axis."""
        if not program:
            return False
        from .crux.field import FIELD_AXES
        return all(s.kind == "MAX" and s.key in FIELD_AXES for s in program)


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
