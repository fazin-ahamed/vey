"""Instruction -> typed decision program.

A clean, documented keyword compiler for the CRUX runtime. It maps a
natural-language priority instruction into an ordered list of typed stages:

    FILTER <concept>     keep candidates that satisfy a categorical predicate
    MAX <axis>           prefer more of a graded qualitative axis
    MIN <axis>           prefer less of a graded qualitative axis

Stages after the filters are the LEXICOGRAPHIC priority, in the order written
("A, then B, then C"). This is deliberately not a learned parser: the neural
part grounds evidence; the program structure is explicit and auditable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Stage:
    kind: str                        # "FILTER" | "MAX" | "MIN"
    key: str                         # concept (FILTER) or axis (MAX/MIN)
    raw: str
    threshold: float | None = None   # numeric floor for a FILTER on a numeric key

    def label(self) -> str:
        if self.threshold is not None:
            return f"{self.kind} {self.key}>={self.threshold}"
        return f"{self.kind} {self.key}"


_FILTER_CONCEPTS = {
    "permitted": ("permitted", "allowed", "is permitted", "that are permitted", "legal"),
    "tool_support": ("supports tools", "tool support", "supports the required tools", "tool-capable"),
    "quality_floor": ("quality floor", "quality >=", "meets the quality", "quality threshold",
                      "meets the quality floor", "quality at least"),
}
_MAX_WORDS = ("most", "greatest", "highest", "more", "largest", "maximize", "maximum", "prefer more", "strongest")
_MIN_WORDS = ("least", "lowest", "fewest", "smallest", "minimize", "minimum", "prefer less", "lower", "cheapest")

_STOP = {"the", "a", "an", "of", "in", "on", "to", "toward", "option", "one", "with", "and",
         "then", "prefer", "choose", "pick", "select", "among", "options", "that", "are", "it",
         "its", "expected", "required", "meets", "meet"}


def _axis_after(clause: str, words) -> str | None:
    low = clause.lower()
    for w in words:
        idx = low.find(w)
        if idx == -1:
            continue
        tail = low[idx + len(w):]
        tail = re.split(r"\bthan\b|,|;|\.", tail)[0]
        toks = [t for t in re.findall(r"[a-z][a-z-]+", tail) if t not in _STOP]
        if toks:
            return " ".join(toks[:3])
    return None


def compile_instruction(instruction: str) -> list[Stage]:
    stages: list[Stage] = []
    # split into ordered clauses
    clauses = re.split(r",?\s*then\s+|;\s*|,\s+(?=prefer|minimi|maximi|choose|pick|select)", instruction, flags=re.I)
    clauses = [c.strip() for c in clauses if c.strip()]
    if not clauses:
        clauses = [instruction]
    seen_filter = set()
    for clause in clauses:
        low = clause.lower()
        # FILTERs (may be several in one clause)
        num = re.search(r"(?:at least|>=|floor(?:\s+of)?|minimum(?:\s+of)?)\s*([0-9]*\.?[0-9]+)", low)
        thr = float(num.group(1)) if num else None
        for concept, kws in _FILTER_CONCEPTS.items():
            if concept in seen_filter:
                continue
            if any(k in low for k in kws):
                # quality_floor -> numeric FILTER on the 'quality' key when a floor is given
                if concept == "quality_floor" and thr is not None:
                    stages.append(Stage("FILTER", "quality", clause, threshold=thr))
                else:
                    stages.append(Stage("FILTER", concept, clause))
                seen_filter.add(concept)
        # generic "that are <X>" / "restrict to <X>" filter
        m = re.search(r"(?:that are|restrict to options that are|of the options that are)\s+([a-z][a-z -]+)", low)
        if m:
            concept = m.group(1).strip().split(" and ")[0].strip()
            if concept and concept not in seen_filter and concept not in ("permitted",):
                stages.append(Stage("FILTER", concept, clause)); seen_filter.add(concept)
        # ordinal MIN / MAX (MIN checked first: 'lowest'/'least' are unambiguous)
        axis = _axis_after(clause, _MIN_WORDS)
        if axis:
            stages.append(Stage("MIN", axis, clause)); continue
        axis = _axis_after(clause, _MAX_WORDS)
        if axis:
            stages.append(Stage("MAX", axis, clause)); continue
        # elided-verb continuation: "minimize latency, then cost" -> MIN cost.
        # A bare noun-phrase clause inherits the previous ordinal direction.
        prev = next((s for s in reversed(stages) if s.kind in ("MIN", "MAX")), None)
        if prev is not None:
            toks = [t for t in re.findall(r"[a-z][a-z-]+", low) if t not in _STOP]
            if toks:
                stages.append(Stage(prev.kind, " ".join(toks[:3]), clause))
    # de-duplicate consecutive identical ordinal stages, keep order
    out: list[Stage] = []
    for s in stages:
        if out and out[-1].kind == s.kind and out[-1].key == s.key:
            continue
        out.append(s)
    return out
