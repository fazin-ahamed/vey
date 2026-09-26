"""Structured / exact lane: deterministic grounding over structured candidate
facts (numbers and enums), no learned model. Used when every axis and filter in
the compiled program resolves against explicit candidate fields, so the ~150M
CRUX backbone need not run.

Fact syntax accepted in candidate text (comma/semicolon separated clauses):
  numeric:   "latency 40 ms", "cost 0.18", "quality: 0.9", "context=128000"
  enum/bool: "permitted" / "not permitted", "supports tools" / "no tool support"
"""
from __future__ import annotations

import re

_NUM_UNIT = re.compile(r"([a-z][a-z _-]*?)\s*[:=]?\s*(-?[0-9]*\.?[0-9]+)\s*(ms|s|k|m|%)?\b", re.I)
_MULT = {"k": 1e3, "m": 1e6}

_ENUM_POS = {
    "permitted": ("permitted", "allowed", "legal"),
    "tool_support": ("supports tools", "tool support", "tool-capable", "supports the required tools"),
}
_ENUM_NEG = {
    "permitted": ("not permitted", "disallowed", "prohibited", "illegal"),
    "tool_support": ("no tool", "not support", "no tools", "tools unsupported", "not tool"),
}


def parse_facts(text: str) -> dict[str, float]:
    facts: dict[str, float] = {}
    for m in _NUM_UNIT.finditer(text):
        key = m.group(1).strip().lower().replace(" ", "_").strip("_")
        if not key:
            continue
        val = float(m.group(2))
        unit = (m.group(3) or "").lower()
        if unit in _MULT:
            val *= _MULT[unit]
        facts[key] = val
    return facts


class DictGrounder:
    """Deterministic grounder. `ordinal_resolution=0` -> numeric axes compared
    exactly (no Semantic Resolution band)."""

    ordinal_resolution = 0.0

    def predicate(self, texts, concept, threshold=None):
        out = []
        for t in texts:
            low = t.lower()
            if threshold is not None:                     # numeric floor on `concept`
                facts = parse_facts(t)
                v = facts.get(concept)
                out.append((v - threshold) if v is not None else -1.0)
                continue
            pos = _ENUM_POS.get(concept, (concept,))
            neg = _ENUM_NEG.get(concept, ("not " + concept, "no " + concept))
            if any(n in low for n in neg):
                out.append(-1.0)
            elif any(p in low for p in pos):
                out.append(1.0)
            else:
                out.append(-1.0)                          # concept not asserted -> not satisfied
        return out

    def ordinal(self, texts, axis):
        key = axis.strip().lower().replace(" ", "_")
        return [parse_facts(t).get(key, 0.0) for t in texts]

    def evidence(self, texts, axis):
        key = axis.strip().lower().replace(" ", "_")
        return [1.0 if key in parse_facts(t) else 0.0 for t in texts]

    def handles(self, program, texts) -> bool:
        """True iff every stage resolves against structured facts in ALL
        candidates (numeric axes/floors present; enum filters recognised)."""
        facts = [parse_facts(t) for t in texts]
        lows = [t.lower() for t in texts]
        for st in program:
            if st.kind in ("MAX", "MIN"):
                key = st.key.strip().lower().replace(" ", "_")
                if not all(key in f for f in facts):
                    return False
            elif st.kind == "FILTER":
                if st.threshold is not None:
                    key = st.key.strip().lower().replace(" ", "_")
                    if not all(key in f for f in facts):
                        return False
                elif st.key in _ENUM_POS:
                    pass  # enum recognised
                else:
                    return False
        return True
