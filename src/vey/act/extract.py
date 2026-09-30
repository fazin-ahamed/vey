"""Exact / span-copy slot extraction: the non-learned control and fallback.

The reviewer's rule: for numeric/date-like fields compare neural extraction
against simple exact regex/parser controls, and if regex wins, use regex. This
module IS that control. It never guesses: an unresolvable required slot returns
MISSING, which the compiler turns into ASK_FOR_INFO.

It also computes, for every slot, the three allowed outcomes:
    VALUE(span) | MISSING | AMBIGUOUS
AMBIGUOUS is raised when the state contains >1 incompatible candidate value for
the slot (e.g. two different dates, two conflicting enum candidates) so the
compiler refuses rather than picking one.
"""
from __future__ import annotations
from .compile import FieldDecision, coerce, MISSING, AMBIGUOUS, VALUE_STATE
import re
from typing import Any

from .tool_card import ArgSpec
from .compile import FieldDecision, coerce

# Slot-name -> per-type extraction patterns. Patterns are deliberately
# conservative: they anchor on the slot name appearing near the value.
_LABEL = r"(?:the\s+)?{label}\s*(?:is|=|:)?\s*"

_ENUM_HINT = re.compile(r"\b(?P<v>[a-z][a-z0-9_\- ]{1,30})\b", re.I)


def _slot_patterns(arg: ArgSpec) -> list[re.Pattern]:
    label = re.escape(arg.name.replace("_", " ").strip())
    lead = _LABEL.format(label=label)
    out: list[re.Pattern] = []
    t = arg.type
    if t in ("integer", "number"):
        out.append(re.compile(lead + r"(?P<v>[-+]?\$?\d[\d,\.]*\s*(?:%|percent|km|miles|m|kg|lbs)?)", re.I))
        out.append(re.compile(r"(?P<v>[-+]?\$?\d[\d,\.]*\s*(?:%|percent|km|miles|m|kg|lbs)?)\s*(?:of\s+)?(?:for\s+)?" + label, re.I))
    elif t == "date":
        out.append(re.compile(lead + r"(?P<v>\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", re.I))
    elif t == "datetime":
        out.append(re.compile(lead + r"(?P<v>\d{4}-\d{2}-\d{2}[T ]\d{1,2}:\d{2}(?::\d{2})?)", re.I))
    elif t in ("email", "url", "phone", "id"):
        # type-shaped search anywhere; the value type disambiguates the field
        pass
    return out


_TYPE_ANYWHERE = {
    "email": re.compile(r"(?P<v>[\w.+-]+@[\w-]+\.[\w.-]+)"),
    "url": re.compile(r"(?P<v>(?:https?://|www\.)[^\s,;]+)", re.I),
    "phone": re.compile(r"(?P<v>(?:\+\d{1,3}[\s-]?)?(?:\(\d{2,4}\)[\s-]?)?\d{3}[\s-]?\d{3,4}(?:[\s-]?\d{2,4})?)"),
    "id": re.compile(r"(?P<v>\b[A-Za-z]{0,4}[-_]?\d{3,}\b)"),
    # value-shaped search: the slot is often unlabeled ("on 2026-09-25"), so the
    # type itself is the disambiguator. These are exact shapes, not heuristics.
    "date": re.compile(r"(?P<v>\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b)"),
    "datetime": re.compile(r"(?P<v>\b\d{4}-\d{2}-\d{2}[T ]\d{1,2}:\d{2}(?::\d{2})?\b)"),
    "integer": re.compile(r"(?P<v>\b[-+]?\d[\d,]*(?=\s|$|[.,;)]))"),
    "number": re.compile(r"(?P<v>\b[-+]?\d[\d,]*(?:\.\d+)?(?=\s|$|[.,;)]))"),
    "boolean": re.compile(r"(?P<v>\b(?:true|false|yes|no|y|n)\b)", re.I),
}


def _enum_candidates(arg: ArgSpec, state: str) -> list[tuple[str, int, int]]:
    """Find enum values in the state. Each match carries a char span."""
    found: list[tuple[str, int, int]] = []
    seen: set[str] = set()
    for e in arg.enum:
        ev = str(e).replace("_", " ")
        pattern = re.compile(r"\b" + re.escape(str(e)) + r"\b", re.I) if "_" in str(e) or re.match(r"^\w+$", str(e)) else None
        if pattern is None:
            continue
        m = pattern.search(state)
        if m and ev.lower() not in seen:
            seen.add(ev.lower())
            found.append((e, m.start(), m.end()))
    return found


def extract_slot(arg: ArgSpec, state: str, *, allow_anywhere: bool = True) -> FieldDecision:
    """Resolve one slot from the state text, exactly. Returns VALUE / MISSING / AMBIGUOUS."""
    t = arg.type

    if t == "enum":
        cands = _enum_candidates(arg, state)
        if not cands:
            return FieldDecision.missing(arg.name, f"no enum value for {arg.name!r}")
        if len(cands) == 1:
            v, a, b = cands[0]
            return FieldDecision.resolved(arg.name, v, (a, b), "enum exact match")
        # multiple enum values present -> ambiguous only if they differ
        vals = {str(c[0]).lower() for c in cands}
        if len(vals) == 1:
            v, a, b = cands[0]
            return FieldDecision.resolved(arg.name, v, (a, b), "enum exact match")
        return FieldDecision.ambiguous(arg.name, [c[0] for c in cands], "multiple enum candidates in state")

    # 1) slot-label-anchored patterns win: an explicit "<slot> is <value>" is
    #    stronger evidence than a bare type-shaped token elsewhere in the state.
    labeled: list[tuple[str, int, int]] = []
    for pat in _slot_patterns(arg):
        for m in pat.finditer(state):
            labeled.append((m.group("v"), m.start(), m.end()))
            break

    def _resolve(found):
        uniq: dict[str, tuple[Any, int, int]] = {}
        for raw, a, b in found:
            val, ok = coerce(arg, raw)
            if not ok:
                continue
            key = str(val)
            if key not in uniq:
                uniq[key] = (val, a, b)
        if not uniq:
            return None
        if len(uniq) == 1:
            return FieldDecision.resolved(arg.name, next(iter(uniq.values()))[0],
                                          next(iter(uniq.values()))[1:], f"{t} exact match")
        vals = list(uniq.values())
        return FieldDecision.ambiguous(arg.name, [v[0] for v in vals],
                                       f"{len(vals)} incompatible {t} candidates")

    if labeled:
        d = _resolve(labeled)
        if d is not None and d.state != AMBIGUOUS:
            return d

    # 2) otherwise fall back to the type-shaped value search
    if t in _TYPE_ANYWHERE and allow_anywhere:
        pat = _TYPE_ANYWHERE[t]
        shaped: list[tuple[str, int, int]] = []
        for m in pat.finditer(state):
            shaped.append((m.group("v"), m.start(), m.end()))
        d = _resolve(shaped)
        if d is not None:
            return d

    return FieldDecision.missing(arg.name, f"no {t} value found for {arg.name!r}")
