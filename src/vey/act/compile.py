"""Typed field values and the exact validator/compiler.

Every field decision is one of exactly three states. This is the mechanism that
removes an entire class of tool hallucination:

    VALUE(v)     a resolved, type-validated value
    MISSING       no supported evidence in state/context
    AMBIGUOUS     >1 incompatible candidate values

The compiler never invents a value. If a required field is not VALUE, the call
cannot be emitted; it becomes ASK_FOR_INFO. The schema supplies the structure;
Vey only decides the values.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from .tool_card import ArgSpec, ToolCard

MISSING = "MISSING"
AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True)
class FieldDecision:
    """One resolved slot. Exactly one of value/span is set when state == VALUE."""
    name: str
    state: str                      # VALUE | MISSING | AMBIGUOUS
    value: Any = None               # already normalized by coerce()
    span: tuple[int, int] | None = None   # char span in the state text, for evidence_refs
    candidates: tuple[Any, ...] = ()     # populated when AMBIGUOUS
    reason: str = ""

    @staticmethod
    def resolved(name, value, span=None, reason=""):
        return FieldDecision(name, VALUE_STATE, value, span, (), reason)

    @staticmethod
    def missing(name, reason="no supported evidence"):
        return FieldDecision(name, MISSING, None, None, (), reason)

    @staticmethod
    def ambiguous(name, candidates, reason="multiple incompatible candidates"):
        return FieldDecision(name, AMBIGUOUS, None, None, tuple(candidates), reason)


VALUE_STATE = "VALUE"


# --------------------------------------------------------------------------- #
# Exact coercers. Each returns (value, ok). No fuzzy fallbacks.               #
# --------------------------------------------------------------------------- #
_NUM = re.compile(r"[-+]?\d{1,3}(?:[ ,]\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_URL = re.compile(r"(?:https?://|www\.)[^\s,;]+", re.I)
_PHONE = re.compile(r"(?:\+\d{1,3}[\s-]?)?(?:\(\d{2,4}\)[\s-]?)?\d{3}[\s-]?\d{3,4}(?:[\s-]?\d{2,4})?")
_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b")
_DATETIME = re.compile(r"\b(\d{4}-\d{2}-\d{2})[T ](\d{1,2}:\d{2}(?::\d{2})?)\b")
_ID = re.compile(r"\b[A-Za-z]{0,4}[-_]?\d{3,}\b")


def coerce(arg: ArgSpec, raw: Any) -> tuple[Any, bool]:
    """Normalize one raw extracted value against its declared type.

    Returns (value, ok). `ok=False` means the raw text is not a valid instance of
    the declared type, so the caller must treat that field as not resolved, never
    as a best-effort guess.
    """
    if raw is None:
        return None, False
    t = arg.type
    s = str(raw).strip() if not isinstance(raw, (int, float, bool)) else raw

    if t == "boolean":
        if isinstance(s, bool):
            return s, True
        sl = str(s).strip().lower()
        if sl in ("true", "yes", "y", "1", "on"):
            return True, True
        if sl in ("false", "no", "n", "0", "off"):
            return False, True
        return None, False

    if t == "integer":
        if isinstance(s, bool):
            return None, False
        m = _NUM.search(str(s))
        if not m:
            return None, False
        try:
            f = float(m.group(0).replace(",", "").replace(" ", ""))
        except ValueError:
            return None, False
        if f != int(f):
            return None, False
        return int(f), True

    if t == "number":
        if isinstance(s, bool):
            return None, False
        m = _NUM.search(str(s))
        if not m:
            return None, False
        try:
            return float(m.group(0).replace(",", "").replace(" ", "")), True
        except ValueError:
            return None, False

    if t == "date":
        if isinstance(s, (date, datetime)):
            return (s.date() if isinstance(s, datetime) else s).isoformat(), True
        m = _DATE.search(str(s))
        if not m:
            return None, False
        return _normalize_date(m.group(1)), (_normalize_date(m.group(1)) is not None)

    if t == "datetime":
        if isinstance(s, datetime):
            return s.isoformat(timespec="seconds"), True
        m = _DATETIME.search(str(s))
        if m:
            d = _normalize_date(m.group(1))
            t2 = m.group(2)
            parts = [int(x) for x in t2.split(":")]
            while len(parts) < 3:
                parts.append(0)
            if d is None or parts[0] > 23 or parts[1] > 59 or parts[2] > 59:
                return None, False
            return f"{d}T{parts[0]:02d}:{parts[1]:02d}:{parts[2]:02d}", True
        return None, False

    if t == "email":
        m = _EMAIL.search(str(s))
        return (m.group(0), True) if m else (None, False)

    if t == "url":
        m = _URL.search(str(s))
        if not m:
            return None, False
        v = m.group(0).rstrip(".,;)")
        return (v if v.lower().startswith("http") else "https://" + v), True

    if t == "phone":
        m = _PHONE.search(str(s))
        return (m.group(0).strip(), True) if m else (None, False)

    if t == "id":
        m = _ID.search(str(s))
        return (m.group(0), True) if m else (None, False)

    if t == "enum":
        sl = str(s).strip().lower()
        for e in arg.enum:
            if str(e).strip().lower() == sl:
                return e, True
        return None, False

    if t in ("string", "object", "array"):
        return (s, True) if str(s).strip() else (None, False)

    return None, False


def _normalize_date(raw: str) -> str | None:
    raw = raw.strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y", "%d/%m/%y", "%m/%d/%y"):
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    return None


# --------------------------------------------------------------------------- #
# The compiler                                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class CompiledCall:
    """Result of compiling a selected tool + field decisions.

    This is a DECISION OBJECT, not an executed action.
    """
    outcome: str                       # ACT | ASK_FOR_INFO | REJECT
    tool_id: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    missing: tuple[str, ...] = ()
    ambiguous: tuple[str, ...] = ()
    evidence_refs: tuple[tuple[str, int, int], ...] = ()
    reasons: tuple[str, ...] = ()

    @property
    def is_actionable(self) -> bool:
        return self.outcome == "ACT"


def compile_call(card: ToolCard, decisions: dict[str, FieldDecision], *,
                 extra_args: tuple[str, ...] = ()) -> CompiledCall:
    """Exact schema compiler. Emits ACT only when every gate passes:
      selected tool exists AND all required args resolved AND types validate
      AND enum constraints validate AND no unsupported args exist.
    Otherwise ASK_FOR_INFO (missing/ambiguous required) or REJECT (schema error).
    """
    missing: list[str] = []
    ambiguous: list[str] = []
    reasons: list[str] = []
    args: dict[str, Any] = {}
    evidence: list[tuple[str, int, int]] = []

    for a in card.args:
        d = decisions.get(a.name)
        if d is None or d.state == MISSING:
            if a.required:
                missing.append(a.name)
                reasons.append(f"required arg {a.name!r} unresolved")
            continue
        if d.state == AMBIGUOUS:
            if a.required:
                ambiguous.append(a.name)
                reasons.append(f"required arg {a.name!r} ambiguous")
            continue
        value, ok = coerce(a, d.value)
        if not ok:
            if a.required:
                missing.append(a.name)
                reasons.append(f"required arg {a.name!r} failed {a.type} validation")
            continue
        if a.type == "object":
            # nested structure: the schema provides the shape, we only fill values
            sub = d.value if isinstance(d.value, dict) else {}
            nested, nmiss, namb, nreasons = _compile_object(a, sub)
            args[a.name] = nested
            missing.extend(f"{a.name}.{m}" for m in nmiss)
            ambiguous.extend(f"{a.name}.{m}" for m in namb)
            reasons.extend(nreasons)
        elif a.type == "array":
            items = d.value if isinstance(d.value, (list, tuple)) else ([d.value] if d.value is not None else [])
            item_spec = ArgSpec(name=f"{a.name}[]", type=a.item_type, required=True)
            out_items = []
            for it in items:
                v, ok2 = coerce(item_spec, it)
                if not ok2:
                    reasons.append(f"array arg {a.name!r} has an invalid {a.item_type} item")
                    continue
                out_items.append(v)
            args[a.name] = out_items
        else:
            args[a.name] = value
        if d.span is not None:
            evidence.append((a.name, d.span[0], d.span[1]))

    unsupported = [k for k in extra_args if k not in {a.name for a in card.args}]
    if unsupported:
        return CompiledCall(
            outcome="REJECT", tool_id=card.tool_id,
            reasons=tuple(reasons + [f"unsupported args: {sorted(unsupported)}"]))
    if missing or ambiguous:
        return CompiledCall(
            outcome="ASK_FOR_INFO", tool_id=card.tool_id, args=args,
            missing=tuple(missing), ambiguous=tuple(ambiguous),
            evidence_refs=tuple(evidence), reasons=tuple(reasons))
    return CompiledCall(
        outcome="ACT", tool_id=card.tool_id, args=args,
        evidence_refs=tuple(evidence), reasons=tuple(reasons))


def _compile_object(a: ArgSpec, values: dict) -> tuple[dict, list[str], list[str], list[str]]:
    out, missing, ambiguous, reasons = {}, [], [], []
    for f in a.fields:
        v = values.get(f.name)
        if v is None or (isinstance(v, str) and not v.strip()):
            if f.required:
                missing.append(f.name)
                reasons.append(f"required nested {a.name}.{f.name!r} unresolved")
            continue
        nv, ok = coerce(f, v)
        if not ok:
            if f.required:
                missing.append(f.name)
                reasons.append(f"nested {a.name}.{f.name!r} failed {f.type} validation")
            continue
        out[f.name] = nv
    return out, missing, ambiguous, reasons


def is_schema_valid(card: ToolCard, args: dict[str, Any]) -> tuple[bool, list[str]]:
    """Pure schema check on a finished args dict (used by the benchmark, not the
    compiler). No defaulting, no coercion: the args must already be valid."""
    errs: list[str] = []
    declared = {a.name: a for a in card.args}
    for k in args:
        if k not in declared:
            errs.append(f"unsupported arg {k!r}")
    for a in card.args:
        if a.name not in args:
            if a.required:
                errs.append(f"missing required arg {a.name!r}")
            continue
        v = args[a.name]
        if a.type == "object":
            if not isinstance(v, dict):
                errs.append(f"arg {a.name!r} must be object")
                continue
            nested = {f.name: f for f in a.fields}
            for k in v:
                if k not in nested:
                    errs.append(f"unsupported nested {a.name}.{k!r}")
            for f in a.fields:
                if f.name not in v:
                    if f.required:
                        errs.append(f"missing required nested {a.name}.{f.name!r}")
                else:
                    _n, ok = coerce(f, v[f.name])
                    if not ok:
                        errs.append(f"nested {a.name}.{f.name!r} invalid {f.type}")
        elif a.type == "array":
            if not isinstance(v, (list, tuple)):
                errs.append(f"arg {a.name!r} must be array")
                continue
            item_spec = ArgSpec(name=f"{a.name}[]", type=a.item_type)
            for it in v:
                _n, ok = coerce(item_spec, it)
                if not ok:
                    errs.append(f"array {a.name!r} has invalid {a.item_type} item")
                    break
        else:
            _n, ok = coerce(a, v)
            if not ok:
                errs.append(f"arg {a.name!r} invalid {a.type}")
    return (not errs), errs
