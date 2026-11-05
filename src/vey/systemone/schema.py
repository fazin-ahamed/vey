"""Request validation for the System One wire format, and text helpers.

The shapes follow TypeSafe's published OpenAPI for ``POST /v1/systemone``
(Jev), which Clef and Laya's server also accept:

    {"model": str, "state": str | object | array,
     "questions": {name: {"type": "noul" | "choice" | "score" | "box", ...}},
     "assets": {placeholder: {"type": "image", ...}}}      # optional

Errors use the same two bodies: ``{"detail": ...}`` with status 400 for a
request that is valid but cannot be served, and FastAPI-style
``{"detail": [{"loc", "msg", "type", ...}]}`` with status 422 for a request that
does not match the schema.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

QUESTION_TYPES = ("noul", "choice", "score", "box")
MAX_CHOICES = 255
MAX_SCORE_LEVELS = 10
MAX_QUESTIONS = 64


class ApiError(Exception):
    """An error with an HTTP status and a JSON body, in the wire format's shape."""

    def __init__(self, status: int, body: dict[str, Any]):
        super().__init__(str(body))
        self.status = status
        self.body = body


def bad_request(message: str) -> ApiError:
    return ApiError(400, {"detail": message})


class _Collector:
    def __init__(self) -> None:
        self.errors: list[dict[str, Any]] = []

    def add(self, loc: list, msg: str, type_: str, value: Any = None) -> None:
        err: dict[str, Any] = {"loc": ["body", *loc], "msg": msg, "type": type_}
        if value is not None:
            preview = json.dumps(value, default=str)
            err["input"] = preview if len(preview) <= 120 else preview[:119] + "…"
        self.errors.append(err)

    def raise_if_any(self) -> None:
        if self.errors:
            raise ApiError(422, {"detail": self.errors})


def to_text(value: Any) -> str:
    """Flatten a string / object / array / scalar to plain text for the encoders.

    Objects become ``key: value`` clauses joined by ``; ``; arrays join their
    items with ``; ``. ``None`` is empty text.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (bool, int, float)):
        return json.dumps(value)
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            t = to_text(v)
            parts.append(f"{k}: {t}" if t else str(k))
        return "; ".join(parts)
    if isinstance(value, (list, tuple)):
        return "; ".join(t for t in (to_text(v) for v in value) if t)
    return str(value)


def state_to_text(state: Any) -> str:
    """The state as text, one ``key: value`` line per field for objects."""
    if isinstance(state, dict):
        lines = []
        for k, v in state.items():
            t = to_text(v)
            lines.append(f"{k}: {t}" if t else str(k))
        return "\n".join(lines)
    if isinstance(state, (list, tuple)):
        return "\n".join(t for t in (to_text(v) for v in state) if t)
    return to_text(state)


def readable(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").strip()


@dataclass
class Question:
    name: str
    type: str
    instructions: Any = None
    criteria: Any = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def instruction_text(self) -> str:
        """The instructions as text; the question name stands in when omitted."""
        return to_text(self.instructions) or readable(self.name)

    @property
    def options(self) -> list[tuple[str, Any]]:
        """``choice``: (name, description) in request order. ``score``: ("0", level), ..."""
        if self.type == "choice":
            return list(self.criteria.items())
        if self.type == "score":
            return [(str(i), c) for i, c in enumerate(self.criteria)]
        return []

    def option_text(self, key: str, description: Any) -> str:
        """Text that represents one option to the encoder: ``name: description``."""
        d = to_text(description)
        if self.type == "choice":
            return f"{readable(key)}: {d}" if d else readable(key)
        return d or f"level {key}"


@dataclass
class Request:
    model: str
    state: Any
    questions: list[Question]
    assets: dict[str, Any] | None = None

    @property
    def state_text(self) -> str:
        return state_to_text(self.state)


def parse_request(body: Any) -> Request:
    """Validate ``body`` against the System One schema, collecting every error."""
    c = _Collector()
    if not isinstance(body, dict):
        c.add([], "Input should be a valid dictionary or object to extract fields from",
              "model_attributes_type", body)
        c.raise_if_any()
    model = body.get("model")
    if "model" not in body:
        c.add(["model"], "Field required", "missing")
    elif not isinstance(model, str):
        c.add(["model"], "Input should be a valid string", "string_type", model)
    if "state" not in body:
        c.add(["state"], "Field required", "missing")
    elif not isinstance(body["state"], (str, dict, list)):
        c.add(["state"], "Input should be a valid string, object or array", "state_type", body["state"])
    qs_raw = body.get("questions")
    questions: list[Question] = []
    if "questions" not in body:
        c.add(["questions"], "Field required", "missing")
    elif not isinstance(qs_raw, dict):
        c.add(["questions"], "Input should be a valid dictionary", "dict_type", qs_raw)
    elif len(qs_raw) < 1:
        c.add(["questions"], "Dictionary should have at least 1 item after validation, not 0", "too_short")
    elif len(qs_raw) > MAX_QUESTIONS:
        c.add(["questions"], f"Dictionary should have at most {MAX_QUESTIONS} items after validation",
              "too_long")
    else:
        for name, q in qs_raw.items():
            questions.append(_parse_question(c, str(name), q))
    assets = body.get("assets")
    if assets is not None and not isinstance(assets, dict):
        c.add(["assets"], "Input should be a valid dictionary", "dict_type", assets)
        assets = None
    c.raise_if_any()
    return Request(model=model, state=body["state"], questions=[q for q in questions if q], assets=assets or None)


def _parse_question(c: _Collector, name: str, q: Any) -> Question | None:
    loc = ["questions", name]
    if not isinstance(q, dict):
        c.add(loc, "Input should be a valid dictionary or object to extract fields from",
              "model_attributes_type", q)
        return None
    t = q.get("type")
    if t not in QUESTION_TYPES:
        c.add([*loc, "type"], "Input tag '%s' found using 'type' does not match any of the expected tags: "
              "'noul', 'choice', 'score', 'box'" % t, "union_tag_invalid" if "type" in q else "union_tag_not_found",
              q)
        return None
    crit = q.get("criteria")
    if t == "choice":
        if not isinstance(crit, dict):
            c.add([*loc, "criteria"], "Field required" if crit is None else "Input should be a valid dictionary",
                  "missing" if crit is None else "dict_type", crit)
            return None
        if not 1 <= len(crit) <= MAX_CHOICES:
            c.add([*loc, "criteria"], f"Dictionary should have between 1 and {MAX_CHOICES} items",
                  "too_short" if not crit else "too_long")
            return None
    elif t == "score":
        if not isinstance(crit, list):
            c.add([*loc, "criteria"], "Field required" if crit is None else "Input should be a valid list",
                  "missing" if crit is None else "list_type", crit)
            return None
        if not 1 <= len(crit) <= MAX_SCORE_LEVELS:
            c.add([*loc, "criteria"], f"List should have between 1 and {MAX_SCORE_LEVELS} items",
                  "too_short" if not crit else "too_long")
            return None
    elif t == "noul" and crit is not None:
        if not isinstance(crit, dict) or set(crit) - {"true", "false"}:
            c.add([*loc, "criteria"], "criteria for a noul question may only contain 'true' and 'false'",
                  "noul_criteria", crit)
            return None
    return Question(name=name, type=t, instructions=q.get("instructions"), criteria=crit, raw=q)


__all__ = ["ApiError", "bad_request", "Question", "Request", "parse_request", "to_text",
           "state_to_text", "readable", "QUESTION_TYPES", "MAX_CHOICES", "MAX_SCORE_LEVELS", "MAX_QUESTIONS"]
