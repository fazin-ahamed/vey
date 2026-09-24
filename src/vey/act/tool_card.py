"""ToolCard: the canonical, model-visible description of a callable tool.

Vey never generates tool-call JSON. A tool is represented as a typed card; a
call is *compiled* from typed field decisions and validated exactly. This module
owns the schema: ToolCard in, validated call or ASK_FOR_INFO out.

A ToolCard is deliberately small and canonical. Everything the matcher sees is
derived from `card_text()` alone, so tool identity, ordering, and candidate
count never enter the representation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Argument type routing. These are the only types the compiler accepts; an
# unknown type is a schema error, never a best-effort guess.
ARG_TYPES = (
    "string",      # free text -> span-copy extraction
    "enum",        # candidate matching over a closed value set
    "boolean",     # exact lexical rule / BOOL head
    "integer",     # span extraction + exact parser
    "number",      # span extraction + exact parser
    "date",        # extraction + normalized date parser
    "datetime",    # extraction + normalized datetime parser
    "email",       # exact/span extractor + validator
    "phone",       # exact/span extractor + validator
    "url",         # exact/span extractor + validator
    "id",          # exact/span extractor + validator
    "object",      # nested -> recursively compiled, never generated
    "array",       # list of scalars
)


@dataclass(frozen=True)
class ArgSpec:
    """One argument slot in a tool's schema."""
    name: str
    type: str
    required: bool = True
    description: str = ""
    enum: tuple[str, ...] = ()          # only for type == "enum"
    fields: tuple["ArgSpec", ...] = ()  # only for type == "object" (nested schema)
    item_type: str = ""                 # only for type == "array"

    def __post_init__(self):
        if self.type not in ARG_TYPES:
            raise ValueError(f"unsupported arg type {self.type!r} for {self.name!r}")
        if self.type == "enum" and not self.enum:
            raise ValueError(f"enum arg {self.name!r} has no values")
        if self.type == "array" and self.item_type not in ARG_TYPES:
            raise ValueError(f"array arg {self.name!r} needs a scalar item_type")
        if self.type == "object" and not self.fields:
            raise ValueError(f"object arg {self.name!r} has no nested fields")

    def required_names(self) -> tuple[str, ...]:
        """Required leaf slots, recursively. Object fields flatten by name path."""
        if self.type == "object":
            out = []
            for f in self.fields:
                out.extend(f.required_names())
            return tuple(out)
        return (self.name,) if self.required else ()


@dataclass(frozen=True)
class ToolCard:
    """Canonical tool description. `args` is the full schema; the matcher only
    ever sees `card_text()`."""
    tool_id: str
    name: str
    description: str
    args: tuple[ArgSpec, ...] = ()
    family: str = ""  # optional grouping used only for NOVEL-split construction

    def arg(self, name: str) -> ArgSpec | None:
        for a in self.args:
            if a.name == name:
                return a
        return None

    def required_args(self) -> tuple[ArgSpec, ...]:
        return tuple(a for a in self.args if a.required)

    def required_paths(self) -> tuple[str, ...]:
        """Every required leaf path, e.g. ('origin', 'passengers.leg') style is
        not needed yet; objects flatten to their own required field names."""
        out: list[str] = []
        for a in self.args:
            if a.type == "object":
                for f in a.required_names():
                    out.append(f"{a.name}.{f}")
            elif a.required:
                out.append(a.name)
        return tuple(out)

    def card_text(self) -> str:
        """The single canonical string the frozen encoder embeds.

        One function, one truth: every representation of this tool in the
        matcher goes through here, so no caller can leak a second, subtly
        different view of the same tool.
        """
        parts = [self.name.replace("_", " "), self.description.strip()]
        if self.args:
            req = ", ".join(a.name for a in self.args if a.required) or "none"
            parts.append(f"required: {req}")
            for a in self.args:
                t = a.enum and f"enum({','.join(a.enum)})" or a.type
                parts.append(f"{a.name}: {t}")
        return "; ".join(p for p in parts if p)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_id": self.tool_id, "name": self.name,
            "description": self.description, "family": self.family,
            "args": [
                {"name": a.name, "type": a.type, "required": a.required,
                 "description": a.description, "enum": list(a.enum),
                 "fields": [
                     {"name": f.name, "type": f.type, "required": f.required,
                      "description": f.description, "enum": list(f.enum)}
                     for f in a.fields],
                 "item_type": a.item_type}
                for a in self.args
            ],
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ToolCard":
        def arg(a: dict[str, Any]) -> ArgSpec:
            return ArgSpec(
                name=a["name"], type=a["type"], required=bool(a.get("required", True)),
                description=a.get("description", ""), enum=tuple(a.get("enum") or ()),
                fields=tuple(arg(f) for f in (a.get("fields") or ())),
                item_type=a.get("item_type", ""),
            )
        return ToolCard(
            tool_id=d["tool_id"], name=d["name"], description=d.get("description", ""),
            family=d.get("family", ""),
            args=tuple(arg(a) for a in (d.get("args") or ())),
        )
