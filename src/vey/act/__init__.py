"""Typed action compiler.

Vey never generates tool-call JSON. A tool is a typed :class:`ToolCard`; a call
is compiled from per-slot :class:`FieldDecision` values and validated exactly.
A required slot that is not resolved becomes ``ASK_FOR_INFO`` rather than an
invented value, which is what keeps the unsupported-argument rate at zero.
"""
from .tool_card import ArgSpec, ToolCard
from .compile import (
    AMBIGUOUS,
    MISSING,
    VALUE_STATE,
    CompiledCall,
    FieldDecision,
    coerce,
    compile_call,
    is_schema_valid,
)

__all__ = [
    "ToolCard",
    "ArgSpec",
    "FieldDecision",
    "CompiledCall",
    "compile_call",
    "is_schema_valid",
    "coerce",
    "VALUE_STATE",
    "MISSING",
    "AMBIGUOUS",
]
