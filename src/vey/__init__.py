"""Vey: a lightweight machine-native decision runtime.

Fast, typed decisions without autoregressive generation. Vey composes narrow,
verifiable mechanisms rather than one learned monolith:

    structured/   exact candidate decisions over typed state (the R0.5 lane)
    semantic/     a frozen 24M sentence encoder; pooled head for fixed labels,
                  cosine matcher for open/novel candidates
    retrieval/    BM25 lexical lane, semantic lane, and reciprocal-rank fusion
    act/          ToolCard schema, typed slot extraction, exact action compiler
    trust/        observation record, calibrated risk controls, action policy
    evaluation/   metrics shared by the lanes (calibration, risk-coverage)

Every lane is replaceable and independently measured. See ``docs/LIMITATIONS.md``
for what Vey does not do.

Quick start::

    from vey import ToolCard, ArgSpec, compile_call, FieldDecision
    from vey.act.extract import extract_slot

    card = ToolCard(
        tool_id="book_flight", name="book_flight",
        description="Book a flight",
        args=(ArgSpec("destination", "string", True, "arrival city"),),
    )
    decision = extract_slot(card.arg("destination"), "Book me a flight to London.")
    call = compile_call(card, {"destination": decision})
    call.outcome  # 'ACT' only when every required slot is resolved and valid
"""

__version__ = "1.0.0"

from .act import (
    ArgSpec,
    CompiledCall,
    FieldDecision,
    ToolCard,
    compile_call,
    coerce,
    is_schema_valid,
    AMBIGUOUS,
    MISSING,
    VALUE_STATE,
)
from .act.extract import extract_slot
from .trust import Observation, PolicyConfig, decide
from .trust.observation import from_scores

__all__ = [
    "__version__",
    # action compiler
    "ToolCard",
    "ArgSpec",
    "FieldDecision",
    "CompiledCall",
    "compile_call",
    "is_schema_valid",
    "coerce",
    "extract_slot",
    "VALUE_STATE",
    "MISSING",
    "AMBIGUOUS",
    # trust
    "Observation",
    "from_scores",
    "PolicyConfig",
    "decide",
]
