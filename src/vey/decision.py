"""Public decision types for the Vey 2 semantic decision runtime."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Candidate:
    """One option: an id and its consequence/description text."""
    id: str
    text: str


@dataclass
class DecisionRequest:
    question: str
    candidates: dict[str, str]
    state: str = ""
    explain: bool = False


@dataclass
class Certificate:
    """Machine-readable decision evidence: never generated reasoning text.

    schema_version : certificate schema version (a public API contract).
    decision_program : typed microcode stages executed, e.g.
        ["FILTER tool_support", "MIN latency"].
    evidence : per-candidate typed grounded values, keyed by candidate id.
    survivors : candidate ids that survived the executed program.
    """
    SCHEMA_VERSION = "crux-certificate/1"

    decision_program: list[str] = field(default_factory=list)
    evidence: dict[str, dict[str, Any]] = field(default_factory=dict)
    survivors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.SCHEMA_VERSION,
                "decision_program": self.decision_program,
                "evidence": self.evidence,
                "survivors": self.survivors}


@dataclass
class DecisionResult:
    answer: str | None
    probabilities: dict[str, float]
    decision_mode: str            # "structured" | "crux" | "abstain"
    trust: dict[str, Any]
    certificate: Certificate | None = None

    def to_dict(self) -> dict[str, Any]:
        out = {"answer": self.answer, "probabilities": self.probabilities,
               "decision_mode": self.decision_mode, "trust": self.trust}
        if self.certificate is not None:
            out["certificate"] = self.certificate.to_dict()
        return out
