"""Constraint + qualitative preference with Vey 2 (CRUX lane).

Shows filtering by a categorical constraint expressed in language, then ranking
survivors on a graded qualitative axis, plus the abstention path when no
candidate satisfies the constraint.

Run:  python examples/semantic_constraints.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import vey

# 1) constraint + preference
r = vey.decide(
    question="Among options that are permitted, choose the most reliable.",
    candidates={
        "a": "the action is permitted; it is rock-solid and dependable",
        "b": "the action is permitted; it is flaky and inconsistent",
        "c": "the action is not permitted",
    },
)
print("constraint+preference ->", r.answer, "(", r.decision_mode, ")")

# 2) abstention: nothing satisfies the constraint
r2 = vey.decide(
    question="Among options that are permitted, choose the most reliable.",
    candidates={
        "a": "the action is not permitted",
        "b": "the action is prohibited",
    },
)
print("all-violate         ->", r2.answer, "(", r2.trust["state"], ")")
