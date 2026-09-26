"""Qualitative choice with Vey 2 (CRUX lane).

Candidates are natural-language consequences with no numeric fields, so the
router uses the CRUX lane: a frozen NLI predicate grounder + a trained
antisymmetric ordinal comparator + the exact microcode. The ~150M semantic
backbone loads lazily on first CRUX call.

Run:  python examples/crux_choice.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json
import vey

result = vey.decide(
    question=("Choose an option that is permitted; among permitted options prefer "
              "more room to maneuver, then stronger progress toward the objective."),
    candidates={
        "north": "the action is permitted; it leaves ample room; it advances strongly toward the objective",
        "east":  "the action is permitted; it leaves little room; it advances strongly toward the objective",
        "south": "the action is not permitted; it collides with an obstacle",
        "west":  "the action is permitted; it leaves ample room; it barely advances",
    },
    explain=True,
)
print("answer       :", result.answer)          # -> north (permitted, most room, then progress)
print("decision_mode:", result.decision_mode)   # -> crux
print("trust        :", result.trust)
print("certificate  :")
print(json.dumps(result.certificate.to_dict(), indent=2))
