"""Model routing with Vey 2 (structured lane, deterministic, no model download).

Candidates expose structured facts; the instruction states a lexicographic
policy. The router uses the structured lane (the ~150M CRUX backbone does not
run). Run:  python examples/route_selection.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json
import vey

result = vey.decide(
    question=("Use a route that supports tools and meets the quality floor 0.7. "
              "Among those, minimize expected latency, then cost."),
    candidates={
        "gpt-4-ish":   "supports tools; quality 0.92; latency 800 ms; cost 0.90",
        "sonnet-ish":  "supports tools; quality 0.88; latency 500 ms; cost 0.60",
        "haiku-ish":   "supports tools; quality 0.74; latency 200 ms; cost 0.10",
        "local-7b":    "no tool support; quality 0.60; latency 50 ms; cost 0.00",
    },
    explain=True,
)
print("answer       :", result.answer)
print("decision_mode:", result.decision_mode)   # -> structured
print("trust        :", result.trust)
print("certificate  :")
print(json.dumps(result.certificate.to_dict(), indent=2))
