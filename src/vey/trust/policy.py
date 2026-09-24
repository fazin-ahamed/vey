"""Deterministic action policy.

The learned/calibrated component estimates RISK. The policy stays explicit and
deterministic: a neural model never directly emits ACT/ASK/ABSTAIN/ESCALATE,
because exact conditions already determine part of the decision:

    if required_field_missing or ambiguous:   ASK_FOR_INFO   (exact schema rule)
    elif hard_constraint_failed:              REJECT         (exact schema rule)
    elif estimated_risk <= act_threshold:     ACT
    elif cheap_lane_may_resolve:              RERUN_CHEAP_LANE
    elif risk >= escalate_threshold:          ESCALATE
    else:                                     ABSTAIN
"""
from __future__ import annotations

from dataclasses import dataclass

from .observation import Observation

ACT = "ACT"
ASK_FOR_INFO = "ASK_FOR_INFO"
REJECT = "REJECT"
ESCALATE = "ESCALATE"
ABSTAIN = "ABSTAIN"
# A request to run the other cheap lane, NOT a decision. Emitting ACT here
# would let a high-risk decision through, which is the exact failure the trust
# phase exists to prevent.
RERUN_CHEAP_LANE = "RERUN_CHEAP_LANE"


@dataclass
class PolicyConfig:
    act_threshold: float = 0.10          # estimated risk <= this => ACT
    escalate_threshold: float = 0.60     # risk >= this, escalation available => ESCALATE
    disagreement_threshold: float = 0.15  # cheap-lane disagreement margin warranting a rerun
    escalation_available: bool = True
    allow_cheap_rerun: bool = True


def decide(obs: Observation, risk: float, cfg: PolicyConfig) -> dict:
    """One decision. Exact conditions first; calibrated risk only gates the rest."""
    # 1) exact schema states are deterministic, never learned
    if obs.n_required_missing > 0:
        return {"decision": ASK_FOR_INFO,
                "reason": f"missing required slots ({obs.n_required_missing})"}
    if obs.n_ambiguous > 0:
        return {"decision": ASK_FOR_INFO,
                "reason": f"ambiguous slots ({obs.n_ambiguous})"}
    if obs.schema_complete < 1.0:
        return {"decision": ASK_FOR_INFO, "reason": "schema incomplete"}

    # 2) calibrated risk gates the uncertain part
    if risk <= cfg.act_threshold:
        return {"decision": ACT, "reason": f"risk {risk:.3f} <= {cfg.act_threshold}"}

    # 3) lane disagreement is a rerun request, never an ACT
    if (cfg.allow_cheap_rerun and obs.lane_agree < 0.5
            and obs.semantic_margin < cfg.disagreement_threshold
            and obs.candidate_count > 1):
        return {"decision": RERUN_CHEAP_LANE, "reason": "lane disagreement -> cheap rerun"}

    if risk >= cfg.escalate_threshold and cfg.escalation_available:
        return {"decision": ESCALATE,
                "reason": f"risk {risk:.3f} >= {cfg.escalate_threshold}"}
    return {"decision": ABSTAIN, "reason": f"risk {risk:.3f} in abstain band"}
