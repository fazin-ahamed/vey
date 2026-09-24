"""Unified observation record for the trust / OOD school.

One record per decision. Features are strictly inference-time observable. No
gold label, no annotation-derived quantity, no post-hoc statistic. Gold outcomes
are attached later as `outcome_*` fields, never as features.

Lane-specific by construction: a cosine from the semantic matcher, a BM25 score,
a pooled softmax, and a compiler state are separate feature columns and are never
mixed into one uncalibrated scale. Calibration happens per lane downstream.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict, field

import numpy as np

_WORD = re.compile(r"\w+")


@dataclass
class Observation:
    """Inference-time observable features for one candidate/tool decision.

    Higher `*_top1` is not universally "better": only the per-lane calibrator
    knows a score's scale. `semantic_top1` is a cosine, `bm25_top1` an unbounded
    lexical score, and they are kept in separate columns on purpose.
    """
    # --- semantic matcher lane (cosine; cached candidate embeddings) ---
    semantic_top1: float
    semantic_top2: float
    semantic_margin: float
    semantic_top1_z: float          # z-score within this row's candidate set
    semantic_ood_dist: float        # 1 - top1: distance to the nearest prototype

    # --- lexical lane (unbounded BM25 score) ---
    bm25_top1: float
    bm25_top2: float
    bm25_margin: float

    # --- cross-lane evidence (the FreshStack complementarity signal) ---
    lane_agree: float                # 1 if semantic top1 == bm25 top1 else 0
    lane_disagree_margin: float      # margin when the lanes disagree, else top1 margin

    # --- candidate-set shape ---
    candidate_count: int
    score_entropy: float             # normalized entropy of the softmax(scores/τ)
    top1_share: float                # softmax mass on the top candidate

    # --- exact / schema state (deterministic, not learned) ---
    n_required_missing: int
    n_ambiguous: int
    schema_complete: float           # 1 if every required slot is resolved
    n_args_emitted: int

    # --- pooled lane (fixed-label only; NaN elsewhere) ---
    pooled_confidence: float = float("nan")

    # --- outcomes, attached AFTER prediction (never features) ---
    outcome_correct: int = 0
    outcome_should_abstain: int = 0
    outcome_kind: str = ""           # answerable | ambiguous | ood | insufficient

    FEATURES = (
        "semantic_top1", "semantic_top2", "semantic_margin", "semantic_top1_z",
        "semantic_ood_dist", "bm25_top1", "bm25_top2", "bm25_margin",
        "lane_agree", "lane_disagree_margin", "candidate_count", "score_entropy",
        "top1_share", "n_required_missing", "n_ambiguous", "schema_complete",
        "n_args_emitted", "pooled_confidence",
    )

    def vector(self) -> np.ndarray:
        return np.array([float(getattr(self, f)) for f in self.FEATURES], dtype=np.float64)

    def to_dict(self) -> dict:
        return asdict(self)


def from_scores(semantic: np.ndarray, bm25: np.ndarray, *, tau: float = 0.05,
                n_required_missing: int = 0, n_ambiguous: int = 0,
                n_args_emitted: int = 0, pooled_confidence: float = float("nan")) -> Observation:
    """Build one observation from a candidate row's score vectors.

    `semantic` and `bm25` are aligned to the same candidate order. All derived
    features are computed from those two arrays only.
    """
    sem = np.asarray(semantic, dtype=np.float64)
    lex = np.asarray(bm25, dtype=np.float64)
    n = len(sem)
    s_order = np.argsort(-sem)
    l_order = np.argsort(-lex)
    s1 = float(sem[s_order[0]])
    s2 = float(sem[s_order[1]]) if n > 1 else s1
    l1 = float(lex[l_order[0]])
    l2 = float(lex[l_order[1]]) if n > 1 else l1
    margin = s1 - s2

    # z-score within the row: how far the winner stands out in ITS OWN set
    if n > 1 and sem.std() > 1e-12:
        z = float((s1 - sem.mean()) / sem.std())
    else:
        z = 0.0

    # softmax mass / entropy over the row (τ-scaled; a shape statistic, not a
    # calibrated probability)
    zt = sem / max(tau, 1e-6)
    zt = zt - zt.max()
    p = np.exp(zt)
    p = p / max(p.sum(), 1e-12)
    top1_share = float(p[0])
    nz = p[p > 0]
    ent = float(-(nz * np.log(nz)).sum() / np.log(max(2, n))) if n > 1 else 0.0

    agree = 1.0 if int(s_order[0]) == int(l_order[0]) else 0.0
    return Observation(
        semantic_top1=s1, semantic_top2=s2, semantic_margin=margin,
        semantic_top1_z=z, semantic_ood_dist=float(1.0 - s1),
        bm25_top1=l1, bm25_top2=l2, bm25_margin=float(l1 - l2),
        lane_agree=agree, lane_disagree_margin=float(margin if agree < 0.5 else s1),
        candidate_count=int(n), score_entropy=ent, top1_share=top1_share,
        n_required_missing=int(n_required_missing), n_ambiguous=int(n_ambiguous),
        schema_complete=float(n_required_missing == 0 and n_ambiguous == 0),
        n_args_emitted=int(n_args_emitted), pooled_confidence=pooled_confidence,
    )


def matrix(obs: list[Observation]) -> np.ndarray:
    return np.stack([o.vector() for o in obs]) if obs else np.zeros((0, len(Observation.FEATURES)))
