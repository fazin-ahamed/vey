"""Shared evaluation metrics.

Calibration and selective-risk primitives used by every lane. Risk-coverage is
reported as an exact prefix curve (safest decisions first) so a narrow operating
region can never be hidden behind a coarse grid.
"""
from .metrics import (
    coverage_at_risk,
    expected_calibration_error,
    mrr_at_k,
    multiclass_brier,
    multiclass_nll,
    ndcg_at_k,
    normalize_probs,
    recall_at_k,
    reorder_agreement,
    risk_at_coverage,
    risk_coverage_curve,
)

__all__ = [
    "normalize_probs",
    "multiclass_brier",
    "multiclass_nll",
    "expected_calibration_error",
    "risk_coverage_curve",
    "risk_at_coverage",
    "coverage_at_risk",
    "reorder_agreement",
    "ndcg_at_k",
    "recall_at_k",
    "mrr_at_k",
]
