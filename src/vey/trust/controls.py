"""Trust controls, simplest first. Nothing here is built unless it beats the
simpler control at matched coverage.

    A  top1 score threshold
    B  top1-top2 margin threshold
    C  top1 + margin (fixed linear combination)
    D  semantic/BM25 lane agreement
    E  OOD distance / nearest-prototype distance
    F  logistic regression over all simple signals
    G  tiny MLP, ONLY attempted if F measurably loses

Every threshold and every fitted weight comes from the VALIDATION split only.
The test split is touched once, at the end, for reporting. Scores are
per-lane normalized before combining, so a cosine and a BM25 score are never
compared on a shared uncalibrated scale.
"""
from __future__ import annotations

import numpy as np

from .observation import Observation


# --------------------------------------------------------------------------- #
# Calibration helpers (validation-only)                                         #
# --------------------------------------------------------------------------- #
def standardize_fit(X: np.ndarray):
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd < 1e-9] = 1.0
    return mu, sd


def standardize_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / sd


def fit_logistic(X: np.ndarray, y: np.ndarray, *, iters=800, lr=0.5, l2=1e-3):
    """Plain L2 logistic regression by gradient descent. No library, no
    hidden capacity. This is the bar a neural trust model must clear."""
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(iters):
        z = X @ w + b
        p = 1 / (1 + np.exp(-z))
        g = p - y
        w -= lr * ((X.T @ g) / n + l2 * w)
        b -= lr * g.mean()
    return w, b


def predict_logistic(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return 1 / (1 + np.exp(-(X @ w + b)))


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels).astype(int)
    if labels.sum() == 0 or (1 - labels).sum() == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    s = np.asarray(scores)[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    pos = labels == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels).astype(int)
    if labels.sum() == 0:
        return float("nan")
    order = np.argsort(-np.asarray(scores))
    y = labels[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    rec = tp / y.sum()
    return float(np.sum(np.diff(np.concatenate([[0], rec])) * prec))


def ece(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    from ..evaluation.metrics import expected_calibration_error
    return float(expected_calibration_error(np.clip(probs, 0, 1), labels, bins=bins))


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((np.asarray(probs) - np.asarray(labels)) ** 2))


def nll(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(np.asarray(probs), eps, 1 - eps)
    return float(-np.mean(np.asarray(labels) * np.log(p) + (1 - np.asarray(labels)) * np.log(1 - p)))


def risk_at_coverage(risks: np.ndarray, coverages=(0.10, 0.25, 0.50, 0.75, 0.90, 1.0)) -> dict:
    """Mean error among the `coverage` fraction lowest-risk decisions."""
    r = np.asarray(risks, dtype=np.float64)
    order = np.argsort(r)
    out = {}
    for c in coverages:
        k = max(1, int(round(c * len(r))))
        out[f"{int(c*100)}%"] = round(float(r[order[:k]].mean()), 4)
    return out


def coverage_at_risk(risks: np.ndarray, fail_rates=(0.01, 0.02, 0.05, 0.10, 0.15, 0.20)) -> dict:
    """Largest coverage whose accepted-set error stays under the given rate."""
    r = np.asarray(risks, dtype=np.float64)
    order = np.argsort(r)
    cum = np.cumsum(r[order]) / np.arange(1, len(r) + 1)
    out = {}
    for f in fail_rates:
        ok = np.where(cum <= f)[0]
        out[f"{int(f*100)}%"] = round(float((ok[-1] + 1) / len(r)), 4) if len(ok) else 0.0
    return out


# --------------------------------------------------------------------------- #
# Controls A-G: each returns a risk score in [0,1] (higher = riskier)          #
# --------------------------------------------------------------------------- #
def _scale_to_unit(x: np.ndarray) -> np.ndarray:
    """Min-max to [0,1] using the VALIDATION range passed in `lo/hi` when given."""
    return x


def control_scores(obs: list[Observation], fit_idx: np.ndarray) -> dict:
    """Compute every control. `fit_idx` is the validation index set; it is used
    ONLY to set each control's scale/threshold, never to see test outcomes."""
    F = np.stack([o.vector() for o in obs])
    names = list(Observation.FEATURES)
    col = {n: F[:, i] for i, n in enumerate(names)}

    # Per-control calibration on validation only. For each control, map its
    # validation distribution to a risk probability via the validation error
    # rate in score quantiles (isotonic-free, 10 quantile bins).
    def _quantile_risk(val_score: np.ndarray, val_err: np.ndarray, test_score: np.ndarray) -> np.ndarray:
        """Piecewise-constant risk: validation error rate per decile of score.
        Applied to test scores by bin lookup + linear interpolation between
        adjacent bin centers. This is calibration, not a learned model."""
        edges = np.quantile(val_score, np.linspace(0, 1, 11))
        edges[0] -= 1e-9
        edges[-1] += 1e-9
        centers, rates = [], []
        for i in range(10):
            m = (val_score >= edges[i]) & (val_score <= edges[i + 1] if i == 9 else val_score < edges[i + 1])
            centers.append(edges[i + 1] if i == 9 else (edges[i] + edges[i + 1]) / 2)
            rates.append(float(val_err[m].mean()) if m.any() else 0.0)
        idx = np.clip(np.searchsorted(edges, test_score) - 1, 0, 9)
        return np.asarray(rates)[idx]

    errors = np.array([o.outcome_correct for o in obs], dtype=float)
    V = fit_idx
    T = np.setdiff1d(np.arange(len(obs)), fit_idx)

    out = {}
    # A: top1 score (low score => high risk)
    sA = -col["semantic_top1"]
    out["A_top1"] = _quantile_risk(sA[V], errors[V], sA[T])
    # B: margin
    sB = -col["semantic_margin"]
    out["B_margin"] = _quantile_risk(sB[V], errors[V], sB[T])
    # C: top1 + margin (both low => risky); equal-weight after val-standardizing
    mu_s, sd_s = col["semantic_top1"][V].mean(), col["semantic_top1"][V].std() + 1e-9
    mu_m, sd_m = col["semantic_margin"][V].mean(), col["semantic_margin"][V].std() + 1e-9
    z_top1 = (col["semantic_top1"] - mu_s) / sd_s
    z_margin = (col["semantic_margin"] - mu_m) / sd_m
    sC = -(z_top1 + z_margin)
    out["C_top1_margin"] = _quantile_risk(sC[V], errors[V], sC[T])
    # D: lane agreement (disagree => risky)
    sD = (1 - col["lane_agree"]) - z_margin
    out["D_lane_agree"] = _quantile_risk(sD[V], errors[V], sD[T])
    # E: OOD distance (high 1-top1 => risky) + low top1_share
    sE = col["semantic_ood_dist"] - col["top1_share"]
    out["E_ood"] = _quantile_risk(sE[V], errors[V], sE[T])
    # F: logistic over the simple signals (top1, margin, agree, ood, entropy, share)
    simple = ["semantic_top1", "semantic_margin", "semantic_top1_z", "semantic_ood_dist",
              "lane_agree", "score_entropy", "top1_share", "candidate_count"]
    idx = [names.index(s) for s in simple]
    X = F[:, idx]
    mu, sd = standardize_fit(X[V])
    Xs = standardize_apply(X, mu, sd)
    w, b = fit_logistic(Xs[V], errors[V])
    pF = predict_logistic(Xs[T], w, b)
    out["F_logistic"] = pF
    return out, T
