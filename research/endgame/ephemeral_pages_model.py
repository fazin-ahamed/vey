"""ECA-1 trainable readers and equal-weight masked objectives.

Inputs are cached FP32 features.  The neural readers never inspect DecisionIR,
metadata, identifiers, grades-as-text, or exact blocks; target arrays are
separate from their feature inputs.
"""
from __future__ import annotations

from typing import NamedTuple
import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F


WIDTH = 384
RANK = 64
CROSS_HIDDEN = 128


class ReaderOutput(NamedTuple):
    score: torch.Tensor
    known_logits: torch.Tensor
    relevance_logits: torch.Tensor
    attention: torch.Tensor
    direction: torch.Tensor
    raw_value: torch.Tensor


def _check_pages(features: torch.Tensor, page_mask: torch.Tensor, name: str) -> None:
    if features.ndim != 3 or features.shape[-1] != WIDTH:
        raise ValueError(f"{name} must have shape [N,P,{WIDTH}]")
    if features.shape[1] < 1:
        raise ValueError(f"{name} must include at least one padded page slot")
    if page_mask.shape != features.shape[:2]:
        raise ValueError("page_mask must have shape [N,P]")
    if not features.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if page_mask.dtype != torch.bool:
        raise TypeError("page_mask must be bool")


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, uniform: bool) -> torch.Tensor:
    valid = mask.to(torch.bool)
    counts = valid.sum(dim=-1, keepdim=True)
    if uniform:
        return valid.to(logits.dtype) / counts.clamp_min(1).to(logits.dtype)
    safe_logits = logits.masked_fill(~valid, -torch.inf)
    # Softmax(all -inf) is NaN.  Empty candidates have exactly zero attention.
    empty = counts == 0
    safe_logits = torch.where(empty, torch.zeros_like(safe_logits), safe_logits)
    weights = torch.softmax(safe_logits, dim=-1) * valid.to(logits.dtype)
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)


def _knownness(
    relevance_logits: torch.Tensor,
    attention: torch.Tensor,
    raw_value: torch.Tensor,
    page_mask: torch.Tensor,
    bk: torch.Tensor,
    uk: torch.Tensor,
    vk: torch.Tensor,
) -> torch.Tensor:
    valid = page_mask.to(torch.bool)
    has_pages = valid.any(dim=-1)
    masked_relevance = relevance_logits.masked_fill(~valid, -torch.inf)
    max_relevance = masked_relevance.max(dim=-1).values
    max_relevance = torch.where(has_pages, max_relevance, torch.zeros_like(max_relevance))
    mean_value = (attention * raw_value).sum(dim=-1, keepdim=True)
    variance = (attention * (raw_value - mean_value).square()).sum(dim=-1)
    logits = bk + F.softplus(uk) * max_relevance - F.softplus(vk) * variance
    # No evidence is explicitly unknown.  This sentinel cannot be mistaken for
    # a learned confidence and BCE remains finite for the required unknown label.
    return logits.masked_fill(~has_pages, -torch.inf)


def _output(
    relevance: torch.Tensor,
    direction: torch.Tensor,
    raw_value: torch.Tensor,
    page_mask: torch.Tensor,
    bk: torch.Tensor,
    uk: torch.Tensor,
    vk: torch.Tensor,
    uniform_attention: bool,
    direct_grade: bool = False,
) -> ReaderOutput:
    valid = page_mask.to(torch.bool)
    relevance = relevance.masked_fill(~valid, -torch.inf)
    attention = _masked_softmax(relevance, valid, uniform_attention)
    direction = torch.where(valid, direction, torch.zeros_like(direction))
    raw_value = torch.where(valid, raw_value, torch.zeros_like(raw_value))
    contributions = raw_value if direct_grade else 0.5 + direction * (raw_value - 0.5)
    score = (attention * contributions).sum(dim=-1)
    has_pages = valid.any(dim=-1)
    score = score.masked_fill(~has_pages, torch.nan)
    known_logits = _knownness(relevance, attention, raw_value, valid, bk, uk, vk)
    return ReaderOutput(score, known_logits, relevance, attention, direction, raw_value)


class PageReader(nn.Module):
    """Cached criterion-conditioned evidence reader with the frozen rank-64 equations."""

    def __init__(self, rank: int = RANK):
        super().__init__()
        if rank != RANK:
            raise ValueError(f"ECA-1 fixes rank={RANK}, got {rank}")
        self.wp = nn.Linear(WIDTH, rank, bias=False)
        self.wqa = nn.Linear(WIDTH, rank, bias=False)
        self.wqd = nn.Linear(WIDTH, rank, bias=False)
        self.wv = nn.Linear(WIDTH, 1, bias=True)
        self.bk = nn.Parameter(torch.zeros(()))
        self.uk = nn.Parameter(torch.zeros(()))
        self.vk = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        q: torch.Tensor,
        pages: torch.Tensor,
        page_mask: torch.Tensor,
        raw_q: torch.Tensor | None = None,
        raw_pages: torch.Tensor | None = None,
        *,
        uniform_attention: bool = False,
    ) -> ReaderOutput:
        _check_pages(pages, page_mask, "pages")
        if q.ndim != 2 or q.shape != (pages.shape[0], WIDTH):
            raise ValueError(f"q must have shape [N,{WIDTH}]")
        z = self.wp(pages)
        relevance = (z * self.wqa(q).unsqueeze(1)).sum(dim=-1) / (RANK ** 0.5)
        direction = torch.tanh((z * self.wqd(q).unsqueeze(1)).sum(dim=-1) / (RANK ** 0.5))
        raw_value = torch.sigmoid(self.wv(pages).squeeze(-1))
        return _output(relevance, direction, raw_value, page_mask,
                       self.bk, self.uk, self.vk, uniform_attention)


class CosineReader(PageReader):
    """Page reader whose attention logits are frozen raw-feature cosine scores."""
    def __init__(self, rank: int = RANK):
        super().__init__(rank)
        # Cosine logits are frozen; do not retain an unused learned attention map.
        del self.wqa


    def forward(
        self,
        q: torch.Tensor,
        pages: torch.Tensor,
        page_mask: torch.Tensor,
        raw_q: torch.Tensor | None = None,
        raw_pages: torch.Tensor | None = None,
        *,
        uniform_attention: bool = False,
    ) -> ReaderOutput:
        _check_pages(pages, page_mask, "pages")
        if q.ndim != 2 or q.shape != (pages.shape[0], WIDTH):
            raise ValueError(f"q must have shape [N,{WIDTH}]")
        if raw_q is None or raw_pages is None:
            raise ValueError("CosineReader requires raw_q and raw_pages")
        if raw_q.shape != q.shape or raw_pages.shape != pages.shape:
            raise ValueError("raw cosine features must match q/pages shapes")
        q_cos = F.normalize(raw_q, p=2, dim=-1, eps=1e-12)
        page_cos = F.normalize(raw_pages, p=2, dim=-1, eps=1e-12)
        relevance = (page_cos * q_cos.unsqueeze(1)).sum(dim=-1)
        z = self.wp(pages)
        direction = torch.tanh((z * self.wqd(q).unsqueeze(1)).sum(dim=-1) / (RANK ** 0.5))
        raw_value = torch.sigmoid(self.wv(pages).squeeze(-1))
        return _output(relevance, direction, raw_value, page_mask,
                       self.bk, self.uk, self.vk, uniform_attention)


class CrossReader(nn.Module):
    """Joint question/page feature control with canonical 128-GELU heads."""

    def __init__(self):
        super().__init__()
        self.relevance_head = nn.Sequential(nn.Linear(WIDTH, CROSS_HIDDEN), nn.GELU(),
                                            nn.Linear(CROSS_HIDDEN, 1))
        self.grade_head = nn.Sequential(nn.Linear(WIDTH, CROSS_HIDDEN), nn.GELU(),
                                        nn.Linear(CROSS_HIDDEN, 1))
        self.direction_head = nn.Sequential(nn.Linear(WIDTH, CROSS_HIDDEN), nn.GELU(),
                                            nn.Linear(CROSS_HIDDEN, 1))
        self.bk = nn.Parameter(torch.zeros(()))
        self.uk = nn.Parameter(torch.zeros(()))
        self.vk = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        pair_features: torch.Tensor,
        page_mask: torch.Tensor,
        *,
        uniform_attention: bool = False,
    ) -> ReaderOutput:
        _check_pages(pair_features, page_mask, "pair_features")
        relevance = self.relevance_head(pair_features).squeeze(-1)
        # Cross predicts g for + orientation and 1-g for - orientation directly.
        raw_value = torch.sigmoid(self.grade_head(pair_features).squeeze(-1))
        direction = torch.tanh(self.direction_head(pair_features).squeeze(-1))
        return _output(relevance, direction, raw_value, page_mask,
                       self.bk, self.uk, self.vk, uniform_attention, direct_grade=True)


class QueryBlindReader(nn.Module):
    """Trainable uniform-attention page-grade control with no question input."""

    def __init__(self):
        super().__init__()
        self.grade_head = nn.Sequential(nn.Linear(WIDTH, CROSS_HIDDEN), nn.GELU(),
                                        nn.Linear(CROSS_HIDDEN, 1))
        self.bk = nn.Parameter(torch.zeros(()))
        self.register_buffer("uk", torch.zeros(()))
        self.vk = nn.Parameter(torch.zeros(()))

    def forward(self, pages: torch.Tensor, page_mask: torch.Tensor) -> ReaderOutput:
        _check_pages(pages, page_mask, "pages")
        relevance = torch.zeros(page_mask.shape, dtype=pages.dtype, device=pages.device)
        raw_value = torch.sigmoid(self.grade_head(pages).squeeze(-1))
        direction = torch.ones_like(raw_value)
        return _output(relevance, direction, raw_value, page_mask,
                       self.bk, self.uk, self.vk, uniform_attention=True)

class LexicalReader:
    """Train-only word/character TF-IDF relevance, grade and orientation control."""

    def __init__(self):
        self.vectorizer = None
        self.relevance_model = None
        self.extent_model = None
        self.orientation_model = None
        self.known_model = None

    @staticmethod
    def _pair_text(question: str, page_text: str) -> str:
        if not question or not page_text:
            raise ValueError("lexical inputs must be nonempty semantic question/page text")
        return question + "\n" + page_text

    @staticmethod
    def _logistic():
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C=1.0, solver="liblinear", max_iter=1000, random_state=7)

    def fit(self, records: list[dict[str, object]]) -> "LexicalReader":
        if not records:
            raise ValueError("LexicalReader requires training records")
        if any(record.get("split") != "train" for record in records):
            raise ValueError("LexicalReader vocabulary/fits may use train records only")
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import FeatureUnion
        from sklearn.linear_model import Ridge

        texts: list[str] = []
        relevance: list[int] = []
        extents: list[float] = []
        extent_mask: list[bool] = []
        orientations: list[float] = []
        orientation_mask: list[bool] = []
        page_ranges: list[tuple[int, int]] = []
        for record in records:
            start = len(texts)
            for index, page in enumerate(record["pages"]):
                texts.append(self._pair_text(str(record["question"]), str(page["text"])))
                relevance.append(int(float(record["relevance_target"][index]) > 0.0))
                grade = record["grade_target"][index]
                extent_mask.append(bool(record["grade_mask"][index]) and
                                   grade is not None and math.isfinite(float(grade)))
                extents.append(float(grade) if grade is not None else 0.0)
                orientation = record["orientation_target"][index]
                orientation_mask.append(bool(record["orientation_mask"][index]) and
                                        orientation is not None and math.isfinite(float(orientation)))
                orientations.append(float(orientation) if orientation is not None else 0.0)
            page_ranges.append((start, len(texts)))
        if not texts:
            raise ValueError("LexicalReader training corpus contains no semantic pages")
        if set(relevance) != {0, 1}:
            raise ValueError("lexical relevance training requires relevant and irrelevant pages")
        self.vectorizer = FeatureUnion([
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2))),
            ("char", TfidfVectorizer(analyzer="char", ngram_range=(3, 5))),
        ])
        matrix = self.vectorizer.fit_transform(texts)
        self.relevance_model = self._logistic().fit(matrix, relevance)
        use_extent = np.asarray(extent_mask, dtype=np.bool_)
        if not use_extent.any():
            raise ValueError("lexical extent training has no defined relevant-page grades")
        self.extent_model = Ridge(alpha=1.0).fit(matrix[use_extent], np.asarray(extents)[use_extent])
        use_orientation = np.asarray(orientation_mask, dtype=np.bool_)
        if not use_orientation.any():
            raise ValueError("lexical orientation training has no defined relevant pages")
        self.orientation_model = Ridge(alpha=1.0).fit(
            matrix[use_orientation], np.asarray(orientations)[use_orientation])

        rel_logits = self.relevance_model.decision_function(matrix)
        extent_predictions = np.clip(self.extent_model.predict(matrix), 0.0, 1.0)
        known_features = []
        known_targets = []
        for record, (start, stop) in zip(records, page_ranges):
            row_logits = rel_logits[start:stop]
            row_values = extent_predictions[start:stop]
            if stop == start:
                known_features.append([0.0, 0.0])
            else:
                stable = row_logits - np.max(row_logits)
                attention = np.exp(stable)
                attention /= attention.sum()
                mean = float(np.dot(attention, row_values))
                variance = float(np.dot(attention, np.square(row_values - mean)))
                known_features.append([float(np.max(row_logits)), variance])
            known_targets.append(int(bool(record["known_target"])))
        if set(known_targets) != {0, 1}:
            raise ValueError("lexical knownness training requires known and unknown candidates")
        self.known_model = self._logistic().fit(np.asarray(known_features), known_targets)
        return self

    def predict_records(self, records: list[dict[str, object]], device: str = "cpu") -> ReaderOutput:
        if any(model is None for model in (self.vectorizer, self.relevance_model,
                                           self.extent_model, self.orientation_model, self.known_model)):
            raise RuntimeError("LexicalReader.fit must run on training records before prediction")
        n = len(records)
        pmax = max(1, max((len(record["pages"]) for record in records), default=0))
        relevance = np.full((n, pmax), -np.inf, dtype=np.float32)
        attention = np.zeros((n, pmax), dtype=np.float32)
        direction = np.zeros((n, pmax), dtype=np.float32)
        values = np.zeros((n, pmax), dtype=np.float32)
        known_features = np.zeros((n, 2), dtype=np.float64)
        pair_texts = [self._pair_text(str(record["question"]), str(page["text"]))
                      for record in records for page in record["pages"]]
        matrix = self.vectorizer.transform(pair_texts) if pair_texts else None
        if matrix is not None:
            logits = self.relevance_model.decision_function(matrix)
            extent = np.clip(self.extent_model.predict(matrix), 0.0, 1.0)
            orient = np.clip(self.orientation_model.predict(matrix), -1.0, 1.0)
        offset = 0
        scores = np.full((n,), np.nan, dtype=np.float32)
        known_logits = np.full((n,), -np.inf, dtype=np.float32)
        for row_index, record in enumerate(records):
            count = len(record["pages"])
            if count == 0:
                continue
            row_logits = np.asarray(logits[offset:offset + count], dtype=np.float64)
            row_values = np.asarray(extent[offset:offset + count], dtype=np.float64)
            row_direction = np.asarray(orient[offset:offset + count], dtype=np.float64)
            offset += count
            stable = row_logits - np.max(row_logits)
            row_attention = np.exp(stable)
            row_attention /= row_attention.sum()
            mean = float(np.dot(row_attention, row_values))
            variance = float(np.dot(row_attention, np.square(row_values - mean)))
            known_features[row_index] = [float(np.max(row_logits)), variance]
            relevance[row_index, :count] = row_logits
            attention[row_index, :count] = row_attention
            direction[row_index, :count] = row_direction
            values[row_index, :count] = row_values
            scores[row_index] = float(np.dot(row_attention, 0.5 + row_direction * (row_values - 0.5)))
        has_pages = np.asarray([bool(record["pages"]) for record in records])
        if has_pages.any():
            known_logits[has_pages] = self.known_model.decision_function(
                known_features[has_pages]).astype(np.float32)
        return ReaderOutput(
            torch.as_tensor(scores, dtype=torch.float32, device=device),
            torch.as_tensor(known_logits, dtype=torch.float32, device=device),
            torch.as_tensor(relevance, dtype=torch.float32, device=device),
            torch.as_tensor(attention, dtype=torch.float32, device=device),
            torch.as_tensor(direction, dtype=torch.float32, device=device),
            torch.as_tensor(values, dtype=torch.float32, device=device),
        )

    def save(self, path: str) -> None:
        if self.vectorizer is None:
            raise RuntimeError("cannot persist an unfitted LexicalReader")
        import joblib
        joblib.dump({
            "schema": "eca1-lexical-v1", "regularization": 1.0,
            "word_ngrams": [1, 2], "char_ngrams": [3, 5],
            "vectorizer": self.vectorizer, "relevance_model": self.relevance_model,
            "extent_model": self.extent_model, "orientation_model": self.orientation_model,
            "known_model": self.known_model,
        }, path, compress=3)

    @classmethod
    def load(cls, path: str) -> "LexicalReader":
        import joblib
        artifact = joblib.load(path)
        if (artifact.get("schema") != "eca1-lexical-v1" or
                artifact.get("regularization") != 1.0 or
                artifact.get("word_ngrams") != [1, 2] or
                artifact.get("char_ngrams") != [3, 5]):
            raise ValueError("persisted lexical model does not match the frozen ECA control")
        instance = cls()
        instance.vectorizer = artifact["vectorizer"]
        instance.relevance_model = artifact["relevance_model"]
        instance.extent_model = artifact["extent_model"]
        instance.orientation_model = artifact["orientation_model"]
        instance.known_model = artifact["known_model"]
        return instance




def intervene_features(
    q: torch.Tensor,
    pages: torch.Tensor,
    page_mask: torch.Tensor,
    raw_q: torch.Tensor | None = None,
    raw_pages: torch.Tensor | None = None,
    *,
    zero_question: bool = False,
    zero_pages: bool = False,
    uniform_attention: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, bool]:
    """Return inference tensors for the preregistered feature/attention interventions.

    Zeroing occurs in reader feature space; raw cosine features are zeroed too,
    so CosineReader cannot bypass either intervention through its frozen logits.
    """
    if zero_question:
        q = torch.zeros_like(q)
        if raw_q is not None:
            raw_q = torch.zeros_like(raw_q)
    if zero_pages:
        pages = torch.zeros_like(pages)
        if raw_pages is not None:
            raw_pages = torch.zeros_like(raw_pages)
    return q, pages, page_mask, raw_q, raw_pages, uniform_attention


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, zero_from: torch.Tensor) -> torch.Tensor:
    use = mask.to(torch.bool) & torch.isfinite(values)
    count = use.sum()
    if not bool(count):
        return zero_from.sum() * 0.0
    return values.masked_select(use).mean()


def eca_loss(output: ReaderOutput, targets: dict[str, torch.Tensor], *,
             directed_grade: bool = False) -> dict[str, torch.Tensor]:
    """Equal-weight relevance CE, normalized grade MSE, orientation MSE and known BCE.

    Required target keys are relevance_target, grade_target, grade_mask,
    orientation_target, orientation_mask, known_target, and page_mask. Undefined,
    irrelevant, padded, or nonfinite targets are excluded component-wise. A
    no-page candidate must have known_target=0; its -inf known logit is an
    explicit structural unknown, not a fabricated score.
    """
    required = ("relevance_target", "grade_target", "grade_mask", "orientation_target",
                "orientation_mask", "known_target", "page_mask")
    missing = [key for key in required if key not in targets]
    if missing:
        raise KeyError(f"missing ECA targets: {missing}")
    page_mask = targets["page_mask"].to(torch.bool)
    if page_mask.shape != output.relevance_logits.shape:
        raise ValueError("target page_mask does not match reader output")
    has_pages = page_mask.any(dim=-1)
    known_target = targets["known_target"].to(output.known_logits.dtype).reshape(-1)
    if known_target.shape != output.known_logits.shape:
        raise ValueError("known_target must have one label per candidate")
    if bool(((~has_pages) & (known_target != 0)).any()):
        raise ValueError("a candidate with no evidence must be explicitly unknown")

    relevance_target = targets["relevance_target"].to(output.relevance_logits.dtype)
    if relevance_target.shape != output.relevance_logits.shape:
        raise ValueError("relevance_target must have shape [N,P]")
    rel_valid = page_mask & torch.isfinite(relevance_target) & (relevance_target >= 0)
    rel_target = torch.where(rel_valid, relevance_target, torch.zeros_like(relevance_target))
    rel_mass = rel_target.sum(dim=-1)
    rel_rows = (rel_mass > 0) & has_pages
    safe_logits = output.relevance_logits.masked_fill(~page_mask, -1e30)
    log_probs = torch.log_softmax(safe_logits, dim=-1)
    per_row_ce = -(rel_target * log_probs).sum(dim=-1) / rel_mass.clamp_min(1.0)
    relevance_loss = per_row_ce[rel_rows].mean() if bool(rel_rows.any()) else output.score.nan_to_num().sum() * 0.0

    grade_key = "directed_grade_target" if directed_grade else "grade_target"
    if grade_key not in targets:
        raise KeyError(f"missing ECA target {grade_key!r}")
    grade_target = targets[grade_key].to(output.raw_value.dtype)
    grade_mask = targets["grade_mask"].to(torch.bool) & page_mask
    if grade_target.shape != output.raw_value.shape or grade_mask.shape != output.raw_value.shape:
        raise ValueError("grade targets/mask must have shape [N,P]")
    grade_valid = grade_mask & torch.isfinite(grade_target) & torch.isfinite(output.raw_value)
    grade_loss = (
        (output.raw_value[grade_valid] - grade_target[grade_valid]).square().mean()
        if bool(grade_valid.any()) else output.raw_value.sum() * 0.0
    )

    orientation_target = targets["orientation_target"].to(output.direction.dtype)
    orientation_mask = targets["orientation_mask"].to(torch.bool) & page_mask
    if orientation_target.shape != output.direction.shape or orientation_mask.shape != output.direction.shape:
        raise ValueError("orientation targets/mask must have shape [N,P]")
    orientation_valid = orientation_mask & torch.isfinite(orientation_target) & torch.isfinite(output.direction)
    orientation_loss = (
        (output.direction[orientation_valid] - orientation_target[orientation_valid]).square().mean()
        if bool(orientation_valid.any()) else output.direction.sum() * 0.0
    )

    known_valid = torch.isfinite(known_target) & ((torch.isfinite(output.known_logits)) |
                   ((~has_pages) & (known_target == 0)))
    if bool((known_valid & ((known_target < 0) | (known_target > 1))).any()):
        raise ValueError("known_target must be in [0,1]")
    if bool(known_valid.any()):
        # Structural no-page UNKNOWN is already certain; BCE(-inf, 0) is NaN.
        safe_known_logits = torch.where(has_pages, output.known_logits, torch.zeros_like(output.known_logits))
        per_candidate = F.binary_cross_entropy_with_logits(
            safe_known_logits, known_target, reduction="none",
        )
        known_loss = per_candidate.masked_fill(~has_pages, 0.0)[known_valid].mean()
    else:
        known_loss = output.score.nan_to_num().sum() * 0.0

    total = relevance_loss + grade_loss + orientation_loss + known_loss
    return {"total": total, "relevance": relevance_loss, "grade": grade_loss,
            "orientation": orientation_loss, "known": known_loss}


def record_loss(history: list[dict[str, float]], epoch: int,
                components: dict[str, torch.Tensor]) -> None:
    """Append all objective components for durable training-history persistence."""
    row = {"epoch": int(epoch)}
    for name in ("total", "relevance", "grade", "orientation", "known"):
        if name not in components:
            raise KeyError(f"missing loss component {name!r}")
        value = float(components[name].detach().cpu())
        if not torch.isfinite(torch.tensor(value)):
            raise FloatingPointError(f"nonfinite ECA loss component {name}: {value}")
        row[name] = value
    history.append(row)
