#!/usr/bin/env python3
"""Train-free ECA-1 calibration and clustered evaluation; development by default.

Final IR is opened only after verification of the immutable five-control
selection/calibration receipt. This authored assay earns no certificate,
performance claim, endgame NI result, or product promotion.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ephemeral_pages_capture as capture
from ephemeral_pages_features import PROTOCOL_SHA256, sha256_file, validate_final_receipt

CONTROLS = ("pages", "cross", "cosine", "lexical", "query_blind")
INTERVENTIONS = ("zero_question", "zero_pages", "uniform_attention")
UNKNOWN = "__unknown__"
PROTOCOL = Path(__file__).with_name("ephemeral_pages_protocol.json")
DEFAULT_RUN = Path("/home/fazinahamed/Documents/vey-data/decisionmix/endgame/ephemeral-pages-v1/runs/seed7-grade-corrected")
SIGMAS = (.025, .05, .075, .1, .15, .2, .3, .4)
LEVELS = np.arange(5, dtype=np.float64) / 4
BOOTSTRAPS = 10000
BOOTSTRAP_SEED = 0
DIAGNOSTIC_VARIANTS = {"base", "relevant_page_erasure", "relevant_page_contradiction"}
ROBUST_VARIANTS = {"candidate_permutation_1", "candidate_permutation_2", "candidate_rename", "page_reorder", "question_reorder"}


def protocol() -> dict:
    if sha256_file(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("ECA protocol changed; no evaluation is authorized")
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> dict:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(plain(value), stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return artifact(path)


def artifact(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def checkpoint_files(run_root: Path) -> dict:
    return {control: artifact(run_root / ("lexical.pkl" if control == "lexical" else f"{control}.pt"))
            for control in CONTROLS}


def load_ir(phase: str, receipt: Path | None = None) -> dict[str, dict]:
    if phase == "final":
        if receipt is None:
            raise RuntimeError("final requires an immutable selection/calibration receipt")
        validate_final_receipt(receipt)
    elif phase not in {"calibration", "development"}:
        raise ValueError("evaluation only opens calibration, development, or receipt-gated final")
    result = {}
    for row in capture.rows_for(phase):
        if row["id"] in result:
            raise RuntimeError("duplicate DecisionIR ID")
        result[row["id"]] = row
    return result


def load_data(phase: str, receipt: Path | None = None) -> tuple[dict, dict]:
    # Gate precedes even capture.load_phase's hash read of final.jsonl.
    if phase == "final":
        if receipt is None:
            raise RuntimeError("final is sealed without a receipt")
        validate_final_receipt(receipt)
    data = capture.load_phase(phase)
    ir = load_ir(phase, receipt)
    if {record["row_id"] for record in data["records"]} != set(ir):
        raise RuntimeError("captured records and DecisionIR do not cover the same rows")
    return data, ir


def prediction_fields(output: Any, data: dict) -> dict[str, np.ndarray]:
    fields = {name: np.asarray(getattr(output, name)) for name in
              ("score", "known_logits", "relevance_logits", "attention", "direction", "raw_value")}
    n, p = data["page_mask"].shape
    for name, values in fields.items():
        expected = (n,) if name in {"score", "known_logits"} else (n, p)
        if values.shape != expected:
            raise RuntimeError(f"prediction {name} shape {values.shape} != {expected}")
    valid = np.asarray(data["page_mask"], dtype=bool)
    for name in ("attention", "direction", "raw_value"):
        if not np.isfinite(fields[name][valid]).all():
            raise RuntimeError(f"nonfinite {name} on a real page")
    if np.isnan(fields["known_logits"]).any() or np.isposinf(fields["known_logits"]).any():
        raise RuntimeError("invalid knownness logits")
    if not np.isfinite(fields["score"][valid.any(axis=1)]).all():
        raise RuntimeError("nonfinite score on a candidate with pages")
    logits = fields["known_logits"].astype(np.float64)
    fields["known_probability"] = np.exp(-np.logaddexp(0, -logits))
    return fields


def ordinal_mass(scores: np.ndarray, sigma: float) -> np.ndarray:
    logits = -.5 * ((np.asarray(scores, dtype=np.float64)[..., None] - LEVELS) / sigma) ** 2
    logits -= np.max(logits, axis=-1, keepdims=True)
    masses = np.exp(logits)
    return masses / masses.sum(axis=-1, keepdims=True)


def term_truth(record: dict) -> float | None:
    values = [target for target, mask in zip(record["directed_grade_target"], record["grade_mask"]) if mask]
    return float(values[0]) if len(values) == 1 else None


def calibration_indices(data: dict, ir: dict) -> list[int]:
    # All authored calibration variants share their world's bootstrap unit.
    return list(range(len(data["records"])))


def calibrate(fields: dict, data: dict, ir: dict) -> dict:
    eligible = calibration_indices(data, ir)
    candidates = defaultdict(list)
    for i in eligible:
        record = data["records"][i]
        candidates[(record["row_id"], record["candidate_id"])].append(i)
    known_probability, known_target = [], []
    for (row_id, cid), indices in sorted(candidates.items()):
        expected_terms = len(ir[row_id]["metadata"]["terms"])
        if len(indices) != expected_terms:
            raise RuntimeError("calibration candidate is missing a required term")
        known_probability.append(float(np.min(fields["known_probability"][indices])))
        known_target.append(bool(ir[row_id]["metadata"]["known"][cid]))
    probs = np.asarray(known_probability)
    targets = np.asarray(known_target, dtype=bool)
    if not targets.any() or targets.all():
        raise RuntimeError("candidate balanced calibration requires both known and unknown classes")
    grid = []
    for integer in range(1, 20):
        threshold = integer / 20
        predicted = probs >= threshold
        balanced = .5 * (np.mean(predicted[targets]) + np.mean(~predicted[~targets]))
        grid.append({"threshold": threshold, "balanced_candidate_accuracy": float(balanced)})
    winner = max(grid, key=lambda row: (row["balanced_candidate_accuracy"], -row["threshold"]))
    ordinal_indices = [i for i in eligible if term_truth(data["records"][i]) is not None
                       and math.isfinite(float(fields["score"][i]))]
    if not ordinal_indices:
        raise RuntimeError("no supported ordinal calibration candidates")
    truths = np.asarray([round(term_truth(data["records"][i]) * 4) for i in ordinal_indices], dtype=int)
    sigma_grid = []
    for sigma in SIGMAS:
        mass = ordinal_mass(fields["score"][ordinal_indices], sigma)
        nll = -np.log(np.maximum(mass[np.arange(len(truths)), truths], np.finfo(float).tiny)).mean()
        sigma_grid.append({"sigma": sigma, "NLL": float(nll)})
    best_sigma = min(sigma_grid, key=lambda row: row["NLL"])
    return {"knownness_threshold": winner["threshold"], "knownness_grid": grid,
            "candidate_count": len(targets), "known_candidates": int(targets.sum()),
            "candidate_rule": "min required-child knownness; every captured calibration decision/candidate",
            "ordinal_sigma": best_sigma["sigma"], "ordinal_sigma_grid": sigma_grid,
            "ordinal_candidate_count": len(truths), "fit_split": "calibration",
            "certificate": "unavailable", "final_outcomes_used": False}


def stable_pick(scores: dict[str, float], ordinals: dict) -> tuple[str, list[str]]:
    if not scores:
        return UNKNOWN, []
    if set(scores) - set(ordinals):
        raise RuntimeError("missing stable identity ordinal")
    best = max(scores.values())
    maxima = [cid for cid, score in scores.items() if math.isclose(score, best, rel_tol=0, abs_tol=1e-12)]
    return min(maxima, key=lambda cid: (int(ordinals[cid]), cid)), maxima


def exact_weight(term: dict) -> Fraction:
    factor = term["weight"]
    if isinstance(factor, bool) or not isinstance(factor, (int, float)) or not math.isfinite(factor):
        raise ValueError("typed term factor must be finite")
    return Fraction(str(factor))


def execute_typed_exact(row: dict, terms: list[dict]) -> dict:
    """Execute explicitly typed numeric facts, never prose or neural arithmetic.

    Each child supplies fact_key, integer weight and explicit sign (+1/-1).
    Parent-owned closed diagnostics can call this without opening fresh worlds.
    """
    if not terms:
        raise ValueError("an exact expression needs a required child")
    parsed = []
    for term in terms:
        key, factor, sign = term["fact_key"], term["weight"], term["sign"]
        if not isinstance(key, str) or not key:
            raise ValueError("exact child needs a typed fact key")
        if type(factor) is not int or type(sign) is not int or sign not in (-1, 1):
            raise ValueError("exact factors must be integers with an explicit +/- sign")
        parsed.append((key, factor * sign))
    meta = row["metadata"]
    candidates = [candidate["id"] for candidate in row["candidates"] if candidate["id"] != UNKNOWN]
    scores, reasons = {}, []
    for cid in candidates:
        facts = meta.get("exact_facts", {}).get(cid, {})
        values = []
        for key, factor in parsed:
            value = facts.get(key)
            if type(value) is not int:
                reasons.append({"candidate_id": cid, "fact_key": key, "reason": "required_exact_fact_missing"})
            else:
                values.append(factor * value)
        scores[cid] = sum(values) if len(values) == len(parsed) else None
    if reasons or not candidates:
        chosen, maxima = UNKNOWN, [UNKNOWN]
    else:
        best = max(scores.values())
        maxima = [cid for cid, score in scores.items() if score == best]
        chosen = min(maxima, key=lambda cid: (meta["stable_ordinals"][cid], cid))
    return {"row_id": row["id"], "chosen": chosen, "maximal_predictions": maxima,
            "candidate_scores": scores, "UNKNOWN_reasons": reasons, "terms": terms,
            "source": "exact", "probability": "not_applicable", "certificate": "not_applicable",
            "fallback": chosen == UNKNOWN}


def aggregate(fields: dict, data: dict, ir: dict, calibration: dict) -> list[dict]:
    groups = defaultdict(dict)
    for i, record in enumerate(data["records"]):
        key = (record["term_index"], record["candidate_id"])
        if key in groups[record["row_id"]]:
            raise RuntimeError("duplicate term/candidate record")
        groups[record["row_id"]][key] = i
    result = []
    threshold = calibration["knownness_threshold"]
    for row_id, row in ir.items():
        meta = row["metadata"]
        terms = meta["terms"]
        candidates = [c["id"] for c in row["candidates"] if c["id"] != UNKNOWN]
        expected = {(t, cid) for t in range(len(terms)) for cid in candidates}
        if set(groups[row_id]) != expected:
            raise RuntimeError(f"required term/candidate missing from capture: {row_id}")
        scores, known, failures, children = {}, {}, [], {}
        for cid in candidates:
            parts = []
            children[cid] = []
            known[cid] = True
            for term_index, term in enumerate(terms):
                i = groups[row_id][(term_index, cid)]
                if exact_weight(term) != Fraction(str(data["records"][i]["term_weight"])):
                    raise RuntimeError("captured term factor differs from typed DecisionIR")
                score = float(fields["score"][i])
                probability = float(fields["known_probability"][i])
                child_known = probability >= threshold and math.isfinite(score)
                children[cid].append({"term_index": term_index, "raw_score": score,
                                      "known_probability": probability, "known": child_known,
                                      "weight": str(exact_weight(term))})
                if not child_known:
                    known[cid] = False
                    failures.append({"candidate_id": cid, "term_index": term_index,
                                     "reason": "no_pages_or_nonfinite_score" if not math.isfinite(score)
                                     else "below_calibrated_knownness_threshold"})
                else:
                    parts.append(exact_weight(term) * Fraction(str(score)))
            scores[cid] = float(sum(parts, Fraction(0))) if known[cid] else None
        if failures or not candidates:
            chosen, maxima = UNKNOWN, [UNKNOWN]
        else:
            chosen, maxima = stable_pick(scores, meta["stable_ordinals"])
        mapping = meta.get("identity_map", {})
        gold = list(row["gold"])
        result.append({"row_id": row_id, "world_id": meta["world_id"], "split": row["split"],
                       "variant": meta["variant"], "query_kind": meta["query_kind"],
                       "query_id": meta["query_id"], "family": meta["family"],
                       "candidate_count": len(candidates), "term_count": len(terms),
                       "parent_row_id": meta.get("parent_decision_id"),
                       "chosen": chosen, "source_chosen": mapping.get(chosen, chosen),
                       "maximal_predictions": maxima, "gold": gold,
                       "source_gold": [mapping.get(cid, cid) for cid in gold],
                       "correct": chosen in gold, "supported": UNKNOWN not in gold,
                       "exact_winner_set_correct": set(maxima) == set(gold),
                       "candidate_scores": scores, "candidate_known": known,
                       "required_children": children, "UNKNOWN_reasons": failures,
                       "source": "semantic", "certificate": "unavailable", "fallback": chosen == UNKNOWN,
                       "ordinal_probability": "descriptive_only; not a decision certificate"})
    return result


def save_predictions(directory: Path, name: str, fields: dict, data: dict, ir: dict,
                     decisions: list[dict], sigma: float) -> dict:
    raw_path = directory / f"{name}_term_candidate.jsonl"
    with raw_path.open("x", encoding="utf-8") as stream:
        for i, record in enumerate(data["records"]):
            p = len(record["pages"])
            raw = {"record_index": i, **record,
                   "prediction": {key: value[i] if value.ndim == 1 else value[i, :p]
                                  for key, value in fields.items()},
                   "ordinal_mass_uncalibrated_sigma_0_1": ordinal_mass(np.asarray(fields["score"][i]), .1)
                   if math.isfinite(float(fields["score"][i])) else None,
                   "ordinal_mass_calibrated": ordinal_mass(np.asarray(fields["score"][i]), sigma)
                   if math.isfinite(float(fields["score"][i])) else None,
                   "variant": ir[record["row_id"]]["metadata"]["variant"],
                   "certificate": "unavailable"}
            stream.write(json.dumps(plain(raw), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    decisions_path = directory / f"{name}_decisions.jsonl"
    with decisions_path.open("x", encoding="utf-8") as stream:
        for decision in decisions:
            stream.write(json.dumps(plain(decision), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return {"term_candidate": artifact(raw_path), "decisions": artifact(decisions_path)}


class Metrics:
    """World-level numerator/denominator ledger, with fixed family macro weights."""

    def __init__(self, worlds: list[str]):
        self.worlds = worlds
        self.index = {world: i for i, world in enumerate(worlds)}
        self.values = {}

    def add(self, name: str, world: str, numerator: float, denominator: float = 1,
            family: str = "all") -> None:
        if name not in self.values:
            self.values[name] = {}
        if family not in self.values[name]:
            self.values[name][family] = np.zeros((len(self.worlds), 2), dtype=np.float64)
        self.values[name][family][self.index[world]] += (numerator, denominator)

    def estimate(self, name: str, weights: np.ndarray | None = None) -> np.ndarray | float:
        ratios = []
        for values in self.values[name].values():
            sums = values.sum(axis=0) if weights is None else weights @ values
            numerator, denominator = sums[..., 0], sums[..., 1]
            ratios.append(np.divide(numerator, denominator, out=np.full_like(numerator, np.nan, dtype=float),
                                    where=denominator > 0))
        # A bootstrap draw lacking any fixed family is undefined, not a lower-dimensional macro.
        return np.mean(ratios, axis=0)

    def summaries(self, weights: np.ndarray) -> dict:
        result = {}
        for name in sorted(self.values):
            samples = self.estimate(name, weights)
            valid = samples[np.isfinite(samples)]
            result[name] = {"point": float(self.estimate(name)),
                            "CI95": np.quantile(valid, [.025, .975]).tolist() if len(valid) else None,
                            "valid_bootstrap_draws": len(valid),
                            "families": sorted(self.values[name]),
                            "denominator": float(sum(values[:, 1].sum() for values in self.values[name].values()))}
        return result

    def ledger(self) -> dict:
        return {"world_ids": self.worlds,
                "metrics": {name: {family: values.tolist() for family, values in groups.items()}
                            for name, groups in self.values.items()},
                "estimator": "sum numerator / sum denominator within each fixed family; equal macro over families"}


def is_primary(row: dict) -> bool:
    return (row["variant"] == "base" and row["supported"] and row["candidate_count"] == 4
            and row["term_count"] == 1)


def distribution_events(metrics: Metrics, prefix: str, world: str, score: float,
                        truth: float, sigma: float) -> None:
    mass = ordinal_mass(np.asarray(score), sigma)
    label = round(truth * 4)
    onehot = np.eye(5)[label]
    metrics.add(prefix + "NLL", world, -math.log(max(float(mass[label]), np.finfo(float).tiny)))
    metrics.add(prefix + "Brier", world, float(np.sum((mass - onehot) ** 2)))
    confidence = float(np.max(mass))
    bucket = min(9, int(confidence * 10))
    metrics.add(prefix + f"ECE_bin{bucket}_confidence", world, confidence)
    metrics.add(prefix + f"ECE_bin{bucket}_accuracy", world, int(np.argmax(mass)) == label)


def metric_ledger(control: str, fields: dict, data: dict, ir: dict, decisions: list[dict],
                  sigma: float, families: list[str], worlds: list[str]) -> tuple[Metrics, list[dict]]:
    metrics = Metrics(worlds)
    by_id = {row["row_id"]: row for row in decisions}
    pairs = []
    for row in decisions:
        world = row["world_id"]
        if is_primary(row):
            if len(row["family"]) != 1:
                raise RuntimeError("supported atomic criterion must map to one family")
            family = row["family"][0]
            # The protocol names exact-winner-set primary accuracy. Concrete
            # winner membership remains a separate, less demanding diagnostic.
            metrics.add("atomic_choice_macro", world, row["exact_winner_set_correct"], family=family)
            metrics.add("atomic_choice_micro", world, row["exact_winner_set_correct"])
            metrics.add("concrete_winner_choice_macro", world, row["correct"], family=family)
            if family in families[-4:]:
                metrics.add("held4_family_choice_macro", world, row["exact_winner_set_correct"], family=family)
            if family in families[12:16]:
                metrics.add("development4_family_choice_macro", world, row["exact_winner_set_correct"], family=family)
        if row["variant"] == "base" and row["supported"] and row["term_count"] > 1:
            metrics.add("composition", world, row["correct"])
        if row["variant"] in DIAGNOSTIC_VARIANTS:
            predicted_unknown = row["chosen"] == UNKNOWN
            gold_unknown = not row["supported"]
            metrics.add("UNKNOWN_precision", world, predicted_unknown and gold_unknown, predicted_unknown)
            metrics.add("UNKNOWN_recall", world, predicted_unknown and gold_unknown, gold_unknown)
            metrics.add("supported_coverage", world, row["supported"] and not predicted_unknown, row["supported"])
        if row["variant"].startswith("probe_k") or row["variant"] == "base":
            metrics.add(f"K{row['candidate_count']}_choice", world, row["correct"])
        parent = by_id.get(row["parent_row_id"])
        if row["variant"] in ROBUST_VARIANTS:
            if parent is None:
                raise RuntimeError("robustness row has no source decision")
            invariant = row["source_chosen"] == parent["source_chosen"]
            metrics.add("permutation_rename_reorder", world, invariant)
            metrics.add(row["variant"] + "_invariance", world, invariant)
            mapping = ir[row["row_id"]]["metadata"].get("identity_map", {})
            mapped_scores = {mapping.get(cid, cid): score for cid, score in row["candidate_scores"].items()}
            parent_scores = parent["candidate_scores"]
            if set(mapped_scores) != set(parent_scores):
                raise RuntimeError("robustness identity mapping changed the source candidate set")
            comparable = [abs(mapped_scores[cid] - parent_scores[cid]) for cid in mapped_scores
                          if mapped_scores[cid] is not None and parent_scores[cid] is not None]
            known_invariant = all((mapped_scores[cid] is None) == (parent_scores[cid] is None) for cid in mapped_scores)
            pairs.append({"kind": row["variant"], "world_id": world, "parent_row_id": parent["row_id"],
                          "new_row_id": row["row_id"], "mapping_invariant": invariant,
                          "mapped_candidate_scores": mapped_scores, "source_candidate_scores": parent_scores,
                          "candidate_knownness_invariant": known_invariant,
                          "maximum_score_deviation": max(comparable, default=0.0)})
        if row["variant"] in {"relevant_page_grade_change", "relevant_page_erasure", "relevant_page_contradiction"}:
            if parent is None:
                raise RuntimeError("intervention row has no source decision")
            add_causal(metrics, pairs, parent, row, "grade_swap" if row["variant"] == "relevant_page_grade_change"
                       else row["variant"])
    # Reusing the exact same base state under every distinct supported atomic criterion.
    criterion_groups = defaultdict(list)
    for row in decisions:
        if is_primary(row):
            criterion_groups[row["world_id"]].append(row)
    for rows in criterion_groups.values():
        for old in rows:
            for new in rows:
                if old["row_id"] != new["row_id"]:
                    add_causal(metrics, pairs, old, new, "criterion_swap")
    for i, record in enumerate(data["records"]):
        row = by_id[record["row_id"]]
        if not is_primary(row):
            continue
        world = record["world_id"]
        mask = np.asarray(record["grade_mask"], dtype=bool)
        orientation_mask = np.asarray(record["orientation_mask"], dtype=bool)
        p = len(record["pages"])
        if mask.any():
            target = np.asarray(record["directed_grade_target"] if control == "cross" else record["grade_target"], dtype=float)
            error = np.abs(fields["raw_value"][i, :p][mask] - target[mask])
            metrics.add("ordinal_MAE", world, float(error.sum()), len(error))
            truth = term_truth(record)
            metrics.add("candidate_directed_ordinal_MAE", world, abs(float(fields["score"][i]) - truth))
            for prefix, use_sigma in (("raw_distribution_", .1), ("calibrated_distribution_", sigma)):
                distribution_events(metrics, prefix, world, float(fields["score"][i]), truth, use_sigma)
                if row["family"][0] in families[12:]:
                    distribution_events(metrics, "OOD_" + prefix, world, float(fields["score"][i]), truth, use_sigma)
        if orientation_mask.any():
            truth_orientation = np.asarray(record["orientation_target"], dtype=float)[orientation_mask]
            predicted = np.sign(fields["direction"][i, :p][orientation_mask])
            metrics.add("orientation", world, int((predicted == truth_orientation).sum()), len(predicted))
        relevance = np.asarray(record["relevance_target"], dtype=float)
        if p and relevance.sum() > 0:
            attention = fields["attention"][i, :p]
            maxima = np.flatnonzero(np.isclose(attention, attention.max(), rtol=0, atol=1e-12))
            selected = min(maxima, key=lambda j: record["pages"][j]["block_id"])
            metrics.add("page_attribution", world, relevance[selected] > 0)
            metrics.add("matched_page_attention_mass", world, float(attention[relevance > 0].sum()))
    return metrics, pairs


def add_causal(metrics: Metrics, pairs: list, old: dict, new: dict, kind: str) -> None:
    teacher_changed = set(old["source_gold"]) != set(new["source_gold"])
    student_changed = old["source_chosen"] != new["source_chosen"]
    correct_new = new["source_chosen"] in new["source_gold"]
    world = new["world_id"]
    pair = {"kind": kind, "world_id": world, "old_row_id": old["row_id"], "new_row_id": new["row_id"],
            "old_gold": old["source_gold"], "new_gold": new["source_gold"],
            "old_chosen": old["source_chosen"], "new_chosen": new["source_chosen"],
            "teacher_changed": teacher_changed, "student_changed": student_changed,
            "correct_new": correct_new, "changed_to_new": student_changed and correct_new,
            "new_supported": new["supported"]}
    pairs.append(pair)
    if teacher_changed:
        metrics.add(kind + "_teacher_changing_correct_new", world, correct_new)
        metrics.add(kind + "_changed_to_new", world, student_changed and correct_new)
        if kind in {"criterion_swap", "grade_swap"} and new["supported"]:
            metrics.add("teacher_changing_correct_new", world, correct_new)
            metrics.add("teacher_changing_changed_to_new", world, student_changed and correct_new)
    else:
        metrics.add(kind + "_teacher_stable_unwanted_changes", world, student_changed)
        if kind in {"criterion_swap", "grade_swap"}:
            metrics.add("teacher_stable_unwanted_changes", world, student_changed)


def bootstrap_indices(directory: Path, worlds: list[str]) -> tuple[np.ndarray, dict]:
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(worlds), size=(BOOTSTRAPS, len(worlds)), dtype=np.int32)
    path = directory / "world_bootstrap_indices.npy"
    with path.open("xb") as stream:
        np.save(stream, indices, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    weights = np.zeros((BOOTSTRAPS, len(worlds)), dtype=np.float64)
    for i, draw in enumerate(indices):
        weights[i] = np.bincount(draw, minlength=len(worlds))
    receipt = {**artifact(path), "world_ids": worlds, "draws": BOOTSTRAPS, "seed": BOOTSTRAP_SEED,
               "unit": "whole world; variants clustered", "macro_weighting": "equal family weights"}
    return weights, receipt


def ece_summary(metrics: Metrics, weights: np.ndarray, prefix: str) -> dict:
    bins = []
    for bucket in range(10):
        accuracy = prefix + f"ECE_bin{bucket}_accuracy"
        confidence = prefix + f"ECE_bin{bucket}_confidence"
        if accuracy in metrics.values:
            a = metrics.values[accuracy]["all"]
            c = metrics.values[confidence]["all"]
            bins.append((a, c))
    if not bins:
        return {"point": None, "CI95": None, "denominator": 0}
    def compute(w):
        total, error = 0, 0
        for a, c in bins:
            av = a.sum(axis=0) if w is None else w @ a
            cv = c.sum(axis=0) if w is None else w @ c
            total = total + av[..., 1]
            error = error + np.abs(av[..., 0] - cv[..., 0])
        return np.divide(error, total, out=np.full_like(np.asarray(error), np.nan, dtype=float), where=total > 0)
    samples = compute(weights)
    valid = samples[np.isfinite(samples)]
    return {"point": float(compute(None)), "CI95": np.quantile(valid, [.025, .975]).tolist() if len(valid) else None,
            "denominator": float(sum(a[:, 1].sum() for a, _ in bins)), "bins": 10}


def comparisons(ledgers: dict[str, Metrics], decisions: dict[str, list], weights: np.ndarray) -> dict:
    from scipy.stats import beta
    pages = ledgers["pages"]
    primary = "atomic_choice_macro"
    pages_samples = pages.estimate(primary, weights)
    result = {}
    for alternative in ("cross", "cosine", "lexical", "query_blind", *INTERVENTIONS):
        other = ledgers[alternative]
        if set(pages.values[primary]) != set(other.values[primary]):
            raise RuntimeError("paired primary comparison changed the fixed family population")
        difference = pages_samples - other.estimate(primary, weights)
        valid = difference[np.isfinite(difference)]
        result["pages_minus_" + alternative] = {
            "point": float(pages.estimate(primary) - other.estimate(primary)),
            "CI95": np.quantile(valid, [.025, .975]).tolist() if len(valid) else None,
            "one_sided95_lower": float(np.quantile(valid, .05)) if len(valid) else None,
            "valid_bootstrap_draws": len(valid)}
    cross_rows = {row["row_id"]: row for row in decisions["cross"] if is_primary(row)}
    world_differences = defaultdict(list)
    discordant = set()
    for row in decisions["pages"]:
        if not is_primary(row):
            continue
        cross = cross_rows[row["row_id"]]
        delta = int(row["exact_winner_set_correct"]) - int(cross["exact_winner_set_correct"])
        world_differences[row["world_id"]].append(delta)
        if delta:
            discordant.add(row["world_id"])
    # A world is harmed if any primary paired choice was worsened; improvement
    # elsewhere in that world cannot cancel the sparse-discordance guard.
    harmed = sum(any(value < 0 for value in values) for values in world_differences.values())
    n = len(world_differences)
    if n == 0:
        raise RuntimeError("internal cross NI has no paired primary worlds")
    upper = 1.0 if harmed == n else float(beta.ppf(.95, harmed + 1, n - harmed))
    lower = result["pages_minus_cross"]["one_sided95_lower"]
    sparse = len(discordant) < 20
    ni_pass = lower is not None and lower > -.05 and (not sparse or upper < .05)
    result["internal_cross_NI"] = {"margin": .05, "one_sided95_lower": lower,
                                   "discordant_worlds": len(discordant), "harmed_worlds": harmed,
                                   "world_count": n, "CP_harmed_world_upper95": upper,
                                   "sparse_guard_required": sparse, "pass": ni_pass,
                                   "status": "passes_internal_screen" if ni_pass else "inconclusive_or_fails",
                                   "endgame_NI": False}
    benefits = {}
    for alternative in ("lexical", "zero_question", "zero_pages"):
        difference = pages_samples - ledgers[alternative].estimate(primary, weights)
        valid = difference[np.isfinite(difference)]
        lower = float(np.quantile(valid, .05 / 3)) if len(valid) else None
        benefits["pages_minus_" + alternative] = {"one_sided_Bonferroni_lower": lower,
                                                  "alpha": .05 / 3, "pass": lower is not None and lower > 0}
    result["learned_benefits"] = {"differences": benefits, "pass": all(item["pass"] for item in benefits.values()),
                                 "familywise_alpha": .05, "claims": 3}
    return result


def gate_report(summary: dict, cfg: dict) -> dict:
    gates = cfg["gates"]
    rules = {"atomic_choice_macro": ("atomic_choice_macro", False),
             "held4_family_choice_macro": ("held4_family_choice_macro", False),
             "ordinal_MAE_max": ("ordinal_MAE", True), "orientation": ("orientation", False),
             "page_attribution": ("page_attribution", False), "UNKNOWN_recall": ("UNKNOWN_recall", False),
             "UNKNOWN_precision": ("UNKNOWN_precision", False), "supported_coverage": ("supported_coverage", False),
             "composition": ("composition", False), "teacher_changing_correct_new": ("teacher_changing_correct_new", False),
             "permutation_rename_reorder": ("permutation_rename_reorder", False)}
    result = {}
    for gate, (metric, upper) in rules.items():
        observed = summary.get(metric, {}).get("point")
        available = observed is not None and math.isfinite(observed)
        result[gate] = {"threshold": gates[gate], "observed": observed if available else None,
                        "pass": (observed <= gates[gate] if upper else observed >= gates[gate]) if available else None,
                        "status": "evaluated_point_screen" if available else "unavailable_in_this_phase"}
    for kind in ("criterion_swap", "grade_swap"):
        metric = kind + "_teacher_changing_correct_new"
        observed = summary.get(metric, {}).get("point")
        available = observed is not None and math.isfinite(observed)
        result[metric] = {"threshold": gates["teacher_changing_correct_new"],
                          "observed": observed if available else None,
                          "pass": observed >= gates["teacher_changing_correct_new"] if available else None,
                          "status": "evaluated_point_screen" if available else "unavailable_in_this_phase"}
    result["exact_literals"] = {"threshold": 1.0, "pass": None,
                               "status": "closed_regression_prerequisite_not_supplied; parent_owned"}
    return result


def evaluate_phase(phase: str, run_root: Path, calibrations: dict | None, device: str,
                   receipt: Path | None = None) -> tuple[dict, dict | None]:
    from ephemeral_pages_train import load_control, predict_control
    cfg = protocol()
    if phase == "final":
        if receipt is None:
            raise RuntimeError("final evaluation requires sealed calibration")
        verified = validate_final_receipt(receipt)
        sealed = json.loads(Path(verified["calibration_file"]["path"]).read_text(encoding="utf-8"))
        if calibrations != sealed.get("controls"):
            raise RuntimeError("final evaluation cannot override sealed calibration")
        if checkpoint_files(run_root) != verified["checkpoint_files"]:
            raise RuntimeError("final checkpoint files differ from selection receipt")
    data, ir = load_data(phase, receipt)
    expected_worlds = cfg["corpus"]["worlds"][phase]
    observed_worlds = {record["world_id"] for record in data["records"]}
    if len(observed_worlds) != expected_worlds:
        raise RuntimeError("captured evaluation does not contain the frozen whole-world count")
    expected_families = cfg["corpus"]["families"][:12 if phase == "calibration" else 16 if phase == "development" else 20]
    observed_families = {family for row in ir.values() if row["metadata"]["variant"] == "base"
                         and row["metadata"]["query_kind"] == "atomic" and UNKNOWN not in row["gold"]
                         for family in row["metadata"]["family"]}
    if observed_families != set(expected_families):
        raise RuntimeError("supported base atomic population omits a frozen family")
    directory = run_root / "evaluation" / phase
    directory.mkdir(parents=True, exist_ok=False)
    worlds = sorted({record["world_id"] for record in data["records"]})
    if not worlds:
        raise RuntimeError("empty evaluation phase")
    weights, bootstrap = bootstrap_indices(directory, worlds)
    ledgers, all_decisions, summaries, saved = {}, {}, {}, {}
    if calibrations is None:
        if phase != "calibration":
            raise RuntimeError("only calibration worlds may fit calibration")
        calibrations = {}
    for control in CONTROLS:
        model, normalizer = load_control(control, run_root, device=device)
        modes = (None, *INTERVENTIONS) if control == "pages" and phase != "calibration" else (None,)
        for intervention in modes:
            name = intervention or control
            output = predict_control(control, model, normalizer, data, intervention=intervention)
            fields = prediction_fields(output, data)
            if phase == "calibration":
                calibrations[control] = calibrate(fields, data, ir)
            cal = calibrations[control]
            decisions = aggregate(fields, data, ir, cal)
            saved[name] = save_predictions(directory, name, fields, data, ir, decisions, cal["ordinal_sigma"])
            ledger, pairs = metric_ledger(control, fields, data, ir, decisions, cal["ordinal_sigma"], cfg["corpus"]["families"], worlds)
            ledgers[name], all_decisions[name] = ledger, decisions
            summaries[name] = ledger.summaries(weights)
            for prefix in ("raw_distribution_", "calibrated_distribution_", "OOD_raw_distribution_", "OOD_calibrated_distribution_"):
                summaries[name][prefix + "ECE"] = ece_summary(ledger, weights, prefix)
            saved[name]["world_metric_ledger"] = write_json(directory / f"{name}_world_metrics.json", ledger.ledger())
            pair_path = directory / f"{name}_paired_interventions.jsonl"
            with pair_path.open("x", encoding="utf-8") as stream:
                for pair in pairs:
                    stream.write(json.dumps(plain(pair), sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            saved[name]["paired_interventions"] = artifact(pair_path)
            del output, fields
        del model, normalizer
    paired = comparisons(ledgers, all_decisions, weights) if phase != "calibration" else None
    screens = {name: gate_report(summary, cfg) for name, summary in summaries.items()}
    if paired is not None:
        causal_pass = all(screens["pages"][kind + "_teacher_changing_correct_new"]["pass"] is True
                          for kind in ("criterion_swap", "grade_swap"))
        screens["pages"]["mechanism"] = {"pass": paired["learned_benefits"]["pass"] and causal_pass,
                                          "corrected_learned_benefits": paired["learned_benefits"],
                                          "required_causal_quality_pass": causal_pass}
        screens["pages"]["internal_cross_NI_margin"] = paired["internal_cross_NI"]
    result = {"schema": "vey.eca.evaluation.v1", "phase": phase,
              "protocol_sha256": PROTOCOL_SHA256, "checkpoint_files": checkpoint_files(run_root),
              "bootstrap": bootstrap, "metrics": summaries, "artifacts": saved,
              "gate_screens": screens, "comparisons": paired,
              "capture_lineage": data["lineage"], "capture_counters": data["counters"],
              "capture_token_receipts": data["token_receipts"], "evaluation_encoder_forwards": 0,
              "closed_regression": {"status": "missing_external_prerequisite", "owner": "parent",
                                    "used_for_selection": False},
              "runtime": {"status": "missing_external_prerequisite", "owner": "parent", "performance_claim": False},
              "scope": cfg["scope"], "statistical_scope": cfg["statistical_scope_clarification"],
              "implementation_sha256": sha256_file(Path(__file__)),
              "metric_populations": {"primary": "supported base atomic K4 exact maximal-winner-set equality; equal family macro; concrete winner-in-gold reported separately",
                                     "UNKNOWN": "base plus paired erasure/contradiction",
                                     "ordinal_orientation_attribution": "supported base atomic K4 candidates",
                                     "criterion_swaps": "ordered distinct base supported atomic criteria within the same world",
                                     "teacher_change": "maximal gold sets differ; correct-new needs no student change"},
              "certificate": "unavailable", "promotion": False, "B_STEF_allowed": False,
              "endgame_complete": False, "final_outcomes_used_for_selection": False}
    manifest = write_json(directory / "evaluation.json", result)
    return {"manifest": manifest, "results": result}, calibrations


def seal_selection(run_root: Path, calibration: dict, development: dict) -> Path:
    calibration_path = run_root / "calibration.json"
    calibration_file = write_json(calibration_path, {
        "schema": "vey.eca.calibration.v1", "protocol_sha256": PROTOCOL_SHA256,
        "controls": calibration, "final_outcomes_used": False,
        "calibration_evaluation": artifact(run_root / "evaluation" / "calibration" / "evaluation.json"),
        "checkpoint_files": checkpoint_files(run_root),
        "fit_split": "calibration", "certificate": "unavailable"})
    selection_file = write_json(run_root / "selection.json", {
        "schema": "vey.eca.selection.v1", "protocol_sha256": PROTOCOL_SHA256,
        "eligible_arm": "pages", "eligible_arms": ["pages"],
        "selection_rule": "Protocol fixes pages; development controls cannot select another architecture",
        "development_evaluation": development["manifest"], "final_outcomes_used": False,
        "gate_screens": development["results"]["gate_screens"],
        "comparisons": development["results"]["comparisons"],
        "closed_regression": development["results"]["closed_regression"],
        "runtime": development["results"]["runtime"],
        "promotion": False, "B_STEF_allowed": False, "endgame_complete": False})
    receipt = run_root / "selection_calibration_receipt.json"
    write_json(receipt, {"schema": "vey.eca.selection-calibration.v1",
                         "protocol_sha256": PROTOCOL_SHA256, "eligible_arm": "pages",
                         "eligible_arms": ["pages"], "final_outcomes_used": False,
                         "checkpoint_files": checkpoint_files(run_root),
                         "selection_file": selection_file, "calibration_file": calibration_file,
                         "development_evaluation": development["manifest"],
                         "corpus_build_manifest": artifact(capture.CORPUS / "build_manifest_v1.json"),
                         "amendment_file": artifact(PROTOCOL.with_name("ephemeral_pages_grade_amendment.json"))})
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "final"), default="development")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    protocol()
    if args.phase == "final":
        if args.receipt is None:
            parser.error("final requires explicit --receipt; development is the default")
        verified = validate_final_receipt(args.receipt)
        if checkpoint_files(args.run_root) != verified["checkpoint_files"]:
            raise RuntimeError("run checkpoints differ from sealed receipt")
        calibration_path = Path(verified["calibration_file"]["path"])
        sealed = json.loads(calibration_path.read_text(encoding="utf-8"))
        if sealed.get("protocol_sha256") != PROTOCOL_SHA256 or sealed.get("final_outcomes_used") is not False:
            raise RuntimeError("invalid sealed calibration")
        if set(sealed["controls"]) != set(CONTROLS):
            raise RuntimeError("sealed calibration omits a fixed control")
        result, _ = evaluate_phase("final", args.run_root, sealed["controls"], args.device, args.receipt)
        print(json.dumps({"phase": "final", "manifest": result["manifest"], "promotion": False}, sort_keys=True))
    else:
        if args.receipt is not None:
            parser.error("development does not consume a final receipt")
        checkpoints = checkpoint_files(args.run_root)
        _, calibration = evaluate_phase("calibration", args.run_root, None, args.device)
        development, _ = evaluate_phase("development", args.run_root, calibration, args.device)
        if checkpoint_files(args.run_root) != checkpoints:
            raise RuntimeError("checkpoint files changed during train-free evaluation")
        receipt = seal_selection(args.run_root, calibration, development)
        print(json.dumps({"phase": "development", "manifest": development["manifest"],
                          "receipt": artifact(receipt), "final_opened": False, "promotion": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
