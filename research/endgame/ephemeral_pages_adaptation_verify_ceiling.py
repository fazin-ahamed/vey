#!/usr/bin/env python3
"""Reconstruct the saved C1 linear readout without any encoder or optimizer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import ephemeral_pages_adaptation_screens as screens

np = screens.np
require = screens._require


def verify(run_root: Path) -> dict:
    root = run_root.resolve()
    receipt_path = root / screens.RECEIPT_NAMES["c1_adapted_ceiling"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(receipt["gradient_protocol"]["sha256"] == screens.C1_GRADIENT_PROTOCOL_SHA256,
            "saved ceiling protocol differs")
    prerequisite, checkpoint = screens.verify_gradient_prerequisite(root)
    require(receipt["population_sha256"] == prerequisite["population_sha256"]
            and receipt["prerequisite_checkpoint"] == prerequisite["checkpoint"],
            "saved ceiling prerequisite differs")
    arrays_path = Path(receipt["arrays"]["path"])
    require(arrays_path.resolve() == (root / "c1_gradient_ceiling_arrays.npz").resolve()
            and screens.evaluator.artifact(arrays_path) == receipt["arrays"],
            "saved ceiling arrays changed")
    rows_path = Path(receipt["rows"]["path"])
    require(screens.evaluator.artifact(rows_path) == receipt["rows"], "saved ceiling rows changed")
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    stats = checkpoint["stats"]
    mean = np.asarray(stats["mean"], dtype=np.float64)
    std = np.asarray(stats["std"], dtype=np.float64)
    require(mean.shape == std.shape == (384,) and np.isfinite(mean).all()
            and np.isfinite(std).all() and np.all(std > 0), "invalid fixed statistics")
    designs = {}
    for phase, count in (("train", 180), ("development", 240)):
        raw = arrays[f"{phase}_rawfeature"]
        target = arrays[f"{phase}_target"]
        prediction = arrays[f"{phase}_prediction"]
        family = arrays[f"{phase}_family"]
        require(raw.shape == (count, 384) and raw.dtype == np.float32
                and target.shape == prediction.shape == family.shape == (count,)
                and target.dtype == prediction.dtype == np.float64,
                f"invalid saved {phase} population")
        require(np.isfinite(raw).all() and np.isfinite(target).all()
                and np.isfinite(prediction).all(), f"nonfinite {phase} arrays")
        require(len(rows[phase]) == count
                and np.array_equal(target, np.asarray([row["target"] for row in rows[phase]]))
                and np.array_equal(family, np.asarray([row["family"] for row in rows[phase]])),
                f"{phase} target/family custody differs")
        # Independent broadcasting implementation, not screens._normalized_design.
        designs[phase] = (raw.astype(np.float64) - mean[None, :]) / std[None, :]
    require(screens.gradient_population_sha256(rows["train"]) == prerequisite["population_sha256"],
            "saved training population hash differs")
    design, target = designs["train"], arrays["train_target"]
    coefficient, residual, rank, singular = np.linalg.lstsq(design, target, rcond=None)
    original = arrays["coefficients"]
    require(original.shape == (384,) and original.dtype == np.float64
            and np.isfinite(original).all(), "invalid original coefficients")
    scale = max(1.0, float(np.linalg.norm(original)))
    coefficient_difference = float(np.linalg.norm(coefficient - original) / scale)
    require(coefficient_difference <= 1e-7, "saved readout is not the reconstructed minimum-norm solution")
    prediction_differences = {}
    for phase in designs:
        predicted = designs[phase] @ coefficient
        difference = float(np.max(np.abs(predicted - arrays[f"{phase}_prediction"])))
        require(difference <= 1e-8, f"{phase} saved prediction reconstruction differs")
        prediction_differences[phase] = difference
    second, _, second_rank, _ = screens.linalg.lstsq(
        design, target, cond=np.finfo(np.float64).eps * max(design.shape), lapack_driver="gelss")
    second_coefficient_difference = float(np.linalg.norm(second - original) / scale)
    second_prediction_difference = float(np.max(np.abs(design @ second - arrays["train_prediction"])))
    require(second_rank == rank and second_coefficient_difference <= 1e-7
            and second_prediction_difference <= 1e-8, "independent GELSS solution differs")
    error = design @ original - target
    gram, cross = design.T @ design, design.T @ target
    normal_scale = max(1.0, float(np.linalg.norm(gram, ord=np.inf)
                                  * np.linalg.norm(original, ord=np.inf)
                                  + np.linalg.norm(cross, ord=np.inf)))
    normal_residual = float(np.linalg.norm(gram @ original - cross, ord=np.inf) / normal_scale)
    direct_residual = float(np.linalg.norm(design.T @ error, ord=np.inf) / normal_scale)
    require(max(normal_residual, direct_residual) <= 1e-10, "saved readout normal equations fail")
    absolute = np.abs(arrays["development_prediction"] - arrays["development_target"])
    family = arrays["development_family"]
    family_mae = {str(name): float(absolute[family == name].mean()) for name in np.unique(family)}
    macro = float(np.mean(list(family_mae.values())))
    require(abs(macro - receipt["development"]["family_macro_unique_text_MAE"]) <= 1e-12,
            "saved primary metric differs")
    quality_verdict = "pass" if macro <= screens.SCREEN_C_THRESHOLD else "fail"
    require(quality_verdict == receipt["verdict"], "saved gate differs")
    output = {
        "schema": "vey.eca2.c1-saved-ceiling-verification.v1", "evidence_class": "MEASURED",
        "verdict": "pass", "quality_verdict": quality_verdict,
        "protocol_sha256": screens.C1_GRADIENT_PROTOCOL_SHA256,
        "population_sha256": prerequisite["population_sha256"],
        "ceiling_receipt": screens.evaluator.artifact(receipt_path),
        "arrays": receipt["arrays"], "rows": receipt["rows"],
        "checkpoint": prerequisite["checkpoint"],
        "verifier_source": screens.evaluator.artifact(Path(__file__)),
        "environment": screens.trainer._runtime_environment("cpu"),
        "rank": int(rank), "design_shape": list(design.shape),
        "unidentified_coefficient_directions": int(design.shape[1] - rank),
        "retained_singular_condition": float(singular[0] / singular[rank - 1]),
        "train_MSE": float(np.square(error).mean()),
        "relative_coefficient_difference": coefficient_difference,
        "prediction_reconstruction_max_abs": prediction_differences,
        "GELSS_relative_coefficient_difference": second_coefficient_difference,
        "GELSS_train_prediction_max_abs": second_prediction_difference,
        "normal_equation_relative_residual": normal_residual,
        "direct_gradient_relative_residual": direct_residual,
        "development_family_macro_unique_text_MAE": macro,
        "development_micro_unique_text_MAE": float(absolute.mean()),
        "development_MAE_by_family": family_mae,
        "threshold": screens.SCREEN_C_THRESHOLD,
        "encoder_forwards": 0, "optimizer_steps": 0,
        "independent_linear_reconstruction": True,
        "original_saved_coefficients_used_for_verdict": True,
        "final_pool_access": False, "promotion": False, "B_STEF_allowed": False,
    }
    output["receipt"] = screens.write_receipt(
        root / "screen_c1_gradient_ceiling_solver_verification.json", output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    result = verify(parser.parse_args().run_root)
    print(json.dumps({key: result[key] for key in (
        "verdict", "quality_verdict", "development_family_macro_unique_text_MAE",
        "rank", "prediction_reconstruction_max_abs", "receipt")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
