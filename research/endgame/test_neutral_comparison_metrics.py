import math

import pytest

from research.endgame import neutral_comparison_verify as verify


def test_choice_ece_uses_predicted_confidence_and_answer_correctness():
    rows = [
        {"locale": "en", "candidate_ids": ["a", "b"], "probs": [.1, .9],
         "gold": "a", "answer": "b"},
        {"locale": "en", "probabilities": {"a": 1., "b": 0.},
         "gold": "a", "answer": "a"},
    ]
    result = verify.choice_stats(rows)
    assert result["top1_accuracy"] == .5
    assert result["ece15"] == pytest.approx(.45)
    assert result["nll"] == pytest.approx(-math.log(.1) / 2)
    assert result["brier"] == pytest.approx(.81)


def test_ordinal_distribution_metrics_retain_native_level_distance():
    # First prediction is one level too low; second is correct. Both targets
    # have three raters, so empirical distribution scores include both rows.
    rows = [
        {"probs": [0., 1., 0.], "expected_level": 1., "mean_level": 2.,
         "native_level_max": 2, "observed_raters": 3,
         "target_distribution": [0., 0., 1.]},
        {"probs": [1., 0., 0.], "expected_level": 0., "mean_level": 0.,
         "native_level_max": 2, "observed_raters": 3,
         "target_distribution": [1., 0., 0.]},
    ]
    result = verify.score_stats(rows)
    assert result["mean_abs_error"] == .5
    assert result["normalized_mean_abs_error"] == .25
    assert result["mean_signed_error"] == -.5
    assert result["signed_error_slope_against_rater_mean"] == -.5
    assert result["normalized_ranked_probability_score"] == .25
    assert result["retained_rater_brier"] == 1.
    assert result["spearman_rank_correlation"] is None  # Fewer than three samples.
