import pytest

from research.endgame import native_field_verify as verify


def test_fractional_component_bootstrap_preserves_micro_estimand():
    rows = [
        {"id": str(i), "component_id": "large", "group_id": str(i)}
        for i in range(3)
    ] + [{"id": "3", "component_id": "small", "group_id": "3"}]
    result = verify.component_ratio_bootstrap(
        rows, {"normalized_RPS": [.1, .1, .1, -.2]}, resamples=200, seed=0)
    metric = result["normalized_RPS"]
    assert metric["observed_delta"] == pytest.approx(.025)
    assert metric["ci95_bootstrap"] == pytest.approx([-.2, .1])
    assert metric["components"] == 2
    assert metric["rows"] == 4


def test_semantic_ties_survive_coordinate_reversal():
    ids = ["z", "a", "middle"]
    masses = [.5, .5, 0.]
    assert verify.semantic_winner(ids, masses) == "a"
    assert verify.semantic_winner(ids[::-1], masses[::-1]) == "a"
    assert verify.canonical_readout(ids[::-1], masses[::-1], "choice") == {
        "answer": "a", "ties": ["a", "z"]}


def test_ordinal_reversal_uses_native_levels_not_coordinate_positions():
    assert verify.canonical_readout(["2", "1", "0"], [.75, .25, 0.], "score") == {
        "expected_level": 1.75, "cdf": [0., .25, 1.]}


def test_cyclic_teacher_change_can_reuse_a_donor():
    rows = [
        {"id": "a", "endpoint": "native.intent", "task": "choice", "component_id": "x", "gold": "zero"},
        {"id": "b", "endpoint": "native.intent", "task": "choice", "component_id": "x", "gold": "zero"},
        {"id": "c", "endpoint": "native.intent", "task": "choice", "component_id": "y", "gold": "one"},
    ]
    assert verify.donor_plan(rows[::-1]) == {"a": "c", "b": "c", "c": "a"}
    for row in rows:
        row["gold"] = "same"
    assert verify.donor_plan(rows) == {"a": None, "b": None, "c": None}


@pytest.mark.parametrize("masses", [[.4, .4], [-.1, 1.1], [float("nan"), 1.], [True, 0.]])
def test_invalid_probability_mass_is_not_silently_normalized(masses):
    with pytest.raises(ValueError):
        verify.distribution(masses, 2, "test")


def test_missing_rater_distribution_is_insufficient_not_a_model_failure():
    cfg = {"screen_gates": {
        "ordinal_prior_minus_model_normalized_RPS_lower_nominal_bound": .005,
        "ordinal_model_minus_prior_nMAE_upper_nominal_bound": .01,
        "scope": "development only",
    }}
    report = {"metrics": {}, "controls": {"teacher_changing_rows": 100, "unavailable_rows": 0},
              "uncertainty": {"prior_minus_model_normalized_RPS_at_least3": None,
                              "model_minus_prior_nMAE": {"ci95_bootstrap": [.02, .03]}}}
    assert verify.screen_field("massive.grammar_score", report, True, cfg)["status"] == "data-insufficient"
    report["uncertainty"]["prior_minus_model_normalized_RPS_at_least3"] = {"ci95_bootstrap": [.02, .03]}
    assert verify.screen_field("massive.grammar_score", report, True, cfg)["status"] == "failed"
    report["uncertainty"]["model_minus_prior_nMAE"]["ci95_bootstrap"] = [-.02, -.01]
    assert verify.screen_field("massive.grammar_score", report, True, cfg)["status"] == "passed"
    report["controls"]["unavailable_rows"] = 1
    assert verify.screen_field("massive.grammar_score", report, True, cfg)["status"] == "data-insufficient"
