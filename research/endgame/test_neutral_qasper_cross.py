"""Consumer-visible reconstruction boundaries, using no neural model or corpus."""
from collections import Counter
import unittest

try:
    from . import neutral_qasper_cross_verify as verify
except ImportError:
    import neutral_qasper_cross_verify as verify


def window(block, indices, offsets, tags, strength=5):
    logits = [[strength if label == tag else 0 for label in range(3)] for tag in tags]
    return {"block_id": block, "window_index": 0, "source_token_indices": indices,
            "offsets": offsets, "bio_logits": logits}


def array_target(values, endpoint=verify.EXTRACTION, blocks=None):
    ratings = [{"annotation_ordinal": index, "present": present, "native_value": value}
               for index, (present, value) in enumerate(values)]
    return verify.native_target({"ratings": ratings}, blocks or [{"id": "b", "text": "red blue green"}], endpoint)


def prediction(spans=None, ranking=None):
    return {"status": "OK", "spans": spans or [], "ranking": ranking or [], "candidate_ids": [],
            "raw_probabilities": [], "probabilities": []}


def record(target, endpoint=verify.EXTRACTION, identity="r", component="c"):
    return {"id": identity, "component_id": component, "endpoint": endpoint, "target": target,
            "candidate_ids": [], "blocks": [{"id": "b", "text": "red blue green"}]}


class SourceDecodeTests(unittest.TestCase):
    def test_orphan_inside_starts_verbatim_unicode_span(self):
        blocks = [{"id": "b", "text": "écho, blue!"}]
        result = verify.decode_bio(blocks, [window("b", [0, 1], [[0, 4], [6, 10]], [2, 2])])
        self.assertEqual([(x["start"], x["end"], x["text"]) for x in result], [(0, 10, "écho, blue")])

    def test_adjacency_does_not_merge_and_blocks_never_merge(self):
        blocks = [{"id": "b", "text": "redblue"}, {"id": "a", "text": "green"}]
        result = verify.decode_bio(blocks, [window("b", [0, 1], [[0, 3], [3, 7]], [1, 1]),
                                           window("a", [0], [[0, 5]], [2])])
        self.assertEqual([(x["block_id"], x["text"]) for x in result], [("b", "red"), ("b", "blue"), ("a", "green")])

    def test_overlap_deduplicates_token_confidence_not_span_means(self):
        blocks = [{"id": "b", "text": "red blue green"}]
        first = window("b", [0, 1], [[0, 3], [4, 8]], [1, 2], 2)
        second = window("b", [1, 2], [[4, 8], [9, 14]], [1, 2], 6)
        duplicate = window("b", [0, 1], [[0, 3], [4, 8]], [1, 2], 1)
        result = verify.decode_bio(blocks, [first, second, duplicate])
        low, high = verify.softmax([0, 2, 0])[1], verify.softmax([0, 6, 0])[1]
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "red blue green")
        self.assertAlmostEqual(result[0]["confidence"], (low + high + high) / 3)

    def test_source_order_survives_trace_order(self):
        blocks = [{"id": "z", "text": "red"}, {"id": "a", "text": "blue"}]
        result = verify.decode_bio(blocks, [window("a", [0], [[0, 4]], [1]), window("z", [0], [[0, 3]], [1])])
        self.assertEqual([x["block_id"] for x in result], ["z", "a"])

    def test_zero_width_token_cannot_generate_or_extend_text(self):
        result = verify.decode_bio([{"id": "b", "text": "red"}],
                                  [window("b", [0, 1], [[0, 0], [0, 3]], [1, 2])])
        self.assertEqual(result[0]["text"], "red")
        self.assertEqual((result[0]["start"], result[0]["end"]), (0, 3))
        continued = verify.decode_bio([{"id": "b", "text": "red blue"}],
                                     [window("b", [0, 1, 2], [[0, 3], [3, 3], [4, 8]], [1, 1, 2])])
        self.assertEqual([x["text"] for x in continued], ["red blue"])

    def test_ties_choose_outside_and_invalid_offsets_fail(self):
        self.assertEqual(verify.decode_bio([{"id": "b", "text": "red"}],
                                          [window("b", [0], [[0, 3]], [0], 0)]), [])
        for offsets, indices in (([[0, 4]], [0]), ([[-1, 2]], [0]), ([[0, 1], [1, 3]], [0, 2])):
            with self.assertRaises(ValueError):
                verify.decode_bio([{"id": "b", "text": "red"}], [window("b", indices, offsets, [1] * len(indices))])


class NativeSupervisionTests(unittest.TestCase):
    def setUp(self):
        self.blocks = [{"id": "b", "text": "red blue green"}]
        self.offsets = {"b": [[0, 3], [4, 8], [9, 14]]}
        self.windows = [window("b", [0, 1, 2], self.offsets["b"], [0, 0, 0])]

    def test_partial_annotation_never_invents_outside_labels(self):
        target = array_target([(True, ["red", "absent"])])
        result = verify.supervision(target, self.blocks, self.offsets, self.windows)
        self.assertEqual(result["token_distributions"]["b"], [[0, 1, 0], None, None])
        self.assertEqual(result["token_annotation_counts"]["b"], [1, 0, 0])
        self.assertTrue(result["annotations"][0]["positive_only"])
        self.assertEqual(result["supervised_source_tokens"], 1)

    def test_missing_null_empty_and_wholly_unsupported_are_unlabelled(self):
        target = array_target([(False, None), (True, None), (True, []), (True, ["absent"])])
        result = verify.supervision(target, self.blocks, self.offsets, self.windows)
        self.assertEqual(result["token_distributions"]["b"], [None, None, None])
        self.assertEqual(result["supervised_source_tokens"], 0)
        sample = verify.score_row(record(target), prediction())
        self.assertEqual(sample["missing_annotations"], 1)
        self.assertEqual(sample["null_annotations"], 1)
        self.assertEqual(sample["empty_annotations"], 1)
        self.assertEqual(sample["f1"], [0, 1])
        self.assertEqual(sample["exact_item_recall"], [0, 1])

    def test_full_and_partial_annotations_have_distinct_token_denominators(self):
        target = array_target([(True, ["red"]), (True, ["blue", "absent"])])
        result = verify.supervision(target, self.blocks, self.offsets, self.windows)
        self.assertEqual(result["token_distributions"]["b"], [[0, 1, 0], [0.5, 0.5, 0], [1, 0, 0]])
        self.assertEqual(result["token_annotation_counts"]["b"], [1, 2, 1])

    def test_char_match_without_token_boundary_or_complete_window_is_unsupported(self):
        target = array_target([(True, ["ed"]), (True, ["red blue"])])
        fragmented = [window("b", [0], [[0, 3]], [0]), window("b", [1, 2], [[4, 8], [9, 14]], [0, 0])]
        result = verify.supervision(target, self.blocks, self.offsets, fragmented)
        self.assertEqual(result["supervised_source_tokens"], 0)
        self.assertEqual(target["unmatched_item_count"], 0)
        self.assertTrue(all(not item["supported_occurrences"] for annotation in result["annotations"] for item in annotation["items"]))
        self.assertEqual(result["support_counts"]["token_boundary_unsupported_items"], 1)
        self.assertEqual(result["support_counts"]["window_length_unsupported_items"], 1)

    def test_duplicate_block_matches_share_one_native_item_mass(self):
        blocks = [{"id": "a", "text": "red"}, {"id": "b", "text": "red"}, {"id": "c", "text": "blue"}]
        target = array_target([(True, ["red", "blue", "absent"])], verify.RETRIEVAL, blocks)
        rec = record(target, verify.RETRIEVAL)
        rec["candidate_ids"] = ["a", "b", "c"]
        self.assertEqual(verify.categorical_target(rec), [0.25, 0.25, 0.5])
        pred = {**prediction(ranking=["b", "c", "a"]), "candidate_ids": ["a", "b", "c"],
                "probabilities": [0.2, 0.5, 0.3], "raw_probabilities": [0.2, 0.5, 0.3]}
        sample = verify.score_row(rec, pred)
        self.assertEqual(sample["recall@1"], [1, 3])
        self.assertEqual(sample["recall@10"], [2, 3])

    def test_null_boolean_is_not_false_and_answerability_complements_only_bools(self):
        ledger = {"ratings": [{"present": True, "native_value": None}, {"present": False, "native_value": None},
                              {"present": True, "native_value": True}, {"present": True, "native_value": False}]}
        self.assertEqual(verify.native_target(ledger, [], verify.BOOL[1])["counts"], [1, 1])
        missing = verify.native_target({"ratings": ledger["ratings"][:2]}, [], verify.BOOL[0])
        self.assertIsNone(missing["distribution"])
        self.assertEqual(missing["observed_raters"], 0)


class MetricBoundaryTests(unittest.TestCase):
    def test_multispan_counter_f1_preserves_multiplicity(self):
        self.assertAlmostEqual(verify.token_f1(["Red red", "blue"], ["red blue"]), 4 / 5)
        self.assertEqual(verify.tokens("ÉCHO, echo; x_1"), Counter({"écho": 1, "echo": 1, "x_1": 1}))

    def test_punctuation_only_native_annotation_is_zero_not_successful_abstention(self):
        target = array_target([(True, [""]), (True, ["..."]), (True, ["red"])])
        sample = verify.score_row(record(target), prediction())
        self.assertEqual(sample["f1"], [0, 3])
        self.assertEqual(sample["exact_set_match"], [0, 3])
        self.assertEqual(sample["exact_item_recall"], [0, 3])
        self.assertEqual(verify.token_f1(["", "??"], ["..."]), 0)

    def test_annotation_denominator_excludes_missing_but_not_unmatched(self):
        target = array_target([(True, ["red", "absent"]), (True, ["blue"]), (True, []), (True, None)])
        spans = [{"block_id": "b", "start": 0, "end": 3, "text": "red", "confidence": 0.9}]
        sample = verify.score_row(record(target), prediction(spans))
        self.assertEqual(sample["f1"][1], 2)
        self.assertAlmostEqual(sample["f1"][0], 2 / 3)
        self.assertEqual(sample["exact_set_match"], [0, 2])
        self.assertEqual(sample["exact_item_recall"], [1, 3])

    def test_zero_native_items_cannot_earn_recall_or_extraction_success(self):
        target = array_target([(True, []), (True, None)])
        summary = verify.summarize([verify.score_row(record(target), prediction())])
        self.assertIsNone(summary["metrics"]["f1"]["value"])
        self.assertIsNone(summary["metrics"]["exact_item_recall"]["value"])
        self.assertIsNone(verify.ratio(0, 0))

    def test_confidence_auc_ties_and_empty_selective_denominators(self):
        metrics = verify.calibration_metrics([0.5, 0.5], [1, 0])
        self.assertEqual(metrics["confidence_AUROC"], 0.5)
        self.assertEqual(metrics["ECE15"], 0)
        self.assertEqual(metrics["selective"][-1]["selected"], 0)
        self.assertIsNone(metrics["selective"][-1]["risk"])
        self.assertIsNone(verify.calibration_metrics([1], [1])["confidence_AUROC"])

    def test_component_ratio_uses_descendant_counts_not_mean_component_scores(self):
        left = {"large": {"component_id": "a", "recall": [90, 100]}, "small": {"component_id": "b", "recall": [0, 1]}}
        right = {"large": {"component_id": "a", "recall": [0, 100]}, "small": {"component_id": "b", "recall": [1, 1]}}
        result = verify.paired_interval(left, right, "recall", ["a", "b"], resamples=128)
        self.assertAlmostEqual(result["estimate"], 89 / 101)
        self.assertNotAlmostEqual(result["estimate"], -0.05)
        self.assertEqual(result["denominator"], 101)

    def test_missing_or_wrong_paired_denominator_fails_closed(self):
        one = {"r": {"component_id": "c", "accuracy": [1, 1]}}
        for other, universe in (({"r": {"component_id": "c"}}, ["c"]),
                                ({"r": {"component_id": "c", "accuracy": [0, 2]}}, ["c"]),
                                (one, ["foreign"]), (one, ["c", "c"])):
            with self.assertRaises(ValueError):
                verify.paired_interval(one, other, "accuracy", universe, resamples=4)
        empty = {"r": {"component_id": "c", "accuracy": [0, 0]}}
        self.assertFalse(verify.interval_pass(verify.paired_interval(empty, empty, "accuracy", ["c"], resamples=4)))

    def test_interval_strict_equality_and_malformed_evidence(self):
        interval = {"status": "MEASURED", "denominator": 1, "estimate": 0.5, "ci95": [0, 1]}
        self.assertFalse(verify.interval_pass(interval))
        self.assertTrue(verify.interval_pass(interval, strict=False))
        for malformed in (None, {}, {**interval, "ci95": [1, 0]}, {**interval, "ci95": [float("nan"), 1]},
                          {**interval, "ci95": [0, float("inf")]}, {**interval, "denominator": 0},
                          {**interval, "denominator": "1"}, {**interval, "denominator": True},
                          {**interval, "denominator": float("nan")}, {**interval, "estimate": float("inf")}):
            self.assertFalse(verify.interval_pass(malformed))

    def test_scientific_negative_is_distinct_from_missing_evidence(self):
        summaries = {endpoint: {"metrics": {"accuracy": {"value": 0.2, "denominator": 10},
                                           "recall@10": {"value": 0.2, "denominator": 10},
                                           "f1": {"value": 0.2, "denominator": 10}}} for endpoint in verify.ENDPOINTS}
        receipt = {"status": "MEASURED", "denominator": 10, "estimate": -0.15, "ci95": [-0.2, -0.1]}
        intervals = {endpoint: {"model_minus_baseline": receipt, "model_minus_question_masked": receipt,
                               "baseline_minus_model_brier": receipt, "model_minus_original_cross": receipt}
                     for endpoint in verify.ENDPOINTS}
        swaps = {endpoint: {"counts": {"teacher_changing": 1, "paired": 1}, "correct_new_denominator": 1,
                            "correct_new_changing": 0} for endpoint in verify.BOOL + (verify.RETRIEVAL, verify.EXTRACTION)}
        gates = verify.quality_gates(summaries, intervals, swaps, True, True)
        self.assertTrue(all(gate["status"] == "FAIL" for gate in gates.values()))
        self.assertTrue(all(gate["status"] != "PASS" for gate in verify.quality_gates(summaries, {}, {}, False, False).values()))


class MembershipAndControlTests(unittest.TestCase):
    def test_semantic_tie_ranking_is_coordinate_order_invariant(self):
        self.assertEqual(verify.ranking(["z", "a", "m"], [1, 1, 0]), ["a", "z", "m"])
        self.assertEqual(verify.ranking(["m", "a", "z"], [0, 1, 1]), ["a", "z", "m"])
        with self.assertRaises(ValueError):
            verify.ranking(["a", "a"], [1, 2])

    def test_duplicate_missing_and_foreign_members_are_rejected(self):
        for rows, expected in (([{"id": "a"}, {"id": "a"}], None), ([{"id": "a"}], {"a", "b"}),
                               ([{"id": "a"}, {"id": "x"}], {"a", "b"})):
            with self.assertRaises(ValueError):
                verify.member_index(rows, expected)

    def test_same_paper_same_endpoint_swap_wraps_and_skips_duplicate_questions(self):
        def row(identity, question, paper="p", endpoint=verify.BOOL[0]):
            return {"id": identity, "group_id": "g", "paper_id": paper, "endpoint": endpoint,
                    "serving": {"question": question}}
        rows = {r["id"]: r for r in [row("a", "one"), row("b", "one"), row("c", "two"),
                                     row("d", "one", "other"), row("e", "three", endpoint=verify.BOOL[1])]}
        self.assertEqual(verify.donor_plan(rows), {"a": "c", "b": "c", "c": "a", "d": None, "e": None})

    def test_prefixes_are_input_only_nested_and_do_not_force_native_membership(self):
        ids = [str(index) for index in range(10)]
        first = verify.nested_prefix(ids, "component", 2)
        larger = verify.nested_prefix(list(reversed(ids)), "component", 5)
        self.assertEqual(larger[:2], first)
        self.assertEqual(len(set(larger)), 5)
        self.assertTrue(set(ids) - set(first))
        self.assertEqual(set(verify.nested_prefix(ids, "component", 1000)), set(ids))

    def test_question_and_state_masks_preserve_coordinates_without_fake_source_identity(self):
        view = {"id": "r", "task": "bool", "locale": "en", "state": "red blue green",
                "question": "Is it red?", "candidates": [{"id": "false", "text": "no"}, {"id": "true", "text": "yes"}]}
        rec = {**record({"distribution": [0, 1]}, verify.BOOL[0]), "serving": view,
               "candidate_ids": ["false", "true"]}
        question_masked = verify.changed_record(rec, "question_masked")
        state_masked = verify.changed_record(rec, "state_masked")
        self.assertEqual(question_masked["serving"]["question"], "")
        self.assertEqual(question_masked["blocks"], rec["blocks"])
        self.assertEqual(state_masked["serving"]["question"], view["question"])
        self.assertEqual(state_masked["candidate_ids"], ["false", "true"])
        self.assertEqual(state_masked["blocks"], [{"id": "b", "text": verify.EMPTY_STATE}])
        self.assertNotEqual(state_masked["input_sha256"], verify.value_digest(view))
        subset = verify.changed_record(rec, "candidate_count", subset=["b"])
        self.assertEqual(subset["candidate_ids"], ["false", "true"])
        self.assertEqual(view["state"], "red blue green")

    def test_question_swap_correct_new_does_not_require_student_to_change(self):
        rows = {}
        for identity, distribution in (("a", [1, 0]), ("b", [0, 1])):
            rows[identity] = {**record({"distribution": distribution}, verify.BOOL[0], identity),
                              "candidate_ids": ["false", "true"]}
        preds = {identity: {**prediction(), "candidate_ids": ["false", "true"], "probabilities": [0.1, 0.9],
                            "raw_probabilities": [0.1, 0.9]} for identity in rows}
        results = verify.swap_results(rows, preds, {"a": "b", "b": "a"})
        self.assertEqual(results[verify.BOOL[0]]["counts"]["teacher_changing"], 2)
        self.assertEqual(results[verify.BOOL[0]]["correct_new_changing"], 0.5)

    def test_question_ids_not_endpoint_prefixes_define_distinct_repeat_questions(self):
        rows = {
            "a": {"id": "a", "paper_id": "p", "question_ordinal": 0, "endpoint": verify.EXTRACTION,
                  "serving": {"question": "one"}},
            "b": {"id": "b", "paper_id": "p", "question_ordinal": 0, "endpoint": verify.BOOL[0],
                  "serving": {"question": "Evidence: one"}},
            "c": {"id": "c", "paper_id": "p", "question_ordinal": 1, "endpoint": verify.RETRIEVAL,
                  "serving": {"question": "Is it red?"}},
            "d": {"id": "d", "paper_id": "p", "question_ordinal": 1, "endpoint": verify.EXTRACTION,
                  "serving": {"question": "Is it red?"}},
        }
        self.assertEqual([row["id"] for row in verify.repeat_plan(rows)], ["a", "c"])

    def test_json_rejects_nonfinite_and_duplicate_keys(self):
        for text in ('{"x":NaN}', '{"x":Infinity}', '{"x":1e999}', '{"x":1,"x":2}'):
            with self.assertRaises(ValueError):
                verify.decode(text)
        with self.assertRaises(ValueError):
            verify.canonical({"x": float("nan")})


if __name__ == "__main__":
    unittest.main()
