import json
import copy
import hashlib
import random
import struct

import pytest

from research.endgame import native_arch_verify as verify


def _native(endpoint="x.intent", ids=("a", "b"), gold="a"):
    return {
        "id": "dec-1", "endpoint": endpoint, "task": "choice",
        "component_id": "comp-1", "group_id": "grp-1", "locale": "en-US",
        "candidate_ids": list(ids), "gold": gold,
    }


def _row(native, temperature=1.0, donor=None):
    logits = [float(len(native["candidate_ids"]) - i) for i in range(len(native["candidate_ids"]))]
    raw = verify.softmax(logits)
    probs = verify.softmax(logits, temperature)
    answer, ties = verify.choice_winner(native["candidate_ids"], probs)
    raw_answer, raw_ties = verify.choice_winner(native["candidate_ids"], raw)
    row = {
        "id": native["id"], "endpoint": native["endpoint"], "task": native["task"],
        "component_id": native["component_id"], "group_id": native["group_id"],
        "locale": native["locale"], "candidate_ids": list(native["candidate_ids"]),
        "logits": logits, "raw_probs": raw, "probs": probs,
        "answer": answer, "ties": ties, "raw_answer": raw_answer, "raw_ties": raw_ties,
    }
    if donor is not None:
        row["donor_id"] = donor
    return row


def _write(tmp_path, rows):
    path = tmp_path / "dev.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_gold_rank_breaks_ties_lexicographically():
    subset = ["c", "a", "b"]
    probs = [.2, .5, .3]
    assert verify.gold_rank(subset, probs, "c") == 3
    assert verify.gold_rank(subset, probs, "a") == 1
    # a and b tied; the lexicographically smaller ID ranks first.
    assert verify.gold_rank(["a", "b"], [.5, .5], "b") == 2
    assert verify.gold_rank(["a", "b"], [.5, .5], "a") == 1
    assert verify.gold_rank(["a", "b"], [.5, .5], "missing") is None


def test_nested_subset_is_input_only_deterministic_component_seeded_prefix():
    ids = ["g", "m", "z", "a", "q"]
    expected = list(ids)
    seed = int.from_bytes(hashlib.sha256(
        b"native-arch-candidate-count|7|comp-1").digest()[:8], "big")
    random.Random(seed).shuffle(expected)
    first = verify.nested_subset(ids, "comp-1", 2)
    assert first == expected[:2]
    assert verify.nested_subset(ids, "comp-1", 4) == expected[:4]
    assert verify.nested_subset(ids, "comp-1", len(ids)) == expected
    assert ids == ["g", "m", "z", "a", "q"]


def test_choice_winner_is_deterministic_across_coordinate_reversal():
    ids = ["z", "a", "middle"]
    masses = [.5, .5, 0.0]
    assert verify.choice_winner(ids, masses) == ("a", ["a", "z"])
    assert verify.choice_winner(ids[::-1], masses[::-1]) == ("a", ["a", "z"])


def _cfg():
    return {"screen_gates": {
        "cross_intent_accuracy_each_source": 0.80,
        "cross_vs_pooled_accuracy_lower_nominal_bound": 0.0,
        "pages_minus_cross_accuracy_lower_nominal_bound": -0.02,
        "dual_minus_cross_accuracy_lower_nominal_bound": -0.05,
        "scope": "development only",
    }}


def test_screen_arch_enforces_protocol_bounds():
    candidate = {"monotone_membership_frequency": True}
    reversal = {"status": "passed"}
    repeated = {"extra_state_encodes": 0, "extra_encoder_forwards": 0,
                "extra_encoded_states": 0, "novel_catalogue_state_reuse": True}
    passing = {"cross_minus_pooled_accuracy": {"ci95_bootstrap": [0.01, 0.05]}}
    assert verify.screen_arch("cross", 0.85, passing, candidate, reversal, repeated,
                              _cfg())["status"] == "passed"
    low = verify.screen_arch("cross", 0.79, passing, candidate, reversal, repeated, _cfg())
    assert low["status"] == "failed"
    assert "cross_intent_accuracy_each_source" in low["failed_checks"]
    weak = {"cross_minus_pooled_accuracy": {"ci95_bootstrap": [-0.01, 0.04]}}
    assert verify.screen_arch("cross", 0.85, weak, candidate, reversal, repeated,
                              _cfg())["status"] == "failed"
    pages_fail = {"pages_minus_cross_accuracy": {"ci95_bootstrap": [-0.03, 0.01]}}
    assert verify.screen_arch("pages", 0.8, pages_fail, candidate, reversal, repeated,
                              _cfg())["status"] == "failed"
    pages_pass = {"pages_minus_cross_accuracy": {"ci95_bootstrap": [-0.01, 0.03]}}
    assert verify.screen_arch("pages", 0.8, pages_pass, candidate, reversal, repeated,
                              _cfg())["status"] == "passed"
    assert "candidate_count_monotone_membership_frequency" in low["checks"]
    assert "novel_catalogue_state_reuse" in low["checks"]
    for key in ("extra_state_encodes", "extra_encoder_forwards", "extra_encoded_states"):
        assert verify.screen_arch("cross", 0.85, passing, candidate, reversal,
                                  {**repeated, key: 1}, _cfg())["status"] == "failed"
    assert verify.screen_arch(
        "cross", 0.85, passing, candidate, reversal,
        {**repeated, "novel_catalogue_state_reuse": False}, _cfg())["status"] == "failed"


def test_prediction_rows_fail_closed_on_empty_duplicate_and_missing(tmp_path):
    native = _native()
    natives = {"x.intent": {native["id"]: native}}
    temperatures = {"x.intent": 1.0}
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        verify.read_dev_predictions(empty, natives, "x", temperatures, "dev", {})
    with pytest.raises(ValueError):
        verify.read_dev_predictions(_write(tmp_path, [_row(native), _row(native)]),
                                    natives, "x", temperatures, "dev", {})
    with pytest.raises(ValueError):
        verify.read_dev_predictions(_write(tmp_path, []), natives, "x", temperatures, "dev", {})
    rows, unavailable = verify.read_dev_predictions(_write(tmp_path, [_row(native)]),
                                                    natives, "x", temperatures, "dev", {})
    assert rows["x.intent"][native["id"]]["answer"] == "a"
    assert unavailable == set()


def test_prediction_rows_fail_closed_on_unnormalized_probabilities(tmp_path):
    native = _native()
    natives = {"x.intent": {native["id"]: native}}
    bad = _row(native)
    bad["probs"] = [.4, .4]
    with pytest.raises(ValueError):
        verify.read_dev_predictions(_write(tmp_path, [bad]), natives, "x",
                                    {"x.intent": 1.0}, "dev", {})


def test_verify_arch_distinguishes_no_data_from_failed(tmp_path):
    cfg = _cfg()
    report, dev = verify.verify_arch(tmp_path, "cross", None, None, None, None, cfg)
    assert report["status"] == "no-data" and dev is None
    directory = tmp_path / "dual"
    directory.mkdir()
    (directory / "metadata.json").write_text(json.dumps(
        {"mode": "train", "status": "failed"}), encoding="utf-8")
    report, dev = verify.verify_arch(tmp_path, "dual", None, None, None, None, cfg)
    assert report["status"] == "failed" and dev is None
    (directory / "metadata.json").write_text(json.dumps(
        {"mode": "train", "status": "train_complete"}), encoding="utf-8")
    report, dev = verify.verify_arch(tmp_path, "dual", None, None, None, None, cfg)
    assert report["status"] == "failed" and "missing prediction artifacts" in report["reason"]


def _candidate_rows(native):
    rows = []
    count = len(native["candidate_ids"])
    for k in sorted({size for size in (*verify.NESTED_K, count) if 2 <= size <= count}):
        subset = verify.nested_subset(native["candidate_ids"], native["component_id"], k)
        row = _row({**native, "candidate_ids": subset})
        row.update({"k": k, "subset_ids": subset,
                    "gold_rank": verify.gold_rank(subset, row["probs"], native["gold"]),
                    "gold_in_shortlist": native["gold"] in subset})
        rows.append(row)
    return rows


def test_candidate_count_rejects_misreported_gold_rank(tmp_path):
    native = _native()
    natives = {"x.intent": {native["id"]: native}}
    rows = _candidate_rows(native)
    verify.read_candidate_count(_write(tmp_path, rows), natives, "x", {"x.intent": 1.0})
    rows[0]["gold_rank"] += 1
    with pytest.raises(ValueError, match="gold rank"):
        verify.read_candidate_count(_write(tmp_path, rows), natives, "x", {"x.intent": 1.0})


def test_candidate_count_accepts_absent_gold_and_rejects_target_forcing(tmp_path):
    native = _native(ids=tuple("abcdef"))
    prefix = verify.nested_subset(native["candidate_ids"], native["component_id"], 2)
    native["gold"] = next(cid for cid in native["candidate_ids"] if cid not in prefix)
    natives = {"x.intent": {native["id"]: native}}
    rows = _candidate_rows(native)
    assert rows[0]["gold_rank"] is None and rows[0]["gold_in_shortlist"] is False
    summary = verify.read_candidate_count(_write(tmp_path, rows), natives, "x", {"x.intent": 1.0})
    metrics = verify.summarize_candidate_count(summary, natives)["x.intent"]
    assert metrics["by_k"]["2"]["conditional_accuracy_gold_present"] is None
    assert metrics["by_k"]["2"]["overall_accuracy"] == 0
    assert metrics["by_k"]["2"]["membership_frequency"] == 0
    assert metrics["by_k"]["6"]["membership_frequency"] == 1
    assert metrics["monotone_membership_frequency"] is True
    assert metrics["learned_retrieval_credit"] is False
    assert "shortlist_recall" not in metrics["by_k"]["2"]
    forced = [native["gold"], prefix[0]]
    rows[0].update({"candidate_ids": forced, "subset_ids": forced})
    with pytest.raises(ValueError, match="seeded nested subset"):
        verify.read_candidate_count(_write(tmp_path, rows), natives, "x", {"x.intent": 1.0})


def test_candidate_summary_separates_conditional_and_overall_accuracy():
    natives = {"x.intent": {
        "one": {"gold": "a"}, "two": {"gold": "b"}, "three": {"gold": "c"}}}
    summary = {
        ("x.intent", "one"): [(2, ["a", "b"], "a", True)],
        ("x.intent", "two"): [(2, ["a", "b"], "a", True)],
        ("x.intent", "three"): [(2, ["a", "b"], "a", False)],
    }
    table = verify.summarize_candidate_count(summary, natives)["x.intent"]["by_k"]["2"]
    assert table["gold_present_rows"] == 2 and table["rows"] == 3
    assert table["overall_accuracy"] == 1 / 3
    assert table["conditional_accuracy_gold_present"] == 1 / 2
    assert table["membership_frequency"] == 2 / 3


def test_candidate_count_requires_every_decision_and_complete_nested_grid(tmp_path):
    first = _native(ids=tuple("abcdef"))
    second = {**first, "id": "dec-2", "gold": "f"}
    natives = {"x.intent": {first["id"]: first, second["id"]: second}}
    rows = _candidate_rows(first)
    with pytest.raises(ValueError, match="missing candidate-count decisions"):
        verify.read_candidate_count(_write(tmp_path, rows), natives, "x", {"x.intent": 1.0})
    second_rows = _candidate_rows(second)
    assert [row["subset_ids"] for row in rows] == [row["subset_ids"] for row in second_rows]
    rows.extend(second_rows)
    verify.read_candidate_count(_write(tmp_path, rows), natives, "x", {"x.intent": 1.0})
    rows.pop()
    with pytest.raises(ValueError, match="registered K grid"):
        verify.read_candidate_count(_write(tmp_path, rows), natives, "x", {"x.intent": 1.0})


def _probe_fixture(arch):
    native = {**_native(ids=tuple("x-%02d" % i for i in range(77)), gold="x-00"),
              "id": "dec-z", "state_id": "S1"}
    other_dev = {**_native("y.intent", ("y-0", "y-1"), "y-0"),
                 "id": "dec-y", "state_id": "S3"}
    later_dev = {**native, "id": "dec-a", "state_id": "S2"}
    foreign = {**_native("y.intent", tuple("y-%02d" % i for i in range(60)), "y-59"),
               "id": "fit-a", "state_id": "FIT1", "component_id": "foreign-comp",
               "group_id": "foreign-group"}
    dev = {"x.intent": {later_dev["id"]: later_dev, native["id"]: native},
           "y.intent": {other_dev["id"]: other_dev}}
    fit = {"x.intent": {"fit-000": {**native, "id": "fit-000", "state_id": "FIT0"}},
           "y.intent": {"fit-z": {**foreign, "id": "fit-z"}, foreign["id"]: foreign}}
    values = {"state_encode_calls": 0, "encoder_forward_calls": 0, "encoded_states": 0}
    rows = []
    for repeat in range(3):
        for slot, source in enumerate((native, foreign)):
            row = _row(source)
            row.update({
                "state_id": native["state_id"], "native_state_decision_id": native["id"],
                "catalogue_source_decision_id": source["id"], "repeat_index": repeat,
                "probe_kind": "native" if slot == 0 else "unlabelled_foreign_catalogue",
                "semantic_metric_eligible": slot == 0,
            })
            deltas = {
                "state_encode_calls": int(repeat == 0 and (arch == "cross" or slot == 0)),
                "encoder_forward_calls": (1 if arch == "cross" or slot == 1 else 2)
                if repeat == 0 else 0,
                "encoded_states": (len(source["candidate_ids"]) if arch == "cross" else int(slot == 0))
                if repeat == 0 else 0,
            }
            for name, delta in deltas.items():
                row[name + "_before"] = values[name]
                values[name] += delta
                row[name + "_after"] = values[name]
            rows.append(row)
    return dev, fit, rows


@pytest.mark.parametrize("arch", verify.ARCHS)
def test_repeated_probe_counts_first_round_exposure_and_generic_memoization(tmp_path, arch):
    dev, fit, rows = _probe_fixture(arch)
    result = verify.read_repeated_state(
        _write(tmp_path, rows), dev, fit, {"x.intent": 1.0, "y.intent": 1.0}, arch)
    assert result["state_id"] == "S1" and result["native_state_decision_id"] == "dec-z"
    assert result["catalogue_source_decision_ids"] == ["dec-z", "fit-a"]
    assert result["catalogue_endpoints"] == ["x.intent", "y.intent"]
    assert result["first_round_encoded_states"] == (137 if arch == "cross" else 1)
    assert result["first_round_state_encode_calls"] == (2 if arch == "cross" else 1)
    assert result["first_round_encoder_forward_calls"] == (2 if arch == "cross" else 3)
    assert result["extra_state_encodes"] == result["extra_encoder_forwards"] == result["extra_encoded_states"] == 0
    assert result["foreign_catalogue_semantic_metrics"] is False
    assert result["learned_retrieval_credit"] == 0
    assert "accuracy" not in result


def test_repeated_probe_never_reads_foreign_gold_or_targets(tmp_path):
    dev, fit, rows = _probe_fixture("pages")
    first = verify.read_repeated_state(
        _write(tmp_path, rows), dev, fit, {"x.intent": 1.0, "y.intent": 1.0}, "pages")
    for source in fit["y.intent"].values():
        source.pop("gold")
        source["target_distribution"] = "not a labelled decision on the DEV state"
    assert verify.read_repeated_state(
        _write(tmp_path, rows), dev, fit, {"x.intent": 1.0, "y.intent": 1.0}, "pages") == first


@pytest.mark.parametrize("key,value", [
    ("state_id", "FIT1"),
    ("native_state_decision_id", "dec-a"),
    ("catalogue_source_decision_id", "fit-z"),
    ("endpoint", "x.intent"),
    ("semantic_metric_eligible", True),
    ("probe_kind", "native"),
    ("gold", "y-59"),
    ("target_distribution", [1.0]),
])
def test_repeated_probe_rejects_wrong_source_or_foreign_semantic_credit(tmp_path, key, value):
    dev, fit, rows = _probe_fixture("dual")
    rows[1][key] = value
    with pytest.raises(ValueError):
        verify.read_repeated_state(
            _write(tmp_path, rows), dev, fit, {"x.intent": 1.0, "y.intent": 1.0}, "dual")


@pytest.mark.parametrize("counter", [
    "state_encode_calls", "encoder_forward_calls", "encoded_states",
])
@pytest.mark.parametrize("arch", verify.ARCHS)
def test_repeated_probe_rejects_undercounted_initial_or_extra_repeat_work(tmp_path, counter, arch):
    dev, fit, rows = _probe_fixture(arch)
    initial = copy.deepcopy(rows)
    initial[0][counter + "_after"] -= 1
    with pytest.raises(ValueError):
        verify.read_repeated_state(
            _write(tmp_path, initial), dev, fit, {"x.intent": 1.0, "y.intent": 1.0}, arch)
    rows[2][counter + "_after"] += 1
    with pytest.raises(ValueError):
        verify.read_repeated_state(
            _write(tmp_path, rows), dev, fit, {"x.intent": 1.0, "y.intent": 1.0}, arch)


def test_repeated_probe_rejects_nonfresh_incomplete_or_changed_repeat(tmp_path):
    dev, fit, rows = _probe_fixture("cross")
    temperatures = {"x.intent": 1.0, "y.intent": 1.0}
    with pytest.raises(ValueError, match="complete two-catalogue"):
        verify.read_repeated_state(_write(tmp_path, rows[:-1]), dev, fit, temperatures, "cross")
    stale = copy.deepcopy(rows)
    for row in stale:
        for suffix in ("_before", "_after"):
            row["encoded_states" + suffix] += 100
    with pytest.raises(ValueError, match="fresh chained"):
        verify.read_repeated_state(_write(tmp_path, stale), dev, fit, temperatures, "cross")
    changed = copy.deepcopy(rows)
    logits = [3.0] + [0.0] * (len(changed[2]["candidate_ids"]) - 1)
    masses = verify.softmax(logits)
    answer, ties = verify.choice_winner(changed[2]["candidate_ids"], masses)
    changed[2].update({"logits": logits, "raw_probs": masses, "probs": masses,
                       "answer": answer, "ties": ties, "raw_answer": answer, "raw_ties": ties})
    with pytest.raises(ValueError, match="identical repeat output"):
        verify.read_repeated_state(_write(tmp_path, changed), dev, fit, temperatures, "cross")


def test_architecture_selection_uses_exposures_and_declared_tie_order():
    architectures = {
        "cross": {"repeated_state": {"first_round_encoded_states": 137,
                                    "first_round_state_encode_calls": 0}},
        "dual": {"repeated_state": {"first_round_encoded_states": 1,
                                   "first_round_state_encode_calls": 1}},
        "pages": {"repeated_state": {"first_round_encoded_states": 1,
                                    "first_round_state_encode_calls": 1}},
    }
    choice = verify.select_architecture(["pages", "dual", "cross"], architectures)
    assert choice["architectures"] == ["dual"]
    assert choice["first_round_encoded_states"] == 1
    assert choice["learned_retrieval_credit"] == 0
    assert verify.select_architecture(["pages", "cross"], architectures)["architectures"] == ["pages"]
    assert verify.select_architecture(["cross"], architectures)["architectures"] == ["cross"]
    assert verify.select_architecture([], architectures)["earned"] is False


def _fixture_digest(tensors):
    rendered = []
    names = {"F32": "torch.float32", "I64": "torch.int64"}
    for name in sorted(tensors):
        dtype, shape, payload = tensors[name]
        description = json.dumps([name, names[dtype], shape], separators=(",", ":")).encode()
        rendered.extend((len(description).to_bytes(8, "little"), description, payload))
    return hashlib.sha256(b"".join(rendered)).hexdigest()


def _checkpoint_fixture(tmp_path, arch, schema="vey.native-field.checkpoint.v1"):
    prefix = {"cross": "scorer.pool.encoder.", "dual": "scorer.pool.encoder.",
              "pages": "encoder."}[arch]
    tensors = {
        prefix + "weight": ("F32", [2], struct.pack("<ff", 0.25, -0.5)),
        prefix + "position_ids": ("I64", [1], struct.pack("<q", 2)),
        "head.weight": ("F32", [1], struct.pack("<f", 0.75)),
    }
    model_hash = _fixture_digest(tensors)
    encoder_hash = _fixture_digest({
        name[len(prefix):]: tensor for name, tensor in tensors.items() if name.startswith(prefix)})
    header = {"__metadata__": {"schema": schema, "state_dict_sha256": model_hash}}
    payload = b""
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [len(payload), len(payload) + len(data)]}
        payload += data
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path = tmp_path / "selected.safetensors"
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + payload)
    return path, model_hash, encoder_hash, prefix


@pytest.mark.parametrize("arch", verify.ARCHS)
def test_selected_checkpoint_independently_hashes_model_and_encoder_tensor_bytes(tmp_path, arch):
    path, model_hash, encoder_hash, prefix = _checkpoint_fixture(tmp_path, arch)
    assert verify.checkpoint_fingerprint(path, arch) == (model_hash, encoder_hash, prefix)
    assert model_hash != encoder_hash


@pytest.mark.parametrize("damage", ["schema", "bytes", "truncated", "encoder_prefix", "offsets"])
def test_selected_checkpoint_fails_closed_on_unrecognized_or_corrupt_actual_bytes(tmp_path, damage):
    path, _, _, _ = _checkpoint_fixture(
        tmp_path, "cross", schema="unknown" if damage == "schema" else "vey.native-field.checkpoint.v1")
    data = path.read_bytes()
    arch = "cross"
    if damage == "bytes":
        path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    elif damage == "truncated":
        path.write_bytes(b"\x01\x00")
    elif damage == "encoder_prefix":
        arch = "pages"
    elif damage == "offsets":
        length = int.from_bytes(data[:8], "little")
        header = json.loads(data[8:8 + length])
        header["head.weight"]["data_offsets"] = [0, 4]
        encoded = json.dumps(header, separators=(",", ":")).encode()
        path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + data[8 + length:])
    with pytest.raises(ValueError):
        verify.checkpoint_fingerprint(path, arch)


def _restore_fixture(tmp_path, arch, schema="vey.native-field.checkpoint.v1"):
    path, model_hash, encoder_hash, prefix = _checkpoint_fixture(tmp_path, arch, schema)
    native = {**_native(), "state_id": "SEL1"}
    probe = {native["id"]: native}
    row = _row(native)
    restore = {
        "expected_fingerprint": model_hash, "restored_fingerprint": model_hash,
        "missing_keys": [], "unexpected_keys": [], "probe_state_ids": ["SEL1"],
        "selected_rows": [row], "restored_rows": [copy.deepcopy(row)],
        "max_abs_difference": 0.0, "checkpoint_sha256": verify.digest(path),
        "selected_epoch": 0, "before_restore_fingerprint": "f" * 64,
    }
    return path, model_hash, encoder_hash, prefix, probe, restore


def test_restore_reconstructs_plural_native_probe_state_schedule(tmp_path):
    path, _, _, _, probe, restore = _restore_fixture(tmp_path, "pages")
    native = {**_native("y.intent", ("y-a", "y-b"), "y-a"),
              "id": "dec-2", "state_id": "SEL2"}
    probe[native["id"]] = native
    row = _row(native)
    restore["probe_state_ids"].append("SEL2")
    restore["selected_rows"].append(row)
    restore["restored_rows"].append(copy.deepcopy(row))
    rebuilt = verify.check_restore(
        restore, probe, {"x.intent", "y.intent"}, path, verify.digest(path), "pages")
    assert rebuilt["numerical_max_abs_difference"] == 0


@pytest.mark.parametrize("arch", verify.ARCHS)
def test_restore_requires_real_model_and_encoder_bytes_not_matching_assertions(tmp_path, arch):
    path, model_hash, encoder_hash, prefix, probe, restore = _restore_fixture(tmp_path, arch)
    result = verify.check_restore(restore, probe, {"x.intent"}, path, verify.digest(path), arch)
    assert result["tensor_fingerprint_independently_reconstructed"] is True
    assert result["state_dict_fingerprint"] == model_hash
    assert result["encoder_state_dict_fingerprint"] == encoder_hash
    assert result["encoder_prefix"] == prefix
    restore["expected_fingerprint"] = restore["restored_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="independent selected checkpoint"):
        verify.check_restore(restore, probe, {"x.intent"}, path, verify.digest(path), arch)


def test_restore_has_no_schema_or_missing_checkpoint_assertion_fallback(tmp_path):
    path, _, _, _, probe, restore = _restore_fixture(tmp_path, "pages", schema="unknown")
    with pytest.raises(ValueError, match="safetensors schema"):
        verify.check_restore(restore, probe, {"x.intent"}, path, verify.digest(path), "pages")
    with pytest.raises(ValueError, match="checkpoint artifact identity"):
        verify.check_restore(restore, probe, {"x.intent"}, None, None, "pages")


@pytest.mark.parametrize("damage", ["restored_id", "restored_drift", "probe_state_ids", "missing_keys"])
def test_restore_rejects_nonidentical_numerical_or_identity_receipts(tmp_path, damage):
    path, _, _, _, probe, restore = _restore_fixture(tmp_path, "dual")
    if damage == "restored_id":
        restore["restored_rows"][0]["id"] = "other"
    elif damage == "restored_drift":
        logits = [3.0, 1.0]
        restore["restored_rows"][0].update({"logits": logits, "raw_probs": verify.softmax(logits)})
        restore["max_abs_difference"] = 1.0
    elif damage == "probe_state_ids":
        restore["probe_state_ids"] = ["other"]
    else:
        restore["missing_keys"] = ["encoder.weight"]
    with pytest.raises(ValueError):
        verify.check_restore(restore, probe, {"x.intent"}, path, verify.digest(path), "dual")


@pytest.mark.parametrize("damage", [
    "metadata_model", "history_model", "selected_epoch", "encoder_hash", "encoder_prefix",
    "liveness_encoder", "before_restore",
])
def test_actual_selected_checkpoint_bytes_bind_model_history_and_encoder_receipts(tmp_path, damage):
    path, model_hash, encoder_hash, prefix, probe, restore = _restore_fixture(tmp_path, "pages")
    checkpoint_sha = verify.digest(path)
    rebuilt = verify.check_restore(restore, probe, {"x.intent"}, path, checkpoint_sha, "pages")
    metadata = {"checkpoint": {
        "path": str(path), "sha256": checkpoint_sha, "schema": "vey.native-field.checkpoint.v1",
        "selected_epoch": 0, "state_dict_fingerprint": model_hash,
        "encoder_prefix": prefix, "encoder_state_dict_fingerprint": encoder_hash,
    }}
    history = {"selected_epoch": 0, "selected_fingerprint": model_hash,
               "epochs": [{"state_dict_fingerprint": model_hash},
                          {"state_dict_fingerprint": restore["before_restore_fingerprint"]}]}
    liveness = {"selected_restore": restore, "encoder_selected_sha256": encoder_hash}
    verify.check_selected_checkpoint(metadata, history, liveness, rebuilt, path, checkpoint_sha)
    if damage == "metadata_model":
        metadata["checkpoint"]["state_dict_fingerprint"] = "0" * 64
    elif damage == "history_model":
        history["epochs"][0]["state_dict_fingerprint"] = "0" * 64
    elif damage == "selected_epoch":
        metadata["checkpoint"]["selected_epoch"] = 1
    elif damage == "encoder_hash":
        metadata["checkpoint"]["encoder_state_dict_fingerprint"] = "0" * 64
    elif damage == "encoder_prefix":
        metadata["checkpoint"]["encoder_prefix"] = "scorer.pool.encoder."
    elif damage == "liveness_encoder":
        liveness["encoder_selected_sha256"] = "0" * 64
    else:
        restore["before_restore_fingerprint"] = "0" * 64
    with pytest.raises(ValueError):
        verify.check_selected_checkpoint(metadata, history, liveness, rebuilt, path, checkpoint_sha)
