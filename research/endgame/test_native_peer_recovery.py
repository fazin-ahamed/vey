"""Pure regressions for the resource-governed peer recovery helpers."""
import json
from pathlib import Path

import pytest

import native_peer_capture_recovery as R


def _meminfo(available_kib, swap_total_kib, swap_free_kib):
    return ("MemTotal: 32658984 kB\nMemAvailable: %d kB\nSwapTotal: %d kB\nSwapFree: %d kB\n"
            % (available_kib, swap_total_kib, swap_free_kib))


def _rows(ids):
    return [{"id": i, "group_id": "g", "component_id": "c", "locale": "en",
             "endpoint": "banking77.intent", "state": "s " + i, "question": "q " + i,
             "candidate_ids": ["x", "y"], "candidates": {"x": "X", "y": "Y"},
             "gold": "x"} for i in ids]


def _valid(row, answer="x"):
    return {"id": row["id"], "group_id": row["group_id"], "component_id": row["component_id"],
            "locale": row["locale"], "endpoint": row["endpoint"],
            "question": row["question"], "state": row["state"],
            "candidate_ids": row["candidate_ids"], "candidates": row["candidates"],
            "gold": row["gold"], "answer": answer, "correct": answer == row["gold"],
            "probabilities": {"x": 0.6, "y": 0.4}}


def test_meminfo_parse_returns_available_and_used_swap(tmp_path):
    path = tmp_path / "meminfo"
    path.write_text(_meminfo(12000000, 8388604, 8000000))
    available, swap = R.read_meminfo(path)
    assert available == 12000000 * 1024
    assert swap == 388604 * 1024


def test_governor_load_floor_stops_before_model(tmp_path):
    path = tmp_path / "meminfo"
    path.write_text(_meminfo(5 * 1024 * 1024, 0, 0))
    with pytest.raises(RuntimeError, match="load floor"):
        R.Governor(lambda: R.read_meminfo(path))


def test_governor_blocks_call_below_ram_floor_and_records_snapshot(tmp_path):
    path = tmp_path / "meminfo"
    path.write_text(_meminfo(12 * 1024 * 1024, 100, 100))
    governor = R.Governor(lambda: R.read_meminfo(path))
    path.write_text(_meminfo(2 * 1024 * 1024, 100, 100))
    with pytest.raises(R.ResourceBreach, match="per-call floor") as breach:
        governor.check("before call 3")
    assert breach.value.snapshot["MemAvailable_bytes"] == 2 * 1024 ** 3


def test_governor_blocks_call_on_swap_growth_without_retry(tmp_path):
    path = tmp_path / "meminfo"
    path.write_text(_meminfo(12 * 1024 * 1024, 8 * 1024 * 1024, 8 * 1024 * 1024))
    governor = R.Governor(lambda: R.read_meminfo(path))
    path.write_text(_meminfo(12 * 1024 * 1024, 8 * 1024 * 1024, 8 * 1024 * 1024 - 300 * 1024))
    with pytest.raises(R.ResourceBreach, match="swap growth"):
        governor.check("before call 1")


def test_retained_pairing_rejects_id_order_mismatch():
    a, b = _rows(["1", "2"]), _rows(["2", "1"])
    with pytest.raises(RuntimeError, match="id order differs"):
        R.retained_universe([_valid(r) for r in a], [{"id": r["id"]} for r in b])


def test_remaining_rows_preserve_original_order_and_reject_forged_retained():
    dev = _rows(["1", "2", "3"])
    retained = {"2": None}
    forged = dict(_valid(dev[1]))
    forged["state"] = "forged"
    with pytest.raises(RuntimeError, match="inputs differ"):
        R.remaining_rows(dev, {"2": (forged, {"id": "2"})})
    del retained
    clean = R.retained_universe([_valid(dev[1])], [{"id": "2"}])
    assert [r["id"] for r in R.remaining_rows(dev, clean)] == ["1", "3"]


def test_validate_prediction_enforces_serialization_bound():
    row = _rows(["1"])[0]
    prediction = {"answers": {"q": {"probabilities": {"x": 0.6, "y": 0.39},
                                    "choice": "x"}}}
    with pytest.raises(RuntimeError, match="Serialization bound"):
        R.validate_prediction(row, prediction)
    good = {"answers": {"q": {"probabilities": {"x": 0.6, "y": 0.4}, "choice": "y"}}}
    record = R.validate_prediction(row, good)
    assert record["correct"] is False and record["answer"] == "y"


def test_merge_full_rejects_rerun_and_preserves_order():
    dev = _rows(["1", "2", "3"])
    retained = R.retained_universe([_valid(dev[0])], [{"id": "1"}])
    cont_valid = [_valid(dev[1]), _valid(dev[2])]
    cont_raw = [{"id": "2"}, {"id": "3"}]
    ordered_validated, ordered_raw = R.merge_full({}, dev, retained, cont_valid, cont_raw)
    assert [r["id"] for r in ordered_validated] == ["1", "2", "3"]
    assert [r["id"] for r in ordered_raw] == ["1", "2", "3"]
    with pytest.raises(RuntimeError, match="re-ran a retained id"):
        R.merge_full({}, dev, retained, [_valid(dev[0])], [{"id": "1"}])


def test_merge_full_requires_complete_population():
    dev = _rows(["1", "2", "3"])
    retained = R.retained_universe([_valid(dev[0])], [{"id": "1"}])
    with pytest.raises(RuntimeError, match="complete DEV population"):
        R.merge_full({}, dev, retained, [_valid(dev[1])], [{"id": "2"}])


def test_load_jsonl_rejects_retained_failure_row(tmp_path):
    path = tmp_path / "capture.jsonl"
    path.write_text(json.dumps({"schema": R.CAPTURE_SCHEMA}) + "\n"
                    + json.dumps({"schema": "vey.native-peer.laya-capture-failure.v1"}) + "\n")
    with pytest.raises(RuntimeError, match="failure row"):
        R.load_jsonl(path)


def test_merge_mode_refuses_existing_outputs(tmp_path):
    existing = tmp_path / "merged.jsonl"
    existing.write_text("{}")
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        R.main(["--attempt-out", str(tmp_path / "cont.jsonl"),
                "--merge-out", str(existing), "--raw-merge-out", str(tmp_path / "r.jsonl")])
