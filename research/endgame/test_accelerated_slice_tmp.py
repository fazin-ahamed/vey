import json, hashlib
from pathlib import Path
def test_screen_slice_deterministic(tmp_path):
    import subprocess
    rows = [{"id": str(i), "component_id": f"c{i%30}", "endpoint": "e", "phase": "fit",
             "paper_id": None, "group_id": "g", "question_ordinal": None,
             "input_sha256": "x", "question": "q", "windows": [], "candidates": [],
             "target_available": True} for i in range(100)]
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    with (ledger / "ledger.jsonl").open("w") as s:
        for r in rows:
            s.write(json.dumps(r) + "\n")
    (ledger / "ledger_manifest.json").write_text(
        json.dumps({"schema": "vey.accel.ledger.v1", "status": "MATERIALIZED"}))
    script = "/home/fazinahamed/Documents/vey-public/research/endgame/accelerated_screen_slice.py"
    out1, out2 = tmp_path / "a", tmp_path / "b"
    out1.mkdir(); out2.mkdir()
    for out in (out1, out2):
        subprocess.run(["/home/fazinahamed/Documents/vey/venvs/vey/bin/python", script,
                        "--ledger-root", str(ledger), "--out-root", str(out)], check=True)
    a = (out1 / "screen_slice.jsonl").read_bytes(); b = (out2 / "screen_slice.jsonl").read_bytes()
    assert a == b
    m = json.loads((out1 / "screen_slice_manifest.json").read_text())
    assert m["rows"] == len([r for r in rows if int(hashlib.sha256(f"accel-screen|7|{r['component_id']}".encode()).hexdigest(), 16) % 20 == 0])
