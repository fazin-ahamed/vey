"""Resource transitions that previously stalled cross-encoder smoke indefinitely."""
import ast
import hashlib
import json
from pathlib import Path

import pytest

import native_arch_train as train

GIB = 1024 ** 3


def resource_scene(monkeypatch, free, reserved, allocated):
    state = {"free": int(free * GIB), "reserved": int(reserved * GIB),
             "allocated": int(allocated * GIB)}
    monkeypatch.setattr(train.native, "meminfo", lambda: (10 * GIB, 0))
    monkeypatch.setattr(train.torch.cuda, "mem_get_info", lambda device: (state["free"], 12 * GIB))
    monkeypatch.setattr(train.torch.cuda, "memory_reserved", lambda device: state["reserved"])
    monkeypatch.setattr(train.torch.cuda, "memory_allocated", lambda device: state["allocated"])

    def release_idle():
        state["free"] += state["reserved"] - state["allocated"]
        state["reserved"] = state["allocated"]

    monkeypatch.setattr(train.torch.cuda, "empty_cache", release_idle)
    cfg = {"resources": {"pause_available_RAM_GiB_below": 4,
                         "pause_swap_growth_MiB_above": 256,
                         "pause_GPU_free_MiB_below": 2048},
           "training": {"batch_states": 4}}
    return train.Resources(cfg, "cuda"), state


def test_idle_owned_cache_does_not_permanently_pause(monkeypatch):
    resources, state = resource_scene(monkeypatch, free=.5, reserved=7, allocated=1)

    def unexpected_pause(seconds):
        raise AssertionError("Idle owned reserve should release enough space to avoid pausing")

    monkeypatch.setattr(train.native.time, "sleep", unexpected_pause)
    snapshot = resources.guard()
    assert snapshot["GPU_free_bytes"] == state["free"] == int(6.5 * GIB)
    assert state["reserved"] == state["allocated"] == GIB
    assert [event["kind"] for event in resources.events] == ["reclaimed_idle_CUDA_cache", "startup"]


def test_live_external_pressure_still_pauses(monkeypatch):
    resources, state = resource_scene(monkeypatch, free=.5, reserved=1, allocated=1)
    pauses = []

    def external_memory_freed(seconds):
        pauses.append(seconds)
        state["free"] = 3 * GIB

    monkeypatch.setattr(train.native.time, "sleep", external_memory_freed)
    snapshot = resources.guard()
    assert pauses == [5]
    assert snapshot["GPU_free_bytes"] == 3 * GIB
    assert resources.events[0]["kind"] == "pause"
    assert resources.events[0]["reasons"] == ["GPU_free"]
    assert resources.events[1]["kind"] == "resumed"
    assert not any(event["kind"] == "reclaimed_idle_CUDA_cache" for event in resources.events)


def test_partial_cache_reclaim_does_not_bypass_threshold(monkeypatch):
    resources, state = resource_scene(monkeypatch, free=.25, reserved=1.25, allocated=1)
    pauses = []

    def external_memory_freed(seconds):
        pauses.append(seconds)
        state["free"] = 3 * GIB

    monkeypatch.setattr(train.native.time, "sleep", external_memory_freed)
    snapshot = resources.guard()
    assert pauses == [5]
    assert snapshot["GPU_free_bytes"] == 3 * GIB
    assert [event["kind"] for event in resources.events] == ["reclaimed_idle_CUDA_cache", "pause", "resumed"]
    assert resources.events[1]["reasons"] == ["GPU_free"]
    assert resources.events[1]["GPU_free_bytes"] == GIB // 2


def modal_guard_namespace(root):
    tree = ast.parse((Path(__file__).parent / "native_arch_modal.py").read_text())
    names = {"require", "digest", "require_preflight", "require_smoke", "check_prerequisites"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    volume = next(ast.literal_eval(node.value) for node in tree.body
                  if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "VOLUME_NAME"
                          for target in node.targets))
    namespace = {"Path": Path, "hashlib": hashlib, "json": json, "ROOT": root,
                 "RECOVERY": root / "recovery.json", "VOLUME_NAME": volume,
                 "ARMS": ("cross", "dual", "pages")}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "modal-receipt-guards", "exec"),
         namespace)
    return namespace


def modal_receipt_scene(root):
    namespace = modal_guard_namespace(root)
    namespace["RECOVERY"].write_text("recovery")
    recovery_hash = hashlib.sha256(b"recovery").hexdigest()
    binding = {"transport_sha256": "current-transport",
               "recovery_protocol_sha256": recovery_hash, "volume": namespace["VOLUME_NAME"]}
    preflight = dict(binding, schema="vey.native-arch.modal-preflight.v1", status="PASS",
                     stock_loading_verified_on_CPU=True, GPU_allocated=False,
                     sealed_phases_accessed=False)
    (root / "modal_preflight.json").write_text(json.dumps(preflight))
    receipts = {}
    for arm in namespace["ARMS"]:
        directory = root / (arm + "-smoke")
        directory.mkdir()
        metadata = directory / "metadata.json"
        metadata.write_text(json.dumps({"status": "smoke_complete", "arm": arm}))
        receipt = dict(binding, schema="vey.native-arch.modal-runtime.v1", status="COMPLETE",
                       arm=arm, mode="smoke", numerical_smoke_verified=True,
                       output=str(directory),
                       metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest())
        (root / ("modal_runtime_" + arm + "_smoke.json")).write_text(json.dumps(receipt))
        receipts[arm] = receipt
    return namespace, preflight, receipts


def test_modal_recovery_requires_all_fresh_volume_smokes(tmp_path):
    namespace, _, _ = modal_receipt_scene(tmp_path)
    namespace["check_prerequisites"]("train", "current-transport")
    (tmp_path / "modal_runtime_pages_smoke.json").unlink()
    with pytest.raises(FileNotFoundError):
        namespace["check_prerequisites"]("train", "current-transport")


@pytest.mark.parametrize("key,value", [
    ("volume", "vey-native-arch-modal-v1"),
    ("transport_sha256", "original-v1-transport"),
    ("recovery_protocol_sha256", "another-recovery"),
    ("status", "STARTED"),
    ("numerical_smoke_verified", False),
    ("arm", "cross"),
    ("mode", "train"),
    ("output", "/original-v1/pages-smoke"),
])
def test_modal_recovery_rejects_stale_or_incomplete_smoke(tmp_path, key, value):
    namespace, _, receipts = modal_receipt_scene(tmp_path)
    receipts["pages"][key] = value
    (tmp_path / "modal_runtime_pages_smoke.json").write_text(json.dumps(receipts["pages"]))
    with pytest.raises(RuntimeError, match="Every actual v2 Modal"):
        namespace["check_prerequisites"]("train", "current-transport")


@pytest.mark.parametrize("key,value", [
    ("volume", "vey-native-arch-modal-v1"),
    ("transport_sha256", "original-v1-transport"),
    ("recovery_protocol_sha256", "another-recovery"),
    ("status", "FAILED"),
    ("stock_loading_verified_on_CPU", False),
    ("GPU_allocated", True),
    ("sealed_phases_accessed", True),
])
def test_modal_recovery_rejects_stale_preflight(tmp_path, key, value):
    namespace, preflight, _ = modal_receipt_scene(tmp_path)
    preflight[key] = value
    (tmp_path / "modal_preflight.json").write_text(json.dumps(preflight))
    with pytest.raises(RuntimeError, match="Fresh v2 preflight"):
        namespace["check_prerequisites"]("smoke", "current-transport")


def test_modal_recovery_rejects_changed_smoke_artifacts(tmp_path):
    namespace, _, _ = modal_receipt_scene(tmp_path)
    (tmp_path / "dual-smoke" / "metadata.json").write_text("{}")
    with pytest.raises(RuntimeError, match="Required v2 smoke artifacts differ"):
        namespace["check_prerequisites"]("train", "current-transport")


def test_modal_recovery_execution_bounds_match_protocol():
    here = Path(__file__).parent
    tree = ast.parse((here / "native_arch_modal.py").read_text())
    constants = {node.targets[0].id: ast.literal_eval(node.value)
                 for node in tree.body if isinstance(node, ast.Assign)
                 and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id in {"TRAIN_TIMEOUT", "SMOKE_TIMEOUT", "VOLUME_NAME", "ARMS"}}
    protocol = json.loads((here / "native_arch_protocol.json").read_text())
    recovery = json.loads((here / "native_arch_modal_timeout_recovery_protocol.json").read_text())
    transport = json.loads((here / "native_arch_modal_transport.json").read_text())
    assert constants["TRAIN_TIMEOUT"] == recovery["train_timeout_seconds_per_arm"] == 43200
    assert protocol["resources"]["modal_train_timeout_seconds_per_arm"] == 43200
    assert constants["SMOKE_TIMEOUT"] == recovery["smoke_timeout_seconds_per_arm"] == 600
    assert protocol["resources"]["modal_smoke_timeout_seconds_per_arm"] == 600
    assert constants["VOLUME_NAME"] == recovery["volume"] == transport["volume"] == "vey-native-arch-modal-v2"
    assert constants["ARMS"] == tuple(recovery["arms"]) == ("cross", "dual", "pages")
    assert protocol["training"]["epochs"] == 10
    assert recovery["resume_allowed"] is recovery["v1_evidence_authorizes_training"] is False
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in {"preflight", "run_arm"}:
            continue
        options = {option.arg: option.value for option in node.decorator_list[0].keywords}
        assert ast.literal_eval(options["max_containers"]) == 1
        assert ast.literal_eval(options["retries"]) == 0
        assert options["timeout"].id == ("TRAIN_TIMEOUT" if node.name == "run_arm" else "SMOKE_TIMEOUT")
        if node.name == "run_arm":
            assert ast.literal_eval(options["gpu"]) == "T4"
