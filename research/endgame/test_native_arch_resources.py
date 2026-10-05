"""Resource transitions that previously stalled cross-encoder smoke indefinitely."""
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
