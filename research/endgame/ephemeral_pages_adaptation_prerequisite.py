#!/usr/bin/env python3
"""Train-only scalar-grade prerequisite for the separately preregistered C1 ceiling.

Only train is loaded. The frozen prefix is evaluated once, and its detached
final-layer call inputs are replayed through the actual transformer block.
Smoke artifacts are permanently ineligible for C1; full runs retain epoch400.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

# The screens module sets offline/thread policy before importing Torch or NumPy.
import ephemeral_pages_adaptation_screens as screens

np = screens.np
torch = screens.torch
features = screens.features
components = screens.components
trainer = screens.trainer
evaluator = screens.evaluator
capture = screens.capture
require = screens._require

GRADIENT_PREREGISTRATION_PATH = screens.C1_GRADIENT_PROTOCOL_PATH
GRADIENT_PREREGISTRATION_SHA256 = screens.C1_GRADIENT_PROTOCOL_SHA256
DATA_ROOT = Path("/home/fazinahamed/Documents/vey-data").resolve()
FULL_TEXTS = 180
FULL_EPOCHS = 400
SMOKE_TEXTS = 8
SMOKE_EPOCHS = 2
STAGE_DIRECTORY = "c1-gradient-v1"
RECEIPT_NAME = "prerequisite_receipt.json"
CHECKPOINT_NAME = "prerequisite_checkpoint.pt"


def _json(path: Path, payload: dict) -> dict:
    return screens.write_receipt(path, evaluator.plain(payload))


def _sources() -> dict:
    pinned = screens._check_pinned_sources()
    return {**pinned, Path(screens.__file__).name: features.sha256_file(screens.__file__),
            Path(__file__).name: features.sha256_file(__file__)}


def _controls(root: Path, experiment) -> dict:
    evidence = {}
    for name in ("smoke", "immutability", "c0_liveness"):
        path = root / screens.RECEIPT_NAMES[name]
        receipt = json.loads(path.read_text(encoding="utf-8"))
        require(receipt.get("verdict") == "pass", f"prerequisite requires passing {name}")
        require(receipt.get("experiment_context") == experiment.context(),
                f"{name} receipt belongs to another experiment")
        require(receipt.get("preregistration", {}).get("sha256")
                == screens.PREREGISTRATION_SHA256, f"{name} parent protocol mismatch")
        if name == "c0_liveness":
            require(receipt.get("c0_amendment", {}).get("sha256")
                    == screens.C0_AMENDMENT_SHA256, "C0 amendment mismatch")
        evidence[name] = evaluator.artifact(path)
    return evidence


def _train_material(experiment) -> dict:
    # Do not use _screen_c_material or _experiment: they inspect other phases.
    require(features.sha256_file(experiment.corpus_root / "train.jsonl")
            == screens.ATOMIC_CORPUS_PINS["train"], "pinned train corpus changed")
    require(features.sha256_file(experiment.cache_root / "train_manifest.json")
            == screens.ATOMIC_FEATURE_MANIFEST_PINS["train"], "pinned train features changed")
    train, evidence = screens._load_phase("train", experiment)
    selected = trainer.optimizer_indices(train, "train", experiment)
    normalizer = trainer.fit_normalizer(train, selected, screens.CHUNK_SIZE)
    require(screens.PAGES_CHECKPOINT.exists(), "pinned pages checkpoint is required; no fresh reader")
    require(features.sha256_file(screens.PAGES_CHECKPOINT) == screens.PAGES_CHECKPOINT_SHA256,
            "pinned pages checkpoint changed")
    checkpoint = torch.load(screens.PAGES_CHECKPOINT, map_location="cpu", weights_only=False)
    require(checkpoint["protocol_sha256"] == experiment.protocol()[1],
            "pages checkpoint belongs to another corpus protocol")
    require(components.fingerprint(checkpoint["normalizer"]) == components.fingerprint(normalizer),
            "common train q/page normalizer identity failed")
    stats = normalizer["pages"]
    require(np.asarray(stats["mean"]).shape == (screens.WIDTH,)
            and np.asarray(stats["std"]).shape == (screens.WIDTH,), "normalizer width changed")
    require(np.isfinite(stats["mean"]).all() and np.isfinite(stats["std"]).all()
            and np.all(np.asarray(stats["std"]) >= np.float32(.01)),
            "common fixed normalizer is nonfinite or lacks the existing .01 floor")
    catalogue = components.source_catalogue(experiment)
    rows, occurrences = components.supervised_unique(
        train, selected, catalogue[2], features._parse_meminfo()[1])
    require(len(rows) == FULL_TEXTS, "prerequisite requires exactly180 unique supervised train texts")
    require(len({row["text_sha256"] for row in rows}) == FULL_TEXTS,
            "supervised population contains duplicate text hashes")
    texts, slots = screens._rows_with_texts(rows, train)
    require(slots == list(range(FULL_TEXTS)), "unique train-text order/remapping changed")
    target = np.asarray([row["target"] for row in rows], dtype=np.float64)
    require(np.isfinite(target).all() and np.all((target >= 0) & (target <= 1)),
            "raw grade targets are not finite scalar extents")
    return {"train": train, "evidence": evidence, "selected": selected,
            "normalizer": normalizer, "rows": rows, "texts": texts, "target": target,
            "occurrences": occurrences, "source_provenance": catalogue[3]}


def _snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True).contiguous()
    if isinstance(value, tuple):
        return tuple(_snapshot(item) for item in value)
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    require(value is None or isinstance(value, (str, bool, int, float)),
            f"unsupported final-layer call input type: {type(value).__name__}")
    return value


def _on_device(value, device: str):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_on_device(item, device) for item in value)
    if isinstance(value, list):
        return [_on_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _on_device(item, device) for key, item in value.items()}
    return value


def _input_description(value):
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "requires_grad": False,
                "sha256": features.sha256_bytes(screens._tensor_bytes(value))}
    if isinstance(value, (tuple, list)):
        return [_input_description(item) for item in value]
    if isinstance(value, dict):
        return {key: _input_description(item) for key, item in value.items()}
    return value


def _layer_hidden(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        hidden = output
    else:
        require(isinstance(output, (tuple, list)) and len(output) >= 1
                and isinstance(output[0], torch.Tensor), "unsupported actual final-layer output contract")
        hidden = output[0]
    require(hidden.ndim == 3 and hidden.shape[-1] == screens.WIDTH
            and hidden.dtype == torch.float32 and bool(torch.isfinite(hidden).all()),
            "actual final-layer hidden output is nonfinite or has a wrong shape/dtype")
    return hidden


def _cache_prefix(adapted, encoder, ids: np.ndarray, masks: np.ndarray,
                  device: str, counters: dict) -> tuple[list[dict], dict]:
    base = adapted.deberta
    require(len(base.encoder.layer) == 12 and base.encoder.conv is None
            and int(base.z_steps) <= 1, "pinned encoder no longer ends at exactly layer11")
    layer = base.encoder.layer[11]
    batches, reports = [], []
    for start in range(0, len(ids), screens.CHUNK_SIZE):
        stop = min(start + screens.CHUNK_SIZE, len(ids))
        features._resource_guard(torch, encoder.initial_swap_mib)
        calls, outputs = [], []

        def capture_inputs(_module, args, kwargs):
            calls.append((_snapshot(args), _snapshot(kwargs)))

        def capture_output(_module, _args, output):
            outputs.append(_snapshot(_layer_hidden(output)))

        pre = layer.register_forward_pre_hook(capture_inputs, with_kwargs=True)
        post = layer.register_forward_hook(capture_output)
        batch_ids = torch.as_tensor(ids[start:stop], dtype=torch.long, device=device)
        batch_mask = torch.as_tensor(masks[start:stop], dtype=torch.long, device=device)
        try:
            # no_grad, not inference_mode: cached tensors must remain usable by autograd.
            with torch.no_grad():
                full = base(input_ids=batch_ids, attention_mask=batch_mask).last_hidden_state
                reference = full.detach().to("cpu", copy=True)
                reference_pool = screens.masked_mean(full, batch_mask).detach().to("cpu", copy=True)
        finally:
            pre.remove()
            post.remove()
        counters["full_encoder_forward_batches"] += 1
        counters["frozen_prefix_texts_encoded"] += stop - start
        require(len(calls) == 1 and len(outputs) == 1, "full encoder did not call final layer exactly once")
        args, kwargs = calls[0]
        require(outputs[0].shape == reference.shape, "final layer differs from full encoder output shape")
        actual_difference = float((outputs[0] - reference).abs().max())
        with torch.no_grad():
            replay_output = layer(*_on_device(args, device), **_on_device(kwargs, device))
            replay_hidden = _layer_hidden(replay_output)
            require(replay_hidden.shape == reference.shape, "cached replay output shape changed")
            replay_pool = screens.masked_mean(replay_hidden, batch_mask)
            token_difference = float((replay_hidden.detach().cpu() - reference).abs().max())
            pool_difference = float((replay_pool.detach().cpu() - reference_pool).abs().max())
        counters["final_layer_replay_batches"] += 1
        require(max(actual_difference, token_difference, pool_difference) <= screens.IDENTITY_TOLERANCE,
                f"cached replay identity failed at rows{start}:{stop}: "
                f"layer/full={actual_difference}, tokens={token_difference}, pooled={pool_difference}")
        batches.append({"start": start, "stop": stop, "args": args, "kwargs": kwargs,
                        "mask": _snapshot(batch_mask)})
        reports.append({"start": start, "stop": stop,
                        "layer_vs_full_max_abs": actual_difference,
                        "replay_tokens_max_abs": token_difference,
                        "replay_pooled_max_abs": pool_difference,
                        "output_contract": type(replay_output).__name__,
                        "args": _input_description(args), "kwargs": _input_description(kwargs)})
    return batches, {"pass": True, "tolerance": screens.IDENTITY_TOLERANCE,
                     "identical_batch_layout": True, "chunk_size": screens.CHUNK_SIZE,
                     "cache_storage": "process-local detached CPU tensors; actual final-layer inputs",
                     "batches": reports}


def _replay(layer, batch: dict, device: str, counters: dict) -> torch.Tensor:
    output = layer(*_on_device(batch["args"], device), **_on_device(batch["kwargs"], device))
    hidden = _layer_hidden(output)
    mask = batch["mask"].to(device)
    require(hidden.shape[:2] == mask.shape, "cached mask and final-layer output shape differ")
    counters["final_layer_replay_batches"] += 1
    return screens.masked_mean(hidden, mask)


def _frozen_inventory(adapted, reader) -> list[tuple[str, torch.Tensor]]:
    return sorted([(f"encoder.{name}", parameter) for name, parameter in adapted.named_parameters()
                   if not parameter.requires_grad]
                  + [(f"reader.{name}", parameter) for name, parameter in reader.named_parameters()
                     if not parameter.requires_grad])


def _frozen_hash(inventory: list[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, parameter in inventory:
        require(not parameter.requires_grad, f"frozen parameter was made trainable: {name}")
        digest.update(screens._entry_bytes(name, parameter))
    return digest.hexdigest()


def _gradient_evidence(adapted, reader) -> dict:
    report = screens.gradient_report(adapted, reader)
    require(report["all_trainable_gradients_finite"] and report["gradient_norm_finite"],
            "missing/nonfinite trainable gradient")
    groups = {"layer11": 0.0, "wv": 0.0}
    nonzero = {}
    for name, parameter in adapted.deberta.named_parameters():
        if parameter.requires_grad:
            require(name.startswith(screens.LAYER11_LIVE_PREFIX), "trainable encoder escaped layer11")
            groups["layer11"] += float(parameter.grad.detach().double().square().sum())
    for name, adapter in adapted.adapters().items():
        groups[name] = float(adapter.grad.detach().double().square().sum())
    for name, parameter in reader.wv.named_parameters():
        squared = float(parameter.grad.detach().double().square().sum())
        groups["wv"] += squared
        nonzero[f"wv.{name}"] = squared > 0
    norms = {name: math.sqrt(squared) for name, squared in groups.items()}
    return {**report, "component_norms": norms, "wv_nonzero_by_parameter": nonzero,
            "all_components_nonzero": all(math.isfinite(value) and value > 0 for value in norms.values())}


def _surface_report(adapted, reader) -> dict:
    state = screens.parameter_hashes(adapted)
    require(state["layer11_parameter_count"] == screens.LAYER11_TENSOR_COUNT
            and state["layer11_scalars"] == screens.LAYER11_SCALARS
            and state["adapter_count"] == 6 and state["adapter_scalars"] == screens.ADAPTER_SCALARS,
            "preregistered trainable encoder surface changed")
    encoder_trainable = sum(parameter.numel() for parameter in adapted.parameters() if parameter.requires_grad)
    require(encoder_trainable == screens.ENCODER_SIDE_TRAINABLE_SCALARS, "encoder trainable fraction changed")
    reader_names = sorted(name for name, parameter in reader.named_parameters() if parameter.requires_grad)
    require(reader_names == ["wv.bias", "wv.weight"], "prerequisite reader must train only existing wv")
    reader_trainable = sum(parameter.numel() for parameter in reader.parameters() if parameter.requires_grad)
    require(reader_trainable == screens.WIDTH + 1, "wv architecture changed")
    require(encoder_trainable / screens.DENOMINATOR_ENCODER <= screens.BOUND_ENCODER_SIDE_FRACTION
            and (encoder_trainable + reader_trainable) / screens.DENOMINATOR_COMBINED
            <= screens.BOUND_COMBINED_FRACTION, "adaptation fraction bound exceeded")
    return {"encoder_trainable_scalars": encoder_trainable, "reader_trainable_scalars": reader_trainable,
            "reader_trainable_names": reader_names, "initial_encoder_hashes": state}


def train_prerequisite(run_root, features_root, device: str = "cuda", smoke: bool = False) -> dict:
    require(features_root is not None, "gradient prerequisite requires --features-root")
    require(isinstance(smoke, bool), "smoke must be a boolean")
    experiment = capture.resolve_experiment(atomic=True, features_root=features_root)
    requested_root = Path(run_root).resolve()
    require(DATA_ROOT in requested_root.parents, "prerequisite artifacts must remain under vey-data")
    root = screens._safe_root(requested_root, experiment)
    directory = root / STAGE_DIRECTORY
    directory.mkdir(exist_ok=False)
    receipt_path = directory / RECEIPT_NAME
    epochs = SMOKE_EPOCHS if smoke else FULL_EPOCHS
    payload = {"schema": "vey.eca2.c1-gradient-prerequisite-receipt.v1", "verdict": "fail",
               "protocol_sha256": GRADIENT_PREREGISTRATION_SHA256,
               "experiment_context": experiment.context(), "device": str(device),
               "seed": screens.SEED, "smoke": smoke, "epochs": epochs,
               "eligible_for_C1": False, "promotion": False, "B_STEF_allowed": False,
               "endgame_complete": False, "final_pool_touched": False,
               "phases_opened": [], "runtime_policy": screens._runtime_policy(), "history": [],
               "counters": {"full_encoder_forward_batches": 0, "frozen_prefix_texts_encoded": 0,
                            "final_layer_replay_batches": 0, "reader_value_forward_chunks": 0,
                            "loss_backward_calls": 0, "optimizer_steps": 0}}
    try:
        protocol = screens.verify_gradient_protocol()
        require(protocol["parent_preregistration_sha256"] == screens.PREREGISTRATION_SHA256
                and protocol["C0_amendment_sha256"] == screens.C0_AMENDMENT_SHA256,
                "gradient parent/amendment lineage differs")
        adaptation = protocol["adaptation"]
        require((adaptation["seed"], adaptation["epochs"], adaptation["updates_per_epoch"],
                 adaptation["chunk_size"], adaptation["optimizer"], adaptation["encoder_lr"],
                 adaptation["adapter_lr"], adaptation["reader_lr"], adaptation["weight_decay"])
                == (7, 400, 1, 32, "AdamW", .0001, .0001, .01, .0001),
                "gradient fitting constants differ from prospective protocol")
        payload["protocol"] = evaluator.artifact(GRADIENT_PREREGISTRATION_PATH)
        payload["source_hashes"] = _sources()
        payload["predecessor_receipts"] = _controls(root, experiment)
        payload["stage"] = "train_material"
        payload["phases_opened"] = ["train"]
        material = _train_material(experiment)
        full_rows = material["rows"]
        rows = full_rows[:SMOKE_TEXTS] if smoke else full_rows
        texts = material["texts"][:len(rows)]
        targets = material["target"][:len(rows)]
        population_hash = screens.gradient_population_sha256(rows)
        payload.update({"population_sha256": population_hash,
                        "full_population_sha256": screens.gradient_population_sha256(full_rows),
                        "unique_train_texts": len(rows), "full_unique_train_texts": len(full_rows),
                        "supervised_occurrences": material["occurrences"],
                        "feature_evidence": material["evidence"],
                        "source_provenance": material["source_provenance"],
                        "normalizer_sha256": components.fingerprint(material["normalizer"]),
                        "selected_row_identity_sha256": components.record_identity_hash(
                            material["train"]["records"], material["selected"])})
        full_population_path = directory / "train_population_rows.jsonl"
        components.save_rows(full_population_path, full_rows)
        input_rows = []
        for order, row in enumerate(rows):
            record = material["train"]["records"][row["record_index"]]
            page = record["pages"][row["page_index"]]
            require(record["split"] == "train", "prerequisite input escaped train")
            input_rows.append({**row, "order": order, "raw_extent": row["target"],
                               "property_id": row["field_key"], "family_id": row["family"],
                               "row_id": record["row_id"], "world_id": record["world_id"],
                               "candidate_id": record["candidate_id"], "term_index": record["term_index"],
                               "page_block_id": page["block_id"]})
        input_path = directory / "train_input_rows.jsonl"
        components.save_rows(input_path, input_rows)
        artifact_paths = {"full_population_rows": full_population_path, "input_rows": input_path}
        for name, array in (("raw_targets", targets), ("optimizer_indices", material["selected"])):
            path = directory / f"train_{name}.npy"
            components.save_array(path, array)
            artifact_paths[name] = path
        trainer._seed()
        payload["stage"] = "encoder_and_cache"
        with screens._open_encoder(device) as encoder:
            require(encoder.parameter_hash() == screens.PINNED_ENCODER_PARAMETERS_SHA256,
                    "loaded encoder parameters differ from pinned base")
            ids, masks = encoder.tokenize(texts)
            require(ids.shape == masks.shape == (len(rows), screens.MAX_TOKENS), "train token shape changed")
            for name, array in (("input_ids", ids), ("attention_mask", masks)):
                path = directory / f"train_{name}.npy"
                components.save_array(path, array)
                artifact_paths[name] = path
            adapted, adapters = screens.attach_trainable_surface(torch, encoder, device)
            reader, checkpoint_normalizer, reader_origin = screens.load_pinned_reader(device)
            require(reader_origin["source"] == "pinned_pages_checkpoint"
                    and reader_origin["sha256"] == screens.PAGES_CHECKPOINT_SHA256,
                    "reader was not initialized from the pinned pages checkpoint")
            require(components.fingerprint(checkpoint_normalizer)
                    == components.fingerprint(material["normalizer"]), "reader common normalizer changed")
            for name, parameter in reader.named_parameters():
                parameter.requires_grad_(name in {"wv.weight", "wv.bias"})
            reader.eval()
            adapted.eval()
            payload["trainable"] = _surface_report(adapted, reader)
            frozen_inventory = _frozen_inventory(adapted, reader)
            frozen_hash = _frozen_hash(frozen_inventory)
            payload["frozen_parameters_sha256"] = frozen_hash
            cache, replay = _cache_prefix(adapted, encoder, ids, masks, device, payload["counters"])
            require(_frozen_hash(frozen_inventory) == frozen_hash, "frozen bytes changed during cache capture")
            replay["frozen_parameters_sha256"] = frozen_hash
            payload["initial_replay"] = _json(directory / "initial_replay_receipt.json", replay)
            payload.update({"environment": trainer._runtime_environment(device),
                            "encoder_lineage": encoder.lineage, "reader_initialization": reader_origin,
                            "input_artifacts": {name: evaluator.artifact(path)
                                                for name, path in artifact_paths.items()},
                            "optimizer_policy": {"name": "AdamW", "encoder_lr": screens.ENCODER_LR,
                                                 "adapter_lr": screens.ADAPTER_LR,
                                                 "reader_lr": screens.READER_LR,
                                                 "weight_decay": screens.WEIGHT_DECAY,
                                                 "chunk_size": screens.CHUNK_SIZE,
                                                 "denominator": len(rows), "updates_per_epoch": 1},
                            "normalizer": evaluator.plain(material["normalizer"])})
            payload["launch_receipt"] = _json(directory / "launch_receipt.json", payload)
            # Every input, lineage and replay receipt above is durable before optimizer creation.
            optimizer = screens._optimizer(adapted, reader)
            stats = material["normalizer"]["pages"]
            mean = torch.as_tensor(stats["mean"], dtype=torch.float32, device=device)
            std = torch.as_tensor(stats["std"], dtype=torch.float32, device=device)
            target_tensor = torch.as_tensor(targets, dtype=torch.float32, device=device)
            layer = adapted.deberta.encoder.layer[11]
            counters = payload["counters"]
            history_path = directory / "epoch_history.jsonl"
            payload["stage"] = "fitting"
            with history_path.open("x", encoding="utf-8") as history_stream:
                for epoch in range(1, epochs + 1):
                    features._resource_guard(torch, encoder.initial_swap_mib)
                    require(not adapted.training and not layer.training, "encoder must remain in eval mode")
                    before = _frozen_hash(frozen_inventory)
                    require(before == frozen_hash, "frozen bytes changed; cached prefix invalidated")
                    optimizer.zero_grad(set_to_none=True)
                    mse = 0.0
                    for batch_index, batch in enumerate(cache):
                        features._resource_guard(torch, encoder.initial_swap_mib)
                        with torch.enable_grad():
                            pooled = _replay(layer, batch, device, counters)
                            normalized = (pooled - mean) / std
                            prediction = torch.sigmoid(reader.wv(normalized).squeeze(-1))
                            difference = prediction - target_tensor[batch["start"]:batch["stop"]]
                            loss = difference.square().sum() / len(rows)
                            require(bool(torch.isfinite(loss)), "nonfinite scalar raw grade MSE")
                            loss.backward()
                        counters["reader_value_forward_chunks"] += 1
                        counters["loss_backward_calls"] += 1
                        mse += float(loss.detach())
                        if epoch == 1 and batch_index == 0:
                            first = _gradient_evidence(adapted, reader)
                            first["pass"] = first["all_components_nonzero"]
                            payload["first_backward"] = _json(directory / "first_backward_receipt.json", first)
                            require(first["pass"],
                                    "first backward must reach layer11, all six adapters and existing wv")
                    gradients = _gradient_evidence(adapted, reader)
                    optimizer.step()
                    counters["optimizer_steps"] += 1
                    after = _frozen_hash(frozen_inventory)
                    payload["last_byte_guard"] = {"epoch": epoch, "before": before, "after": after,
                                                  "expected": frozen_hash}
                    require(after == frozen_hash, "frozen parameter bytes changed after optimizer step")
                    require(all(bool(torch.isfinite(parameter).all()) for parameter in
                                (*adapted.parameters(), *reader.wv.parameters()) if parameter.requires_grad),
                            "optimizer produced nonfinite trainable parameters")
                    history = {"epoch": epoch, "train_grade_MSE_pre_update": mse,
                               "gradient_norm": gradients["gradient_norm"],
                               "frozen_sha256_before": before, "frozen_sha256_after": after,
                               "all_gradients_finite": True, "optimizer_steps": counters["optimizer_steps"]}
                    payload["history"].append(history)
                    history_stream.write(json.dumps(history, sort_keys=True, allow_nan=False) + "\n")
                    history_stream.flush()
                    os.fsync(history_stream.fileno())
            require(counters["optimizer_steps"] == epochs, "fixed accumulated update count changed")
            require(counters["frozen_prefix_texts_encoded"] == len(rows), "frozen prefix was not reused once")
            final_prediction = np.empty(len(rows), dtype=np.float32)
            with torch.no_grad():
                for batch in cache:
                    features._resource_guard(torch, encoder.initial_swap_mib)
                    pooled = _replay(layer, batch, device, counters)
                    prediction = torch.sigmoid(reader.wv((pooled - mean) / std).squeeze(-1))
                    require(bool(torch.isfinite(prediction).all()), "nonfinite final train raw grade prediction")
                    final_prediction[batch["start"]:batch["stop"]] = prediction.cpu().numpy()
                    counters["reader_value_forward_chunks"] += 1
            prediction_path = directory / "final_train_raw_predictions.npy"
            components.save_array(prediction_path, final_prediction)
            require(_sources() == payload["source_hashes"], "implementation sources changed during fitting")
            require(features.sha256_file(GRADIENT_PREREGISTRATION_PATH) == GRADIENT_PREREGISTRATION_SHA256,
                    "gradient protocol changed during fitting")
            checkpoint = {"schema": "vey.eca2.c1-gradient-prerequisite-checkpoint.v1",
                          "adapted_state_dict": {name: value.detach().cpu().clone()
                                                 for name, value in adapted.state_dict().items()},
                          "reader_value_state_dict": {name: value.detach().cpu().clone()
                                                      for name, value in reader.wv.state_dict().items()},
                          "stats": stats, "normalizer": material["normalizer"],
                          "protocol_sha256": GRADIENT_PREREGISTRATION_SHA256,
                          "corpus_protocol_sha256": experiment.protocol()[1],
                          "experiment_context": experiment.context(), "population_sha256": population_hash,
                          "full_population_sha256": payload["full_population_sha256"],
                          "source_hashes": payload["source_hashes"], "smoke": smoke, "epochs": epochs,
                          "seed": screens.SEED, "unique_train_texts": len(rows),
                          "adapter_names": list(screens.ADAPTER_NAMES),
                          "frozen_parameters_sha256": frozen_hash, "eligible_for_C1": not smoke,
                          "launch_receipt": payload["launch_receipt"]}
            checkpoint_path = directory / CHECKPOINT_NAME
            with checkpoint_path.open("xb") as stream:
                torch.save(checkpoint, stream)
                stream.flush()
                os.fsync(stream.fileno())
            payload["checkpoint"] = evaluator.artifact(checkpoint_path)
            payload["stage"] = "strict_cpu_reload"
            reloaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            require(reloaded["source_hashes"] == _sources()
                    and reloaded["protocol_sha256"] == GRADIENT_PREREGISTRATION_SHA256
                    and reloaded["population_sha256"] == screens.gradient_population_sha256(rows)
                    and reloaded["smoke"] is smoke and reloaded["epochs"] == epochs
                    and components.fingerprint(reloaded["normalizer"])
                    == components.fingerprint(material["normalizer"]), "saved prerequisite custody mismatch")
            adapted.to("cpu")
            reader.to("cpu")
            adapted.load_state_dict(reloaded["adapted_state_dict"], strict=True)
            reader.wv.load_state_dict(reloaded["reader_value_state_dict"], strict=True)
            require(_frozen_hash(_frozen_inventory(adapted, reader)) == frozen_hash,
                    "strict CPU reload changed frozen bytes")
            payload.update({"strict_cpu_reload": True, "epoch_history": evaluator.artifact(history_path),
                            "final_train_predictions": evaluator.artifact(prediction_path),
                            "final_train_grade_MSE": float(np.square(final_prediction.astype(np.float64)
                                                                      - targets).mean()),
                            "final_encoder_hashes": screens.parameter_hashes(adapted, adapters),
                            "eligible_for_C1": not smoke, "stage": "complete", "verdict": "pass",
                            "quality_claim": False,
                            "scope": "execution prerequisite only; fresh CPU FP64 C1 ceiling owns quality verdict"})
        payload["receipt"] = _json(receipt_path, payload)
        return payload
    except BaseException as exc:
        payload["failure_reason"] = f"{type(exc).__name__}: {exc}"
        payload["failure_class"] = "execution_or_invariant; no model quality interpretation"
        payload["eligible_for_C1"] = False
        payload["receipt"] = _json(receipt_path, payload)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--features-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    receipt = train_prerequisite(args.run_root, args.features_root, args.device, args.smoke)
    print(json.dumps({"verdict": receipt["verdict"], "smoke": receipt["smoke"],
                      "epochs": receipt["epochs"], "eligible_for_C1": receipt["eligible_for_C1"],
                      "receipt": receipt["receipt"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
