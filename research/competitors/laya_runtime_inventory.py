#!/usr/bin/env python3
"""Inventory pinned Laya artifacts and exercise routing without loading models."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys

SOURCE_REVISION = "859b8ee595cc04f84dd2af476d6d1d90ec1fea46"
BUNDLE_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
ROUTES = {"english": "", "typed-decisions": "typed-decisions", "multilingual": "multilingual"}
ARTIFACTS = (
    "rl_agent_config.json", "encoder/config.json",
    "tokenizer/tokenizer_config.json", "tokenizer/tokenizer.json", "model.safetensors",
)


def file_hash(path, *, git_blob=False):
    digest = hashlib.sha1(usedforsecurity=False) if git_blob else hashlib.sha256()
    if git_blob:
        digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def inventory_bundle(bundle_root, upstream_manifest):
    from safetensors import safe_open

    if bundle_root.name != BUNDLE_REVISION:
        raise ValueError("bundle directory does not identify the reviewed revision")
    if (upstream_manifest["repository"] != "convaiinnovations/laya"
            or upstream_manifest["revision"] != BUNDLE_REVISION):
        raise ValueError("upstream identities do not identify the reviewed bundle")
    repository_root = bundle_root.parent.parent.resolve()
    shared_blob_root = repository_root.parent / "blobs"
    results = {}
    for route, subfolder in ROUTES.items():
        root = bundle_root / subfolder
        files = {}
        for relative in ARTIFACTS:
            path = root / relative
            resolved = path.resolve(strict=True)
            if not (resolved.is_relative_to(repository_root)
                    or resolved.is_relative_to(shared_blob_root)):
                raise ValueError(f"artifact escapes its Hub cache: {path}")
            stat = resolved.stat()
            if not resolved.is_file() or stat.st_size == 0:
                raise ValueError(f"missing or empty artifact: {path}")
            upstream_path = f"{subfolder}/{relative}".lstrip("/")
            expected = upstream_manifest["files"][upstream_path]
            if stat.st_size != expected["size"]:
                raise ValueError(f"upstream artifact size mismatch: {path}")
            digest = file_hash(resolved)
            identity_kind = "lfs_sha256" if "lfs" in expected else "git_blob_sha1"
            expected_identity = expected["lfs"]["oid"] if "lfs" in expected else expected["oid"]
            actual_identity = digest if "lfs" in expected else file_hash(resolved, git_blob=True)
            if actual_identity != expected_identity:
                raise ValueError(f"upstream artifact identity mismatch: {path}")
            files[relative] = {
                "path": str(path), "resolved_path": str(resolved),
                "bytes": stat.st_size, "sha256": digest,
                "upstream_identity_kind": identity_kind,
                "upstream_identity": expected_identity, "upstream_identity_verified": True,
                "cache_blob_name": resolved.name,
                "upstream_xet_hash": expected.get("xetHash"),
            }
        with safe_open(root / "model.safetensors", framework="np") as checkpoint:
            shapes = {name: checkpoint.get_slice(name).get_shape() for name in checkpoint.keys()}
        config = json.loads((root / "rl_agent_config.json").read_text())
        results[route] = {
            "repository": "convaiinnovations/laya", "revision": BUNDLE_REVISION,
            "subfolder": subfolder or None, "files": files,
            "safetensors_header_valid": True,
            "stored_tensor_count": len(shapes),
            "stored_scalar_count_including_buffers": sum(math.prod(shape) for shape in shapes.values()),
            "stored_scalar_count_is_live_parameter_count": False,
            "agent_config": config,
        }
    return results


def route_probe(source_root):
    actual_revision = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True,
    ).strip()
    if actual_revision != SOURCE_REVISION:
        raise ValueError("source checkout differs from the reviewed revision")
    if "laya" in sys.modules or "torch" in sys.modules:
        raise RuntimeError("routing probe requires a fresh non-model process")
    sys.path.insert(0, str(source_root))
    laya = importlib.import_module("laya")
    if Path(laya.__file__).resolve() != source_root / "laya/__init__.py":
        raise ValueError("import resolved the legacy installed package")
    router = laya.Router(device="cpu", revision=BUNDLE_REVISION, preload=False)
    decisions = {
        "english": dict(router.route("Non-sensitive routing probe", lang="en")),
        "typed-decisions": dict(router.route("Non-sensitive routing probe", task="typed_decisions")),
        "multilingual": dict(router.route("Non-sensitive routing probe", lang="hi")),
    }
    if any(decision["model"] != route for route, decision in decisions.items()):
        raise AssertionError("recommended explicit route did not select its checkpoint")
    if router.loaded or "torch" in sys.modules:
        raise AssertionError("route-only probe loaded a model or imported torch")
    return {
        "source_root": str(source_root), "source_revision": actual_revision,
        "package_version": laya.__version__, "imported_file": laya.__file__,
        "source_files": {str(path.relative_to(source_root)): file_hash(path)
                         for path in sorted((source_root / "laya").rglob("*.py"))},
        "routing_decisions": decisions, "resident_models": router.loaded,
        "torch_imported": False, "model_forwards": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--upstream-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to replace an inventory receipt")
    if os.environ.get("HF_TOKEN") or os.environ.get("LAYA_SHA256_DIGESTS"):
        raise ValueError("run with credential and ambient digest variables removed")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise ValueError("offline guards are required")
    result = {
        "schema": "vey.laya.runtime-artifact-custody.v1",
        "scope": "Pinned source/routing and artifact byte/header custody only; inference compatibility, model quality, latency and residency unmeasured.",
        "routing": route_probe(args.source_root.resolve()),
        "bundle": inventory_bundle(args.bundle_root.resolve(), json.loads(args.upstream_manifest.read_text())),
        "upstream_manifest": {"path": str(args.upstream_manifest.resolve()),
                              "sha256": file_hash(args.upstream_manifest)},
        "environment": {
            "python": sys.version, "executable": sys.executable,
            "packages": {name: importlib.metadata.version(name)
                         for name in ("torch", "transformers", "huggingface_hub", "safetensors", "numpy")},
        },
        "inventory_source_sha256": file_hash(Path(__file__)),
        "network_model_access": False, "final_benchmark_access": False,
        "gpu_or_model_execution": False, "neutral_comparison_earned": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"receipt": str(args.output), "sha256": file_hash(args.output),
                      "routes": sorted(result["routing"]["routing_decisions"]),
                      "model_forwards": 0, "verified": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
