"""Tier 2: Liquid AI's open d1-omni-600M decision model, int8, on this machine.

d1-omni-600M (587M parameters: a 381M bidirectional LFM2.5-Encoder-350M trunk and
decision head, a 94M SigLIP2 vision tower, a 112M audio tower) reads text and
images and answers ``noul``, ``choice`` and ``score`` questions in one forward
pass per question. The full server starts at about 0.8 GB RSS and reaches about
1.0 GB after images; its measured peak exceeds 1 GiB. The audio tower is not loaded.

License: the weights are under the **LFM Open License v1.0**, not Vey's. In short,
commercial use is licensed only while you and your affiliates stay under $10M in
annual revenue, redistribution must carry the license text and notices of any
change, and an int8 conversion is a Derivative Work. So Vey never ships or hosts
these weights. :func:`prepare` downloads them from the Hub at a pinned revision and
converts them **on your machine**, writes the license and a modification notice
beside the result, and the code below only loads that local copy. Read the license
(``LICENSE`` in the cache directory) before you use it commercially.

It runs the model's own code from that pinned revision (the same trust as
``trust_remote_code``) and needs PyTorch: ``pip install 'vey[d1]'``.

How it is quantized, and why:

  * Large weights are per-output-channel symmetric int8; activations are quantized
    per token on the fly and multiplied with ``torch._int_mm``. This is not a
    bitwise-equivalent replacement for fp32; evaluate it on your own tasks.
  * The 16 ``out_proj`` layers stay weight-only: with them quantized too, an
    answer that was a near-tie in fp32 flipped.
  * One request at a time, a 256-token batching budget, and the heap is trimmed
    after each request. An individual long question can exceed that budget.
"""
from __future__ import annotations

import ctypes
import importlib.util
import json
import os
import sys
import threading
import types
from pathlib import Path
from typing import Any

D1_REPO = "LiquidAI/d1-omni-600M"
D1_REVISION = "02b55d7076f15129e59ab3f94783f32c4b088674"
LICENSE_URL = "https://huggingface.co/LiquidAI/d1-omni-600M/blob/main/LICENSE"
WEIGHT_ONLY = ("out_proj",)  # kept weight-only: quantizing their activations flipped a near-tie
SCALE = ".__scale"
GROUPS = {"text": ("encoder.", "head."), "vision": ("vision.",)}  # audio is not loaded
CODE_FILES = ["audio.py", "encoder.py", "modeling_d1.py", "prompt.py", "vision.py"]


class D1Unavailable(RuntimeError):
    """d1 cannot be used here: PyTorch missing, weights not prepared, or a version mismatch."""


def default_cache() -> Path:
    return Path(os.environ.get("VEY_CACHE", Path.home() / ".cache" / "vey")) / "d1" / D1_REVISION[:12]


def _torch():
    try:
        import torch
        import torch.nn as nn
        from safetensors import safe_open  # noqa: F401
    except ModuleNotFoundError as e:
        raise D1Unavailable("the d1 lane needs PyTorch: install it with `pip install 'vey[d1]'`") from e
    return torch, nn


# ---------------------------------------------------------------- conversion

def prepare(cache: Path | str | None = None, log=lambda m: print(m, file=sys.stderr, flush=True)) -> Path:
    """Download the pinned model and convert its weights to int8 locally. Idempotent.

    Conversion needs more memory than inference; run it on a machine with several
    GB available. The converted files are Derivative Works under the LFM Open
    License v1.0 and stay in ``cache``; Vey does not distribute them.
    """
    torch, _ = _torch()
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    cache = Path(cache) if cache else default_cache()
    required = [*(f"{g}.safetensors" for g in GROUPS), *CODE_FILES, "config.json", "tokenizer.json",
                "tokenizer_config.json", "LICENSE", "NOTICE.txt"]
    if all((cache / name).exists() for name in required):
        return cache
    cache.mkdir(parents=True, exist_ok=True)
    repo, revision = D1_REPO, D1_REVISION
    log(f"d1: downloading {repo}@{revision[:8]} (about 2.3 GB). The weights are under the LFM Open License v1.0 "
        f"({LICENSE_URL}): commercial use requires annual revenue under $10M.")
    src = Path(snapshot_download(repo, revision=revision, allow_patterns=[
        *CODE_FILES, "config.json", "tokenizer.json", "tokenizer_config.json", "LICENSE", "model.safetensors"]))
    f = safe_open(str(src / "model.safetensors"), "pt")
    for group, prefixes in GROUPS.items():
        out: dict[str, Any] = {}
        for k in f.keys():
            if not k.startswith(prefixes):
                continue
            t = f.get_tensor(k)
            if t.ndim == 2 and t.numel() >= 50_000 and "position_embedding" not in k:
                w = t.float()
                scale = (w.abs().amax(1) / 127).clamp_min(1e-10)
                out[k] = torch.clamp((w / scale[:, None]).round(), -127, 127).to(torch.int8).contiguous()
                out[k + SCALE] = scale.contiguous()
            else:
                out[k] = t.float().contiguous()
            del t
        save_file(out, str(cache / f"{group}.safetensors"))
        log(f"d1: converted {group} ({os.path.getsize(cache / f'{group}.safetensors') / 1e6:.0f} MB)")
        del out
    for name in [*CODE_FILES, "config.json", "tokenizer.json", "tokenizer_config.json", "LICENSE"]:
        (cache / name).write_bytes((src / name).read_bytes())
    (cache / "NOTICE.txt").write_text(
        f"These files are a modified form of {repo} at revision {revision}, Copyright Liquid AI, Inc.,\n"
        "used under the LFM Open License v1.0 (see LICENSE in this directory).\n"
        "Modification: the large weight matrices were converted to per-channel int8 by Vey on this machine,\n"
        "and the audio tower was dropped. Redistribution carries the license text and this notice, and the\n"
        "license limits commercial use to entities with annual revenue under $10,000,000.\n")
    return cache


# ------------------------------------------------------------------ int8 ops

_INT_MM: bool | None = None


def _int_mm_works() -> bool:
    global _INT_MM
    if _INT_MM is None:
        torch, _ = _torch()
        try:
            torch._int_mm(torch.zeros(32, 8, dtype=torch.int8), torch.zeros(8, 8, dtype=torch.int8))
            _INT_MM = True
        except Exception:  # an older or unusual build: fall back to weight-only int8
            _INT_MM = False
    return _INT_MM


def _make_classes():
    torch, nn = _torch()

    class QLinear(nn.Module):
        """int8 weight, per-channel scale. W8A8 through ``_int_mm`` unless ``weight_only``."""

        def __init__(self, q, scale, bias, weight_only: bool):
            super().__init__()
            self.out_features, self.in_features = q.shape  # the vision projector reads these
            self.weight = torch.empty(0, dtype=torch.float32)  # callers read .weight.dtype
            self.bias = None if bias is None else nn.Parameter(bias, requires_grad=False)
            self.weight_only = weight_only or not _int_mm_works()
            if self.weight_only:
                self.register_buffer("q", q)
            else:
                self.register_buffer("qT", q.t().contiguous())  # (K, N): the only copy
            self.register_buffer("scale", scale)

        def forward(self, x):
            if self.weight_only:
                return nn.functional.linear(x, self.q.float() * self.scale[:, None], self.bias)
            shape = x.shape
            x2 = x.reshape(-1, shape[-1]).float()
            s = x2.abs().amax(1, keepdim=True).clamp_min(1e-8) / 127
            xi = torch.clamp((x2 / s).round(), -127, 127).to(torch.int8)
            m = xi.shape[0]
            if m < 32:
                xi = torch.cat([xi, xi.new_zeros(32 - m, xi.shape[1])])
            y = torch._int_mm(xi, self.qT)[:m].float() * s * self.scale[None, :]
            if self.bias is not None:
                y = y + self.bias
            return y.reshape(*shape[:-1], -1)

    class QEmbedding(nn.Module):
        def __init__(self, q, scale):
            super().__init__()
            self.register_buffer("q", q)
            self.register_buffer("scale", scale)
            self.weight = torch.empty(0, dtype=torch.float32)

        def forward(self, ids):
            return self.q[ids].float() * self.scale[ids][..., None]

    class LazyDequant:
        """Head layer weights stay int8 and become fp32 only while that layer runs."""

        def __init__(self, layer, items):
            self.items = items
            layer.register_forward_pre_hook(self.pre)
            layer.register_forward_hook(self.post)

        def pre(self, *_):
            for p, q, s in self.items:
                p.data = q.float() * s[:, None]

        def post(self, *_):
            for p, _q, _s in self.items:
                p.data = torch.empty(0)

    class Reader:
        """safetensors reader with plain reads: no mmap, so no file pages stay mapped."""
        DT = {"F32": torch.float32, "I8": torch.int8}

        def __init__(self, path):
            self.fp = open(path, "rb")
            n = int.from_bytes(self.fp.read(8), "little")
            self.h = json.loads(self.fp.read(n))
            self.h.pop("__metadata__", None)
            self.base = 8 + n
            # transformers 5 flattened the vision tower: its keys lose "vision_model."
            self.alias = {k.replace("tower.vision_model.", "tower."): k for k in self.h}

        def keys(self):
            return list(self.alias)

        def get_tensor(self, k):
            e = self.h[self.alias[k]]
            a, b = e["data_offsets"]
            t = torch.empty(b - a, dtype=torch.uint8)
            self.fp.seek(self.base + a)
            self.fp.readinto(memoryview(t.numpy()))
            return t.view(self.DT[e["dtype"]]).reshape(e["shape"])

        def close(self):
            self.fp.close()

    return QLinear, QEmbedding, LazyDequant, Reader


# --------------------------------------------------------------------- model

def _trim_heap() -> None:
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:  # not glibc
        pass


class D1Model:
    """d1-omni-600M, int8, text and image questions. One request at a time."""

    def __init__(self, cache: Path | str | None = None, max_image_side: int = 512, batch_tokens: int = 256,
                 threads: int | None = None):
        self.cache = Path(cache) if cache else default_cache()
        required = [*(f"{g}.safetensors" for g in GROUPS), *CODE_FILES, "config.json", "tokenizer.json",
                    "tokenizer_config.json", "LICENSE", "NOTICE.txt"]
        if not all((self.cache / name).is_file() for name in required):
            raise D1Unavailable(f"d1 weights are not prepared in {self.cache}; run `vey-d1-prepare` first")
        self.torch, self.nn = _torch()
        if threads:
            self.torch.set_num_threads(threads)
        self.max_image_side = max_image_side
        self._classes = _make_classes()
        self._lock = threading.Lock()
        self._loaded: set[str] = set()
        self.model = self._skeleton()
        self._load("text")
        orig = self.model._run
        self.model._run = lambda rows, max_tokens=batch_tokens: orig(rows, max_tokens)
        self.requests = 0

    # -- loading
    def _skeleton(self):
        pkg = types.ModuleType("vey_d1_code")
        pkg.__path__ = [str(self.cache)]
        sys.modules["vey_d1_code"] = pkg
        mods = {}
        for n in ["audio", "encoder", "prompt", "vision", "modeling_d1"]:
            spec = importlib.util.spec_from_file_location(f"vey_d1_code.{n}", self.cache / f"{n}.py")
            m = importlib.util.module_from_spec(spec)
            sys.modules[f"vey_d1_code.{n}"] = m
            try:
                spec.loader.exec_module(m)
            except ModuleNotFoundError as e:
                raise D1Unavailable(f"d1 needs more packages ({e}); install them with `pip install 'vey[d1]'`") from e
            mods[n] = m
        raw = json.load(open(self.cache / "config.json"))
        keys = ("text_config", "vision_config", "audio_config", "projector_hidden_size", "head_layers",
                "max_length", "image_text_length", "audio_text_length", "temperatures")
        cfg = mods["modeling_d1"].D1OmniConfig(**{k: v for k, v in raw.items() if k in keys})
        cfg._name_or_path = str(self.cache)
        try:
            with self.torch.device("meta"):
                model = mods["modeling_d1"].D1OmniModel(cfg)
        except Exception as e:  # a transformers too old or new for the model's code
            raise D1Unavailable(f"d1's code did not build with this transformers ({e}); it needs transformers>=5.15") from e
        return model.eval()

    def _load(self, group: str) -> None:
        nn, torch = self.nn, self.torch
        QLinear, QEmbedding, LazyDequant, Reader = self._classes
        f = Reader(str(self.cache / f"{group}.safetensors"))
        keys = f.keys()
        have = set(keys)
        lazy: dict[str, list] = {}
        for k in keys:
            if k.endswith(SCALE):
                continue
            t = f.get_tensor(k)
            *path, name = k.split(".")
            parent = self.model.get_submodule(".".join(path)) if path else self.model
            if k + SCALE not in have:
                if name in parent._buffers:
                    parent._buffers[name] = t
                else:
                    parent._parameters[name] = nn.Parameter(t, requires_grad=False)
                continue
            scale = f.get_tensor(k + SCALE)
            if isinstance(parent, nn.Embedding) and "embed_tokens" in k:
                repl = QEmbedding(t, scale)
            elif isinstance(parent, nn.Linear) and name == "weight" and "head.head.layers" not in k:
                bias = f.get_tensor(".".join(path) + ".bias") if ".".join(path) + ".bias" in have else None
                repl = QLinear(t, scale, bias, any(w in k for w in WEIGHT_ONLY))
            else:  # head transformer layers: the fused fast path reads their weights directly
                lazy.setdefault(".".join(k.split(".")[:4]), []).append((k, t, scale))
                continue
            gp = ".".join(path[:-1])
            setattr(self.model.get_submodule(gp) if gp else self.model, path[-1], repl)
        for layer_path, items in lazy.items():
            layer, resolved = self.model.get_submodule(layer_path), []
            for k, t, s in items:
                *path, name = k.split(".")
                parent = self.model.get_submodule(".".join(path))
                parent._parameters[name] = nn.Parameter(torch.empty(0), requires_grad=False)
                resolved.append((parent._parameters[name], t, s))
            LazyDequant(layer, resolved)
        f.close()
        self._loaded.add(group)
        _trim_heap()

    # -- inference
    def _fit(self, image):
        """Shrink an image so its longest side is at most ``max_image_side`` (one 256-token tile)."""
        w, h = image.size
        s = max(w, h)
        if s <= self.max_image_side:
            return image.convert("RGB")
        from PIL import Image

        r = self.max_image_side / s
        return image.convert("RGB").resize((max(1, round(w * r)), max(1, round(h * r))), Image.BILINEAR)

    def probabilities(self, state: Any, questions: list[dict[str, Any]], images: list | None = None
                      ) -> list[list[float]]:
        """Each question's distribution over its options: ``[yes, no]`` for a noul."""
        with self._lock:
            if images:
                if "vision" not in self._loaded:
                    self._load("vision")
                images = [self._fit(i) for i in images]
            try:
                out = self.model.probabilities(state, questions, images=images or None)
            finally:
                _trim_heap()
            self.requests += 1
            return out

    def info(self) -> dict[str, Any]:
        return {"repo": D1_REPO, "revision": D1_REVISION[:8], "loaded": sorted(self._loaded),
                "max_image_side": self.max_image_side, "requests": self.requests}


def decode_image(asset: dict[str, Any], name: str = "asset", max_bytes: int = 8 << 20):
    """A PIL image from an ``{"type": "image", "data_base64": ...}`` asset, or a 400."""
    import base64
    import binascii
    import io

    from .schema import bad_request

    if not isinstance(asset, dict):
        raise bad_request(f"asset {name!r} must be an image object")
    if asset.get("type") != "image":
        raise bad_request(f"asset {name!r} must have type 'image'")
    if asset.get("part"):
        raise bad_request(f"asset {name!r}: multipart image parts are not read locally; send data_base64")
    encoded = asset.get("data_base64")
    if not isinstance(encoded, str):
        raise bad_request(f"asset {name!r}: data_base64 must be a string")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise bad_request(f"asset {name!r}: data_base64 is not valid base64") from None
    if not raw or len(raw) > max_bytes:
        raise bad_request(f"asset {name!r}: image must be between 1 byte and {max_bytes} bytes")
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception:
        raise bad_request(f"asset {name!r}: could not decode the image (png, jpg and jpeg are supported)") from None
    return img


def prepare_main(argv: list[str] | None = None) -> int:
    """``vey-d1-prepare``: download d1-omni-600M and convert it to int8 on this machine."""
    import argparse

    p = argparse.ArgumentParser(prog="vey-d1-prepare", description=prepare.__doc__)
    p.add_argument("--cache", help=f"output directory (default {default_cache()})")
    a = p.parse_args(argv)
    d = prepare(a.cache)
    print(f"ready: {d}")
    return 0


__all__ = ["D1Model", "D1Unavailable", "decode_image", "prepare", "prepare_main", "default_cache", "D1_REPO",
           "D1_REVISION"]


if __name__ == "__main__":
    raise SystemExit(prepare_main())
