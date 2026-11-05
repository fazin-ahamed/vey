"""Torch-free embedder: the same frozen 24M encoder, run on ONNX Runtime.

``OnnxEmbedder`` has the same contract as :class:`vey.semantic.Embedder`
(``embed(texts) -> (n, dim)`` L2-normalized float32, cached by exact text), so
:class:`vey.semantic.CandidateMatcher` and the System One engine use either one.

It needs only ``onnxruntime``, ``tokenizers`` and ``numpy``. No PyTorch and no
``transformers`` are imported, which is what lets a serving process stay small:
see ``docs/SYSTEMONE.md`` for the measured memory.

The default weights are the ONNX exports published in the encoder's own Hugging
Face repository, pinned to a revision. ``variant`` picks the file:

    model_int8   dynamic int8 weights (default; smallest and fastest)
    model        fp32 (the reference; use it when int8 drift matters)

Pooling matches Vey's torch embedder exactly: masked mean over the last hidden
state, then L2 normalization, with the same 96-token truncation.

Each text is encoded on its own (``batch_size=1``). That is deliberate: ONNX
Runtime's dynamic int8 quantization scales activations over the whole batch, so a
text's embedding depends on which other texts share its batch. Measured on the
int8 export, the same text embedded alone and inside a batch differs by cosine
0.984 at worst (elementwise up to 0.03); the fp32 export is exactly unchanged.
Encoding alone keeps a candidate's score independent of the other candidates, the
property the matcher relies on, at about 1.5 ms per new text (each is cached).
Raise ``batch_size`` only for fp32, or when that invariance does not matter.
"""
from __future__ import annotations

import os
import threading
from collections import OrderedDict

import numpy as np

ONNX_REPO = "mixedbread-ai/mxbai-embed-xsmall-v1"
ONNX_REVISION = "e6ac24e5d6efb8782b59de1647b3ececb4ece94e"
DEFAULT_VARIANT = "model_int8"


class OnnxEmbedder:
    """Masked-mean, L2-normalized embeddings on ONNX Runtime, cached by text.

    The cache is a bounded LRU so a long-lived server cannot grow without limit,
    and every cache access holds a lock, so one instance serves many threads.
    """

    def __init__(self, session, tokenizer, max_len: int = 96, cache_size: int = 50_000, batch_size: int = 1):
        self.batch_size = batch_size
        self.session = session
        self.tok = tokenizer
        self.max_len = max_len
        self.cache_size = cache_size
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self._inputs = {i.name for i in session.get_inputs()}
        self.dim = int(session.get_outputs()[0].shape[-1])
        # A separate, untruncated copy for counting keeps encode() free of shared mutable state.
        from tokenizers import Tokenizer
        self._count_tok = Tokenizer.from_str(tokenizer.to_str())
        self._count_tok.no_padding()
        self._count_tok.no_truncation()
        self.tok.no_padding()
        self.tok.enable_truncation(max_length=max_len)
        self.tok.enable_padding(pad_id=0, pad_token="[PAD]")  # pad to the longest in the batch

    # ------------------------------------------------------------ construction

    @classmethod
    def from_dir(cls, path: str, variant: str = DEFAULT_VARIANT, threads: int | None = None,
                 max_len: int = 96, cache_size: int = 50_000, batch_size: int = 1) -> "OnnxEmbedder":
        """Load ``<path>/onnx/<variant>.onnx`` and ``<path>/tokenizer.json``."""
        import onnxruntime as ort
        from tokenizers import Tokenizer

        onnx_path = os.path.join(path, "onnx", f"{variant}.onnx")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads or min(4, os.cpu_count() or 1)
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
        tok = Tokenizer.from_file(os.path.join(path, "tokenizer.json"))
        return cls(session, tok, max_len=max_len, cache_size=cache_size, batch_size=batch_size)

    @classmethod
    def from_hub(cls, repo: str = ONNX_REPO, revision: str = ONNX_REVISION,
                 variant: str = DEFAULT_VARIANT, **kw) -> "OnnxEmbedder":
        """Download the pinned ONNX export once (cached by ``huggingface_hub``) and load it."""
        from huggingface_hub import hf_hub_download

        files = [hf_hub_download(repo, f, revision=revision)
                 for f in (f"onnx/{variant}.onnx", "tokenizer.json")]
        root = os.path.dirname(os.path.dirname(files[0]))
        return cls.from_dir(root, variant=variant, **kw)

    # ------------------------------------------------------------------- api

    def count_tokens(self, text: str) -> int:
        """Tokens this encoder would read for ``text`` (before truncation)."""
        return len(self._count_tok.encode(text).ids)

    def _encode(self, texts: list[str], batch: int | None = None) -> np.ndarray:
        batch = batch or self.batch_size
        out = []
        for i in range(0, len(texts), batch):
            enc = self.tok.encode_batch(texts[i:i + batch])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            hidden = self.session.run(None, feed)[0]
            m = mask[:, :, None].astype(np.float32)
            v = (hidden * m).sum(1) / np.maximum(m.sum(1), 1.0)
            out.append(v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12))
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, self.dim), np.float32)

    def embed(self, texts: list[str]) -> np.ndarray:
        with self._lock:
            missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            vecs = self._encode(missing)
            with self._lock:
                for t, v in zip(missing, vecs):
                    self._cache[t] = v
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
        with self._lock:
            rows = []
            for t in texts:
                v = self._cache.get(t)
                if v is None:  # evicted between the two steps by another thread
                    v = self._encode([t])[0]
                else:
                    self._cache.move_to_end(t)
                rows.append(v)
        return np.stack(rows) if rows else np.zeros((0, self.dim), np.float32)


__all__ = ["OnnxEmbedder", "ONNX_REPO", "ONNX_REVISION", "DEFAULT_VARIANT"]
