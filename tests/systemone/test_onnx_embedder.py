"""OnnxEmbedder behavior that does not need the real model: a stub session stands in for ONNX Runtime."""
import threading

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from vey.semantic.onnx_encoder import OnnxEmbedder

VOCAB = {"[PAD]": 0, "[UNK]": 1, **{w: i + 2 for i, w in enumerate("a b c d e f g h".split())}}


class StubSession:
    """Returns each token's id broadcast over 4 dims, so pooling is checkable by hand."""

    class _Spec:
        def __init__(self, name, shape):
            self.name, self.shape = name, shape

    def __init__(self):
        self.batches = []

    def get_inputs(self):
        return [self._Spec("input_ids", [None, None]), self._Spec("attention_mask", [None, None])]

    def get_outputs(self):
        return [self._Spec("last_hidden_state", [None, None, 4])]

    def run(self, _, feed):
        self.batches.append(feed["input_ids"].shape)
        ids = feed["input_ids"].astype(np.float32)
        return [np.repeat(ids[:, :, None], 4, axis=2) + np.array([0, 1, 2, 3], np.float32)]


def make(**kw):
    tok = Tokenizer(models.WordLevel(VOCAB, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    sess = StubSession()
    return OnnxEmbedder(sess, tok, **kw), sess


def test_vectors_are_unit_length_and_pooled_over_real_tokens_only():
    e, _ = make()
    v = e.embed(["a b", "a b c d"])
    assert v.shape == (2, 4) and v.dtype == np.float32
    assert np.linalg.norm(v, axis=1) == pytest.approx(1.0, abs=1e-6)
    # "a b" -> ids 2,3 -> mean 2.5 + [0,1,2,3] ; padding must not enter the mean
    expect = np.array([2.5, 3.5, 4.5, 5.5]); expect /= np.linalg.norm(expect)
    assert v[0] == pytest.approx(expect, abs=1e-6)


def test_each_text_is_encoded_alone_so_neighbours_cannot_change_it():
    e, sess = make()
    e.embed(["a b c d e f", "a"])
    assert sorted(shape[0] for shape in sess.batches) == [1, 1]  # never batched together
    assert sorted(shape[1] for shape in sess.batches) == [1, 6]  # and never padded to each other
    alone = e.embed(["a"])[0]
    e2, _ = make()
    assert e2.embed(["a b c d e f", "a"])[1] == pytest.approx(alone, abs=0)


def test_cache_encodes_each_distinct_text_once_and_is_bounded():
    e, sess = make(cache_size=3)
    e.embed(["a", "b", "a", "b"])
    assert len(sess.batches) == 2
    e.embed(["a", "b", "c", "d"])  # c and d are new; capacity 3 evicts the oldest
    assert len(e._cache) == 3
    n = len(sess.batches)
    e.embed(["d"])
    assert len(sess.batches) == n  # still cached


def test_empty_input_and_token_counting():
    e, _ = make()
    assert e.embed([]).shape == (0, 4)
    assert e.count_tokens("a b c") == 3


def test_concurrent_embedding_is_safe():
    e, _ = make()
    out, errs = {}, []

    def work(i):
        try:
            out[i] = e.embed(["a b", f"c {'d ' * (i % 4)}", "e f"])
        except Exception as ex:  # noqa: BLE001
            errs.append(ex)

    ts = [threading.Thread(target=work, args=(i,)) for i in range(16)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert not errs and all(np.allclose(out[i][0], out[0][0]) for i in out)
