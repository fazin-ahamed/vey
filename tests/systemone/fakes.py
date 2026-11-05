"""Deterministic stand-ins for the two models, so the System One tests need no network."""
import re
import zlib

import numpy as np

_WORD = re.compile(r"[a-z0-9]+")
STOP = {"the", "a", "an", "is", "are", "this", "that", "it", "of", "to", "and", "or", "in", "on", "for", "example"}


class FakeEmbedder:
    """Hashed bag of words, L2-normalized. Texts that share words have a high cosine."""
    dim = 64

    def __init__(self):
        self.calls = 0
        self._cache = {}

    def count_tokens(self, text):
        return len(_WORD.findall(text.lower()))

    def embed(self, texts):
        self.calls += 1
        out = []
        for t in texts:
            if t not in self._cache:
                v = np.zeros(self.dim, np.float32)
                for w in _WORD.findall(t.lower()):
                    if w not in STOP:
                        v[zlib.crc32(w.encode()) % self.dim] += 1.0
                n = np.linalg.norm(v)
                self._cache[t] = v / n if n else v
            out.append(self._cache[t])
        return np.stack(out)


class FakeNLI:
    """Entails a hypothesis when one of its content words occurs in the premise."""

    def __init__(self):
        self.calls = 0

    def count_tokens(self, text):
        return len(_WORD.findall(text.lower()))

    def margin(self, pairs):
        self.calls += 1
        out = []
        for premise, hyp in pairs:
            p = set(_WORD.findall(premise.lower()))
            h = [w for w in _WORD.findall(hyp.lower()) if w not in STOP]
            out.append(3.0 if any(w in p for w in h) else -3.0)
        return np.array(out)


class UniformEmbedder(FakeEmbedder):
    """Every text gets the same vector: the matcher is maximally unsure."""

    def embed(self, texts):
        self.calls += 1
        return np.ones((len(texts), self.dim), np.float32) / np.sqrt(self.dim)
