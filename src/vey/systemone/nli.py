"""Yes/no (``noul``) answers from a zero-shot NLI model, on ONNX Runtime.

A cosine matcher cannot judge "does this message express urgency?". An NLI
cross-encoder can: premise = the state, hypothesis = the statement to check, and
P(yes) is the model's probability of entailment. The default model is
``MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33`` (MIT, 22M parameters of
transformer plus a 128k-token embedding table), run as the ONNX export published
in its Hugging Face repository, int8, with no PyTorch.

The model was trained on *statements*. A question collapses it: measured on 200
SST-2 reviews, "This movie review is positive." scored 0.795 accuracy and "Is
this movie review positive?" scored 0.485. :func:`to_statement` therefore
rewrites common yes/no questions into statements, deterministically.
"""
from __future__ import annotations

import json
import os
import re
import threading

import numpy as np

NLI_REPO = "MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33"
NLI_REVISION = "262ae02f29173eec1c250f90804dc7edc677dcff"
NLI_FILE = "model_quantized"

_AUX = ("is", "are", "was", "were", "does", "do", "did", "has", "have", "had", "can", "could", "will", "would",
        "should", "may", "might", "must")
_DET = {"the", "this", "that", "these", "those", "a", "an", "my", "our", "your", "their", "his", "her", "its",
        "any", "each", "every", "no", "some"}
_PRONOUN = {"i", "you", "we", "they", "he", "she", "it", "there"}
# Words that start a predicate rather than continue a noun phrase.
_PRED_START = {"about", "above", "across", "after", "against", "among", "around", "at", "before", "below",
               "between", "by", "for", "from", "in", "inside", "into", "near", "of", "on", "over", "to", "under",
               "with", "within", "without", "not", "very", "too", "also", "still", "currently", "already",
               "a", "an", "the", "this", "that", "these", "those", "any", "no", "some", "both", "all", "more",
               "less", "same", "likely", "able", "related", "regarding", "using", "be", "been", "being"}
_VERB_NEXT = {"does", "do", "did", "can", "could", "will", "would", "should", "may", "might", "must"}
_QUESTION = re.compile(r"^\s*(?P<aux>[A-Za-z]+)\s+(?P<rest>.+?)\s*\?\s*$", re.DOTALL)
_PLURAL = re.compile(r"(?:^|\s)(?:i|you|we|they|these|those)$|[a-z]s$", re.IGNORECASE)


def _third_person(verb: str) -> str:
    v = verb.lower()
    if v == "have":
        return "has"
    if re.search(r"(s|x|z|ch|sh|o)$", v):
        return v + "es"
    if re.search(r"[^aeiou]y$", v):
        return v[:-1] + "ies"
    return v + "s"


def _split_subject(words: list[str], head_only: bool) -> int:
    """Index where the predicate starts: determiner, head noun, an optional second
    noun, and any ``of`` phrases. The predicate always keeps at least one word."""
    n = len(words)
    if words[0].lower() in _PRONOUN:
        return 1
    i = 1 if words[0].lower() in _DET else 0
    if i == 1 and words[1].lower() in _DET:  # "this a billing issue": the determiner stands alone
        return 1
    i += 1  # head noun
    while i < n - 1 and words[i].lower() == "of":  # "the sentiment of this review"
        i += 1
        if i < n - 1 and words[i].lower() in _DET:
            i += 1
        i += 1
    if not head_only and i < n - 1:
        w = words[i].lower()
        if (w not in _PRED_START and not w.endswith(("ing", "ed")) and w not in _AUX and w not in _DET):
            i += 1
    return min(i, n - 1)


def to_statement(instructions: str) -> str:
    """Rewrite a yes/no question as a declarative statement; leave statements alone.

    ``Is this message about billing?`` -> ``This message is about billing.``
    ``Does this message express urgency?`` -> ``This message expresses urgency.``
    ``Is the sentiment of this movie review positive?`` -> ``The sentiment of this movie review is positive.``

    Text that is not a yes/no question (no leading auxiliary verb, or no question
    mark) is returned unchanged, so statements pass straight through. Questions
    whose subject cannot be found are returned unchanged too.
    """
    text = (instructions or "").strip()
    m = _QUESTION.match(text)
    if not m or m.group("aux").lower() not in _AUX:
        return text
    aux, words = m.group("aux").lower(), m.group("rest").split()
    if len(words) < 2:
        return text
    do_form = aux in ("does", "do", "did")
    k = _split_subject(words, head_only=aux in _VERB_NEXT)
    subj, pred = " ".join(words[:k]), " ".join(words[k:])
    subj = subj[:1].upper() + subj[1:]
    if do_form:
        verb, _, tail = pred.partition(" ")
        if aux == "did":
            conj = _past(verb)
        elif aux == "do" or _PLURAL.search(subj.split(" of ")[0]) or subj.lower() in ("i", "you", "we", "they"):
            conj = verb
        else:
            conj = _third_person(verb)
        return f"{subj} {conj} {tail}".strip().rstrip(".") + "."
    return f"{subj} {aux} {pred}".rstrip(".") + "."


def _past(verb: str) -> str:
    v = verb.lower()
    irregular = {"say": "said", "go": "went", "get": "got", "make": "made", "take": "took", "give": "gave",
                 "come": "came", "see": "saw", "pay": "paid", "buy": "bought", "send": "sent", "ask": "asked"}
    if v in irregular:
        return irregular[v]
    if v.endswith("e"):
        return v + "d"
    if re.search(r"[^aeiou]y$", v):
        return v[:-1] + "ied"
    return v + "ed"


class OnnxNLI:
    """Entailment logit of (premise, hypothesis) pairs. Thread-safe, torch-free."""

    def __init__(self, session, tokenizer, entailment_index: int = 0, max_len: int = 256):
        self.session = session
        self.tok = tokenizer
        self.entail = entailment_index
        self.max_len = max_len
        self._inputs = {i.name for i in session.get_inputs()}
        self.tok.enable_truncation(max_length=max_len)
        self.tok.enable_padding()
        from tokenizers import Tokenizer
        self._count_tok = Tokenizer.from_str(tokenizer.to_str())
        self._count_tok.no_padding()
        self._count_tok.no_truncation()

    @classmethod
    def from_dir(cls, path: str, file: str = NLI_FILE, threads: int | None = None) -> "OnnxNLI":
        import onnxruntime as ort
        from tokenizers import Tokenizer

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads or min(4, os.cpu_count() or 1)
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(os.path.join(path, "onnx", f"{file}.onnx"), opts,
                                       providers=["CPUExecutionProvider"])
        cfg = json.load(open(os.path.join(path, "config.json")))
        labels = {int(k): v.lower() for k, v in cfg["id2label"].items()}
        idx = next(i for i, name in labels.items() if name.startswith("entail"))
        return cls(session, Tokenizer.from_file(os.path.join(path, "tokenizer.json")), idx)

    @classmethod
    def from_hub(cls, repo: str = NLI_REPO, revision: str = NLI_REVISION, file: str = NLI_FILE,
                 **kw) -> "OnnxNLI":
        from huggingface_hub import hf_hub_download

        paths = [hf_hub_download(repo, f, revision=revision)
                 for f in (f"onnx/{file}.onnx", "tokenizer.json", "config.json")]
        return cls.from_dir(os.path.dirname(os.path.dirname(paths[0])), file=file, **kw)

    def count_tokens(self, text: str) -> int:
        return len(self._count_tok.encode(text).ids)

    def margin(self, pairs: list[tuple[str, str]], batch: int = 1) -> np.ndarray:
        """``logit(entailment) - logit(not entailment)`` for each (premise, hypothesis).

        Pairs are scored one at a time by default. With dynamic int8 quantization a
        pair's score depends on which pairs share its batch, and a rerank candidate's
        score must not depend on the other candidates.
        """
        out = []
        for i in range(0, len(pairs), batch):
            enc = self.tok.encode_batch(pairs[i:i + batch])
            feed = {"input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                    "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64)}
            if "token_type_ids" in self._inputs:
                feed["token_type_ids"] = np.array([e.type_ids for e in enc], dtype=np.int64)
            logits = self.session.run(None, feed)[0]
            other = [j for j in range(logits.shape[1]) if j != self.entail]
            out.append(logits[:, self.entail] - np.log(np.exp(logits[:, other]).sum(1)))
        return np.concatenate(out) if out else np.zeros(0)


class LazyNLI:
    """Loads the NLI model on first use, once, so a process that never gets a
    ``noul`` question never pays for it."""

    def __init__(self, loader=None):
        self._loader = loader or OnnxNLI.from_hub
        self._model: OnnxNLI | None = None
        self._lock = threading.Lock()
        self.error: str | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def get(self) -> OnnxNLI | None:
        if self._model is None and self.error is None:
            with self._lock:
                if self._model is None and self.error is None:
                    try:
                        self._model = self._loader()
                    except Exception as e:  # network, missing file, ...
                        self.error = f"{type(e).__name__}: {e}"
        return self._model


__all__ = ["OnnxNLI", "LazyNLI", "to_statement", "NLI_REPO", "NLI_REVISION", "NLI_FILE"]
