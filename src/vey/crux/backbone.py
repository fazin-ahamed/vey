"""Frozen NLI backbone: the CRUX predicate grounder.

A frozen natural-language-inference model answers narrow typed questions by the
sign of an entailment-minus-contradiction margin. It never produces a final
action utility. Vendored, self-contained: no research-tree or private-path
dependencies. Honors the standard Hugging Face cache environment
(``HF_HOME`` / ``HF_HUB_CACHE``); models download from the public Hub on first
use.
"""
from __future__ import annotations

import torch

HIGH_TMPL = ["This option has high {a}.", "This option exhibits strong {a}.",
             "The option delivers substantial {a}."]
LOW_TMPL = ["This option has low {a}.", "This option exhibits weak {a}.",
            "The option delivers minimal {a}."]


class NLIGrounder:
    """Frozen sequence-classification NLI model. Provides categorical predicate
    grounding (bipolar margins) and an evidence-based support state."""

    def __init__(self, model_id: str, device: str = "cpu", batch_size: int = 128):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device).eval()
        self.device = device
        self.bs = batch_size
        self.model_id = model_id
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.ent = next(i for i, l in id2label.items() if "entail" in l)
        self.con = next((i for i, l in id2label.items() if "contra" in l), None)

    @torch.no_grad()
    def _logits(self, premises, hyps):
        outs = []
        for i in range(0, len(premises), self.bs):
            enc = self.tok(premises[i:i + self.bs], hyps[i:i + self.bs], padding=True,
                           truncation=True, max_length=96, return_tensors="pt").to(self.device)
            outs.append(self.model(**enc).logits.float().cpu())
        return torch.cat(outs, 0)

    def margin(self, premises, hyps):
        """logit(entail) - logit(contradict); binary heads use 2*logit(entail)."""
        lg = self._logits(premises, hyps)
        if self.con is not None:
            return lg[:, self.ent] - lg[:, self.con]
        return 2.0 * lg[:, self.ent]

    def entail_prob(self, premises, hyps):
        return torch.softmax(self._logits(premises, hyps), dim=-1)[:, self.ent]

    def predicate(self, options, prop_pos: str, prop_neg: str):
        """Bipolar margin per option for a categorical predicate. >0 satisfies."""
        prem, hyp = [], []
        for o in options:
            prem.append(o); hyp.append(prop_pos)
            prem.append(o); hyp.append(prop_neg)
        m = self.margin(prem, hyp).view(len(options), 2)
        return 0.5 * (m[:, 0] - m[:, 1])

    def bipolar(self, options, axis: str):
        pos = [t.format(a=axis) for t in HIGH_TMPL]
        neg = [t.format(a=axis) for t in LOW_TMPL]
        prem, hyp = [], []
        for o in options:
            for hp in pos:
                prem.append(o); hyp.append(hp)
            for hn in neg:
                prem.append(o); hyp.append(hn)
        m = self.margin(prem, hyp).view(len(options), len(pos) + len(neg))
        return 0.5 * (m[:, :len(pos)].median(dim=1).values - m[:, len(pos):].median(dim=1).values)

    def support_state(self, options, axis: str, *, evidence_tau: float = 0.30, sign_tau: float = 0.5):
        """Per-option SUPPORTED / CONTRADICTED / UNKNOWN / AMBIGUOUS on an axis.
        UNKNOWN = the axis has no evidence in the option text (used for abstention)."""
        pos = [t.format(a=axis) for t in HIGH_TMPL]
        neg = [t.format(a=axis) for t in LOW_TMPL]
        prem = [o for o in options for _ in range(len(pos) + len(neg))]
        hyp = [h for _ in options for h in (pos + neg)]
        ep = self.entail_prob(prem, hyp).view(len(options), len(pos) + len(neg))
        evidence = ep.max(dim=1).values
        b = self.bipolar(options, axis)
        states = []
        for i in range(len(options)):
            if float(evidence[i]) < evidence_tau:
                states.append("UNKNOWN")
            elif float(b[i]) >= sign_tau:
                states.append("SUPPORTED")
            elif float(b[i]) <= -sign_tau:
                states.append("CONTRADICTED")
            else:
                states.append("AMBIGUOUS")
        return states, b
