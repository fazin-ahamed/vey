"""Antisymmetric ordinal comparator — the CRUX ranking primitive.

For an axis and two option texts A, B:

    f(axis, A, B) = head(CLS_encode("On {axis}: A = {A} | B = {B}"))
    r(A, B) = 0.5 * (f(axis, A, B) - f(axis, B, A))          # antisymmetric by construction

For an option set, the ordinal value of option i is the scalar potential

    q_i = mean_{j != i} r(i, j)

which is transitive and PERMUTATION-INVARIANT: q_i is a symmetric aggregation
over the other options, so reindexing the options reindexes q identically.

Vendored, self-contained. The trained head + fine-tuned encoder are loaded from a
published artifact (see grounding.SageGrounder); this module only defines the
architecture and the permutation-exact scoring.
"""
from __future__ import annotations

import torch
from torch import nn


def seq_ab(axis: str, a: str, b: str) -> str:
    return f"On {axis}: A = {a} | B = {b}"


class OrdinalHead(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Dropout(0.1), nn.Linear(hidden, 1))

    def forward(self, feats):
        return self.net(feats).squeeze(-1)


class OrdinalComparator:
    """Encoder (frozen at inference) + trained OrdinalHead."""

    def __init__(self, model_id: str, device: str = "cpu"):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.enc = AutoModel.from_pretrained(model_id).to(device).eval()
        self.device = device
        self.hidden = self.enc.config.hidden_size
        self.head = OrdinalHead(self.hidden).to(device)
        for p in self.enc.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def _cls(self, texts, bs=96):
        outs = []
        for i in range(0, len(texts), bs):
            enc = self.tok(texts[i:i + bs], padding=True, truncation=True,
                           max_length=48, return_tensors="pt").to(self.device)
            outs.append(self.enc(**enc).last_hidden_state[:, 0].detach().cpu())
        return torch.cat(outs, 0)

    @torch.no_grad()
    def ordinal_values(self, options, axis: str):
        """Permutation-invariant scalar potential per option (higher = more)."""
        K = len(options)
        seqs, idx = [], []
        for i in range(K):
            for j in range(K):
                if i == j:
                    continue
                seqs.append(seq_ab(axis, options[i], options[j])); idx.append((i, j))
        if not seqs:
            return torch.zeros(K)
        fvals = self.head(self._cls(seqs).to(self.device))
        r = torch.zeros(K, K, device=self.device)
        for (i, j), v in zip(idx, fvals):
            r[i, j] = v
        rr = 0.5 * (r - r.t())
        return (rr.sum(1) / max(1, K - 1)).detach().cpu()

    def load_state(self, enc_state, head_state):
        self.enc.load_state_dict(enc_state)
        self.head.load_state_dict(head_state)
        self.enc.eval(); self.head.eval()
