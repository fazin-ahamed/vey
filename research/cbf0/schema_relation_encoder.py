"""CBF-6 stock FP32 criterion/schema pairs and ternary relation heads."""
import hashlib
import json
from pathlib import Path

MODEL = 'microsoft/deberta-v3-xsmall'
REVISION = 'eb2d654bf0a5b628c8be6c4be7d29118fbef95b8'
WEIGHT_SHA256 = '964ceb3612da6cfdb45997d380fdb95f92c7499ffcabb50cbeea55e04756cafd'
WIDTH = 384
TOKEN_LIMIT = 128
SCHEMA = ('reliability', 'purchase expense', 'operating expense', 'convenience')


def load_encoder():
    """Return the fast tokenizer, frozen CUDA stock encoder, and JSON-safe lineage."""
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, use_fast=True)
    assert tokenizer.is_fast
    weight_path = Path(hf_hub_download(MODEL, 'model.safetensors', revision=REVISION))
    with weight_path.open('rb') as stream:
        weight_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert weight_hash == WEIGHT_SHA256, weight_hash
    model, loading = AutoModel.from_pretrained(
        MODEL, revision=REVISION, use_safetensors=True,
        dtype=torch.float32, output_loading_info=True,
    )
    assert not loading['missing_keys'] and not loading['mismatched_keys'], loading
    assert not loading.get('error_msgs'), loading
    assert model.config._commit_hash == REVISION, model.config._commit_hash
    assert model.config.hidden_size == WIDTH, model.config.hidden_size
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    lineage = dict(
        encoder=MODEL, revision=REVISION, stock_not_Vey2=True,
        weight_sha256=weight_hash, config=model.config.to_dict(),
        loading={key: sorted(value, key=repr) if isinstance(value, (list, tuple, set)) else value
                 for key, value in sorted(loading.items())},
        tokenizer_sha256=hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
        dtype='float32', width=WIDTH, token_limit=TOKEN_LIMIT,
        pair_format='[CLS] criterion [SEP] canonical_field [SEP]',
        feature='Raw final-layer CLS; no normalization',
        schema=list(SCHEMA), candidate_forwards=0,
    )
    lineage = json.loads(json.dumps(lineage, sort_keys=True))
    model.to('cuda').eval()
    model.requires_grad_(False)
    return tokenizer, model, lineage


def encode_pairs(pairs, tokenizer, model, batch_size=32):
    """Return raw CLS float32 [N,384] and evidence for ordered (criterion, field) pairs."""
    import numpy as np
    import torch

    assert isinstance(batch_size, int) and batch_size > 0
    assert tokenizer.is_fast
    pairs = list(pairs)
    for criterion, field in pairs:
        assert isinstance(criterion, str) and criterion
        assert field in SCHEMA, field
    features = np.empty((len(pairs), WIDTH), dtype=np.float32)
    evidence = dict(
        pairs=[dict(criterion=criterion, field=field) for criterion, field in pairs],
        batches=[], lengths=[], forwards=0, tokens=0, maximum_tokens=0,
        candidate_forwards=0, feature='Raw final-layer CLS; no normalization',
        truncation=False,
    )
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            encoded = tokenizer(
                [criterion for criterion, _ in batch],
                text_pair=[field for _, field in batch],
                add_special_tokens=True, padding=True, truncation=False,
                return_attention_mask=True, return_tensors='pt',
            )
            padded_length = int(encoded['input_ids'].shape[1])
            assert padded_length <= TOKEN_LIMIT, 'Never truncate a criterion/schema pair'
            lengths = encoded['attention_mask'].sum(dim=1).tolist()
            assert torch.all(encoded['input_ids'][:, 0] == tokenizer.cls_token_id)
            batch_evidence = dict(
                start=start, count=len(batch), lengths=lengths,
                input_ids=encoded['input_ids'].tolist(),
                attention_mask=encoded['attention_mask'].tolist(),
            )
            if 'token_type_ids' in encoded:
                batch_evidence['token_type_ids'] = encoded['token_type_ids'].tolist()
            inputs = {key: value.to(device) for key, value in encoded.items()}
            cls = model(**inputs).last_hidden_state[:, 0, :]
            assert cls.shape == (len(batch), WIDTH) and cls.dtype == torch.float32
            features[start:start + len(batch)] = cls.cpu().numpy()
            evidence['batches'].append(batch_evidence)
            evidence['lengths'].extend(lengths)
            evidence['forwards'] += 1
            evidence['tokens'] += sum(lengths)
            evidence['maximum_tokens'] = max(evidence['maximum_tokens'], padded_length)
    return features, evidence


def head(kind):
    """Build a linear or 64-unit GELU ternary head; preprocessing belongs to the caller."""
    import torch
    from torch import nn

    if kind == 'linear':
        return nn.Linear(WIDTH, 3, dtype=torch.float32)
    if kind == 'mlp':
        return nn.Sequential(
            nn.Linear(WIDTH, 64, dtype=torch.float32), nn.GELU(),
            nn.Linear(64, 3, dtype=torch.float32),
        )
    raise ValueError(f'Unknown relation head: {kind!r}')


def set_adaptation(model):
    """Enable gradients only for the last transformer layer, retaining eval mode."""
    model.requires_grad_(False)
    model.encoder.layer[-1].requires_grad_(True)
    model.eval()
