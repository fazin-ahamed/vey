"""CBF-7 relation-pretrained frozen pair interfaces (stock MLM and SNLI/MNLI NLI checkpoints).

Every model is loaded from the pinned revision with a byte hash check on
``model.safetensors``, fully frozen and in eval mode (including the native NLI
pooler and classifier).  Features are cached per (model, format, feature) so
each pair batch is encoded exactly once and shared across every head.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


PROTOCOL = Path(__file__).with_name('relation_pretrained_protocol.json')
SCHEMA = ('reliability', 'purchase expense', 'operating expense', 'convenience')
WIDTH = {'stock_xsmall': 384, 'nli_xsmall': 384, 'nli_small': 768}
TOKEN_LIMIT = 128
HYPOTHESIS = {field: f'Higher {field} is preferred.' for field in SCHEMA}
CLASS_ORDER = (-1, 0, 1)
# Native NLI mapping by name, never by numeric index: relation class +1 is
# entailment, -1 is contradiction, and the empirical neutral bridge is 0.
NATIVE_NAME_FOR_SIGN = {-1: 'contradiction', 0: 'neutral', 1: 'entailment'}


def _sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pair_second(format_name, field):
    """Return the tokenizer text_pair for one pair format and canonical field."""
    assert format_name in ('bare', 'higher'), format_name
    assert field in SCHEMA, field
    return field if format_name == 'bare' else HYPOTHESIS[field]


def load_model(model_key, device='cuda'):
    """Return the fast tokenizer, fully frozen eval model, and JSON-safe lineage.

    The stock MLM arm uses AutoModel; the pretrained NLI arms use
    AutoModelForSequenceClassification with trust_remote_code disabled and the
    classifier weights kept frozen so the native readout is reproducible.
    """
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

    cfg = json.loads(PROTOCOL.read_text())
    spec = cfg['models'][model_key]
    repo, revision, expected = spec['repo'], spec['revision'], spec['weight_sha256']
    weight_path = Path(hf_hub_download(repo, 'model.safetensors', revision=revision))
    actual = _sha256(weight_path)
    assert actual == expected, f'{model_key} weight hash {actual} != pinned {expected}'
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision, use_fast=True)
    assert tokenizer.is_fast
    if model_key == 'stock_xsmall':
        model, loading = AutoModel.from_pretrained(
            repo, revision=revision, use_safetensors=True, dtype=torch.float32,
            output_loading_info=True, trust_remote_code=False,
        )
    else:
        model, loading = AutoModelForSequenceClassification.from_pretrained(
            repo, revision=revision, use_safetensors=True, dtype=torch.float32,
            output_loading_info=True, trust_remote_code=False,
        )
    assert not loading['missing_keys'] and not loading['mismatched_keys'], loading
    assert not loading.get('error_msgs'), loading
    assert model.config._commit_hash == revision, model.config._commit_hash
    assert model.config.hidden_size == spec['width'], model.config.hidden_size
    assert model.config.hidden_size == WIDTH[model_key], model.config.hidden_size
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    floating = sum(parameter.numel() for parameter in model.parameters())
    if 'floating_parameters_metadata' in spec:
        assert floating == spec['floating_parameters_metadata'], floating
    label_map = {str(key): value for key, value in model.config.id2label.items()} if hasattr(model.config, 'id2label') else {}
    lineage = dict(
        model_key=model_key, encoder=repo, revision=revision, family=spec['family'],
        weight_sha256=actual, floating_parameters=floating, width=spec['width'],
        license_category=spec.get('eligibility', 'unspecified'),
        weight_license=spec.get('weight_license', 'unspecified'),
        config=model.config.to_dict(), label_map=label_map,
        loading={key: sorted(value, key=repr) if isinstance(value, (list, tuple, set)) else value
                 for key, value in sorted(loading.items())},
        tokenizer_sha256=hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
        dtype='float32', token_limit=TOKEN_LIMIT,
        pair_formats=dict(bare='[CLS] criterion [SEP] canonical_field [SEP]',
                          higher="[CLS] criterion [SEP] 'Higher '+canonical_field+' is preferred.' [SEP]"),
        schema=list(SCHEMA), candidate_forwards=0, frozen='all parameters eval, no adaptation',
    )
    if hasattr(model, 'pooler'):
        lineage['native_pooler'] = dict(
            hidden=model.config.pooler_hidden_size, act=model.config.pooler_hidden_act,
            dropout=model.config.pooler_dropout, frozen=True,
        )
    lineage = json.loads(json.dumps(lineage, sort_keys=True))
    model.to(device).eval()
    model.requires_grad_(False)
    return tokenizer, model, lineage


def native_head(model):
    """Return the frozen native (pooler, classifier) pair for an NLI model."""
    assert hasattr(model, 'pooler') and hasattr(model, 'classifier')
    return model.pooler, model.classifier


def native_class_indices(model, cfg=None):
    """Map relation classes (-1, 0, +1) to native classifier columns by label NAME."""
    if cfg is None:
        cfg = json.loads(PROTOCOL.read_text())
    id2label = {int(key): str(value) for key, value in model.config.id2label.items()}
    for name in NATIVE_NAME_FOR_SIGN.values():
        assert name in id2label.values(), f'native label map lacks {name!r}: {id2label}'
    mapping = {}
    for relation, expected in NATIVE_NAME_FOR_SIGN.items():
        matches = [index for index, name in id2label.items() if name == expected]
        assert len(matches) == 1, (expected, id2label)
        mapping[relation] = matches[0]
    assert set(mapping) == set(CLASS_ORDER), mapping
    names = cfg['native_mapping']
    assert names['lower'] == 'contradiction' and names['unrelated'] == 'neutral' and names['higher'] == 'entailment'
    return mapping


def reordered_native_weights(model, mapping):
    """Return native classifier weight/bias rows reordered to relation classes (-1,0,+1)."""
    import torch

    assert set(mapping) == set(CLASS_ORDER)
    weight = model.classifier.weight.detach()
    bias = model.classifier.bias.detach()
    rows = [mapping[relation] for relation in CLASS_ORDER]
    return weight[rows].clone(), bias[rows].clone()


def encode_pairs(pairs, tokenizer, model, format_name, batch_size=32, device='cuda'):
    """Return raw CLS features [N,width] and evidence for (criterion, field) pairs.

    ``format_name`` fixes the text_pair content: 'bare' passes the canonical
    field, 'higher' passes the fixed generic hypothesis.  No truncation ever
    occurs; a pair longer than the 128-token limit is a hard failure.
    """
    import torch

    assert isinstance(batch_size, int) and batch_size > 0
    assert tokenizer.is_fast
    pairs = list(pairs)
    for criterion, field in pairs:
        assert isinstance(criterion, str) and criterion
        assert field in SCHEMA, field
    width = model.config.hidden_size
    features = np.empty((len(pairs), width), dtype=np.float32)
    evidence = dict(
        pairs=[dict(criterion=criterion, field=field, hypothesis=pair_second(format_name, field))
               for criterion, field in pairs],
        batches=[], lengths=[], forwards=0, tokens=0, maximum_tokens=0,
        candidate_forwards=0, truncation=False, format=format_name,
        feature='Raw final-layer CLS; no normalization',
    )
    target = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            encoded = tokenizer(
                [criterion for criterion, _ in batch],
                text_pair=[pair_second(format_name, field) for _, field in batch],
                add_special_tokens=True, padding=True, truncation=False,
                return_attention_mask=True, return_tensors='pt',
            )
            padded = int(encoded['input_ids'].shape[1])
            assert padded <= TOKEN_LIMIT, 'Never truncate a criterion/schema pair'
            lengths = encoded['attention_mask'].sum(dim=1).tolist()
            assert torch.all(encoded['input_ids'][:, 0] == tokenizer.cls_token_id)
            record = dict(start=start, count=len(batch), lengths=lengths,
                         input_ids=encoded['input_ids'].tolist(),
                         attention_mask=encoded['attention_mask'].tolist())
            if 'token_type_ids' in encoded:
                record['token_type_ids'] = encoded['token_type_ids'].tolist()
            inputs = {key: value.to(target) for key, value in encoded.items()}
            cls = model.base_model(**inputs).last_hidden_state[:, 0, :]
            assert cls.shape == (len(batch), width) and cls.dtype == torch.float32
            features[start:start + len(batch)] = cls.cpu().numpy()
            evidence['batches'].append(record)
            evidence['lengths'].extend(lengths)
            evidence['forwards'] += 1
            evidence['tokens'] += sum(lengths)
            evidence['maximum_tokens'] = max(evidence['maximum_tokens'], padded)
    return features, evidence


def pooler_features(cls_features, model, device='cuda'):
    """Apply the frozen native ContextPooler to cached CLS features.

    The pooler is exactly ``gelu(pooler.dense(cls))`` in eval mode
    (pooler_dropout=0), so it replays on the cached CLS row without a second
    encoder forward.  Assertions pin that structural identity.
    """
    import torch

    assert model.config.pooler_dropout == 0, 'pooler dropout must be identity in eval'
    assert model.config.pooler_hidden_act == 'gelu'
    assert model.config.pooler_hidden_size == model.config.hidden_size
    dense = model.pooler.dense
    with torch.inference_mode():
        x = torch.tensor(cls_features, device=device)
        pooled = torch.nn.functional.gelu(dense(x)).cpu().numpy()
    return pooled.astype(np.float32)


def tokenizer_parity(texts, tokenizer, format_name, model_key):
    """Compare fast input IDs against the pinned native SentencePiece model."""
    import sentencepiece
    from huggingface_hub import hf_hub_download

    spec = json.loads(PROTOCOL.read_text())['models'][model_key]
    path = hf_hub_download(spec['repo'], 'spm.model', revision=spec['revision'])
    native = sentencepiece.SentencePieceProcessor(model_file=path)
    pairs = [(criterion, field) for criterion in texts for field in SCHEMA]
    fast_ids = tokenizer([q for q, _ in pairs],
                         text_pair=[pair_second(format_name, f) for _, f in pairs])['input_ids']
    native_ids = [[tokenizer.cls_token_id] + native.encode(q) + [tokenizer.sep_token_id] +
                  native.encode(pair_second(format_name, f)) + [tokenizer.sep_token_id]
                  for q, f in pairs]
    assert fast_ids == native_ids, 'fast/native SentencePiece input-ID divergence'
    return dict(format=format_name, pairs=len(pairs), parity=True,
                spm_sha256=_sha256(path), comparator='Pinned native SentencePiece model')


def full_forward_parity(pairs, tokenizer, model, format_name, device='cuda'):
    """Verify full native forward equals pooler/classifier replay on a tiny batch.

    This is the anti-blind-fix guard for the direct-native-head readout: the
    native model's own logits must be reproduced exactly by applying the
    frozen pooler to the encoder final state and the classifier to the pooled
    row.  The tiny batch uses the first two live pairs of this arm.
    """
    import torch

    pooler, classifier = native_head(model)
    mapping = native_class_indices(model)
    checked = 0
    with torch.inference_mode():
        for criterion, field in pairs[:2]:
            encoded = tokenizer(criterion, text_pair=pair_second(format_name, field),
                                return_tensors='pt')
            inputs = {key: value.to(device) for key, value in encoded.items()}
            reference = model(**inputs).logits
            hidden = model.deberta(**inputs).last_hidden_state
            pooled = pooler(hidden)
            replay = classifier(pooled)
            assert torch.equal(replay, reference), 'native classifier replay diverged'
            checked += 1
    assert checked > 0, 'no pairs available for parity check'
    return dict(
        pairs=len(pairs), checked=checked, exact=True,
        details='model(**inputs).logits == classifier(pooler(encoder(...).last_hidden_state)) bitwise',
        class_index_map={str(relation): index for relation, index in mapping.items()},
    )
