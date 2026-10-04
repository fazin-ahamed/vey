"""Capture CBF-8 frozen NLI CLS features and a real tiny-pair execution preflight."""
import os

for _name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import argparse
import hashlib
import io
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from audit import put
from build import canonical

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / 'schema_support_protocol.json'
TOKEN_LIMIT = 128
BATCH_SIZE = 32
FORMATS = ('generic', 'defined')


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def _json_bytes(value):
    return canonical(value) + b'\n'


def _protocol(path=PROTOCOL):
    cfg = load(path)
    assert cfg['model']['repo'] == 'cross-encoder/nli-deberta-v3-xsmall'
    assert cfg['model']['revision'] == 'a150876415327c80daeff35ca6f68f5ed8cf5c24'
    assert cfg['model']['weight_sha256'] == '4e4fc4977f8d29d2a164255c8f69b9d6c158deeb309bb5e70445b94666ccd9e9'
    assert cfg['model']['floating_parameters'] == 70831107 and cfg['model']['width'] == 384
    assert cfg['resource_budget']['max_pair_tokens'] == TOKEN_LIMIT
    assert cfg['resource_budget']['pair_batch'] == BATCH_SIZE
    assert cfg['resource_budget']['threads'] == 4 and cfg['resource_budget']['BLAS'] == 1
    assert cfg['resource_budget']['paid'] is False
    return cfg

PREPARATION_SOURCES = {
    'schema_support_prepare.py', 'schema_support_encoder.py', 'schema_support_run.py',
    'schema_support_training_corpus.py', 'schema_support_final_corpus.py',
    'schema_support_protocol.json', 'schema_grounding_compiler.py', 'schema_relation_evaluate.py',
    'relation_pretrained_run.py', 'relation_pretrained_encoder.py', 'audit.py', 'build.py',
}


def verify_preparation_sources(manifest):
    sources = manifest['preparation_source_sha256']
    assert set(sources) == PREPARATION_SOURCES, 'Prepared manifest source list differs from the frozen source set.'
    for name, expected in sources.items():
        assert sha(HERE / name) == expected, f'Prepared code source changed after corpus freeze: {name}'



def _root_and_manifest(cfg):
    root = Path(cfg['output_root'])
    manifest_path = root / 'corpus_manifest.json'
    manifest = load(manifest_path)
    assert manifest['experiment'] == 'CBF8'
    assert manifest['protocol_sha256'] == sha(PROTOCOL)
    assert manifest['source_root'] == cfg['source_root']
    verify_preparation_sources(manifest)
    for name, expected in manifest['files'].items():
        assert sha(root / name) == expected, f'Prepared corpus hash mismatch: {name}'
    assert not (root / 'selection.json').exists() and not (root / 'results.json').exists()
    return root, manifest


def require_meaning_audit(root, manifest):
    audit_path = root / 'meaning_audit.json'
    review = manifest['blind_review']
    predictions_path = root / review['predictions_file']
    assert audit_path.is_file(), 'Independent meaning_audit.json is required before model execution.'
    assert predictions_path.is_file(), 'Independent blind predictions are required before model execution.'
    files = manifest['files']
    question_path = root / review['question_file']
    ids_path = root / review['review_ids_file']
    key_path = root / review['key_file']
    expected_count = review['expected_questions']
    assert review['question_file'] == 'blind_meaning_questions.json'
    assert review['review_ids_file'] == 'blind_meaning_review_ids.json'
    assert review['key_file'] == 'blind_meaning_key.json'
    assert review['predictions_file'] == 'blind_meaning_predictions.json'
    assert review['schema_version'] == 'cbf8-blind-meaning-review-v1'
    assert expected_count == manifest['counts']['blind_review_rows'] == 352
    assert sha(question_path) == review['packet_sha256'] == files[review['question_file']]
    assert sha(ids_path) == review['review_ids_sha256'] == files[review['review_ids_file']]
    assert sha(key_path) == review['key_sha256'] == files[review['key_file']]
    expected_ids = load(ids_path)
    assert len(expected_ids) == len(set(expected_ids)) == expected_count
    predictions = load(predictions_path)
    assert set(predictions) == {'schema_version', 'packet_sha256', 'judgments'}
    assert predictions['schema_version'] == review['schema_version']
    assert predictions['packet_sha256'] == review['packet_sha256']
    judgments = predictions['judgments']
    assert isinstance(judgments, list) and len(judgments) == expected_count
    actual_ids = [row['review_id'] for row in judgments]
    assert len(set(actual_ids)) == expected_count and set(actual_ids) == set(expected_ids)

    for row in judgments:
        assert row['decision'] == 'accept'
        if row['kind'] == 'atomic':
            assert set(row) == {'review_id', 'decision', 'kind', 'axis', 'sign'}
            assert type(row['axis']) is int and 0 <= row['axis'] < 4
            assert type(row['sign']) is int and row['sign'] in (-1, 1)
        else:
            assert row['kind'] == 'composition'
            assert set(row) == {'review_id', 'decision', 'kind', 'components', 'weights'}
            assert len(row['components']) == 2 and len(row['weights']) == 4
            assert all(set(term) == {'axis', 'sign'} and type(term['axis']) is int and
                       0 <= term['axis'] < 4 and type(term['sign']) is int and term['sign'] in (-1, 1)
                       for term in row['components'])
            assert all(type(weight) is int for weight in row['weights'])

    audit = load(audit_path)
    assert audit['all_meanings_accepted'] is True
    assert audit['final_atoms_sha256'] == files['final_atoms.json']
    assert audit['final_compositions_sha256'] == files['final_compositions.json']
    assert audit['blind_predictions_sha256'] == sha(predictions_path)
    return audit, sha(audit_path)


def _device_helpers():
    import torch
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        assert torch.get_num_interop_threads() == 1
    from relation_pretrained_encoder import load_model
    from relation_pretrained_run import gpu_serialization, tensor_hash, guard_resources
    return torch, load_model, gpu_serialization, tensor_hash, guard_resources


def _format_pair(cfg, fmt, field):
    assert fmt in FORMATS and field in cfg['schema']
    generic = f'Higher {field} is preferred.'
    if fmt == 'generic':
        return generic
    return generic + ' Definition: ' + cfg['field_definitions'][field]


def _development_texts(root):
    from schema_relation_evaluate import semantic_contexts
    names = ('training_atoms.json', 'validation_atoms.json', 'semantic_training_atoms.json',
             'semantic_validation_atoms.json', 'literal_holdout_atoms.json')
    texts = {row['text'] for name in names for row in load(root / name)}
    development = load(root / 'development_atoms.json')
    compositions = load(root / 'development_compositions.json')
    g0 = load(root / 'literal_holdout_atoms.json')
    texts.update(semantic_contexts(development, compositions))
    texts.update(row['text'] for row in g0)
    return sorted(texts)


def _rows(texts, schema, fmt, cfg):
    return [dict(criterion=text, field=field, text_pair=_format_pair(cfg, fmt, field))
            for text in texts for field in schema]


def _tokenize_rows(rows, tokenizer, fmt):
    batches = []
    row_evidence = []
    all_lengths = []
    for start in range(0, len(rows), BATCH_SIZE):
        chunk = rows[start:start + BATCH_SIZE]
        encoded = tokenizer(
            [row['criterion'] for row in chunk],
            text_pair=[row['text_pair'] for row in chunk],
            add_special_tokens=True, padding=False, truncation=False,
            return_attention_mask=True, return_token_type_ids=True,
        )
        if 'token_type_ids' not in encoded:
            raise RuntimeError('Pinned fast tokenizer did not return token_type_ids; refusing incomplete pair evidence.')
        ids_rows = encoded['input_ids']
        type_rows = encoded['token_type_ids']
        assert len(ids_rows) == len(type_rows) == len(chunk)
        padded_width = max(map(len, ids_rows))
        assert padded_width <= TOKEN_LIMIT, 'Never truncate a criterion/schema pair.'
        pad_id = tokenizer.pad_token_id
        assert pad_id is not None
        pad_type = int(getattr(tokenizer, 'pad_token_type_id', 0) or 0)
        padded_ids, padded_types, masks = [], [], []
        lengths = []
        for row, input_ids, token_types in zip(chunk, ids_rows, type_rows):
            assert len(input_ids) == len(token_types)
            length = len(input_ids)
            assert length > 0 and input_ids[0] == tokenizer.cls_token_id
            lengths.append(length)
            all_lengths.append(length)
            padded_ids.append(list(input_ids) + [int(pad_id)] * (padded_width - length))
            padded_types.append(list(token_types) + [pad_type] * (padded_width - length))
            masks.append([1] * length + [0] * (padded_width - length))
            row_evidence.append(dict(
                criterion=row['criterion'], field=row['field'], text_pair=row['text_pair'],
                input_ids=list(map(int, input_ids)), token_type_ids=list(map(int, token_types)),
                attention_mask=[1] * length,
            ))
        batches.append(dict(
            start=start, count=len(chunk), lengths=lengths,
            input_ids=padded_ids, token_type_ids=padded_types, attention_mask=masks,
            padded_width=padded_width,
        ))
    assert len(row_evidence) == len(rows)
    return dict(
        format=fmt, feature='Raw final-layer CLS from frozen base_model; no normalization',
        pairs=row_evidence, batches=batches, pair_count=len(rows),
        candidate_forwards=0, truncation=False, padding_bytes_explicit=True,
        max_tokens=TOKEN_LIMIT, maximum_sequence_length=max(all_lengths),
        tokens=sum(all_lengths), tokenizer_padding='Right padding recorded in every batch input array; per-row inputs are unpadded.',
    )


def _native_sentencepiece_parity(rows, tokenizer, fmt, cfg):
    import sentencepiece
    from huggingface_hub import hf_hub_download

    spec = cfg['model']
    spm_path = Path(hf_hub_download(spec['repo'], 'spm.model', revision=spec['revision']))
    native = sentencepiece.SentencePieceProcessor(model_file=str(spm_path))
    max_error = 0
    delimiter_records = []
    for row in rows:
        first = native.encode(row['criterion'], out_type=int)
        second = native.encode(row['text_pair'], out_type=int)
        expected = tokenizer.build_inputs_with_special_tokens(first, second)
        encoded = tokenizer(row['criterion'], text_pair=row['text_pair'], add_special_tokens=True,
                            padding=False, truncation=False, return_attention_mask=True,
                            return_token_type_ids=True)
        assert encoded['input_ids'] == expected, f'Native SentencePiece pair-ID mismatch: {row["criterion"]!r}/{row["field"]}'
        assert encoded['attention_mask'] == [1] * len(expected)
        type_ids = tokenizer.create_token_type_ids_from_sequences(first, second)
        assert encoded['token_type_ids'] == type_ids, 'Native SentencePiece segment-delimiter/type-ID mismatch'
        max_error = max(max_error, abs(len(expected) - len(encoded['input_ids'])))
        delimiter_records.append(dict(
            criterion=row['criterion'], field=row['field'], input_ids=expected,
            token_type_ids=type_ids,
            special_token_positions=[i for i, token in enumerate(expected)
                                     if token in (tokenizer.cls_token_id, tokenizer.sep_token_id)],
        ))
    return dict(
        format=fmt, pairs=len(rows), exact=True,
        spm_model_path=str(spm_path), spm_sha256=sha(spm_path),
        pair_id_rows_sha256=sha_bytes(canonical(delimiter_records)),
        pair_ids=delimiter_records,
        delimiter_comparator='Pinned SentencePiece ids passed through tokenizer.build_inputs_with_special_tokens and tokenizer.create_token_type_ids_from_sequences.',
        checked_rows=len(delimiter_records), maximum_length_delta=max_error,
    )


def _batch_to_torch(batch, device):
    import torch
    return {
        'input_ids': torch.tensor(batch['input_ids'], dtype=torch.long, device=device),
        'token_type_ids': torch.tensor(batch['token_type_ids'], dtype=torch.long, device=device),
        'attention_mask': torch.tensor(batch['attention_mask'], dtype=torch.long, device=device),
    }




def _atomic_progress(path, value):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_bytes(_json_bytes(value))
    os.replace(temporary, path)


def _cache_manifest(fmt, cache_key, feature_name, input_name, input_hash, feature_path,
                    expected_shape, model_hash, cfg):
    return dict(format=fmt, cache_key=cache_key, feature=feature_name, input=input_name,
                input_sha256=input_hash, feature_sha256=sha(feature_path),
                model_repo=cfg['model']['repo'], model_revision=cfg['model']['revision'],
                model_weight_sha256=cfg['model']['weight_sha256'],
                encoder_parameters_sha256=model_hash, shape=list(expected_shape), dtype='float32',
                raw_final_layer_cls=True, base_model_last_hidden_state=True,
                classifier_invoked=False, pooler_invoked=False, truncation=False,
                all_tokenizer_ids_saved=True, padding_bytes_saved=True)


def _verify_progress(state, evidence, input_hash, model_hash, cfg, features, expected_shape):
    assert state['input_sha256'] == input_hash
    assert state['model_weight_sha256'] == cfg['model']['weight_sha256']
    assert state['encoder_parameters_sha256'] == model_hash
    assert tuple(state['shape']) == expected_shape and state['dtype'] == 'float32'
    completed = 0
    for batch_index, record in enumerate(state['completed_batches']):
        assert batch_index < len(evidence['batches'])
        batch = evidence['batches'][batch_index]
        assert record['start'] == completed and record['start'] == batch['start']
        assert record['stop'] == batch['start'] + batch['count']
        assert record['input_batch_sha256'] == sha_bytes(canonical(batch))
        assert record['feature_sha256'] == sha_bytes(
            np.ascontiguousarray(features[record['start']:record['stop']]).tobytes())
        completed = record['stop']
    assert state['completed_rows'] == completed
    return completed

def _cache_one(root, fmt, evidence, evidence_payload, model, model_hash, cfg, cache_key=None):
    cache_key = cache_key or fmt
    input_name = f'inputs_{cache_key}.json'
    feature_name = f'features_{cache_key}.npy'
    cache_name = f'feature_cache_{cache_key}.json'
    input_path, feature_path, cache_path = root / input_name, root / feature_name, root / cache_name
    if input_path.exists():
        assert input_path.read_bytes() == evidence_payload, f'Existing {input_name} is not byte/input compatible.'
    else:
        put(root, input_name, evidence_payload)
    input_hash = sha(input_path)
    expected_shape = (evidence['pair_count'], cfg['model']['width'])

    partial = root / f'.{feature_name}.partial'
    progress = root / f'.{feature_name}.progress.json'
    if feature_path.exists():
        features = np.load(feature_path, mmap_mode='r', allow_pickle=False)
        assert features.shape == expected_shape and features.dtype == np.float32
        if cache_path.exists():
            cache = load(cache_path)
            assert cache['cache_key'] == cache_key and cache['format'] == fmt
            assert cache['feature'] == feature_name and cache['input'] == input_name
            assert cache['input_sha256'] == input_hash and cache['shape'] == list(expected_shape)
            assert cache['dtype'] == 'float32' and cache['model_repo'] == cfg['model']['repo']
            assert cache['model_revision'] == cfg['model']['revision']
            assert cache['model_weight_sha256'] == cfg['model']['weight_sha256']
            assert cache['encoder_parameters_sha256'] == model_hash
            assert cache['feature_sha256'] == sha(feature_path)
            assert cache['raw_final_layer_cls'] and cache['base_model_last_hidden_state']
            assert cache['classifier_invoked'] is False and cache['pooler_invoked'] is False
            assert cache['truncation'] is False and cache['all_tokenizer_ids_saved']
        else:
            assert progress.exists(), 'Completed cache has no compatible progress or cache manifest.'
            state = load(progress)
            completed = _verify_progress(state, evidence, input_hash, model_hash, cfg, features, expected_shape)
            assert completed == expected_shape[0] and len(state['completed_batches']) == len(evidence['batches'])
            cache = _cache_manifest(fmt, cache_key, feature_name, input_name, input_hash,
                                    feature_path, expected_shape, model_hash, cfg)
            put(root, cache_name, _json_bytes(cache))
        if progress.exists():
            progress.unlink()
        return features, cache
    assert not cache_path.exists(), f'Cache manifest exists without its feature file: {cache_name}'

    if progress.exists():
        state = load(progress)
        if partial.exists():
            features = np.load(partial, mmap_mode='r+', allow_pickle=False)
            assert features.shape == expected_shape and features.dtype == np.float32
        else:
            features = np.lib.format.open_memmap(partial, mode='w+', dtype=np.float32, shape=expected_shape)
        completed = _verify_progress(state, evidence, input_hash, model_hash, cfg, features, expected_shape)
    else:
        assert not partial.exists(), 'Partial feature file has no compatible resume ledger.'
        state = dict(input_sha256=input_hash, model_weight_sha256=cfg['model']['weight_sha256'],
                     encoder_parameters_sha256=model_hash, shape=list(expected_shape), dtype='float32',
                     completed_rows=0, completed_batches=[])
        _atomic_progress(progress, state)
        features = np.lib.format.open_memmap(partial, mode='w+', dtype=np.float32, shape=expected_shape)
        completed = 0

    device = next(model.parameters()).device
    for batch_index, batch in enumerate(evidence['batches']):
        start, stop = batch['start'], batch['start'] + batch['count']
        if stop <= completed:
            continue
        assert start == completed, 'Resume ledger is not a contiguous prefix.'
        inputs = _batch_to_torch(batch, device)
        with __import__('torch').inference_mode():
            hidden = model.base_model(**inputs).last_hidden_state
            cls = hidden[:, 0, :]
        assert tuple(cls.shape) == (batch['count'], cfg['model']['width'])
        assert cls.dtype == __import__('torch').float32
        values = cls.detach().cpu().numpy().astype(np.float32, copy=False)
        features[start:stop] = values
        features.flush()
        state['completed_batches'].append(dict(
            start=start, stop=stop, input_batch_sha256=sha_bytes(canonical(batch)),
            feature_sha256=sha_bytes(np.ascontiguousarray(values).tobytes()),
        ))
        state['completed_rows'] = stop
        _atomic_progress(progress, state)
        completed = stop
    assert completed == expected_shape[0]
    features.flush()
    del features
    os.replace(partial, feature_path)
    cache = _cache_manifest(fmt, cache_key, feature_name, input_name, input_hash,
                            feature_path, expected_shape, model_hash, cfg)
    put(root, cache_name, _json_bytes(cache))
    progress.unlink()
    features = np.load(feature_path, mmap_mode='r', allow_pickle=False)
    return features, cache


def _preflight(root, cfg, meaning_sha, tokenizer, model, lineage, model_hash, tensor_hash):
    import torch
    literal = load(root / 'training_atoms.json')[0]
    field = cfg['schema'][literal['axis']]
    rows = [dict(criterion=literal['text'], field=field, text_pair=_format_pair(cfg, fmt, field))
            for fmt in FORMATS]
    evidence = {fmt: _tokenize_rows([row], tokenizer, fmt) for fmt, row in zip(FORMATS, rows)}
    parity = {fmt: _native_sentencepiece_parity([row], tokenizer, fmt, cfg) for fmt, row in zip(FORMATS, rows)}
    features = {}
    device = next(model.parameters()).device
    for fmt in FORMATS:
        inputs = _batch_to_torch(evidence[fmt]['batches'][0], device)
        with torch.inference_mode():
            cls = model.base_model(**inputs).last_hidden_state[:, 0, :]
        assert cls.shape == (1, cfg['model']['width']) and cls.dtype == torch.float32
        features[fmt] = cls.detach().cpu().numpy().astype(np.float32, copy=True)
    assert all(parameter.requires_grad is False and parameter.grad is None for parameter in model.parameters())
    torch.manual_seed(7)
    head = torch.nn.Linear(cfg['model']['width'], 3, dtype=torch.float32)
    target = torch.tensor([literal['sign'] + 1], dtype=torch.long)
    logits = head(torch.tensor(features['generic'], dtype=torch.float32))
    loss = torch.nn.functional.cross_entropy(logits, target)
    loss.backward()
    gradient_l1 = float(sum(parameter.grad.abs().sum().item() for parameter in head.parameters()))
    assert gradient_l1 > 0 and all(parameter.grad is None for parameter in model.parameters())
    before_after_hash = tensor_hash(model.named_parameters())
    assert before_after_hash == model_hash
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    attempt_id = f'{stamp}_{uuid.uuid4().hex}'
    inputs_name = f'preflight_inputs_{attempt_id}.json'
    features_name = f'preflight_features_{attempt_id}.npy'
    attempt_name = f'preflight_attempt_{attempt_id}.json'
    evidence_value = dict(formats={fmt: evidence[fmt] for fmt in FORMATS},
                          tokenizer_parity={fmt: parity[fmt] for fmt in FORMATS})
    input_payload = _json_bytes(evidence_value)
    feature_array = np.stack((features['generic'][0], features['defined'][0])).astype(np.float32, copy=False)
    buffer = io.BytesIO()
    np.save(buffer, feature_array, allow_pickle=False)
    features_payload = buffer.getvalue()
    put(root, inputs_name, input_payload)
    put(root, features_name, features_payload)
    cached = np.load(root / features_name, mmap_mode='r', allow_pickle=False)
    assert cached.shape == (2, cfg['model']['width']) and cached.dtype == np.float32
    assert np.array_equal(cached, feature_array)
    attempt = dict(
        preflight=True, passed=True, attempt_id=attempt_id,
        protocol_sha256=sha(PROTOCOL), corpus_manifest_sha256=sha(root / 'corpus_manifest.json'),
        meaning_audit_sha256=meaning_sha, source='one literal training criterion and one canonical field only',
        formats=list(FORMATS), tokenizer_parity_exact=True,
        tokenizer_evidence_file=inputs_name, tokenizer_evidence_sha256=sha(root / inputs_name),
        feature_evidence_file=features_name, feature_evidence_sha256=sha(root / features_name),
        feature_mmap_verified=True, feature_shape=list(cached.shape), feature_dtype=str(cached.dtype),
        model_revision=lineage['revision'], model_weight_sha256=lineage['weight_sha256'],
        floating_parameters=lineage['floating_parameters'], width=lineage['width'],
        encoder_parameters_before_sha256=model_hash, encoder_parameters_after_sha256=before_after_hash,
        encoder_parameters_frozen=True, encoder_parameter_gradient_count=0,
        actual_base_model_forward=True, classifier_invoked=False, pooler_invoked=False,
        head_gradient_L1=gradient_l1, head_only_gradient=True,
        final_inputs_captured=False, alias_inputs_captured=False,
    )
    put(root, attempt_name, _json_bytes(attempt))
    return attempt


def _load_runtime_model(cfg, load_model):
    tokenizer, model, lineage = load_model('nli_xsmall', device='cuda')
    assert lineage['revision'] == cfg['model']['revision']
    assert lineage['weight_sha256'] == cfg['model']['weight_sha256']
    assert lineage['floating_parameters'] == cfg['model']['floating_parameters'] == 70831107
    assert lineage['width'] == cfg['model']['width'] == 384
    assert model.training is False and all(parameter.requires_grad is False for parameter in model.parameters())
    return tokenizer, model, lineage


def preflight(protocol_path=PROTOCOL):
    cfg = _protocol(protocol_path)
    root, manifest = _root_and_manifest(cfg)
    _, meaning_sha = require_meaning_audit(root, manifest)
    torch, load_model, gpu_serialization, tensor_hash, guard_resources = _device_helpers()
    attempt = None
    tokenizer = model = None
    with gpu_serialization():
        guard_resources()
        try:
            tokenizer, model, lineage = _load_runtime_model(cfg, load_model)
            before = tensor_hash(model.named_parameters())
            attempt = _preflight(root, cfg, meaning_sha, tokenizer, model, lineage, before, tensor_hash)
            after = tensor_hash(model.named_parameters())
            assert after == before
        finally:
            if model is not None:
                model.cpu()
            model = tokenizer = None
            torch.cuda.empty_cache()
    return attempt


def capture_development(protocol_path=PROTOCOL):
    cfg = _protocol(protocol_path)
    root, manifest = _root_and_manifest(cfg)
    _, meaning_sha = require_meaning_audit(root, manifest)
    verify_preflight(root, manifest, meaning_sha, cfg)
    texts = _development_texts(root)
    schema = tuple(cfg['schema'])
    lineage_path = root / 'encoder_lineage.json'
    torch, load_model, gpu_serialization, tensor_hash, guard_resources = _device_helpers()
    lineage_value = None
    caches, parity_records, evidence_hashes = {}, {}, {}
    tokenizer = model = None
    with gpu_serialization():
        guard_resources()
        try:
            tokenizer, model, lineage = _load_runtime_model(cfg, load_model)
            before_hash = tensor_hash(model.named_parameters())
            assert lineage['weight_sha256'] == cfg['model']['weight_sha256']
            for fmt in FORMATS:
                rows = _rows(texts, schema, fmt, cfg)
                evidence = _tokenize_rows(rows, tokenizer, fmt)
                parity = _native_sentencepiece_parity(rows, tokenizer, fmt, cfg)
                parity_records[fmt] = parity
                parity_name = f'native_sentencepiece_parity_{fmt}.json'
                parity_payload = _json_bytes(parity)
                if (root / parity_name).exists():
                    assert (root / parity_name).read_bytes() == parity_payload
                else:
                    put(root, parity_name, parity_payload)
                features, cache = _cache_one(root, fmt, evidence, _json_bytes(evidence), model, before_hash, cfg)
                caches[fmt] = cache
                evidence_hashes[fmt] = cache['input_sha256']
                assert features.shape == (len(rows), cfg['model']['width'])
            after_hash = tensor_hash(model.named_parameters())
            assert after_hash == before_hash, 'Frozen NLI parameters changed during feature capture.'
            lineage_value = dict(
                experiment='CBF8', model_repo=lineage['encoder'], revision=lineage['revision'],
                weight_sha256=lineage['weight_sha256'], floating_parameters=lineage['floating_parameters'],
                width=lineage['width'], dtype=lineage['dtype'], license_class=cfg['model']['license_class'],
                tokenizer_sha256=lineage['tokenizer_sha256'],
                sentencepiece_sha256={fmt: parity_records[fmt]['spm_sha256'] for fmt in FORMATS},
                encoder_parameters_before_sha256=before_hash, encoder_parameters_after_sha256=after_hash,
                all_parameters_frozen=True, eval_mode=True, feature='model.base_model(...).last_hidden_state[:,0,:]',
                pooler_invoked=False, classifier_invoked=False, candidate_forwards=0,
                cache_feature_sha256={fmt: caches[fmt]['feature_sha256'] for fmt in FORMATS},
                cache_input_sha256=evidence_hashes, cache_metadata_sha256={
                    fmt: sha(root / f'feature_cache_{fmt}.json') for fmt in FORMATS},
                protocol_sha256=sha(PROTOCOL), corpus_manifest_sha256=sha(root / 'corpus_manifest.json'),
                meaning_audit_sha256=meaning_sha,
            )
            payload = _json_bytes(lineage_value)
            if lineage_path.exists():
                assert lineage_path.read_bytes() == payload, 'Existing encoder lineage differs from resumed capture.'
            else:
                put(root, lineage_path.name, payload)
        finally:
            if model is not None:
                model.cpu()
            model = tokenizer = None
            torch.cuda.empty_cache()
    return dict(encoder_lineage=lineage_value, caches=caches, parity=parity_records,
                texts=len(texts), pairs=len(texts) * len(schema), meaning_audit_sha256=meaning_sha)


def verify_preflight(root, manifest, meaning_sha, cfg):
    records = sorted(root.glob('preflight_attempt_*.json'))
    assert records, 'No execution-proof preflight attempt found.'
    valid = []
    for path in records:
        row = load(path)
        assert row['preflight'] and row['passed'] and row['protocol_sha256'] == manifest['protocol_sha256'] == sha(PROTOCOL)
        assert row['corpus_manifest_sha256'] == sha(root / 'corpus_manifest.json')
        assert row['meaning_audit_sha256'] == meaning_sha
        assert row['floating_parameters'] == cfg['model']['floating_parameters'] == 70831107
        assert row['width'] == cfg['model']['width'] == 384
        assert row['model_revision'] == cfg['model']['revision']
        assert row['model_weight_sha256'] == cfg['model']['weight_sha256']
        assert row['encoder_parameters_before_sha256'] == row['encoder_parameters_after_sha256']
        assert row['encoder_parameters_frozen'] and row['encoder_parameter_gradient_count'] == 0
        assert row['actual_base_model_forward'] and row['tokenizer_parity_exact']
        assert row['head_only_gradient'] and row['head_gradient_L1'] > 0
        assert not row['final_inputs_captured'] and not row['alias_inputs_captured']
        assert row['feature_mmap_verified'] and row['feature_shape'] == [2, 384]
        assert row['feature_dtype'] == 'float32'
        assert sha(root / row['tokenizer_evidence_file']) == row['tokenizer_evidence_sha256']
        assert sha(root / row['feature_evidence_file']) == row['feature_evidence_sha256']
        evidence = load(root / row['tokenizer_evidence_file'])
        for fmt in FORMATS:
            assert evidence['formats'][fmt]['format'] == fmt
            assert evidence['formats'][fmt]['pair_count'] == len(evidence['formats'][fmt]['pairs']) == 1
            assert evidence['tokenizer_parity'][fmt]['exact'] is True
            assert evidence['formats'][fmt]['truncation'] is False
        cached = np.load(root / row['feature_evidence_file'], mmap_mode='r', allow_pickle=False)
        assert cached.shape == (2, 384) and cached.dtype == np.float32
        valid.append(path.name)
    return valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    parser.add_argument('--preflight', action='store_true', help='Run a real literal-pair forward/head-gradient preflight and save unique evidence.')
    args = parser.parse_args()
    result = preflight(args.protocol) if args.preflight else capture_development(args.protocol)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
