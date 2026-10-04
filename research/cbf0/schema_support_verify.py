"""Independently reconstruct CBF-8 from persisted cohorts, features, and heads."""
import os

for _name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import argparse
import hashlib
import itertools
import json
import math
import re
import subprocess
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from scipy.special import logsumexp
from transformers import AutoTokenizer

from build import canonical
from exact_state_build import parse as parse_state
from exact_state_run import score as exact_score, winner as exact_winner
from schema_grounding_compiler import compile_criterion, parse_criterion

PROTOCOL = Path(__file__).with_name('schema_support_protocol.json')
SCHEMA = ('reliability', 'purchase expense', 'operating expense', 'convenience')
ARMS = ('literal_generic', 'literal_defined', 'semantic_generic', 'semantic_defined')
ELIGIBLE = ('literal_defined', 'semantic_generic', 'semantic_defined')
FROZEN_VEY2 = 'e6b046ffbd138cbdbfb2f89c6ae77525fe6b0b18'
FLOAT_DTYPES = {'F16', 'BF16', 'F32', 'F64'}
COHORTS = ('training_atoms.json', 'validation_atoms.json', 'literal_holdout_atoms.json',
           'semantic_training_atoms.json', 'semantic_validation_atoms.json', 'development_atoms.json',
           'development_compositions.json', 'literal_cases.json', 'states.json', 'final_atoms.json',
           'final_compositions.json')
BASELINE_FILES = ('nli_xsmall_higher_cls_checkpoint.pt', 'nli_xsmall_higher_cls_training.json',
                  'normalizer_nli_xsmall_higher_cls.npz', 'features_nli_xsmall_higher_cls.npy',
                  'inputs_nli_xsmall_higher.json', 'encoder_lineage_nli_xsmall.json')


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()



def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
def hash_pair(record):
    for before, after in (
            ('encoder_parameters_before_sha256', 'encoder_parameters_after_sha256'),
            ('tensor_hash_before', 'tensor_hash_after'),
            ('full_tensor_hash_before', 'full_tensor_hash_after'),
            ('frozen_parameter_before_sha256', 'frozen_parameter_after_sha256'),
            ('frozen_encoder_before_sha256', 'frozen_encoder_after_sha256')):
        if before in record or after in record:
            return record.get(before), record.get(after)
    return None, None


def compare(actual, expected, label, tolerance=1e-12):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), label + ': keys')
        for key, value in expected.items():
            compare(actual[key], value, label + '/' + str(key), tolerance)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + ': length')
        for index, value in enumerate(expected):
            compare(actual[index], value, label + '/' + str(index), tolerance)
    elif isinstance(expected, float):
        require(isinstance(actual, (int, float)) and not isinstance(actual, bool) and math.isfinite(actual) and
                abs(actual - expected) <= tolerance, f'{label}: {actual!r} != {expected!r}')
    else:
        require(actual == expected, f'{label}: {actual!r} != {expected!r}')


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == {'mean', 'std'}, str(path) + ': normalizer keys')
        return {'mean': archive['mean'].copy(), 'std': archive['std'].copy()}


def finite_array(value, label):
    require(isinstance(value, np.ndarray) and value.dtype == np.float32 and np.isfinite(value).all(),
            label + ': expected finite float32')


def expected_hypothesis(cfg, field, format_name):
    sentence = 'Higher ' + field + ' is preferred.'
    if format_name == 'defined':
        sentence += ' Definition: ' + cfg['field_definitions'][field]
    return sentence


def row_texts(atoms, compositions, g0):
    questions = {c['text'] for c in atoms + g0}
    for c in compositions:
        for term in parse_criterion(c['text']):
            lo, hi = term['context_span']
            questions.add(c['text'][lo:hi].rstrip('.') + '.')
    return sorted(questions)


def contexts_for_queries(atoms, compositions):
    questions = {c['text'] for c in atoms}
    for c in compositions:
        for term in parse_criterion(c['text']):
            lo, hi = term['context_span']
            questions.add(c['text'][lo:hi].rstrip('.') + '.')
    return sorted(questions)


def field_context(text, term):
    lo, hi = term['context_span']
    return text[lo:hi].rstrip('.') + '.'


def compare_tokens(encoded, saved, label, prefix=''):
    ids_key = prefix + 'input_ids'
    mask_key = prefix + 'attention_mask'
    types_key = prefix + 'token_type_ids'
    expected_ids = np.asarray(encoded['input_ids'])
    expected_mask = np.asarray(encoded['attention_mask'])
    saved_ids = np.asarray(saved[ids_key])
    saved_mask = np.asarray(saved[mask_key])
    if saved_ids.ndim == 1:
        saved_ids = saved_ids[None, :]
    if saved_mask.ndim == 1:
        saved_mask = saved_mask[None, :]
    require(expected_ids.shape == saved_ids.shape and np.array_equal(expected_ids, saved_ids), label + ': input_ids')
    require(expected_mask.shape == saved_mask.shape and np.array_equal(expected_mask, saved_mask), label + ': attention_mask')
    if 'token_type_ids' in encoded:
        require(types_key in saved and np.array_equal(np.asarray(encoded['token_type_ids']),
                np.asarray(saved[types_key])), label + ': token_type_ids')
    else:
        require(types_key not in saved, label + ': unexpected token_type_ids')


def retokenize(evidence, fast, native, ordered_texts, cfg, format_name, label, source_higher=False,
               fields=None):
    accepted_formats = (format_name, 'higher') if source_higher and format_name == 'generic' else (format_name,)
    require(evidence.get('format') in accepted_formats and evidence.get('truncation') is False and
            evidence.get('candidate_forwards') == 0, label + ': format/truncation/candidate forwarding')
    pairs = evidence.get('pairs')
    require(isinstance(pairs, list), label + ': missing pair evidence')
    expected_rows = [(q, field) for q in ordered_texts for field in (fields or SCHEMA)]
    actual_rows = [(p.get('criterion'), p.get('field')) for p in pairs]
    require(actual_rows == expected_rows, label + ': sorted unique text x canonical field row order')
    require(len(set(actual_rows)) == len(actual_rows), label + ': duplicate pair rows')
    forbidden = {'axis', 'sign', 'weights', 'gold_atoms', 'candidate', 'candidate_text',
                 'teacher', 'teacher_scores', 'composition', 'pair_id', 'semantic_id'}
    total_tokens = 0
    maximum = 0
    native_total = 0
    native_maximum = 0
    row_lengths = []
    for index, pair in enumerate(pairs):
        require(set(pair) == ({'criterion', 'field', 'hypothesis'} if source_higher else
                              {'criterion', 'field', 'text_pair', 'input_ids',
                               'token_type_ids', 'attention_mask'}),
                label + ': exact neural pair evidence keys')
        require(not forbidden.intersection(pair), label + ': oracle/candidate metadata in neural input evidence')
        field = pair['field']
        hypothesis = expected_hypothesis(cfg, field, format_name)
        require(pair.get('text_pair', pair.get('hypothesis')) == hypothesis, label + ': exact canonical hypothesis')
        if 'hypothesis' in pair:
            require(pair['hypothesis'] == hypothesis, label + ': hypothesis field')
        expected_fast = fast([pair['criterion']], text_pair=[hypothesis], add_special_tokens=True,
                             padding=False, truncation=False, return_attention_mask=True, return_tensors='np')
        expected_native = native([pair['criterion']], text_pair=[hypothesis], add_special_tokens=True,
                                 padding=False, truncation=False, return_attention_mask=True, return_tensors='np')
        row_saved = {key: pair[key] for key in ('input_ids', 'attention_mask', 'token_type_ids') if key in pair}
        if row_saved:
            compare_tokens(expected_fast, row_saved, f'{label}/row{index}/fast')
        elif 'batches' not in evidence:
            raise AssertionError(label + ': token IDs absent from both rows and batches')
        fast_ids = np.asarray(expected_fast['input_ids'])
        native_ids = np.asarray(expected_native['input_ids'])
        require(np.array_equal(fast_ids, native_ids), label + ': fast/native SentencePiece pair IDs')
        require(np.array_equal(np.asarray(expected_fast['token_type_ids']),
                               np.asarray(expected_native['token_type_ids'])),
                label + ': fast/native SentencePiece segment IDs')
        fast_len = int(np.asarray(expected_fast['attention_mask']).sum())
        native_len = int(np.asarray(expected_native['attention_mask']).sum())
        require(fast_len > 0 and native_len > 0 and fast_len <= 128 and native_len <= 128,
                label + ': token limit or empty pair')
        require(pair.get('input_ids', fast_ids[0].tolist()) == fast_ids[0].tolist() and
                pair.get('attention_mask', [1] * fast_len) == [1] * fast_len,
                label + ': per-row input/mask evidence')
        if 'token_type_ids' in pair:
            require(pair['token_type_ids'] == np.asarray(expected_fast['token_type_ids'])[0].tolist(),
                    label + ': per-row segment IDs')
        row_lengths.append(fast_len)
        total_tokens += fast_len
        native_total += native_len
        maximum = max(maximum, int(fast_ids.shape[-1]))
        native_maximum = max(native_maximum, int(native_ids.shape[-1]))
        if 'native_input_ids' in pair:
            native_saved = {key: pair[key] for key in
                            ('native_input_ids', 'native_attention_mask', 'native_token_type_ids')
                            if key in pair}
            compare_tokens(expected_native, native_saved, f'{label}/row{index}/native', prefix='native_')
    if 'batches' in evidence:
        offset = 0
        batch_tokens = 0
        for batch_index, batch in enumerate(evidence['batches']):
            require(batch.get('start') == offset and 0 < batch.get('count', 0) <= 32,
                    label + ': batch coverage/order')
            chunk = pairs[offset:offset + batch['count']]
            second = [expected_hypothesis(cfg, pair['field'], format_name) for pair in chunk]
            encoded = fast([pair['criterion'] for pair in chunk], text_pair=second, add_special_tokens=True,
                           padding=True, truncation=False, return_attention_mask=True, return_tensors='np')
            stored = {key: batch[key] for key in ('input_ids', 'attention_mask', 'token_type_ids') if key in batch}
            compare_tokens(encoded, stored, f'{label}/batch{batch_index}')
            chunk_lengths = row_lengths[offset:offset + batch['count']]
            expected_width = max(chunk_lengths)
            require(batch.get('lengths', chunk_lengths) == chunk_lengths and
                    batch.get('padded_width', expected_width) == expected_width and
                    encoded['input_ids'].shape[1] == expected_width,
                    label + ': batch lengths/padded width')
            batch_tokens += int(np.asarray(encoded['attention_mask']).sum())
            offset += len(chunk)
        require(offset == len(pairs) and batch_tokens == total_tokens, label + ': batch token accounting')
        require(len(evidence['batches']) == evidence.get('forwards', len(evidence['batches'])),
                label + ': forward count')
    require(evidence.get('pair_count', len(pairs)) == len(pairs) and
            evidence.get('lengths', row_lengths) == row_lengths,
            label + ': pair count/per-row lengths')
    require(total_tokens == evidence.get('tokens', total_tokens), label + ': fast token count')
    require(native_total == evidence.get('native_tokens', native_total), label + ': native token count')
    maximum_sequence_length = evidence.get('maximum_sequence_length', evidence.get('maximum_tokens'))
    if maximum_sequence_length is not None:
        require(maximum == maximum_sequence_length, label + ': maximum sequence length')
    if 'native_maximum_tokens' in evidence:
        require(native_maximum == evidence['native_maximum_tokens'], label + ': native max length')
    return dict(pairs=len(pairs), fast_valid_tokens=total_tokens, native_valid_tokens=native_total,
                fast_max_tokens=maximum, native_max_tokens=native_maximum,
                batches=len(evidence.get('batches', [])))



def model_package(cfg, root):
    model = cfg['model']
    require(model['repo'] == 'cross-encoder/nli-deberta-v3-xsmall' and
            model['revision'] == 'a150876415327c80daeff35ca6f68f5ed8cf5c24' and
            model['width'] == 384, 'pinned NLI xsmall model contract')
    weight_path = Path(hf_hub_download(model['repo'], 'model.safetensors', revision=model['revision'],
                                       local_files_only=True))
    config_path = Path(hf_hub_download(model['repo'], 'config.json', revision=model['revision'],
                                       local_files_only=True))
    require(sha(weight_path) == model['weight_sha256'], 'actual pinned safetensors hash')
    model_config = load(config_path)
    require(model_config.get('hidden_size') == model['width'] and model_config.get('num_hidden_layers') == 12 and
            model_config.get('model_type') == 'deberta-v2', 'actual model config/width/depth')
    floating = 0
    exclusions = {}
    with safe_open(weight_path, framework='np', device='cpu') as archive:
        for name in archive.keys():
            tensor = archive.get_slice(name)
            dtype = tensor.get_dtype()
            shape = tensor.get_shape()
            count = math.prod(shape)
            if dtype in FLOAT_DTYPES:
                floating += count
            else:
                exclusions[name] = dict(dtype=dtype, shape=shape, elements=count)
    require(floating == model['floating_parameters'] == 70831107,
            f'live floating parameter count {floating}')
    lineage_path = root / 'encoder_lineage.json'
    lineage = load(lineage_path)
    require(lineage.get('model_repo', lineage.get('repo', lineage.get('encoder'))) == model['repo'] and
            lineage['revision'] == model['revision'] and lineage['weight_sha256'] == model['weight_sha256'] and
            lineage['floating_parameters'] == floating and lineage['width'] == model['width'],
            'encoder lineage/model package mismatch')
    require(lineage.get('candidate_forwards') == 0 and lineage.get('dtype') == 'float32',
            'encoder candidate/dtype lineage')
    for name in ('missing_keys', 'mismatched_keys', 'error_msgs'):
        require(not lineage.get('loading', {}).get(name), 'model loading ' + name)
    fast = AutoTokenizer.from_pretrained(model['repo'], revision=model['revision'], use_fast=True,
                                         local_files_only=True, trust_remote_code=False)
    native = AutoTokenizer.from_pretrained(model['repo'], revision=model['revision'], use_fast=False,
                                           local_files_only=True, trust_remote_code=False)
    require(fast.is_fast and not native.is_fast and 'DebertaV2' in type(native).__name__ and
            (hasattr(native, 'sp_model') or hasattr(native, 'spm_processor')),
            'pinned fast tokenizer and native SentencePiece implementation')
    fast_digest = hashlib.sha256(fast.backend_tokenizer.to_str().encode()).hexdigest()
    spm = getattr(native, 'sp_model', getattr(native, 'spm_processor', None))
    spm_digest = hashlib.sha256(spm.serialized_model_proto()).hexdigest()
    spm_path = Path(hf_hub_download(model['repo'], 'spm.model', revision=model['revision'],
                                    local_files_only=True))
    spm_file_digest = sha(spm_path)

    def matches_digest(value, actual):
        if isinstance(value, dict):
            return bool(value) and set(value.values()) == {actual}
        return value == actual

    require(matches_digest(lineage.get('tokenizer_sha256'), fast_digest), 'fast tokenizer pinned hash')
    require(matches_digest(lineage.get('sentencepiece_sha256'), spm_file_digest),
            'native SentencePiece file hash')
    require(spm_digest == spm_file_digest, 'SentencePiece processor/file byte parity')
    before, after = hash_pair(lineage)
    require(isinstance(before, str) and re.fullmatch(r'[0-9a-f]{64}', before) and before == after,
            'frozen encoder full tensor hash unchanged during capture')
    position_ids = [name for name in exclusions if 'position_ids' in name.casefold()]
    return dict(model=model, lineage=lineage, fast=fast, native=native, parameter_count=floating,
                parameter_exclusions=exclusions, position_id_buffers=position_ids,
                fast_tokenizer_sha256=fast_digest, sentencepiece_sha256=spm_file_digest,
                full_tensor_hash_sha256=before, spm_path=spm_path,
                model_hashes={str(weight_path): sha(weight_path), str(config_path): sha(config_path),
                              str(spm_path): spm_file_digest, str(lineage_path): sha(lineage_path)})
 
 
def verify_sentencepiece_parity(root, evidence, package, format_name, suffix=None):
    import sentencepiece

    suffix = suffix or format_name
    path = root / f'native_sentencepiece_parity_{suffix}.json'
    parity = load(path)
    tokenizer = package['fast']
    native = sentencepiece.SentencePieceProcessor(model_file=str(package['spm_path']))
    delimiter_records = []
    maximum_delta = 0
    for row in evidence['pairs']:
        first = native.encode(row['criterion'], out_type=int)
        second = native.encode(row['text_pair'], out_type=int)
        expected = tokenizer.build_inputs_with_special_tokens(first, second)
        type_ids = tokenizer.create_token_type_ids_from_sequences(first, second)
        encoded = tokenizer(row['criterion'], text_pair=row['text_pair'], add_special_tokens=True,
                            padding=False, truncation=False, return_attention_mask=True,
                            return_token_type_ids=True)
        require(encoded['input_ids'] == expected and encoded['token_type_ids'] == type_ids,
                f'{suffix}: independent native SentencePiece delimiter parity')
        require(encoded['attention_mask'] == [1] * len(expected) and len(expected) <= 128,
                f'{suffix}: native SentencePiece truncation/length')
        delimiter_records.append(dict(
            criterion=row['criterion'], field=row['field'], input_ids=expected, token_type_ids=type_ids,
            special_token_positions=[i for i, token in enumerate(expected)
                                     if token in (tokenizer.cls_token_id, tokenizer.sep_token_id)],
        ))
        maximum_delta = max(maximum_delta, abs(len(expected) - len(encoded['input_ids'])))
    require(parity.get('format') == format_name and parity.get('pairs') == len(delimiter_records) and
            parity.get('exact') is True and parity.get('spm_sha256') == package['sentencepiece_sha256'] and
            parity.get('spm_model_path') == str(package['spm_path']) and
            parity.get('pair_id_rows_sha256') == sha_bytes(canonical(delimiter_records)) and
            parity.get('pair_ids') == delimiter_records and
            parity.get('checked_rows') == len(delimiter_records) and
            parity.get('maximum_length_delta') == maximum_delta == 0,
            f'{suffix}: independently reconstructed native SentencePiece parity record')
    return dict(file=path.name, sha256=sha(path), rows=len(delimiter_records), exact=True)


def verify_source(root, cfg, manifest):
    source = Path(cfg['source_root'])
    require(source.is_dir(), 'CBF7 source root missing')
    c7_protocol = Path(__file__).with_name('relation_pretrained_protocol.json')
    c7_cfg = load(c7_protocol)
    require(Path(c7_cfg['output_root']).resolve() == source.resolve(), 'CBF7 source root/protocol mismatch')
    inventory_path = Path(__file__).with_name('relation_pretrained_result_manifest.json')
    inventory = load(inventory_path)
    source_verify = load(source / 'verification.json')
    source_corpus = load(source / 'corpus_manifest.json')
    verification_path = source / 'verification.json'
    require(source_verify.get('all_checks_pass') is True and
            inventory.get('all_independent_checks_pass') is True,
            'CBF7 independent verification not passed')
    require(manifest.get('source_result_manifest_sha256') == sha(inventory_path) and
            manifest.get('source_verification_sha256') == sha(verification_path) and
            manifest.get('source_results_sha256') == sha(source / 'results.json') == inventory['results_sha256'] and
            manifest.get('source_corpus_manifest_sha256') == sha(source / 'corpus_manifest.json'),
            'CBF7 source inventory/results/verification/corpus hashes')
    require(inventory.get('protocol_sha256') == sha(c7_protocol) and
            inventory.get('verification_sha256') == sha(verification_path) and
            source_corpus.get('experiment') == 'CBF-7' and
            source_corpus.get('final_closed_until_selection') is True and
            source_corpus.get('protocol_sha256') == sha(c7_protocol),
            'CBF7 source corpus and inventory lineage')
    require(manifest.get('source_measurement_git_revision') == inventory['measurement_git'],
            'CBF7 source measurement revision')
    checked = source_verify['checked_source_and_model_sha256']
    c8_source_verified = manifest.get('source_verified_files', {})
    result = {}
    for name in COHORTS:
        if name in ('semantic_training_atoms.json', 'semantic_validation_atoms.json', 'final_atoms.json',
                    'final_compositions.json'):
            continue
        path = source / name
        actual = sha(path)
        require(checked.get(str(path)) == actual and c8_source_verified.get(name) == actual,
                'CBF7 independently verified/reused cohort ' + name)
        result[str(path)] = actual
    refs = manifest.get('baseline_references')
    require(isinstance(refs, dict) and refs.get('arm') == 'literal_generic' and
            refs.get('format') == 'generic' and refs.get('no_retrain') is True and
            refs.get('no_reencode') is True and isinstance(refs.get('files'), dict),
            'missing explicit byte-identical CBF7 baseline references')
    roles = dict(zip(('checkpoint', 'training', 'normalizer', 'features', 'inputs', 'lineage'),
                     BASELINE_FILES))
    require(set(refs['files']) == set(roles), 'CBF7 baseline reference roles')
    for role, name in roles.items():
        path = source / name
        actual = sha(path)
        custody = ('CBF8 preparation-time metadata byte pin; CBF7 reconstructed head fields checked below'
                   if role == 'training' else 'CBF7 independent verification byte hash')
        require(refs['files'][role] == dict(file=name, source_path=str(path), sha256=actual,
                                          custody_basis=custody) and
                c8_source_verified.get(name) == actual, 'CBF8 baseline custody ' + role)
        if role == 'training':
            details = load(path)
            proof = source_verify['stages']['nli_xsmall_higher_cls']['head']
            require(details['selected_epoch'] == proof['selected_epoch'] and
                    abs(details['validation_CE'] - proof['unweighted_validation_CE']) <= 2e-5 and
                    details['frozen_encoder_before_sha256'] == proof['frozen_parameter_sha256'],
                    'CBF7 independently reconstructed baseline metadata fields')
        else:
            require(checked.get(str(path)) == actual, 'CBF7 independently verified baseline bytes ' + role)
        result[str(path)] = actual
    return source, source_verify, result


def validate_atom(atom, label, atomic=False):
    require(isinstance(atom.get('text'), str) and atom['text'].strip(), label + ': text')
    require(isinstance(atom.get('id'), str) and atom['id'], label + ': ID')
    weights = atom.get('weights')
    require(isinstance(weights, list) and len(weights) == 4 and all(type(v) is int for v in weights),
            label + ': four integer weights')
    if 'axis' in atom:
        require(type(atom['axis']) is int and atom['axis'] in range(4), label + ': axis')
        require(type(atom.get('sign')) is int and atom['sign'] in (-1, 1), label + ': sign')
        require(weights[atom['axis']] * atom['sign'] > 0 and all(weights[j] == 0 for j in range(4) if j != atom['axis']),
                label + ': atomic orientation/weights')
    if atomic:
        require(sum(v != 0 for v in weights) == 1, label + ': expected one supported axis')


def validate_reversal_cohort(rows, expected_count, per_direction, label):
    require(len(rows) == expected_count and len({r['id'] for r in rows}) == expected_count,
            label + ': row/unique-ID count')
    require(Counter((r['axis'], r['sign']) for r in rows) ==
            Counter({(axis, sign): per_direction for axis in range(4) for sign in (-1, 1)}),
            label + ': field/sign balance')
    pairs = {}
    for row in rows:
        validate_atom(row, label + '/' + row['id'], atomic=True)
        pairs.setdefault(row['pair_id'], []).append(row)
    require(len(pairs) == expected_count // 2 and all(len(pair) == 2 for pair in pairs.values()),
            label + ': reversal groups')
    for pair in pairs.values():
        require(pair[0]['weights'] == [-value for value in pair[1]['weights']] and
                pair[0]['axis'] == pair[1]['axis'] and pair[0]['sign'] == -pair[1]['sign'],
                label + ': opposite directions')
    return pairs


def normalized_tokens(text):
    return tuple(re.findall(r'[a-z0-9]+', text.casefold()))


def includes_tokens(text_tokens, phrase_tokens):
    return bool(phrase_tokens) and len(phrase_tokens) <= len(text_tokens) and any(
        text_tokens[index:index + len(phrase_tokens)] == phrase_tokens
        for index in range(len(text_tokens) - len(phrase_tokens) + 1))


def historical_reference_rows(cfg, manifest, final_audit):
    c7_protocol = PROTOCOL.with_name('relation_pretrained_protocol.json')
    c6_root = Path(load(c7_protocol)['source_root'])
    source_roots = {'CBF6': c6_root, 'CBF7': Path(cfg['source_root'])}
    expected_counts = {
        'development_atoms.json': 32, 'development_compositions.json': 50,
        'final_atoms.json': 128, 'final_compositions.json': 128,
        'training_atoms.json': 32, 'validation_atoms.json': 32,
        'literal_holdout_atoms.json': 32, 'literal_cases.json': 155,
    }
    lineage = manifest.get('lexical_reference_lineage', {})
    final_lineage = final_audit.get('source_lineage', {})
    protocol_lineage = final_lineage.get('CBF6_protocol', {})
    require(protocol_lineage == dict(path=str(c7_protocol), sha256=sha(c7_protocol)),
            'final-corpus historical protocol lineage')
    rows, checked = [], {}
    for source_name, source_root in source_roots.items():
        source_map = final_lineage.get(f'{source_name}_source_files', {})
        for filename, count in expected_counts.items():
            key = f'{source_name}/{filename}'
            path = source_root / filename
            record = lineage.get(key)
            expected = dict(path=str(path), sha256=sha(path), rows=count)
            require(record == expected, 'manifest lexical reference lineage ' + key)
            require(source_map.get(filename) == expected, 'final audit source lineage ' + key)
            require(manifest.get('source_verified_files', {}).get('lexical_reference:' + key) == expected['sha256'],
                    'manifest source-verified lexical reference ' + key)
            source_rows = load(path)
            require(len(source_rows) == count and all(isinstance(row.get('text'), str) for row in source_rows),
                    'historical reference row schema/count ' + key)
            rows.extend(dict(source=key, id=row['id'], text=row['text']) for row in source_rows)
            checked[key] = expected['sha256']
    require(set(lineage) == {f'{source_name}/{filename}' for source_name in source_roots
                             for filename in expected_counts}, 'complete CBF6/CBF7 lexical reference inventory')
    return rows, checked


def verify_compiler_checks(root, groups):
    atom_rows = (('semantic_training', groups['semantic_training_atoms']),
                 ('semantic_validation', groups['semantic_validation_atoms']),
                 ('final_atoms', groups['final_atoms']))
    by_phrase = {normalized_tokens(row['text']): row
                 for _, cohort in atom_rows for row in cohort}
    require(len(by_phrase) == sum(len(cohort) for _, cohort in atom_rows),
            'compiler semantic atom phrase uniqueness')
    expected_records = []

    def compile_row(cohort, row, calls_expected):
        calls = []

        def resolve(term):
            lo, hi = term['context_span']
            phrase = normalized_tokens(row['text'][lo:hi].rstrip('.') + '.')
            atom = by_phrase.get(phrase)
            require(atom is not None, 'compiler semantic phrase support ' + row['id'])
            calls.append(atom['id'])
            return dict(axis=atom['axis'], sign=atom['sign'])

        vector, ir = compile_criterion(row['text'], resolve)
        parsed = parse_criterion(row['text'])
        require(vector == row['weights'] and len(parsed) == calls_expected and
                all(term['literal_axis'] is None for term in parsed) and len(calls) == calls_expected,
                'independent compiler AST/weights/callbacks ' + row['id'])
        expected_records.append(dict(cohort=cohort, id=row['id'], weights=vector,
                                     atoms=ir, atomic_ids=calls))

    for cohort_name, cohort in atom_rows:
        for row in cohort:
            compile_row(cohort_name, row, 1)
    for row in groups['final_compositions']:
        compile_row('final_compositions', row, 2)
    compare(load(root / 'compiler_checks.json'), expected_records, 'independent corpus compiler records')
    return dict(rows=len(expected_records), semantic_atoms=sum(len(rows) for _, rows in atom_rows),
                compositions=len(groups['final_compositions']))


def verify_training_audit(groups, audit):
    require(audit.get('experiment') == 'CBF8' and audit.get('license_class') == 'shipping-train',
            'semantic training audit identity/license')
    provenance = audit.get('provenance', {})
    generator_path = PROTOCOL.with_name('schema_support_training_corpus.py')
    require(provenance.get('protocol_sha256') == sha(PROTOCOL) and
            provenance.get('generator_path') == str(generator_path) and
            provenance.get('generator_sha256') == sha(generator_path),
            'semantic training generator/protocol lineage')
    require(audit.get('output_sha256') == dict(
        training_atoms_json=sha_bytes(canonical(groups['semantic_training_atoms']) + b'\n'),
        validation_atoms_json=sha_bytes(canonical(groups['semantic_validation_atoms']) + b'\n')),
        'semantic training/validation serialized output hashes')
    meanings = audit.get('meaning_review', {})
    template = audit.get('template_separation', {})
    scopes = (
        ('training', 'semantic_training_atoms', meanings.get('training_pairs', []), 32,
         'semantic_train'),
        ('validation', 'semantic_validation_atoms', meanings.get('validation_pairs', []), 16,
         'semantic_validation'),
    )
    template_ids, signatures, quantities = {}, {}, {}
    for scope, group_name, records, count, stratum in scopes:
        rows = {row['id']: row for row in groups[group_name]}
        require(len(records) == count and len(rows) == 2 * count, scope + ': independent pair meanings count')
        ids, signature_rows, quantity_rows = set(), set(), set()
        seen_pairs = set()
        for record in records:
            axis = record.get('axis')
            require(type(axis) is int and axis in range(4) and record.get('canonical_field') == SCHEMA[axis],
                    scope + ': meaning field grounding')
            pair_id = record.get('pair_id')
            require(pair_id not in seen_pairs and isinstance(pair_id, str), scope + ': meaning pair IDs')
            seen_pairs.add(pair_id)
            positive = rows.get(record.get('positive_atom_id'))
            negative = rows.get(record.get('negative_atom_id'))
            require(positive is not None and negative is not None and
                    positive['text'] == record.get('positive_text') and
                    negative['text'] == record.get('negative_text') and
                    positive['pair_id'] == negative['pair_id'] == pair_id and
                    positive['axis'] == negative['axis'] == axis and
                    positive['sign'] == 1 and negative['sign'] == -1 and
                    positive['stratum'] == negative['stratum'] == stratum,
                    scope + ': positive/negative audit labels and cohort rows')
            template_text = record.get('template', '')
            require(template_text.count('{direction}') == 1, scope + ': one direction slot')
            prefix, suffix = template_text.split('{direction}')

            def direction_slot(text):
                require(text.startswith(prefix) and text.endswith(suffix), scope + ': template boundary')
                middle_end = len(text) - len(suffix) if suffix else len(text)
                middle = text[len(prefix):middle_end]
                require(middle and not any(character.isspace() for character in middle),
                        scope + ': single direction token')
                return middle

            pos_direction = direction_slot(positive['text'])
            neg_direction = direction_slot(negative['text'])
            require(pos_direction != neg_direction and
                    record.get('template_signature') ==
                    ' '.join(normalized_tokens(template_text.replace('{direction}', 'direction'))),
                    scope + ': direction-only template and signature')
            template_id = record.get('template_id')
            quantity = record.get('quantity')
            require(isinstance(template_id, str) and template_id and isinstance(quantity, str) and quantity and
                    isinstance(record.get('unit'), str) and record['unit'] and
                    isinstance(record.get('fixed_exposure'), str) and record['fixed_exposure'],
                    scope + ': template and measurement metadata')
            require(template_id not in ids, scope + ': duplicate template ID')
            ids.add(template_id)
            signature_rows.add(record['template_signature'])
            quantity_rows.add(quantity)
        template_ids[scope] = ids
        signatures[scope] = signature_rows
        quantities[scope] = quantity_rows
    require(template.get('disjoint') is True and
            template.get('training_template_ids') == sorted(template_ids['training']) and
            template.get('validation_template_ids') == sorted(template_ids['validation']) and
            template.get('training_quantity_templates') == sorted(quantities['training']) and
            template.get('validation_quantity_templates') == sorted(quantities['validation']) and
            template_ids['training'].isdisjoint(template_ids['validation']) and
            signatures['training'].isdisjoint(signatures['validation']) and
            quantities['training'].isdisjoint(quantities['validation']),
            'semantic training/validation template and quantity separation')
    require('diagnostic' in audit.get('validation_usage', '').casefold() and
            'MUST NOT' in audit['validation_usage'], 'semantic validation diagnostic-only policy')
    for key in ('normalized_duplicates', 'normalized_atomic_semantic_duplicates',
                'literal_schema_name_hits', 'byte_exact_reference_matches',
                'normalized_reference_phrase_matches', 'complete_old_phrase_inclusions'):
        require(not audit.get('lexical_checks', {}).get(key), 'semantic training lexical exclusion ' + key)
    return dict(training_pairs=len(template_ids['training']), validation_pairs=len(template_ids['validation']),
                template_and_quantity_disjoint=True)


def validate_corpus(root, cfg, manifest, meaning):
    groups = {Path(name).stem: load(root / name) for name in COHORTS}
    expected_counts = dict(training_atoms=32, validation_atoms=32, literal_holdout_atoms=32,
                           semantic_training_atoms=64, semantic_validation_atoms=32, development_atoms=32,
                           development_compositions=50, literal_cases=155, states=64,
                           final_atoms=128, final_compositions=128)
    for name, count in expected_counts.items():
        require(len(groups[name]) == count, name + ': row count')
        if name in ('training_atoms', 'validation_atoms', 'literal_holdout_atoms',
                    'semantic_training_atoms', 'semantic_validation_atoms', 'development_atoms',
                    'final_atoms'):
            for row in groups[name]:
                validate_atom(row, name + '/' + row['id'], atomic=True)
    source, source_verify, source_hashes = verify_source(root, cfg, manifest)
    for name in ('training_atoms.json', 'validation_atoms.json', 'literal_holdout_atoms.json',
                 'development_atoms.json', 'development_compositions.json', 'literal_cases.json', 'states.json'):
        require((root / name).read_bytes() == (source / name).read_bytes(), 'CBF7 byte-identical reuse ' + name)
    audit = load(root / 'final_corpus_audit.json')
    training_audit = load(root / 'training_corpus_audit.json')
    old_rows, historical_hashes = historical_reference_rows(cfg, manifest, audit)
    old_phrases = []
    for old in old_rows:
        old_phrases.append((old['source'], old['id'], 'full', old['text']))
        for term in parse_criterion(old['text']):
            lo, hi = term['span']
            old_phrases.append((old['source'], old['id'], 'semantic', old['text'][lo:hi]))
    semantic_groups = (groups['semantic_training_atoms'], groups['semantic_validation_atoms'],
                       groups['final_atoms'], groups['final_compositions'])
    new_phrase_records = []
    for rows in semantic_groups:
        for row in rows:
            new_phrase_records.append((row['id'], 'full', row['text']))
            for term in parse_criterion(row['text']):
                lo, hi = term['span']
                new_phrase_records.append((row['id'], 'semantic', row['text'][lo:hi]))
    historical_inclusions = []
    for new_id, kind, phrase in new_phrase_records:
        new_tokens = normalized_tokens(phrase)
        for old_source, old_id, old_kind, old_phrase in old_phrases:
            if includes_tokens(new_tokens, normalized_tokens(old_phrase)):
                historical_inclusions.append((new_id, kind, old_source, old_id, old_kind))
    require(not historical_inclusions, 'new corpus contains a complete historical phrase')
    train_validation_records = []
    for cohort in (groups['semantic_training_atoms'], groups['semantic_validation_atoms']):
        for row in cohort:
            train_validation_records.append(normalized_tokens(row['text']))
            train_validation_records.extend(normalized_tokens(row['text'][term['span'][0]:term['span'][1]])
                                            for term in parse_criterion(row['text']))
    training_validation_keys = set(train_validation_records)
    final_phrase_keys = []
    for row in groups['final_atoms'] + groups['final_compositions']:
        final_phrase_keys.append(normalized_tokens(row['text']))
        final_phrase_keys.extend(normalized_tokens(row['text'][term['span'][0]:term['span'][1]])
                                 for term in parse_criterion(row['text']))
    require(training_validation_keys.isdisjoint(final_phrase_keys),
            'fresh final phrases exactly collide with C8 training/validation phrases')
    literal_schema_hits = [
        (row_id, kind, field)
        for row_id, kind, phrase in new_phrase_records for field in SCHEMA
        if includes_tokens(normalized_tokens(phrase), normalized_tokens(field))
    ]
    require(not literal_schema_hits, 'new semantic corpora contain canonical field-name leakage')
    protected_tokens = [normalized_tokens(phrase) for _, _, _, phrase in old_phrases]
    for definition in cfg['field_definitions'].values():
        require(not any(includes_tokens(normalized_tokens(definition), phrase) for phrase in protected_tokens),
                'evaluation phrase leaks into field definition')

    literal_texts = [{c['text'] for c in groups[k]} for k in
                     ('training_atoms', 'validation_atoms', 'literal_holdout_atoms')]
    require(all(not literal_texts[i] & literal_texts[j] for i in range(3) for j in range(i + 1)),
            'literal cohort text disjointness')
    require(not ({c['text'] for c in groups['semantic_training_atoms']} &
                 set().union(*literal_texts)), 'semantic training overlaps literal cohorts')
    require(not ({c['text'] for c in groups['semantic_validation_atoms']} &
                 set().union(*literal_texts)), 'semantic validation overlaps literal cohorts')

    states = groups['states']
    require(len({s['id'] for s in states}) == 64 and
            sum(s['source_split'] != 'train' for s in states) == 32, 'exact state partitions')
    for state in states:
        facts = [list(parse_state(text)) for text in state['candidates']]
        require(facts == state['parsed_facts'] and len(facts) == state['K'] and
                all(len(row) == 4 and all(type(v) is int and 0 <= v <= 100 for v in row) for row in facts),
                'integer parsed facts ' + state['id'])
        require(set(SCHEMA) == set(re.findall(r'([^:;]+): \d+ percent', state['candidates'][0])),
                'candidate field schema ' + state['id'])

    atomic_by_text = {a['text'].rstrip('.') + '.': a for a in groups['final_atoms']}
    require(len(atomic_by_text) == 128, 'unique final atomic question text')
    composition_uses = Counter()
    coverage = Counter()
    composition_checks = []
    for composition in groups['final_compositions']:
        validate_atom(composition, 'final composition ' + composition['id'])
        terms = parse_criterion(composition['text'])
        require(len(terms) == 2 and all(term['literal_axis'] is None for term in terms),
                'final composition grammar/literal dispatch ' + composition['id'])
        vector = [0] * 4
        component_key = []
        teacher_components = []
        components = composition.get('components')
        require(isinstance(components, list) and len(components) == 2, 'final composition components')
        for component, term in zip(components, terms):
            question = field_context(composition['text'], term)
            atom = atomic_by_text.get(question)
            require(atom is not None, 'fresh composition component phrase absent from atoms ' + composition['id'])
            require(component.get('atom_id') == atom['id'] and component.get('factor') == term['factor'],
                    'fresh component ID/factor ' + composition['id'])
            vector[atom['axis']] += term['factor'] * atom['sign']
            composition_uses[atom['id']] += 1
            component_key.append((atom['axis'], atom['sign'], term['factor']))
            teacher_components.append(dict(atom_id=atom['id'], axis=atom['axis'], sign=atom['sign'],
                                           factor=component['factor'], semantic_span=term['span'],
                                           context_span=term['context_span']))
        require(vector == composition['weights'], 'exact compiler composition integer vector ' + composition['id'])
        require(component_key[0][0] != component_key[1][0], 'composition distinct support fields')
        coverage[tuple(component_key)] += 1
        composition_checks.append(dict(id=composition['id'], pair_id=composition['pair_id'],
                                       teacher_components=teacher_components, teacher_weights=vector))
    require(set(composition_uses) == {a['id'] for a in groups['final_atoms']} and
            set(composition_uses.values()) == {2}, 'all fresh atoms used exactly twice')
    expected_coverage = {tuple(zip(axes, signs, factors))
                         for axes in itertools.combinations(range(4), 2)
                         for signs in itertools.product((-1, 1), repeat=2)
                         for factors in itertools.product((1, 2), repeat=2)}
    require(set(coverage) == expected_coverage, 'all field/sign/factor composition patterns')
    composition_pairs = {}
    final_by_id = {row['id']: row for row in groups['final_atoms']}
    for composition in groups['final_compositions']:
        composition_pairs.setdefault(composition['pair_id'], []).append(composition)
    require(len(composition_pairs) == 64 and
            all(len(pair) == 2 and pair[0]['weights'] == [-value for value in pair[1]['weights']]
                for pair in composition_pairs.values()), 'final composition full-vector reversal groups')
    for pair in composition_pairs.values():
        def terms_by_axis(row):
            values = []
            for component in row['components']:
                atom = final_by_id[component['atom_id']]
                values.append((atom['axis'], atom['sign'], component['factor']))
            return sorted(values)
        require(sorted((axis, -sign, factor) for axis, sign, factor in terms_by_axis(pair[0])) ==
                terms_by_axis(pair[1]), 'final composition reversal preserves terms/factors')
    for atom in groups['final_atoms']:
        terms = parse_criterion(atom['text'])
        require(len(terms) == 1 and terms[0]['literal_axis'] is None, 'fresh atom compiler grammar')
        require(terms[0]['factor'] * atom['sign'] == atom['weights'][atom['axis']],
                'fresh atom exact compiler weight')

    require(audit.get('protocol', {}).get('sha256') == sha(PROTOCOL), 'final corpus audit protocol lineage')
    coverage_records = [dict(components=[dict(axis=axis, sign=sign, factor=factor)
                                        for axis, sign, factor in key], count=coverage[key])
                        for key in sorted(coverage)]
    compare(audit.get('composition_teacher_checks'), composition_checks,
            'final audit compiler component-to-atom records')
    compare(audit.get('coverage'), coverage_records, 'final audit field/sign/factor coverage')
    compare(audit.get('atom_composition_uses'), dict(sorted(composition_uses.items())),
            'final audit exact atom composition uses')
    compare(parent_final.get('atom_composition_uses'), dict(sorted(composition_uses.items())),
            'preparation atom composition uses')
    compare(parent_final.get('composition_coverage'),
            {repr(key): value for key, value in sorted(coverage.items())},
            'preparation composition coverage')
    training_proof = verify_training_audit(groups, training_audit)
    compiler_proof = verify_compiler_checks(root, groups)
    compiler_rows = load(root / 'compiler_checks.json')
    parent_training = training_audit.get('parent_preparation_checks', {})
    parent_final = audit.get('parent_preparation_checks', {})
    compare(parent_training.get('compiler_ast_weight_checks'),
            [row for row in compiler_rows if row['cohort'] in ('semantic_training', 'semantic_validation')],
            'training audit compiler replay')
    compare(parent_final.get('compiler_ast_weight_checks'),
            [row for row in compiler_rows if row['cohort'] in ('final_atoms', 'final_compositions')],
            'final audit compiler replay')
    require(parent_training.get('semantic_validation_never_selects_checkpoint_or_arm') is True and
            parent_final.get('independent_meaning_audit_not_created') is True,
            'preparation stage selection/meaning-audit separation')
    final_outputs = dict(
        final_atoms_json=sha_bytes((json.dumps(groups['final_atoms'], sort_keys=True, indent=2,
                                               ensure_ascii=False) + '\n').encode('utf-8')),
        final_compositions_json=sha_bytes((json.dumps(groups['final_compositions'], sort_keys=True, indent=2,
                                                      ensure_ascii=False) + '\n').encode('utf-8')))
    require(audit.get('output_sha256') == final_outputs and
            audit.get('counts') == dict(atoms=128, compositions=128, atom_reversal_pairs=64,
                                        composition_reversal_pairs=64, training_validation_phrase_rows=192),
            'final corpus serialized output hashes/counts')
    final_meanings = audit.get('meaning_review', {}).get('pairs', [])
    require(len(final_meanings) == 64, 'final audit meaning-pair count')
    final_by_id = {row['id']: row for row in groups['final_atoms']}
    seen_final_pairs = set()
    for record in final_meanings:
        axis = record.get('axis')
        pair_id = record.get('pair_id')
        higher = final_by_id.get(record.get('higher_atom_id'))
        lower = final_by_id.get(record.get('lower_atom_id'))
        require(type(axis) is int and axis in range(4) and record.get('canonical_field') == SCHEMA[axis] and
                pair_id not in seen_final_pairs and higher is not None and lower is not None and
                higher['pair_id'] == lower['pair_id'] == pair_id and
                higher['axis'] == lower['axis'] == axis and higher['sign'] == 1 and lower['sign'] == -1 and
                higher['text'] == record.get('higher_text') and lower['text'] == record.get('lower_text') and
                isinstance(record.get('quantity_exposure'), str) and record['quantity_exposure'] and
                isinstance(record.get('rationale'), str) and record['rationale'],
                'final audit meaning record/atom linkage')
        seen_final_pairs.add(pair_id)
    for key in ('normalized_full_text_duplicates', 'normalized_atomic_semantic_duplicates',
                'literal_schema_name_hits', 'complete_historical_phrase_inclusions',
                'exact_C8_training_validation_phrase_collisions'):
        require(not audit.get('lexical_checks', {}).get(key), 'final corpus audit lexical exclusion ' + key)
    for key in ('normalized_duplicates', 'normalized_atomic_semantic_duplicates',
                'literal_schema_name_hits', 'byte_exact_reference_matches',
                'normalized_reference_phrase_matches', 'complete_old_phrase_inclusions'):
        require(not training_audit.get('lexical_checks', {}).get(key),
                'training corpus audit lexical exclusion ' + key)
    packet_path = root / 'blind_meaning_questions.json'
    packet = load(packet_path)
    questions = packet.get('questions', [])
    question_by_id = {row.get('review_id'): row for row in questions}
    review_ids = load(root / 'blind_meaning_review_ids.json')
    blind_key = load(root / 'blind_meaning_key.json')
    require(set(packet) == {'schema_version', 'instructions', 'axis_order', 'field_definitions',
                            'response_schema', 'questions'} and
            packet.get('schema_version') == 'cbf8-blind-meaning-review-v1' and
            packet.get('axis_order') == list(SCHEMA) and
            packet.get('field_definitions') == cfg['field_definitions'] and
            isinstance(packet.get('instructions'), list) and len(packet['instructions']) == 6,
            'sealed opaque blind-review packet schema and fixed definitions')
    require(len(questions) == len(question_by_id) == len(review_ids) == 352 and
            all(set(row) == {'review_id', 'text'} and isinstance(row['text'], str) and
                re.fullmatch(r'[0-9a-f]{32}', row['review_id']) for row in questions) and
            review_ids == sorted(question_by_id) and set(blind_key) == set(question_by_id),
            'sealed blind question/review-ID/key coverage')
    review_rows = (
        [('semantic_training', row) for row in groups['semantic_training_atoms']] +
        [('semantic_validation', row) for row in groups['semantic_validation_atoms']] +
        [('final_atoms', row) for row in groups['final_atoms']] +
        [('final_compositions', row) for row in groups['final_compositions']]
    )
    by_text = {row['text']: (cohort, row) for cohort, row in review_rows}
    require(len(by_text) == 352 and {question['text'] for question in questions} == set(by_text),
            'blind packet includes every new training, validation, atom, and composition meaning')
    expected_blind_key = {}
    for question in questions:
        cohort, row = by_text[question['text']]
        expected = dict(weights=row['weights'])
        if 'axis' in row:
            expected.update(axis=row['axis'], sign=row['sign'])
        if 'components' in row:
            final_by_id = {atom['id']: atom for atom in groups['final_atoms']}
            expected['components'] = [
                dict(axis=final_by_id[component['atom_id']]['axis'],
                     sign=final_by_id[component['atom_id']]['sign'], factor=component['factor'])
                for component in row['components']
            ]
        expected_blind_key[question['review_id']] = dict(cohort=cohort, record_id=row['id'], expected=expected)
    require(blind_key == expected_blind_key, 'sealed blind key complete cohort/label/component match')
    predictions_path = root / 'blind_meaning_predictions.json'
    require(predictions_path.is_file() and meaning.get('all_meanings_accepted') is True and
            meaning.get('final_atoms_sha256') == sha(root / 'final_atoms.json') and
            meaning.get('final_compositions_sha256') == sha(root / 'final_compositions.json') and
            meaning.get('blind_predictions_sha256') == sha(predictions_path),
            'independent final meaning audit and blind-prediction hashes')
    predictions = load(predictions_path)
    require(isinstance(predictions, dict) and
            set(predictions) == {'schema_version', 'packet_sha256', 'judgments'},
            'strict blind-prediction top-level schema')
    judgments = predictions.get('judgments', []) if isinstance(predictions, dict) else []
    judgment_by_id = {row.get('review_id'): row for row in judgments}
    require(predictions.get('schema_version') == 'cbf8-blind-meaning-review-v1' and
            predictions.get('packet_sha256') == sha(packet_path) and
            len(judgments) == len(judgment_by_id) == 352 and
            set(judgment_by_id) == set(expected_blind_key),
            'independent predictions bind the exact complete blind packet')
    for review_id, judgment in judgment_by_id.items():
        entry = expected_blind_key[review_id]
        expected = entry['expected']
        common = {'review_id', 'decision', 'kind'}
        require(judgment.get('decision') == 'accept', 'blind review has ambiguous/rejected phrase')
        if 'axis' in expected:
            require(set(judgment) == common | {'axis', 'sign'} and judgment.get('kind') == 'atomic' and
                    type(judgment.get('axis')) is int and type(judgment.get('sign')) is int and
                    judgment.get('axis') == expected['axis'] and judgment.get('sign') == expected['sign'],
                    'blind atomic meaning matches sealed key')
        else:
            expected_components = [dict(axis=row['axis'], sign=row['sign'])
                                   for row in expected['components']]
            require(set(judgment) == common | {'components', 'weights'} and
                    judgment.get('kind') == 'composition' and
                    judgment.get('components') == expected_components and
                    all(type(term['axis']) is int and type(term['sign']) is int
                        for term in judgment['components']) and
                    all(type(weight) is int for weight in judgment['weights']) and
                    judgment.get('weights') == expected['weights'],
                    'blind composition meaning/components/weights match sealed key')
    require(meaning['assembler_sha256'] == manifest['preparation_source_sha256']['schema_support_meaning.py'],
            'exact audit assembler source hash')
    raw_sources = meaning['raw_judgment_sources']
    require(len(raw_sources) == 2 and sum(source['rows'] for source in raw_sources) == 352,
            'complete two-slice raw independent judgments')
    raw_rows = []
    for source_record in raw_sources:
        raw_path = Path(source_record['path']).resolve()
        raw_path.relative_to(root.resolve())
        require(sha(raw_path) == source_record['sha256'], 'raw independent judgment bytes')
        raw = load(raw_path)
        require(set(raw) == {'schema_version', 'packet_sha256', 'judgments'} and
                raw['schema_version'] == predictions['schema_version'] and
                raw['packet_sha256'] == predictions['packet_sha256'] and
                len(raw['judgments']) == source_record['rows'], 'raw judgment packet binding')
        raw_rows.extend(raw['judgments'])
    require(len({row['review_id'] for row in raw_rows}) == 352 and
            {row['review_id'] for row in raw_rows} == set(judgment_by_id), 'raw judgment exact ID coverage')
    review_texts = {row['review_id']: row['text'] for row in load(packet_path)['questions']}
    vector_differences = []
    for raw in raw_rows:
        rendered = judgment_by_id[raw['review_id']]
        require(raw['decision'] == 'accept', 'raw ambiguous/rejected judgment cannot be accepted')
        if raw['kind'] == 'atomic':
            require(raw == rendered and type(raw['axis']) is int and type(raw['sign']) is int,
                    'atomic semantic judgment remains untouched')
        else:
            terms = parse_criterion(review_texts[raw['review_id']])
            require(len(terms) == len(raw['components']) == 2, 'exact audit clause count')
            weights = [0] * 4
            for term, component in zip(terms, raw['components']):
                require(type(component['axis']) is int and type(component['sign']) is int and
                        term['literal_axis'] is None and type(term['factor']) is int,
                        'strict raw semantic types and exact factors')
                weights[component['axis']] += term['factor'] * component['sign']
            require(rendered == dict(raw, weights=weights), 'canonical weights derived without sealed gold')
            if raw['weights'] != weights:
                vector_differences.append(dict(review_id=raw['review_id'], raw_weights=raw['weights'],
                                              exact_weights=weights))
    require(meaning['raw_vector_differences'] == vector_differences, 'raw arithmetic differences disclosed')
    proof = dict(copied_cbf7_cohorts=7, semantic_training_atoms=64, semantic_validation_atoms=32,
                 final_atoms=128, final_compositions=128, training_reversal_pairs=len(train_pairs),
                 semantic_validation_reversal_pairs=len(validation_pairs), final_reversal_pairs=len(final_pairs),
                 final_component_use_counts=dict(composition_uses), final_composition_patterns=len(coverage),
                 integer_states=64, blind_meanings_accepted=True, training_audit=training_proof,
                 compiler_reconstruction=compiler_proof, historical_reference_sha256=historical_hashes,
                 source_lineage_sha256=dict(final_corpus_audit=sha(root / 'final_corpus_audit.json'),
                                            training_corpus_audit=sha(root / 'training_corpus_audit.json')))
    return groups, proof, source, source_verify, source_hashes


def verify_manifest(root, cfg):
    manifest = load(root / 'corpus_manifest.json')
    require(manifest.get('experiment') == 'CBF8' and manifest.get('protocol_sha256') == sha(PROTOCOL) and
            manifest.get('source_root') == cfg['source_root'] and
            manifest.get('output_root') == cfg['output_root'], 'corpus manifest identity/protocol/source/output')
    expected_files = set(COHORTS) | {
        'training_corpus_audit.json', 'final_corpus_audit.json', 'compiler_checks.json',
        'blind_meaning_questions.json', 'blind_meaning_review_ids.json',
        'blind_meaning_key.json', 'baseline_references.json',
    }
    files = manifest.get('files')
    require(isinstance(files, dict) and set(files) == expected_files, 'exact immutable preparation file inventory')
    for name, expected in files.items():
        require(sha(root / name) == expected, 'corpus manifest file hash ' + name)
    require(manifest.get('final_closed_until_selection') is True and
            manifest.get('meaning_audit') == dict(
                required=True, supplied_by='independent blind auditor', created_by_preparation=False,
                before_model_execution=True,
                review_cohorts=['semantic_training', 'semantic_validation', 'final_atoms', 'final_compositions']),
            'final sealing and independent meaning-audit contract')
    count_names = {
        'training_atoms.json': 'literal_training', 'validation_atoms.json': 'literal_validation',
        'literal_holdout_atoms.json': 'literal_holdout', 'semantic_training_atoms.json': 'semantic_training',
        'semantic_validation_atoms.json': 'semantic_validation', 'development_atoms.json': 'development_atoms',
        'development_compositions.json': 'development_compositions', 'literal_cases.json': 'literal_cases',
        'states.json': 'states', 'final_atoms.json': 'final_atoms', 'final_compositions.json': 'final_compositions',
    }
    counts = manifest.get('counts', {})
    require(set(counts) == set(count_names.values()) | {
        'final_atom_reversal_pairs', 'final_composition_reversal_pairs', 'blind_review_rows'},
        'corpus count key inventory')
    for name, key in count_names.items():
        count = len(load(root / name))
        require(counts.get(key) == count, 'corpus count mismatch ' + key)
    require(counts['final_atom_reversal_pairs'] == 64 and counts['final_composition_reversal_pairs'] == 64 and
            counts['blind_review_rows'] == 352, 'final/meaning-review row counts')
    require(manifest.get('fixed_field_definitions') == cfg['field_definitions'] and
            manifest.get('field_definitions_sha256') ==
            sha_bytes((json.dumps(cfg['field_definitions'], sort_keys=True, indent=2, ensure_ascii=False) +
                       '\n').encode('utf-8')), 'frozen field-definition hash')
    preparation_sources = manifest.get('preparation_source_sha256', {})
    compare(load(root / 'baseline_references.json'), manifest.get('baseline_references'),
            'persisted CBF7 baseline-reference map')
    expected_sources = (
        'schema_support_prepare.py', 'schema_support_encoder.py', 'schema_support_run.py',
        'schema_support_training_corpus.py', 'schema_support_final_corpus.py',
        'schema_support_meaning.py',
        'schema_support_protocol.json', 'schema_grounding_compiler.py', 'schema_relation_evaluate.py',
        'relation_pretrained_run.py', 'relation_pretrained_encoder.py', 'audit.py', 'build.py')
    require(set(preparation_sources) == set(expected_sources), 'preparation source inventory')
    for name, expected in preparation_sources.items():
        require(sha(PROTOCOL.with_name(name)) == expected, 'preparation source hash ' + name)
    packet = load(root / 'blind_meaning_questions.json')
    review_ids = load(root / 'blind_meaning_review_ids.json')
    expected_review = dict(
        schema_version=packet['schema_version'], question_file='blind_meaning_questions.json',
        packet_sha256=sha(root / 'blind_meaning_questions.json'),
        review_ids_file='blind_meaning_review_ids.json',
        review_ids_sha256=sha(root / 'blind_meaning_review_ids.json'),
        key_file='blind_meaning_key.json', key_sha256=sha(root / 'blind_meaning_key.json'),
        predictions_file='blind_meaning_predictions.json', expected_questions=352)
    compare(manifest.get('blind_review'), expected_review, 'manifest blind review packet/IDs/key hashes')
    require(isinstance(review_ids, list) and len(review_ids) == len(set(review_ids)) == 352,
            'manifest blind review ID inventory')
    revision = manifest.get('preparation_git_revision')
    require(isinstance(revision, str) and re.fullmatch(r'[0-9a-f]{40}', revision),
            'preparation Git revision')
    return manifest


def model_features(features, evidence, texts):
    require(features.shape[0] == len(evidence['pairs']), 'feature/evidence row count')
    lookup = {(p['criterion'], p['field']): i for i, p in enumerate(evidence['pairs'])}
    require(len(lookup) == len(evidence['pairs']), 'duplicate cache key')
    return features[[lookup[(text, field)] for text in texts for field in SCHEMA]]


def evidence_by_text(evidence):
    return {(pair['criterion'], pair['field']): i for i, pair in enumerate(evidence['pairs'])}


def validate_feature_cache(root, features, evidence, texts, label):
    expected = [(text, field) for text in texts for field in SCHEMA]
    actual = [(pair['criterion'], pair['field']) for pair in evidence['pairs']]
    require(actual == expected, label + ': canonical sorted question/field cache order')
    finite_array(features, label)
    require(features.shape == (4 * len(texts), 384), label + ': feature dimensions')
    require(len(evidence_by_text(evidence)) == len(expected), label + ': unique evidence rows')
    return dict(rows=len(expected), feature_sha256=array_sha(features))


def verify_cache_manifest(root, name, evidence, features, fmt, cache_key, encoder_hash, cfg):
    manifest = load(root / name)
    feature_name = {
        'generic': 'features_generic.npy', 'defined': 'features_defined.npy',
    }.get(cache_key, 'features_final_' + cache_key.removeprefix('final_') + '.npy')
    input_name = {
        'generic': 'inputs_generic.json', 'defined': 'inputs_defined.json',
    }.get(cache_key, 'inputs_final_' + cache_key.removeprefix('final_') + '.json')
    expected = dict(
        format=fmt, cache_key=cache_key, feature=feature_name, input=input_name,
        input_sha256=sha(root / input_name), feature_sha256=sha(root / feature_name),
        model_repo=cfg['model']['repo'], model_revision=cfg['model']['revision'],
        model_weight_sha256=cfg['model']['weight_sha256'],
        encoder_parameters_sha256=encoder_hash, shape=list(features.shape), dtype='float32',
        raw_final_layer_cls=True, base_model_last_hidden_state=True,
        classifier_invoked=False, pooler_invoked=False, truncation=False,
        all_tokenizer_ids_saved=True, padding_bytes_saved=True)
    compare(manifest, expected, name + ': exact cache manifest')
    require(manifest['input_sha256'] == sha(root / input_name) and
            manifest['feature_sha256'] == sha(root / feature_name) and
            evidence['pair_count'] == features.shape[0], name + ': cache input/feature/pair-count binding')
    return dict(file=name, sha256=sha(root / name), rows=int(features.shape[0]))


def head_values(checkpoint, width=384):
    require(isinstance(checkpoint, dict) and set(checkpoint) == {'weight', 'bias'}, 'simple linear head state dict')
    weights = {key: value.detach().cpu().numpy() for key, value in checkpoint.items()}
    require(weights['weight'].shape == (3, width) and weights['bias'].shape == (3,), 'linear head dimensions')
    require(all(value.dtype == np.float32 and np.isfinite(value).all() for value in weights.values()),
            'head dtype/finite values')
    return weights


def probability_table(features, normalizer, weights):
    require(features.ndim == 2 and features.shape[1] == weights['weight'].shape[1], 'head feature width')
    mean, std = normalizer['mean'], normalizer['std']
    require(mean.shape == std.shape == (features.shape[1],) and np.isfinite(mean).all() and
            np.isfinite(std).all() and np.all(std > 0), 'normalizer dimensions/finite')
    # The inference contract first performs normalized head inputs in FP32, then replay accumulates in FP64.
    values = ((features - mean) / std).astype(np.float32).astype(np.float64)
    logits = values @ weights['weight'].astype(np.float64).T + weights['bias'].astype(np.float64)
    exponent = np.exp(logits - logits.max(1, keepdims=True))
    return exponent / exponent.sum(1, keepdims=True), logits


def decode(matrix):
    p = np.asarray(matrix, dtype=np.float64)
    require(p.shape == (4, 3) and np.isfinite(p).all() and np.all(p >= 0) and
            np.allclose(p.sum(1), 1, atol=1e-6), 'invalid four-field uncalibrated masses')
    classes = p.argmax(1) - 1
    active = np.flatnonzero(classes)
    if not len(active):
        return None
    evidence = np.maximum(p[:, 0], p[:, 2]) - p[:, 1]
    axis = int(max(active, key=lambda i: (evidence[i], -i)))
    return dict(axis=axis, sign=int(classes[axis]), relation_matrix=p.tolist(),
                row_classes=classes.tolist(), nonzero_fields=int(len(active)))


def checkpoint_details(root, arm, cache, evidence, groups, cfg, source_baseline):
    if arm == 'literal_generic':
        source = source_baseline['root']
        checkpoint_path = source / 'nli_xsmall_higher_cls_checkpoint.pt'
        training_path = source / 'nli_xsmall_higher_cls_training.json'
        normalizer_path = source / 'normalizer_nli_xsmall_higher_cls.npz'
    else:
        checkpoint_path = root / (arm + '_checkpoint.pt')
        training_path = root / (arm + '_training.json')
        normalizer_path = root / ('normalizer_' + arm + '.npz')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    weights = head_values(checkpoint)
    normalizer = read_npz(normalizer_path)
    require(np.isfinite(normalizer['mean']).all() and np.isfinite(normalizer['std']).all() and
            np.all(normalizer['std'] > 0), arm + ': valid mean/std')
    details = load(training_path)
    if arm == 'literal_generic':
        require(details.get('arm_id') == 'nli_xsmall_higher_cls' and details.get('kind') == 'linear',
                'CBF7 baseline training record')
        train_e = load(source / 'inputs_nli_xsmall_higher.json')
        train_h = np.load(source / 'features_nli_xsmall_higher_cls.npy', mmap_mode='r', allow_pickle=False)
        finite_array(train_h, 'CBF7 baseline persisted CLS features')
        train_rows = [evidence_by_text(train_e)[(c['text'], field)]
                      for c in groups['training_atoms'] for field in SCHEMA]
        Htrain = train_h[train_rows]
        require(len(train_rows) == 128 and normalizer['mean'].dtype == normalizer['std'].dtype == np.float32 and
                np.array_equal(Htrain.mean(0), normalizer['mean']) and
                np.array_equal(np.maximum(Htrain.std(0), .01), normalizer['std']),
                'CBF7 baseline ORIGINAL literal training row order normalizer')
        normalizer_meta = details.get('normalizer', {})
        require(normalizer_meta.get('arm_id') == 'nli_xsmall_higher_cls' and
                normalizer_meta.get('train_only') is True and normalizer_meta.get('floor') == .01 and
                normalizer_meta.get('mean_sha256') == array_sha(normalizer['mean']) and
                normalizer_meta.get('std_sha256') == array_sha(normalizer['std']),
                'CBF7 baseline normalizer metadata/hash')
        trace = details.get('trace')
        require(isinstance(trace, list) and len(trace) == cfg['training']['epochs'] == 400 and
                [item.get('epoch') for item in trace] == list(range(400)) and
                all(math.isfinite(item.get(key, float('nan')))
                    for item in trace for key in ('loss', 'CE', 'validation_CE')),
                'CBF7 baseline finite 400-epoch trace')
        selected = min(range(len(trace)), key=lambda index: trace[index]['validation_CE'])
        require(details.get('selected_epoch') == selected and
                details.get('validation_CE') == trace[selected]['validation_CE'] and
                details.get('seed') == cfg['training']['seed'] and details.get('epochs') == 400 and
                details.get('optimizer') == 'AdamW' and details.get('lr') == cfg['training']['lr'] and
                details.get('weight_decay') == cfg['training']['weight_decay'] and
                details.get('first_gradient_L1', 0) > 0,
                'CBF7 baseline earlier-tie checkpoint/recipe/gradient')
        val_H = model_features(train_h, train_e, [atom['text'] for atom in groups['validation_atoms']])
        val_labels = np.asarray([atom['sign'] + 1 if field_index == atom['axis'] else 1
                                 for atom in groups['validation_atoms'] for field_index in range(4)],
                                dtype=np.int64)
        _, validation_logits = probability_table(val_H, normalizer, weights)
        validation_ce = float(np.mean(logsumexp(validation_logits, axis=1) -
                                      validation_logits[np.arange(len(val_labels)), val_labels]))
        require(abs(validation_ce - details['validation_CE']) < 2e-5,
                'CBF7 independently reconstructed literal validation CE')
        torch.manual_seed(cfg['training']['seed'])
        initial = torch.nn.Linear(384, 3, dtype=torch.float32)
        x = torch.tensor((Htrain - normalizer['mean']) / normalizer['std'], dtype=torch.float32)
        y = torch.tensor([atom['sign'] + 1 if field_index == atom['axis'] else 1
                          for atom in groups['training_atoms'] for field_index in range(4)], dtype=torch.long)
        class_weights = torch.tensor(cfg['training']['class_weights'], dtype=torch.float32)
        torch.nn.functional.cross_entropy(initial(x), y, weight=class_weights).backward()
        first_gradient = float(sum(parameter.grad.abs().sum().item() for parameter in initial.parameters()))
        require(first_gradient > 0 and abs(first_gradient - details['first_gradient_L1']) < 2e-5,
                'CBF7 independently reconstructed first head gradient')
        training_proof = dict(kind='reused_CBF7_raw_higher_CLS', normalizer_original_training_order=True,
                              unweighted_literal_validation_CE=validation_ce,
                              selected_epoch=selected, first_gradient_L1=first_gradient,
                              feature_sha256=sha(source / 'features_nli_xsmall_higher_cls.npy'),
                              checkpoint_sha256=sha(checkpoint_path), normalizer_sha256=sha(normalizer_path))
        require(details.get('frozen_encoder_before_sha256') == details.get('frozen_encoder_after_sha256'),
                'CBF7 baseline full frozen hash equality')
        return weights, normalizer, details, training_proof, checkpoint_path, train_e, train_h

    train_atoms = list(groups['training_atoms'])
    if arm.startswith('semantic_'):
        train_atoms += list(groups['semantic_training_atoms'])
    train_rows = [index for atom in train_atoms for index in (evidence_by_text(evidence)[(atom['text'], field)] for field in SCHEMA)]
    Htrain = cache[train_rows]
    mean = Htrain.mean(0)
    std = np.maximum(Htrain.std(0), .01)
    require(np.array_equal(normalizer['mean'], mean) and np.array_equal(normalizer['std'], std),
            arm + ': normalizer from exact original train order')
    normalizer_meta = details.get('normalizer', {})
    require(normalizer_meta.get('mean_sha256') == array_sha(mean) and normalizer_meta.get('std_sha256') == array_sha(std) and
            normalizer_meta.get('floor') == .01 and normalizer_meta.get('train_only') is True,
            arm + ': truthful train-only normalizer metadata')
    require(details.get('normalizer_npz_sha256') == sha(normalizer_path) and
            details.get('normalizer_file') == normalizer_path.name and
            details.get('normalizer_mean_sha256') == array_sha(mean) and
            details.get('normalizer_std_sha256') == array_sha(std),
            arm + ': NPZ/array hashes and filenames')
    require(details.get('arm_id') == arm and details.get('kind') == 'linear' and details.get('width') == 384 and
            details.get('trainable_parameters') == 3 * 384 + 3 and details.get('floating_parameters') == 70831107 and
            details.get('feature_sha256') == sha(root / ('features_' + arm.split('_', 1)[1] + '.npy')) and
            details.get('evidence_sha256') == sha(root / ('inputs_' + arm.split('_', 1)[1] + '.json')) and
            details.get('checkpoint_file') == checkpoint_path.name and
            details.get('checkpoint_sha256') == sha(checkpoint_path),
            arm + ': head/model/cache/checkpoint accounting')
    expected_semantic_count = 64 if arm.startswith('semantic_') else 0
    require(details.get('literal_training_atoms') == 32 and
            details.get('semantic_training_atoms') == expected_semantic_count and
            details.get('validation_atoms') == 32 and details.get('semantic_validation_used') is False and
            details.get('development_or_G0_or_final_used_for_training') is False,
            arm + ': exact training/diagnostic scopes')
    trace = details.get('trace')
    require(isinstance(trace, list) and len(trace) == cfg['training']['epochs'] == 400 and
            [r.get('epoch') for r in trace] == list(range(400)), arm + ': exact 400-epoch history')
    for index, item in enumerate(trace):
        require(all(math.isfinite(item.get(key, float('nan'))) for key in ('loss', 'CE', 'validation_CE')),
                arm + ': finite history entry ' + str(index))
    selected_epoch = min(range(len(trace)), key=lambda i: trace[i]['validation_CE'])
    require(details.get('selected_epoch') == selected_epoch and
            details.get('validation_CE') == trace[selected_epoch]['validation_CE'],
            arm + ': earlier-tie minimum literal validation CE')
    require(details.get('seed') == cfg['training']['seed'] and details.get('epochs') == 400 and
            details.get('optimizer') == 'AdamW' and details.get('lr') == cfg['training']['lr'] and
            details.get('weight_decay') == cfg['training']['weight_decay'], arm + ': frozen recipe')
    grad = details.get('gradient_proof', {})
    require(grad.get('head_l1', 0) > 0 and grad.get('encoder_l1') == 0.0 and
            grad.get('frozen_gradients') is True and grad.get('optimizer_parameters') == 'head_only' and
            details.get('first_gradient_L1') == grad.get('head_l1') and
            details.get('encoder_parameter_gradient_count') == 0,
            arm + ': positive head gradient and strictly frozen encoder')
    before, after = hash_pair(details)
    require(isinstance(before, str) and re.fullmatch(r'[0-9a-f]{64}', before) and before == after,
            arm + ': full frozen model hash unchanged')
    validation_h = model_features(cache, evidence, [a['text'] for a in groups['validation_atoms']])
    validation_rows = np.arange(len(groups['validation_atoms']) * 4)
    validation_labels = np.asarray([a['sign'] + 1 if field_index == a['axis'] else 1
                                    for a in groups['validation_atoms']
                                    for field_index in range(4)], dtype=np.int64)
    _, validation_logits = probability_table(validation_h, normalizer, weights)
    ce = float(np.mean(logsumexp(validation_logits, axis=1) -
                       validation_logits[validation_rows, validation_labels]))
    require(abs(ce - details['validation_CE']) < 2e-5, arm + ': independent literal-validation CE')
    # Recompute the initial full-batch head gradient from the seed and persisted rows.
    torch.manual_seed(cfg['training']['seed'])
    initial = torch.nn.Linear(384, 3, dtype=torch.float32)
    x = torch.tensor(((Htrain - mean) / std), dtype=torch.float32)
    y = torch.tensor([atom['sign'] + 1 if field_index == atom['axis'] else 1
                      for atom in train_atoms for field_index in range(4)], dtype=torch.long)
    weights_t = torch.tensor(cfg['training']['class_weights'], dtype=torch.float32)
    initial.zero_grad(set_to_none=True)
    torch.nn.functional.cross_entropy(initial(x), y, weight=weights_t).backward()
    initial_gradient = float(sum(parameter.grad.abs().sum().item() for parameter in initial.parameters()))
    require(initial_gradient > 0 and abs(initial_gradient - details['first_gradient_L1']) < 2e-5,
            arm + ': independently reconstructed initial head gradient')
    training_proof = dict(normalizer_original_training_order=True, train_pairs=len(train_rows),
                          selected_epoch=selected_epoch, unweighted_literal_validation_CE=ce,
                          validation_CE_error=abs(ce - details['validation_CE']),
                          first_gradient_L1=initial_gradient, frozen_parameter_sha256=before,
                          checkpoint_sha256=sha(checkpoint_path), normalizer_sha256=sha(normalizer_path))
    return weights, normalizer, details, training_proof, checkpoint_path, evidence, cache


def verify_head_selection_integrity(root, arm, details, cfg):
    if arm == 'literal_generic':
        return
    trace = details['trace']
    selected = min(range(len(trace)), key=lambda i: trace[i]['validation_CE'])
    require(details['selected_epoch'] == selected, arm + ': selection must use only literal validation trace')
    require(details.get('semantic_validation_used_for_selection') is False and
            details.get('semantic_validation_used_for_checkpoint') is False and
            details.get('development_used_for_checkpoint') is False and
            details.get('final_used_for_checkpoint') is False,
            arm + ': nonliteral/checkpoint leakage flags')


def summary(rows):
    require(rows, 'empty decision summary')
    return dict(n=len(rows), **{key: float(np.mean([row[key] for row in rows]))
                                for key in ('top1', 'exact_winner', 'pairwise')},
                coverage=float(np.mean([row['covered'] for row in rows])))


def verify_stage(root, tag, atoms, compositions, literals, states, H, evidence, weights, normalizer, g0, cfg):
    output = root / tag
    texts = row_texts(atoms, compositions, g0)
    features = model_features(H, evidence, texts)
    probabilities, _ = probability_table(features, normalizer, weights)
    matrices = {text: probabilities[4 * index:4 * index + 4] for index, text in enumerate(texts)}
    persisted = load(output / 'relation_matrices.json')
    require([row.get('text') for row in persisted] == texts, tag + ': complete canonical probability table')
    resolutions = {}
    persisted_resolutions = {}
    maximum_error = 0.0
    matrix_records = []
    for record in persisted:
        require(record.get('fields') == list(SCHEMA) and record.get('classes') == [-1, 0, 1],
                tag + ': matrix schema/classes')
        saved = np.asarray(record['matrix'], dtype=np.float64)
        expected = matrices[record['text']]
        require(saved.shape == (4, 3) and np.isfinite(saved).all() and np.all(saved >= 0) and
                np.allclose(saved.sum(1), 1, atol=1e-6), tag + ': valid raw class masses')
        error = float(np.max(np.abs(saved - expected)))
        require(error < 2e-5, tag + ': probability table mismatch ' + str(error))
        maximum_error = max(maximum_error, error)
        resolution = decode(saved)
        compare(record.get('resolution'), resolution, tag + ': saved decoder output', 2e-5)
        reconstructed = decode(expected)
        require((None if resolution is None else (resolution['axis'], resolution['sign'])) ==
                (None if reconstructed is None else (reconstructed['axis'], reconstructed['sign'])),
                tag + ': UNKNOWN/axis/sign decode')
        resolutions[record['text']] = reconstructed
        persisted_resolutions[record['text']] = resolution
        matrix_records.append(dict(text=record['text'], matrix=expected.tolist(), resolution=reconstructed))

    axis_correct = sum(resolutions[a['text']] is not None and resolutions[a['text']]['axis'] == a['axis'] for a in atoms)
    joint_correct = sum(resolutions[a['text']] is not None and
                        (resolutions[a['text']]['axis'], resolutions[a['text']]['sign']) == (a['axis'], a['sign'])
                        for a in atoms)
    atomic = dict(n=len(atoms), axis_correct=axis_correct, axis_accuracy=axis_correct / len(atoms),
                  sign_given_axis_correct=joint_correct, sign_given_axis_denominator=axis_correct,
                  conditional_sign_accuracy=joint_correct / axis_correct if axis_correct else None,
                  joint_correct=joint_correct, joint_accuracy=joint_correct / len(atoms),
                  unknown=sum(resolutions[a['text']] is None for a in atoms))
    gold_relations = np.asarray([[row['sign'] if field_index == row['axis'] else 0
                                  for field_index in range(4)] for row in g0], dtype=np.int64)
    predicted_relations = np.asarray([matrices[row['text']].argmax(1) - 1 for row in g0], dtype=np.int64)
    equal = gold_relations == predicted_relations
    matched = gold_relations != 0
    relation = dict(n=int(equal.size), correct=int(equal.sum()), accuracy=float(equal.mean()),
                    matched_n=int(matched.sum()), matched_accuracy=float(equal[matched].mean()),
                    per_class={str(value): dict(n=int((gold_relations == value).sum()),
                                                accuracy=float(equal[gold_relations == value].mean()))
                               for value in (-1, 0, 1)})

    queries = [dict(row, stratum='alias') for row in atoms] + \
              [dict(row, stratum='alias_composition') for row in compositions] + list(literals)
    saved_compiled = load(output / 'compiled_queries.json')
    require([row.get('id') for row in saved_compiled] == [row['id'] for row in queries],
            tag + ': compiled query completeness/order')
    vectors = {}
    expected_compiled = []
    semantic_callbacks = literal_callbacks = 0
    for query, saved in zip(queries, saved_compiled):
        vector = [0, 0, 0, 0]
        unknown = False
        records = []
        for term in parse_criterion(query['text']):
            lo, hi = term['span']
            phrase = query['text'][lo:hi]
            if term['literal_axis'] is None:
                require(query['stratum'] != 'literal', tag + ': literal query requested neural parse')
                semantic_callbacks += 1
                resolution = persisted_resolutions[field_context(query['text'], term)]
            else:
                resolution = dict(axis=term['literal_axis'], sign=term['literal_sign'])
            if resolution is None:
                unknown = True
            else:
                vector[resolution['axis']] += term['factor'] * resolution['sign']
            records.append(dict(term, term=phrase, resolution=resolution))
        expected_vector = None if unknown else vector
        expected_ir = dict(id=query['id'], stratum=query['stratum'], vector=expected_vector,
                           coefficient_exact=expected_vector == query['weights'], atoms=records)
        compare(saved, expected_ir, tag + ': exact compiler AST/vector/UNKNOWN')
        vectors[query['id']] = expected_vector
        expected_compiled.append(expected_ir)
        if query['stratum'] == 'literal':
            def forbidden(_term):
                nonlocal literal_callbacks
                literal_callbacks += 1
                raise AssertionError('literal callback invoked')
            exact, _ = compile_criterion(query['text'], forbidden)
            require(exact == query['weights'] == expected_vector, tag + ': exact literal compiler')
    require(literal_callbacks == 0, tag + ': literal callbacks')

    test_states = [state for state in states if state['source_split'] != 'train']
    require(len(test_states) == 32, tag + ': test state count')
    rows = []
    cached = {}
    permutation = dict(comparisons=0, numeric_comparisons=0, unknown_comparisons=0,
                       score_mismatches=0, winner_mismatches=0)
    rng = np.random.default_rng(7)
    with (output / 'decisions.jsonl').open() as stream:
        for query in queries:
            direction = vectors[query['id']]
            for state in test_states:
                actual_line = stream.readline()
                require(bool(actual_line), tag + ': missing decision row')
                facts = np.asarray(state['parsed_facts'], dtype=np.int64)
                gold = facts @ np.asarray(query['weights'], dtype=np.int64)
                teacher = max(range(len(facts)), key=lambda i: (int(gold[i]), tuple(facts[i])))
                values = None if direction is None else exact_score(direction, state['candidates'])
                selected = exact_winner(values, state['parsed_facts'])
                pair_i, pair_j = np.triu_indices(state['K'], 1)
                actual = json.loads(actual_line)
                expected = dict(criterion_id=query['id'], stratum=query['stratum'], scenario_id=state['id'],
                                source_split=state['source_split'], K=state['K'], teacher=teacher,
                                teacher_scores=gold.tolist(), teacher_top_set=np.flatnonzero(gold == gold.max()).tolist(),
                                winner=selected, scores=None if values is None else values.tolist(), covered=selected is not None,
                                top1=int(selected is not None and gold[selected] == gold.max()),
                                exact_winner=int(selected == teacher),
                                pairwise=0.0 if values is None else
                                float(np.mean(np.sign(values[pair_i] - values[pair_j]) ==
                                               np.sign(gold[pair_i] - gold[pair_j]))))
                compare(actual, expected, tag + ': exact integer-score decision')
                rows.append(expected)
                cached[(query['id'], state['id'])] = expected
                for order in (np.arange(state['K'])[::-1], rng.permutation(state['K'])):
                    permutation['comparisons'] += 1
                    if values is None:
                        permutation['unknown_comparisons'] += 1
                        continue
                    permuted_values = exact_score(direction, [state['candidates'][j] for j in order])
                    permuted_winner = exact_winner(permuted_values, facts[order])
                    restored = int(order[permuted_winner]) if permuted_winner is not None else None
                    permutation['numeric_comparisons'] += 1
                    permutation['winner_mismatches'] += int(restored != selected)
                    permutation['score_mismatches'] += int(not np.array_equal(permuted_values[np.argsort(order)], values))
        require(not stream.readline(), tag + ': extra decisions')

    def included(row, stratum):
        return row['stratum'] == stratum and (stratum == 'literal' or row['source_split'] ==
                                              ('unseen_wording' if stratum == 'alias' else 'unseen_criterion'))
    strata = ('alias', 'alias_composition', 'literal')
    summaries = {name: summary([row for row in rows if included(row, name)]) for name in strata}
    per_k = {name: {str(k): summary([row for row in rows if included(row, name) and row['K'] == k])
                    for k in (2, 4, 8, 16)} for name in strata}
    representatives = {}
    for query in sorted((query for query in queries if query['stratum'] != 'literal'), key=lambda row: row['text']):
        divisor = math.gcd(*(abs(int(weight)) for weight in query['weights']))
        require(divisor > 0, tag + ': zero teacher vector')
        ray = tuple(int(weight) // divisor for weight in query['weights'])
        representatives.setdefault(ray, query)
    reps = list(representatives.values())
    rep_records = [dict(id=query['id'], text=query['text'], weights=query['weights']) for query in reps]
    compare(load(output / 'causal_representatives.json'), rep_records, tag + ': exact causal representatives')
    causal_counts = Counter()
    eligible = 0
    causal_n = 0
    with (output / 'teacher_changing_swaps.jsonl').open() as stream:
        for state in test_states:
            for first, second in itertools.permutations(reps, 2):
                eligible += 1
                before = cached[(first['id'], state['id'])]
                after = cached[(second['id'], state['id'])]
                if before['teacher'] == after['teacher']:
                    continue
                a, b = before['winner'], after['winner']
                flags = dict(correct_new=int(b == after['teacher']), student_changed=int(a != b),
                             changed_to_new=int(b == after['teacher'] and a is not None and b is not None and a != b),
                             both_endpoints=int(a == before['teacher'] and b == after['teacher']),
                             top_set_new=after['top1'])
                expected = dict(scenario_id=state['id'], before=first['id'], after=second['id'],
                                teacher_before=before['teacher'], teacher_after=after['teacher'], **flags)
                line = stream.readline()
                require(bool(line), tag + ': missing teacher-changing swap')
                compare(json.loads(line), expected, tag + ': causal row')
                causal_counts.update(flags)
                causal_n += 1
        require(not stream.readline(), tag + ': extra teacher-changing swaps')
    require(causal_n > 0, tag + ': empty causal denominator')
    causal = dict(representatives=len(reps), eligible_pairs=eligible, n=causal_n,
                  **{key + '_rate': causal_counts[key] / causal_n for key in
                     ('correct_new', 'student_changed', 'changed_to_new', 'both_endpoints', 'top_set_new')})
    thresholds = cfg['gates']
    gates = dict(G0=relation['accuracy'] >= thresholds['G0_literal_relation'],
                 G1=atomic['axis_accuracy'] >= thresholds['G1_axis'],
                 G2=axis_correct > 0 and joint_correct / axis_correct >= thresholds['G2_sign_given_axis'],
                 G3=atomic['joint_accuracy'] >= thresholds['G3_joint_atom'],
                 G4=summaries['alias']['top1'] >= thresholds['G4_alias_decision'],
                 G5=summaries['alias_composition']['top1'] >= thresholds['G5_alias_composition'],
                 G6=causal['correct_new_rate'] >= thresholds['G6_correct_new'],
                 G7=summaries['literal']['top1'] == summaries['literal']['exact_winner'] ==
                 summaries['literal']['pairwise'] == thresholds['G7_exact'] and literal_callbacks == 0 and
                 permutation['score_mismatches'] == permutation['winner_mismatches'] == 0)
    result = dict(literal_relation=relation, atoms=atomic, summary=summaries, per_K=per_k,
                  causal=causal, permutation=permutation,
                  compiler_accuracy={name: dict(n=sum(query['stratum'] == name for query in queries),
                                                correct=sum(row['coefficient_exact'] for row in expected_compiled
                                                            if row['stratum'] == name)) for name in strata},
                  semantic_callbacks=semantic_callbacks, literal_callbacks=literal_callbacks,
                  gates=gates, passed=all(gates.values()), decision_rows=len(rows))
    compare(load(output / 'results.json'), result, tag + ': all reconstructed statistics')
    if (output / 'probability_table.json').exists():
        compare(load(output / 'probability_table.json'), matrix_records, tag + ': per-question probability table', 2e-5)
    proof = dict(head_probability_maximum_error=maximum_error, relation_matrices=len(texts),
                 probability_table_sha256=hashlib.sha256(json.dumps(matrix_records, sort_keys=True,
                                                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
                 atomic_queries=len(atoms), compiled_queries=len(queries), decision_rows=len(rows),
                 permutation_comparisons=permutation['comparisons'], causal_rows=causal_n,
                 gates=gates, passed=result['passed'], statistics_reconstructed=True)
    return result, proof, rows, resolutions, matrices


def interval(values):
    return np.quantile(np.asarray(values, dtype=np.float64), [.025, .975]).tolist()


def verify_final_intervals(root, atoms, compositions, rows, resolutions):
    atom_pairs = sorted({atom['pair_id'] for atom in atoms})
    atom_axes, atom_joints, alias_decisions = [], [], []
    for pair_id in atom_pairs:
        group = [atom for atom in atoms if atom['pair_id'] == pair_id]
        require(len(group) == 2, 'final atom bootstrap reversal pair')
        atom_axes.append(sum(resolutions[a['text']] is not None and resolutions[a['text']]['axis'] == a['axis']
                             for a in group))
        atom_joints.append(sum(resolutions[a['text']] is not None and
                               (resolutions[a['text']]['axis'], resolutions[a['text']]['sign']) ==
                               (a['axis'], a['sign']) for a in group))
        ids = {a['id'] for a in group}
        alias_decisions.append(float(np.mean([r['top1'] for r in rows if r['criterion_id'] in ids and
                                              r['source_split'] == 'unseen_wording'])))
    rng = np.random.default_rng(0)
    atom_draws = rng.integers(0, len(atom_pairs), (10000, len(atom_pairs)))
    axes = np.asarray(atom_axes)[atom_draws].sum(1)
    joints = np.asarray(atom_joints)[atom_draws].sum(1)
    output = dict(repetitions=10000, seed=0, opposite_atom_pairs=len(atom_pairs),
                  axis_CI95=interval(axes / (2 * len(atom_pairs))),
                  joint_CI95=interval(joints / (2 * len(atom_pairs))),
                  sign_given_axis_CI95=interval(joints[axes > 0] / axes[axes > 0]) if np.any(axes > 0) else None,
                  alias_decision_CI95=interval(np.asarray(alias_decisions)[atom_draws].mean(1)), point_gates_only=True)
    composition_pairs = sorted({row['pair_id'] for row in compositions})
    comp_means = []
    for pair_id in composition_pairs:
        group = [row for row in compositions if row['pair_id'] == pair_id]
        require(len(group) == 2, 'final composition bootstrap reversal pair')
        ids = {row['id'] for row in group}
        comp_means.append(float(np.mean([r['top1'] for r in rows if r['criterion_id'] in ids and
                                         r['source_split'] == 'unseen_criterion'])))
    comp_draws = rng.integers(0, len(composition_pairs), (10000, len(composition_pairs)))
    output.update(opposite_composition_pairs=len(composition_pairs),
                  composition_decision_CI95=interval(np.asarray(comp_means)[comp_draws].mean(1)))
    compare(load(root / 'final' / 'intervals.json'), output, 'fresh final reversal-pair intervals')
    return output


def paired_development_controls(groups, stage_rows, resolutions, outcomes, output_value):
    development = groups['development_atoms']
    pair_ids = sorted({atom['pair_id'] for atom in development})
    held_state_ids = sorted(state['id'] for state in groups['states']
                            if state['source_split'] == 'unseen_wording')
    require(len(pair_ids) == 16 and len(held_state_ids) == 16, 'paired development reversal/held-state counts')
    atoms_by_pair = {pair: [atom for atom in development if atom['pair_id'] == pair] for pair in pair_ids}
    per_arm = {}
    for arm in ARMS:
        row_index = {(row['criterion_id'], row['scenario_id']): row for row in stage_rows[arm]}
        require(len(row_index) == len(stage_rows[arm]), arm + ': duplicate decision rows in paired-control replay')
        atom_metrics = {}
        for atom in development:
            resolved = resolutions[arm][atom['text']]
            axis_correct = int(resolved is not None and resolved['axis'] == atom['axis'])
            joint_correct = int(resolved is not None and (resolved['axis'], resolved['sign']) ==
                                (atom['axis'], atom['sign']))
            cases = [row_index[(atom['id'], state_id)] for state_id in held_state_ids]
            require(all(row['stratum'] == 'alias' and row['source_split'] == 'unseen_wording'
                        for row in cases), arm + ': held-state paired decision scope')
            atom_metrics[atom['id']] = dict(
                axis_correct=axis_correct, joint_correct=joint_correct,
                held_state_top1_mean=float(np.mean([row['top1'] for row in cases])),
                held_states={row['scenario_id']: int(row['top1']) for row in cases},
            )
        per_arm[arm] = atom_metrics
    pair_rows = []
    for pair_id, atom_pair in atoms_by_pair.items():
        require(len(atom_pair) == 2, 'development atom reversal pair shape')
        arm_metrics = {}
        for arm in ARMS:
            values = per_arm[arm]
            arm_metrics[arm] = {
                metric: float(np.mean([values[atom['id']][metric] for atom in atom_pair]))
                for metric in ('axis_correct', 'joint_correct', 'held_state_top1_mean')
            }
        pair_rows.append(dict(pair_id=pair_id, atom_ids=[atom['id'] for atom in atom_pair],
                              arm_metrics=arm_metrics))
    atom_rows = [
        dict(id=atom['id'], pair_id=atom.get('pair_id'), axis=atom['axis'], sign=atom['sign'],
             arms={arm: per_arm[arm][atom['id']] for arm in ARMS})
        for atom in development
    ]
    atom_state_rows = [
        dict(criterion_id=atom['id'], pair_id=atom.get('pair_id'), scenario_id=state_id,
             arms={arm: per_arm[arm][atom['id']]['held_states'][state_id] for arm in ARMS})
        for atom in development for state_id in held_state_ids
    ]
    pair_values = {row['pair_id']: row['arm_metrics'] for row in pair_rows}
    metrics = ('axis_correct', 'joint_correct', 'held_state_top1_mean')
    simple_specs = {
        'data_at_generic': ('semantic_generic', 'literal_generic'),
        'data_at_defined': ('semantic_defined', 'literal_defined'),
        'definition_at_literal': ('literal_defined', 'literal_generic'),
        'definition_at_semantic': ('semantic_defined', 'semantic_generic'),
    }
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(pair_ids), size=(10000, len(pair_ids)))
    simple_values, simple_effects = {}, {}
    for name, (left_arm, right_arm) in simple_specs.items():
        simple_values[name] = {
            metric: np.asarray([pair_values[pair][left_arm][metric] - pair_values[pair][right_arm][metric]
                                for pair in pair_ids], dtype=np.float64)
            for metric in metrics
        }
        discordance = {}
        for metric in ('axis_correct', 'joint_correct'):
            discordance[metric] = _discordance(
                [row['arms'][left_arm][metric] for row in atom_rows],
                [row['arms'][right_arm][metric] for row in atom_rows])
        discordance['atom_by_held_state_top1'] = _discordance(
            [row['arms'][left_arm] for row in atom_state_rows],
            [row['arms'][right_arm] for row in atom_state_rows])
        simple_effects[name] = dict(
            left_arm=left_arm, right_arm=right_arm,
            raw_pair_effects={metric: simple_values[name][metric].tolist() for metric in metrics},
            mean_effect={metric: float(simple_values[name][metric].mean()) for metric in metrics},
            CI95={metric: np.quantile(simple_values[name][metric][draws].mean(axis=1),
                                      [.025, .975]).tolist() for metric in metrics},
            matched_discordance=discordance, bootstrap_pairs=10000, bootstrap_seed=0)
    main_arrays = {
        'data': {metric: (simple_values['data_at_generic'][metric] +
                          simple_values['data_at_defined'][metric]) / 2 for metric in metrics},
        'definition': {metric: (simple_values['definition_at_literal'][metric] +
                                simple_values['definition_at_semantic'][metric]) / 2 for metric in metrics},
        'interaction': {metric: simple_values['data_at_defined'][metric] -
                        simple_values['data_at_generic'][metric] for metric in metrics},
    }
    main_effects = {}
    for effect, arrays in main_arrays.items():
        main_effects[effect] = dict(
            raw_pair_effects={metric: arrays[metric].tolist() for metric in metrics},
            mean_effect={metric: float(arrays[metric].mean()) for metric in metrics},
            CI95={metric: np.quantile(arrays[metric][draws].mean(axis=1), [.025, .975]).tolist()
                  for metric in metrics},
            bootstrap_pairs=10000, bootstrap_seed=0,
            formula=('(semantic_defined-literal_defined)-(semantic_generic-literal_generic)'
                     if effect == 'interaction' else 'average of the two corresponding simple effects'))
    expected = dict(
        design='paired 2x2 format×training-data intervention; baseline is exact CBF7 reuse',
        metrics=list(metrics), reversal_clusters=16, atom_cases=32,
        held_states_per_atom=len(held_state_ids), atom_by_held_state_rows=len(atom_state_rows),
        paired_bootstrap=dict(repetitions=10000, seed=0, common_index_draws=True,
                              index_matrix_sha256=sha_bytes(draws.astype(np.int64, copy=False).tobytes()),
                              gate_use='descriptive only; development gate selection uses declared point metrics'),
        simple_effects=simple_effects, main_effects=main_effects,
        raw_pair_rows=pair_rows, raw_atom_rows=atom_rows,
        raw_atom_by_held_state_rows=atom_state_rows,
        raw_arm_point_metrics={arm: dict(axis_accuracy=outcomes[arm]['atoms']['axis_accuracy'],
                                         joint_accuracy=outcomes[arm]['atoms']['joint_accuracy'],
                                         alias_top1=outcomes[arm]['summary']['alias']['top1'])
                               for arm in ARMS},
        semantic_validation_used=False,
    )
    compare(output_value, expected, 'paired development controls')
    return expected


def verify_semantic_validation(root, arm, atoms, H, evidence, weights, normalizer):
    path = root / ('semantic_validation_' + arm + '.json')
    artifact = load(path)
    require(artifact.get('usage') == 'diagnostic_only_never_selection' and
            artifact.get('arm') == arm, arm + ': semantic validation restricted to diagnostics')
    questions = [atom['text'] for atom in atoms]
    subset = model_features(H, evidence, questions)
    probabilities, _ = probability_table(subset, normalizer, weights)
    rows = artifact.get('atoms')
    require(artifact.get('n') == len(atoms) and isinstance(rows, list) and
            [row.get('text') for row in rows] == questions,
            arm + ': semantic validation complete/original cohort order')
    correct_axis = correct_joint = maximum_error = 0
    for index, (atom, record) in enumerate(zip(atoms, rows)):
        require(record.get('id') == atom['id'] and record.get('pair_id') == atom['pair_id'] and
                record.get('gold_axis') == atom['axis'] and record.get('gold_sign') == atom['sign'],
                arm + ': semantic diagnostic labels')
        matrix = probabilities[4 * index:4 * index + 4]
        saved = np.asarray(record['relation_matrix'], dtype=np.float64)
        require(saved.shape == (4, 3), arm + ': semantic diagnostic matrix shape')
        error = float(np.max(np.abs(saved - matrix)))
        require(error < 2e-5, arm + ': semantic validation head matrix discrepancy')
        maximum_error = max(maximum_error, error)
        resolution = decode(matrix)
        compare(record.get('resolution'), resolution, arm + ': semantic validation resolution', 2e-5)
        axis_correct = int(resolution is not None and resolution['axis'] == atom['axis'])
        joint_correct = int(resolution is not None and (resolution['axis'], resolution['sign']) ==
                            (atom['axis'], atom['sign']))
        require(record.get('axis_correct') == axis_correct and record.get('joint_correct') == joint_correct,
                arm + ': semantic diagnostic per-atom scores')
        correct_axis += axis_correct
        correct_joint += joint_correct
    summary_value = dict(axis_correct=correct_axis, axis_accuracy=correct_axis / len(atoms),
                         joint_correct=correct_joint, joint_accuracy=correct_joint / len(atoms),
                         checkpoint_or_arm_selection_used=False)
    compare(artifact.get('summary'), summary_value, arm + ': semantic validation diagnostic summary')
    require(artifact.get('selection_used', False) is False and artifact.get('checkpoint_used', False) is False,
            arm + ': semantic validation must not select checkpoint/arm')
    return dict(semantic_validation_rows=len(atoms), maximum_probability_error=maximum_error,
                diagnostic_only=True, summary=summary_value)


def check_final_order(root, selected):
    selection_path = root / 'selection.json'
    require(selection_path.is_file(), 'immutable development selection missing')
    parity_name = 'native_sentencepiece_parity_final_' + selected + '.json'
    final_paths = [root / ('inputs_final_' + selected + '.json'),
                   root / ('features_final_' + selected + '.npy'),
                   root / parity_name, root / ('feature_cache_final_' + selected + '.json'),
                   root / 'final_readout.json', root / 'final' / 'results.json', root / 'results.json']
    for path in final_paths:
        require(path.is_file() and selection_path.stat().st_mtime_ns <= path.stat().st_mtime_ns,
                'selection persisted before final artifact: ' + str(path))
    require({path.name for path in root.glob('inputs_final_*.json')} ==
            {'inputs_final_' + selected + '.json'}, 'only selected final token evidence')
    require({path.name for path in root.glob('features_final_*.npy')} ==
            {'features_final_' + selected + '.npy'}, 'only selected final feature cache')
    require({path.name for path in root.glob('native_sentencepiece_parity_final_*.json')} == {parity_name} and
            {path.name for path in root.glob('feature_cache_final_*.json')} ==
            {'feature_cache_final_' + selected + '.json'}, 'only selected final tokenizer/feature manifests')


def verify_git_lineage(root, cfg, results, environment, manifest):
    repo = Path(__file__).parents[2]
    frozen = subprocess.check_output(['git', 'rev-parse', 'vey-2-final^{commit}'], cwd=repo, text=True).strip()
    require(frozen == FROZEN_VEY2, 'frozen Vey2 tag peel')
    revision = results.get('measurement_git_revision')
    require(isinstance(revision, str) and re.fullmatch(r'[0-9a-f]{40}', revision) and
            results.get('git_revision') == revision and environment.get('measurement_git_revision') == revision and
            environment.get('git_revision') == revision and manifest.get('preparation_git_revision') == revision and
            results.get('preparation_git_revision') == revision and
            environment.get('preparation_git_revision') == revision,
            'preparation/measurement source Git revisions are not identical')
    source_files = set(manifest.get('preparation_source_sha256', {})) | {
        'schema_support_verify.py', 'exact_state_build.py', 'exact_state_run.py',
    }
    hashes = {}
    for name in sorted(source_files):
        path = PROTOCOL.with_name(name)
        require(path.is_file(), 'pre-outcome source absent: ' + name)
        recorded = subprocess.check_output(['git', 'show', revision + ':research/cbf0/' + name], cwd=repo)
        require(path.read_bytes() == recorded, 'working source differs from measurement revision: ' + name)
        hashes[name] = sha(path)
    return frozen, hashes


def verify_results_policy(root, cfg, results, selection, dev, final, intervals, paired,
                          manifest, package, selected_details):
    require(results.get('experiment') == 'CBF8 Schema-support domain control' and
            results.get('source_root') == cfg['source_root'] and results.get('output_root') == cfg['output_root'],
            'result identity/source/output roots')
    compare(results.get('selection'), selection, 'results selection')
    compare(results.get('development'), dev, 'all four reconstructed development outcomes')
    compare(results.get('final'), final, 'selected final statistics')
    compare(results.get('final_intervals'), intervals, 'final intervals')
    compare(results.get('paired_development_controls'), paired, 'paired development controls')
    require(results.get('protocol_sha256') == sha(PROTOCOL) and
            results.get('corpus_manifest_sha256') == sha(root / 'corpus_manifest.json') and
            results.get('meaning_audit_sha256') == sha(root / 'meaning_audit.json') and
            results.get('source_result_manifest_sha256') == manifest['source_result_manifest_sha256'] and
            results.get('baseline_references') == manifest['baseline_references'],
            'result protocol/corpus/meaning/baseline source lineage')
    require(results.get('blind_review') == manifest.get('blind_review') and
            results.get('blind_predictions_sha256') == sha(root / 'blind_meaning_predictions.json'),
            'result binds sealed blind-review evidence')
    passed = bool(final['passed'])
    verdict = ('CBF8_SUPPORT_SCREEN_PASS_REPLICATION_REQUIRED' if passed else
               'CBF8_SUPPORT_NOT_EARNED_ON_FRESH_CORPUS')
    require(results.get('replication_required') is passed and results.get('B_STEF_allowed') is False and
            results.get('verdict') == verdict and
            results.get('semantic_validation_used_for_selection') is False and
            results.get('final_outcomes_never_select_arm') is True and
            results.get('no_final_tuning') is True,
            'replication/verdict/B-STEF/frozen-selection policy')
    license_category = cfg['model']['license_class']
    require(results.get('license_category') == license_category and
            results.get('license_warning') ==
            'The NLI checkpoint remains conditional/review and research-only; this result does not clear shipping obligations.',
            'conditional model license disclosure')
    selected_outcome = dev[selection['selected']]
    if passed:
        next_branch = cfg['replication']
    elif (selected_outcome['atoms']['axis_accuracy'] >= cfg['gates']['G1_axis'] and
          selected_outcome['atoms']['conditional_sign_accuracy'] is not None and
          selected_outcome['atoms']['conditional_sign_accuracy'] < cfg['gates']['G2_sign_given_axis']):
        next_branch = cfg['failure_branches']['axis_pass_sign_fail']
    elif all(not dev[arm]['passed'] for arm in ELIGIBLE):
        next_branch = cfg['failure_branches']['all_support_arms_fail']
    else:
        next_branch = ('Fresh-final gates did not all pass; replication and B-STEF remain blocked. '
                       'The frozen CBF8 protocol defines no additional intervention for this outcome.')
    require(results.get('next_branch') == next_branch, 'result next-branch decision derived from frozen outcomes')
    readout = load(root / 'final_readout.json')
    selected = selection['selected']
    fmt = 'generic' if selected.endswith('generic') else 'defined'
    lineage_hash = package['lineage']['encoder_parameters_before_sha256']
    parity_path = root / ('native_sentencepiece_parity_final_' + selected + '.json')
    expected_readout = dict(
        selected_arm=selected, format=fmt, model_repo=cfg['model']['repo'],
        model_revision=cfg['model']['revision'], model_weight_sha256=cfg['model']['weight_sha256'],
        floating_parameters=cfg['model']['floating_parameters'],
        encoder_parameters_before_sha256=lineage_hash,
        encoder_parameters_after_training_sha256=lineage_hash,
        encoder_parameters_after_final_capture_sha256=lineage_hash,
        feature_file='features_final_' + selected + '.npy',
        feature_sha256=sha(root / ('features_final_' + selected + '.npy')),
        input_file='inputs_final_' + selected + '.json',
        input_sha256=sha(root / ('inputs_final_' + selected + '.json')),
        native_sentencepiece_parity_sha256=sha(parity_path),
        checkpoint_file=selected + '_checkpoint.pt',
        checkpoint_sha256=sha(root / (selected + '_checkpoint.pt')),
        normalizer_file='normalizer_' + selected + '.npz',
        normalizer_sha256=sha(root / ('normalizer_' + selected + '.npz')),
        normalizer_mean_sha256=selected_details['normalizer_mean_sha256'],
        normalizer_std_sha256=selected_details['normalizer_std_sha256'],
        meaning_audit_sha256=sha(root / 'meaning_audit.json'), final_openings=1)
    compare(readout, expected_readout, 'immutable final selected readout')
    compare(results.get('final_readout'), expected_readout, 'results final readout')
    require(results.get('final_readout_sha256') == sha(root / 'final_readout.json'),
            'results final-readout file hash')
    forbidden = ('calibrated_probability', 'probability_calibration', 'confidence_certificate',
                 'certificate_emitted', 'certified_rate', 'certified_accuracy', 'speed_claim',
                 'latency_claim', 'throughput_claim')
    for key in forbidden:
        if key in results:
            require(results[key] in (False, None), 'unsupported calibration/certification/performance claim: ' + key)
    return dict(verdict=verdict, replication_required=passed, B_STEF_allowed=False,
                license_category=license_category, final_readout_sha256=sha(root / 'final_readout.json'))


def verify_preflight(root, environment, cfg, groups, package, generic_features, generic_evidence,
                    defined_features, defined_evidence):
    attempts = environment.get('preflight_attempts')
    files = sorted(path.name for path in root.glob('preflight_attempt_*.json'))
    require(isinstance(attempts, list) and attempts and attempts == files,
            'all persisted preflight attempts are recorded exactly')
    expected_text = groups['training_atoms'][0]['text']
    expected_field = cfg['schema'][groups['training_atoms'][0]['axis']]
    generic_lookup = evidence_by_text(generic_evidence)
    defined_lookup = evidence_by_text(defined_evidence)
    maximum_length = 0
    for name in attempts:
        attempt = load(root / name)
        require(attempt.get('preflight') is True and attempt.get('passed') is True and
                attempt.get('protocol_sha256') == sha(PROTOCOL) and
                attempt.get('corpus_manifest_sha256') == sha(root / 'corpus_manifest.json') and
                attempt.get('meaning_audit_sha256') == sha(root / 'meaning_audit.json') and
                attempt.get('source') == 'one literal training criterion and one canonical field only' and
                attempt.get('formats') == ['generic', 'defined'] and
                attempt.get('tokenizer_parity_exact') is True,
                'actual tiny-pair preflight identity and format evidence')
        lineage_hash = package['lineage']['encoder_parameters_before_sha256']
        require(attempt.get('model_revision') == cfg['model']['revision'] and
                attempt.get('model_weight_sha256') == cfg['model']['weight_sha256'] and
                attempt.get('floating_parameters') == cfg['model']['floating_parameters'] and
                attempt.get('width') == cfg['model']['width'] and
                attempt.get('encoder_parameters_before_sha256') ==
                attempt.get('encoder_parameters_after_sha256') == lineage_hash and
                attempt.get('encoder_parameters_frozen') is True and
                attempt.get('encoder_parameter_gradient_count') == 0 and
                attempt.get('actual_base_model_forward') is True and
                attempt.get('classifier_invoked') is False and attempt.get('pooler_invoked') is False and
                attempt.get('head_only_gradient') is True and attempt.get('head_gradient_L1', 0) > 0 and
                attempt.get('final_inputs_captured') is False and
                attempt.get('alias_inputs_captured') is False,
                'actual preflight base encoder/head gradient/frozen/model lineage')
        require(attempt.get('feature_mmap_verified') is True and
                attempt.get('feature_shape') == [2, cfg['model']['width']] and
                attempt.get('feature_dtype') == 'float32',
                'preflight persisted float32 mmap dimensions')
        input_path = root / attempt['tokenizer_evidence_file']
        feature_path = root / attempt['feature_evidence_file']
        require(sha(input_path) == attempt['tokenizer_evidence_sha256'] and
                sha(feature_path) == attempt['feature_evidence_sha256'],
                'preflight input/feature evidence hashes')
        payload = load(input_path)
        require(set(payload) == {'formats', 'tokenizer_parity'}, 'preflight evidence exact sections')
        for fmt in ('generic', 'defined'):
            evidence = payload['formats'][fmt]
            require(evidence.get('pair_count') == 1 and len(evidence.get('pairs', [])) == 1 and
                    evidence['pairs'][0]['criterion'] == expected_text and
                    evidence['pairs'][0]['field'] == expected_field and
                    payload['tokenizer_parity'][fmt].get('exact') is True,
                    'preflight captures only first literal train criterion/canonical field')
            token_proof = retokenize(evidence, package['fast'], package['native'], [expected_text],
                                     cfg, fmt, 'actual ' + fmt + ' preflight', fields=[expected_field])
            maximum_length = max(maximum_length, token_proof['maximum_sequence_length'])
        features = np.load(feature_path, mmap_mode='r', allow_pickle=False)
        require(features.shape == (2, cfg['model']['width']) and features.dtype == np.float32 and
                np.array_equal(features[0], generic_features[generic_lookup[(expected_text, expected_field)]]) and
                np.array_equal(features[1], defined_features[defined_lookup[(expected_text, expected_field)]]),
                'preflight real frozen feature rows equal matching canonical cache rows')
    return dict(attempts=len(attempts), maximum_sequence_length=maximum_length,
                actual_literal_pair_forward=True, feature_cache_parity=True)


def main(root, cfg):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    require(tuple(arm['id'] for arm in cfg['arms']) == ARMS, 'frozen four-arm order')
    require([arm['eligible'] for arm in cfg['arms']] == [False, True, True, True], 'eligible arm contract')
    require(cfg['B_STEF_allowed'] is False, 'protocol B-STEF prohibited')
    manifest = verify_manifest(root, cfg)
    meaning = load(root / 'meaning_audit.json')
    groups, corpus_proof, source, source_verify, source_hashes = validate_corpus(root, cfg, manifest, meaning)
    for name in COHORTS:
        require(sha(root / name) == manifest['files'][name], 'cohort changed after validation ' + name)

    env = load(root / 'environment.json')
    require(env.get('experiment') == 'CBF8' and
            env.get('protocol_sha256') == sha(PROTOCOL) and
            env.get('corpus_manifest_sha256') == sha(root / 'corpus_manifest.json') and
            env.get('meaning_audit_sha256') == sha(root / 'meaning_audit.json') and
            env.get('source_root') == cfg['source_root'] and env.get('output_root') == cfg['output_root'] and
            env.get('source_result_manifest_sha256') == manifest['source_result_manifest_sha256'] and
            env.get('blind_review') == manifest['blind_review'] and
            env.get('blind_predictions_sha256') == sha(root / manifest['blind_review']['predictions_file']) and
            env.get('model_repo') == cfg['model']['repo'] and
            env.get('model_revision') == cfg['model']['revision'] and
            env.get('model_weight_sha256') == cfg['model']['weight_sha256'] and
            env.get('floating_parameters') == cfg['model']['floating_parameters'] and
            env.get('license_category') == cfg['model']['license_class'] and
            env.get('paid_resources') is False and env.get('paid') is False and
            env.get('threads') <= 4 and env.get('threads_budget') == 4 and
            env.get('interop_threads') == 1 and env.get('BLAS_threads') == 1 and
            env.get('pair_batch') == 32 and env.get('token_limit') == 128 and
            env.get('one_model_resident_at_a_time') is True and
            env.get('semantic_validation_checkpoint_selection') is False,
            'runtime environment/preflight lineage')
    result_source = load(root / 'results.json')
    frozen_tag, code_hashes = verify_git_lineage(root, cfg, result_source, env, manifest)
    package = model_package(cfg, root)
    lineage = package['lineage']
    lineage_before, lineage_after = hash_pair(lineage)
    require(lineage_before == lineage_after == package['full_tensor_hash_sha256'],
            'encoder full hash unchanged across all new feature capture')

    training = groups['training_atoms']
    validation = groups['validation_atoms']
    g0 = groups['literal_holdout_atoms']
    dev_atoms = groups['development_atoms']
    dev_comps = groups['development_compositions']
    literals = groups['literal_cases']
    semantic_train = groups['semantic_training_atoms']
    semantic_val = groups['semantic_validation_atoms']
    dev_texts = row_texts(dev_atoms, dev_comps, g0)
    generic_evidence = load(root / 'inputs_generic.json')
    defined_evidence = load(root / 'inputs_defined.json')
    generic_texts = sorted({pair['criterion'] for pair in generic_evidence['pairs']})
    defined_texts = sorted({pair['criterion'] for pair in defined_evidence['pairs']})
    require(generic_texts == defined_texts, 'generic/defined cache criterion set equality')
    literal_texts = {row['text'] for cohort in (training, validation, g0, dev_atoms) for row in cohort}
    semantic_texts = {row['text'] for row in semantic_train + semantic_val}
    expected_cache_texts = sorted(literal_texts | semantic_texts | set(dev_texts))
    require(generic_texts == expected_cache_texts, 'new caches are training/validation/development only; no final')
    final_texts = set(row_texts(groups['final_atoms'], groups['final_compositions'], g0))
    require(not final_texts.intersection(generic_texts), 'fresh final phrases appeared before selection')

    generic_H = np.load(root / 'features_generic.npy', mmap_mode='r', allow_pickle=False)
    defined_H = np.load(root / 'features_defined.npy', mmap_mode='r', allow_pickle=False)
    generic_evidence_check = retokenize(generic_evidence, package['fast'], package['native'],
                                        generic_texts, cfg, 'generic', 'generic cache')
    defined_evidence_check = retokenize(defined_evidence, package['fast'], package['native'],
                                        defined_texts, cfg, 'defined', 'defined cache')
    generic_cache_proof = validate_feature_cache(root, generic_H, generic_evidence, generic_texts, 'generic')
    defined_cache_proof = validate_feature_cache(root, defined_H, defined_evidence, defined_texts, 'defined')
    generic_parity_proof = verify_sentencepiece_parity(root, generic_evidence, package, 'generic')
    defined_parity_proof = verify_sentencepiece_parity(root, defined_evidence, package, 'defined')
    generic_manifest_proof = verify_cache_manifest(
        root, 'feature_cache_generic.json', generic_evidence, generic_H, 'generic', 'generic',
        lineage['encoder_parameters_before_sha256'], cfg)
    defined_manifest_proof = verify_cache_manifest(
        root, 'feature_cache_defined.json', defined_evidence, defined_H, 'defined', 'defined',
        lineage['encoder_parameters_before_sha256'], cfg)
    require(generic_H.shape == defined_H.shape, 'format feature cache shape parity')
    preflight_proof = verify_preflight(root, env, cfg, groups, package, generic_H, generic_evidence,
                                       defined_H, defined_evidence)

    # Retokenize and hash-check the original higher-CLS baseline, then require exact cached feature reuse.
    baseline_inputs = load(source / 'inputs_nli_xsmall_higher.json')
    base_texts = sorted({pair['criterion'] for pair in baseline_inputs['pairs']})
    baseline_token_proof = retokenize(baseline_inputs, package['fast'], package['native'], base_texts,
                                      cfg, 'generic', 'CBF7 raw-higher baseline', source_higher=True)
    baseline_H = np.load(source / 'features_nli_xsmall_higher_cls.npy', mmap_mode='r', allow_pickle=False)
    finite_array(baseline_H, 'CBF7 raw higher CLS')
    require(baseline_H.shape == (4 * len(base_texts), 384), 'CBF7 baseline feature shape')
    baseline_lookup = evidence_by_text(baseline_inputs)
    generic_lookup = evidence_by_text(generic_evidence)
    baseline_feature_lookup = {key: baseline_H[index] for key, index in baseline_lookup.items()}
    for key, index in generic_lookup.items():
        if key in baseline_feature_lookup:
            require(np.array_equal(generic_H[index], baseline_feature_lookup[key]),
                    'generic new capture differs from byte-identical CBF7 raw higher CLS ' + repr(key))
    source_reference_hashes = {name: sha(source / name) for name in BASELINE_FILES}
    require(source_verify['checked_source_and_model_sha256'].get(str(source / 'features_nli_xsmall_higher_cls.npy')) ==
            source_reference_hashes['features_nli_xsmall_higher_cls.npy'], 'CBF7 source feature hash reference')

    arm_data = {}
    source_baseline = dict(root=source)
    caches = {'literal_generic': (baseline_H, baseline_inputs, base_texts)}
    for arm in ARMS[1:]:
        fmt = 'generic' if arm.endswith('generic') else 'defined'
        H, evidence = (generic_H, generic_evidence) if fmt == 'generic' else (defined_H, defined_evidence)
        weights, normalizer, details, training_proof, checkpoint_path, training_evidence, training_H = \
            checkpoint_details(root, arm, H, evidence, groups, cfg, source_baseline)
        verify_head_selection_integrity(root, arm, details, cfg)
        arm_data[arm] = dict(weights=weights, normalizer=normalizer, details=details,
                             training_proof=training_proof, H=H, evidence=evidence, checkpoint=checkpoint_path)
        caches[arm] = (H, evidence, sorted({p['criterion'] for p in evidence['pairs']}))
        require(details.get('evidence_sha256') in (None, sha(root / ('inputs_' + fmt + '.json'))),
                arm + ': exact evidence file hash')
        require(details.get('feature_sha256') in (None, sha(root / ('features_' + fmt + '.npy'))),
                arm + ': exact frozen feature file hash')
        require(details.get('frozen_parameter_before_sha256', details.get('frozen_encoder_before_sha256')) ==
                lineage.get('tensor_hash_before', lineage.get('full_tensor_hash_before')),
                arm + ': head-training frozen model hash links to encoder lineage')

    weights, normalizer, details, training_proof, checkpoint_path, _, _ = \
        checkpoint_details(root, 'literal_generic', baseline_H, baseline_inputs, groups, cfg, source_baseline)
    arm_data['literal_generic'] = dict(weights=weights, normalizer=normalizer, details=details,
                                       training_proof=training_proof, H=baseline_H,
                                       evidence=baseline_inputs, checkpoint=checkpoint_path)

    selection = load(root / 'selection.json')
    outcomes = {}
    stage_checks = {}
    stage_rows = {}
    resolutions = {}
    matrix_cache = {}
    for arm in ARMS:
        data = arm_data[arm]
        H, evidence = data['H'], data['evidence']
        result, proof, rows, resolved, matrices = verify_stage(
            root, 'development_' + arm, dev_atoms, dev_comps, literals, groups['states'],
            H, evidence, data['weights'], data['normalizer'], g0, cfg)
        outcomes[arm] = result
        stage_checks[arm] = dict(**proof, training=data['training_proof'])
        stage_rows[arm] = rows
        resolutions[arm] = resolved
        matrix_cache[arm] = matrices
        if arm != 'literal_generic':
            validation_proof = verify_semantic_validation(root, arm, semantic_val, H, evidence,
                                                          data['weights'], data['normalizer'])
            stage_checks[arm]['semantic_validation'] = validation_proof
    paired = paired_development_controls(groups, stage_rows, resolutions, outcomes,
                                         result_source['paired_development_controls'])

    passes = [arm for arm in ELIGIBLE if outcomes[arm]['passed']]
    if passes:
        selected = passes[0]
        reason = 'first eligible all-gate development pass in preregistered arm order'
    else:
        selected = min(ELIGIBLE, key=lambda arm: (-outcomes[arm]['atoms']['joint_accuracy'],
                                                  -outcomes[arm]['summary']['alias']['top1'],
                                                  ELIGIBLE.index(arm)))
        reason = 'highest eligible development joint atom, then alias decisions, then earlier arm order'
    baseline_refs = {role: dict(file=value['file'], sha256=value['sha256'])
                     for role, value in manifest['baseline_references']['files'].items()}
    expected_selection = dict(
        selected=selected, reason=reason, passed=selected in passes,
        development={arm: dict(passed=outcomes[arm]['passed'], gates=outcomes[arm]['gates'],
                               joint_accuracy=outcomes[arm]['atoms']['joint_accuracy'],
                               alias_decision_accuracy=outcomes[arm]['summary']['alias']['top1'])
                     for arm in ARMS},
        arm_order=list(ARMS), eligible_order=list(ELIGIBLE), semantic_validation_used=False,
        final_outcomes_never_select_arm=True, final_openings=1,
        protocol_sha256=sha(PROTOCOL), meaning_audit_sha256=sha(root / 'meaning_audit.json'),
        baseline_references=baseline_refs,
        paired_development_controls_sha256=sha(root / 'paired_development_controls.json'))
    compare(selection, expected_selection, 'independent development-only first-pass/joint-alias selection')
    check_final_order(root, selected)

    final_atoms = groups['final_atoms']
    final_comps = groups['final_compositions']
    final_text_order = row_texts(final_atoms, final_comps, g0)
    final_evidence = load(root / ('inputs_final_' + selected + '.json'))
    final_H = np.load(root / ('features_final_' + selected + '.npy'), mmap_mode='r', allow_pickle=False)
    final_fmt = 'generic' if selected.endswith('generic') else 'defined'
    final_token_proof = retokenize(final_evidence, package['fast'], package['native'], final_text_order,
                                   cfg, final_fmt, 'selected final ' + selected)
    final_parity_proof = verify_sentencepiece_parity(
        root, final_evidence, package, final_fmt, suffix='final_' + selected)
    validate_feature_cache(root, final_H, final_evidence, final_text_order, 'selected final ' + selected)
    final_manifest_proof = verify_cache_manifest(
        root, 'feature_cache_final_' + selected + '.json', final_evidence, final_H, final_fmt,
        'final_' + selected, lineage['encoder_parameters_before_sha256'], cfg)
    selected_data = arm_data[selected]
    final, final_proof, final_rows, final_resolutions, final_matrices = verify_stage(
        root, 'final', final_atoms, final_comps, literals, groups['states'], final_H, final_evidence,
        selected_data['weights'], selected_data['normalizer'], g0, cfg)
    dev_g0 = matrix_cache[selected]
    final_g0_error = max(float(np.max(np.abs(dev_g0[c['text']] - final_matrices[c['text']]))) for c in g0)
    require(final_g0_error < 2e-5, 'matched G0 development/final probability replay')
    intervals = verify_final_intervals(root, final_atoms, final_comps, final_rows, final_resolutions)
    results = load(root / 'results.json')
    compare(results.get('development'), outcomes, 'result development all four arms')
    policy_proof = verify_results_policy(root, cfg, results, selection, outcomes, final, intervals, paired,
                                         manifest, package, selected_data['details'])

    proof = dict(all_checks_pass=True, evidence_class='MEASURED: independent CPU persisted-artifact reconstruction',
                 experiment='CBF8', source_root=str(source), output_root=str(root),
                 source_hashes=source_hashes, baseline_reference_sha256=source_reference_hashes,
                 model_package=dict(weight_sha256=cfg['model']['weight_sha256'], floating_parameters=package['parameter_count'],
                                    nonfloating_position_and_other_buffers=package['parameter_exclusions'],
                                    fast_tokenizer_sha256=package['fast_tokenizer_sha256'],
                                    sentencepiece_sha256=package['sentencepiece_sha256'],
                                    full_frozen_tensor_hash_unchanged=True),
                 corpus=corpus_proof, preflight=preflight_proof,
                 cache_retokenization=dict(generic=generic_evidence_check,
                                           defined=defined_evidence_check,
                                           baseline=baseline_token_proof,
                                           final=final_token_proof),
                 tokenizer_parity=dict(generic=generic_parity_proof, defined=defined_parity_proof,
                                       final=final_parity_proof),
                 feature_caches=dict(generic=generic_cache_proof, defined=defined_cache_proof,
                                     generic_manifest=generic_manifest_proof,
                                     defined_manifest=defined_manifest_proof,
                                     final_manifest=final_manifest_proof),
                 stages=stage_checks, selected=selected, reconstructed_selection=expected_selection,
                 final=final_proof, final_intervals=intervals,
                 paired_development_controls=paired, matched_G0_max_probability_error=final_g0_error,
                 final_outcome_never_selects_arm=True, final_opening_count=1, selected_only_final=True,
                 exact_literal_neural_calls=0, cpu_only=True, uncalibrated_softmax_not_certificate=True,
                 conditional_review_not_shipping_clearance=True, B_STEF_allowed=False,
                 policy=policy_proof, frozen_vey2_tag=frozen_tag,
                 measurement_source_sha256=code_hashes,
                 mtime_order_observed_not_crypto_seal=True, protocol_sha256=sha(PROTOCOL))
    return proof


def save_new(path, value):
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(payload)


def main_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, help='Artifact root; must match the frozen protocol output root')
    args = parser.parse_args()
    cfg = load(PROTOCOL)
    root = args.root or Path(cfg['output_root'])
    try:
        require(root.resolve() == Path(cfg['output_root']).resolve(), 'artifact root differs from protocol')
        result = main(root, cfg)
        save_new(root / 'verification.json', result)
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    except Exception as error:
        failure = dict(all_checks_pass=False, error_type=type(error).__name__, error=str(error),
                       traceback=traceback.format_exc(), protocol_sha256=sha(PROTOCOL))
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        if root.is_dir():
            try:
                save_new(root / ('verification_failure_' + stamp + '_' + str(os.getpid()) + '.json'), failure)
            except FileExistsError:
                pass
        raise


if __name__ == '__main__':
    main_cli()
