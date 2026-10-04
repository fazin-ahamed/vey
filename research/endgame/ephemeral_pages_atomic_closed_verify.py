#!/usr/bin/env python3
"""Independent CPU NumPy replay of the sealed atomic closed CBF8 diagnostic.

Torch is used only to deserialize the hash-pinned checkpoint, never to execute
its reader. The frozen grammar parser supplies spans; all coefficient and
candidate arithmetic is reconstructed here. No current ECA final data is read.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import sys

for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'ARROW_NUM_THREADS', 'TOKENIZERS_PARALLELISM'):
    os.environ[_key] = 'false' if _key == 'TOKENIZERS_PARALLELISM' else '1'
import numpy as np

HERE = Path(__file__).resolve().parent
AXES = ('reliability', 'purchase expense', 'operating expense', 'convenience')
CONTROLS = {'pages', 'cross', 'cosine', 'lexical', 'query_blind'}


class Audit:
    def __init__(self):
        self.hashes = {}
        self.counts = Counter()
        self.errors = []
        self.max_numeric_error = 0.0

    def require(self, condition, label):
        self.counts['assertions'] += 1
        if not condition:
            if len(self.errors) < 32:
                self.errors.append(label)
            raise ValueError(label)

    def hash(self, path, expected=None):
        path = Path(path).resolve()
        self.require(path.is_file(), 'missing custody artifact')
        # Current ECA final captures are never an allowed input, even for hashing.
        self.require(not ('ephemeral-pages' in str(path) and
                          ('final' in path.name or 'final' in path.parts)),
                     'forbidden current ECA final input')
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1048576), b''):
                h.update(chunk)
        value = h.hexdigest()
        self.hashes[str(path)] = value
        if expected is not None:
            self.require(value == expected, 'artifact SHA256 mismatch')
        return value

    def load(self, path):
        self.hash(path)
        def pairs(items):
            out = {}
            for key, value in items:
                self.require(key not in out, 'duplicate JSON key')
                out[key] = value
            return out
        return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))

    def rows(self, path):
        self.hash(path)
        with Path(path).open(encoding='utf-8') as stream:
            return [json.loads(line) for line in stream]

    def artifact(self, entry, base):
        path = Path(entry['path'])
        path = path if path.is_absolute() else base / path
        self.hash(path, entry['sha256'])
        return path.resolve()

    def equal(self, actual, expected, label):
        self.counts['compared_values'] += 1
        self.require(actual == expected, label)

    def numeric(self, actual, expected, label):
        x, y = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
        self.require(x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all(), label + '.shape/finiteness')
        if x.size:
            self.max_numeric_error = max(self.max_numeric_error, float(np.max(np.abs(x-y))))
        self.require(np.allclose(x, y, rtol=2e-5, atol=2e-5), label + '.numeric')
        self.counts['numeric_scalars'] += x.size


def sigmoid(x):
    x = np.asarray(x, dtype=np.float32)
    return np.exp(-np.logaddexp(np.float32(0), -x))


def replay_reader(audit, checkpoint, queries, pages):
    params = {key: value.detach().cpu().numpy() for key, value in checkpoint['state_dict'].items()}
    expected = {'wp.weight': (64, 384), 'wqa.weight': (64, 384),
                'wqd.weight': (64, 384), 'wv.weight': (1, 384),
                'wv.bias': (1,), 'bk': (), 'uk': (), 'vk': ()}
    audit.equal(set(params), set(expected), 'reader parameter names')
    for key, shape in expected.items():
        audit.require(params[key].shape == shape and params[key].dtype == np.float32
                      and np.isfinite(params[key]).all(), 'reader parameter contract')
    stats = checkpoint['normalizer']['pages']
    mean, std = (np.asarray(stats[k], dtype=np.float32) for k in ('mean', 'std'))
    audit.require(mean.shape == std.shape == (384,) and np.isfinite(mean).all()
                  and np.isfinite(std).all() and (std > 0).all(), 'normalizer contract')
    q, p = (queries - mean) / std, (pages - mean) / std
    z = p @ params['wp.weight'].T
    out = []
    for question in q:
        relevance = np.sum(z * (question @ params['wqa.weight'].T), axis=1) / np.float32(8)
        direction = np.tanh(np.sum(z * (question @ params['wqd.weight'].T), axis=1) / np.float32(8))
        values = sigmoid((p @ params['wv.weight'].T).reshape(-1) + params['wv.bias'][0])
        attention = np.exp(relevance - relevance.max())
        attention /= attention.sum()
        attention /= attention.sum()
        average = np.sum(attention * values)
        variance = np.sum(attention * (values-average)**2)
        known = params['bk'] + np.logaddexp(np.float32(0), params['uk']) * relevance.max() - np.logaddexp(np.float32(0), params['vk']) * variance
        score = np.sum(attention * (np.float32(.5) + direction * (values-np.float32(.5))))
        out.append({'score': [float(score)], 'known_logits': [float(known)],
                    'relevance_logits': [relevance.tolist()], 'attention': [attention.tolist()],
                    'direction': [direction.tolist()], 'raw_value': [values.tolist()]})
    return out


def facts(audit, text):
    fields = text.rstrip('.').split('; ')
    parsed = {}
    for field in fields:
        match = re.fullmatch(r'([^:;]+): (\d+) percent', field)
        audit.require(match is not None, 'invalid integer source fact')
        name, value = match[1], int(match[2])
        audit.require(name not in parsed and 0 <= value <= 100, 'duplicate/out-of-range source fact')
        parsed[name] = value
    audit.equal(set(parsed), set(AXES), 'source fact axes')
    return [parsed[name] for name in AXES]


def choice(vector, rows):
    if vector is None:
        return None, [], None
    scores = [sum(coefficient * value for coefficient, value in zip(vector, row)) for row in rows]
    top = [ordinal for ordinal, score in enumerate(scores) if score == max(scores)]
    return scores, top, min(top)


def verify(audit, protocol_path, result_path):
    cfg, result = audit.load(protocol_path), audit.load(result_path)
    audit.equal(result['protocol_sha256'], audit.hash(protocol_path), 'closed protocol hash')
    audit.equal(result['schema'], 'vey.eca.closed-diagnostic.v1', 'result schema')
    audit.equal(result['claim'], cfg['claim'], 'diagnostic claim')
    for key in ('selection', 'training', 'threshold_fitting', 'protocol_mutation'):
        audit.equal(cfg[key], False, 'closed diagnostic prohibition')
    for entry in cfg['authority'].values():
        if isinstance(entry, dict) and 'sha256' in entry:
            audit.artifact(entry, protocol_path.parent)
    audit.artifact(cfg['implementation'], protocol_path.parent)
    audit.equal(result['adapter_sha256'], cfg['implementation']['sha256'], 'producer hash')
    audit.equal(result['source_hashes'], cfg['source_hashes'], 'source ledger')
    for path, sha in cfg['source_hashes'].items():
        audit.hash(path, sha)
    legacy = audit.load(cfg['authority']['unchanged_closed_recipe']['path'])
    for key in ('recipe', 'cohorts', 'source_hashes', 'source_root', 'claim'):
        audit.equal(cfg[key], legacy[key], 'unchanged recipe.' + key)
    context = cfg['authority']['active_context']
    audit.equal(result['experiment_context'], context, 'explicit atomic context')
    active = audit.load(cfg['authority']['active_atomic_protocol']['path'])
    inherited_path = HERE.parents[1] / active['parent_protocol_path']
    audit.hash(inherited_path, active['parent_protocol_sha256'])
    encoder_spec = audit.load(inherited_path)['encoder']
    receipt_path = Path(cfg['authority']['sealed_corrected_selection']['path'])
    receipt = audit.load(receipt_path)
    projected_receipt = {key: receipt[key] for key in (
        'protocol_sha256', 'eligible_arm', 'final_outcomes_used',
        'checkpoint_files', 'calibration_file', 'selection_file', 'corpus_build_manifest')}
    projected_receipt.update({
        'receipt_path': str(receipt_path.resolve()),
        'receipt_sha256': audit.hash(receipt_path),
        'amendment_file': cfg['authority']['active_atomic_protocol']})
    audit.equal(result['receipt'], projected_receipt, 'sealed receipt projection identity')
    audit.equal(receipt['schema'], 'vey.eca.selection-calibration.v1', 'receipt schema')
    audit.equal(receipt['protocol_sha256'], context['protocol_sha256'], 'receipt protocol')
    for key, value in (('eligible_arm', 'pages'), ('eligible_arms', ['pages']), ('final_outcomes_used', False)):
        audit.equal(receipt[key], value, 'receipt.' + key)
    audit.equal(set(receipt['checkpoint_files']), CONTROLS, 'checkpoint coverage')
    for name, entry in receipt['checkpoint_files'].items():
        path = audit.artifact(entry, receipt_path.parent)
        audit.equal(path, receipt_path.parent / ('lexical.pkl' if name == 'lexical' else name+'.pt'), 'checkpoint run custody')
    custody = receipt['checkpoint_custody']
    launch = audit.load(audit.artifact(custody['launch_receipt'], receipt_path.parent))
    audit.equal(launch['experiment_context'], context, 'launch atomic context')
    audit.equal(launch['protocol_sha256'], context['protocol_sha256'], 'launch atomic protocol')
    audit.equal(launch['smoke'], False, 'launch non-smoke')
    audit.equal(set(custody['checkpoint_histories']), CONTROLS, 'history coverage')
    for name, entry in custody['checkpoint_histories'].items():
        history = audit.load(audit.artifact(entry, receipt_path.parent))
        audit.equal(history['control'], name, 'history control')
        audit.equal(history['experiment_context'], context, 'history atomic context')
        audit.equal(history['protocol_sha256'], context['protocol_sha256'], 'history protocol')
        audit.equal(history['smoke'], False, 'history non-smoke')
        audit.equal(history['checkpoint_sha256'], receipt['checkpoint_files'][name]['sha256'], 'history checkpoint hash')
        audit.equal(history['launch_receipt_sha256'], custody['launch_receipt']['sha256'], 'history launch hash')
    selection = audit.load(audit.artifact(receipt['selection_file'], receipt_path.parent))
    calibration = audit.load(audit.artifact(receipt['calibration_file'], receipt_path.parent))
    development = audit.load(audit.artifact(receipt['development_evaluation'], receipt_path.parent))
    for obj, schema in ((selection, 'vey.eca.selection.v1'), (calibration, 'vey.eca.calibration.v1')):
        audit.equal(obj['schema'], schema, 'sealed artifact schema')
        audit.equal(obj['protocol_sha256'], context['protocol_sha256'], 'sealed atomic protocol')
        audit.equal(obj['final_outcomes_used'], False, 'no final selection')
    audit.equal(selection['eligible_arm'], 'pages', 'selected arm')
    audit.equal(selection['eligible_arms'], ['pages'], 'selected arms')
    audit.equal(selection['development_evaluation'], receipt['development_evaluation'], 'development custody')
    audit.equal(selection['gate_screens'], development['gate_screens'], 'selection gates')
    audit.equal(selection['comparisons'], development['comparisons'], 'selection comparisons')
    audit.equal(development['phase'], 'development', 'development phase')
    audit.equal(development['final_outcomes_used_for_selection'], False, 'development no final')
    audit.equal(calibration['checkpoint_files'], receipt['checkpoint_files'], 'calibration checkpoints')
    audit.equal(calibration['fit_split'], 'calibration', 'calibration split')
    audit.artifact(calibration['calibration_evaluation'], receipt_path.parent)
    audit.equal(set(calibration['controls']), CONTROLS, 'calibration control coverage')
    for settings in calibration['controls'].values():
        audit.equal(settings['fit_split'], 'calibration', 'control calibration split')
        audit.equal(settings['final_outcomes_used'], False, 'control calibration no final')
    audit.equal(calibration['controls']['pages']['knownness_threshold'], .4, 'frozen knownness threshold')
    audit.equal(result['threshold'], .4, 'reported frozen threshold')
    base = result_path.parent.resolve()
    ledger = result['artifacts']
    actual_files = {str(p.relative_to(base)) for p in base.rglob('*') if p.is_file() and p != result_path}
    audit.equal(set(ledger), actual_files, 'complete output artifact coverage')
    for relative, sha in ledger.items():
        path = (base / relative).resolve()
        audit.require(path.is_relative_to(base), 'escaping artifact path')
        audit.hash(path, sha)
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    checkpoint = torch.load(cfg['authority']['corrected_pages_checkpoint']['path'], map_location='cpu', weights_only=False)
    audit.equal(checkpoint['experiment_context'], context, 'checkpoint atomic context')
    audit.equal(checkpoint['protocol_sha256'], context['protocol_sha256'], 'checkpoint protocol')
    audit.equal(checkpoint['control'], 'pages', 'checkpoint control')
    audit.equal(checkpoint['smoke'], False, 'non-smoke checkpoint')
    parser_path = HERE.parent / 'cbf0' / 'schema_grounding_compiler.py'
    audit.hash(parser_path, cfg['source_hashes'][str(parser_path)])
    spec = importlib.util.spec_from_file_location('closed_grammar_only', parser_path)
    grammar = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grammar)
    source = Path(cfg['source_root'])
    states = audit.load(source / 'states.json')
    held = [s for s in states if s['source_split'] != 'train']
    audit.require(bool(held), 'nonzero held states')
    literals = audit.load(source / 'literal_cases.json')
    cohorts = {phase: [dict(q, stratum='alias') for q in audit.load(source / item['atoms'])] +
               [dict(q, stratum='alias_composition') for q in audit.load(source / item['compositions'])] + literals
               for phase, item in cfg['cohorts'].items()}
    texts = sorted({q['text'][a['span'][0]:a['span'][1]] for qs in cohorts.values() for q in qs
                    for a in grammar.parse_criterion(q['text']) if a['literal_axis'] is None})
    definitions = audit.load(HERE.parent / 'cbf0' / 'schema_support_protocol.json')
    page_texts = [definitions['field_definitions'][name] for name in definitions['schema']]
    inputs = audit.load(base / 'encoder_inputs.json')
    arrays = {}
    for modality, strings in (('page', page_texts), ('query', texts)):
        audit.equal(inputs[modality]['texts'], strings, 'exact original encoder strings')
        arrays[modality] = np.load(base / (modality+'_features.npy'), allow_pickle=False)
        ids = np.load(base / (modality+'_input_ids.npy'), allow_pickle=False)
        mask = np.load(base / (modality+'_attention_mask.npy'), allow_pickle=False)
        audit.require(arrays[modality].shape == (len(strings), 384) and arrays[modality].dtype == np.float32
                      and np.isfinite(arrays[modality]).all(), 'raw feature contract')
        audit.require(ids.shape == mask.shape == (len(strings), 128) and np.issubdtype(ids.dtype, np.integer)
                      and np.isin(mask, [0, 1]).all(), 'token array contract')
        audit.equal(inputs[modality]['token_count'], int(mask.sum()), 'token coverage')
        audit.counts[modality+'_tokens'] = int(mask.sum())
    audit.equal(len(page_texts), 4, 'four field pages')
    lineage = result['encoder_lineage']
    audit.equal(lineage['protocol_sha256'], context['protocol_sha256'], 'encoder atomic protocol')
    audit.hash(HERE / 'ephemeral_pages_features.py', lineage['feature_code_sha256'])
    audit.hash(HERE / 'ephemeral_pages_model.py', lineage['reader_code_sha256'])
    canonical_root = Path(lineage['canonical_root'])
    for relative, sha in lineage['canonical_files_sha256'].items():
        path = Path(relative)
        audit.hash(path if path.is_absolute() else canonical_root / path, sha)
    audit.equal(lineage['encoder_repo'], encoder_spec['repo'], 'frozen encoder repository')
    audit.equal(lineage['token_limit'], 128, 'encoder token limit')
    audit.equal(lineage['truncation'], False, 'no token truncation')
    audit.equal(lineage['encoder_revision'], encoder_spec['revision'], 'frozen encoder revision')
    audit.equal(lineage['encoder_weight_sha256'], encoder_spec['weight_sha256'], 'frozen encoder weights')
    audit.equal(lineage['encoder_parameters_sha256_before'], lineage['encoder_parameters_sha256_after'], 'unchanged encoder parameters')
    audit.equal(result['counters'], {'forward_counts': {'page': 1, 'query': math.ceil(len(texts)/32), 'cross': 0},
                                   'encoded_counts': {'page': 4, 'query': len(texts), 'cross': 0},
                                   'cache_hits': 0, 'unique_page_texts': 4, 'unique_query_texts': len(texts)}, 'encoder coverage')
    raw_outputs = replay_reader(audit, checkpoint, arrays['query'], arrays['page'])
    saved_relations = audit.load(base / 'relations.json')
    audit.equal(len(saved_relations), len(texts), 'relation coverage')
    resolutions = {}
    for text, raw, saved in zip(texts, raw_outputs, saved_relations):
        audit.equal(saved['text'], text, 'relation text order')
        audit.equal(set(saved['full_reader_output']), set(raw), 'full output keys')
        for key, values in raw.items():
            audit.numeric(values, saved['full_reader_output'][key], 'reader.'+key)
        rel, direction = np.asarray(raw['relevance_logits'][0]), np.asarray(raw['direction'][0])
        known = float(sigmoid(raw['known_logits'][0]))
        top = np.flatnonzero(rel == rel.max())
        reason = ('nonfinite_full_output' if not all(np.isfinite(v).all() for v in raw.values()) else
                  'below_unchanged_knownness_threshold' if known < .4 else
                  'exact_relevance_tie' if len(top) != 1 else
                  'zero_direction' if direction[top[0]] == 0 else None)
        resolution = None if reason else {'axis': int(top[0]), 'sign': 1 if direction[top[0]] > 0 else -1}
        audit.equal(saved['resolution'], resolution, 'independent relation decode')
        audit.equal(saved['unknown_reason'], reason, 'independent unknown reason')
        audit.numeric(known, saved['known_probability'], 'known probability')
        resolutions[text] = resolution
        audit.counts['relations'] += 1
    reports = {}
    for phase, queries in cohorts.items():
        original_root = source / cfg['cohorts'][phase]['selected_output']
        originals_list = audit.rows(original_root / 'decisions.jsonl')
        originals = {(r['criterion_id'], r['scenario_id']): r for r in originals_list}
        audit.equal(len(originals), len(originals_list), 'unique original decisions')
        saved_list = audit.rows(base / phase / 'decisions.jsonl')
        saved_rows = {(r['criterion_id'], r['scenario_id']): r for r in saved_list}
        audit.equal(len(saved_rows), len(saved_list), 'unique replay decisions')
        expected_keys = {(q['id'], s['id']) for q in queries for s in held}
        audit.equal(set(saved_rows), expected_keys, 'exact decision Cartesian coverage')
        audit.equal(set(originals), expected_keys, 'original closed coverage')
        compiled_saved = audit.load(base / phase / 'compiled_queries.json')
        audit.equal(len(compiled_saved), len(queries), 'compiler coverage')
        rebuilt = {}
        for query, compiled in zip(queries, compiled_saved):
            coefficients, atoms, unknown = [0]*4, [], False
            for atom in grammar.parse_criterion(query['text']):
                text = query['text'][slice(*atom['span'])]
                resolution = resolutions[text] if atom['literal_axis'] is None else {'axis': atom['literal_axis'], 'sign': atom['literal_sign']}
                atoms.append(dict(atom, term=text, resolution=resolution))
                if resolution is None:
                    unknown = True
                else:
                    audit.require(type(atom['factor']) is int, 'integer literal coefficient')
                    coefficients[resolution['axis']] += atom['factor'] * resolution['sign']
            vector = None if unknown else coefficients
            audit.equal(compiled, {'query': query, 'vector': vector, 'atoms': atoms}, 'independent compiler replay')
            if query['stratum'] == 'literal':
                audit.equal(vector, query['weights'], 'original literal coefficients')
            for state in held:
                key = (query['id'], state['id'])
                old, saved = originals[key], saved_rows[key]
                integer_facts = [facts(audit, c) for c in state['candidates']]
                audit.require(len(query['weights']) == 4 and all(type(v) is int for v in query['weights']), 'teacher integer vector')
                gold_scores, gold, _ = choice(query['weights'], integer_facts)
                scores, top, winner = choice(vector, integer_facts)
                audit.equal(old['teacher_scores'], gold_scores, 'original teacher integer scores')
                audit.equal(old['teacher_top_set'], gold, 'original teacher maximal set')
                old_scores = old['scores']
                old_top = [] if old_scores is None else [i for i, v in enumerate(old_scores) if v == max(old_scores)]
                row = {'criterion_id': query['id'], 'scenario_id': state['id'], 'stratum': query['stratum'],
                       'query_provenance': query, 'state_provenance': state, 'exact_integer_facts': integer_facts,
                       'vector': vector, 'scores_integer': scores, 'teacher_scores': gold_scores,
                       'teacher_top_set': gold, 'predicted_top_set': top, 'winner': winner,
                       'stable_ordinals': list(range(len(integer_facts))), 'maximal_set_correct': top == gold,
                       'chosen_in_gold': winner in gold, 'unknown': vector is None,
                       'original_selected_decision': old, 'original_predicted_top_set': old_top,
                       'closed_maximal_set_difference': top != old_top,
                       'closed_concrete_difference': winner != old['winner']}
                audit.equal(saved, row, 'full independent decision row')
                if query['stratum'] == 'literal':
                    audit.require(top == gold == old_top, 'literal exact maximal set')
                rebuilt[key] = row
                audit.counts[phase+'_decisions'] += 1
                audit.counts[phase+'_integer_scores'] += len(gold_scores) + (0 if scores is None else len(scores))
        original_swaps = audit.rows(original_root / 'teacher_changing_swaps.jsonl')
        saved_swaps = audit.rows(base / phase / 'teacher_changing_swaps.jsonl')
        audit.equal(len(saved_swaps), len(original_swaps), 'exact causal coverage')
        swaps = []
        for old, saved in zip(original_swaps, saved_swaps):
            before, after = (rebuilt[(old[k], old['scenario_id'])] for k in ('before', 'after'))
            audit.require(before['teacher_top_set'] != after['teacher_top_set'], 'actual teacher-changing swap')
            correct = after['winner'] in after['teacher_top_set']
            changed = before['winner'] != after['winner']
            row = {'original_selected_swap': old, 'scenario_id': old['scenario_id'], 'before': old['before'],
                   'after': old['after'], 'correct_new': correct, 'student_changed': changed,
                   'changed_to_new': correct and changed, 'closed_correct_new_difference': correct != bool(old['top_set_new'])}
            audit.equal(saved, row, 'independent causal correct-new')
            swaps.append(row)
        rows = list(rebuilt.values())
        literal_rows = [r for r in rows if r['stratum'] == 'literal']
        audit.require(bool(literal_rows) and bool(swaps), 'nonzero literal and causal denominator')
        report = {'literal_rows': len(literal_rows),
                  'literal_maximal_set_accuracy': sum(r['maximal_set_correct'] for r in literal_rows)/len(literal_rows),
                  'decision_rows': len(rows), 'strata': {}, 'causal_n': len(swaps),
                  'causal_correct_new': sum(r['correct_new'] for r in swaps)/len(swaps)}
        audit.equal(report['literal_maximal_set_accuracy'], 1.0, 'actual literal accuracy 100 percent')
        for stratum in ('alias', 'alias_composition', 'literal'):
            selected = [r for r in rows if r['stratum'] == stratum]
            audit.require(bool(selected), 'nonzero stratum')
            report['strata'][stratum] = {'n': len(selected),
                'maximal_set_accuracy': sum(r['maximal_set_correct'] for r in selected)/len(selected),
                'chosen_in_gold': sum(r['chosen_in_gold'] for r in selected)/len(selected),
                'unknown': sum(r['unknown'] for r in selected),
                'closed_maximal_set_differences': sum(r['closed_maximal_set_difference'] for r in selected)}
        audit.counts[phase+'_causal_swaps'] = len(swaps)
        audit.counts[phase+'_closed_concrete_differences'] = sum(r['closed_concrete_difference'] for r in rows)
        audit.counts[phase+'_closed_correct_new_differences'] = sum(r['closed_correct_new_difference'] for r in swaps)
        reports[phase] = report
    audit.equal(reports, result['reports'], 'all independently reconstructed report strata')
    return reports, {'numpy': np.__version__, 'torch_deserialization_only': torch.__version__}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'result', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args(argv)
    output = args.output.absolute()
    repo = HERE.parents[1]
    if output.resolve().is_relative_to(repo) or output.exists():
        parser.error('output must be an exclusive new private receipt outside the repository')
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    audit = Audit()
    reports, versions, status = {}, {}, 'failed'
    try:
        audit.hash(Path(__file__))
        reports, versions = verify(audit, args.protocol.resolve(), args.result.resolve())
        status = 'verified_atomic_closed_diagnostic_only'
    except Exception as exc:
        if not audit.errors:
            audit.errors.append(type(exc).__name__)
    receipt = {'schema': 'vey.eca.atomic-closed-independent.v1', 'status': status,
               'artifact_sha256': audit.hashes, 'exact_counts': dict(audit.counts),
               'bounded_errors': audit.errors, 'maximum_reader_absolute_error': audit.max_numeric_error,
               'reader_tolerance': {'absolute': 2e-5, 'relative': 2e-5}, 'reports': reports,
               'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                               'packages': versions, 'thread_limits': {k: os.environ[k] for k in
                               ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'ARROW_NUM_THREADS', 'TOKENIZERS_PARALLELISM')}},
               'encoder_forwards': 0, 'model_forwards': 0, 'fitting': False,
               'calibration_changes': False, 'final_selection': False, 'promotion': False,
               'claim': 'Closed CBF8 diagnostic reconstruction only; no fresh/current ECA final credit.'}
    receipt['environment_sha256'] = hashlib.sha256(
        json.dumps(receipt['environment'], sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as stream:
        json.dump(receipt, stream, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': status, 'checks': audit.counts['assertions'], 'error_count': len(audit.errors)}))
    return 0 if status.startswith('verified_') else 1


if __name__ == '__main__':
    raise SystemExit(main())
