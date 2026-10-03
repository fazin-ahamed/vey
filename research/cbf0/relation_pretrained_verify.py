"""Independently replay CBF-7 persisted evidence on CPU; never encode or fit models."""
import os

for _variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_variable] = '1'
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
from scipy.special import erf, logsumexp
from transformers import AutoTokenizer

from schema_grounding_compiler import compile_criterion, parse_criterion

PROTOCOL = Path(__file__).with_name('relation_pretrained_protocol.json')
SCHEMA = ('reliability', 'purchase expense', 'operating expense', 'convenience')
ARM_ORDER = ('stock_bare', 'stock_higher_cls', 'nli_xsmall_bare_cls',
             'nli_xsmall_higher_cls', 'nli_xsmall_higher_pooler', 'nli_xsmall_native',
             'nli_small_higher_pooler', 'nli_small_native')
FROZEN_VEY2 = 'e6b046ffbd138cbdbfb2f89c6ae77525fe6b0b18'


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


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
        require(isinstance(actual, (int, float)) and math.isfinite(actual) and
                abs(actual - expected) <= tolerance, f'{label}: {actual!r} != {expected!r}')
    else:
        require(actual == expected, f'{label}: {actual!r} != {expected!r}')


def terms(text):
    return [text[t['context_span'][0]:t['context_span'][1]].rstrip('.') + '.'
            for t in parse_criterion(text) if t['literal_axis'] is None]


def contexts(atoms, compositions, g0):
    return sorted({c['text'] for c in atoms + g0} |
                  {q for c in compositions for q in terms(c['text'])})


def tokens(text):
    return tuple(re.findall(r'[a-z0-9]+', text.casefold()))


def contains(text, phrase):
    return bool(phrase) and any(text[i:i + len(phrase)] == phrase
                               for i in range(len(text) - len(phrase) + 1))


def corpus_checks(root, cfg):
    source = Path(cfg['source_root'])
    manifest = load(root / 'corpus_manifest.json')
    inventory_path = PROTOCOL.with_name('schema_relation_result_manifest.json')
    inventory = load(inventory_path)
    require(manifest['protocol_sha256'] == sha(PROTOCOL), 'protocol lineage')
    require(manifest['source_inventory_sha256'] == sha(inventory_path), 'source inventory lineage')
    require(inventory['data_root'] == str(source), 'source root')
    require(manifest['source_measurement_git_revision'] == inventory['measurement_git_revision'], 'source revision')
    require(load(source / 'verification.json')['all_checks_pass'], 'CBF6 verification')
    checked = {}
    for name, expected in manifest['files'].items():
        require(sha(root / name) == expected, 'corpus hash ' + name)
        checked[str(root / name)] = expected
    for name, expected in manifest['source_verified_files'].items():
        require(sha(source / name) == expected == inventory['files'][name]['sha256'], 'source hash ' + name)
        if (root / name).exists():
            require((root / name).read_bytes() == (source / name).read_bytes(), 'source reuse ' + name)
        checked[str(source / name)] = expected
    require(sha(source / 'verification.json') == inventory['files']['verification.json']['sha256'], 'source verification hash')
    groups = {name: load(root / (name + '.json')) for name in
              ('training_atoms', 'validation_atoms', 'literal_holdout_atoms', 'development_atoms',
               'development_compositions', 'literal_cases', 'states', 'final_atoms', 'final_compositions')}
    atoms, compositions = groups['final_atoms'], groups['final_compositions']
    require(len(atoms) == len(compositions) == 128, 'fresh pool counts')
    require(len({c['id'] for c in atoms + compositions}) == 256, 'fresh unique IDs')
    require(Counter((c['axis'], c['sign']) for c in atoms) ==
            Counter({(a, s): 16 for a in range(4) for s in (-1, 1)}), 'fresh direction balance')
    for cohort in (atoms, compositions):
        pairs = {}
        for c in cohort:
            pairs.setdefault(c['pair_id'], []).append(c)
        require(len(pairs) == 64 and all(len(p) == 2 and p[0]['weights'] == [-w for w in p[1]['weights']]
                                       for p in pairs.values()), 'semantic reversal groups')
    exclusions = []
    audit = load(root / 'final_corpus_audit.json')
    require(audit['provenance']['generator_sha256'] == sha(PROTOCOL.with_name('relation_pretrained_corpus.py')), 'generator hash')
    require(audit['provenance']['protocol_sha256'] == sha(PROTOCOL), 'generator protocol')
    for name, reference in audit['source_lineage'].items():
        require(sha(source / name) == reference['sha256'] == inventory['files'][name]['sha256'], 'lexical source ' + name)
        rows = load(source / name)
        require(len(rows) == reference['rows'], 'lexical source row count')
        for c in rows:
            exclusions.append(tokens(c['text']))
            for t in parse_criterion(c['text']):
                lo, hi = t['span']
                exclusions.append(tokens(c['text'][lo:hi]))
    all_texts = [tokens(c['text']) for c in atoms + compositions]
    require(len(set(all_texts)) == 256, 'normalized fresh duplicate')
    semantic = [tokens(c['text'][t['span'][0]:t['span'][1]]) for c in atoms for t in parse_criterion(c['text'])]
    require(len(set(semantic)) == 128, 'normalized atomic duplicate')
    for c, text in zip(atoms + compositions, all_texts):
        require(not any(contains(text, tokens(f)) for f in SCHEMA), 'literal schema leakage ' + c['id'])
        require(not any(contains(text, old) for old in exclusions), 'old full/semantic phrase leakage ' + c['id'])
    text_sets = [{c['text'] for c in groups[n]} for n in
                 ('training_atoms', 'validation_atoms', 'literal_holdout_atoms', 'final_atoms')]
    require(all(not text_sets[i] & text_sets[j] for i in range(4) for j in range(i)), 'literal/final disjointness')
    by_text = {c['text'].rstrip('.') + '.': c for c in atoms}
    uses, coverage = Counter(), Counter()
    grammar_count = 0
    for c in atoms + compositions:
        parsed = parse_criterion(c['text'])
        require(len(parsed) == (1 if c in atoms else 2), 'fresh grammar arity')
        vector = [0] * 4
        for t in parsed:
            require(t['literal_axis'] is None, 'fresh literal dispatch')
            lo, hi = t['context_span']
            a = by_text[c['text'][lo:hi].rstrip('.') + '.']
            vector[a['axis']] += t['factor'] * a['sign']
            if c in compositions:
                uses[a['id']] += 1
        require(vector == c['weights'], 'fresh teacher coefficients ' + c['id'])
        if c in compositions:
            components = c['components']
            require(len(components) == 2, 'composition components')
            key = []
            for component, term in zip(components, parsed):
                atom = next(a for a in atoms if a['id'] == component['atom_id'])
                lo, hi = term['context_span']
                require(c['text'][lo:hi].rstrip('.') == atom['text'].rstrip('.') and
                        component['factor'] == term['factor'], 'composition component span')
                key.append((atom['axis'], atom['sign'], term['factor']))
            coverage[tuple(key)] += 1
        grammar_count += 1
    require(set(uses) == {a['id'] for a in atoms} and set(uses.values()) == {2}, 'all final atoms used twice')
    expected_coverage = {tuple(zip(axes, signs, factors)) for axes in itertools.combinations(range(4), 2)
                         for signs in itertools.product((-1, 1), repeat=2)
                         for factors in itertools.product((1, 2), repeat=2)}
    require(set(coverage) == expected_coverage, 'composition field/sign/weight coverage')
    meaning = load(root / 'meaning_audit.json')
    predictions = load(root / 'blind_meaning_predictions.json')
    key = load(root / 'blind_meaning_key.json')
    questions = load(root / 'blind_meaning_questions.json')
    require(len(predictions) == len(questions) == len(key) == 128, 'blind counts')
    require(meaning['blind_audit_sha256'] == sha(root / 'final_corpus_audit.json'), 'blind audit hash')
    require(meaning['blind_predictions_sha256'] == sha(root / 'blind_meaning_predictions.json'), 'blind predictions hash')
    require(meaning['final_atoms_sha256'] == sha(root / 'final_atoms.json') and
            meaning['final_compositions_sha256'] == sha(root / 'final_compositions.json'), 'blind corpus hashes')
    qmap = {q['review_id']: q for q in questions}
    amap = {a['id']: a for a in atoms}
    require(len(qmap) == len({p['review_id'] for p in predictions}) == 128, 'blind unique IDs')
    require({v['atom_id'] for v in key.values()} == set(amap), 'blind all atoms covered')
    for p in predictions:
        k = key[p['review_id']]
        a = amap[k['atom_id']]
        require(qmap[p['review_id']]['text'] == a['text'] and
                (p['axis'], p['sign']) == (k['axis'], k['sign']) == (a['axis'], a['sign']) and
                not p['ambiguous'] and k['weights'] == a['weights'], 'blind accepted meaning ' + p['review_id'])
    require(meaning['all_meanings_accepted'] and meaning['correct'] == meaning['n'] == 128 and meaning['ambiguous'] == 0,
            'blind aggregate')
    for state in groups['states']:
        facts = []
        for text in state['candidates']:
            fields = re.findall(r'([^:;]+): (\d+) percent', text)
            values = {name.strip(): int(value) for name, value in fields}
            require(len(fields) == 4 and set(values) == set(SCHEMA), 'integer state syntax')
            facts.append([values[field] for field in SCHEMA])
        require(facts == state['parsed_facts'] and len(facts) == state['K'], 'integer state facts ' + state['id'])
    require(len(groups['states']) == 64 and sum(s['source_split'] != 'train' for s in groups['states']) == 32,
            'exact state cohort counts')
    return groups, checked, dict(blind_meanings=128, lexical_questions=256, grammar_queries=grammar_count,
                                 atom_reversal_groups=64, composition_reversal_groups=64, integer_states=64)


def re_tokenize(evidence, tokenizer, text_order, format_name):
    require(tokenizer.is_fast and evidence['truncation'] is False and evidence['candidate_forwards'] == 0, 'tokenizer/truncation')
    expected_pairs = [(q, f) for q in text_order for f in SCHEMA]
    require([(p['criterion'], p['field']) for p in evidence['pairs']] == expected_pairs, 'canonical pair row order')
    require(evidence.get('format', 'bare') == format_name, 'pair format')
    for p in evidence['pairs']:
        h = p['field'] if format_name == 'bare' else 'Higher ' + p['field'] + ' is preferred.'
        require(p.get('hypothesis', h) == h, 'canonical hypothesis only')
    offset, token_count, maximum, lengths = 0, 0, 0, []
    for b in evidence['batches']:
        require(b['start'] == offset and 0 < b['count'] <= 32, 'token batch coverage')
        rows = evidence['pairs'][offset:offset + b['count']]
        second = [p['field'] if format_name == 'bare' else 'Higher ' + p['field'] + ' is preferred.' for p in rows]
        encoded = tokenizer([p['criterion'] for p in rows], text_pair=second, add_special_tokens=True,
                            padding=True, truncation=False, return_attention_mask=True, return_tensors='np')
        for name in ('input_ids', 'attention_mask', 'token_type_ids'):
            if name in encoded:
                require(name in b and np.array_equal(encoded[name], b[name]), 'retokenized ' + name)
            else:
                require(name not in b, 'unexpected token_type_ids')
        n = encoded['attention_mask'].sum(1).tolist()
        require(encoded['input_ids'].shape[1] <= 128 and np.all(encoded['input_ids'][:, 0] == tokenizer.cls_token_id), 'token limit/CLS')
        if 'lengths' in b:
            require(b['lengths'] == n, 'batch token lengths')
        lengths.extend(n)
        offset += len(rows)
        token_count += sum(n)
        maximum = max(maximum, encoded['input_ids'].shape[1])
    require(offset == len(expected_pairs) and token_count == evidence['tokens'] and
            len(evidence['batches']) == evidence['forwards'], 'token totals')
    if 'lengths' in evidence:
        require(evidence['lengths'] == lengths, 'all token lengths')
    if 'maximum_tokens' in evidence:
        require(evidence['maximum_tokens'] == maximum, 'maximum tokens')
    return dict(pairs=offset, valid_tokens=token_count, batches=len(evidence['batches']))


def feature_rows(features, evidence, texts):
    lookup = {(p['criterion'], p['field']): i for i, p in enumerate(evidence['pairs'])}
    require(len(lookup) == len(evidence['pairs']), 'duplicate feature pair')
    return features[[lookup[(q, f)] for q in texts for f in SCHEMA]]


def head_probabilities(features, normalizer, weights):
    # Match the persisted FP32 head inputs, then independently accumulate in FP64.
    values = ((features - normalizer['mean']) / normalizer['std']).astype(np.float32).astype(np.float64)
    logits = values @ weights['weight'].astype(np.float64).T + weights['bias'].astype(np.float64)
    exponent = np.exp(logits - logits.max(1, keepdims=True))
    return exponent / exponent.sum(1, keepdims=True), logits


def resolve(matrix):
    classes = matrix.argmax(1) - 1
    active = [i for i in range(4) if classes[i] != 0]
    if not active:
        return None
    axis = max(active, key=lambda i: (max(matrix[i, 0], matrix[i, 2]) - matrix[i, 1], -i))
    return dict(axis=axis, sign=int(classes[axis]), relation_matrix=matrix.tolist(),
                row_classes=classes.tolist(), nonzero_fields=len(active))


def summary(rows):
    require(bool(rows), 'empty decision summary')
    return dict(n=len(rows), **{key: float(np.mean([r[key] for r in rows])) for key in
                               ('top1', 'exact_winner', 'pairwise')},
                coverage=float(np.mean([r['covered'] for r in rows])))


def verify_stage(root, tag, atoms, compositions, literals, states, H, evidence, weights, normalizer, g0, cfg):
    output = root / tag
    texts = contexts(atoms, compositions, g0)
    features = feature_rows(H, evidence, texts)
    p, _ = head_probabilities(features, normalizer, weights)
    matrices = {q: p[4 * i:4 * i + 4] for i, q in enumerate(texts)}
    persisted = load(output / 'relation_matrices.json')
    require([r['text'] for r in persisted] == texts, tag + ': all relation matrices')
    resolutions, maximum = {}, 0.0
    for r in persisted:
        require(r['fields'] == list(SCHEMA) and r['classes'] == [-1, 0, 1], 'matrix axes/classes')
        matrix = np.asarray(r['matrix'])
        require(matrix.shape == (4, 3) and np.isfinite(matrix).all() and np.all(matrix >= 0) and
                np.allclose(matrix.sum(1), 1, atol=1e-6), 'valid class masses')
        error = float(np.max(np.abs(matrix - matrices[r['text']])))
        require(error < 2e-5, tag + ': head probability discrepancy ' + str(error))
        maximum = max(maximum, error)
        saved_resolution = resolve(matrix)
        compare(r['resolution'], saved_resolution, 'persisted resolution')
        independent = resolve(matrices[r['text']])
        require((None if independent is None else (independent['axis'], independent['sign'])) ==
                (None if saved_resolution is None else (saved_resolution['axis'], saved_resolution['sign'])), 'independent UNKNOWN/axis/sign')
        resolutions[r['text']] = independent
    axis = sum(resolutions[c['text']] is not None and resolutions[c['text']]['axis'] == c['axis'] for c in atoms)
    joint = sum(resolutions[c['text']] is not None and
                (resolutions[c['text']]['axis'], resolutions[c['text']]['sign']) == (c['axis'], c['sign']) for c in atoms)
    atomic = dict(n=len(atoms), axis_correct=axis, axis_accuracy=axis / len(atoms), sign_given_axis_correct=joint,
                  sign_given_axis_denominator=axis, conditional_sign_accuracy=joint / axis if axis else None,
                  joint_correct=joint, joint_accuracy=joint / len(atoms), unknown=sum(resolutions[c['text']] is None for c in atoms))
    gold = np.array([[c['sign'] if j == c['axis'] else 0 for j in range(4)] for c in g0])
    pred = np.array([matrices[c['text']].argmax(1) - 1 for c in g0])
    correct = gold == pred
    relation = dict(n=int(gold.size), correct=int(correct.sum()), accuracy=float(correct.mean()),
                    matched_n=int((gold != 0).sum()), matched_accuracy=float(correct[gold != 0].mean()),
                    per_class={str(k): dict(n=int((gold == k).sum()), accuracy=float(correct[gold == k].mean())) for k in (-1, 0, 1)})
    queries = [dict(c, stratum='alias') for c in atoms] + [dict(c, stratum='alias_composition') for c in compositions] + literals
    compiled_saved = load(output / 'compiled_queries.json')
    require([c['id'] for c in compiled_saved] == [c['id'] for c in queries], 'compiled query completeness/order')
    vectors, compiled = {}, []
    semantic_callbacks = literal_callbacks = 0
    for c, saved in zip(queries, compiled_saved):
        vector, unknown, records = [0] * 4, False, []
        for term in parse_criterion(c['text']):
            lo, hi = term['span']
            if term['literal_axis'] is None:
                require(c['stratum'] != 'literal', 'neural literal callback')
                semantic_callbacks += 1
                a, b = term['context_span']
                resolution = resolutions[c['text'][a:b].rstrip('.') + '.']
            else:
                resolution = dict(axis=term['literal_axis'], sign=term['literal_sign'])
            if resolution is None:
                unknown = True
            else:
                vector[resolution['axis']] += term['factor'] * resolution['sign']
            # Matrix values in compiler IR came from FP32 probabilities; check them separately above.
            recorded_resolution = resolution
            if term['literal_axis'] is None:
                a, b = term['context_span']
                recorded_resolution = next(r['resolution'] for r in persisted if r['text'] == c['text'][a:b].rstrip('.') + '.')
            records.append(dict(term, term=c['text'][lo:hi], resolution=recorded_resolution))
        vector = None if unknown else vector
        expected = dict(id=c['id'], stratum=c['stratum'], vector=vector,
                        coefficient_exact=vector == c['weights'], atoms=records)
        compare(saved, expected, 'compiled IR')
        vectors[c['id']] = vector
        compiled.append(expected)
        if c['stratum'] == 'literal':
            def forbidden(_term):
                raise AssertionError('Neural literal dispatch')
            exact, _ = compile_criterion(c['text'], forbidden)
            require(exact == vector == c['weights'], 'literal exact coefficients')
    test_states = [s for s in states if s['source_split'] != 'train']
    rows, cached = [], {}
    invariance = dict(comparisons=0, numeric_comparisons=0, unknown_comparisons=0, score_mismatches=0, winner_mismatches=0)
    rng = np.random.default_rng(7)
    with (output / 'decisions.jsonl').open() as stream:
        for c in queries:
            for state in test_states:
                line = stream.readline()
                require(bool(line), 'missing decision')
                actual = json.loads(line)
                facts = np.asarray(state['parsed_facts'], dtype=np.int64)
                gold = facts @ np.asarray(c['weights'], dtype=np.int64)
                def choose(values, candidates=facts):
                    return None if values is None else max(range(len(values)), key=lambda i: (int(values[i]), tuple(candidates[i])))
                teacher = choose(gold)
                vector = vectors[c['id']]
                values = None if vector is None else facts @ np.asarray(vector, dtype=np.int64)
                winner = choose(values)
                a, b = np.triu_indices(state['K'], 1)
                expected = dict(criterion_id=c['id'], stratum=c['stratum'], scenario_id=state['id'], source_split=state['source_split'],
                                K=state['K'], teacher=teacher, teacher_scores=gold.tolist(), teacher_top_set=np.flatnonzero(gold == gold.max()).tolist(),
                                winner=winner, scores=None if values is None else (values / 100).tolist(), covered=winner is not None,
                                top1=int(winner is not None and gold[winner] == gold.max()), exact_winner=int(winner == teacher),
                                pairwise=0.0 if values is None else float(np.mean(np.sign(values[a] - values[b]) == np.sign(gold[a] - gold[b]))))
                compare(actual, expected, 'decision')
                rows.append(expected)
                cached[(c['id'], state['id'])] = expected
                for order in (np.arange(state['K'])[::-1], rng.permutation(state['K'])):
                    invariance['comparisons'] += 1
                    if values is None:
                        invariance['unknown_comparisons'] += 1
                    else:
                        other = facts[order] @ np.asarray(vector, dtype=np.int64)
                        selected = choose(other, facts[order])
                        invariance['numeric_comparisons'] += 1
                        invariance['winner_mismatches'] += int(int(order[selected]) != winner)
                        invariance['score_mismatches'] += int(not np.array_equal(other[np.argsort(order)], values))
        require(not stream.readline(), 'extra/duplicate decision rows')
    strata = ('alias', 'alias_composition', 'literal')
    def included(r, stratum):
        return r['stratum'] == stratum and (stratum == 'literal' or r['source_split'] ==
                                            ('unseen_wording' if stratum == 'alias' else 'unseen_criterion'))
    summaries = {s: summary([r for r in rows if included(r, s)]) for s in strata}
    per_k = {s: {str(k): summary([r for r in rows if included(r, s) and r['K'] == k]) for k in (2, 4, 8, 16)} for s in strata}
    representatives = {}
    for c in sorted((c for c in queries if c['stratum'] != 'literal'), key=lambda c: c['text']):
        divisor = math.gcd(*c['weights'])
        require(divisor > 0, 'nonzero causal ray')
        representatives.setdefault(tuple(w // divisor for w in c['weights']), c)
    reps = list(representatives.values())
    compare(load(output / 'causal_representatives.json'), [dict(id=c['id'], text=c['text'], weights=c['weights']) for c in reps], 'causal representatives')
    counts, eligible, n = Counter(), 0, 0
    with (output / 'teacher_changing_swaps.jsonl').open() as stream:
        for state in test_states:
            for first, second in itertools.permutations(reps, 2):
                eligible += 1
                before, after = cached[(first['id'], state['id'])], cached[(second['id'], state['id'])]
                if before['teacher'] == after['teacher']:
                    continue
                a, b = before['winner'], after['winner']
                correct_new = int(b == after['teacher'])
                flags = dict(correct_new=correct_new, student_changed=int(a != b),
                             changed_to_new=int(correct_new and a is not None and b is not None and a != b),
                             both_endpoints=int(a == before['teacher'] and correct_new), top_set_new=after['top1'])
                expected = dict(scenario_id=state['id'], before=first['id'], after=second['id'],
                                teacher_before=before['teacher'], teacher_after=after['teacher'], **flags)
                line = stream.readline()
                require(bool(line), 'missing teacher-changing causal row')
                compare(json.loads(line), expected, 'teacher-changing causal row')
                counts.update(flags)
                n += 1
        require(not stream.readline(), 'extra/duplicate causal rows')
    require(n > 0, 'causal denominator')
    causal = dict(representatives=len(reps), eligible_pairs=eligible, n=n,
                  **{k + '_rate': counts[k] / n for k in ('correct_new', 'student_changed', 'changed_to_new', 'both_endpoints', 'top_set_new')})
    thresholds = cfg['gates']
    gates = dict(G0=relation['accuracy'] >= thresholds['G0_literal_relation'], G1=atomic['axis_accuracy'] >= thresholds['G1_axis'],
                 G2=axis > 0 and joint / axis >= thresholds['G2_sign_given_axis'], G3=atomic['joint_accuracy'] >= thresholds['G3_joint_atom'],
                 G4=summaries['alias']['top1'] >= thresholds['G4_alias_decision'], G5=summaries['alias_composition']['top1'] >= thresholds['G5_alias_composition'],
                 G6=causal['correct_new_rate'] >= thresholds['G6_correct_new'],
                 G7=summaries['literal']['top1'] == summaries['literal']['exact_winner'] == summaries['literal']['pairwise'] == thresholds['G7_exact'] and
                 literal_callbacks == invariance['score_mismatches'] == invariance['winner_mismatches'] == 0)
    result = dict(literal_relation=relation, atoms=atomic, summary=summaries, per_K=per_k, causal=causal, permutation=invariance,
                  compiler_accuracy={s: dict(n=sum(c['stratum'] == s for c in compiled),
                                             correct=sum(c['coefficient_exact'] for c in compiled if c['stratum'] == s)) for s in strata},
                  semantic_callbacks=semantic_callbacks, literal_callbacks=literal_callbacks, gates=gates, passed=all(gates.values()), decision_rows=len(rows))
    compare(load(output / 'results.json'), result, tag + ': all statistics')
    proof = dict(head_probability_maximum_error=maximum, relation_matrices=len(texts), atomic_queries=len(atoms),
                 compiled_queries=len(queries), decision_rows=len(rows), permutation_comparisons=invariance['comparisons'],
                 causal_rows=n, gates=gates, passed=result['passed'], statistics_reconstructed=True)
    return result, proof, rows, resolutions


def verify_intervals(root, atoms, compositions, rows, resolutions):
    pairs = sorted({c['pair_id'] for c in atoms})
    axes, joints, decisions = [], [], []
    for pair in pairs:
        group = [c for c in atoms if c['pair_id'] == pair]
        require(len(group) == 2, 'bootstrap atom group')
        axes.append(sum(resolutions[c['text']] is not None and resolutions[c['text']]['axis'] == c['axis'] for c in group))
        joints.append(sum(resolutions[c['text']] is not None and
                          (resolutions[c['text']]['axis'], resolutions[c['text']]['sign']) == (c['axis'], c['sign']) for c in group))
        ids = {c['id'] for c in group}
        decisions.append(np.mean([r['top1'] for r in rows if r['criterion_id'] in ids and r['source_split'] == 'unseen_wording']))
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(pairs), (10000, len(pairs)))
    a, j = np.asarray(axes)[draws].sum(1), np.asarray(joints)[draws].sum(1)
    def interval(values):
        return np.quantile(values, [.025, .975]).tolist()
    result = dict(repetitions=10000, seed=0, opposite_atom_pairs=len(pairs), axis_CI95=interval(a / (2 * len(pairs))),
                  joint_CI95=interval(j / (2 * len(pairs))), sign_given_axis_CI95=interval(j[a > 0] / a[a > 0]) if np.any(a > 0) else None,
                  alias_decision_CI95=interval(np.asarray(decisions)[draws].mean(1)), point_gates_only=True)
    cpairs = sorted({c['pair_id'] for c in compositions})
    means = []
    for pair in cpairs:
        group = [c for c in compositions if c['pair_id'] == pair]
        require(len(group) == 2, 'bootstrap composition group')
        ids = {c['id'] for c in group}
        means.append(np.mean([r['top1'] for r in rows if r['criterion_id'] in ids and r['source_split'] == 'unseen_criterion']))
    draws = rng.integers(0, len(cpairs), (10000, len(cpairs)))
    result.update(opposite_composition_pairs=len(cpairs), composition_decision_CI95=interval(np.asarray(means)[draws].mean(1)))
    compare(load(root / 'final' / 'intervals.json'), result, 'semantic reversal-pair intervals')
    return result


def model_artifacts(root, model_key, cfg, baseline=False):
    spec = cfg['models'][model_key]
    filename = ('encoder_lineage.json' if baseline else
                'encoder_lineage_stock_xsmall_higher_cls.json' if model_key == 'stock_xsmall' else
                'encoder_lineage_' + model_key + '.json')
    lineage = load(root / filename)
    require(lineage['encoder'] == spec['repo'] and lineage['revision'] == spec['revision'] and
            lineage['weight_sha256'] == spec['weight_sha256'], 'pinned canonical model ' + model_key)
    require(lineage['dtype'] == 'float32' and lineage['candidate_forwards'] == 0 and
            lineage['config']['hidden_size'] == spec['width'], 'model dtype/width')
    for name in ('missing_keys', 'mismatched_keys', 'error_msgs'):
        require(not lineage['loading'].get(name), 'model loading ' + name)
    weight_path = Path(hf_hub_download(spec['repo'], 'model.safetensors', revision=spec['revision'], local_files_only=True))
    config_path = Path(hf_hub_download(spec['repo'], 'config.json', revision=spec['revision'], local_files_only=True))
    require(sha(weight_path) == spec['weight_sha256'], 'actual pretrained weight bytes ' + model_key)
    config = load(config_path)
    for name in ('hidden_size', 'num_hidden_layers', 'model_type', 'id2label', 'pooler_hidden_act', 'pooler_hidden_size'):
        if name in config:
            compare(lineage['config'][name], config[name], 'actual model config ' + name)
    tokenizer = AutoTokenizer.from_pretrained(spec['repo'], revision=spec['revision'], use_fast=True,
                                             local_files_only=True, trust_remote_code=False)
    require(hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest() == lineage['tokenizer_sha256'], 'actual tokenizer hash')
    parameters = 0
    classifier, pooler = {}, {}
    with safe_open(weight_path, framework='np', device='cpu') as tensors:
        excluded = set(lineage['loading'].get('unexpected_keys', [])) if model_key == 'stock_xsmall' else set()
        for name in tensors.keys():
            if name not in excluded:
                parameters += math.prod(tensors.get_slice(name).get_shape())
        if model_key != 'stock_xsmall':
            for name in ('weight', 'bias'):
                classifier[name] = tensors.get_tensor('classifier.' + name)
                pooler[name] = tensors.get_tensor('pooler.dense.' + name)
    if 'floating_parameters' in lineage:
        require(parameters == lineage['floating_parameters'], 'actual model parameter count')
    if 'floating_parameters_metadata' in spec:
        require(parameters == spec['floating_parameters_metadata'], 'preregistered parameter count')
    return dict(lineage=lineage, tokenizer=tokenizer, parameters=parameters, classifier=classifier, pooler=pooler,
                hashes={str(weight_path): sha(weight_path), str(config_path): sha(config_path), str(root / filename): sha(root / filename)})


def verify_pooler(cls, pooled, model):
    require(cls.shape == pooled.shape and model['lineage']['config']['pooler_hidden_act'] == 'gelu', 'native pooler interface')
    require(pooled.dtype == np.float32 and np.isfinite(pooled).all(), 'finite FP32 native pooler features')
    maximum = 0.0
    for start in range(0, len(cls), 256):
        x = np.asarray(cls[start:start + 256], dtype=np.float64)
        z = x @ model['pooler']['weight'].astype(np.float64).T + model['pooler']['bias'].astype(np.float64)
        expected = .5 * z * (1 + erf(z / np.sqrt(2)))
        maximum = max(maximum, float(np.max(np.abs(expected - pooled[start:start + 256]))))
    require(maximum < 2e-4, 'independent native pooler maximum error ' + str(maximum))
    return maximum


def frozen_hashes(details):
    before, after = details['frozen_encoder_before_sha256'], details['frozen_encoder_after_sha256']
    require(isinstance(before, str) and re.fullmatch('[0-9a-f]{64}', before) is not None and before == after, 'full frozen parameter before/after hashes')
    return before


def head_artifacts(root, arm, model, H, evidence, groups, cfg, native):
    state = torch.load(root / (arm + '_checkpoint.pt'), map_location='cpu', weights_only=True)
    require(set(state) == {'weight', 'bias'}, 'plain linear state dict')
    weights = {k: v.detach().cpu().numpy() for k, v in state.items()}
    require(weights['weight'].shape == (3, cfg['models'][arm_model(cfg, arm)]['width']) and
            weights['bias'].shape == (3,) and all(v.dtype == np.float32 and np.isfinite(v).all() for v in weights.values()), 'head dimensions/dtype')
    with np.load(root / ('normalizer_' + arm + '.npz'), allow_pickle=False) as n:
        normalizer = {k: n[k] for k in ('mean', 'std')}
    require(np.isfinite(normalizer['mean']).all() and np.isfinite(normalizer['std']).all() and np.all(normalizer['std'] > 0), 'normalizer finite')
    if native:
        details = load(root / (arm + '_native_readout.json'))
        require(details['kind'] == 'native-frozen' and details['mean_0_std_1'] is True and
                not any(k in details for k in ('trace', 'selected_epoch', 'epochs', 'optimizer')), 'native no training')
        require(not (root / (arm + '_training.json')).exists(), 'native training forbidden')
        labels = model['lineage']['config']['id2label']
        mapping = {str(sign): next(int(i) for i, value in labels.items() if value == name)
                   for sign, name in ((-1, 'contradiction'), (0, 'neutral'), (1, 'entailment'))}
        require(len(set(mapping.values())) == 3 and details['mapping'] == mapping, 'native actual label-name mapping')
        order = [mapping[str(k)] for k in (-1, 0, 1)]
        for name in ('weight', 'bias'):
            require(np.array_equal(weights[name], model['classifier'][name][order]), 'actual reordered native classifier ' + name)
            require(details['classifier_' + name + '_sha256'] == array_sha(weights[name]), 'native classifier hash')
        require(np.array_equal(normalizer['mean'], np.zeros(weights['weight'].shape[1])) and
                np.array_equal(normalizer['std'], np.ones(weights['weight'].shape[1])), 'native mean0/std1')
        training_proof = dict(native_parameters_exact=True, no_training=True, mapping=mapping)
    else:
        details = load(root / (arm + '_training.json'))
        train = feature_rows(H, evidence, [c['text'] for c in groups['training_atoms']])
        mean, std = train.mean(0), np.maximum(train.std(0), .01)
        require(np.array_equal(normalizer['mean'], mean) and np.array_equal(normalizer['std'], std), 'ORIGINAL training order mean/std')
        require(details['normalizer']['mean_sha256'] == array_sha(mean) and details['normalizer']['std_sha256'] == array_sha(std) and
                details['normalizer']['train_only'] and details['normalizer']['floor'] == .01, 'train normalizer metadata')
        trace = details['trace']
        require(len(trace) == cfg['training']['epochs'] and [t['epoch'] for t in trace] == list(range(len(trace))), 'literal history epochs')
        require(all(math.isfinite(t['validation_CE']) and math.isfinite(t['CE']) and math.isfinite(t['loss']) for t in trace), 'finite training history')
        selected = min(range(len(trace)), key=lambda i: trace[i]['validation_CE'])
        require(details['selected_epoch'] == selected and details['validation_CE'] == trace[selected]['validation_CE'], 'minimum unweighted validation earlier tie')
        validation = feature_rows(H, evidence, [c['text'] for c in groups['validation_atoms']])
        _, logits = head_probabilities(validation, normalizer, weights)
        labels = np.array([c['sign'] + 1 if j == c['axis'] else 1 for c in groups['validation_atoms'] for j in range(4)])
        ce = float(np.mean(logsumexp(logits, axis=1) - logits[np.arange(len(labels)), labels]))
        require(abs(ce - details['validation_CE']) < 2e-5, 'checkpoint independent UNWEIGHTED literal-validation CE')
        require(details['kind'] == 'linear' and details['trainable_parameters'] == sum(v.size for v in weights.values()) and
                details['first_gradient_L1'] > 0 and details['seed'] == cfg['training']['seed'] and
                details['lr'] == cfg['training']['lr'] and details['weight_decay'] == cfg['training']['weight_decay'], 'fixed literal-head recipe')
        training_proof = dict(normalizer_original_training_order=True, selected_epoch=selected,
                              unweighted_validation_CE=ce, validation_CE_error=abs(ce - details['validation_CE']))
    require(details['arm_id'] == arm and details['floating_parameters'] == model['parameters'], 'arm model count')
    training_proof['frozen_parameter_sha256'] = frozen_hashes(details)
    return weights, normalizer, details, training_proof


def arm_model(cfg, arm):
    return next(a['model'] for a in cfg['arms'] if a['id'] == arm)


def main(root, cfg):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    groups, checked, corpus_proof = corpus_checks(root, cfg)
    source = Path(cfg['source_root'])
    env = load(root / 'environment.json')
    require(env['protocol_sha256'] == sha(PROTOCOL) and env['corpus_sha256'] == sha(root / 'corpus_manifest.json') and
            env['meaning_audit_sha256'] == sha(root / 'meaning_audit.json') and env['paid_resources'] is False and
            env['threads'] <= 4 and env['one_model_resident_at_a_time'] is True and
            env['token_limit'] == 128 and env['pair_batch'] == 32, 'measurement environment lineage')
    repo = Path(__file__).parents[2]
    frozen = subprocess.check_output(['git', 'rev-parse', 'vey-2-final'], cwd=repo, text=True).strip()
    require(frozen == FROZEN_VEY2, 'frozen Vey2 tag')
    code_hashes = {}
    for name in ('relation_pretrained_protocol.json', 'relation_pretrained_run.py', 'relation_pretrained_encoder.py',
                 'relation_pretrained_prepare.py', 'relation_pretrained_corpus.py', 'schema_relation_evaluate.py',
                 'schema_grounding_compiler.py'):
        path = PROTOCOL.with_name(name)
        recorded = subprocess.check_output(['git', 'show', env['git_revision'] + ':research/cbf0/' + name], cwd=repo)
        require(path.read_bytes() == recorded, 'measurement source bytes ' + name)
        code_hashes[name] = sha(path)
    selection, results = load(root / 'selection.json'), load(root / 'results.json')
    require(results['selection'] == selection and selection['arm_order'] == list(ARM_ORDER), 'selection record consistency')
    dev, compositions, g0 = groups['development_atoms'], groups['development_compositions'], groups['literal_holdout_atoms']
    literals, states = groups['literal_cases'], groups['states']
    outcomes, stages, models, caches, training_details = {}, {}, {}, {}, {}
    base = model_artifacts(source, 'stock_xsmall', cfg, baseline=True)
    checked.update(base['hashes'])
    base_e = load(source / 'frozen_pair_inputs.json')
    all_texts = sorted({p['criterion'] for p in base_e['pairs']})
    retokenized = {'stock_bare': re_tokenize(base_e, base['tokenizer'], all_texts, 'bare')}
    require(set(all_texts).isdisjoint({c['text'] for c in groups['final_atoms']}), 'development final disjointness')
    H = np.load(source / 'frozen_pair_features.npy', mmap_mode='r', allow_pickle=False)
    base_ckpt = torch.load(source / 'A_checkpoint.pt', map_location='cpu', weights_only=True)
    require(base_ckpt['kind'] == 'linear' and base_ckpt['last_layer'] is None, 'exact CBF6 checkpoint wrapper')
    weights = {k: v.detach().cpu().numpy() for k, v in base_ckpt['head'].items()}
    with np.load(source / 'normalizer.npz', allow_pickle=False) as n:
        normalizer = {k: n[k] for k in ('mean', 'std')}
    baseline_details = load(source / 'A_training.json')
    frozen_hashes(baseline_details)
    outcomes['stock_bare'], stages['stock_bare'], _, _ = verify_stage(root, 'development_stock_bare', dev, compositions,
                                                                     literals, states, H, base_e, weights, normalizer, g0, cfg)
    train_set = {c['text'] for c in groups['training_atoms']}
    baseline_train = H[[i for i, p in enumerate(base_e['pairs']) if p['criterion'] in train_set]]
    require(len(baseline_train) == 128 and np.array_equal(baseline_train.mean(0), normalizer['mean']) and
            np.array_equal(np.maximum(baseline_train.std(0), .01), normalizer['std']), 'exact CBF6 train-only normalizer order')
    trace = baseline_details['trace']
    minimum_epoch = min(range(len(trace)), key=lambda i: trace[i]['validation_CE'])
    require(baseline_details['selected_epoch'] == minimum_epoch and
            baseline_details['validation_CE'] == trace[minimum_epoch]['validation_CE'], 'baseline validation checkpoint minimum')
    base_validation = feature_rows(H, base_e, [c['text'] for c in groups['validation_atoms']])
    _, validation_logits = head_probabilities(base_validation, normalizer, weights)
    validation_labels = np.array([c['sign'] + 1 if j == c['axis'] else 1 for c in groups['validation_atoms'] for j in range(4)])
    baseline_ce = float(np.mean(logsumexp(validation_logits, axis=1) -
                                validation_logits[np.arange(len(validation_labels)), validation_labels]))
    require(abs(baseline_ce - baseline_details['validation_CE']) < 2e-5, 'baseline independent validation CE')
    stages['stock_bare']['head'] = dict(exact_source_checkpoint_reused=True, selected_epoch=minimum_epoch,
                                      unweighted_validation_CE=baseline_ce,
                                      validation_CE_error=abs(baseline_ce - baseline_details['validation_CE']),
                                      frozen_parameter_sha256=baseline_details['frozen_encoder_before_sha256'])
    expected_baseline = dict(kind='reuse-cbf6-a', no_retrain=True, no_reencode=True, feature_path=str(source / 'frozen_pair_features.npy'),
                             feature_sha256=sha(source / 'frozen_pair_features.npy'), normalizer_sha256=sha(source / 'normalizer.npz'),
                             checkpoint_sha256=sha(source / 'A_checkpoint.pt'), training_sha256=sha(source / 'A_training.json'),
                             frozen_encoder_before_sha256=baseline_details['frozen_encoder_before_sha256'],
                             frozen_encoder_after_sha256=baseline_details['frozen_encoder_after_sha256'], lineage_sha256=sha(source / 'encoder_lineage.json'))
    compare(results['native_readouts']['stock_bare'], expected_baseline, 'byte-identical CBF6 baseline reuse')
    require(not (root / 'stock_bare_checkpoint.pt').exists(), 'baseline no retuning')
    arm_specs = {a['id']: a for a in cfg['arms']}
    require(tuple(arm_specs) == ARM_ORDER, 'protocol canonical arm order')
    present = [a for a in ARM_ORDER if (root / ('development_' + a)).exists()]
    require(present[:6] == list(ARM_ORDER[:6]) and set(present).issubset(ARM_ORDER), 'all six controls completed')
    require({p.name for p in root.glob('development_*') if p.is_dir()} ==
            {'development_' + a for a in present}, 'no undeclared development arms')
    for arm in present[1:]:
        spec = arm_specs[arm]
        model_key, fmt = spec['model'], spec['format']
        if model_key not in models:
            models[model_key] = model_artifacts(root, model_key, cfg)
            checked.update(models[model_key]['hashes'])
        model = models[model_key]
        tag = model_key + '_' + fmt
        if tag not in caches:
            evidence = load(root / ('inputs_' + tag + '.json'))
            retokenized[tag] = re_tokenize(evidence, model['tokenizer'], all_texts, fmt)
            cls = np.load(root / ('features_' + tag + '_cls.npy'), mmap_mode='r', allow_pickle=False)
            require(cls.shape == (4 * len(all_texts), cfg['models'][model_key]['width']) and cls.dtype == np.float32 and np.isfinite(cls).all(), 'CLS feature dimensions')
            pooled = None
            if model_key != 'stock_xsmall':
                pooled = np.load(root / ('features_' + tag + '_pooler.npy'), mmap_mode='r', allow_pickle=False)
                model.setdefault('pooler_errors', {})[tag] = verify_pooler(cls, pooled, model)
            caches[tag] = (cls, pooled, evidence)
        cls, pooled, evidence = caches[tag]
        native = spec['head'].startswith('native')
        feature_name = 'cls' if spec['feature'] == 'CLS' else 'pooler'
        H = cls if feature_name == 'cls' else pooled
        weights, normalizer, details, training_proof = head_artifacts(root, arm, model, H, evidence, groups, cfg, native)
        training_details[arm] = (weights, normalizer, details, training_proof)
        require(details['evidence_sha256'] == sha(root / ('inputs_' + tag + '.json')), 'head input hash')
        feature_hash = sha(root / ('features_' + tag + '_' + feature_name + '.npy'))
        require(details['replay_sha256' if native else 'feature_sha256'] == feature_hash, 'head feature hash')
        if native:
            without_replay = {k: v for k, v in details.items() if k != 'replay_sha256'}
            compare(results['native_readouts'][arm], without_replay, 'native consumer metadata')
        outcomes[arm], stages[arm], _, _ = verify_stage(root, 'development_' + arm, dev, compositions, literals, states,
                                                      H, evidence, weights, normalizer, g0, cfg)
        stages[arm]['head'] = training_proof
        for filename in ('inputs_' + tag + '.json', 'features_' + tag + '_' + feature_name + '.npy',
                         arm + '_checkpoint.pt', 'normalizer_' + arm + '.npz'):
            checked[str(root / filename)] = sha(root / filename)
    small_trigger = not any(outcomes[a]['passed'] for a in ARM_ORDER[1:6])
    require(present == list(ARM_ORDER if small_trigger else ARM_ORDER[:6]), 'development-only small-model trigger')
    eligible = [a for a in present if arm_specs[a]['eligible']]
    passes = [a for a in eligible if outcomes[a]['passed']]
    if passes:
        selected, reason = passes[0], 'first eligible all-gate pass in arm order'
    else:
        selected = min(eligible, key=lambda a: (-outcomes[a]['atoms']['joint_accuracy'], -outcomes[a]['summary']['alias']['top1'],
                                               models[arm_specs[a]['model']]['parameters'], ARM_ORDER.index(a)))
        reason = 'highest eligible development joint atom, alias decisions, lower parameter count, earlier order'
    expected_selection = dict(selected=selected, reason=reason, passed=selected in passes,
                              development={a: dict(gates=r['gates'], passed=r['passed'], joint_accuracy=r['atoms']['joint_accuracy'],
                                                   alias_top1=r['summary']['alias']['top1']) for a, r in outcomes.items()},
                              arm_order=list(ARM_ORDER), small_triggered=small_trigger, final_openings=1,
                              final_outcomes_never_select_arm=True, meaning_audit_sha256=sha(root / 'meaning_audit.json'), protocol_sha256=sha(PROTOCOL))
    compare(selection, expected_selection, 'independent development-only selection')
    compare(results['development'], outcomes, 'consumer-visible development statistics')
    model_key, fmt = arm_specs[selected]['model'], arm_specs[selected]['format']
    final_tag = model_key + '_final_' + fmt
    final_texts = contexts(groups['final_atoms'], groups['final_compositions'], g0)
    final_inputs = load(root / ('inputs_' + final_tag + '.json'))
    require((root / 'selection.json').stat().st_mtime_ns <=
            (root / ('inputs_' + final_tag + '.json')).stat().st_mtime_ns and
            (root / 'selection.json').stat().st_mtime_ns <= (root / 'final' / 'results.json').stat().st_mtime_ns,
            'persisted selection precedes final evidence')
    retokenized[final_tag] = re_tokenize(final_inputs, models[model_key]['tokenizer'], final_texts, fmt)
    require({p.name for p in root.glob('inputs_*_final_*.json')} == {'inputs_' + final_tag + '.json'}, 'selected-only final inputs')
    require({p.name for p in root.glob('features_*_final_*.npy')}.issubset(
            {'features_' + final_tag + '_cls.npy', 'features_' + final_tag + '_pooler.npy'}), 'selected-only final feature interfaces')
    final_cls = np.load(root / ('features_' + final_tag + '_cls.npy'), mmap_mode='r', allow_pickle=False)
    require(final_cls.shape == (4 * len(final_texts), cfg['models'][model_key]['width']) and np.isfinite(final_cls).all(), 'final feature dimensions')
    feature_name = 'cls' if arm_specs[selected]['feature'] == 'CLS' else 'pooler'
    final_h = final_cls
    if model_key != 'stock_xsmall':
        final_pooled = np.load(root / ('features_' + final_tag + '_pooler.npy'), mmap_mode='r', allow_pickle=False)
        models[model_key].setdefault('pooler_errors', {})[final_tag] = verify_pooler(final_cls, final_pooled, models[model_key])
        if feature_name == 'pooler':
            final_h = final_pooled
    weights, normalizer, details, _ = training_details[selected]
    final, stages['final'], final_rows, resolutions = verify_stage(root, 'final', groups['final_atoms'], groups['final_compositions'], literals,
                                                                  states, final_h, final_inputs, weights, normalizer, g0, cfg)
    # Held literals are shared across openings; a different cache must not change their consumer-visible masses.
    dev_records = {r['text']: r for r in load(root / ('development_' + selected) / 'relation_matrices.json')}
    final_records = {r['text']: r for r in load(root / 'final' / 'relation_matrices.json')}
    g0_error = max(float(np.max(np.abs(np.asarray(dev_records[c['text']]['matrix']) - final_records[c['text']]['matrix']))) for c in g0)
    require(g0_error < 2e-5, 'matched G0 development/final replay')
    compare(results['final'], final, 'consumer-visible final statistics')
    ci = verify_intervals(root, groups['final_atoms'], groups['final_compositions'], final_rows, resolutions)
    compare(results['final_intervals'], ci, 'consumer-visible intervals')
    expected_readout = dict(kind='native-frozen' if arm_specs[selected]['head'].startswith('native') else 'linear',
                            evidence_sha256=sha(root / ('inputs_' + final_tag + '.json')),
                            checkpoint_sha256=sha(root / (selected + '_checkpoint.pt')), normalizer_sha256=sha(root / ('normalizer_' + selected + '.npz')))
    compare(results['final_readout'], expected_readout, 'immutable selected checkpoint final reuse')
    require(results['verdict'] == ('C7_RESEARCH_ONLY_SCREEN_PASS_REPLICATION_REQUIRED' if final['passed'] else 'C7_NOT_EARNED_ON_FRESH_CORPUS') and
            results['replication_required'] == final['passed'], 'final verdict/replication')
    require(results['B_STEF_allowed'] is False and cfg['B_STEF_allowed'] is False and results['ASG_closed_permanently'] is True and
            results['license_category'] == cfg['models'][model_key]['eligibility'], 'research-only policy')
    require(results['source_root'] == str(source) and results['output_root'] == str(root), 'consumer evidence roots')
    require(set(results['native_readouts']) == {'stock_bare'} |
            {a for a in present if arm_specs[a]['head'].startswith('native')}, 'declared native readouts only')
    for model_key in models:
        hashes = {training_details[a][3]['frozen_parameter_sha256'] for a in present[1:]
                  if arm_specs[a]['model'] == model_key}
        require(len(hashes) == 1, 'same complete frozen model hash across readouts ' + model_key)
    for key, model in models.items():
        session_name = 'gpu_session_' + key + '.json'
        session = load(root / session_name)
        lineage_name = 'encoder_lineage_stock_xsmall_higher_cls.json' if key == 'stock_xsmall' else 'encoder_lineage_' + key + '.json'
        require(session['locked'] and session['model_key'] == key and session['frozen_model_sha256'] == sha(root / lineage_name), 'GPU residency lineage')
    for path in root.glob('features_*_final_*.npy'):
        checked[str(path)] = sha(path)
    checked[str(root / ('inputs_' + final_tag + '.json'))] = sha(root / ('inputs_' + final_tag + '.json'))
    return dict(all_checks_pass=True, evidence_class='MEASURED: independent persisted-artifact reconstruction',
                corpus=corpus_proof, checked_source_and_model_sha256=checked, measurement_source_sha256=code_hashes,
                frozen_vey2_tag=frozen, pair_inputs_retokenized=retokenized, stages=stages,
                selected=selected, reconstructed_selection=expected_selection, final_intervals=ci,
                matched_G0_maximum_probability_error=g0_error,
                native_pooler_maximum_errors={key: model.get('pooler_errors', {}) for key, model in models.items()},
                final_outcome_never_selects_arm=True, baseline_byte_identical_reuse=True,
                selected_only_final=True, exact_literal_neural_calls=0, cpu_only=True,
                uncalibrated_softmax_not_certificate=True, conditional_review_not_shipping_clearance=True,
                B_STEF_allowed=False, protocol_sha256=sha(PROTOCOL))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, help='Artifact root; must equal the frozen protocol output root')
    args = parser.parse_args()
    config = load(PROTOCOL)
    artifact_root = args.root or Path(config['output_root'])
    try:
        require(artifact_root.resolve() == Path(config['output_root']).resolve(), 'frozen output root')
        proof = main(artifact_root, config)
        save(artifact_root / 'verification.json', proof)
        print(json.dumps(proof, sort_keys=True, indent=2, allow_nan=False))
    except Exception as error:
        failure = dict(all_checks_pass=False, error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        if artifact_root.is_dir():
            save(artifact_root / ('verification_failure_' + stamp + '_' + str(os.getpid()) + '.json'), failure)
        raise
