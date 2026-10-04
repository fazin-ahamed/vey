#!/usr/bin/env python3
"""Diagnostic-only four-field replay of already closed CBF8 evidence."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from ephemeral_pages_features import (FeatureEncoder, FeatureNormalizer, sha256_file,
                                      validate_final_receipt)
from ephemeral_pages_capture import DEFAULT_EXPERIMENT, resolve_experiment

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / 'ephemeral_pages_closed_protocol.json'
sys.path.insert(0, str(HERE.parent / 'cbf0'))
from schema_grounding_compiler import compile_criterion, parse_criterion
from exact_state_build import parse


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def run(receipt_path, calibration_path, output, device='cuda',
        experiment=DEFAULT_EXPERIMENT, protocol_path=PROTOCOL):
    import numpy as np
    import torch
    from ephemeral_pages_train import load_control

    cfg = load(protocol_path)
    repo = HERE.parents[1]
    for path, expected in cfg['source_hashes'].items():
        if sha256_file(path) != expected:
            raise RuntimeError(f'preregistered source changed: {path}')
    for path in (protocol_path, Path(__file__).resolve()):
        relative = str(path.relative_to(repo))
        committed = subprocess.check_output(['git', 'show', f'HEAD:{relative}'], cwd=repo)
        if committed != path.read_bytes():
            raise RuntimeError('commit diagnostic specification and adapter before replay')
    receipt = validate_final_receipt(receipt_path, experiment.protocol_path)
    if Path(calibration_path).resolve() != Path(receipt['calibration_file']['path']).resolve():
        raise RuntimeError('explicit calibration must be the sealed receipt artifact')
    calibration = load(calibration_path)
    if calibration['protocol_sha256'] != receipt['protocol_sha256'] or calibration['final_outcomes_used'] is not False:
        raise RuntimeError('invalid sealed calibration')
    threshold = calibration['controls']['pages']['knownness_threshold']
    if not 0 < threshold < 1:
        raise RuntimeError('invalid calibrated threshold')
    checkpoint = Path(receipt['checkpoint_files']['pages']['path'])
    if checkpoint.name != 'pages.pt':
        raise RuntimeError('expected unchanged pages checkpoint')
    reader, stats = load_control('pages', checkpoint.parent, device, experiment=experiment)
    normalizer = FeatureNormalizer(stats['pages']['mean'], stats['pages']['std'])
    root = Path(cfg['source_root'])
    states = load(root / 'states.json')
    literals = load(root / 'literal_cases.json')
    cohorts = {}
    for phase, spec in cfg['cohorts'].items():
        cohorts[phase] = ([dict(q, stratum='alias') for q in load(root / spec['atoms'])] +
                          [dict(q, stratum='alias_composition') for q in load(root / spec['compositions'])] + literals)
    texts = sorted({q['text'][t['span'][0]:t['span'][1]] for queries in cohorts.values()
                    for q in queries for t in parse_criterion(q['text']) if t['literal_axis'] is None})
    definitions = load(HERE.parent / 'cbf0' / 'schema_support_protocol.json')
    pages = [definitions['field_definitions'][name] for name in definitions['schema']]
    output = Path(output).resolve()
    if output.is_relative_to(repo):
        raise RuntimeError('closed replay data must remain outside Git')
    output.mkdir(parents=True, exist_ok=False)
    features = {}
    token_receipts = {}
    with FeatureEncoder(device=device, protocol_path=experiment.protocol_path) as encoder:
        for modality, strings in [('page', pages), ('query', texts)]:
            ids, mask = encoder.tokenize(strings)
            chunks = [encoder.encode_batch(ids[i:i+32], mask[i:i+32], modality)
                      for i in range(0, len(strings), 32)]
            features[modality] = np.concatenate(chunks)
            np.save(output / f'{modality}_features.npy', features[modality], allow_pickle=False)
            np.save(output / f'{modality}_input_ids.npy', ids, allow_pickle=False)
            np.save(output / f'{modality}_attention_mask.npy', mask, allow_pickle=False)
            token_receipts[modality] = {'texts': strings, 'token_count': int(mask.sum())}
        lineage = encoder.finalize_lineage()
        counters = {'forward_counts': dict(encoder.forward_counts), 'encoded_counts': dict(encoder.encoded_counts),
                    'cache_hits': 0, 'unique_page_texts': len(pages), 'unique_query_texts': len(texts)}
    resolutions, relations = {}, []
    p = torch.as_tensor(normalizer.transform(features['page']), device=device).unsqueeze(0)
    mask = torch.ones((1, 4), dtype=torch.bool, device=device)
    with torch.inference_mode():
        for index, text in enumerate(texts):
            q = torch.as_tensor(normalizer.transform(features['query'][index:index+1]), device=device)
            prediction = reader(q, p, mask)
            raw = {name: value.detach().cpu().numpy().tolist() for name, value in zip(prediction._fields, prediction)}
            rel = prediction.relevance_logits[0].detach().cpu().numpy()
            direction = prediction.direction[0].detach().cpu().numpy()
            known = float(torch.sigmoid(prediction.known_logits[0]).cpu())
            tied = np.flatnonzero(rel == rel.max())
            reason = None
            if not np.isfinite(known) or not np.isfinite(rel).all() or not np.isfinite(direction).all() or not np.isfinite(float(prediction.score[0].cpu())):
                reason = 'nonfinite_full_output'
            elif known < threshold:
                reason = 'below_unchanged_knownness_threshold'
            elif len(tied) != 1:
                reason = 'exact_relevance_tie'
            elif direction[tied[0]] == 0:
                reason = 'zero_direction'
            resolution = None if reason else {'axis': int(tied[0]), 'sign': int(np.sign(direction[tied[0]]))}
            resolutions[text] = resolution
            relations.append({'text': text, 'resolution': resolution, 'unknown_reason': reason,
                              'known_probability': known, 'full_reader_output': raw})
    save(output / 'relations.json', relations)
    save(output / 'encoder_inputs.json', token_receipts)
    reports = {}
    for phase, queries in cohorts.items():
        spec = cfg['cohorts'][phase]
        original_root = root / spec['selected_output']
        originals = {(r['criterion_id'], r['scenario_id']): r for r in jsonl(original_root / 'decisions.jsonl')}
        directory = output / phase
        directory.mkdir()
        decisions, compiled, lookup = [], [], {}
        for query in queries:
            text = query['text']
            vector, atoms = compile_criterion(text, lambda t: resolutions[text[t['span'][0]:t['span'][1]]])
            if query['stratum'] == 'literal' and vector != query['weights']:
                raise RuntimeError('literal compiler differs from original')
            compiled.append({'query': query, 'vector': vector, 'atoms': atoms})
            for state in states:
                if state['source_split'] == 'train':
                    continue
                facts = [parse(candidate) for candidate in state['candidates']]
                gold_scores = [sum(int(a)*int(b) for a,b in zip(query['weights'], fact)) for fact in facts]
                scores = None if vector is None else [sum(int(a)*int(b) for a,b in zip(vector, fact)) for fact in facts]
                gold = [i for i,s in enumerate(gold_scores) if s == max(gold_scores)]
                top = [] if scores is None else [i for i,s in enumerate(scores) if s == max(scores)]
                chosen = top[0] if top else None
                old = originals[(query['id'], state['id'])]
                if old['teacher_scores'] != gold_scores or old['teacher_top_set'] != gold:
                    raise RuntimeError('original integer states/gold replay mismatch')
                old_scores = old['scores']
                old_top = [] if old_scores is None else [i for i,s in enumerate(old_scores) if s == max(old_scores)]
                if query['stratum'] == 'literal' and (top != old_top or top != gold):
                    raise RuntimeError('literal maximal-set regression')
                row = {'criterion_id': query['id'], 'scenario_id': state['id'], 'stratum': query['stratum'],
                       'query_provenance': query, 'state_provenance': state, 'exact_integer_facts': facts,
                       'vector': vector, 'scores_integer': scores, 'teacher_scores': gold_scores,
                       'teacher_top_set': gold, 'predicted_top_set': top, 'winner': chosen,
                       'stable_ordinals': list(range(len(facts))), 'maximal_set_correct': top == gold,
                       'chosen_in_gold': chosen in gold, 'unknown': vector is None,
                       'original_selected_decision': old, 'original_predicted_top_set': old_top,
                       'closed_maximal_set_difference': top != old_top, 'closed_concrete_difference': chosen != old['winner']}
                decisions.append(row)
                lookup[(query['id'], state['id'])] = row
        swaps = []
        for old in jsonl(original_root / 'teacher_changing_swaps.jsonl'):
            before = lookup[(old['before'], old['scenario_id'])]
            after = lookup[(old['after'], old['scenario_id'])]
            correct = after['chosen_in_gold']
            changed = before['winner'] != after['winner']
            swaps.append({'original_selected_swap': old, 'scenario_id': old['scenario_id'],
                          'before': old['before'], 'after': old['after'], 'correct_new': correct,
                          'student_changed': changed, 'changed_to_new': correct and changed,
                          'closed_correct_new_difference': correct != bool(old['top_set_new'])})
        save(directory / 'compiled_queries.json', compiled)
        for name, rows in [('decisions', decisions), ('teacher_changing_swaps', swaps)]:
            with (directory / f'{name}.jsonl').open('x') as stream:
                for row in rows:
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
        reports[phase] = {'literal_rows': sum(r['stratum']=='literal' for r in decisions),
                         'literal_maximal_set_accuracy': 1.0, 'decision_rows': len(decisions),
                         'strata': {s: {'n': len(rr), 'maximal_set_accuracy': sum(r['maximal_set_correct'] for r in rr)/len(rr),
                                        'chosen_in_gold': sum(r['chosen_in_gold'] for r in rr)/len(rr),
                                        'unknown': sum(r['unknown'] for r in rr),
                                        'closed_maximal_set_differences': sum(r['closed_maximal_set_difference'] for r in rr)}
                                    for s in ('alias','alias_composition','literal')
                                    for rr in [[r for r in decisions if r['stratum']==s]]},
                         'causal_n': len(swaps), 'causal_correct_new': sum(r['correct_new'] for r in swaps)/len(swaps)}
    result = {'schema': 'vey.eca.closed-diagnostic.v1', 'protocol_sha256': sha256_file(protocol_path),
              'adapter_sha256': sha256_file(__file__),
              'git_revision': subprocess.check_output(['git','rev-parse','HEAD'], cwd=repo, text=True).strip(),
              'experiment_context': experiment.context(),
              'receipt': receipt, 'threshold': threshold, 'encoder_lineage': lineage, 'counters': counters,
              'reports': reports, 'source_hashes': cfg['source_hashes'],
              'claim': cfg['claim'], 'artifacts': {str(p.relative_to(output)): sha256_file(p)
                                                 for p in sorted(output.rglob('*')) if p.is_file()}}
    save(output / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', required=True, type=Path)
    parser.add_argument('--calibration', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--atomic', action='store_true')
    args = parser.parse_args()
    protocol_path = HERE / 'ephemeral_pages_atomic_closed_protocol.json' if args.atomic else PROTOCOL
    print(json.dumps(run(args.receipt, args.calibration, args.output, args.device,
                         resolve_experiment(args.atomic), protocol_path), sort_keys=True))


if __name__ == '__main__':
    main()
