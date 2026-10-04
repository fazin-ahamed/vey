#!/usr/bin/env python3
"""Audit child-local supervision against retained ECA-1 projection artifacts."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/fazinahamed/Documents/vey-data/decisionmix/endgame/ephemeral-pages-v1')
PHASES = ('train', 'validation', 'calibration', 'development')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def leaf_truth(record, row):
    term = row['metadata']['terms'][record['term_index']]
    meta = row['metadata']
    matching = [block for block, owner in meta['page_owners'].items()
                if owner == record['candidate_id'] and meta['page_fields'][block] == term['field_key']]
    known = len(matching) == 1 and matching[0] in meta['page_grades']
    grade_mask = [known and page['field_key'] == term['field_key'] and page['grade_target'] is not None
                  for page in record['pages']]
    return known, grade_mask


def audit_phase(phase, features, corpus):
    if phase not in PHASES:
        raise ValueError('this prospective target audit never opens ECA final')
    manifest_path = features / f'{phase}_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    ir_path = corpus / f'{phase}.jsonl'
    if sha(ir_path) != manifest['corpus_sha256']:
        raise RuntimeError('IR does not match retained projection')
    records_path = features / manifest['files']['records']['path']
    if sha(records_path) != manifest['files']['records']['sha256']:
        raise RuntimeError('retained record projection changed')
    with ir_path.open() as stream:
        ir = {row['id']: row for row in map(json.loads, stream)}
    counts, variants, examples = Counter(), Counter(), []
    with records_path.open() as stream:
        for line in stream:
            record = json.loads(line)
            row = ir[record['row_id']]
            known, mask = leaf_truth(record, row)
            counts['records'] += 1
            if bool(record['known_target']) != known:
                counts['wrong_child_knownness'] += 1
                variants[row['metadata']['variant']] += 1
                if len(examples) < 8:
                    visible = {'question': record['question'],
                               'ordered_page_texts': [page['text'] for page in record['pages']]}
                    signature = hashlib.sha256(json.dumps(visible, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                    examples.append({'row_id': row['id'], 'candidate_id': record['candidate_id'],
                                     'term_index': record['term_index'], 'variant': row['metadata']['variant'],
                                     'model_visible_input': visible, 'input_sha256': signature,
                                     'retained_known_target': record['known_target'], 'required_atomic_known_target': known,
                                     'parent_terms': row['metadata']['terms'],
                                     'explanation': 'Other parent requirements are absent from the atomic reader input; parent UNKNOWN cannot relabel an independently supported child.'})
            if record['grade_mask'] != mask:
                counts['wrong_grade_mask_records'] += 1
            if record['orientation_mask'] != mask:
                counts['wrong_orientation_mask_records'] += 1
    return {'phase': phase, 'capture_manifest': {'path': str(manifest_path), 'sha256': sha(manifest_path)},
            'corpus': {'path': str(ir_path), 'sha256': sha(ir_path)},
            'records': {'path': str(records_path), 'sha256': sha(records_path)},
            'counts': dict(counts), 'affected_variants': dict(variants), 'examples': examples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=PHASES, action='append')
    parser.add_argument('--features-root', type=Path, default=ROOT / 'features-grade-corrected')
    parser.add_argument('--corpus-root', type=Path, default=ROOT / 'corpus')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    phases = args.phase or PHASES
    reports = [audit_phase(phase, args.features_root, args.corpus_root) for phase in phases]
    wrong = sum(report['counts'].get('wrong_child_knownness', 0) for report in reports)
    result = {'schema': 'vey.eca.atomic-target-audit.v1', 'implementation_sha256': sha(__file__),
              'reports': reports, 'wrong_child_knownness_records': wrong,
              'status': 'invalid_atomic_supervision' if wrong else 'child_local_targets_verified',
              'neural_encoder_forwards': 0, 'model_training': False, 'final_opened': False,
              'scope': 'Function contract audit, not a fit/quality/representation conclusion.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'wrong_child_knownness_records': wrong,
                      'phase_counts': {report['phase']: report['counts'] for report in reports},
                      'output': str(args.output), 'sha256': sha(args.output)}, sort_keys=True))


if __name__ == '__main__':
    main()
