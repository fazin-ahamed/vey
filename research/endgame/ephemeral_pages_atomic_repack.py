#!/usr/bin/env python3
"""Reproject ECA-2 child targets without encoding or changing any input bytes."""
from __future__ import annotations
import argparse
import copy
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ephemeral_pages_capture as capture
from ephemeral_pages_features import rows_to_examples, sha256_file

ORIGINAL_ROOT = Path('/home/fazinahamed/Documents/vey-data/decisionmix/endgame/ephemeral-pages-v1')
PHASES = ('train', 'validation', 'calibration', 'development')
CHANGED_TARGETS = ('known_target', 'grade_mask', 'orientation_mask',
                   'orientation_target', 'directed_grade_target')


def artifact(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha256_file(path)}


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    path.chmod(0o600)


def repack(phase, proof_path):
    exp = capture.ATOMIC_EXPERIMENT
    cfg, protocol_hash = exp.protocol()
    proof_path = Path(proof_path).resolve()
    proof = json.loads(proof_path.read_text())
    feature_hash = sha256_file(Path(__file__).with_name('ephemeral_pages_features.py'))
    if (proof['feature_implementation_sha256'] != feature_hash or
            proof.get('status') != 'child_local_projection_and_parent_conjunction_verified' or
            proof.get('model_training') is not False or proof.get('final_opened') is not False):
        raise RuntimeError('projection proof does not describe the current child-local implementation')
    source_root = ORIGINAL_ROOT / 'features-grade-corrected'
    parent_path = source_root / f'{phase}_manifest.json'
    parent = json.loads(parent_path.read_text())
    source_ir = ORIGINAL_ROOT / 'corpus' / f'{phase}.jsonl'
    if sha256_file(source_ir) != parent['corpus_sha256']:
        raise RuntimeError('retained parent IR changed')
    if (exp.cache_root / f'{phase}_manifest.json').exists():
        raise FileExistsError('refusing to replace completed correction')
    exp.cache_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    exp.corpus_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    target_ir = exp.corpus_root / source_ir.name
    if target_ir.exists():
        if sha256_file(target_ir) != parent['corpus_sha256']:
            raise RuntimeError('borrowed IR copy changed')
    else:
        with target_ir.open('xb') as dst, source_ir.open('rb') as src:
            shutil.copyfileobj(src, dst)
        target_ir.chmod(0o600)
    with source_ir.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    records = rows_to_examples(rows)
    source_records = source_root / parent['files']['records']['path']
    if sha256_file(source_records) != parent['files']['records']['sha256']:
        raise RuntimeError('parent record projection changed')
    with source_records.open() as stream:
        previous = [json.loads(line) for line in stream if line.strip()]
    if len(previous) != len(records):
        raise RuntimeError('child correction changed record population')
    changes = Counter()
    for old, new in zip(previous, records):
        if {k:v for k,v in old.items() if k not in CHANGED_TARGETS} != {k:v for k,v in new.items() if k not in CHANGED_TARGETS}:
            raise RuntimeError('correction changed model-visible inputs or unaffected labels')
        for key in CHANGED_TARGETS:
            changes[key] += old[key] != new[key]
    grouped = {}
    for rec in records:
        grouped.setdefault((rec['row_id'], rec['candidate_id']), []).append(rec['known_target'])
    for row in rows:
        for cid, known in row['metadata']['known'].items():
            if cid != '__unknown__' and all(grouped[(row['id'], cid)]) != known:
                raise RuntimeError('child conjunction disagrees with parent UNKNOWN')
    files = {}
    for key, entry in parent['files'].items():
        old_path = source_root / entry['path']
        if sha256_file(old_path) != entry['sha256']:
            raise RuntimeError('parent packed artifact changed: '+key)
        if key == 'records':
            path = exp.cache_root / f'{phase}_records.jsonl'
            with path.open('x', encoding='utf-8') as stream:
                for rec in records:
                    stream.write(json.dumps(rec, sort_keys=True, ensure_ascii=False, allow_nan=False)+'\n')
            path.chmod(0o600)
            files[key] = artifact(path)
        elif key in CHANGED_TARGETS:
            old_array = np.load(old_path, mmap_mode='r', allow_pickle=False)
            changed = np.array(old_array, copy=True)
            for i, rec in enumerate(records):
                if changed.ndim == 1:
                    changed[i] = rec[key]
                else:
                    changed[i, :len(rec['pages'])] = np.asarray(rec[key], dtype=changed.dtype)
            path = exp.cache_root / f'{phase}_{key}.npy'
            with path.open('xb') as stream:
                np.save(stream, changed, allow_pickle=False)
            path.chmod(0o600)
            files[key] = artifact(path)
        else:
            files[key] = artifact(old_path)
    lineage = copy.deepcopy(parent['lineage'])
    lineage.pop('grade_correction', None)
    lineage.update(protocol_sha256=protocol_hash, capture_phase=phase, atomic_supervision='child-local-v1',
                   feature_code_sha256=feature_hash)
    lineage['atomic_correction'] = {
        'correction_protocol':artifact(exp.protocol_path), 'original_capture_manifest':artifact(parent_path),
        'original_corpus_sha256':parent['corpus_sha256'], 'feature_implementation_sha256':lineage['feature_code_sha256'],
        'child_local_records':len(records), 'changed_known_target_records':changes['known_target'],
        'changed_grade_mask_records':changes['grade_mask'], 'changed_orientation_mask_records':changes['orientation_mask'],
        'changed_directed_grade_target_records':changes['directed_grade_target'],
        'changed_orientation_target_records':changes['orientation_target'],
        'existing_inputs_reencoded':0, 'parent_conjunction_verified':True,
        'projection_verification':artifact(proof_path), 'implementation':artifact(Path(__file__))}
    counters = copy.deepcopy(parent['counters'])
    for modality in ('query','page','cross'):
        counters[modality].update(forward_calls_this_capture=0, encoded_examples_this_capture=0,
                                  cache_misses=0, cache_hits=counters[modality]['unique_inputs'])
    counters.update(total_encoder_forward_calls_this_capture=0,total_encoded_examples_this_capture=0)
    manifest={'phase':phase,'protocol_sha256':protocol_hash,'record_count':len(records),
              'corpus_sha256':sha256_file(target_ir),'files':files,'lineage':lineage,
              'counters':counters,'token_receipts':copy.deepcopy(parent['token_receipts']),
              'experiment_context':exp.context(),'selection_calibration_receipt':None}
    path=exp.cache_root/f'{phase}_manifest.json'
    write_json(path,manifest)
    print(json.dumps({'phase':phase,'records':len(records),'changed_records':dict(changes),
                      'encoder_forwards':0,'byte_identical_inputs':['q','pages','cross','page_mask'],
                      'manifest':artifact(path)},sort_keys=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=PHASES,required=True)
    parser.add_argument('--projection-proof', type=Path,
                        default=capture.ATOMIC_ROOT / 'projection_verification.custody-v2.json')
    args=parser.parse_args()
    repack(args.phase, args.projection_proof)


if __name__=='__main__':
    main()
