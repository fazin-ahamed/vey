#!/usr/bin/env python3
"""Count substantive QASPER input tokens without accessing answer values."""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sqlite3
import statistics
import sys

import neutral_paper_acquire as paper


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def check_pin(pin):
    path = Path(pin['path'])
    if file_hash(path) != pin['sha256']:
        raise ValueError(f'Pinned input changed: {path}')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if protocol['schema'] != 'vey.neutral.qasper.input-length-protocol.v1':
        raise ValueError('Unsupported census protocol')
    if args.output_root.exists():
        raise FileExistsError('Refusing to overwrite a census')
    if os.environ.get('TOKENIZERS_PARALLELISM') != 'false':
        raise ValueError('Tokenizer parallelism must be disabled')
    check_pin(protocol['implementation'])
    for pin in protocol['helper_sources']:
        check_pin(pin)
    check_pin(protocol['source_verification'])
    database = check_pin(protocol['database'])
    from tokenizers import Tokenizer
    tokenizers = {name: Tokenizer.from_file(str(check_pin(pin)))
                  for name, pin in protocol['tokenizers'].items()}
    for tokenizer in tokenizers.values():
        tokenizer.no_truncation()
        tokenizer.no_padding()
    args.output_root.mkdir(mode=0o700, parents=True)
    row_path = args.output_root / 'paper_token_counts.jsonl'
    counts = defaultdict(Counter)
    lengths = defaultdict(list)
    split_counts = Counter()
    groups = defaultdict(set)
    components = defaultdict(set)
    allowed_columns = {
        'papers': {'paper_id', 'source_partition', 'group_id', 'model_state_json'},
        'source_groups': {'group_id', 'component_id', 'final_split'},
    }
    reads = set()

    def authorizer(action, first, second, database_name, trigger):
        if action == sqlite3.SQLITE_READ:
            if first not in allowed_columns or second not in allowed_columns[first]:
                return sqlite3.SQLITE_DENY
            reads.add((first, second))
        return sqlite3.SQLITE_OK

    query = ('SELECT p.paper_id,p.source_partition,p.group_id,p.model_state_json,'
             'g.component_id,g.final_split FROM papers p JOIN source_groups g '
             'ON p.group_id=g.group_id ORDER BY g.final_split,p.group_id,p.paper_id')
    with sqlite3.connect(database.as_uri()+'?mode=ro', uri=True) as connection:
        connection.execute('PRAGMA query_only=ON')
        connection.execute('PRAGMA trusted_schema=OFF')
        connection.execute('PRAGMA cache_size=-16384')
        connection.set_authorizer(authorizer)
        with row_path.open('x') as stream:
            for paper_id, source_partition, group_id, state_json, component_id, split in connection.execute(query):
                state = json.loads(state_json)
                if set(state) - {'title', 'abstract', 'full_text'}:
                    raise ValueError('Model-state projection contains a forbidden field')
                document = paper._document_text(state)
                row = {'paper_id': paper_id, 'source_partition': source_partition,
                       'group_id': group_id, 'component_id': component_id, 'split': split,
                       'model_state_sha256': hashlib.sha256(state_json.encode()).hexdigest(),
                       'substantive_text_sha256': hashlib.sha256(document.encode()).hexdigest(),
                       'characters': len(document), 'token_counts': {}}
                for name, tokenizer in tokenizers.items():
                    size = len(tokenizer.encode(document, add_special_tokens=False).ids)
                    row['token_counts'][name] = size
                    for scope in ('all', split):
                        key = (name, scope)
                        lengths[key].append(size)
                        counts[key]['papers'] += 1
                        for threshold in protocol['thresholds']:
                            if size >= threshold:
                                counts[key][f'ge_{threshold}'] += 1
                split_counts[split] += 1
                groups[split].add(group_id)
                components[split].add(component_id)
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
    if dict(split_counts) != protocol['expected_papers_by_split']:
        raise ValueError('Source split census differs from independently verified custody')
    if file_hash(database) != protocol['database']['sha256']:
        raise ValueError('Read-only source database changed during census')
    if 'torch' in sys.modules:
        raise RuntimeError('Input-only census imported a model framework')
    report = {'schema': 'vey.neutral.qasper.input-length-result.v1',
              'protocol': {'path': str(args.protocol.resolve()), 'sha256': file_hash(args.protocol)},
              'implementation_sha256': file_hash(Path(__file__)),
              'database_sha256': protocol['database']['sha256'],
              'input_only_sql': query, 'actual_SQL_READ_columns': sorted(reads),
              'source_database_unchanged': True, 'source_split_unchanged': True,
              'answer_values_or_questions_read': False, 'model_forwards': 0,
              'final_quality_evaluation': False, 'long_context_capability_earned': False,
              'tokenizers_version': importlib.metadata.version('tokenizers'),
              'python': sys.version, 'tokenizer_pins': protocol['tokenizers'],
              'paper_rows': {'path': str(row_path.resolve()), 'sha256': file_hash(row_path)},
              'split_counts': dict(split_counts),
              'group_counts': {split: len(ids) for split, ids in groups.items()},
              'component_counts': {split: len(ids) for split, ids in components.items()},
              'lengths': {name: {scope: {**dict(counts[(name, scope)]),
                                       'minimum': min(values), 'median': statistics.median(values),
                                       'maximum': max(values)}
                                 for (tokenizer_name, scope), values in lengths.items() if tokenizer_name == name}
                          for name in tokenizers}}
    report_path = args.output_root / 'report.json'
    with report_path.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'report': str(report_path), 'sha256': file_hash(report_path),
                      'paper_count': sum(split_counts.values()), 'model_forwards': 0}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
