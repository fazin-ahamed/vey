#!/usr/bin/env python3
"""Acquire pinned NQ validation bytes and count document tokens, not labels."""
import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import shutil
import statistics
import sys
import unicodedata
import urllib.parse

import neutral_paper_length_census as shared


INPUT_COLUMNS = ('id', 'document.title', 'document.url', 'document.tokens')
TOKEN_FIELDS = {'token', 'is_html', 'start_byte', 'end_byte'}
GIB = 1024 ** 3


def checked_batches(parquet, columns=INPUT_COLUMNS):
    if tuple(columns) != INPUT_COLUMNS:
        raise ValueError('Only the committed document/identity projection may be decoded')
    return parquet.iter_batches(batch_size=1, columns=list(columns), use_threads=False)


def document_text(document):
    if set(document) != {'title', 'url', 'tokens'}:
        raise ValueError('Document projection contains missing or forbidden fields')
    if not all(isinstance(document[key], str) and document[key] for key in ('title', 'url')):
        raise ValueError('Native document identity is missing')
    tokens = document['tokens']
    if isinstance(tokens, dict):
        if set(tokens) != TOKEN_FIELDS or any(not isinstance(value, list) for value in tokens.values()):
            raise ValueError('Invalid native struct-of-token-lists')
        if len({len(value) for value in tokens.values()}) != 1:
            raise ValueError('Native token lists have different lengths')
        rows = zip(tokens['token'], tokens['is_html'], tokens['start_byte'], tokens['end_byte'])
    elif isinstance(tokens, list):
        if any(not isinstance(token, dict) or set(token) != TOKEN_FIELDS for token in tokens):
            raise ValueError('Invalid native list-of-token-structs')
        rows = ((token['token'], token['is_html'], token['start_byte'], token['end_byte'])
                for token in tokens)
    else:
        raise ValueError('Unsupported native token representation')
    visible = []
    previous_start = -1
    for token, is_html, start, end in rows:
        if not isinstance(token, str) or type(is_html) is not bool:
            raise ValueError('Invalid native token text or HTML flag')
        if type(start) is not int or type(end) is not int or not 0 <= start <= end:
            raise ValueError('Invalid native byte offsets')
        if start < previous_start:
            raise ValueError('Native token order is not monotonic')
        previous_start = start
        if not is_html:
            visible.append(token)
    text = ' '.join(visible)
    if not text.strip():
        raise ValueError('Native document has no visible text')
    return text


def identity_keys(document, text_hash):
    def normalized_title(title):
        return ' '.join(unicodedata.normalize('NFC', title).replace('_', ' ').split()).casefold()

    url = urllib.parse.urlsplit(document['url'])
    if url.scheme not in ('http', 'https') or url.hostname not in ('en.wikipedia.org', 'en.m.wikipedia.org'):
        raise ValueError('Native document URL is outside the Wikipedia source contract')
    if url.username is not None or url.password is not None:
        raise ValueError('Native document URL contains credentials')
    title = None
    if url.path.startswith('/wiki/'):
        title = urllib.parse.unquote(url.path[len('/wiki/'):])
    elif url.path == '/w/index.php':
        titles = urllib.parse.parse_qs(url.query).get('title', [])
        if len(titles) == 1:
            title = titles[0]
    if not title:
        raise ValueError('Native Wikipedia article identity cannot be recovered')
    return ('article:' + normalized_title(document['title']),
            'article:' + normalized_title(title), 'text:' + text_hash)


def memory_guard():
    fields = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    if int(fields['MemAvailable'].split()[0]) * 1024 < 4 * GIB:
        raise RuntimeError('Available memory is below the committed 4GiB floor')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 2 * GIB:
        raise RuntimeError('Source process exceeded the committed 2GiB RSS ceiling')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if protocol['schema'] != 'vey.neutral.nq.input-length-protocol.v1':
        raise ValueError('Unsupported census protocol')
    if args.output_root.exists():
        raise FileExistsError('Refusing to overwrite an acquisition or census')
    required_env = {'TOKENIZERS_PARALLELISM': 'false', 'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1',
                    'HF_HUB_DISABLE_XET': '1', 'OMP_NUM_THREADS': '1',
                    'OPENBLAS_NUM_THREADS': '1', 'ARROW_NUM_THREADS': '1'}
    if any(os.environ.get(key) != value for key, value in required_env.items()):
        raise ValueError('Committed anonymous/thread-limited environment is missing')
    os.umask(0o077)
    os.nice(10)
    resource.setrlimit(resource.RLIMIT_AS, (6 * GIB, 6 * GIB))
    memory_guard()
    repo = Path(__file__).resolve().parents[2]
    relative_inputs = [str(args.protocol.resolve().relative_to(repo)),
                       str(Path(__file__).resolve().relative_to(repo))]
    for pin in protocol['helper_sources']:
        relative_inputs.append(str(Path(pin['path']).resolve().relative_to(repo)))
        shared.check_pin(pin)
    commit = shared.paper._assert_committed_inputs(repo, relative_inputs)
    shared.check_pin(protocol['implementation'])
    if tuple(protocol['allowed_columns']) != INPUT_COLUMNS:
        raise ValueError('Protocol projection differs from the no-gold guard')
    for pin in protocol['source_metadata'].values():
        shared.check_pin(pin)
    tree = json.loads(Path(protocol['source_metadata']['file_tree']['path']).read_text())
    files = protocol['source']['files']
    if files != tree or sum(item['size'] for item in files) != protocol['source']['total_bytes']:
        raise ValueError('Committed shard inventory differs from the pinned primary metadata')
    data_root = Path(protocol['data_root']).resolve()
    if not args.output_root.resolve().is_relative_to(data_root):
        raise ValueError('Source output is outside the data root')
    cache_root = Path(protocol['cache_root']).resolve()
    if not cache_root.is_relative_to(data_root):
        raise ValueError('Source cache is outside the data root')
    if shutil.disk_usage(data_root).free < protocol['resource_budget']['minimum_free_disk_bytes']:
        raise RuntimeError('Insufficient disk headroom for the committed source budget')
    from huggingface_hub import hf_hub_download
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    tokenizers = {name: Tokenizer.from_file(str(shared.check_pin(pin)))
                  for name, pin in protocol['tokenizers'].items()}
    for tokenizer in tokenizers.values():
        tokenizer.no_truncation()
        tokenizer.no_padding()
    args.output_root.mkdir(mode=0o700, parents=True)
    custody_path = args.output_root / 'shard_custody.jsonl'
    ledger_path = args.output_root / 'document_token_counts.jsonl'
    custody = []
    records = []
    seen_ids = set()
    first_key = {}
    parents = {}
    lengths = defaultdict(list)

    def find(identity):
        while parents[identity] != identity:
            parents[identity] = parents[parents[identity]]
            identity = parents[identity]
        return identity

    with custody_path.open('x') as custody_stream, ledger_path.open('x') as ledger:
        for item in files:
            memory_guard()
            source_path = Path(hf_hub_download(
                repo_id=protocol['source']['repository'], repo_type='dataset',
                revision=protocol['source']['revision'], filename=item['path'],
                cache_dir=str(cache_root), token=False, endpoint='https://huggingface.co',
            ))
            if not source_path.resolve().is_relative_to(cache_root):
                raise ValueError('Resolved source blob escaped the committed cache')
            before = shared.file_hash(source_path)
            if source_path.stat().st_size != item['size'] or before != item['lfs']['oid']:
                raise ValueError('Downloaded shard failed upstream size/content identity')
            parquet = pq.ParquetFile(source_path)
            source_record = {'path': item['path'], 'local_path': str(source_path),
                             'bytes': item['size'], 'sha256': before,
                             'rows': parquet.metadata.num_rows,
                             'schema': str(parquet.schema_arrow),
                             'decoded_columns': list(INPUT_COLUMNS)}
            custody_stream.write(json.dumps(source_record, sort_keys=True)+'\n')
            custody_stream.flush()
            custody.append(source_record)
            for batch in checked_batches(parquet):
                row = batch.to_pylist()[0]
                if set(row) != {'id', 'document'}:
                    raise ValueError('Decoded record contains a forbidden top-level field')
                row_id = row['id']
                if not isinstance(row_id, str) or not row_id or row_id in seen_ids:
                    raise ValueError('Native example identity is empty or duplicated')
                seen_ids.add(row_id)
                text = document_text(row['document'])
                text_hash = hashlib.sha256(text.encode()).hexdigest()
                keys = identity_keys(row['document'], text_hash)
                parents[row_id] = row_id
                for key in keys:
                    if key in first_key:
                        left, right = find(row_id), find(first_key[key])
                        parents[max(left, right)] = min(left, right)
                    else:
                        first_key[key] = row_id
                counts = {name: len(tokenizer.encode(text, add_special_tokens=False).ids)
                          for name, tokenizer in tokenizers.items()}
                record = {'id': row_id, 'source_partition': 'official_validation',
                          'study_allocation': 'confirmation_only',
                          'title': row['document']['title'], 'url': row['document']['url'],
                          'substantive_text_sha256': text_hash, 'characters': len(text),
                          'identity_keys': list(keys), 'token_counts': counts}
                ledger.write(json.dumps(record, sort_keys=True, allow_nan=False)+'\n')
                records.append((row_id, counts, text_hash))
                for name, size in counts.items():
                    lengths[name].append(size)
                memory_guard()
            if shared.file_hash(source_path) != before:
                raise ValueError('Source shard changed during the read-only census')
    if len(records) != protocol['source']['expected_rows']:
        raise ValueError('Native source row count differs from the pinned metadata')
    if 'torch' in sys.modules:
        raise RuntimeError('Input-only source census imported a model framework')
    group_lengths = defaultdict(dict)
    membership_path = args.output_root / 'document_groups.jsonl'
    with membership_path.open('x') as stream:
        for row_id, counts, _ in records:
            root = find(row_id)
            stream.write(json.dumps({'id': row_id, 'component_id': root}, sort_keys=True)+'\n')
            for name, count in counts.items():
                group_lengths[name][root] = max(group_lengths[name].get(root, 0), count)
    summary = {}
    for name, values in lengths.items():
        grouped = list(group_lengths[name].values())
        summary[name] = {'rows': len(values), 'minimum': min(values),
                         'median': statistics.median(values), 'maximum': max(values),
                         'components': len(grouped),
                         'rows_by_threshold': {str(t): sum(n >= t for n in values)
                                               for t in protocol['thresholds']},
                         'components_with_any_qualifying_row_by_threshold': {
                             str(t): sum(n >= t for n in grouped) for t in protocol['thresholds']}}
    report = {'schema': 'vey.neutral.nq.input-length-result.v1', 'source_revision': protocol['source']['revision'],
              'protocol': {'path': str(args.protocol.resolve()), 'sha256': shared.file_hash(args.protocol)},
              'git_commit': commit, 'implementation_sha256': shared.file_hash(Path(__file__)),
              'shards': custody, 'source_shards_unchanged': True,
              'decoded_columns': list(INPUT_COLUMNS), 'questions_or_annotation_values_decoded': False,
              'native_annotations_preserved_in_immutable_raw_shards': True,
              'source_allocation': 'All official validation stays confirmation-only; no reassignment.',
              'model_forwards': 0, 'model_quality_or_long_context_capability_earned': False,
              'prior_training_or_selection_exposure': 'UNKNOWN',
              'grouping': protocol['grouping'], 'statistical_independence_earned': False,
              'tokenizers': protocol['tokenizers'], 'lengths': summary,
              'unique_substantive_text_hashes': len({record[2] for record in records}),
              'artifacts': {name: {'path': str(path.resolve()), 'sha256': shared.file_hash(path)}
                            for name, path in {'custody': custody_path, 'document_lengths': ledger_path,
                                               'document_groups': membership_path}.items()},
              'environment': {name: importlib.metadata.version(name)
                              for name in ('huggingface_hub', 'pyarrow', 'tokenizers')},
              'python': sys.version, 'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    report_path = args.output_root / 'report.json'
    with report_path.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'report': str(report_path), 'sha256': shared.file_hash(report_path),
                      'rows': len(records), 'lengths': summary, 'model_forwards': 0}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
