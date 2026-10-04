#!/usr/bin/env python3
"""Independently replay the pinned NQ input-only census from local Parquet bytes."""
import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import sys
import unicodedata
from urllib.parse import parse_qs, unquote, urlsplit

COLUMNS = ('id', 'document.title', 'document.url', 'document.tokens')
FIELDS = {'token', 'is_html', 'start_byte', 'end_byte'}
THREAD_ENV = {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
              'MKL_NUM_THREADS': '1', 'ARROW_NUM_THREADS': '1',
              'TOKENIZERS_PARALLELISM': 'false'}
for _key, _value in THREAD_ENV.items():
    os.environ[_key] = _value
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'


class VerificationFailure(Exception):
    pass


class Receipt:
    def __init__(self):
        self.data = {
            'schema': 'vey.neutral.nq.input-length-independent-verification.v1',
            'status': 'FAIL', 'assertions': {}, 'failures': [], 'pins': [],
            'coverage': {'rows': 0, 'shards': 0, 'components': 0},
            'decoded_columns': list(COLUMNS), 'model_forwards': 0,
            'questions_or_annotation_values_decoded': False,
            'boundaries': [
                'Connected components control source overlap, not author/template/user independence.',
                'No model capability or statistical-quality claim is earned.',
                'Official validation remains confirmation-only; no allocation migration.',
                'Prior training or selection exposure remains UNKNOWN.',
                'No current ECA-2 final IR, features or outcomes are accessed.'],
        }

    def check(self, label, condition):
        entry = self.data['assertions'].setdefault(label, {'checked': 0, 'failed': 0})
        entry['checked'] += 1
        if not condition:
            entry['failed'] += 1
            self.data['failures'].append(label)
            raise VerificationFailure(label)

    def pin(self, pin, label):
        path = Path(pin['path'])
        actual = digest(path)
        self.data['pins'].append({'role': label, 'path': str(path.resolve()),
                                  'sha256': actual, 'bytes': path.stat().st_size})
        self.check(label + '_sha256', actual == pin['sha256'])
        return path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise VerificationFailure('duplicate_json_key')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs)


def json_lines(path):
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip():
                raise VerificationFailure('blank_ledger_line')
            yield json.loads(line)


def source_text(document, receipt):
    receipt.check('document_projection', set(document) == {'title', 'url', 'tokens'})
    receipt.check('native_identity_types', all(isinstance(document[k], str) and document[k]
                                             for k in ('title', 'url')))
    native = document['tokens']
    if isinstance(native, dict):
        receipt.check('token_struct_fields', set(native) == FIELDS)
        receipt.check('token_list_types', all(isinstance(v, list) for v in native.values()))
        receipt.check('token_list_lengths', len({len(v) for v in native.values()}) == 1)
        iterator = zip(native['token'], native['is_html'], native['start_byte'], native['end_byte'])
    else:
        receipt.check('token_layout', isinstance(native, list))
        receipt.check('token_row_fields', all(isinstance(v, dict) and set(v) == FIELDS for v in native))
        iterator = ((v['token'], v['is_html'], v['start_byte'], v['end_byte']) for v in native)
    visible = []
    last = -1
    for token, html, start, end in iterator:
        receipt.check('token_native_types', isinstance(token, str) and type(html) is bool
                      and type(start) is int and type(end) is int)
        receipt.check('token_offsets_order', 0 <= start <= end and start >= last)
        last = start
        if not html:
            visible.append(token)
    text = chr(32).join(visible)
    receipt.check('nonempty_substantive_text', bool(text.strip()))
    return text


def keys_for(document, text_hash, receipt):
    url = urlsplit(document['url'])
    receipt.check('url_source_contract', url.scheme in {'http', 'https'}
                  and url.hostname in {'en.wikipedia.org', 'en.m.wikipedia.org'}
                  and url.username is None and url.password is None)
    dispatch = '/' + url.path.lstrip('/')
    if dispatch.startswith('/wiki/'):
        title = unquote(dispatch[6:])
    elif dispatch == '/w/index.php':
        values = parse_qs(url.query).get('title', [])
        receipt.check('url_single_query_title', len(values) == 1)
        title = values[0]
    else:
        raise VerificationFailure('url_article_dispatch')
    receipt.check('url_nonempty_title', bool(title))
    def normal(value):
        return ' '.join(unicodedata.normalize('NFC', value).replace('_', ' ').split()).casefold()
    return ['article:' + normal(document['title']), 'article:' + normal(title), 'text:' + text_hash]


def verify(args, receipt):
    protocol = load_json(args.protocol)
    report = load_json(args.report)
    for role, path in [('protocol', args.protocol), ('producer_report', args.report),
                       ('verifier', Path(__file__))]:
        receipt.data['pins'].append({'role': role, 'path': str(path.resolve()),
                                    'sha256': digest(path), 'bytes': path.stat().st_size})
    receipt.check('protocol_schema', protocol['schema'] == 'vey.neutral.nq.input-length-protocol.v1')
    receipt.check('report_schema', report['schema'] == 'vey.neutral.nq.input-length-result.v1')
    receipt.check('protocol_binding', report['protocol']['sha256'] == digest(args.protocol)
                  and Path(report['protocol']['path']).resolve() == args.protocol.resolve())
    receipt.check('projection_authority', protocol['allowed_columns'] == list(COLUMNS)
                  and report['decoded_columns'] == list(COLUMNS))
    root = Path(protocol['data_root']).resolve()
    receipt.check('private_output_scope', args.output.resolve().is_relative_to(root))
    receipt.check('report_source_revision', report['source_revision'] == protocol['source']['revision'])
    receipt.pin(protocol['implementation'], 'producer_source')
    receipt.check('runtime_package_versions',
                  all(importlib.metadata.version(name) == protocol['environment'][name]
                      for name in ('pyarrow', 'tokenizers')))
    receipt.check('producer_environment_binding',
                  all(report['environment'][name] == protocol['environment'][name]
                      for name in ('huggingface_hub', 'pyarrow', 'tokenizers'))
                  and report['python'] == protocol['environment']['python'])
    receipt.check('producer_source_binding', report['implementation_sha256'] == protocol['implementation']['sha256'])
    for index, pin in enumerate(protocol['helper_sources']):
        receipt.pin(pin, 'helper_' + str(index))
    for name, pin in protocol['source_metadata'].items():
        receipt.pin(pin, 'source_metadata_' + name)
    receipt.check('primary_file_tree', load_json(protocol['source_metadata']['file_tree']['path'])
                  == protocol['source']['files'])
    receipt.check('tokenizer_report_binding', report['tokenizers'] == protocol['tokenizers'])
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    tokenizers = {}
    for name, pin in protocol['tokenizers'].items():
        tokenizers[name] = Tokenizer.from_file(str(receipt.pin(pin, 'tokenizer_' + name)))
        tokenizers[name].no_padding()
        tokenizers[name].no_truncation()
    receipt.check('two_laya_tokenizers', set(tokenizers) == {'english', 'multilingual'})
    artifacts = {name: receipt.pin(pin, 'artifact_' + name) for name, pin in report['artifacts'].items()}
    receipt.check('artifact_exact_roles', set(artifacts) == {'custody', 'document_lengths', 'document_groups'})
    receipt.check('artifact_data_scope', all(p.resolve().is_relative_to(root) for p in artifacts.values()))
    custody = list(json_lines(artifacts['custody']))
    receipt.check('custody_report_equality', custody == report['shards'])
    files = protocol['source']['files']
    receipt.check('seven_exact_objects', len(files) == len(custody) == 7)
    receipt.check('total_source_bytes', sum(f['size'] for f in files) == protocol['source']['total_bytes'])
    ledger = iter(json_lines(artifacts['document_lengths']))
    groups = iter(json_lines(artifacts['document_groups']))
    seen = set()
    records = []
    adjacency = defaultdict(set)
    first = {}
    before_after = []
    for item, claim in zip(files, custody):
        path = Path(claim['local_path'])
        receipt.check('cache_scope', path.resolve().is_relative_to(Path(protocol['cache_root']).resolve()))
        before = digest(path)
        receipt.check('shard_upstream_identity', path.stat().st_size == item['size']
                      and before == item['lfs']['oid'])
        parquet = pq.ParquetFile(path)
        expected_custody = {'path': item['path'], 'local_path': str(path), 'bytes': item['size'],
                            'sha256': before, 'rows': parquet.metadata.num_rows,
                            'schema': str(parquet.schema_arrow), 'decoded_columns': list(COLUMNS)}
        receipt.check('shard_custody_exact', claim == expected_custody)
        count = 0
        for batch in parquet.iter_batches(batch_size=1, columns=list(COLUMNS), use_threads=False):
            receipt.check('one_row_batch', batch.num_rows == 1)
            source = batch.to_pylist()[0]
            receipt.check('top_level_projection', set(source) == {'id', 'document'})
            identity = source['id']
            receipt.check('unique_native_id', isinstance(identity, str) and bool(identity) and identity not in seen)
            seen.add(identity)
            text = source_text(source['document'], receipt)
            text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
            keys = keys_for(source['document'], text_hash, receipt)
            counts = {name: len(tok.encode(text, add_special_tokens=False).ids)
                      for name, tok in tokenizers.items()}
            rebuilt = {'id': identity, 'source_partition': 'official_validation',
                       'study_allocation': 'confirmation_only', 'title': source['document']['title'],
                       'url': source['document']['url'], 'substantive_text_sha256': text_hash,
                       'characters': len(text), 'identity_keys': keys, 'token_counts': counts}
            receipt.check('row_ledger_exact', next(ledger, None) == rebuilt)
            adjacency[identity]
            for key in keys:
                if key in first:
                    neighbor = first[key]
                    adjacency[identity].add(neighbor)
                    adjacency[neighbor].add(identity)
                else:
                    first[key] = identity
            records.append((identity, counts, text_hash))
            count += 1
            receipt.data['coverage']['rows'] += 1
        receipt.check('shard_full_rows', count == parquet.metadata.num_rows)
        after = digest(path)
        receipt.check('shard_before_after', before == after)
        before_after.append({'path': item['path'], 'bytes': path.stat().st_size,
                             'sha256_before': before, 'sha256_after': after, 'rows': count})
        receipt.data['coverage']['shards'] += 1
    receipt.check('ledger_no_extra_rows', next(ledger, None) is None)
    receipt.check('all_expected_rows', len(records) == protocol['source']['expected_rows'] == 7830)
    components = {}
    remaining = set(seen)
    while remaining:
        seed = min(remaining)
        stack = [seed]
        members = set()
        while stack:
            node = stack.pop()
            if node in members:
                continue
            members.add(node)
            stack.extend(adjacency[node] - members)
        remaining.difference_update(members)
        representative = min(members)
        for member in members:
            components[member] = representative
    grouped = {name: defaultdict(int) for name in tokenizers}
    lengths = {name: [] for name in tokenizers}
    for identity, counts, _ in records:
        receipt.check('dfs_group_ledger_exact', next(groups, None) ==
                      {'id': identity, 'component_id': components[identity]})
        for name, count in counts.items():
            lengths[name].append(count)
            group = components[identity]
            grouped[name][group] = max(grouped[name][group], count)
    receipt.check('group_ledger_no_extra_rows', next(groups, None) is None)
    summary = {}
    for name, values in lengths.items():
        summary[name] = {'rows': len(values), 'minimum': min(values),
                         'median': statistics.median(values), 'maximum': max(values),
                         'components': len(grouped[name]),
                         'rows_by_threshold': {str(t): sum(v >= t for v in values) for t in protocol['thresholds']},
                         'components_with_any_qualifying_row_by_threshold': {
                             str(t): sum(v >= t for v in grouped[name].values()) for t in protocol['thresholds']}}
    receipt.check('full_length_summary', summary == report['lengths'])
    unique_hashes = len({h for _, _, h in records})
    receipt.check('unique_text_hashes', unique_hashes == report['unique_substantive_text_hashes'])
    receipt.check('grouping_authority', report['grouping'] == protocol['grouping'])
    receipt.check('allocation_claim', report['source_allocation'] ==
                  'All official validation stays confirmation-only; no reassignment.')
    receipt.check('report_boundaries', report['source_shards_unchanged'] is True
                  and report['questions_or_annotation_values_decoded'] is False
                  and report['native_annotations_preserved_in_immutable_raw_shards'] is True
                  and report['model_forwards'] == 0
                  and report['model_quality_or_long_context_capability_earned'] is False
                  and report['statistical_independence_earned'] is False
                  and report['prior_training_or_selection_exposure'] == 'UNKNOWN')
    receipt.check('no_model_framework_import', not any(k.split('.')[0] in
                  {'torch', 'transformers', 'tensorflow', 'jax'} for k in sys.modules))
    receipt.data.update({'lengths': summary, 'unique_substantive_text_hashes': unique_hashes,
                         'shard_before_after': before_after})
    receipt.data['coverage']['components'] = len(set(components.values()))
    receipt.check('component_reconstruction_expectation', len(set(components.values())) == 6930)
    for name, expected in {'english': ((2810, 2288), (1052, 815)),
                           'multilingual': ((3045, 2483), (1242, 956))}.items():
        for threshold, (rows, component_count) in zip((8192, 16384), expected):
            receipt.check('reconstructed_threshold_expectation',
                          summary[name]['rows_by_threshold'][str(threshold)] == rows
                          and summary[name]['components_with_any_qualifying_row_by_threshold'][str(threshold)]
                          == component_count)
    for name, pin in report['artifacts'].items():
        receipt.check('artifact_unchanged_after_replay', digest(artifacts[name]) == pin['sha256'])
    receipt.check('protocol_unchanged_after_replay', digest(args.protocol) == report['protocol']['sha256'])
    receipt.data['status'] = 'PASS'


def package_versions():
    versions = {}
    for name in ('pyarrow', 'tokenizers'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    output_root = Path(load_json(args.protocol)['data_root']).resolve()
    if not args.output.resolve().is_relative_to(output_root):
        raise SystemExit('Verification receipt must stay inside the private data root')
    if args.output.exists():
        raise SystemExit('Refusing to overwrite a verification receipt')
    # Reserve the exclusive receipt before any reconstruction so failures survive.
    with args.output.open('x', encoding='utf-8') as stream:
        receipt = Receipt()
        try:
            verify(args, receipt)
        except Exception as exc:
            label = str(exc) if isinstance(exc, VerificationFailure) else type(exc).__name__
            if label not in receipt.data['failures']:
                receipt.data['failures'].append(label)
            receipt.data['status'] = 'FAIL'
        receipt.data['environment'] = {
            'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
            'threads': {k: os.environ[k] for k in THREAD_ENV},
            'packages': package_versions(),
            'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        }
        receipt.data['environment_sha256'] = hashlib.sha256(json.dumps(
            receipt.data['environment'], sort_keys=True).encode()).hexdigest()
        json.dump(receipt.data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': receipt.data['status'], 'output': str(args.output),
                      'sha256': digest(args.output), 'coverage': receipt.data['coverage'],
                      'failure_count': len(receipt.data['failures'])}, sort_keys=True))
    return 0 if receipt.data['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
