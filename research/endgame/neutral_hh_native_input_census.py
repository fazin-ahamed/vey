#!/usr/bin/env python3
"""Census the pinned helpful-base TRAIN inputs without exporting preference roles."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import unicodedata

PROTOCOL_SHA256 = '44dbcc32b446654aac816403e6d9f6206e2ff3ced7affe35a0f34f14d03b3686'
MAX_LINE_BYTES = 2097152
FLAGS = dict.fromkeys(('quality_credit', 'shipping_credit', 'sealed_TEST_accessed', 'B_STEF_allowed'), False)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def _constant(value):
    raise ValueError('Nonfinite JSON constant')


def decode_row(raw_line):
    if not isinstance(raw_line, bytes) or not raw_line or len(raw_line) > MAX_LINE_BYTES:
        raise ValueError('Invalid native line size')
    value = json.loads(raw_line.decode('utf-8'), object_pairs_hook=_object,
                       parse_constant=_constant)
    _validate(value)
    return value


def _validate(value):
    if not isinstance(value, dict) or set(value) != {'chosen', 'rejected'}:
        raise ValueError('Invalid native object keys')
    if any(not isinstance(text, str) for text in value.values()):
        raise ValueError('Invalid native transcript type')
    for text in value.values():
        text.encode('utf-8')


def row_features(value, raw_line, ordinal):
    """Return unordered input metadata; native-line identity alone retains role order."""
    _validate(value)
    if type(ordinal) is not int or ordinal < 0:
        raise ValueError('Invalid source ordinal')
    if decode_row(raw_line) != value:
        raise ValueError('Parsed value differs from native bytes')
    texts = list(value.values())
    hashes = sorted(digest(text.encode('utf-8')) for text in texts)
    row = {'ordinal': ordinal, 'source_line_sha256': digest(raw_line),
           'full_transcript_sha256s': hashes,
           'input_pair_sha256': digest(json.dumps(hashes, separators=(',', ':')).encode('utf-8')),
           'projectable': False, 'reason': None}
    parts = [text.rpartition('\n\nAssistant:') for text in texts]
    contexts = [part[0] for part in parts]
    responses = [part[2] for part in parts]
    if any(not part[1] for part in parts):
        reason = 'missing_last_assistant_marker'
    elif contexts[0] != contexts[1]:
        reason = 'shared_context_mismatch'
    elif not contexts[0].startswith('\n\nHuman:'):
        reason = 'unexpected_context_prefix'
    elif any(not response.strip() for response in responses):
        reason = 'blank_response'
    elif responses[0] == responses[1]:
        reason = 'identical_response'
    else:
        reason = None
    row['reason'] = reason
    if reason is None:
        context = contexts[0]
        root = context[len('\n\nHuman:'):].split('\n\nAssistant:', 1)[0]
        root = ' '.join(unicodedata.normalize('NFKC', root).casefold().split())
        row.update(projectable=True, context_sha256=digest(context.encode('utf-8')),
                   response_sha256s=sorted(digest(text.encode('utf-8')) for text in responses),
                   context_characters=len(context), response_characters=sorted(map(len, responses)),
                   root_prompt_proxy_sha256=digest(root.encode('utf-8')))
    return row


class Counts:
    def __init__(self):
        self.rows = 0
        self.reasons = Counter()
        self.contexts = set()
        self.roots = Counter()
        self.pairs = Counter()
        self.max_transcript = self.max_context = self.max_response = 0

    def add(self, row, value):
        self.rows += 1
        self.reasons[row['reason'] or 'projectable'] += 1
        self.pairs[row['input_pair_sha256']] += 1
        self.max_transcript = max(self.max_transcript, *(len(text) for text in value.values()))
        if row['projectable']:
            self.contexts.add(row['context_sha256'])
            self.roots[row['root_prompt_proxy_sha256']] += 1
            self.max_context = max(self.max_context, row['context_characters'])
            self.max_response = max(self.max_response, *row['response_characters'])

    def result(self):
        return {'rows': self.rows, 'projectable_rows': self.reasons['projectable'],
                'unprojectable_rows': self.rows - self.reasons['projectable'],
                'projection_reason_counts': dict(sorted(self.reasons.items())),
                'unique_contexts': len(self.contexts), 'unique_root_prompt_proxies': len(self.roots),
                'max_root_prompt_proxy_rows': max(self.roots.values(), default=0),
                'unique_input_pairs': len(self.pairs),
                'duplicate_input_pair_rows': sum(count - 1 for count in self.pairs.values()),
                'max_transcript_characters': self.max_transcript,
                'max_context_characters': self.max_context,
                'max_response_characters': self.max_response}

def memory_guard():
    fields = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    if int(fields['MemAvailable'].split()[0]) * 1024 < 4 * 1024 ** 3:
        raise RuntimeError('Available RAM below registered floor')

def pin_stream(path, stream, guard=True):
    stream.seek(0)
    hasher = hashlib.sha256()
    size = 0
    while block := stream.read(1024 * 1024):
        if guard:
            memory_guard()
        hasher.update(block)
        size += len(block)
    stream.seek(0)
    return {'path': str(path.resolve()), 'bytes': size, 'sha256': hasher.hexdigest()}


def pin(path, guard=True):
    with path.open('rb') as stream:
        return pin_stream(path, stream, guard=guard)


def committed_identity(repo, paths):
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, timeout=10).decode().strip()
    relative = [str(path.resolve().relative_to(repo)) for path in paths]
    for path, name in zip(paths, relative):
        stored = subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=repo, timeout=10)
        if stored != path.read_bytes():
            raise ValueError('Registered source differs from committed identity')
        staged = subprocess.check_output(['git', 'show', f':{name}'], cwd=repo, timeout=10)
        if staged != stored:
            raise ValueError('Registered index differs from committed identity')
    return {'commit': commit, 'paths': relative}


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def _timeout(signum, frame):
    raise TimeoutError('Registered wall deadline exceeded')


def run(protocol_path):
    protocol_path = protocol_path.resolve()
    protocol_bytes = protocol_path.read_bytes()
    if digest(protocol_bytes) != PROTOCOL_SHA256:
        raise ValueError('Unregistered protocol')
    protocol = json.loads(protocol_bytes)
    repo = Path(__file__).resolve().parents[2]
    root = Path(protocol['data']['root']).resolve()
    if root.is_relative_to(repo):
        raise ValueError('Raw source cannot reside in the repository')
    output = root / 'input-census-v1'
    os.umask(0o077)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    ledger_path = output / 'rows.jsonl'
    counts = Counts()
    try:
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[key] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        if os.getpriority(os.PRIO_PROCESS, 0) < 10:
            os.setpriority(os.PRIO_PROCESS, 0, 10)
        signal.signal(signal.SIGALRM, _timeout)
        signal.setitimer(signal.ITIMER_REAL, 600)
        memory_guard()
        census_path = Path(__file__).resolve()
        verifier_path = census_path.with_name('neutral_hh_native_input_verify.py')
        identity = committed_identity(repo, [protocol_path, census_path, verifier_path])
        protocol_pin = pin(protocol_path)
        if protocol_pin['sha256'] != PROTOCOL_SHA256:
            raise ValueError('Protocol changed during custody check')
        code_pins = [pin(path) for path in (census_path, verifier_path)]
        card_path = Path(protocol['source']['card']['path'])
        card_pin = pin(card_path)
        if any(card_pin[key] != protocol['source']['card'][key] for key in ('bytes', 'sha256')):
            raise ValueError('Source card identity mismatch')
        source_path = root / 'raw' / 'helpful-base' / 'train.jsonl.gz'
        expected = protocol['source']['train_file']
        with source_path.open('rb') as source:
            source_pin = pin_stream(source_path, source)
            if any(source_pin[key] != expected[key] for key in ('bytes', 'sha256')):
                raise ValueError('TRAIN compressed identity mismatch')
            with ledger_path.open('x', encoding='utf-8') as ledger, gzip.GzipFile(fileobj=source, mode='rb') as native:
                ordinal = 0
                while True:
                    memory_guard()
                    raw_line = native.readline(MAX_LINE_BYTES + 1)
                    if not raw_line:
                        break
                    value = decode_row(raw_line)
                    row = row_features(value, raw_line, ordinal)
                    ledger.write(json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n')
                    ledger.flush()
                    counts.add(row, value)
                    ordinal += 1
                os.fsync(ledger.fileno())
            if pin_stream(source_path, source) != source_pin or pin(source_path) != source_pin:
                raise ValueError('TRAIN changed during census')
        if pin(protocol_path) != protocol_pin or [pin(path) for path in (census_path, verifier_path)] != code_pins:
            raise ValueError('Registered sources changed during census')
        if pin(card_path) != card_pin or committed_identity(repo, [protocol_path, census_path, verifier_path]) != identity:
            raise ValueError('Committed custody changed during census')
        manifest = {'schema': 'vey.neutral.hh-native-input-census-result.v1', 'status': 'COMPLETE',
                    'committed_identity': identity, 'protocol': protocol_pin, 'source_card': card_pin,
                    'source_gzip': source_pin, 'source_code': code_pins, 'ledger': pin(ledger_path),
                    'counts': counts.result(), **FLAGS}
        write_json(output / 'manifest.json', manifest)
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        receipt = {'schema': 'vey.neutral.hh-native-input-census-failure.v1', 'status': 'FAILED',
                   'error_type': type(error).__name__, 'completed_rows': counts.rows, **FLAGS}
        if ledger_path.exists():
            receipt['ledger'] = pin(ledger_path, guard=False)
        write_json(output / 'failure.json', receipt)
        raise RuntimeError('Census failed; metadata-only receipt retained') from None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=Path(__file__).with_name('neutral_hh_native_input_protocol.json'))
    args = parser.parse_args()
    try:
        run(args.protocol)
    except Exception as error:
        parser.exit(1, f'Census failed ({type(error).__name__}); no source payload displayed.\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
