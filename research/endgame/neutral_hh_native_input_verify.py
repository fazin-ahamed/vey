#!/usr/bin/env python3
"""Independently reconstruct the registered TRAIN-only HH input census."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import unicodedata

PROTOCOL_SHA256 = '44dbcc32b446654aac816403e6d9f6206e2ff3ced7affe35a0f34f14d03b3686'
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DEFAULT_PROTOCOL = HERE / 'neutral_hh_native_input_protocol.json'
FLAGS = ('quality_credit', 'shipping_credit', 'sealed_TEST_accessed', 'B_STEF_allowed')
LIMIT = 2097152


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def decode(raw):
    def invalid_constant(value):
        raise ValueError('Nonfinite JSON constant')
    return json.loads(raw.decode('utf-8'), object_pairs_hook=strict_object,
                      parse_constant=invalid_constant)


def digest_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def identity(path):
    digest = hashlib.sha256()
    size = 0
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return {'path': str(path.resolve()), 'bytes': size, 'sha256': digest.hexdigest()}


def check_identity(pin, expected_path):
    if not isinstance(pin, dict) or set(pin) != {'path', 'bytes', 'sha256'}:
        raise ValueError('Malformed artifact pin')
    if type(pin['bytes']) is not int or pin['bytes'] < 0:
        raise ValueError('Malformed byte count')
    if Path(pin['path']).resolve() != expected_path.resolve():
        raise ValueError('Unregistered artifact path')
    observed = identity(expected_path)
    if pin != observed:
        raise ValueError('Artifact identity mismatch')
    return observed


def committed(path):
    relative = path.resolve().relative_to(REPO).as_posix()
    stored = subprocess.run(['git', '-C', str(REPO), 'show', 'HEAD:' + relative],
                            check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout
    if stored != path.read_bytes():
        raise ValueError('Implementation or protocol is not committed unchanged')


def resource_guard():
    with Path('/proc/meminfo').open() as stream:
        available = next(int(line.split()[1]) * 1024 for line in stream
                         if line.startswith('MemAvailable:'))
    if available < 4 * 1024 ** 3:
        raise RuntimeError('Available RAM below registered minimum')
    if any(name in sys.modules for name in ('torch', 'tensorflow', 'jax')):
        raise RuntimeError('Model framework loaded')


def reconstruct(raw, ordinal):
    native = decode(raw)
    if not isinstance(native, dict) or set(native) != {'chosen', 'rejected'}:
        raise ValueError('Native row must contain exactly two registered keys')
    if any(type(value) is not str for value in native.values()):
        raise ValueError('Native transcript must be a string')
    transcripts = list(native.values())
    hashes = sorted(digest_text(text) for text in transcripts)
    row = {'ordinal': ordinal, 'source_line_sha256': hashlib.sha256(raw).hexdigest(),
           'full_transcript_sha256s': hashes,
           'input_pair_sha256': digest_text(json.dumps(hashes, separators=(',', ':'))),
           'projectable': False, 'reason': None}
    partitions = [text.rpartition('\n\nAssistant:') for text in transcripts]
    contexts = [part[0] for part in partitions]
    responses = [part[2] for part in partitions]
    if any(not part[1] for part in partitions):
        reason = 'missing_last_assistant_marker'
    elif contexts[0] != contexts[1]:
        reason = 'shared_context_mismatch'
    elif not contexts[0].startswith('\n\nHuman:'):
        reason = 'unexpected_context_prefix'
    elif any(not text.strip() for text in responses):
        reason = 'blank_response'
    elif responses[0] == responses[1]:
        reason = 'identical_response'
    else:
        reason = None
    row['reason'] = reason
    if reason is None:
        context = contexts[0]
        root = context[len('\n\nHuman:'):].split('\n\nAssistant:', 1)[0]
        normalized = ' '.join(unicodedata.normalize('NFKC', root).casefold().split())
        row.update(projectable=True, context_sha256=digest_text(context),
                   response_sha256s=sorted(digest_text(text) for text in responses),
                   context_characters=len(context),
                   response_characters=sorted(len(text) for text in responses),
                   root_prompt_proxy_sha256=digest_text(normalized))
    return row, max(map(len, transcripts))


def same_json(left, right):
    # Serialization preserves the distinction between booleans and integer counts.
    return json.dumps(left, sort_keys=True, allow_nan=False, separators=(',', ':')) == \
        json.dumps(right, sort_keys=True, allow_nan=False, separators=(',', ':'))


def verify(protocol_path):
    if protocol_path.resolve() != DEFAULT_PROTOCOL:
        raise ValueError('Only the registered protocol path is allowed')
    protocol_pin = identity(protocol_path)
    if protocol_pin['sha256'] != PROTOCOL_SHA256:
        raise ValueError('Immutable protocol changed')
    committed(protocol_path)
    resource_guard()
    protocol = decode(protocol_path.read_bytes())
    if protocol['schema'] != 'vey.neutral.hh-native-input-census.v1':
        raise ValueError('Unsupported protocol')
    if any(protocol.get(flag) is not False for flag in FLAGS):
        raise ValueError('Protocol grants forbidden credit')
    root = Path(protocol['data']['root']).resolve()
    output = root / 'input-census-v1'
    verification = output / 'verification.json'
    if verification.exists():
        raise FileExistsError('Verification evidence already exists')
    if (output / 'failure.json').exists():
        raise ValueError('Census has a failure receipt')
    manifest_path = output / 'manifest.json'
    manifest_pin = identity(manifest_path)
    manifest = decode(manifest_path.read_bytes())
    if manifest.get('schema') != 'vey.neutral.hh-native-input-census-result.v1' or manifest.get('status') != 'COMPLETE':
        raise ValueError('Census is not COMPLETE')
    if any(manifest.get(flag) is not False for flag in FLAGS):
        raise ValueError('Manifest grants forbidden credit')
    check_identity(manifest['protocol'], protocol_path)
    code_paths = [HERE / 'neutral_hh_native_input_census.py', Path(__file__).resolve()]
    pins = manifest['source_code']
    if not isinstance(pins, list) or len(pins) != len(code_paths):
        raise ValueError('Incomplete implementation pins')
    code_by_path = {str(Path(pin['path']).resolve()): pin for pin in pins}
    if set(code_by_path) != {str(path) for path in code_paths}:
        raise ValueError('Implementation pin membership mismatch')
    for path in code_paths:
        committed(path)
        check_identity(code_by_path[str(path)], path)
    card_path = Path(protocol['source']['card']['path']).resolve()
    card_pin = check_identity(protocol['source']['card'], card_path)
    check_identity(manifest['source_card'], card_path)
    gzip_path = root / 'raw' / 'helpful-base' / 'train.jsonl.gz'
    custody = manifest.get('committed_identity')
    expected_paths = sorted(path.relative_to(REPO).as_posix()
                            for path in [protocol_path, *code_paths])
    if not isinstance(custody, dict) or sorted(custody.get('paths', [])) != expected_paths:
        raise ValueError('Committed identity membership mismatch')
    revision = custody.get('commit')
    if not isinstance(revision, str) or len(revision) != 40 or any(
            character not in '0123456789abcdef' for character in revision):
        raise ValueError('Malformed committed identity')
    for relative in expected_paths:
        stored = subprocess.run(
            ['git', '-C', str(REPO), 'show', revision + ':' + relative],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout
        if stored != (REPO / relative).read_bytes():
            raise ValueError('Census commit source drift')
    registered = protocol['source']['train_file']
    if registered['path'] != 'helpful-base/train.jsonl.gz':
        raise ValueError('Unregistered source tranche')
    gzip_pin = check_identity({'path': str(gzip_path), 'bytes': registered['bytes'],
                               'sha256': registered['sha256']}, gzip_path)
    check_identity(manifest['source_gzip'], gzip_path)
    ledger_path = output / 'rows.jsonl'
    ledger_pin = check_identity(manifest['ledger'], ledger_path)
    resource_guard()
    reasons, pairs, roots = Counter(), Counter(), Counter()
    contexts = set()
    rows = projectable = max_transcript = max_context = max_response = 0
    with gzip.open(gzip_path, 'rb') as source, ledger_path.open('rb') as ledger:
        while raw := source.readline(LIMIT + 1):
            if len(raw) > LIMIT:
                raise ValueError('Native JSON line exceeds registered bound')
            if rows % 1024 == 0:
                resource_guard()
            row, transcript_length = reconstruct(raw, rows)
            exported = ledger.readline(LIMIT + 1)
            if not exported or len(exported) > LIMIT:
                raise ValueError('Missing or oversized ledger row')
            if not same_json(decode(exported), row):
                raise ValueError('Independent ledger reconstruction mismatch at ordinal ' + str(rows))
            rows += 1
            reasons[row['reason'] or 'projectable'] += 1
            pairs[row['input_pair_sha256']] += 1
            max_transcript = max(max_transcript, transcript_length)
            if row['projectable']:
                projectable += 1
                contexts.add(row['context_sha256'])
                roots[row['root_prompt_proxy_sha256']] += 1
                max_context = max(max_context, row['context_characters'])
                max_response = max(max_response, *row['response_characters'])
        if ledger.read(1):
            raise ValueError('Extra ledger rows')
    counts = {'rows': rows, 'projectable_rows': projectable,
              'unprojectable_rows': rows - projectable,
              'projection_reason_counts': dict(reasons), 'unique_contexts': len(contexts),
              'unique_root_prompt_proxies': len(roots),
              'max_root_prompt_proxy_rows': max(roots.values(), default=0),
              'unique_input_pairs': len(pairs),
              'duplicate_input_pair_rows': sum(count - 1 for count in pairs.values()),
              'max_transcript_characters': max_transcript,
              'max_context_characters': max_context, 'max_response_characters': max_response}
    if not same_json(manifest.get('counts'), counts):
        raise ValueError('Independent summary reconstruction mismatch')
    # Recheck custody after reading so concurrent replacement cannot earn verification.
    for pin in [protocol_pin, card_pin, gzip_pin, ledger_pin, manifest_pin, *pins]:
        check_identity(pin, Path(pin['path']))
    for path in [protocol_path, *code_paths]:
        committed(path)
    result = {'schema': 'vey.neutral.hh-native-input-census-verification.v1',
              'status': 'VERIFIED', 'counts': counts, 'protocol': protocol_pin,
              'source_card': card_pin, 'source_gzip': gzip_pin, 'source_code': pins,
              'ledger': ledger_pin, 'manifest': manifest_pin,
              'independent_reconstruction': True, 'capability_credit': False,
              'model_loads': 0, 'model_forwards': 0, **dict.fromkeys(FLAGS, False)}
    with verification.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    current_nice = os.getpriority(os.PRIO_PROCESS, 0)
    if current_nice < 10:
        os.nice(10 - current_nice)
    def expired(signum, frame):
        raise TimeoutError('Registered wall-time bound exceeded')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(600)
    try:
        result = verify(args.protocol)
    except Exception as error:
        failure_root = Path('/home/fazinahamed/Documents/vey-data/decisionmix/endgame/hh-native-v1/input-census-v1')
        if failure_root.is_dir() and not (failure_root / 'verification.json').exists():
            receipt = {'schema': 'vey.neutral.hh-native-input-census-failure.v1',
                       'status': 'FAILED', 'stage': 'independent_verification',
                       'error_type': type(error).__name__, **dict.fromkeys(FLAGS, False)}
            try:
                with (failure_root / 'failure.json').open('x', encoding='utf-8') as stream:
                    json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
                    stream.write('\n')
            except OSError:
                pass
        # Never print parser exception text: it can contain source transcript bytes.
        print(json.dumps({'status': 'FAILED', 'error_type': type(error).__name__,
                          **dict.fromkeys(FLAGS, False)}, sort_keys=True), file=sys.stderr)
        return 1
    finally:
        signal.alarm(0)
    print(json.dumps({'status': result['status'], 'counts': result['counts'],
                      **dict.fromkeys(FLAGS, False)}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
