#!/usr/bin/env python3
"""Offline synthetic typed compatibility of the exact published Laya 0.3.27."""
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import socket
import sys
import traceback

ROOT = Path('/home/fazinahamed/Documents/vey-data/decisionmix/endgame/competitor-runtime/laya-0.3.27-preparation')
MANIFEST = Path(__file__).with_name('laya_published_0_3_27_custody_manifest.json')
PROTOCOL = Path(__file__).with_name('laya_published_0_3_27_protocol.json')
OUTPUT = ROOT / 'cpu-compatibility-result.json'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def offline(*args, **kwargs):
    raise RuntimeError('Network denied by offline compatibility probe')


def main():
    require(not OUTPUT.exists(), 'Refusing to overwrite exercised receipt')
    protocol = json.loads(PROTOCOL.read_text())
    require(sha(MANIFEST) == protocol['custody_manifest_sha256'], 'Custody manifest changed')
    require(sha(Path(__file__)) == protocol['probe_sha256'], 'Probe differs from prospective protocol')
    manifest = json.loads(MANIFEST.read_text())
    state = {'text': 'I was charged twice. Please refund the duplicate charge.'}
    questions = {
        'department': {'type': 'choice', 'instructions': 'Which team should handle this?',
                       'criteria': {'billing': 'money and invoices', 'technical': 'bugs and outages', 'sales': 'pricing and contracts'}},
        'refund': {'type': 'noul', 'instructions': 'Does the customer request a refund?'},
        'severity': {'type': 'score', 'instructions': 'How severe is this?',
                     'criteria': ['trivial', 'minor', 'moderate', 'serious', 'critical']},
    }
    result = {'passed': False, 'scope': protocol['scope'], 'state': state, 'questions': questions,
              'script_sha256': sha(Path(__file__)), 'protocol_sha256': sha(PROTOCOL),
              'custody_manifest_sha256': sha(MANIFEST), 'release_commit': manifest['release_commit'],
              'routes': {}, 'final_benchmark_access': False, 'quality_claim': False}
    try:
        for variable, value in protocol['environment_variables'].items():
            require(os.environ.get(variable) == value, f'Environment guard: {variable}')
        for variable in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'LAYA_SHA256_DIGESTS'):
            require(not os.environ.get(variable), f'Forbidden credential or override: {variable}')
        require(sys.executable == protocol['executable'], 'Unexpected interpreter')
        for package, version in protocol['packages'].items():
            require(importlib.metadata.version(package) == version, f'Dependency pin changed: {package}')
        for artifact in manifest['artifacts']:
            require(sha(Path(artifact['path'])) == artifact['actual_sha256'] == artifact['advertised_sha256'], 'Published artifact changed')
        require(sha(Path(manifest['historical_inventory']['path'])) == manifest['historical_inventory']['sha256'], 'Historical custody changed')
        require(sha(Path(manifest['release_tree']['path'])) == manifest['release_tree']['sha256'], 'Release tree changed')
        source_root = Path(manifest['wheel_source_root'])
        actual = {str(path.relative_to(source_root)) for path in (source_root / 'laya').rglob('*') if path.is_file() and '__pycache__' not in path.parts}
        require(actual == set(manifest['wheel_source_sha256']), 'Unexpected package source member')
        for name, digest in manifest['wheel_source_sha256'].items():
            require(sha(source_root / name) == digest, f'Published source changed: {name}')
        socket.socket.connect = offline
        socket.socket.connect_ex = offline
        socket.create_connection = offline
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(source_root))
        import laya
        import torch
        require(laya.__version__ == '0.3.27', 'Wrong runtime version')
        require(Path(laya.__file__).resolve() == (source_root / 'laya/__init__.py').resolve(), 'Wrong runtime source')
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        result['environment'] = {'python': sys.version, 'packages': protocol['packages'], 'laya_source': laya.__file__}
        for route in ('english', 'typed-decisions', 'multilingual'):
            fields = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
            require(int(fields['MemAvailable'].split()[0]) * 1024 >= 10 * 1024**3, 'Insufficient host RAM')
            entry = manifest['routes'][route]
            directory = ROOT / 'probe-artifacts' / route
            directory.mkdir(parents=True, exist_ok=False)
            expected = {}
            for relative, info in entry['files'].items():
                source = Path(info['resolved_path'])
                require(source.stat().st_size == info['bytes'] and sha(source) == info['sha256'], f'Payload changed: {route}/{relative}')
                target = directory / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if relative == 'model.safetensors':
                    target.symlink_to(source)
                else:
                    shutil.copyfile(source, target)
                expected[relative] = info['sha256']
            print(json.dumps({'route': route, 'stage': 'loading'}), flush=True)
            agent = laya.Agent(str(directory), device='cpu', backend='eager', expected_sha256=expected)
            prediction = agent.system_one(state, questions)
            result['routes'][route] = {'prediction': prediction, 'repository': entry['repository'], 'revision': entry['revision'],
                                       'device': str(agent.device), 'runtime_dtype': str(agent.dtype), 'amp_enabled': agent.amp_enabled,
                                       'live_parameter_count': sum(p.numel() for p in agent.model.parameters()),
                                       'tokenizer_config_sha256_before': expected['tokenizer/tokenizer_config.json'],
                                       'tokenizer_config_sha256_after': sha(directory / 'tokenizer/tokenizer_config.json')}
            with (directory / 'prediction-capture.json').open('x') as capture:
                json.dump(result['routes'][route], capture, indent=2, sort_keys=True, allow_nan=True)
                capture.write('\n')
            require(str(agent.device) == 'cpu', 'Non-CPU device')
            answers = prediction['answers']
            require(set(answers) == set(questions), 'Question keys differ')
            require(answers['department']['choice'] in questions['department']['criteria'], 'Invalid choice')
            noul = float(answers['refund']['noul'])
            score = float(answers['severity']['score'])
            require(math.isfinite(noul) and 0 <= noul <= 1, 'Invalid Noul')
            require(math.isfinite(score) and 0 <= score <= 4, 'Invalid Score')
            for name in ('department', 'severity'):
                probabilities = answers[name]['probabilities']
                keys = set(questions[name]['criteria']) if name == 'department' else {str(i) for i in range(5)}
                require(set(probabilities) == keys, 'Invalid probability keys')
                values = [float(value) for value in probabilities.values()]
                require(all(math.isfinite(value) and 0 <= value <= 1 for value in values), 'Invalid probabilities')
                require(abs(sum(values) - 1) <= len(values) * 0.00005 + 1e-6, 'Serialization normalization bound exceeded')
            del agent, prediction, answers
            gc.collect()
        result['passed'] = True
    except BaseException:
        result['error'] = traceback.format_exc()
        raise
    finally:
        with OUTPUT.open('x') as stream:
            json.dump(result, stream, sort_keys=True, indent=2, allow_nan=True)
            stream.write('\n')
        print(json.dumps({'receipt': str(OUTPUT), 'sha256': sha(OUTPUT), 'passed': result['passed']}), flush=True)


if __name__ == '__main__':
    main()
