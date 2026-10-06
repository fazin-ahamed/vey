#!/usr/bin/env python3
"""Run the QNATIVE-2 joint cross study on serialized, hash-verified Modal calls.

Stages: upload -> preflight (CPU custody/tokenizer census, no model) ->
smoke (T4 numerical liveness) -> train (single 10-epoch joint fit).
Every stage verifies the committed transport inventory before any source read;
training requires the fresh smoke receipt under the same transport bytes.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import modal

REPO = Path('/home/fazinahamed/Documents/vey-public')
HERE = REPO / 'research/endgame'
DATA = Path('/home/fazinahamed/Documents/vey-data/decisionmix')
ROOT = DATA / 'endgame/qnative-cross-v1'
TRANSPORT = HERE / 'qnative_cross_modal_transport.json'
PREFLIGHT_TIMEOUT = 900
SMOKE_TIMEOUT = 1200
TRAIN_TIMEOUT = 86400
PACKAGES = {'torch': '2.5.1+cu121', 'transformers': '5.17.0', 'numpy': '2.4.6',
            'safetensors': '0.8.0', 'huggingface_hub': '1.32.0', 'tokenizers': '0.23.2',
            'sentencepiece': '0.2.2', 'protobuf': '7.36.2'}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def transport():
    manifest = json.loads(TRANSPORT.read_text())
    require(manifest['schema'] == 'vey.qnative.cross.modal-transport.v1', 'Transport schema differs')
    for key, expected in (('gpu', 'T4'), ('preflight_timeout_seconds', PREFLIGHT_TIMEOUT),
                          ('smoke_timeout_seconds', SMOKE_TIMEOUT),
                          ('train_timeout_seconds', TRAIN_TIMEOUT), ('max_containers', 1),
                          ('retries', 0), ('packages', PACKAGES)):
        require(manifest[key] == expected, 'Transport execution bound differs: ' + key)
    return manifest


def check_transport(expected_sha256):
    require(digest(TRANSPORT) == expected_sha256, 'Transport identity differs')
    manifest = transport()
    for entry in (*manifest['source_files'], *manifest['input_files']):
        path = Path(entry['path'])
        require(path.stat().st_size == entry['bytes'] and digest(path) == entry['sha256'],
                'Remote immutable input/source differs: ' + str(path))
    for name, version in PACKAGES.items():
        require(importlib.metadata.version(name) == version, 'Remote dependency differs: ' + name)
    return manifest


volume_name = transport()['volume']
app = modal.App(transport()['app'])
volume = modal.Volume.from_name(volume_name, create_if_missing=True, version=2)
image = (
    modal.Image.debian_slim(python_version='3.11')
    .pip_install('torch==2.5.1+cu121', index_url='https://download.pytorch.org/whl/cu121')
    .pip_install(*(f'{name}=={version}' for name, version in PACKAGES.items() if name != 'torch'))
    .env({'HF_HOME': str(DATA / 'hf-home'), 'HF_HUB_OFFLINE': '1',
          'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1', 'TRANSFORMERS_OFFLINE': '1',
          'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1',
          'TOKENIZERS_PARALLELISM': 'false', 'USE_TF': '0',
          'QNATIVE_DATASET_ROOT': transport()['derived_dataset']['root'],
          'QNATIVE_TRANSPORT_MANIFEST': str(TRANSPORT),
          'QNATIVE_TRANSPORT_MANIFEST_SHA256': digest(TRANSPORT)})
)
for entry in transport()['source_files']:
    image = image.add_local_file(entry['path'], remote_path=entry['path'], copy=True)
image = image.add_local_file(TRANSPORT, remote_path=str(TRANSPORT), copy=True)


def run_stage(mode, expected_sha256):
    check_transport(expected_sha256)
    require(not (ROOT / mode).exists(), 'Refusing to overwrite a stage output: ' + mode)
    sys.path.insert(0, str(REPO))
    from research.endgame import neutral_qasper_cross_train as trainer
    require(Path(trainer.__file__).resolve() == HERE / 'neutral_qasper_cross_train.py',
            'Wrong trainer module')
    return trainer.run(ROOT, mode)


@app.function(image=image, volumes={str(DATA): volume}, cpu=2, memory=8192,
              timeout=PREFLIGHT_TIMEOUT, max_containers=1, retries=0)
def preflight(expected_sha256):
    require(not (ROOT / 'modal_preflight.json').exists(),
            'Refusing to overwrite a remote preflight receipt')
    directory = run_stage('preflight', expected_sha256)
    result = {'schema': 'vey.qnative.cross.modal-preflight.v1', 'status': 'PASS',
              'transport_sha256': expected_sha256, 'volume': volume_name,
              'model_loads': 0, 'GPU_allocated': False, 'sealed_phases_accessed': False,
              'metadata_sha256': digest(directory / 'metadata.json'),
              'performance_credit': False, 'quality_credit': False}
    (ROOT / 'modal_preflight.json').write_text(json.dumps(result, indent=2) + '\n')
    volume.commit()
    return result


@app.function(image=image, volumes={str(DATA): volume}, gpu='T4', cpu=2, memory=8192,
              timeout=SMOKE_TIMEOUT, max_containers=1, retries=0)
def smoke(expected_sha256):
    receipt_path = ROOT / 'modal_runtime_smoke.json'
    require(not receipt_path.exists(), 'Refusing to overwrite a remote smoke receipt')
    existing = json.loads((ROOT / 'modal_preflight.json').read_text())
    require(existing.get('status') == 'PASS' and existing.get('transport_sha256') == expected_sha256,
            'Fresh preflight PASS under this transport required')
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            'Exactly one Modal GPU required')
    result = {'schema': 'vey.qnative.cross.modal-runtime.v1', 'mode': 'smoke', 'status': 'STARTED',
              'transport_sha256': expected_sha256, 'volume': volume_name,
              'device': torch.cuda.get_device_name(0),
              'GPU_total_bytes': torch.cuda.get_device_properties(0).total_memory,
              'performance_credit': False, 'quality_credit': False}
    receipt_path.write_text(json.dumps(result, indent=2) + '\n')
    volume.commit()
    try:
        directory = run_stage('smoke', expected_sha256)
        metadata = json.loads((directory / 'metadata.json').read_text())
        result.update(status='COMPLETE', output=str(directory), metadata_sha256=digest(directory / 'metadata.json'),
                      metadata_status=metadata['status'],
                      own_GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated())
        proof = json.loads((directory / 'liveness.json').read_text())['smoke']
        require(proof['status'] == 'PASS', 'Remote smoke failed')
        require(proof['strict_restore']['strict'] is True
                and proof['strict_restore']['fingerprint'] == proof['after_step_sha256'],
                'Remote restore differs')
        require(proof['initial_model_sha256'] != proof['after_step_sha256'], 'Remote model unchanged')
        require(all(value > 0 for value in proof['named_gradients'].values())
                and all(value > 0 for value in proof['named_parameter_deltas'].values()),
                'Remote optimizer delta absent')
        result['numerical_smoke_verified'] = True
        return result
    except BaseException as error:
        result.update(status='FAILED', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        receipt_path.write_text(json.dumps(result, indent=2) + '\n')
        volume.commit()


@app.function(image=image, volumes={str(DATA): volume}, gpu='T4', cpu=2, memory=8192,
              timeout=TRAIN_TIMEOUT, max_containers=1, retries=0)
def train(expected_sha256):
    receipt_path = ROOT / 'modal_runtime_train.json'
    require(not receipt_path.exists(), 'Refusing to overwrite a remote train receipt')
    smoke_receipt = json.loads((ROOT / 'modal_runtime_smoke.json').read_text())
    require(smoke_receipt.get('status') == 'COMPLETE'
            and smoke_receipt.get('numerical_smoke_verified') is True
            and smoke_receipt.get('transport_sha256') == expected_sha256,
            'Fresh smoke PASS under this transport required')
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            'Exactly one Modal GPU required')
    result = {'schema': 'vey.qnative.cross.modal-runtime.v1', 'mode': 'train', 'status': 'STARTED',
              'transport_sha256': expected_sha256, 'volume': volume_name,
              'device': torch.cuda.get_device_name(0),
              'GPU_total_bytes': torch.cuda.get_device_properties(0).total_memory,
              'performance_credit': False, 'quality_credit': False}
    receipt_path.write_text(json.dumps(result, indent=2) + '\n')
    volume.commit()
    try:
        directory = run_stage('train', expected_sha256)
        check_transport(expected_sha256)
        metadata = json.loads((directory / 'metadata.json').read_text())
        require(metadata['status'] == 'train_complete', 'Incomplete remote training')
        result.update(status='COMPLETE', output=str(directory),
                      metadata_sha256=digest(directory / 'metadata.json'),
                      own_GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated())
        return result
    except BaseException as error:
        result.update(status='FAILED', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        receipt_path.write_text(json.dumps(result, indent=2) + '\n')
        volume.commit()


def committed(manifest):
    for entry in manifest['source_files']:
        path = Path(entry['path'])
        require(path.stat().st_size == entry['bytes'] and digest(path) == entry['sha256'],
                'Local source changed after transport freeze')
        if path.is_relative_to(REPO):
            blob = subprocess.check_output(['git', 'show', 'HEAD:' + path.relative_to(REPO).as_posix()], cwd=REPO)
            require(hashlib.sha256(blob).hexdigest() == entry['sha256'],
                    'Uncommitted transport source: ' + str(path))
    blob = subprocess.check_output(['git', 'show', 'HEAD:' + TRANSPORT.relative_to(REPO).as_posix()], cwd=REPO)
    require(hashlib.sha256(blob).hexdigest() == digest(TRANSPORT), 'Uncommitted transport inventory')


@app.local_entrypoint()
def main(stage: str = 'preflight'):
    require(stage in ('upload', 'preflight', 'smoke', 'train'), 'Unknown stage')
    manifest = transport()
    if stage != 'upload':
        committed(manifest)
    if stage == 'upload':
        # Explicit one-time input transfer; existing remote inputs refused, never overwritten.
        with volume.batch_upload(force=False) as batch:
            for entry in manifest['input_files']:
                path = Path(entry['path'])
                require(path.stat().st_size == entry['bytes'] and digest(path) == entry['sha256'],
                        'Local input changed before upload: ' + str(path))
                batch.put_local_file(path, str(path.relative_to(DATA)))
        print(json.dumps({'uploaded': len(manifest['input_files'])}))
        return
    expected = digest(TRANSPORT)
    if stage == 'preflight':
        result = preflight.remote(expected)
    elif stage == 'smoke':
        raw = b''.join(volume.read_file('endgame/qnative-cross-v1/modal_preflight.json'))
        receipt = json.loads(raw)
        require(receipt.get('status') == 'PASS' and receipt.get('transport_sha256') == expected,
                'Remote preflight PASS under this transport required')
        result = smoke.remote(expected)
    else:
        for name in ('modal_preflight.json', 'modal_runtime_smoke.json'):
            raw = b''.join(volume.read_file('endgame/qnative-cross-v1/' + name))
            receipt = json.loads(raw)
            require(receipt.get('transport_sha256') == expected, 'Transport drift in ' + name)
            require(receipt.get('status') in ('PASS', 'COMPLETE'), 'Prior stage incomplete: ' + name)
        result = train.remote(expected)
    print(json.dumps(result, sort_keys=True), flush=True)
