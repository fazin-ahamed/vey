#!/usr/bin/env python3
"""Run the frozen NATIVE-2 trainer on serialized, hash-verified Modal T4 calls."""
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
ROOT = DATA / 'endgame/native-field-v1'
TRANSPORT = HERE / 'native_arch_modal_transport.json'
RECOVERY = HERE / 'native_arch_modal_timeout_recovery_protocol.json'
VOLUME_NAME = 'vey-native-arch-modal-v2'
TRAIN_TIMEOUT = 43200
SMOKE_TIMEOUT = 600
ARMS = ('cross', 'dual', 'pages')
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
    require(manifest['schema'] == 'vey.native-arch.modal-transport.v1', 'Transport schema differs')
    require(manifest['volume'] == VOLUME_NAME, 'Transport volume differs')
    return manifest


def recovery(manifest, retained=False):
    pin = manifest['timeout_recovery_protocol']
    require(pin['path'] == str(RECOVERY) and RECOVERY.stat().st_size == pin['bytes']
            and digest(RECOVERY) == pin['sha256'], 'Recovery protocol identity differs')
    cfg = json.loads(RECOVERY.read_text())
    require(cfg['schema'] == 'vey.native-arch.modal-timeout-recovery-protocol.v1',
            'Recovery protocol schema differs')
    require(cfg['volume'] == VOLUME_NAME and cfg['original_volume'] == 'vey-native-arch-modal-v1'
            and cfg['app'] == 'vey-native-arch-v2', 'Recovery isolation differs')
    require(tuple(cfg['arms']) == ARMS and cfg['gpu'] == 'T4'
            and cfg['train_timeout_seconds_per_arm'] == TRAIN_TIMEOUT
            and cfg['smoke_timeout_seconds_per_arm'] == SMOKE_TIMEOUT
            and cfg['preflight_timeout_seconds'] == SMOKE_TIMEOUT
            and cfg['max_containers'] == 1 and cfg['retries'] == 0,
            'Recovery execution bounds differ')
    require(cfg['resume_allowed'] is False and cfg['input_files_changed'] is False
            and cfg['quality_recipe_changed'] is False
            and cfg['v1_evidence_authorizes_training'] is False,
            'Recovery must remain a fresh unchanged-recipe restart')
    failure_pin = cfg['original_failure_manifest']
    failure_path = Path(failure_pin['path'])
    require(failure_path.stat().st_size == failure_pin['bytes']
            and digest(failure_path) == failure_pin['sha256'], 'Original failure identity differs')
    failure = json.loads(failure_path.read_text())
    require(failure['status'] == 'EXECUTION_TIMEOUT_NO_QUALITY_RESULT'
            and failure['original_protocol'] == cfg['original_protocol']
            and failure['source_transport'] == cfg['original_transport'],
            'Recovery no longer binds the original timeout snapshots')
    if retained:
        for key in ('original_protocol', 'original_transport'):
            original = cfg[key]
            path = Path(original['path'])
            require(path.stat().st_size == original['bytes'] and digest(path) == original['sha256'],
                    'Original retained snapshot differs: ' + key)
        original = json.loads(Path(cfg['original_protocol']['path']).read_text())
        current = json.loads((HERE / 'native_arch_protocol.json').read_text())
        require(current.pop('modal_timeout_recovery_amendment', None),
                'Operational timeout recovery amendment absent')
        current['resources']['modal_train_timeout_seconds_per_arm'] = original['resources']['modal_train_timeout_seconds_per_arm']
        require(current == original, 'Recovery changed the original scientific protocol')
    return cfg


def require_preflight(receipt, expected_sha256, recovery_sha256):
    require(receipt.get('schema') == 'vey.native-arch.modal-preflight.v1'
            and receipt.get('status') == 'PASS'
            and receipt.get('transport_sha256') == expected_sha256
            and receipt.get('recovery_protocol_sha256') == recovery_sha256
            and receipt.get('volume') == VOLUME_NAME
            and receipt.get('stock_loading_verified_on_CPU') is True
            and receipt.get('GPU_allocated') is False
            and receipt.get('sealed_phases_accessed') is False,
            'Fresh v2 preflight must pass under the current transport and recovery')


def require_smoke(receipt, arm, expected_sha256, recovery_sha256):
    require(receipt.get('schema') == 'vey.native-arch.modal-runtime.v1'
            and receipt.get('arm') == arm and receipt.get('mode') == 'smoke'
            and receipt.get('status') == 'COMPLETE'
            and receipt.get('numerical_smoke_verified') is True
            and receipt.get('transport_sha256') == expected_sha256
            and receipt.get('recovery_protocol_sha256') == recovery_sha256
            and receipt.get('volume') == VOLUME_NAME
            and receipt.get('output') == str(ROOT / (arm + '-smoke')),
            'Every actual v2 Modal liveness smoke must pass under the current transport and recovery')


def check_prerequisites(mode, expected_sha256):
    recovery_sha256 = digest(RECOVERY)
    require_preflight(json.loads((ROOT / 'modal_preflight.json').read_text()),
                      expected_sha256, recovery_sha256)
    if mode == 'train':
        for arm in ARMS:
            receipt = json.loads((ROOT / ('modal_runtime_' + arm + '_smoke.json')).read_text())
            require_smoke(receipt, arm, expected_sha256, recovery_sha256)
            require(digest(ROOT / (arm + '-smoke') / 'metadata.json') == receipt['metadata_sha256'],
                    'Required v2 smoke artifacts differ: ' + arm)


app = modal.App('vey-native-arch-v2')
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)
image = (
    modal.Image.debian_slim(python_version='3.11')
    .pip_install('torch==2.5.1+cu121', index_url='https://download.pytorch.org/whl/cu121')
    .pip_install(*(f'{name}=={version}' for name, version in PACKAGES.items() if name != 'torch'))
    .env({'HF_HOME': str(DATA / 'hf-home'), 'HF_HUB_OFFLINE': '1',
          'HF_HUB_DISABLE_IMPLICIT_TOKEN': '1', 'TRANSFORMERS_OFFLINE': '1',
          'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1',
          'TOKENIZERS_PARALLELISM': 'false', 'USE_TF': '0'})
)
for entry in transport()['source_files']:
    image = image.add_local_file(entry['path'], remote_path=entry['path'], copy=True)
image = image.add_local_file(TRANSPORT, remote_path=str(TRANSPORT), copy=True)


def check_transport(expected_sha256):
    require(digest(TRANSPORT) == expected_sha256, 'Transport identity differs')
    manifest = transport()
    for entry in (*manifest['source_files'], *manifest['input_files']):
        path = Path(entry['path'])
        require(path.stat().st_size == entry['bytes'] and digest(path) == entry['sha256'],
                'Remote immutable input/source differs: ' + str(path))
    recovery(manifest)
    for name, version in PACKAGES.items():
        require(importlib.metadata.version(name) == version, 'Remote dependency differs: ' + name)
    return manifest


@app.function(image=image, volumes={str(DATA): volume}, cpu=2, memory=8192,
              timeout=SMOKE_TIMEOUT, max_containers=1, retries=0)
def preflight(expected_sha256):
    check_transport(expected_sha256)
    require(not (ROOT / 'modal_preflight.json').exists(),
            'Refusing to overwrite a remote preflight receipt')
    sys.path.insert(0, str(REPO))
    from research.endgame import native_field_data as data
    from research.endgame.native_arch_model import canonical_modules, load_arch_encoder
    manifest, phases = data.load_dataset(ROOT)
    require(Path(data.__file__).resolve() == HERE / 'native_field_data.py', 'Wrong data module')
    canonical_modules()
    tokenizer, encoder, custody = load_arch_encoder(
        data.protocol(), {endpoint: manifest['catalogues'][endpoint]
                          for endpoint in manifest['eligible_endpoints']}, 'cpu')
    require(tokenizer.is_fast and tokenizer('Offline native architecture preflight')['input_ids'],
            'CPU tokenizer conversion failed')
    del tokenizer, encoder
    result = {'schema': 'vey.native-arch.modal-preflight.v1', 'status': 'PASS',
              'transport_sha256': expected_sha256, 'GPU_allocated': False,
              'volume': VOLUME_NAME, 'recovery_protocol_sha256': digest(RECOVERY),
              'model_forwards': 0, 'sealed_phases_accessed': False,
              'dataset_manifest_sha256': digest(ROOT / 'dataset_manifest.json'),
              'phase_counts': {phase: len(records) for phase, records in phases.items()},
              'eligible_endpoints': manifest['eligible_endpoints'],
              'stock_loading_verified_on_CPU': True,
              'stock_encoder_initial_sha256': custody['encoder_initial_sha256'],
              'packages': PACKAGES, 'python': sys.version}
    (ROOT / 'modal_preflight.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    volume.commit()
    return result


@app.function(image=image, volumes={str(DATA): volume}, gpu='T4', cpu=2, memory=8192,
              timeout=TRAIN_TIMEOUT, max_containers=1, retries=0)
def run_arm(arm, mode, expected_sha256):
    require(arm in ARMS and mode in ('smoke', 'train'), 'Unregistered arm/mode')
    check_transport(expected_sha256)
    check_prerequisites(mode, expected_sha256)
    require(not (ROOT / (arm if mode == 'train' else arm + '-smoke')).exists(),
            'Recovery requires a fresh arm, never a partial-checkpoint resume')
    sys.path.insert(0, str(REPO))
    import torch
    from research.endgame import native_arch_train as trainer
    require(Path(trainer.__file__).resolve() == HERE / 'native_arch_train.py', 'Wrong trainer module')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Exactly one Modal GPU required')
    name = torch.cuda.get_device_name(0)
    require('T4' in name, 'Unregistered remote GPU: ' + name)
    require(trainer.arch_protocol()['resources']['gpu'] == 'Serialized Modal T4 16GB; never local GPU',
            'Modal environment amendment absent')
    props = torch.cuda.get_device_properties(0)
    receipt_path = ROOT / ('modal_runtime_' + arm + '_' + mode + '.json')
    require(not receipt_path.exists(), 'Refusing to overwrite a remote execution receipt')
    result = {'schema': 'vey.native-arch.modal-runtime.v1', 'arm': arm, 'mode': mode,
              'status': 'STARTED', 'transport_sha256': expected_sha256,
              'volume': VOLUME_NAME, 'recovery_protocol_sha256': digest(RECOVERY),
              'device': name, 'GPU_total_bytes': props.total_memory,
              'capability': list(torch.cuda.get_device_capability(0)),
              'torch': torch.__version__, 'CUDA_runtime': torch.version.cuda,
              'python': sys.version, 'packages': PACKAGES,
              'training_recomputation_in_inference_counters': False,
              'performance_credit': False, 'quality_credit': False}
    receipt_path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    volume.commit()
    try:
        directory = trainer.run(ROOT, arm, mode, 'cuda')
        check_transport(expected_sha256)
        metadata_path = directory / 'metadata.json'
        metadata = json.loads(metadata_path.read_text())
        expected_status = 'smoke_complete' if mode == 'smoke' else 'train_complete'
        require(metadata['status'] == expected_status, 'Incomplete remote arm')
        result.update(status='COMPLETE', output=str(directory),
                      metadata_sha256=digest(metadata_path),
                      own_GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated())
        if mode == 'smoke':
            proof = json.loads((directory / 'liveness.json').read_text())['smoke']
            require(proof['strict_restore']['max_abs_difference'] == 0, 'Remote restore differs')
            require(proof['encoder_before_sha256'] != proof['encoder_after_sha256'], 'Remote encoder unchanged')
            require(all(item['nonzero'] > 0 for item in proof['parameter_deltas'].values()), 'Remote optimizer delta absent')
            result['numerical_smoke_verified'] = True
        return result
    except BaseException as error:
        result.update(status='FAILED', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        receipt_path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        volume.commit()


def committed(manifest):
    recovery(manifest, retained=True)
    for entry in manifest['source_files']:
        path = Path(entry['path'])
        require(path.stat().st_size == entry['bytes'] and digest(path) == entry['sha256'],
                'Local source changed after transport freeze')
        if path.is_relative_to(REPO):
            blob = subprocess.check_output(['git', 'show', 'HEAD:' + path.relative_to(REPO).as_posix()], cwd=REPO)
            require(hashlib.sha256(blob).hexdigest() == entry['sha256'], 'Uncommitted transport source')
    blob = subprocess.check_output(['git', 'show', 'HEAD:' + TRANSPORT.relative_to(REPO).as_posix()], cwd=REPO)
    require(hashlib.sha256(blob).hexdigest() == digest(TRANSPORT), 'Uncommitted transport manifest')


@app.local_entrypoint()
def main(stage: str = 'preflight'):
    require(stage in ('preflight', 'smoke', 'train'), 'Stage must be preflight, smoke or train')
    manifest = transport()
    committed(manifest)
    expected = digest(TRANSPORT)
    if stage == 'preflight':
        # Inputs are transferred explicitly before preflight; never overwrite them.
        for entry in manifest['input_files']:
            require(digest(entry['path']) == entry['sha256'], 'Local input changed before preflight')
        result = preflight.remote(expected)
        print(json.dumps(result, sort_keys=True), flush=True)
        return
    recovery_sha256 = digest(RECOVERY)
    raw = b''.join(volume.read_file('endgame/native-field-v1/modal_preflight.json'))
    require_preflight(json.loads(raw), expected, recovery_sha256)
    if stage == 'train':
        for arm in ARMS:
            path = 'endgame/native-field-v1/modal_runtime_' + arm + '_smoke.json'
            raw = b''.join(volume.read_file(path))
            require_smoke(json.loads(raw), arm, expected, recovery_sha256)
    for arm in ARMS:
        result = run_arm.with_options(timeout=SMOKE_TIMEOUT if stage == 'smoke' else TRAIN_TIMEOUT).remote(arm, stage, expected)
        print(json.dumps(result, sort_keys=True), flush=True)
