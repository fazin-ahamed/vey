"""Run the frozen CBF-8 two-by-two field-support experiment once."""
import os

for _name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import argparse
import hashlib
import io
import json
import platform
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from audit import put
from relation_pretrained_run import (
    gpu_serialization, guard_resources, matrices_for, tensor_hash, train_linear_head,
)
from schema_support_encoder import (
    _json_bytes, _native_sentencepiece_parity, _protocol,
    _rows, _tokenize_rows, _cache_one,
    require_meaning_audit, verify_preflight, verify_preparation_sources,
)

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / 'schema_support_protocol.json'
SOURCE_PROTOCOL = HERE / 'schema_relation_protocol.json'
ARM_ORDER = ('literal_generic', 'literal_defined', 'semantic_generic', 'semantic_defined')
ELIGIBLE_ORDER = ARM_ORDER[1:]


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def _source_refs(manifest):
    source = Path(manifest['source_root'])
    refs = manifest['baseline_references']['files']
    return {role: source / row['file'] for role, row in refs.items()}


def _root_state(cfg):
    root = Path(cfg['output_root'])
    manifest_path = root / 'corpus_manifest.json'
    manifest = load(manifest_path)
    assert manifest['experiment'] == 'CBF8'
    assert manifest['protocol_sha256'] == sha(PROTOCOL)
    assert manifest['source_root'] == cfg['source_root']
    verify_preparation_sources(manifest)
    for name, expected in manifest['files'].items():
        assert sha(root / name) == expected, f'Prepared CBF8 artifact hash mismatch: {name}'
    if (root / 'selection.json').exists() or (root / 'results.json').exists():
        raise RuntimeError('Refusing to overwrite CBF8 selection or results.')
    for arm in ARM_ORDER:
        if (root / f'development_{arm}').exists():
            raise RuntimeError(f'Refusing existing development stage directory: development_{arm}')
    if (root / 'final').exists():
        raise RuntimeError('Refusing existing CBF8 final stage directory.')
    _, meaning_sha = require_meaning_audit(root, manifest)
    preflight = verify_preflight(root, manifest, meaning_sha, cfg)
    return root, manifest, meaning_sha, preflight


def assert_gate_equivalence(cfg):
    old = load(SOURCE_PROTOCOL)['gates']
    keys = {
        'G0_literal_relation': 'G0_literal_relation',
        'G1_axis': 'G1_axis',
        'G2_sign_given_axis': 'G2_sign_given_axis',
        'G3_joint_atom': 'G3_joint_atom',
        'G4_alias_decision': 'G4_alias_decision',
        'G5_alias_composition': 'G5_alias_composition',
        'G6_correct_new': 'G6_correct_new',
        'G7_exact': 'G7_exact',
    }
    assert {key: cfg['gates'][key] for key in keys} == {key: old[value] for key, value in keys.items()}, \
        'CBF8 gate thresholds differ from the unchanged CBF6 evaluator gates.'


def _training_labels(atoms, schema):
    return torch.tensor([atom['sign'] + 1 if index == atom['axis'] else 1
                         for atom in atoms for index, _ in enumerate(schema)], dtype=torch.long)


def _feature_lookup(input_path, features, schema):
    evidence = load(input_path)
    expected = [(text, field) for text in sorted({row['criterion'] for row in evidence['pairs']})
                for field in schema]
    pairs = [(row['criterion'], row['field']) for row in evidence['pairs']]
    assert pairs == expected, 'Feature evidence rows are not sorted unique text×field pairs.'
    assert all(set(row) == {'criterion', 'field', 'text_pair', 'input_ids', 'token_type_ids',
                             'attention_mask'} for row in evidence['pairs']), \
        'Feature evidence includes missing IDs or forbidden extra data.'
    assert features.shape == (len(pairs), 384) and features.dtype == np.float32
    return evidence, {(text, field): index for index, (text, field) in enumerate(pairs)}


def _rows_for_texts(texts, features, lookup, schema):
    return np.asarray(features[[lookup[(text, field)] for text in texts for field in schema]], dtype=np.float32)


def _make_matrices(module, values, mean, std, texts):
    return matrices_for(module, values, mean, std, texts)


def _load_baseline(root, cfg, manifest, g0, dev, comps):
    refs = _source_refs(manifest)
    source = Path(manifest['source_root'])
    reference_files = manifest['baseline_references']['files']
    for role, path in refs.items():
        assert sha(path) == reference_files[role]['sha256'], f'CBF7 baseline reference changed: {role}'
    input_path = refs['inputs']
    evidence = load(input_path)
    schema = tuple(cfg['schema'])
    pairs = [(row['criterion'], row['field']) for row in evidence['pairs']]
    assert pairs == [(text, field) for text in sorted({q for q, _ in pairs}) for field in schema]
    baseline_features = np.load(refs['features'], mmap_mode='r', allow_pickle=False)
    assert baseline_features.shape == (len(pairs), 384) and baseline_features.dtype == np.float32
    baseline_lookup = {pair: index for index, pair in enumerate(pairs)}
    score_texts = sorted(set(__import__('schema_relation_evaluate').semantic_contexts(dev, comps)) |
                         {row['text'] for row in g0})
    assert all((text, field) in baseline_lookup for text in score_texts for field in schema), \
        'CBF7 source cache does not cover every development/G0 generic pair.'
    score_values = _rows_for_texts(score_texts, baseline_features, baseline_lookup, schema)
    normalizer = np.load(refs['normalizer'], allow_pickle=False)
    mean, std = normalizer['mean'], normalizer['std']
    assert mean.shape == std.shape == (384,)
    payload = torch.load(refs['checkpoint'], map_location='cpu', weights_only=True)
    if isinstance(payload, dict) and 'head' in payload:
        state = payload['head']
    else:
        state = payload
    module = nn.Linear(384, 3, dtype=torch.float32)
    module.load_state_dict(state, strict=True)
    module.eval()
    matrices = _make_matrices(module, score_values, mean, std, score_texts)
    training = load(refs['training'])
    lineage = load(refs['lineage'])
    readout = dict(
        kind='exact-CBF7-literal_generic-reuse', source_root=str(source),
        source_files={role: dict(file=reference_files[role]['file'], sha256=reference_files[role]['sha256'])
                      for role in reference_files},
        feature_reused=True, checkpoint_reused=True, normalizer_reused=True,
        no_retraining=True, no_reencoding_for_development=True,
        baseline_encoder_parameters_sha256=training['frozen_encoder_before_sha256'],
        model_revision=lineage['revision'], model_weight_sha256=lineage['weight_sha256'],
        floating_parameters=lineage['floating_parameters'], feature_sha256=sha(refs['features']),
        inputs_sha256=sha(input_path), checkpoint_sha256=sha(refs['checkpoint']),
        training_sha256=sha(refs['training']), normalizer_npz_sha256=sha(refs['normalizer']),
        lineage_sha256=sha(refs['lineage']),
    )
    return matrices, readout, module, (mean, std)


def _normalizer_payload(mean, std):
    buffer = io.BytesIO()
    np.savez(buffer, mean=mean, std=std)
    return buffer.getvalue()


def _arm_data(arm, literal_training, semantic_training):
    semantic = arm.startswith('semantic_')
    atoms = literal_training + semantic_training if semantic else literal_training
    format_name = arm.split('_', 1)[1]
    return atoms, format_name


def _train_arm(arm, cfg, root, literal_training, literal_validation, semantic_training,
               feature, lookup, encoder_lineage):
    schema = tuple(cfg['schema'])
    atoms, fmt = _arm_data(arm, literal_training, semantic_training)
    train_texts = [row['text'] for row in atoms]
    val_texts = [row['text'] for row in literal_validation]
    train_h = _rows_for_texts(train_texts, feature, lookup, schema)
    val_h = _rows_for_texts(val_texts, feature, lookup, schema)
    mean = train_h.mean(axis=0)
    std = np.maximum(train_h.std(axis=0), .01)
    assert mean.shape == std.shape == (384,) and np.isfinite(mean).all() and np.isfinite(std).all()
    normalizer_bytes = _normalizer_payload(mean, std)
    normalizer_name = f'normalizer_{arm}.npz'
    checkpoint_name = f'{arm}_checkpoint.pt'
    normalizer_sha = sha_bytes(normalizer_bytes)
    normalizer_mean_sha = sha_bytes(np.ascontiguousarray(mean).tobytes())
    normalizer_std_sha = sha_bytes(np.ascontiguousarray(std).tobytes())
    normalizer_record = dict(arm_id=arm, file=normalizer_name, npz_sha256=normalizer_sha,
                             mean_sha256=normalizer_mean_sha, std_sha256=normalizer_std_sha,
                             shape=[384], dtype=str(mean.dtype), floor=.01, train_only=True,
                             train_atom_ids=[row['id'] for row in atoms],
                             train_pair_order='original literal cohort order followed by original semantic cohort order; criterion×schema pairs in row order')
    X = torch.tensor((train_h - mean) / std, dtype=torch.float32)
    V = torch.tensor((val_h - mean) / std, dtype=torch.float32)
    labels = _training_labels(atoms, schema)
    val_labels = _training_labels(literal_validation, schema)
    module, details = train_linear_head(X, labels, V, val_labels, cfg['training'], normalizer_record)
    head_state = {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}
    checkpoint_buffer = io.BytesIO()
    torch.save(head_state, checkpoint_buffer)
    checkpoint_bytes = checkpoint_buffer.getvalue()
    gradient_l1 = float(details['first_gradient_L1'])
    details.update(dict(
        arm_id=arm, eligible=True, data_condition='literal32+semantic64' if atoms is not literal_training else 'literal32',
        format=fmt, model_repo=cfg['model']['repo'], model_revision=cfg['model']['revision'],
        model_weight_sha256=cfg['model']['weight_sha256'], floating_parameters=cfg['model']['floating_parameters'],
        feature='raw frozen final-layer CLS from base_model', feature_sha256=sha(root / f'features_{fmt}.npy'),
        evidence_sha256=sha(root / f'inputs_{fmt}.json'),
        literal_training_atoms=32, semantic_training_atoms=len(atoms) - 32,
        validation_atoms=len(literal_validation), semantic_validation_used=False,
        development_or_G0_or_final_used_for_training=False,
        normalizer_mean_sha256=normalizer_mean_sha, normalizer_std_sha256=normalizer_std_sha,
        normalizer_npz_sha256=normalizer_sha, normalizer_file=normalizer_name,
        checkpoint_file=checkpoint_name, checkpoint_sha256=sha_bytes(checkpoint_bytes),
        encoder_parameters_before_sha256=encoder_lineage['encoder_parameters_before_sha256'],
        encoder_parameters_after_sha256=encoder_lineage['encoder_parameters_before_sha256'],
        encoder_parameter_gradient_count=0,
        gradient_proof=dict(head_l1=gradient_l1, encoder_l1=0.0, frozen_gradients=True,
                            optimizer_parameters='head_only'),
        optimizer='AdamW', learning_rate=cfg['training']['lr'],
        weight_decay=cfg['training']['weight_decay'], epochs=cfg['training']['epochs'], seed=cfg['training']['seed'],
    ))
    module.eval()
    return dict(module=module, state=head_state, mean=mean, std=std, atoms=atoms, fmt=fmt,
                details=details, normalizer_bytes=normalizer_bytes,
                checkpoint_bytes=checkpoint_bytes, normalizer_name=normalizer_name,
                checkpoint_name=checkpoint_name)


def _semantic_validation(root, arm, module, mean, std, features, lookup, atoms, schema):
    from schema_relation_evaluate import decode
    texts = [atom['text'] for atom in atoms]
    values = _rows_for_texts(texts, features, lookup, schema)
    with torch.no_grad():
        logits = module(torch.tensor((values - mean) / std, dtype=torch.float32))
        probabilities = logits.softmax(-1).cpu().numpy().reshape(len(atoms), 4, 3)
    cases = []
    for atom, matrix in zip(atoms, probabilities):
        resolution = decode(matrix)
        axis_correct = int(resolution is not None and resolution['axis'] == atom['axis'])
        joint_correct = int(axis_correct and resolution['sign'] == atom['sign'])
        cases.append(dict(id=atom['id'], pair_id=atom['pair_id'], text=atom['text'],
                          gold_axis=atom['axis'], gold_sign=atom['sign'],
                          relation_matrix=matrix.tolist(), resolution=resolution,
                          axis_correct=axis_correct, joint_correct=joint_correct))
    diagnostic = dict(arm=arm, usage='diagnostic_only_never_selection', n=len(cases), atoms=cases,
                      summary=dict(axis_correct=sum(row['axis_correct'] for row in cases),
                                   axis_accuracy=float(np.mean([row['axis_correct'] for row in cases])),
                                   joint_correct=sum(row['joint_correct'] for row in cases),
                                   joint_accuracy=float(np.mean([row['joint_correct'] for row in cases])),
                                   checkpoint_or_arm_selection_used=False))
    put(root, f'semantic_validation_{arm}.json', _json_bytes(diagnostic))
    return diagnostic


def _development_pair_records(root, arms, development):
    row_by_arm = {}
    for arm in arms:
        stage = root / f'development_{arm}'
        matrices = {row['text']: row['resolution']
                    for row in load(stage / 'relation_matrices.json')}
        decision_rows = [json.loads(line) for line in (stage / 'decisions.jsonl').read_text().splitlines()]
        per_atom = {}
        held_state_ids = None
        for atom in development:
            resolution = matrices[atom['text']]
            axis_correct = int(resolution is not None and resolution['axis'] == atom['axis'])
            joint_correct = int(axis_correct and resolution['sign'] == atom['sign'])
            cases = [row for row in decision_rows if row['criterion_id'] == atom['id'] and
                     row['stratum'] == 'alias' and row['source_split'] == 'unseen_wording']
            ids = sorted(row['scenario_id'] for row in cases)
            if held_state_ids is None:
                held_state_ids = ids
            assert ids == held_state_ids and ids, 'Development aliases must share the same held state rows.'
            per_atom[atom['id']] = dict(
                axis_correct=axis_correct, joint_correct=joint_correct,
                held_state_top1_mean=float(np.mean([row['top1'] for row in cases])),
                held_states={row['scenario_id']: int(row['top1']) for row in cases},
            )
        row_by_arm[arm] = dict(per_atom=per_atom, held_state_ids=held_state_ids)
    pairs = defaultdict(list)
    for atom in development:
        pair_id = atom.get('pair_id') or f'axis-{atom["axis"]}-variant-{atom.get("variant", "unknown")}'
        pairs[pair_id].append(atom)
    assert len(pairs) == 16 and all(len(group) == 2 for group in pairs.values())
    for group in pairs.values():
        assert group[0]['axis'] == group[1]['axis'] and group[0]['sign'] == -group[1]['sign']
    rows = []
    for pair_id, group in sorted(pairs.items()):
        arm_metrics = {}
        for arm in arms:
            values = row_by_arm[arm]['per_atom']
            arm_metrics[arm] = {
                'axis_correct': float(np.mean([values[atom['id']]['axis_correct'] for atom in group])),
                'joint_correct': float(np.mean([values[atom['id']]['joint_correct'] for atom in group])),
                'held_state_top1_mean': float(np.mean([values[atom['id']]['held_state_top1_mean'] for atom in group])),
            }
        rows.append(dict(pair_id=pair_id, atom_ids=[atom['id'] for atom in group],
                         arm_metrics=arm_metrics))
    atom_rows = [dict(id=atom['id'], pair_id=atom.get('pair_id'), axis=atom['axis'], sign=atom['sign'],
                      arms={arm: row_by_arm[arm]['per_atom'][atom['id']] for arm in arms})
                 for atom in development]
    atom_state_rows = []
    for atom in development:
        for state_id in row_by_arm[arms[0]]['held_state_ids']:
            atom_state_rows.append(dict(
                criterion_id=atom['id'], pair_id=atom.get('pair_id'), scenario_id=state_id,
                arms={arm: row_by_arm[arm]['per_atom'][atom['id']]['held_states'][state_id] for arm in arms}))
    return rows, atom_rows, atom_state_rows


def _discordance(left, right):
    left = np.asarray(left, dtype=np.int8)
    right = np.asarray(right, dtype=np.int8)
    assert left.shape == right.shape
    return dict(n=int(left.size), left_only=int(np.sum((left == 1) & (right == 0))),
                right_only=int(np.sum((left == 0) & (right == 1))),
                concordant=int(np.sum(left == right)))


def _paired_controls(root, outcomes, development):
    arms = list(ARM_ORDER)
    pair_rows, atom_rows, atom_state_rows = _development_pair_records(root, arms, development)
    pair_values = {row['pair_id']: row['arm_metrics'] for row in pair_rows}
    metrics = ('axis_correct', 'joint_correct', 'held_state_top1_mean')
    simple_specs = {
        'data_at_generic': ('semantic_generic', 'literal_generic'),
        'data_at_defined': ('semantic_defined', 'literal_defined'),
        'definition_at_literal': ('literal_defined', 'literal_generic'),
        'definition_at_semantic': ('semantic_defined', 'semantic_generic'),
    }
    pair_ids = [row['pair_id'] for row in pair_rows]
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(pair_ids), size=(10000, len(pair_ids)))
    simple_values = {}
    simple = {}
    for name, (left_arm, right_arm) in simple_specs.items():
        simple_values[name] = {
            metric: np.asarray([pair_values[pair][left_arm][metric] - pair_values[pair][right_arm][metric]
                                for pair in pair_ids], dtype=np.float64)
            for metric in metrics
        }
        discordance = {}
        for metric in ('axis_correct', 'joint_correct'):
            left = [a['arms'][left_arm][metric] for a in atom_rows]
            right = [a['arms'][right_arm][metric] for a in atom_rows]
            discordance[metric] = _discordance(left, right)
        state_left = [row['arms'][left_arm] for row in atom_state_rows]
        state_right = [row['arms'][right_arm] for row in atom_state_rows]
        discordance['atom_by_held_state_top1'] = _discordance(state_left, state_right)
        simple[name] = dict(left_arm=left_arm, right_arm=right_arm,
                            raw_pair_effects={metric: simple_values[name][metric].tolist() for metric in metrics},
                            mean_effect={metric: float(simple_values[name][metric].mean()) for metric in metrics},
                            CI95={metric: np.quantile(simple_values[name][metric][draws].mean(axis=1),
                                                      [.025, .975]).tolist() for metric in metrics},
                            matched_discordance=discordance, bootstrap_pairs=10000, bootstrap_seed=0)
    main_arrays = {
        'data': {metric: (simple_values['data_at_generic'][metric] + simple_values['data_at_defined'][metric]) / 2
                 for metric in metrics},
        'definition': {metric: (simple_values['definition_at_literal'][metric] +
                                simple_values['definition_at_semantic'][metric]) / 2
                       for metric in metrics},
        'interaction': {metric: simple_values['data_at_defined'][metric] - simple_values['data_at_generic'][metric]
                        for metric in metrics},
    }
    main_effects = {}
    for effect, arrays in main_arrays.items():
        main_effects[effect] = dict(
            raw_pair_effects={metric: arrays[metric].tolist() for metric in metrics},
            mean_effect={metric: float(arrays[metric].mean()) for metric in metrics},
            CI95={metric: np.quantile(arrays[metric][draws].mean(axis=1), [.025, .975]).tolist()
                  for metric in metrics},
            bootstrap_pairs=10000, bootstrap_seed=0,
        )
        if effect == 'interaction':
            main_effects[effect]['formula'] = '(semantic_defined-literal_defined)-(semantic_generic-literal_generic)'
        else:
            main_effects[effect]['formula'] = 'average of the two corresponding simple effects'
    controls = dict(
        design='paired 2x2 format×training-data intervention; baseline is exact CBF7 reuse',
        metrics=list(metrics), reversal_clusters=16, atom_cases=32,
        held_states_per_atom=len(atom_rows[0]['arms']['literal_generic']['held_states']),
        atom_by_held_state_rows=len(atom_state_rows),
        paired_bootstrap=dict(repetitions=10000, seed=0, common_index_draws=True,
                              index_matrix_sha256=sha_bytes(draws.astype(np.int64, copy=False).tobytes()),
                              gate_use='descriptive only; development gate selection uses declared point metrics'),
        simple_effects=simple, main_effects=main_effects,
        raw_pair_rows=pair_rows, raw_atom_rows=atom_rows, raw_atom_by_held_state_rows=atom_state_rows,
        raw_arm_point_metrics={arm: dict(
            axis_accuracy=outcomes[arm]['atoms']['axis_accuracy'],
            joint_accuracy=outcomes[arm]['atoms']['joint_accuracy'],
            alias_top1=outcomes[arm]['summary']['alias']['top1']) for arm in arms},
        semantic_validation_used=False,
    )
    return controls


def _resource_snapshot():
    fields = {line.split(':')[0]: int(line.split()[1])
              for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line}
    return dict(available_RAM_MiB=fields['MemAvailable'] / 1024,
                swap_used_MiB=(fields['SwapTotal'] - fields['SwapFree']) / 1024,
                load1=os.getloadavg()[0])


def _environment(cfg, manifest, meaning_sha, preflight, git_revision, resource_before):
    return dict(
        experiment='CBF8', git_revision=git_revision, measurement_git_revision=git_revision,
        preparation_git_revision=manifest['preparation_git_revision'],
        python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
        platform=platform.platform(), cuda=torch.version.cuda,
        threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
        resource_before=resource_before, paid_resources=False,
        protocol_sha256=sha(PROTOCOL), corpus_manifest_sha256=sha(Path(cfg['output_root']) / 'corpus_manifest.json'),
        meaning_audit_sha256=meaning_sha, preflight_attempts=preflight,
        blind_review=manifest['blind_review'],
        blind_predictions_sha256=sha(Path(cfg['output_root']) / manifest['blind_review']['predictions_file']),
        source_root=cfg['source_root'], output_root=cfg['output_root'],
        source_result_manifest_sha256=manifest['source_result_manifest_sha256'],
        model_repo=cfg['model']['repo'], model_revision=cfg['model']['revision'],
        model_weight_sha256=cfg['model']['weight_sha256'], floating_parameters=cfg['model']['floating_parameters'],
        license_category=cfg['model']['license_class'], one_model_resident_at_a_time=True,
        gpu_serialization='exclusive /tmp/vey-gpu.lock held during NLI model residency',
        token_limit=cfg['resource_budget']['max_pair_tokens'], pair_batch=cfg['resource_budget']['pair_batch'],
        threads_budget=cfg['resource_budget']['threads'], BLAS_threads=cfg['resource_budget']['BLAS'],
        paid=cfg['resource_budget']['paid'], semantic_validation_checkpoint_selection=False,
    )


def _save_new_arm(root, arm):
    put(root, arm['normalizer_name'], arm['normalizer_bytes'])
    put(root, arm['checkpoint_name'], arm['checkpoint_bytes'])
    put(root, f'{arm["details"]["arm_id"]}_training.json', _json_bytes(arm['details']))


def _select(outcomes):
    passed = [arm for arm in ELIGIBLE_ORDER if outcomes[arm]['passed']]
    if passed:
        selected = passed[0]
        reason = 'first eligible all-gate development pass in preregistered arm order'
    else:
        selected = sorted(ELIGIBLE_ORDER, key=lambda arm: (
            -outcomes[arm]['atoms']['joint_accuracy'],
            -outcomes[arm]['summary']['alias']['top1'],
            ELIGIBLE_ORDER.index(arm),
        ))[0]
        reason = 'highest eligible development joint atom, then alias decisions, then earlier arm order'
    return dict(
        selected=selected, reason=reason, passed=selected in passed,
        development={arm: dict(passed=outcomes[arm]['passed'], gates=outcomes[arm]['gates'],
                               joint_accuracy=outcomes[arm]['atoms']['joint_accuracy'],
                               alias_decision_accuracy=outcomes[arm]['summary']['alias']['top1'])
                     for arm in ARM_ORDER},
        arm_order=list(ARM_ORDER), eligible_order=list(ELIGIBLE_ORDER),
        semantic_validation_used=False, final_outcomes_never_select_arm=True,
        final_openings=1,
    )


def _next_branch(cfg, outcomes, selection, final):
    if final['passed']:
        return cfg['replication']
    selected = outcomes[selection['selected']]
    if (selected['atoms']['axis_accuracy'] >= cfg['gates']['G1_axis'] and
            selected['atoms']['conditional_sign_accuracy'] is not None and
            selected['atoms']['conditional_sign_accuracy'] < cfg['gates']['G2_sign_given_axis']):
        return cfg['failure_branches']['axis_pass_sign_fail']
    if all(not outcomes[arm]['passed'] for arm in ELIGIBLE_ORDER):
        return cfg['failure_branches']['all_support_arms_fail']
    return ('Fresh-final gates did not all pass; replication and B-STEF remain blocked. '
            'The frozen CBF8 protocol defines no additional intervention for this outcome.')


def _encode_final(root, selected, fmt, texts, cfg, tokenizer, model, before_hash):
    rows = _rows(texts, tuple(cfg['schema']), fmt, cfg)
    evidence = _tokenize_rows(rows, tokenizer, fmt)
    parity = _native_sentencepiece_parity(rows, tokenizer, fmt, cfg)
    parity_name = f'native_sentencepiece_parity_final_{selected}.json'
    put(root, parity_name, _json_bytes(parity))
    return _cache_one(root, fmt, evidence, _json_bytes(evidence), model, before_hash,
                      cfg, cache_key=f'final_{selected}')


def run(protocol_path=PROTOCOL):
    cfg = _protocol(protocol_path)
    assert_gate_equivalence(cfg)
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        assert torch.get_num_interop_threads() == 1
    torch.manual_seed(cfg['training']['seed'])
    root, manifest, meaning_sha, preflight = _root_state(cfg)
    resource_before = _resource_snapshot()
    guard_resources()
    git_revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE.parents[1], text=True).strip()
    environment = _environment(cfg, manifest, meaning_sha, preflight, git_revision, resource_before)
    put(root, 'environment.json', _json_bytes(environment))
    literal_training = load(root / 'training_atoms.json')
    literal_validation = load(root / 'validation_atoms.json')
    g0 = load(root / 'literal_holdout_atoms.json')
    development = load(root / 'development_atoms.json')
    compositions = load(root / 'development_compositions.json')
    literals = load(root / 'literal_cases.json')
    states = load(root / 'states.json')
    semantic_training = load(root / 'semantic_training_atoms.json')
    semantic_validation = load(root / 'semantic_validation_atoms.json')
    assert len(literal_training) == len(literal_validation) == len(g0) == len(development) == 32
    assert len(compositions) == 50 and len(literals) == 155 and len(states) == 64
    assert len(semantic_training) == 64 and len(semantic_validation) == 32
    assert not ({row['text'] for row in semantic_training} & {row['text'] for row in semantic_validation})

    encoder_lineage = load(root / 'encoder_lineage.json')
    assert encoder_lineage['protocol_sha256'] == sha(PROTOCOL)
    assert encoder_lineage['meaning_audit_sha256'] == meaning_sha
    assert encoder_lineage['encoder_parameters_before_sha256'] == encoder_lineage['encoder_parameters_after_sha256']
    assert encoder_lineage['weight_sha256'] == cfg['model']['weight_sha256']
    assert encoder_lineage['revision'] == cfg['model']['revision']
    assert encoder_lineage['floating_parameters'] == 70831107 and encoder_lineage['width'] == 384
    assert encoder_lineage['pooler_invoked'] is False and encoder_lineage['classifier_invoked'] is False
    score_texts = sorted(set(__import__('schema_relation_evaluate').semantic_contexts(development, compositions)) |
                         {row['text'] for row in g0})
    schema = tuple(cfg['schema'])
    features_by_format, lookups, format_pairs = {}, {}, {}
    for fmt in ('generic', 'defined'):
        cache = load(root / f'feature_cache_{fmt}.json')
        assert cache['feature_sha256'] == sha(root / f'features_{fmt}.npy')
        assert cache['input_sha256'] == sha(root / f'inputs_{fmt}.json')
        assert cache['encoder_parameters_sha256'] == encoder_lineage['encoder_parameters_before_sha256']
        features = np.load(root / f'features_{fmt}.npy', mmap_mode='r', allow_pickle=False)
        evidence, lookup = _feature_lookup(root / f'inputs_{fmt}.json', features, schema)
        format_pairs[fmt] = [(row['criterion'], row['field']) for row in evidence['pairs']]
        assert all((text, field) in lookup for text in score_texts for field in schema)
        features_by_format[fmt], lookups[fmt] = features, lookup
    assert format_pairs['generic'] == format_pairs['defined']

    from schema_relation_evaluate import evaluate, uncertainty
    baseline_matrices, baseline_readout, baseline_module, (base_mean, base_std) = \
        _load_baseline(root, cfg, manifest, g0, development, compositions)
    outcomes = {}
    outcomes['literal_generic'] = evaluate(root, 'development_literal_generic', development,
                                           compositions, literals, states, baseline_matrices, g0)
    new_arms = {}
    for arm in ELIGIBLE_ORDER:
        fmt = arm.split('_', 1)[1]
        features = features_by_format[fmt]
        module_data = _train_arm(arm, cfg, root, literal_training, literal_validation,
                                 semantic_training, features, lookups[fmt], encoder_lineage)
        new_arms[arm] = module_data
        module = module_data['module']
        mean, std = module_data['mean'], module_data['std']
        eval_values = _rows_for_texts(score_texts, features, lookups[fmt], schema)
        matrices = _make_matrices(module, eval_values, mean, std, score_texts)
        outcomes[arm] = evaluate(root, f'development_{arm}', development,
                                 compositions, literals, states, matrices, g0)
        _semantic_validation(root, arm, module, mean, std, features,
                             lookups[fmt], semantic_validation, schema)
    # The ineligible baseline semantic-validation file is diagnostic only; it uses the shared generic CLS cache
    # because CBF7 did not encode this new diagnostic cohort. Its source development readout remains untouched.
    baseline_feature = features_by_format['generic']
    baseline_lookup = lookups['generic']
    _semantic_validation(root, 'literal_generic', baseline_module, base_mean, base_std,
                         baseline_feature, baseline_lookup, semantic_validation, schema)
    assert set(outcomes) == set(ARM_ORDER)

    paired = _paired_controls(root, outcomes, development)

    # The encoder was absent during CPU-only head fitting. Reload the pinned weights,
    # verify their full parameter hash, then keep that sole model resident for final capture.
    from relation_pretrained_encoder import load_model
    with gpu_serialization():
        guard_resources()
        tokenizer, model, actual_lineage = load_model('nli_xsmall', device='cuda')
        encoder_after_training_hash = tensor_hash(model.named_parameters())
        assert encoder_after_training_hash == encoder_lineage['encoder_parameters_before_sha256']
        assert actual_lineage['weight_sha256'] == cfg['model']['weight_sha256']
        for arm, data in new_arms.items():
            data['details']['encoder_parameters_after_sha256'] = encoder_after_training_hash
            data['details']['frozen_encoder_before_sha256'] = encoder_lineage['encoder_parameters_before_sha256']
            data['details']['frozen_encoder_after_sha256'] = encoder_after_training_hash
            data['details']['gradient_proof'].update(dict(encoder_l1=0.0, frozen_gradients=True,
                                                          optimizer_parameters='head_only'))
            _save_new_arm(root, data)
        for name in ARM_ORDER:
            assert not (root / f'development_{name}').exists() or (root / f'development_{name}/results.json').is_file()
        selection = _select(outcomes)
        selection.update(dict(
            protocol_sha256=sha(PROTOCOL), meaning_audit_sha256=meaning_sha,
            baseline_references=baseline_readout['source_files'],
            paired_development_controls_sha256=sha_bytes(_json_bytes(paired)),
        ))
        put(root, 'paired_development_controls.json', _json_bytes(paired))
        put(root, 'selection.json', _json_bytes(selection))

        # No final atom or composition is loaded before the immutable selection above.
        final_atoms = load(root / 'final_atoms.json')
        final_compositions = load(root / 'final_compositions.json')
        from schema_relation_evaluate import semantic_contexts
        final_texts = sorted(set(semantic_contexts(final_atoms, final_compositions)) |
                             {row['text'] for row in g0})
        selected = selection['selected']
        final_fmt = selected.split('_', 1)[1]
        final_features, final_cache = _encode_final(
            root, selected, final_fmt, final_texts, cfg, tokenizer, model, encoder_after_training_hash)
        assert tensor_hash(model.named_parameters()) == encoder_after_training_hash
        final_lookup = {(row['criterion'], row['field']): index
                        for index, row in enumerate(load(root / f'inputs_final_{selected}.json')['pairs'])}
        final_h = _rows_for_texts(final_texts, final_features, final_lookup, schema)
        selected_head = nn.Linear(384, 3, dtype=torch.float32)
        selected_head.load_state_dict(new_arms[selected]['state'], strict=True)
        selected_head.eval()
        selected_mean, selected_std = new_arms[selected]['mean'], new_arms[selected]['std']
        final_matrices = _make_matrices(selected_head, final_h, selected_mean, selected_std, final_texts)
        final_readout = dict(
            selected_arm=selected, format=final_fmt, model_repo=cfg['model']['repo'],
            model_revision=cfg['model']['revision'], model_weight_sha256=cfg['model']['weight_sha256'],
            floating_parameters=cfg['model']['floating_parameters'],
            encoder_parameters_before_sha256=encoder_lineage['encoder_parameters_before_sha256'],
            encoder_parameters_after_training_sha256=encoder_after_training_hash,
            encoder_parameters_after_final_capture_sha256=tensor_hash(model.named_parameters()),
            feature_file=f'features_final_{selected}.npy', feature_sha256=final_cache['feature_sha256'],
            input_file=f'inputs_final_{selected}.json', input_sha256=final_cache['input_sha256'],
            native_sentencepiece_parity_sha256=sha(root / f'native_sentencepiece_parity_final_{selected}.json'),
            checkpoint_file=f'{selected}_checkpoint.pt', checkpoint_sha256=sha(root / f'{selected}_checkpoint.pt'),
            normalizer_file=f'normalizer_{selected}.npz', normalizer_sha256=sha(root / f'normalizer_{selected}.npz'),
            normalizer_mean_sha256=new_arms[selected]['details']['normalizer_mean_sha256'],
            normalizer_std_sha256=new_arms[selected]['details']['normalizer_std_sha256'],
            meaning_audit_sha256=meaning_sha, final_openings=1,
        )
        put(root, 'final_readout.json', _json_bytes(final_readout))
        del model, tokenizer
        torch.cuda.empty_cache()

    final = evaluate(root, 'final', final_atoms, final_compositions, literals, states, final_matrices, g0)
    final_intervals = uncertainty(root, 'final', final_atoms, final_compositions)
    result = dict(
        experiment='CBF8 Schema-support domain control', evidence_class='MEASURED: one preregistered frozen-encoder screen',
        git_revision=git_revision, measurement_git_revision=git_revision,
        preparation_git_revision=manifest['preparation_git_revision'],
        protocol_sha256=sha(PROTOCOL), corpus_manifest_sha256=sha(root / 'corpus_manifest.json'),
        meaning_audit_sha256=meaning_sha, source_root=cfg['source_root'], output_root=cfg['output_root'],
        source_result_manifest_sha256=manifest['source_result_manifest_sha256'],
        blind_review=manifest['blind_review'],
        blind_predictions_sha256=sha(root / manifest['blind_review']['predictions_file']),
        development=outcomes, selection=selection, final=final, final_intervals=final_intervals,
        paired_development_controls=paired, baseline_readout=baseline_readout,
        baseline_references=manifest['baseline_references'],
        final_readout=final_readout,
        final_readout_sha256=sha(root / 'final_readout.json'),
        replication_required=bool(final['passed']), B_STEF_allowed=False,
        verdict='CBF8_SUPPORT_SCREEN_PASS_REPLICATION_REQUIRED' if final['passed']
                else 'CBF8_SUPPORT_NOT_EARNED_ON_FRESH_CORPUS',
        next_branch=_next_branch(cfg, outcomes, selection, final),
        license_category=cfg['model']['license_class'],
        license_warning='The NLI checkpoint remains conditional/review and research-only; this result does not clear shipping obligations.',
        semantic_validation_used_for_selection=False,
        final_outcomes_never_select_arm=True,
        no_final_tuning=True,
    )
    put(root, 'results.json', _json_bytes(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    args = parser.parse_args()
    print(json.dumps(run(args.protocol), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
