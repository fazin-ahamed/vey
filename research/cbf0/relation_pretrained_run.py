"""CBF-7 relation-pretrained staged frozen-encoder screen with one fresh final opening.

Six stock/xsmall arms complete before selection. NLI-small runs only if no
eligible stock/xsmall arm passes development. Literal criteria train R heads;
aliases are evaluation inputs. Canonical fields or fixed generic hypotheses
form the second pair text. Labels, numeric weights, full compositions and
candidate descriptions never enter the encoder.
"""
import hashlib
import json
import os
import platform
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch import nn

from audit import put
from build import canonical
from relation_pretrained_encoder import (
    SCHEMA,
    encode_pairs,
    full_forward_parity,
    load_model,
    native_class_indices,
    pooler_features,
    reordered_native_weights,
    tokenizer_parity,
)

PROTOCOL = Path(__file__).with_name('relation_pretrained_protocol.json')
SOURCE_CBF6 = Path(__file__).with_name('schema_relation_protocol.json')
ARM_ORDER = ('stock_bare', 'stock_higher_cls', 'nli_xsmall_bare_cls', 'nli_xsmall_higher_cls',
             'nli_xsmall_higher_pooler', 'nli_xsmall_native', 'nli_small_higher_pooler', 'nli_small_native')
@contextmanager
def gpu_serialization():
    """Serialize model residency with every experiment using the shared GPU lock."""
    import fcntl

    guard_resources()
    lock_path = Path('/tmp/vey-gpu.lock')
    handle = lock_path.open('a')
    fcntl.flock(handle, fcntl.LOCK_EX)
    print(json.dumps(dict(gpu_lock=str(lock_path), held=True)), flush=True)
    try:
        yield
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def load(root, name):
    return json.loads((root / name).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def resource_snapshot():
    fields = {line.split(':')[0]: int(line.split()[1])
              for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line}
    return dict(available_RAM_MiB=fields['MemAvailable'] / 1024,
                swap_used_MiB=(fields['SwapTotal'] - fields['SwapFree']) / 1024,
                load1=os.getloadavg()[0])


def guard_resources():
    snapshot = resource_snapshot()
    while snapshot['available_RAM_MiB'] < 4096 or snapshot['swap_used_MiB'] > INITIAL_SWAP + 256:
        torch.set_num_threads(2)
        print(json.dumps(dict(resource_pause=snapshot)), flush=True)
        time.sleep(5)
        snapshot = resource_snapshot()
    if snapshot['load1'] > max(4, os.cpu_count() * .75):
        torch.set_num_threads(2)
        time.sleep(1)


INITIAL_SWAP = resource_snapshot()['swap_used_MiB']


def pairs_for(atoms):
    return ([(c['text'], field) for c in atoms for field in SCHEMA],
            torch.tensor([c['sign'] + 1 if j == c['axis'] else 1 for c in atoms for j in range(4)],
                         dtype=torch.long))


def state_copy(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def tensor_hash(parameters):
    digest = hashlib.sha256()
    for name, parameter in parameters:
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def assert_gate_equivalence():
    new_gates = json.loads(PROTOCOL.read_text())['gates']
    old_gates = json.loads(SOURCE_CBF6.read_text())['gates']
    assert new_gates == old_gates, f'CBF-7 gates diverge from the CBF-6 helper gates: {new_gates} != {old_gates}'


def feature_interface(arm_id):
    """Map an arm id to its frozen feature interface and native-head flag."""
    spec = {
        'stock_bare': dict(feature='CLS', native=False, reuse_cbf6_a=True),
        'stock_higher_cls': dict(feature='CLS', native=False),
        'nli_xsmall_bare_cls': dict(feature='CLS', native=False),
        'nli_xsmall_higher_cls': dict(feature='CLS', native=False),
        'nli_xsmall_higher_pooler': dict(feature='pooler', native=False),
        'nli_xsmall_native': dict(feature='pooler', native=True),
        'nli_small_higher_pooler': dict(feature='pooler', native=False),
        'nli_small_native': dict(feature='pooler', native=True),
    }[arm_id]
    spec['arm_id'] = arm_id
    return spec


def train_linear_head(X, y, V, vy, config, normalizer):
    """Train one new linear width-to-3 R head, full-batch, on CPU from cached features."""
    torch.manual_seed(config['seed'])
    module = nn.Linear(X.shape[1], 3, dtype=torch.float32)
    optimizer = torch.optim.AdamW(module.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    weights = torch.tensor(config['class_weights'])
    best, best_state, trace, first_gradient, selected_epoch = float('inf'), None, [], None, None
    for epoch in range(config['epochs']):
        guard_resources()
        module.train()
        optimizer.zero_grad(set_to_none=True)
        logits = module(X)
        ce = nn.functional.cross_entropy(logits, y, weight=weights)
        loss = ce
        assert torch.isfinite(loss)
        loss.backward()
        if epoch == 0:
            first_gradient = float(sum(parameter.grad.abs().sum() for parameter in module.parameters()))
            assert first_gradient > 0
        optimizer.step()
        module.eval()
        with torch.no_grad():
            val_logits = module(V)
            validation = float(nn.functional.cross_entropy(val_logits, vy))
            accuracy = float((val_logits.argmax(1) == vy).float().mean())
        trace.append(dict(epoch=epoch, loss=float(loss), CE=float(ce),
                         validation_CE=validation, validation_accuracy=accuracy))
        if validation < best:
            best, best_state, selected_epoch = validation, state_copy(module), epoch
    module.load_state_dict(best_state, strict=True)
    module.eval()
    details = dict(
        arm_id=normalizer['arm_id'], kind='linear', width=X.shape[1],
        trainable_parameters=sum(parameter.numel() for parameter in module.parameters()),
        epochs=config['epochs'], selected_epoch=selected_epoch, validation_CE=best,
        first_gradient_L1=first_gradient, trace=trace,
        normalizer=normalizer, loss='full-batch weighted CE, unweighted validation checkpoint',
        optimizer='AdamW', lr=config['lr'], weight_decay=config['weight_decay'], seed=config['seed'],
    )
    return module, details


def matrices_for(module, H, mean, std, texts, device='cpu'):
    with torch.no_grad():
        p = module(torch.tensor((H - mean) / std, dtype=torch.float32, device=device)).softmax(-1).numpy()
    assert p.shape == (4 * len(texts), 3)
    return {q: p[4 * i:4 * i + 4] for i, q in enumerate(texts)}


def native_matrices(model, pooled_features, mapping, texts):
    """Apply the reordered frozen classifier to already-cached native pooler rows."""
    weight, bias = reordered_native_weights(model, mapping)
    module = nn.Linear(pooled_features.shape[1], 3, dtype=torch.float32)
    with torch.no_grad():
        module.weight.copy_(weight.cpu())
        module.bias.copy_(bias.cpu())
    module.eval()
    mean = np.zeros(pooled_features.shape[1], dtype=np.float64)
    std = np.ones(pooled_features.shape[1], dtype=np.float64)
    return matrices_for(module, pooled_features, mean, std, texts), dict(
        module=module, mean=mean, std=std)


def cache_features(root, model_key, texts, tokenizer, model, format_name, device='cuda'):
    """Encode every text-field pair once and cache raw CLS features under the data root.

    Pooler features are derived on the same cached CLS rows for NLI models so
    the literal-linear and native readouts share one forward pass per batch.
    """
    tag = f'{model_key}_{format_name}'
    cls_path = root / f'features_{tag}_cls.npy'
    pooler_path = root / f'features_{tag}_pooler.npy'
    inputs_path = root / f'inputs_{tag}.json'
    if cls_path.exists():
        cls = np.load(cls_path)
        evidence = load(root, inputs_path.name)
        assert [dict(criterion=p['criterion'], field=p['field']) for p in evidence['pairs']] == \
            [dict(criterion=q, field=f) for q in texts for f in SCHEMA]
        pooled = np.load(pooler_path) if pooler_path.exists() else None
        return cls, pooled, evidence
    pairs = [(q, f) for q in texts for f in SCHEMA]
    guard_resources()
    cls, evidence = encode_pairs(pairs, tokenizer, model, format_name, batch_size=32, device=device)
    assert evidence['candidate_forwards'] == 0 and not evidence['truncation']
    np.save(cls_path, cls)
    if hasattr(model, 'pooler'):
        pooled = pooler_features(cls, model, device=device)
        np.save(pooler_path, pooled)
    else:
        pooled = None
    put(root, inputs_path.name, canonical(evidence) + b'\n')
    return cls, pooled, evidence


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(7)
    cfg = json.loads(PROTOCOL.read_text())
    root = Path(cfg['output_root'])
    c6_root = Path(cfg['source_root'])
    corpus = load(root, 'corpus_manifest.json')
    assert corpus['protocol_sha256'] == sha(PROTOCOL)
    assert corpus['experiment'] == 'CBF-7'
    assert corpus['final_closed_until_selection'] is True
    for name, expected in corpus['files'].items():
        assert sha(root / name) == expected, name
    for name, expected in corpus['source_verified_files'].items():
        assert sha(c6_root / name) == expected, name
    audit_path = root / 'meaning_audit.json'
    audit = load(root, 'meaning_audit.json')
    assert audit['all_meanings_accepted']
    assert audit['final_atoms_sha256'] == sha(root / 'final_atoms.json')
    assert audit['final_compositions_sha256'] == sha(root / 'final_compositions.json')
    assert_gate_equivalence()
    if (root / 'selection.json').exists() or (root / 'results.json').exists():
        raise RuntimeError('Refusing to overwrite CBF-7 stage/final evidence')

    training = load(root, 'training_atoms.json')
    validation = load(root, 'validation_atoms.json')
    g0 = load(root, 'literal_holdout_atoms.json')
    dev = load(root, 'development_atoms.json')
    comps = load(root, 'development_compositions.json')
    literals = load(root, 'literal_cases.json')
    states = load(root, 'states.json')
    from schema_relation_evaluate import evaluate, semantic_contexts, uncertainty
    environment_git = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).parents[2],
                                              text=True).strip()
    environment = dict(
        git_revision=environment_git,
        python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
        platform=platform.platform(), cuda=torch.version.cuda, device=torch.cuda.get_device_name(0),
        threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
        resource_before=resource_snapshot(), initial_swap_MiB=INITIAL_SWAP,
        protocol_sha256=sha(PROTOCOL), corpus_sha256=sha(root / 'corpus_manifest.json'),
        seed=cfg['training']['seed'], paid_resources=False,
        meaning_audit_sha256=sha(audit_path),
        HF_HOME=os.environ.get('HF_HOME'), HF_HUB_OFFLINE=os.environ.get('HF_HUB_OFFLINE'),
        source_inventory_sha256=corpus['source_inventory_sha256'],
        license_category=cfg['models']['nli_xsmall']['eligibility'],
        data_license_class=corpus.get('data_license_class'),
        one_model_resident_at_a_time=True, cached_features=True,
        gpu_serialization='exclusive /tmp/vey-gpu.lock held throughout each model residency',
        token_limit=cfg['resource_budget']['tokens'], pair_batch=cfg['resource_budget']['pair_batch'],
        source_root=str(c6_root), output_root=str(root),
        actual_model_git_revision=environment_git,
    )
    put(root, 'environment.json', canonical(environment) + b'\n')
    score_texts = sorted(set(semantic_contexts(dev, comps)) | {c['text'] for c in g0})
    base_features = np.load(c6_root / 'frozen_pair_features.npy')
    base_evidence = load(c6_root, 'frozen_pair_inputs.json')
    base_pairs = [dict(criterion=p['criterion'], field=p['field']) for p in base_evidence['pairs']]
    all_texts = sorted({p['criterion'] for p in base_pairs})
    assert base_pairs == [dict(criterion=q, field=f) for q in all_texts for f in SCHEMA]
    assert len(base_features) == 4 * len(all_texts)
    base_normalizer = np.load(c6_root / 'normalizer.npz')
    base_mean, base_std = base_normalizer['mean'], base_normalizer['std']
    base_lookup = {q: i for i, q in enumerate(all_texts)}
    base_score_h = np.concatenate([base_features[4 * base_lookup[q]:4 * base_lookup[q] + 4] for q in score_texts])
    base_module = nn.Linear(384, 3, dtype=torch.float32)
    base_ckpt = torch.load(c6_root / 'A_checkpoint.pt', map_location='cpu', weights_only=True)
    assert base_ckpt['kind'] == 'linear' and base_ckpt['last_layer'] is None
    base_module.load_state_dict(base_ckpt['head'], strict=True)
    base_module.eval()
    base_training = load(c6_root, 'A_training.json')
    assert base_training['frozen_encoder_before_sha256'] == base_training['frozen_encoder_after_sha256']
    base_matrices = matrices_for(base_module, base_score_h, base_mean, base_std, score_texts)
    base_outcome = evaluate(root, 'development_stock_bare', dev, comps, literals, states, base_matrices, g0)
    outcomes = dict(stock_bare=base_outcome)
    native_readouts = dict(
        stock_bare=dict(kind='reuse-cbf6-a', no_retrain=True, no_reencode=True,
                       feature_path=str(c6_root / 'frozen_pair_features.npy'),
                       feature_sha256=sha(c6_root / 'frozen_pair_features.npy'),
                       normalizer_sha256=sha(c6_root / 'normalizer.npz'),
                       checkpoint_sha256=sha(c6_root / 'A_checkpoint.pt'),
                       training_sha256=sha(c6_root / 'A_training.json'),
                       frozen_encoder_before_sha256=base_training['frozen_encoder_before_sha256'],
                       frozen_encoder_after_sha256=base_training['frozen_encoder_after_sha256'],
                       lineage_sha256=sha(c6_root / 'encoder_lineage.json')))
    print(json.dumps(dict(arm='stock_bare',
                         literal_relation=base_outcome['literal_relation'],
                         atoms=base_outcome['atoms'], gates=base_outcome['gates']), indent=2), flush=True)

    holder = dict(tokenizer=None, model=None)
    checkpoints = {}

    def encode_and_train(arm_id, model_key, format_name):
        interface = feature_interface(arm_id)
        cls, pooled, _evidence = cache_features(root, model_key, all_texts, holder['tokenizer'],
                                               holder['model'], format_name)
        lookup = {q: i for i, q in enumerate(all_texts)}

        def rows(texts, source):
            return np.concatenate([source[4 * lookup[q]:4 * lookup[q] + 4] for q in texts])

        source = cls if interface['feature'] == 'CLS' else pooled
        train_labels = pairs_for(training)[1]
        val_labels = pairs_for(validation)[1]
        train_h, val_h, score_h = rows([c['text'] for c in training], source), \
            rows([c['text'] for c in validation], source), rows(score_texts, source)
        floating = sum(parameter.numel() for parameter in holder['model'].parameters())
        frozen_before = tensor_hash(holder['model'].named_parameters())
        if interface['native']:
            mapping = native_class_indices(holder['model'])
            dev_matrices, replay = native_matrices(holder['model'], score_h, mapping, score_texts)
            weight, bias = reordered_native_weights(holder['model'], mapping)
            details = dict(
                arm_id=arm_id, kind='native-frozen',
                mapping={str(k): v for k, v in mapping.items()},
                floating_parameters=floating, mean_0_std_1=True,
                classifier_weight_sha256=hashlib.sha256(
                    weight.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
                classifier_bias_sha256=hashlib.sha256(
                    bias.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
                normalizer=dict(arm_id=arm_id, mean_sha256='zeros', std_sha256='ones', floor=0.0),
                evidence_sha256=sha(root / f"inputs_{model_key}_{format_name}.json"),
                frozen_encoder_before_sha256=frozen_before,
                frozen_encoder_after_sha256=tensor_hash(holder['model'].named_parameters()))
            assert details['frozen_encoder_after_sha256'] == frozen_before, 'native NLI model drifted during encoding'
            native_readouts[arm_id] = dict(details)
            put(root, f'{arm_id}_native_readout.json', canonical(dict(details, replay_sha256=sha(root / f"features_{model_key}_{format_name}_pooler.npy"))) + b'\n')
            torch.save(state_copy(replay['module']), root / f'{arm_id}_checkpoint.pt')
            np.savez(root / f'normalizer_{arm_id}.npz', mean=replay['mean'], std=replay['std'])
            outcome = evaluate(root, f'development_{arm_id}', dev, comps, literals, states, dev_matrices, g0)
            outcomes[arm_id] = outcome
            checkpoints[arm_id] = dict(module=state_copy(replay['module']), arm_id=arm_id, native=True,
                                       model_key=model_key, format_name=format_name, floating_parameters=floating,
                                       mean=replay['mean'], std=replay['std'])
            print(json.dumps(dict(arm=arm_id, literal_relation=outcome['literal_relation'],
                                  atoms=outcome['atoms'], gates=outcome['gates']), indent=2), flush=True)
            return
        mean = train_h.mean(0)
        std = np.maximum(train_h.std(0), .01)
        np.savez(root / f'normalizer_{arm_id}.npz', mean=mean, std=std)
        normalizer = dict(arm_id=arm_id, mean_sha256=hashlib.sha256(mean.tobytes()).hexdigest(),
                          std_sha256=hashlib.sha256(std.tobytes()).hexdigest(), floor=0.01, train_only=True)
        X = torch.tensor((train_h - mean) / std, dtype=torch.float32)
        V = torch.tensor((val_h - mean) / std, dtype=torch.float32)
        module, details = train_linear_head(X, train_labels, V, val_labels, cfg['training'], normalizer)
        frozen_after = tensor_hash(holder['model'].named_parameters())
        assert frozen_after == frozen_before, f'{arm_id} encoder drifted during head training'
        details.update(dict(feature=interface['feature'], floating_parameters=floating,
                            evidence_sha256=sha(root / f"inputs_{model_key}_{format_name}.json"),
                            feature_sha256=sha(root / f"features_{model_key}_{format_name}_{'pooler' if interface['feature'] == 'pooler' else 'cls'}.npy"),
                            frozen_encoder_before_sha256=frozen_before,
                            frozen_encoder_after_sha256=frozen_after))
        put(root, f'{arm_id}_training.json', canonical(details) + b'\n')
        torch.save(state_copy(module), root / f'{arm_id}_checkpoint.pt')
        matrices = matrices_for(module, score_h, mean, std, score_texts)
        outcome = evaluate(root, f'development_{arm_id}', dev, comps, literals, states, matrices, g0)
        outcomes[arm_id] = outcome
        checkpoints[arm_id] = dict(module=state_copy(module), arm_id=arm_id, native=False,
                                   model_key=model_key, format_name=format_name,
                                   mean=mean, std=std, floating_parameters=floating)
        print(json.dumps(dict(arm=arm_id, literal_relation=outcome['literal_relation'],
                              atoms=outcome['atoms'], gates=outcome['gates']), indent=2), flush=True)

    def run_stock_higher():
        with gpu_serialization():
            tokenizer, model, lineage = load_model('stock_xsmall')
            holder['tokenizer'], holder['model'] = tokenizer, model
            put(root, 'encoder_lineage_stock_xsmall_higher_cls.json', canonical(lineage) + b'\n')
            encode_and_train('stock_higher_cls', 'stock_xsmall', 'higher')
            holder.clear()
            del model, tokenizer
            torch.cuda.empty_cache()
        put(root, 'gpu_session_stock_xsmall.json', canonical(dict(
            model_key='stock_xsmall', formats=('higher',),
            frozen_model_sha256=sha(root / 'encoder_lineage_stock_xsmall_higher_cls.json'),
            locked=True)) + b'\n')

    def run_nli_xsmall():
        with gpu_serialization():
            tokenizer, model, lineage = load_model('nli_xsmall')
            holder['tokenizer'], holder['model'] = tokenizer, model
            put(root, 'encoder_lineage_nli_xsmall.json', canonical(lineage) + b'\n')
            for fmt in ('bare', 'higher'):
                put(root, f'tokenizer_parity_nli_xsmall_{fmt}.json',
                    canonical(tokenizer_parity(score_texts, tokenizer, fmt, 'nli_xsmall')) + b'\n')
            put(root, 'native_forward_parity_nli_xsmall.json',
                canonical(full_forward_parity([(q, f) for q in score_texts[:1] for f in SCHEMA][:2],
                                              tokenizer, model, 'higher')) + b'\n')
            encode_and_train('nli_xsmall_bare_cls', 'nli_xsmall', 'bare')
            encode_and_train('nli_xsmall_higher_cls', 'nli_xsmall', 'higher')
            encode_and_train('nli_xsmall_higher_pooler', 'nli_xsmall', 'higher')
            encode_and_train('nli_xsmall_native', 'nli_xsmall', 'higher')
            holder.clear()
            del model, tokenizer
            torch.cuda.empty_cache()
        put(root, 'gpu_session_nli_xsmall.json', canonical(dict(
            model_key='nli_xsmall', formats=('bare', 'higher'),
            frozen_model_sha256=sha(root / 'encoder_lineage_nli_xsmall.json'),
            locked=True)) + b'\n')

    def run_nli_small():
        with gpu_serialization():
            tokenizer, model, lineage = load_model('nli_small')
            holder['tokenizer'], holder['model'] = tokenizer, model
            put(root, 'encoder_lineage_nli_small.json', canonical(lineage) + b'\n')
            put(root, 'tokenizer_parity_nli_small_higher.json',
                canonical(tokenizer_parity(score_texts, tokenizer, 'higher', 'nli_small')) + b'\n')
            put(root, 'native_forward_parity_nli_small.json',
                canonical(full_forward_parity([(q, f) for q in score_texts[:1] for f in SCHEMA][:2],
                                              tokenizer, model, 'higher')) + b'\n')
            encode_and_train('nli_small_higher_pooler', 'nli_small', 'higher')
            encode_and_train('nli_small_native', 'nli_small', 'higher')
            holder.clear()
            del model, tokenizer
            torch.cuda.empty_cache()
        put(root, 'gpu_session_nli_small.json', canonical(dict(
            model_key='nli_small', formats=('higher',),
            frozen_model_sha256=sha(root / 'encoder_lineage_nli_small.json'),
            locked=True)) + b'\n')

    run_stock_higher()
    run_nli_xsmall()
    xsmall_arms = ('stock_higher_cls', 'nli_xsmall_bare_cls', 'nli_xsmall_higher_cls',
                   'nli_xsmall_higher_pooler', 'nli_xsmall_native')
    if not any(outcomes[a]['passed'] for a in xsmall_arms if a in outcomes):
        print(json.dumps(dict(trigger='no eligible xsmall/stock arm passed development; running nli_small')), flush=True)
        run_nli_small()

    pass_candidates = [a for a in ARM_ORDER if a in outcomes and a != 'stock_bare' and outcomes[a]['passed']]
    if pass_candidates:
        selected, selection_reason = pass_candidates[0], 'first eligible all-gate pass in arm order'
    else:
        ranked = [(a, outcomes[a]['atoms']['joint_accuracy'],
                   outcomes[a]['summary']['alias']['top1'],
                   checkpoints[a]['floating_parameters'])
                  for a in ARM_ORDER if a in outcomes and a in checkpoints]
        ranked.sort(key=lambda item: (-item[1], -item[2], item[3], ARM_ORDER.index(item[0])))
        selected, selection_reason = ranked[0][0], 'highest eligible development joint atom, alias decisions, lower parameter count, earlier order'
    selection = dict(selected=selected, reason=selection_reason, passed=selected in pass_candidates,
                     development={name: dict(gates=outcome['gates'], passed=outcome['passed'],
                                            joint_accuracy=outcome['atoms']['joint_accuracy'],
                                            alias_top1=outcome['summary']['alias']['top1'])
                                 for name, outcome in outcomes.items()},
                     arm_order=list(ARM_ORDER), small_triggered='nli_small_higher_pooler' in outcomes,
                     final_openings=1, final_outcomes_never_select_arm=True,
                     meaning_audit_sha256=sha(audit_path), protocol_sha256=sha(PROTOCOL))
    put(root, 'selection.json', canonical(selection) + b'\n')
    print(json.dumps(selection, indent=2), flush=True)

    final_readout = None
    final_atoms = load(root, 'final_atoms.json')
    final_comps = load(root, 'final_compositions.json')
    final_texts = sorted(set(semantic_contexts(final_atoms, final_comps)) | {c['text'] for c in g0})
    checkpoint = checkpoints[selected]
    model_key = checkpoint['model_key']
    format_name = checkpoint['format_name']
    with gpu_serialization():
        tokenizer, model, _ = load_model(model_key)
        final_cls, final_pooled, _ = cache_features(
            root, f'{model_key}_final', final_texts, tokenizer, model, format_name)
        module = nn.Linear(final_cls.shape[1], 3, dtype=torch.float32)
        module.load_state_dict(checkpoint['module'], strict=True)
        module.eval()
        source = final_cls if feature_interface(selected)['feature'] == 'CLS' else final_pooled
        final_matrices = matrices_for(module, source, checkpoint['mean'], checkpoint['std'], final_texts)
        final_readout = dict(
            kind='native-frozen' if checkpoint['native'] else 'linear',
            evidence_sha256=sha(root / f'inputs_{model_key}_final_{format_name}.json'),
            checkpoint_sha256=sha(root / f'{selected}_checkpoint.pt'),
            normalizer_sha256=sha(root / f'normalizer_{selected}.npz'))
        del model, tokenizer
        torch.cuda.empty_cache()
    final = evaluate(root, 'final', final_atoms, final_comps, literals, states, final_matrices, g0)
    ci = uncertainty(root, 'final', final_atoms, final_comps)
    result = dict(
        experiment='CBF-7', selection=selection, development=outcomes,
        final=final, final_intervals=ci, native_readouts=native_readouts,
        final_readout=final_readout,
        verdict='C7_RESEARCH_ONLY_SCREEN_PASS_REPLICATION_REQUIRED' if final['passed'] else 'C7_NOT_EARNED_ON_FRESH_CORPUS',
        B_STEF_allowed=False, ASG_closed_permanently=True,
        native_bridge_warning='Empirical contradiction/neutral/entailment -> -1/0/+1 bridge for unconditional monotone one-field preferences only; not a universal logical reduction or certificate.',
        license_category=cfg['models'][model_key]['eligibility'],
        source_root=str(c6_root), output_root=str(root),
        replication_required=final['passed'],
        replication='learned R head: retrain head seeds 11/13, every seed must pass before capability promotion; native frozen classifier: two independently authored blinded new 128-atom/128-composition confirmation pools, inference repeats/shuffles are not seed replication; shipping: checkpoint provenance conditional/review blocks shipping release until resolved',
    )
    put(root, 'results.json', canonical(result) + b'\n')
    print(json.dumps(result, indent=2), flush=True)




if __name__ == '__main__':
    main()
