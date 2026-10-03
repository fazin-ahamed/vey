"""CBF-2: closed-form ceilings and support-aware checkpoint interventions."""
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from audit import ARMS, direction, numeric, put, swap_summary
from build import CATALOG, canonical
from run import corr, fit, load, metrics, paired_ci, score_one
from scipy.stats import pearsonr, spearmanr

PROTOCOL = Path(__file__).with_name('ceiling_protocol.json')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def factor(H):
    u, s, vt = np.linalg.svd(H, full_matrices=False)
    cutoff = np.finfo(np.float64).eps * max(H.shape) * s[0]
    keep = s > cutoff
    p = s[s > 0] / s.sum()
    spectrum = dict(shape=list(H.shape), singular_values=s.tolist(),
                    cutoff=float(cutoff), rank=int(keep.sum()),
                    condition_number=float(s[0] / s[-1]) if s[-1] > 0 else None,
                    entropy_singular_value_effective_rank=float(np.exp(-np.sum(p * np.log(p)))))
    return (u, s, vt, keep), spectrum


def solve(decomp, Y, lam=0.):
    u, s, vt, keep = decomp
    if lam:
        gain = s / (s * s + lam)
    else:
        gain = np.zeros_like(s)
        gain[keep] = 1 / s[keep]
    return (vt.T * gain) @ (u.T @ Y)


def examples(rows):
    items = {}
    counts = Counter()
    for r in rows:
        for text, y in zip(r['candidates'], numeric(r)):
            counts[text] += 1
            if text in items:
                assert np.array_equal(items[text]['target'], y)
                assert items[text]['scenario'] == r['scenario_id']
            else:
                items[text] = dict(text=text, target=y, scenario=r['scenario_id'],
                                   family=r['family'], K=r['K'])
    return [items[t] for t in sorted(items)], counts


def recovery(items, pred):
    y = np.stack([r['target'] for r in items])
    groups = {}
    for i, r in enumerate(items):
        groups.setdefault(r['scenario'], []).append(i)
    ordering = [[] for _ in range(4)]
    for inds in groups.values():
        a, b = np.triu_indices(len(inds), 1)
        for j in range(4):
            target = y[inds, j]
            scores = pred[inds, j]
            ordering[j].append(float(np.mean(np.sign(target[a] - target[b]) ==
                                            np.sign(scores[a] - scores[b]))))
    out = []
    for j in range(4):
        variance = float(np.sum((y[:, j] - y[:, j].mean()) ** 2))
        squared = float(np.sum((pred[:, j] - y[:, j]) ** 2))
        out.append(dict(slot=j, n=len(items), mse=squared / len(items),
                        R2=1 - squared / variance if variance else None,
                        pearson=corr(pearsonr, pred[:, j], y[:, j]),
                        spearman=corr(spearmanr, pred[:, j], y[:, j]),
                        pairwise=float(np.mean(ordering[j])), scenarios=len(groups)))
    return out


def decision_summary(records, arms):
    out = dict(n=len(records))
    for arm in arms:
        out[arm] = {}
        for key in ('top1', 'pairwise', 'spearman', 'kendall', 'pearson'):
            values = [r[arm][key] for r in records if r[arm][key] is not None]
            out[arm][key] = float(np.mean(values)) if values else None
            out[arm][key + '_undefined'] = len(records) - len(values)
    return out


def bins(records, summarize):
    out = summarize(records)
    out['per_K'] = {str(k): summarize([r for r in records if r['K'] == k])
                    for k in sorted({r['K'] for r in records})}
    out['per_family'] = {f: summarize([r for r in records if r['family'] == f])
                         for f in sorted({r['family'] for r in records})}
    return out


def resplit(rows, criteria):
    groups = {}
    for r in rows:
        if r['split'] != 'held_family':
            groups.setdefault((r['family'], r['split'], r['K']), set()).add(r['scenario_id'])
    rng = np.random.default_rng(7)
    reserve = set()
    for key in sorted(groups):
        scenarios = sorted(groups[key])
        order = rng.permutation(len(scenarios))
        reserve.update(scenarios[i] for i in order[:max(1, len(scenarios) // 4)])
    ids = {'train': [], 'unseen_wording': [], 'unseen_direction': [], 'held_family': []}
    for r in rows:
        c = criteria[r['criterion_id']]
        index = c['forms'].index(r['criterion'])
        v = direction(c)
        excluded = v[3] == -1
        if r['split'] == 'held_family':
            ids['held_family'].append(r['id'])
        elif r['scenario_id'] not in reserve and index < 2 and not excluded:
            ids['train'].append(r['id'])
    by_id = {r['id']: r for r in rows}
    train_criteria = {by_id[i]['criterion_id'] for i in ids['train']}
    for r in rows:
        if r['scenario_id'] not in reserve:
            continue
        c = criteria[r['criterion_id']]
        if direction(c)[3] == -1:
            ids['unseen_direction'].append(r['id'])
        elif r['criterion_id'] in train_criteria and c['forms'].index(r['criterion']) >= 2:
            ids['unseen_wording'].append(r['id'])
    train = [by_id[i] for i in ids['train']]
    targets = np.stack([direction(criteria[i]) for i in sorted(train_criteria)])
    _, s, vt = np.linalg.svd(targets, full_matrices=False)
    rank = int(sum(s > 1e-10))
    assert rank == 5
    basis = vt[:rank]
    train_scenarios = {r['scenario_id'] for r in train}
    train_texts = {t for r in train for t in r['candidates']}
    errors = []
    for split in ('unseen_wording', 'unseen_direction', 'held_family'):
        assert ids[split]
        rr = [by_id[i] for i in ids[split]]
        assert not train_scenarios & {r['scenario_id'] for r in rr}
        assert not train_texts & {t for r in rr for t in r['candidates']}
        if split != 'unseen_wording':
            assert not train_criteria & {r['criterion_id'] for r in rr}
        for r in rr:
            v = direction(criteria[r['criterion_id']])
            errors.append(float(np.linalg.norm(v - (v @ basis.T) @ basis)))
    assert max(errors) < 1e-10
    return dict(ids={k: sorted(v) for k, v in ids.items()},
                counts={k: len(v) for k, v in ids.items()}, train_target_rank=rank,
                maximum_held_direction_span_residual=max(errors),
                held_direction='negative slot 3; existing corpus only; no mixed-coordinate examples')


def main():
    protocol = json.loads(PROTOCOL.read_text())
    root, source, audit_root = map(Path, (protocol['output_root'], protocol['source_root'], protocol['audit_root']))
    manifest, cs, rows = load(source)
    criteria = {c['id']: c for c in cs}
    previous_inventory = json.loads(Path(__file__).with_name('audit_result_manifest.json').read_text())
    for name, expected in previous_inventory['artifact_hashes'].items():
        assert sha(audit_root / name) == expected, name
    source_inventory = previous_inventory['run_manifest']['source_hashes']
    for name, expected in source_inventory.items():
        assert sha(source / name) == expected, name
    root.mkdir(parents=True, exist_ok=True)
    run_manifest = dict(protocol=protocol, protocol_sha256=sha(PROTOCOL), source_sha256=sha(Path(__file__)),
                        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                        source_manifest=manifest, source_hashes=source_inventory,
                        audit_hashes=previous_inventory['artifact_hashes'])
    put(root, 'run_manifest.json', canonical(run_manifest) + b'\n')
    cache = np.load(source / 'embeddings.npz', allow_pickle=False)
    assert cache['values'].dtype == np.float32
    E = dict(zip(cache['texts'].tolist(), cache['values']))
    train_rows = [r for r in rows if r['split'] == 'train']
    train, multiplicities = examples(train_rows)
    assert len(set(multiplicities.values())) == 1
    H = np.stack([E[r['text']] for r in train]).astype(np.float64)
    Y = np.stack([r['target'] for r in train])
    decomp, spectrum = factor(H)
    W = solve(decomp, Y)
    residual = H @ W - Y
    spectrum['normal_equation_residual_inf'] = float(np.max(np.abs(H.T @ residual)))
    spectrum['deduplicated_training_examples'] = len(train)
    spectrum['flattened_training_examples'] = sum(multiplicities.values())
    spectrum['uniform_multiplicity'] = next(iter(multiplicities.values()))
    assert spectrum['normal_equation_residual_inf'] < 1e-7
    groups = {}
    for r in train:
        groups.setdefault((r['family'], r['K']), set()).add(r['scenario'])
    inner_val = set()
    rng = np.random.default_rng(7)
    for key in sorted(groups):
        scenarios = sorted(groups[key]); assert len(scenarios) == 8
        inner_val.update(scenarios[i] for i in rng.permutation(8)[:2])
    val = np.array([r['scenario'] in inner_val for r in train])
    inner, _ = factor(H[~val])
    trials = []
    for lam in protocol['ridge']['lambdas']:
        w = solve(inner, Y[~val], lam)
        trials.append(dict(lam=lam, validation_numeric_mse=float(np.mean((H[val] @ w[:, :4] - Y[val, :4]) ** 2))))
    selected = min(trials, key=lambda r: (r['validation_numeric_mse'], -r['lam']))['lam']
    Wr = solve(decomp, Y, selected)
    weights = dict(candidate_OLS=W, candidate_ridge=Wr)
    put(root, 'candidate_spectrum.json', canonical(spectrum) + b'\n')
    inner_ids = dict(train_scenarios=sorted({r['scenario'] for r in train} - inner_val),
                     validation_scenarios=sorted(inner_val), n_train=int((~val).sum()), n_validation=int(val.sum()))
    put(root, 'ridge_inner_split.json', canonical(inner_ids) + b'\n')
    old_weights = load_file(str(audit_root / 'candidate_probe.safetensors'))['weight'][:5]
    previous = {r['id']: r for r in map(json.loads, (audit_root / 'predictions.jsonl').read_text().splitlines())}
    candidate_predictions = []
    candidate_recovery = {}
    for split in ('train', 'unseen_wording', 'unseen_criterion', 'held_family'):
        rr = [r for r in rows if r['split'] == split]
        ds, _ = examples(rr)
        values = np.stack([E[r['text']] for r in ds]).astype(np.float64)
        pred = dict(OLS=values @ W, ridge=values @ Wr,
                    SGD=np.stack([old_weights @ E[r['text']] for r in ds]))
        report = {a: dict(per_slot=recovery(ds, p), per_named_attribute={}) for a, p in pred.items()}
        for family in sorted({r['family'] for r in ds}):
            take = [i for i, r in enumerate(ds) if r['family'] == family]
            for arm, values in pred.items():
                stats = recovery([ds[i] for i in take], values[take])
                for j, (name, _) in enumerate(CATALOG[family]):
                    report[arm]['per_named_attribute'][f'{family}/{name}'] = stats[j]
        candidate_recovery[split] = report
        for r in rr:
            v = direction(criteria[r['criterion_id']])
            rec = {k: r[k] for k in ('id', 'scenario_id', 'split', 'family', 'K', 'criterion_id')}
            rec['gold'] = r['scores']; rec['supported'] = bool(np.linalg.norm(v[[2, 3]]) == 0)
            rec['scores'] = dict(OLS=[float((E[t].astype(np.float64) @ W) @ v) for t in r['candidates']],
                                 ridge=[float((E[t].astype(np.float64) @ Wr) @ v) for t in r['candidates']],
                                 SGD=previous[r['id']]['scores']['text_candidate_oracle_criterion'])
            for arm in ('OLS', 'ridge', 'SGD'):
                rec[arm] = metrics(rec['scores'][arm], r['scores'])
            candidate_predictions.append(rec)
    put(root, 'candidate_predictions.jsonl', b''.join(canonical(r) + b'\n' for r in candidate_predictions))
    candidate_stats = {s: bins([r for r in candidate_predictions if r['split'] == s],
                              lambda rs: decision_summary(rs, ('OLS', 'ridge', 'SGD')))
                       for s in ('train', 'unseen_wording', 'unseen_criterion', 'held_family')}
    supported_held = decision_summary([r for r in candidate_predictions if r['split'] == 'held_family' and r['supported']], ('OLS', 'ridge', 'SGD'))

    supported = sorted([c for c in cs if c['role'] == 'train'], key=lambda c: c['id'])
    assert len(supported) == 20
    centroids = np.array([np.mean([E[q] for q in c['forms'][:2]], axis=0) for c in supported])
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    Q = np.stack([E[q] for c in supported for q in c['forms'][:2]]).astype(np.float64)
    T = np.stack([direction(c) for c in supported for _ in range(2)])
    original_decomp, original_spectrum = factor(Q)
    original_w = solve(original_decomp, T)
    weights['criterion_original_OLS'] = original_w
    query_predictions, criterion_decisions = [], []
    folds = {}
    for fold in (-1, 0, 1, 2, 3):
        if fold == -1:
            w, spec = original_w, original_spectrum
            train_strings = [q for c in supported for q in c['forms'][:2]]
            test_strings = [q for c in supported for q in c['forms'][2:]]
        else:
            train_strings = [q for c in supported for j, q in enumerate(c['forms']) if j != fold]
            test_strings = [c['forms'][fold] for c in supported]
            q = np.stack([E[t] for t in train_strings]).astype(np.float64)
            target = np.stack([direction(c) for c in supported for j in range(4) if j != fold])
            d, spec = factor(q)
            w = solve(d, target)
            weights[f'criterion_fold{fold}'] = w
        assert not set(train_strings) & set(test_strings)
        assert np.count_nonzero(w[:, 2:4]) == 0
        label = 'original_two_wordings' if fold == -1 else str(fold)
        fold_queries = []
        for c in supported:
            for query in c['forms']:
                if query not in test_strings:
                    continue
                true = direction(c); predicted = E[query].astype(np.float64) @ w
                cosine = float(predicted @ true / (np.linalg.norm(predicted) * np.linalg.norm(true)))
                active = true != 0
                sign = np.where(np.abs(predicted[[0, 1, 4]]) <= 1e-8, 0., np.sign(predicted[[0, 1, 4]]))
                qr = dict(fold=label, criterion_id=c['id'], family=c['family'], query=query,
                          oracle=true.tolist(), predicted=predicted.tolist(), cosine=cosine,
                          vector_mse=float(np.mean((predicted - true) ** 2)),
                          nonzero_sign_accuracy=float(np.mean(np.sign(predicted[active]) == np.sign(true[active]))),
                          supported_ternary_sign_accuracy=float(np.mean(sign == np.sign(true[[0, 1, 4]]))))
                if fold == -1:
                    nearest = supported[int(np.argmax(centroids @ E[query]))]
                    qr['retrieved_id'] = nearest['id']; qr['semantic_id_correct'] = nearest['id'] == c['id']
                    qr['oracle_direction_correct'] = bool(np.array_equal(direction(nearest), true))
                fold_queries.append(qr); query_predictions.append(qr)
                for r in rows:
                    if r['criterion_id'] != c['id'] or r['criterion'] != query:
                        continue
                    rec = {k: r[k] for k in ('id', 'family', 'K', 'criterion_id')}
                    rec['fold'] = label; rec['gold'] = r['scores']; rec['scores'] = dict(OLS=(numeric(r) @ predicted).tolist())
                    rec['OLS'] = metrics(rec['scores']['OLS'], r['scores'])
                    if fold == -1:
                        rec['scores']['centroid'] = (numeric(r) @ direction(nearest)).tolist()
                        rec['scores']['SGD'] = previous[r['id']]['scores']['oracle_candidate_text_criterion']
                        for arm in ('centroid', 'SGD'):
                            rec[arm] = metrics(rec['scores'][arm], r['scores'])
                    criterion_decisions.append(rec)
        arms = ('OLS', 'centroid', 'SGD') if fold == -1 else ('OLS',)
        fold_decisions = [r for r in criterion_decisions if r['fold'] == label]
        folds[label] = dict(train_strings=train_strings, test_strings=test_strings, spectrum=spec,
                            queries=len(fold_queries), mean_cosine=float(np.mean([r['cosine'] for r in fold_queries])),
                            vector_mse=float(np.mean([r['vector_mse'] for r in fold_queries])),
                            nonzero_sign_accuracy=float(np.mean([r['nonzero_sign_accuracy'] for r in fold_queries])),
                            supported_ternary_sign_accuracy=float(np.mean([r['supported_ternary_sign_accuracy'] for r in fold_queries])),
                            decisions=bins(fold_decisions, lambda rs: decision_summary(rs, arms)))
    put(root, 'criterion_queries.jsonl', b''.join(canonical(r) + b'\n' for r in query_predictions))
    put(root, 'criterion_predictions.jsonl', b''.join(canonical(r) + b'\n' for r in criterion_decisions))
    original_queries = [r for r in query_predictions if r['fold'] == 'original_two_wordings']
    centroid_accuracy = float(np.mean([r['semantic_id_correct'] for r in original_queries]))
    loo_top1 = float(np.mean([folds[str(i)]['decisions']['OLS']['top1'] for i in range(4)]))
    loo_cosine = float(np.mean([folds[str(i)]['mean_cosine'] for i in range(4)]))
    criterion_stats = dict(folds=folds, centroid_semantic_id_accuracy=centroid_accuracy,
                           centroid_direction_accuracy=float(np.mean([r['oracle_direction_correct'] for r in original_queries])),
                           leave_one_wording_out_macro_top1=loo_top1,
                           leave_one_wording_out_macro_cosine=loo_cosine,
                           leave_one_wording_out_micro=decision_summary([r for r in criterion_decisions if r['fold'] != 'original_two_wordings'], ('OLS',)))

    swap_rows = list(map(json.loads, (audit_root / 'swaps.jsonl').read_text().splitlines()))
    causal = {}
    for scope in ('all', 'supported_both_criteria'):
        use = swap_rows if scope == 'all' else [s for s in swap_rows if previous[s['before_id']]['criterion_supported'] and previous[s['after_id']]['criterion_supported']]
        causal[scope] = {s: bins([r for r in use if r['split'] == s], swap_summary)
                         for s in ('unseen_wording', 'unseen_criterion', 'held_family')
                         if any(r['split'] == s for r in use)}
    gate_candidate = bool(candidate_stats['train']['OLS']['top1'] >= .95 and
                          all(r['R2'] >= .95 for r in candidate_recovery['train']['OLS']['per_slot']))
    gate_transfer = bool(candidate_stats['unseen_wording']['ridge']['top1'] >= .95 and supported_held['ridge']['top1'] >= .95)
    gates = dict(candidate_linear_ceiling_high=gate_candidate, candidate_supported_transfer=gate_transfer,
                 centroid_high=centroid_accuracy >= .95, criterion_OLS_high=loo_top1 >= .95 and loo_cosine >= .99)
    do_resplit = all(gates.values())
    split_plan = resplit(rows, criteria)
    put(root, 'support_complete_split.json', canonical(split_plan) + b'\n')
    conditional = dict(ran=False, reason='supported_span_gates_failed', B_STEF_allowed=False)
    if do_resplit:
        conditional = run_conditional(root, rows, split_plan, E)
    np.savez_compressed(root / 'linear_weights.npz', **weights)
    result = dict(candidate_spectrum=spectrum, candidate_recovery=candidate_recovery,
                  candidate_decisions=candidate_stats, supported_held_candidate_decisions=supported_held,
                  ridge=dict(selected_lambda=selected, trials=trials, inner_split=inner_ids),
                  criterion=criterion_stats, causal=causal, gates=gates,
                  conditional_resplit=conditional, support_complete_split_audit={k: v for k, v in split_plan.items() if k != 'ids'},
                  B_STEF_allowed=conditional['B_STEF_allowed'], neural_retraining=do_resplit,
                  artifact_hashes={p.name: sha(p) for p in root.iterdir() if p.is_file()})
    put(root, 'results.json', canonical(result) + b'\n')
    print('candidate_rank', spectrum['rank'], 'condition', spectrum['condition_number'], flush=True)
    print('candidate_train_R2', [r['R2'] for r in candidate_recovery['train']['OLS']['per_slot']], flush=True)
    print('candidate_top1', {s: {a: v[a]['top1'] for a in ('OLS', 'ridge', 'SGD')} for s, v in candidate_stats.items()}, flush=True)
    print('criterion_centroid', centroid_accuracy, 'original_OLS', folds['original_two_wordings']['decisions']['OLS']['top1'],
          'LOO', loo_top1, 'cosine', loo_cosine, flush=True)
    print(json.dumps(dict(gates=gates, conditional_resplit=conditional)), flush=True)


def run_conditional(root, rows, plan, E):
    import torch
    from safetensors.torch import save_file
    torch.set_num_threads(1)
    by_id = {r['id']: r for r in rows}
    train = [by_id[i] for i in plan['ids']['train']]
    model, losses = fit(train, E, 32)
    save_file(model.state_dict(), str(root / 'support_complete_cbf32.safetensors'))
    A, B = model.criterion.weight.detach().numpy(), model.candidate.weight.detach().numpy()
    summaries, predictions = {}, []
    for split in ('unseen_wording', 'unseen_direction', 'held_family'):
        records = []
        for ident in plan['ids'][split]:
            r = by_id[ident]; q = E[r['criterion']]
            scores = [score_one(A, B, q, E[t]) for t in r['candidates']]
            moved = [score_one(A, B, q, E[t]) for t in reversed(r['candidates'])]
            assert scores == list(reversed(moved))
            rec = {k: r[k] for k in ('id', 'criterion_id', 'family', 'K')}
            rec['split'] = split; rec['scores'] = scores; rec['gold'] = r['scores']
            rec['cbf32'] = metrics(scores, r['scores'])
            rec['cosine'] = metrics([float(q @ E[t]) for t in r['candidates']], r['scores'])
            records.append(rec); predictions.append(rec)
        summary = bins(records, lambda rs: decision_summary(rs, ('cbf32', 'cosine')))
        summary['paired_CI'] = paired_ci(records, 'cbf32', 'cosine')
        summary['family_deltas'] = {f: v['cbf32']['top1'] - v['cosine']['top1'] for f, v in summary['per_family'].items()}
        summaries[split] = summary
    passed = (all(summaries[s]['paired_CI']['ci95'][0] > 0 for s in ('unseen_direction', 'held_family')) and
              sum(d > 0 for d in summaries['held_family']['family_deltas'].values()) >= 2 and
              summaries['held_family']['paired_CI']['delta'] > 0)
    put(root, 'support_complete_predictions.jsonl', b''.join(canonical(r) + b'\n' for r in predictions))
    return dict(ran=True, summaries=summaries, loss_curve=losses,
                exact_permutation=True, B_STEF_allowed=bool(passed), B_STEF_launched=False)


if __name__ == '__main__':
    main()
