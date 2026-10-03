"""CBF-3: fixed-dimensional cached and frozen-interface diagnostics."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from audit import direction, numeric, put
from build import canonical
from ceiling import bins, examples, factor, recovery, solve
from run import load

PROTOCOL = Path(__file__).with_name('tomography_protocol.json')


def fit_map(H, Y):
    if not np.any(H):
        return np.zeros((H.shape[1], Y.shape[1])), dict(rank=0, shape=list(H.shape), condition_number=None, zero_signal=True)
    d, spectrum = factor(H)
    W = solve(d, Y)
    spectrum['normal_equation_residual_inf'] = float(np.max(np.abs(H.T @ (H @ W - Y))))
    return W, spectrum


def core(scores, gold):
    s, y = np.asarray(scores), np.asarray(gold)
    i, j = np.triu_indices(len(s), 1)
    return dict(top1=int(y[int(np.argmax(s))] == y.max()),
                pairwise=float(np.mean(np.sign(s[i] - s[j]) == np.sign(y[i] - y[j]))))


def summarize(records):
    return dict(n=len(records), top1=float(np.mean([r['top1'] for r in records])),
                pairwise=float(np.mean([r['pairwise'] for r in records])))


def decisions(items, values, rows, criteria):
    lookup = {r['text']: i for i, r in enumerate(items)}
    out = []
    for r in rows:
        scores = values[[lookup[t] for t in r['candidates']]] @ direction(criteria[r['criterion_id']])[:4]
        out.append(dict(id=r['id'], family=r['family'], K=r['K'], **core(scores, r['scores'])))
    return bins(out, summarize)


def all_items(rows):
    return examples(rows)[0]


def difference_rows(items, H, Y):
    groups = {}
    for i, item in enumerate(items):
        groups.setdefault(item['scenario'], []).append(i)
    x, y, pairs = [], [], 0
    for indices in groups.values():
        h, target = H[indices], Y[indices]
        # Subtract a reference first: identical CLS vectors must center to exact zero.
        h = h - h[0]; target = target - target[0]
        K = len(indices)
        x.append((h - h.mean(0)) * np.sqrt(K))
        y.append((target - target.mean(0)) * np.sqrt(K))
        pairs += K * (K - 1) // 2
    return np.concatenate(x), np.concatenate(y), pairs


def candidate_audit(E, rows, criteria, root, name, absolute=None):
    items = all_items(rows)
    indices = {r['text']: i for i, r in enumerate(items)}
    H = np.stack([E[r['text']] for r in items]).astype(np.float64)
    Y = np.stack([r['target'][:4] for r in items])
    train_texts = {t for r in rows if r['split'] == 'train' for t in r['candidates']}
    take = np.array([r['text'] in train_texts for r in items])
    train_items = [r for r, yes in zip(items, take) if yes]
    Wa, sa = fit_map(H[take], Y[take])
    if absolute is not None:
        assert np.max(np.abs(H[take] @ (Wa - absolute))) < 1e-8
        Wa = absolute
    DH, DY, pairs = difference_rows(train_items, H[take], Y[take])
    Wd, sd = fit_map(DH, DY)
    pred = dict(absolute=H @ Wa, difference=H @ Wd)
    result = dict(absolute_spectrum=sa, difference_spectrum=sd, unordered_train_pairs=pairs, splits={})
    for split in ('train', 'unseen_wording', 'unseen_criterion', 'held_family'):
        rr = [r for r in rows if r['split'] == split]
        ds = all_items(rr); inds = [indices[r['text']] for r in ds]
        result['splits'][split] = {}
        for arm, values in pred.items():
            result['splits'][split][arm] = dict(per_slot=recovery(ds, values[inds]),
                                                decisions=decisions(ds, values[inds], rr, criteria))
    hard = result['splits']['held_family']['difference']['decisions']
    parts = [result['splits'][s]['difference']['decisions'] for s in ('unseen_criterion', 'held_family')]
    parts += list(hard['per_family'].values())
    result['difference_adequate'] = all(x['top1'] >= .95 and x['pairwise'] >= .95 for x in parts)
    np.savez(root / f'candidate_{name}.npz', texts=np.array([r['text'] for r in items]),
             absolute_weights=Wa, difference_weights=Wd,
             absolute_attributes=pred['absolute'], difference_attributes=pred['difference'])
    put(root, f'candidate_{name}.json', canonical(result) + b'\n')
    return result


def family_audit(E, rows, criteria, root):
    items = all_items(rows)
    families = sorted({r['family'] for r in rows})
    weights, cvweights, cvpreds, matrix, cv = {}, {}, {}, {}, {}
    split_ids = {}
    for family in families:
        rr = [r for r in rows if r['family'] == family]
        ds = [r for r in items if r['family'] == family]
        H = np.stack([E[r['text']] for r in ds]).astype(np.float64)
        Y = np.stack([r['target'][:4] for r in ds])
        W, spec = fit_map(H, Y); weights[family] = W
        groups = {}
        for r in rr:
            groups.setdefault((r['split'], r['K']), set()).add(r['scenario_id'])
        rng = np.random.default_rng(7)
        fold_by_scenario = {}
        for key in sorted(groups):
            ss = sorted(groups[key])
            for i, index in enumerate(rng.permutation(len(ss))):
                fold_by_scenario[ss[index]] = i % 4
        predicted = np.empty_like(Y)
        split_ids[family] = []
        for fold in range(4):
            test = np.array([fold_by_scenario[r['scenario']] == fold for r in ds])
            assert test.any() and (~test).any()
            wf, _ = fit_map(H[~test], Y[~test]); cvweights[f'{family}_{fold}'] = wf
            predicted[test] = H[test] @ wf
            split_ids[family].append(dict(fold=fold, test_scenarios=sorted({r['scenario'] for r, flag in zip(ds, test) if flag}),
                                          n_train=int((~test).sum()), n_test=int(test.sum())))
        cvpreds[family] = predicted
        cv[family] = dict(training_spectrum=spec, diagnostic_only=True,
                           resubstitution_recovery=recovery(ds, H @ W),
                           resubstitution_decisions=decisions(ds, H @ W, rr, criteria),
                           CV_recovery=recovery(ds, predicted), CV_decisions=decisions(ds, predicted, rr, criteria))
        matrix[family] = {}
        for target in families:
            target_rows = [r for r in rows if r['family'] == target]
            target_items = [r for r in items if r['family'] == target]
            hh = np.stack([E[r['text']] for r in target_items]).astype(np.float64)
            pp = hh @ W
            matrix[family][target] = dict(per_slot=recovery(target_items, pp),
                                          decisions=decisions(target_items, pp, target_rows, criteria),
                                          self_resubstitution=family == target)
    high = [f for f, x in cv.items() if x['CV_decisions']['top1'] >= .95 and all(y['R2'] >= .95 for y in x['CV_recovery'])]
    result = dict(diagnostic_only=True, families=families, CV=cv, transfer_matrix=matrix,
                  high_CV_families=high, previously_held_high_CV_families=[f for f in high if f in ('compliance', 'similarity', 'intent-routing')])
    np.savez(root / 'family_weights.npz', **weights, **cvweights,
             **{f'CV_predictions_{f}': v for f, v in cvpreds.items()})
    put(root, 'family_CV_ids.json', canonical(split_ids) + b'\n')
    put(root, 'family_results.json', canonical(result) + b'\n')
    return result


def nuisance(E, donors):
    means = np.array([np.mean([E[c['forms'][t]].astype(np.float64) for c in donors], axis=0) for t in range(4)])
    contrasts = means - means.mean(0)
    _, s, vt = np.linalg.svd(contrasts, full_matrices=False)
    cutoff = np.finfo(np.float64).eps * max(contrasts.shape) * s[0]
    basis = vt[s > cutoff]
    assert len(basis) <= 3
    return basis, dict(rank=len(basis), singular_values=s.tolist(), donor_ids=[c['id'] for c in donors],
                       donor_strings=[q for c in donors for q in c['forms']])


def criterion_audit(E, rows, cs, root, name, template=False):
    supported = sorted([c for c in cs if c['role'] == 'train'], key=lambda c: c['id'])
    families = sorted({c['family'] for c in supported})
    shuffled = np.random.default_rng(7).permutation(families).tolist()
    groups = [set(shuffled[:5]), set(shuffled[5:])] if template else [set(families)]
    queries, decisions_rows, stats, saved = [], [], {}, {}
    for group, recipient_families in enumerate(groups):
        recipient = [c for c in supported if c['family'] in recipient_families]
        donors = [c for c in supported if c['family'] not in recipient_families]
        basis, info = nuisance(E, donors) if template else (np.empty((0, 384)), {})
        stats[str(group)] = info
        saved[f'projection_{group}'] = basis
        qE = {}
        for c in supported:
            for q in c['forms']:
                h = E[q].astype(np.float64)
                qE[q] = h - (h @ basis.T) @ basis
        centers = np.array([np.mean([qE[q] for q in c['forms'][:2]], axis=0) for c in supported])
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        for fold in (-1, 0, 1, 2, 3):
            selected = lambda j: j < 2 if fold == -1 else j != fold
            train = [q for c in supported for j, q in enumerate(c['forms']) if selected(j)]
            H = np.stack([qE[q] for q in train])
            Y = np.stack([direction(c) for c in supported for j in range(4) if selected(j)])
            W, spectrum = fit_map(H, Y)
            saved[f'weights_{group}_{fold}'] = W
            label = 'original' if fold == -1 else str(fold)
            stats[str(group)][label] = dict(train_strings=train, spectrum=spectrum)
            for c in recipient:
                for j, query in enumerate(c['forms']):
                    if (j < 2 if fold == -1 else j != fold):
                        continue
                    if template:
                        assert query not in info['donor_strings']
                    target = direction(c); pred = qE[query] @ W
                    cosine = float(pred @ target / (np.linalg.norm(pred) * np.linalg.norm(target)))
                    rec = dict(group=group, fold=label, id=c['id'], query=query, predicted=pred.tolist(),
                               cosine=cosine, vector_mse=float(np.mean((pred - target) ** 2)),
                               nonzero_sign_accuracy=float(np.mean(np.sign(pred[target != 0]) == np.sign(target[target != 0]))))
                    if fold == -1:
                        hh = qE[query] / np.linalg.norm(qE[query])
                        match = supported[int(np.argmax(centers @ hh))]
                        rec['retrieved_id'] = match['id']; rec['semantic_ID_correct'] = match['id'] == c['id']
                        rec['direction_correct'] = bool(np.array_equal(direction(match), target))
                    queries.append(rec)
                    for r in rows:
                        if r['criterion_id'] != c['id'] or r['criterion'] != query:
                            continue
                        score = numeric(r) @ pred
                        dr = dict(id=r['id'], group=group, fold=label, family=r['family'], K=r['K'], OLS=core(score, r['scores']))
                        if fold == -1:
                            dr['centroid'] = core(numeric(r) @ direction(match), r['scores'])
                        decisions_rows.append(dr)
    results = {}
    for label in ('original', '0', '1', '2', '3'):
        qq = [q for q in queries if q['fold'] == label]
        dd = [r for r in decisions_rows if r['fold'] == label]
        entry = dict(n_queries=len(qq), cosine=float(np.mean([q['cosine'] for q in qq])),
                     vector_mse=float(np.mean([q['vector_mse'] for q in qq])),
                     nonzero_sign_accuracy=float(np.mean([q['nonzero_sign_accuracy'] for q in qq])),
                     OLS=bins([dict(family=r['family'], K=r['K'], **r['OLS']) for r in dd], summarize))
        if label == 'original':
            entry['semantic_ID_accuracy'] = float(np.mean([q['semantic_ID_correct'] for q in qq]))
            entry['oracle_direction_accuracy'] = float(np.mean([q['direction_correct'] for q in qq]))
            entry['centroid'] = bins([dict(family=r['family'], K=r['K'], **r['centroid']) for r in dd], summarize)
        results[label] = entry
    results['LOO_macro_top1'] = float(np.mean([results[str(f)]['OLS']['top1'] for f in range(4)]))
    results['LOO_macro_cosine'] = float(np.mean([results[str(f)]['cosine'] for f in range(4)]))
    results['adequate'] = (results['original']['semantic_ID_accuracy'] >= .95 and
                           results['LOO_macro_top1'] >= .95 and results['LOO_macro_cosine'] >= .99)
    results['projection_crossfit'] = template; results['fold_metadata'] = stats
    np.savez(root / f'criterion_{name}.npz', **saved)
    put(root, f'criterion_{name}_queries.jsonl', b''.join(canonical(q) + b'\n' for q in queries))
    put(root, f'criterion_{name}_decisions.jsonl', b''.join(canonical(q) + b'\n' for q in decisions_rows))
    put(root, f'criterion_{name}.json', canonical(results) + b'\n')
    return results


def inputs(protocol):
    source = Path(protocol['source_root'])
    manifest, cs, rows = load(source)
    inventory = json.loads(Path(__file__).with_name('ceiling_result_manifest.json').read_text())
    for name, expected in inventory['artifact_hashes'].items():
        assert hashlib.sha256((Path(protocol['ceiling_root']) / name).read_bytes()).hexdigest() == expected
    for name, expected in inventory['run_manifest']['source_hashes'].items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == expected
    return source, manifest, cs, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('cached', 'encode', 'recover', 'layers'), required=True)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    root = Path(protocol['output_root']); root.mkdir(parents=True, exist_ok=True)
    source, manifest, cs, rows = inputs(protocol)
    criteria = {c['id']: c for c in cs}
    if args.stage == 'cached':
        put(root, 'run_manifest.json', canonical(dict(protocol=protocol, source_manifest=manifest,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
            encoder_source_sha256=hashlib.sha256(Path(__file__).with_name('tomography_encode.py').read_bytes()).hexdigest(),
            commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())) + b'\n')
        cache = np.load(source / 'embeddings.npz', allow_pickle=False)
        E = dict(zip(cache['texts'].tolist(), cache['values']))
        old = np.load(Path(protocol['ceiling_root']) / 'linear_weights.npz', allow_pickle=False)['candidate_OLS'][:, :4]
        candidate = candidate_audit(E, rows, criteria, root, 'cached', old)
        family = family_audit(E, rows, criteria, root)
        criterion = criterion_audit(E, rows, cs, root, 'cached')
        projected = criterion_audit(E, rows, cs, root, 'template', template=True)
        result = dict(candidate=candidate, family=family, criterion_baseline=criterion, template_projection=projected,
                      new_encoder_pass_required=not (candidate['difference_adequate'] and projected['adequate']),
                      B_STEF_allowed=False)
        put(root, 'cached_results.json', canonical(result) + b'\n')
        print('Difference top1', {s: v['difference']['decisions']['top1'] for s, v in candidate['splits'].items()}, flush=True)
        print('Family CV top1', {f: v['CV_decisions']['top1'] for f, v in family['CV'].items()}, flush=True)
        print('Template retrieval', projected['original']['semantic_ID_accuracy'], 'LOO', projected['LOO_macro_top1'], flush=True)
        print('New encoder pass required', result['new_encoder_pass_required'], flush=True)
    elif args.stage == 'encode':
        cached = json.loads((root / 'cached_results.json').read_text())
        if not cached['new_encoder_pass_required']:
            print('Encoder capture gated off: cached interfaces adequate', flush=True)
            return
        from tomography_encode import encode
        encode(root, source, protocol)
    elif args.stage == 'recover':
        from tomography_encode import recover_manifest
        recover_manifest(root, source)
    else:
        cached = json.loads((root / 'cached_results.json').read_text())
        if not cached['new_encoder_pass_required']:
            print('Layer sweep gated off: cached interfaces adequate', flush=True)
            return
        meta = json.loads((root / 'encoder_manifest.json').read_text())
        texts = np.load(root / 'texts.npy', allow_pickle=False).tolist()
        candidate_results, criterion_results = {}, {}
        for layer in range(meta['layers']):
            for pool in ('shipped', 'cls', 'content', 'numeric', 'namevalue', 'keyword'):
                name = f'L{layer:02d}_{pool}'
                values = np.load(root / (name + '.npy'), mmap_mode='r')
                E = dict(zip(texts, values))
                if pool != 'keyword':
                    result = candidate_audit(E, rows, criteria, root, name)
                    candidate_results[name] = result
                    print(name, 'candidate held diff', result['splits']['held_family']['difference']['decisions']['top1'], flush=True)
                if pool in ('shipped', 'cls', 'content', 'keyword'):
                    result = criterion_audit(E, rows, cs, root, name)
                    criterion_results[name] = result
                    print(name, 'criterion retrieval', result['original']['semantic_ID_accuracy'], 'LOO', result['LOO_macro_top1'], flush=True)
        final = dict(candidate_interfaces=candidate_results, criterion_interfaces=criterion_results,
                     candidate_adequate=[k for k, v in candidate_results.items() if v['difference_adequate']],
                     criterion_adequate=[k for k, v in criterion_results.items() if v['adequate']],
                     exploratory_scan=True, full_unseen_criterion_transfer_proven=False, B_STEF_allowed=False)
        put(root, 'layer_results.json', canonical(final) + b'\n')
        print('Adequate candidate interfaces', final['candidate_adequate'], flush=True)
        print('Adequate criterion interfaces', final['criterion_adequate'], flush=True)


if __name__ == '__main__':
    main()
