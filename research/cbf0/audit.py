"""CBF-1: fixed oracle basis, two privileged linear probes, causal swaps."""
import argparse
import hashlib
import json
import subprocess
from itertools import permutations
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from build import CATALOG, canonical
from run import CONFIG, load, metrics, paired_ci, score_one

PROTOCOL = Path(__file__).with_name('audit_protocol.json')
ARMS = ('oracle_oracle', 'text_candidate_oracle_criterion',
        'oracle_candidate_text_criterion', 'cbf32', 'cosine', 'blind32', 'blind64')


def put(root, name, payload):
    path = root / name
    if path.exists():
        raise RuntimeError(f'Refusing to overwrite audit evidence: {path}')
    path.write_bytes(payload)


def numeric(row):
    attrs = [a for a, _ in CATALOG[row['family']]]
    return np.array([[*[f[a] / 100 for a in attrs], 1.] for f in row['facts']],
                    dtype=np.float64)


def direction(criterion):
    attrs = [a for a, _ in CATALOG[criterion['family']]]
    out = np.zeros(5, dtype=np.float64)
    out[attrs.index(criterion['attribute'])] = criterion['direction']
    out[4] = float(criterion['direction'] == -1)
    return out


def oracle_sanity(criteria, rows):
    by_id = {c['id']: c for c in criteria}
    maximum = 0.
    for r in rows:
        scores = numeric(r) @ direction(by_id[r['criterion_id']])
        maximum = max(maximum, float(np.max(np.abs(scores - r['scores']))))
        m = metrics(scores, r['scores'])
        assert m['top1'] == m['pairwise'] == 1, r['id']
    assert maximum <= 1e-12
    return dict(rows=len(rows), maximum_score_error=maximum, gate0=True)


def probe(inputs, targets, protocol, name):
    config = protocol['hybrid_training']
    torch.manual_seed(config['seed'])
    model = nn.Linear(inputs.shape[1], protocol['width'], bias=False)
    with torch.no_grad():
        model.weight[5:].zero_()
    x = torch.tensor(inputs, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])
    generator = torch.Generator().manual_seed(config['seed'])
    losses = []
    for epoch in range(config['epochs']):
        order = torch.randperm(len(x), generator=generator)
        total = 0.
        for start in range(0, len(x), config['batch_size']):
            take = order[start:start + config['batch_size']]
            pred = nn.functional.linear(x[take], model.weight[:5])
            loss = (pred - y[take]).square().mean()
            optimizer.zero_grad()
            loss.backward()
            if epoch == start == 0:
                assert model.weight.grad[:5].abs().max() > 0
            optimizer.step()
            total += float(loss.detach()) * len(take)
        losses.append(total / len(x))
        if epoch % 50 == 0 or epoch == config['epochs'] - 1:
            print(f'{name} epoch={epoch} vector_mse={losses[-1]:.6f}', flush=True)
    assert torch.count_nonzero(model.weight[5:]) == 0
    return model.eval(), losses


def summarize(records):
    out = dict(n=len(records))
    for arm in ARMS:
        out[arm] = {}
        for key in ('top1', 'pairwise', 'spearman', 'kendall', 'pearson'):
            vals = [r[arm][key] for r in records if r[arm][key] is not None]
            out[arm][key] = float(np.mean(vals)) if vals else None
            out[arm][key + '_undefined'] = len(records) - len(vals)
        if arm not in ('cosine', 'oracle_oracle'):
            out[arm + '_vs_cosine'] = paired_ci(records, arm, 'cosine')
    return out


def ratio(num, den):
    return num / den if den else None


def swap_summary(rows):
    changed = sum(r['teacher_changed'] for r in rows)
    out = dict(n=len(rows), teacher_changed=changed,
               teacher_change_rate=ratio(changed, len(rows)))
    for arm in ARMS:
        values = [r['arms'][arm] for r in rows]
        actual = sum(v['student_changed'] for v in values)
        response = sum(r['teacher_changed'] and v['student_changed']
                       for r, v in zip(rows, values))
        new_correct = sum(r['teacher_changed'] and v['changed_to_new_teacher']
                          for r, v in zip(rows, values))
        both = sum(r['teacher_changed'] and v['both_winners_correct']
                   for r, v in zip(rows, values))
        stable_changes = actual - response
        out[arm] = dict(student_changed=actual, student_change_rate=ratio(actual, len(rows)),
                        CRA=ratio(sum(v['student_changed'] == r['teacher_changed']
                                      for r, v in zip(rows, values)), len(rows)),
                        changed_given_teacher_change=ratio(response, changed),
                        changed_to_new_teacher_given_teacher_change=ratio(new_correct, changed),
                        both_winners_correct_given_teacher_change=ratio(both, changed),
                        changed_given_teacher_stable=ratio(stable_changes, len(rows) - changed),
                        response_count=response, correct_new_count=new_correct,
                        both_correct_count=both, stable_change_count=stable_changes)
    return out


def swaps(records):
    groups = {}
    for r in records:
        if r['split'] != 'train':
            groups.setdefault(r['scenario_id'], []).append(r)
    out = []
    for sid, rs in groups.items():
        assert all(r['gold'] is not None for r in rs)
        for before, after in permutations(rs, 2):
            if before['criterion_id'] == after['criterion_id']:
                continue
            t0, t1 = int(np.argmax(before['gold'])), int(np.argmax(after['gold']))
            row = dict(scenario_id=sid, before_id=before['id'], after_id=after['id'],
                       before_criterion=before['criterion_id'], after_criterion=after['criterion_id'],
                       split=before['split'], family=before['family'], K=before['K'],
                       teacher_before=t0, teacher_after=t1, teacher_changed=t0 != t1, arms={})
            for arm in ARMS:
                s0 = int(np.argmax(before['scores'][arm]))
                s1 = int(np.argmax(after['scores'][arm]))
                row['arms'][arm] = dict(before=s0, after=s1, student_changed=s0 != s1,
                                        changed_to_new_teacher=s0 != s1 and s1 == t1,
                                        both_winners_correct=s0 == t0 and s1 == t1)
            out.append(row)
    return out


def support(criteria, rows):
    by_id = {c['id']: c for c in criteria}
    train_ids = sorted({r['criterion_id'] for r in rows if r['split'] == 'train'})
    values = np.array([direction(by_id[i]) for i in train_ids])
    _, singular, vh = np.linalg.svd(values, full_matrices=False)
    rank = int(sum(singular > 1e-10))
    basis = vh[:rank]
    residuals = {}
    for c in criteria:
        v = direction(c)
        residuals[c['id']] = float(np.linalg.norm(v - (v @ basis.T) @ basis))
    return dict(train_target_span_rank=rank, active_dimensions=5, padded_dimensions=32,
                train_slots=[0, 1], candidate_supervised_slots=[0, 1, 2, 3],
                unsupported=residuals,
                caveat='Unsupported criterion directions cannot identify an encoder bottleneck from this probe alone.')


def adequacy(splits, arm):
    parts = [splits[s][arm] for s in ('unseen_criterion', 'held_family')]
    parts += [v[arm] for v in splits['held_family']['per_family'].values()]
    return all(p['top1'] >= .95 and p['pairwise'] >= .95 for p in parts)


def audit(protocol, root, stage):
    source = Path(protocol['source_root'])
    manifest, criteria, rows = load(source)
    sanity = oracle_sanity(criteria, rows)
    if stage == 'sanity':
        print(json.dumps(sanity), flush=True)
        return
    root.mkdir(parents=True, exist_ok=True)
    pinned = json.loads((Path(__file__).with_name('result_manifest.json')).read_text())
    for name, expected in pinned['artifact_hashes'].items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == expected, name
    run_manifest = dict(protocol=protocol, source_manifest=manifest,
                        source_hashes=pinned['artifact_hashes'],
                        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
                        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    put(root, 'run_manifest.json', canonical(run_manifest) + b'\n')
    torch.set_num_threads(1)
    cache = np.load(source / 'embeddings.npz', allow_pickle=False)
    assert cache['values'].dtype == np.float32
    E = dict(zip(cache['texts'].tolist(), cache['values']))
    by_id = {c['id']: c for c in criteria}
    q_inputs, x_inputs, q_targets, x_targets = [], [], [], []
    for r in rows:
        if r['split'] != 'train':
            continue
        for text, phi in zip(r['candidates'], numeric(r)):
            q_inputs.append(E[r['criterion']]); x_inputs.append(E[text])
            q_targets.append(direction(by_id[r['criterion_id']])); x_targets.append(phi)
    candidate, candidate_loss = probe(np.stack(x_inputs), np.stack(x_targets), protocol, 'candidate')
    criterion, criterion_loss = probe(np.stack(q_inputs), np.stack(q_targets), protocol, 'criterion')
    save_file(candidate.state_dict(), str(root / 'candidate_probe.safetensors'))
    save_file(criterion.state_dict(), str(root / 'criterion_probe.safetensors'))
    B = candidate.weight.detach().numpy()[:5]
    A = criterion.weight.detach().numpy()[:5]
    base = load_file(str(source / 'm32.safetensors'))
    A0, B0 = base['criterion.weight'].numpy(), base['candidate.weight'].numpy()
    previous = {r['id']: r for r in map(json.loads, (source / 'predictions.jsonl').read_text().splitlines())}
    train_queries = sorted({r['criterion'] for r in rows if r['split'] == 'train'})
    b32 = np.mean([A0 @ E[q] for q in train_queries], axis=0)
    base64 = load_file(str(source / 'm64.safetensors'))
    a64, b64 = base64['criterion.weight'].numpy(), base64['candidate.weight'].numpy()
    blind64 = np.mean([a64 @ E[q] for q in train_queries], axis=0)
    records = []
    mse = {}
    permutation_checks = 0
    for r in rows:
        phi, psi = numeric(r), direction(by_id[r['criterion_id']])
        cq = A @ E[r['criterion']]
        zx = np.array([B @ E[text] for text in r['candidates']])
        rec = {k: r[k] for k in ('id', 'scenario_id', 'split', 'criterion_id', 'family', 'K')}
        rec['gold'] = r['scores']
        rec['criterion_supported'] = np.linalg.norm(psi[[2, 3]]) == 0
        rec['scores'] = dict(oracle_oracle=(phi @ psi).tolist(),
                            text_candidate_oracle_criterion=(zx.astype(np.float64) @ psi).tolist(),
                            oracle_candidate_text_criterion=(phi @ cq.astype(np.float64)).tolist(),
                            cbf32=[score_one(A0, B0, E[r['criterion']], E[text]) for text in r['candidates']],
                            cosine=[float(E[r['criterion']] @ E[text]) for text in r['candidates']],
                            blind32=[float(np.sum(b32 * (B0 @ E[text]), dtype=np.float32)) for text in r['candidates']],
                            blind64=[float(np.sum(blind64 * (b64 @ E[text]), dtype=np.float32)) for text in r['candidates']])
        if r['split'] != 'train':
            for arm in ('cbf32', 'cosine', 'blind32', 'blind64'):
                assert rec['scores'][arm] == previous[r['id']]['scores'][arm], (r['id'], arm)
        for arm in ARMS:
            rec[arm] = metrics(rec['scores'][arm], r['scores'])
        order = list(reversed(range(r['K'])))
        moved_x = np.array([B @ E[r['candidates'][i]] for i in order])
        moved_candidate = (moved_x.astype(np.float64) @ psi).tolist()
        moved_criterion = (phi[order] @ cq.astype(np.float64)).tolist()
        for arm, moved in [('text_candidate_oracle_criterion', moved_candidate),
                           ('oracle_candidate_text_criterion', moved_criterion)]:
            assert list(reversed(moved)) == rec['scores'][arm]
            permutation_checks += 1
        mse.setdefault(r['split'], []).append(dict(
            candidate_coordinates=((zx - phi) ** 2).mean(axis=0).tolist(),
            criterion_coordinates=((cq - psi) ** 2).tolist()))
        records.append(rec)
    put(root, 'predictions.jsonl', b''.join(canonical(r) + b'\n' for r in records))
    splits = {}
    for split in ('train', 'unseen_wording', 'unseen_criterion', 'held_family'):
        rr = [r for r in records if r['split'] == split]
        splits[split] = summarize(rr)
        splits[split]['per_K'] = {str(k): summarize([r for r in rr if r['K'] == k]) for k in (2, 4, 8, 16)}
        splits[split]['per_family'] = {f: summarize([r for r in rr if r['family'] == f]) for f in sorted({r['family'] for r in rr})}
        splits[split]['criterion_support'] = {str(flag): summarize([r for r in rr if r['criterion_supported'] == flag])
                                              for flag in (True, False) if any(r['criterion_supported'] == flag for r in rr)}
    swap_rows = swaps(records)
    put(root, 'swaps.jsonl', b''.join(canonical(r) + b'\n' for r in swap_rows))
    swap_stats = {}
    for split in ('unseen_wording', 'unseen_criterion', 'held_family'):
        rr = [r for r in swap_rows if r['split'] == split]
        swap_stats[split] = swap_summary(rr)
        swap_stats[split]['per_K'] = {str(k): swap_summary([r for r in rr if r['K'] == k]) for k in (2, 4, 8, 16)}
        swap_stats[split]['per_family'] = {f: swap_summary([r for r in rr if r['family'] == f]) for f in sorted({r['family'] for r in rr})}
    basis_support = support(criteria, rows)
    put(root, 'criterion_basis_support.json', canonical(basis_support) + b'\n')
    ca = adequacy(splits, 'text_candidate_oracle_criterion')
    qa = adequacy(splits, 'oracle_candidate_text_criterion')
    if not ca:
        verdict = 'candidate_hybrid_inadequate' if qa else 'both_hybrids_inadequate_no_unique_bottleneck'
    elif not qa:
        verdict = 'criterion_hybrid_inadequate_unsupported_basis_caveat'
    else:
        verdict = 'consistent_with_joint_alignment_or_optimization_failure'
    result = dict(gate0=sanity, candidate_hybrid_adequate=ca, criterion_hybrid_adequate=qa,
                  verdict=verdict, B_STEF_allowed=False, splits=splits, swaps=swap_stats,
                  loss_curves=dict(candidate=candidate_loss, criterion=criterion_loss),
                  vector_mse={s: {side: np.mean([v[side] for v in vs], axis=0).tolist()
                                  for side in ('candidate_coordinates', 'criterion_coordinates')}
                              for s, vs in mse.items()}, basis_support=basis_support,
                  exact_prior_predictions=True, hybrid_permutation_checks=permutation_checks,
                  artifact_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()})
    put(root, 'results.json', canonical(result) + b'\n')
    for split in ('unseen_wording', 'unseen_criterion', 'held_family'):
        print(split, {a: splits[split][a]['top1'] for a in ARMS[:4]}, flush=True)
        print('swaps', split, swap_stats[split]['cbf32'], flush=True)
    print(json.dumps(dict(gate0=sanity, candidate_hybrid_adequate=ca,
                          criterion_hybrid_adequate=qa, verdict=verdict, B_STEF_allowed=False)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('sanity', 'run'), default='sanity')
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    audit(protocol, Path(protocol['output_root']), args.stage)


if __name__ == '__main__':
    main()
