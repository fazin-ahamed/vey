"""Verify CBF-4 from persisted evidence, without encoder forwards or refitting."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from audit import put
from build import canonical
from exact_state_build import AXES, DONORS, PROTOCOL
from exact_state_run import ALL_ARMS, ARMS, STRATA, read_corpus
from run import load


def main():
    p = argparse.ArgumentParser(); p.add_argument('--replay', action='store_true'); p.add_argument('--root', type=Path); args = p.parse_args()
    protocol = json.loads(PROTOCOL.read_text()); root = args.root or Path(protocol['output_root'])
    measured = root / 'score-corrected' if args.replay else root
    cc, ss = read_corpus(root); encoder = json.loads((root / 'encoder_manifest.json').read_text())
    result = json.loads((measured / 'results.json').read_text())
    assert not encoder['loading']['missing_keys'] and not encoder['loading']['mismatched_keys']
    assert encoder['weight_sha256'] == protocol['encoder']['weight_sha256']
    assert encoder['candidate_forwards'] == 0 and encoder['forwards'] == len(encoder['raw_shards'])
    texts = np.load(root / 'texts.npy', allow_pickle=False).tolist(); lookup = {t: i for i, t in enumerate(texts)}
    features = {arm: np.load(root / f'{arm}.npy', allow_pickle=False) for arm in ARMS}
    errors = {arm: 0. for arm in ARMS}; visited = []; tokens = 0
    for shard in encoder['raw_shards']:
        shard_texts = np.load(root / shard.get('texts_file', 'texts.npy'), allow_pickle=False).tolist()
        with np.load(root / shard['file'], allow_pickle=False) as data:
            assert data['hidden'].dtype == np.float32 and data['hidden'].shape[1:] == (2, 384)
            cursor = 0
            for index, length in zip(data['indices'], data['lengths']):
                index, length = int(index), int(length); end = cursor + length
                text = shard_texts[index]
                if text not in lookup:
                    assert text in encoder['superseded_texts']
                    cursor = end
                    continue
                index = lookup[text]; visited.append(index); tokens += length
                offsets = data['offsets'][cursor:end]
                prefix = re.match(r'(?:Choose the option with the |Choose using |Prefer |Aim for )', text, re.I)
                lo = prefix.end() if prefix else 0; hi = len(text.rstrip('.'))
                expected = ~data['special'][cursor:end] & (offsets[:, 0] < hi) & (offsets[:, 1] > lo)
                assert np.array_equal(expected, data['span'][cursor:end])
                for arm, layer, mask in (('final_mean', 1, np.ones(length, bool)),
                                         ('layer1_mean', 0, np.ones(length, bool)), ('layer1_span', 0, expected)):
                    pooled = data['hidden'][cursor:end, layer][mask].mean(0)
                    pooled /= np.linalg.norm(pooled)
                    errors[arm] = max(errors[arm], float(np.max(np.abs(pooled - features[arm][index]))))
                cursor = end
            assert cursor == len(data['hidden'])
    assert sorted(visited) == list(range(len(texts))) and tokens == encoder['tokens']
    assert max(errors.values()) <= 1e-6
    _, _, originals = load(Path(protocol['source_root']))
    source = {r['scenario_id']: r['candidates'] for r in originals}
    exact_fields = 0
    for state in ss:
        for index, text in enumerate(state['candidates']):
            fact = dict((a.strip(), int(v)) for a, v in re.findall(r'([^;]+): (\d+) percent', text.rstrip('.')))
            assert set(fact) == set(AXES)
            for axis, donor in zip(AXES, DONORS):
                original = source[state['donors'][donor]][index]
                parsed = dict((a.strip(), int(v)) for a, v in re.findall(r'([^;]+): (\d+) percent', original.rstrip('.')))
                assert fact[axis] == parsed[axis]; exact_fields += 1
    c_by_id = {c['id']: c for c in cc}; s_by_id = {s['id']: s for s in ss}
    ordered_features = {arm: features[arm][[lookup[c['text']] for c in cc]].astype(np.float64) for arm in ARMS}
    with np.load(root / 'criterion_vectors.npz', allow_pickle=False) as vv:
        assert vv['ids'].tolist() == [c['id'] for c in cc]
        directions = {arm: vv[arm].copy() for arm in ALL_ARMS}
        query_errors = {}
        for arm in ARMS:
            W = np.load(root / f'{arm}_weights.npy', allow_pickle=False); assert W.shape == (384, 4)
            predicted = ordered_features[arm] @ W
            query_errors[arm] = float(np.max(np.abs(predicted - directions[arm])))
            assert np.array_equal(predicted, directions[arm])
    qindex = {c['id']: i for i, c in enumerate(cc)}
    rows = {}; checked_scores = 0; concrete_ties = 0
    for line in (measured / 'decisions.jsonl').read_text().splitlines():
        row = json.loads(line); rows[(row['criterion_id'], row['scenario_id'])] = row
        c, state = c_by_id[row['criterion_id']], s_by_id[row['scenario_id']]
        facts = [tuple(dict((a.strip(), int(v)) for a, v in re.findall(r'([^;]+): (\d+) percent', t.rstrip('.')))[a] for a in AXES) for t in state['candidates']]
        gold = [sum(w * v for w, v in zip(c['weights'], f)) for f in facts]
        teacher = max(range(len(facts)), key=lambda j: (gold[j], facts[j]))
        assert row['teacher_scores'] == gold and row['teacher'] == teacher
        assert row['teacher_top_set'] == [j for j, g in enumerate(gold) if g == max(gold)]
        concrete_ties += int(len(row['teacher_top_set']) > 1)
        for arm in ALL_ARMS:
            v = directions[arm][qindex[c['id']]]; saved = row['arms'][arm]
            if not np.isfinite(v).all():
                assert saved['winner'] is None and saved['scores'] is None and saved['top1'] == 0
                continue
            scores = [(((v[0] * f[0] + v[1] * f[1]) + v[2] * f[2]) + v[3] * f[3]) / 100 for f in facts]
            assert np.array_equal(scores, saved['scores']); checked_scores += len(scores)
            win = max(range(len(facts)), key=lambda j: (scores[j], facts[j]))
            assert saved['winner'] == win and saved['top1'] == int(gold[win] == max(gold))
            assert saved['exact_winner'] == int(win == teacher)
            a, b = np.triu_indices(len(facts), 1)
            pairwise = float(np.mean(np.sign(np.array(scores)[a] - np.array(scores)[b]) == np.sign(np.array(gold)[a] - np.array(gold)[b])))
            assert pairwise == saved['pairwise']
    assert len(rows) == result['decision_rows']
    totals = {s: {arm: dict(n=0, correct_new=0, changed_to_new=0, both_endpoints=0, top_set_new=0) for arm in ALL_ARMS} for s in STRATA}
    swaps = 0; unmoved_correct = {arm: 0 for arm in ALL_ARMS}
    for line in (measured / 'teacher_changing_swaps.jsonl').read_text().splitlines():
        row = json.loads(line); swaps += 1
        before, after = rows[(row['before'], row['scenario_id'])], rows[(row['after'], row['scenario_id'])]
        assert before['teacher'] != after['teacher']
        for arm in ALL_ARMS:
            b, a = before['arms'][arm]['winner'], after['arms'][arm]['winner']
            correct = int(a == after['teacher']); changed = b is not None and a is not None and b != a
            expected = dict(correct_new=correct, changed_to_new=int(correct and changed),
                            both_endpoints=int(b == before['teacher'] and correct), top_set_new=after['arms'][arm]['top1'])
            assert row['arms'][arm] == expected
            unmoved_correct[arm] += int(correct and not changed)
            count = totals[row['stratum']][arm]; count['n'] += 1
            for key, val in expected.items(): count[key] += val
    assert swaps == result['causal']['teacher_changing_pairs']
    for stratum, arms in totals.items():
        for arm, count in arms.items():
            saved = result['causal']['per_stratum'][stratum][arm]; assert saved['n'] == count['n']
            for key in ('correct_new', 'changed_to_new', 'both_endpoints', 'top_set_new'):
                assert saved[key + '_rate'] == count[key] / count['n']
    report = dict(raw_criteria_verified=len(visited), raw_tokens_verified=tokens, raw_pooling_max_errors=errors,
                  source_exact_fields_verified=exact_fields, query_projection_max_errors=query_errors,
                  decision_rows_verified=len(rows), scalar_scores_verified=checked_scores,
                  composed_gold_tie_rows=concrete_ties, teacher_changing_swaps_verified=swaps,
                  correct_new_without_student_change=unmoved_correct, encoder_forwards_during_verification=0,
                  fitting_during_verification=False)
    put(measured, 'verification.json', canonical(report) + b'\n'); print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
