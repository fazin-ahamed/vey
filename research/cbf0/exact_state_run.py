"""CBF-4 train-only criterion fits and exact-state decision/causal gates."""
import argparse
import hashlib
import itertools
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from audit import put
from build import canonical
from ceiling import factor, solve
from exact_state_build import AXES, PROTOCOL, parse, validate
from exact_state_encode import ARMS

ALL_ARMS = ('lexical',) + ARMS
STRATA = ('alias', 'polarity', 'composition', 'held_combination')
CUES = re.compile(r'\b(maximize|minimize|higher|lower|more|less|reward|penalize|positive|negative|highest|lowest|greatest|least|high|low|larger|smaller|adding|subtracting|penalty)\b', re.I)
NEGATIVE = {'minimize', 'lower', 'less', 'penalize', 'negative', 'lowest', 'least', 'low', 'smaller', 'subtracting', 'penalty'}
NUMBER = {'one': 1, 'two': 2, '1': 1, '2': 2}


def lexical(text):
    """Schema-name lookup with generic surface cues; no alias/criterion catalog."""
    text = text.lower()
    matches = sorted((m.start(), m.end(), i) for i, axis in enumerate(AXES)
                     for m in re.finditer(r'\b' + re.escape(axis) + r'\b', text))
    if not matches: return None
    weights = np.zeros(4)
    for j, (lo, hi, axis) in enumerate(matches):
        before = text[matches[j-1][1] if j else 0:lo]
        before = re.split(r';|\.|\band\b|\bwhile\b|together with', before)[-1]
        after = text[hi:matches[j+1][0] if j + 1 < len(matches) else len(text)]
        pre = list(CUES.finditer(before)); post = list(CUES.finditer(after))
        cue = pre[-1][0] if pre else (post[0][0] if post else 'positive')
        magnitude = 1
        lead = re.search(r'\b(one|two|1|2) times\s*$', before)
        trailing = re.search(r'\bweight (one|two|1|2)\b', after)
        if lead: magnitude = NUMBER[lead[1]]
        elif trailing: magnitude = NUMBER[trailing[1]]
        weights[axis] += (-1 if cue in NEGATIVE else 1) * magnitude
    positional = (re.search(r'Use weights (one|two|1|2) and (one|two|1|2) respectively', text, re.I)
                  or re.search(r'first preference a weight of (one|two|1|2) and the second a weight of (one|two|1|2)', text))
    counts = re.search(r'first preference (once|twice) and the second preference (once|twice)', text)
    if len(matches) == 2:
        if positional:
            for j, (_, _, axis) in enumerate(matches): weights[axis] = np.sign(weights[axis]) * NUMBER[positional[j+1].lower()]
        elif counts:
            for j, (_, _, axis) in enumerate(matches): weights[axis] = np.sign(weights[axis]) * (1 if counts[j+1] == 'once' else 2)
        elif 'matters twice as much as' in text:
            weights[matches[0][2]] *= 2
    elif len(matches) == 1 and 'assign double weight' in text:
        weights *= 2
    return weights


def score(direction, texts):
    if direction is None: return None
    phi = np.array([parse(t) for t in texts], dtype=np.float64)
    # Sum in exact percent units before normalization: integer directions retain ties.
    return (((direction[0] * phi[:, 0] + direction[1] * phi[:, 1]) +
             direction[2] * phi[:, 2]) + direction[3] * phi[:, 3]) / 100


def winner(scores, facts):
    if scores is None: return None
    return max(range(len(facts)), key=lambda i: (scores[i], tuple(facts[i])))


def read_corpus(root):
    corpus = json.loads((root / 'corpus_manifest.json').read_text())
    assert corpus['protocol_sha256'] == hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    for name, h in corpus['files'].items(): assert hashlib.sha256((root / name).read_bytes()).hexdigest() == h
    cc = json.loads((root / 'criteria.json').read_text()); ss = json.loads((root / 'states.json').read_text())
    assert validate(cc, ss) == corpus['checks']
    return cc, ss


def fit_maps(root, cc, protocol):
    texts = np.load(root / 'texts.npy', allow_pickle=False).tolist()
    lookup = {t: i for i, t in enumerate(texts)}
    assert set(texts) == {c['text'] for c in cc}
    Y = np.array([c['weights'] for c in cc], dtype=np.float64)
    train = np.array([c['stratum'] == 'train' for c in cc])
    templates = np.array([int(c['template'].split('/')[-1]) for c in cc])
    grid = protocol['fit']['ridge_grid']; details = {}; vectors = {}
    lv = [lexical(c['text']) for c in cc]
    vectors['lexical'] = np.array([v if v is not None else np.full(4, np.nan) for v in lv])
    for arm in ARMS:
        H = np.load(root / f'{arm}.npy', allow_pickle=False)[[lookup[c['text']] for c in cc]].astype(np.float64)
        assert H.shape == (len(cc), 384) and np.isfinite(H).all()
        losses = np.zeros((6, len(grid)))
        for t in range(6):
            fitting = train & (templates != t); validation = train & (templates == t)
            dec, _ = factor(H[fitting])
            for j, lam in enumerate(grid):
                W = solve(dec, Y[fitting], lam)
                losses[t, j] = np.mean((H[validation] @ W - Y[validation]) ** 2)
        best = min(range(len(grid)), key=lambda j: (float(losses[:, j].mean()), -grid[j]))
        lam = grid[best]; dec, spectrum = factor(H[train]); W = solve(dec, Y[train], lam)
        vectors[arm] = H @ W
        np.save(root / f'{arm}_weights.npy', W)
        details[arm] = dict(ridge=lam, template_fold_MSE=losses.tolist(), mean_fold_MSE=losses.mean(0).tolist(),
                            training_coefficient_MSE=float(np.mean((vectors[arm][train] - Y[train]) ** 2)),
                            training_spectrum=spectrum, map_shape=list(W.shape), active_parameters=int(W.size),
                            test_queries_used_for_fit_or_selection=0)
    np.savez(root / 'criterion_vectors.npz', ids=np.array([c['id'] for c in cc]), gold=Y, **vectors)
    put(root, 'fit.json', canonical(details) + b'\n')
    return vectors, details


def direction(vectors, arm, index):
    value = vectors[arm][index]
    return value if np.isfinite(value).all() else None


def evaluate(root, cc, ss, vectors):
    records = []; cached = {}; rng = np.random.default_rng(7)
    permutations = {arm: dict(comparisons=0, numeric_comparisons=0, abstention_comparisons=0,
                              score_mismatches=0, winner_mismatches=0, maximum_score_error=0.) for arm in ALL_ARMS}
    for i, c in enumerate(cc):
        if c['stratum'] == 'train': continue
        source = 'unseen_wording' if c['stratum'] in ('alias', 'polarity') else 'unseen_criterion'
        for s in ss:
            if s['source_split'] != source: continue
            facts = [parse(t) for t in s['candidates']]
            gold = np.array(facts) @ np.array(c['weights'], dtype=np.int64)
            teacher = winner(gold, facts); a, b = np.triu_indices(s['K'], 1)
            row = dict(criterion_id=c['id'], semantic_id=c['semantic_id'], stratum=c['stratum'],
                       scenario_id=s['id'], K=s['K'], teacher=teacher,
                       teacher_top_set=np.flatnonzero(gold == gold.max()).tolist(), teacher_scores=gold.tolist(), arms={})
            orders = [np.arange(s['K'])[::-1], rng.permutation(s['K'])]
            for arm in ALL_ARMS:
                v = direction(vectors, arm, i); pred = score(v, s['candidates']); win = winner(pred, facts)
                row['arms'][arm] = dict(winner=win, top1=int(win is not None and gold[win] == gold.max()),
                                       exact_winner=int(win == teacher), covered=win is not None,
                                       pairwise=float(np.mean(np.sign(pred[a] - pred[b]) == np.sign(gold[a] - gold[b]))) if pred is not None else 0.,
                                       scores=pred.tolist() if pred is not None else None)
                for order in orders:
                    pp = score(v, [s['candidates'][j] for j in order]); pf = [facts[j] for j in order]
                    pw = winner(pp, pf); restored_win = int(order[pw]) if pw is not None else None
                    check = permutations[arm]; check['comparisons'] += 1
                    check['winner_mismatches'] += int(restored_win != win)
                    if pp is None:
                        check['abstention_comparisons'] += 1
                    else:
                        restored = pp[np.argsort(order)]; check['numeric_comparisons'] += 1
                        check['score_mismatches'] += int(not np.array_equal(restored, pred))
                        check['maximum_score_error'] = max(check['maximum_score_error'], float(np.max(np.abs(restored - pred))))
            records.append(row); cached[(c['id'], s['id'])] = row
    put(root, 'decisions.jsonl', b''.join(canonical(r) + b'\n' for r in records))
    return records, cached, permutations


def aggregate(records):
    return {arm: dict(n=len(records), top1=float(np.mean([r['arms'][arm]['top1'] for r in records])),
                      exact_winner=float(np.mean([r['arms'][arm]['exact_winner'] for r in records])),
                      pairwise=float(np.mean([r['arms'][arm]['pairwise'] for r in records])),
                      coverage=float(np.mean([r['arms'][arm]['covered'] for r in records]))) for arm in ALL_ARMS}


def alias_intervals(records, protocol):
    groups = defaultdict(list)
    for r in records:
        if r['stratum'] == 'alias': groups[r['semantic_id']].append(r)
    ids = sorted(groups); rng = np.random.default_rng(protocol['metrics']['bootstrap']['seed'])
    draws = rng.integers(0, len(ids), size=(protocol['metrics']['bootstrap']['repetitions'], len(ids)))
    results = {}
    for arm in ARMS:
        delta = np.array([np.mean([r['arms'][arm]['top1'] - r['arms']['lexical']['top1'] for r in groups[key]]) for key in ids])
        samples = delta[draws].mean(1)
        results[arm] = dict(semantic_direction_clusters=len(ids), cluster_ids=ids, delta=float(delta.mean()),
                           CI95=np.quantile(samples, [0.025, 0.975]).tolist(),
                           simultaneous_CI=np.quantile(samples, [0.05 / (2 * len(ARMS)), 1 - 0.05 / (2 * len(ARMS))]).tolist())
    return results


def reversals(root, cc, ss, vectors, cached):
    lookup = {(c['stratum'], tuple(c['weights']), c['template']): (i, c) for i, c in enumerate(cc)}
    records = []; result = {}
    for stratum in ('polarity', 'alias'):
        pairs = []
        for i, c in enumerate(cc):
            if c['stratum'] != stratum or sum(c['weights']) != 1: continue
            j, opposite = lookup[(stratum, tuple(-w for w in c['weights']), c['template'])]
            axis = next(k for k, w in enumerate(c['weights']) if w)
            row = dict(stratum=stratum, axis=AXES[axis], positive=c['id'], negative=opposite['id'], arms={})
            for arm in ALL_ARMS:
                vp, vn = direction(vectors, arm, i), direction(vectors, arm, j)
                norm = float(np.linalg.norm(vp) * np.linalg.norm(vn)) if vp is not None and vn is not None else 0.
                cos = float(np.dot(vp, vn) / norm) if norm else None
                changes = []
                for state in ss:
                    if state['source_split'] != 'unseen_wording': continue
                    rp, rn = cached[(c['id'], state['id'])], cached[(opposite['id'], state['id'])]
                    if rp['teacher'] != rn['teacher']:
                        wp, wn = rp['arms'][arm]['winner'], rn['arms'][arm]['winner']
                        changes.append(int(wp is not None and wn is not None and wp != wn))
                row['arms'][arm] = dict(cosine=cos, teacher_changing_states=len(changes), student_changes=sum(changes))
            pairs.append(row); records.append(row)
        result[stratum] = {}
        for arm in ALL_ARMS:
            vals = [p['arms'][arm] for p in pairs]; cosines = [v['cosine'] for v in vals if v['cosine'] is not None]
            n = sum(v['teacher_changing_states'] for v in vals)
            result[stratum][arm] = dict(pairs=len(vals), defined_cosines=len(cosines),
                                        negative_cosines=sum(v < 0 for v in cosines),
                                        mean_cosine=float(np.mean(cosines)) if cosines else None,
                                        maximum_cosine=max(cosines) if cosines else None,
                                        teacher_changing_states=n, student_change_rate=sum(v['student_changes'] for v in vals) / n)
    put(root, 'polarity_pairs.json', canonical(records) + b'\n')
    return result


def swaps(root, cc, ss, cached):
    groups = defaultdict(list)
    for c in cc:
        if c['stratum'] != 'train': groups[(c['stratum'], c['template'])].append(c)
    counts = {s: {arm: dict(n=0, correct_new=0, changed_to_new=0, both_endpoints=0, top_set_new=0) for arm in ALL_ARMS} for s in STRATA}
    payload = []; eligible = 0
    for (stratum, template), queries in sorted(groups.items()):
        source = 'unseen_wording' if stratum in ('alias', 'polarity') else 'unseen_criterion'
        for state in ss:
            if state['source_split'] != source: continue
            for first, second in itertools.permutations(queries, 2):
                assert first['ray'] != second['ray']
                eligible += 1
                r1, r2 = cached[(first['id'], state['id'])], cached[(second['id'], state['id'])]
                if r1['teacher'] == r2['teacher']: continue
                row = dict(stratum=stratum, template=template, scenario_id=state['id'],
                           before=first['id'], after=second['id'], teacher_before=r1['teacher'], teacher_after=r2['teacher'], arms={})
                for arm in ALL_ARMS:
                    a, b = r1['arms'][arm]['winner'], r2['arms'][arm]['winner']
                    correct = int(b == r2['teacher']); changed = a is not None and b is not None and a != b
                    flags = dict(correct_new=correct, changed_to_new=int(correct and changed),
                                 both_endpoints=int(a == r1['teacher'] and correct), top_set_new=r2['arms'][arm]['top1'])
                    row['arms'][arm] = flags; count = counts[stratum][arm]; count['n'] += 1
                    for key, value in flags.items(): count[key] += value
                payload.append(row)
    put(root, 'teacher_changing_swaps.jsonl', b''.join(canonical(r) + b'\n' for r in payload))
    def rates(d): return dict(n=d['n'], **{key + '_rate': d[key] / d['n'] for key in ('correct_new', 'changed_to_new', 'both_endpoints', 'top_set_new')})
    pooled = {arm: rates({key: sum(counts[s][arm][key] for s in STRATA) for key in counts[STRATA[0]][arm]}) for arm in ALL_ARMS}
    return dict(eligible_ordered_same_state_pairs=eligible, teacher_changing_pairs=len(payload),
                pooled=pooled, per_stratum={s: {arm: rates(d) for arm, d in aa.items()} for s, aa in counts.items()})


def main():
    p = argparse.ArgumentParser(); p.add_argument('--replay', action='store_true'); p.add_argument('--root', type=Path); args = p.parse_args()
    protocol = json.loads(PROTOCOL.read_text()); evidence = args.root or Path(protocol['output_root'])
    root = evidence / 'score-corrected' if args.replay else evidence
    if (root / 'results.json').exists(): raise RuntimeError('Refusing to overwrite a measured experiment')
    cc, ss = read_corpus(evidence)
    if args.replay:
        with np.load(evidence / 'criterion_vectors.npz', allow_pickle=False) as data:
            assert data['ids'].tolist() == [c['id'] for c in cc]
            vectors = {arm: data[arm].copy() for arm in ALL_ARMS}
        fits = json.loads((evidence / 'fit.json').read_text()); root.mkdir()
    else:
        vectors, fits = fit_maps(root, cc, protocol)
    records, cached, permutations = evaluate(root, cc, ss, vectors)
    summaries = {s: aggregate([r for r in records if r['stratum'] == s]) for s in STRATA}
    per_K = {s: {str(k): aggregate([r for r in records if r['stratum'] == s and r['K'] == k]) for k in (2, 4, 8, 16)} for s in STRATA}
    per_axis = {a: aggregate([r for r in records if r['stratum'] == 'polarity' and cc[next(i for i, c in enumerate(cc) if c['id'] == r['criterion_id'])]['weights'][j]]) for j, a in enumerate(AXES)}
    intervals = alias_intervals(records, protocol); reversal = reversals(root, cc, ss, vectors, cached)
    causal = swaps(root, cc, ss, cached); gates = {}
    for arm in ALL_ARMS:
        p = reversal['polarity'][arm]; inv = permutations[arm]
        gates[arm] = dict(G1_alias=summaries['alias'][arm]['top1'] >= .8 and arm in intervals and intervals[arm]['simultaneous_CI'][0] > 0,
                          G2_polarity=summaries['polarity'][arm]['top1'] >= .9 and p['negative_cosines'] == p['pairs'] and p['student_change_rate'] == 1.,
                          G3_composition=summaries['composition'][arm]['top1'] >= .8 and summaries['held_combination'][arm]['top1'] >= .8,
                          G4_correct_new=causal['pooled'][arm]['correct_new_rate'] >= .8,
                          G5_permutation=inv['score_mismatches'] == inv['winner_mismatches'] == 0)
        gates[arm]['all'] = all(gates[arm].values())
    passed = [arm for arm in ARMS if gates[arm]['all']]
    result = dict(experiment='CBF-4', protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
                  corpus=json.loads((evidence / 'corpus_manifest.json').read_text()),
                  encoder=json.loads((evidence / 'encoder_manifest.json').read_text()),
                  evidence_root=str(evidence), percent_sum_before_normalization=True,
                  replay_without_refitting=args.replay,
                  fits=fits, decision_rows=len(records), summary=summaries, per_K=per_K, polarity_per_axis=per_axis,
                  alias_paired_intervals=intervals, reversal=reversal, causal=causal, permutation=permutations,
                  gates=gates, passing_learned_interfaces=passed,
                  verdict='REPLICATION_REQUIRED' if passed else 'CRITERION_MAPPING_NOT_EARNED',
                  B_STEF_allowed=False)
    put(root, 'results.json', canonical(result) + b'\n')
    print(json.dumps({k: result[k] for k in ('summary', 'reversal', 'gates', 'passing_learned_interfaces', 'verdict')}, indent=2))


if __name__ == '__main__': main()
