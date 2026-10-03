"""CBF-5: signed-atom diagnostics, exact compiler, controls and causal gates."""
import itertools
import json
from pathlib import Path

import numpy as np

from audit import put
from build import canonical
from exact_state_build import parse
from exact_state_run import score, winner
from schema_grounding_capture import PROTOCOL, atom_input, sha
from schema_grounding_compiler import compile_criterion, parse_criterion
from schema_grounding_resolver import METHODS, fit_prototypes, resolve


def load(root):
    manifest = json.loads((root / 'corpus_manifest.json').read_text())
    assert manifest['protocol_sha256'] == sha(PROTOCOL)
    for name, h in manifest['files'].items(): assert sha(root / name) == h, name
    fm = json.loads((root / 'feature_manifest.json').read_text())
    assert sha(root / 'features.npy') == fm['features_sha256']
    cc = json.loads((root / 'criteria.json').read_text()); ss = json.loads((root / 'states.json').read_text())
    tt = json.loads((root / 'prototype_queries.json').read_text()); specs = json.loads((root / 'atom_inputs.json').read_text())
    H = np.load(root / 'features.npy', allow_pickle=False); assert H.shape == (len(specs), 384) and np.isfinite(H).all()
    return manifest, fm, cc, ss, tt, specs, H


def compile_cases(root, cc, tt, specs, H):
    lookup = {s['id']: i for i, s in enumerate(specs)}
    member_features = np.stack([H[lookup[atom_input(c['text'], parse_criterion(c['text'])[0])['id']]] for c in tt])
    prototypes, basis, metadata = fit_prototypes(member_features, tt)
    metadata['prototype_pair_cosines'] = [float(prototypes[2*j] @ prototypes[2*j+1]) for j in range(4)]
    metadata['CBF4_reversal_space'] = 'Regressed four-coordinate vectors, not raw layer1 features'
    np.savez(root / 'prototypes.npz', prototypes=prototypes, nuisance_basis=basis,
             member_features=member_features, member_ids=np.array([c['id'] for c in tt]))
    put(root, 'prototype_manifest.json', canonical(metadata) + b'\n')
    unknown_ids = {atom_input(c['text'], t)['id'] for c in cc for t in parse_criterion(c['text']) if t['literal_axis'] is None}
    cached = {m: {key: resolve(H[lookup[key]], prototypes, basis, m) for key in sorted(unknown_ids)} for m in METHODS}
    compiled = []; atom_records = []
    for c in cc:
        row = dict(id=c['id'], stratum=c['stratum'], methods={})
        for method in METHODS:
            vector, atoms = compile_criterion(c['text'], lambda t: cached[method][atom_input(c['text'], t)['id']])
            assert len(atoms) == len(c['gold_atoms']), c['text']
            if c['stratum'] == 'literal':
                assert vector == c['weights'] and all(t['literal_axis'] is not None for t in atoms), c
            row['methods'][method] = dict(vector=vector, coefficient_exact=vector == c['weights'], atoms=atoms)
            for index, (prediction, gold) in enumerate(zip(atoms, c['gold_atoms'])):
                if prediction['literal_axis'] is not None: continue
                found = prediction['resolution']
                axis = bool(found is not None and found['axis'] == gold['axis'])
                sign = bool(found is not None and found['sign'] == gold['sign'])
                atom_records.append(dict(criterion_id=c['id'], semantic_id=c['semantic_id'], stratum=c['stratum'],
                                         component=index, method=method, gold=gold, prediction=found,
                                         factor=prediction['factor'], axis_correct=int(axis), sign_correct=int(sign),
                                         joint_correct=int(axis and sign)))
        compiled.append(row)
    put(root, 'compiled_queries.json', canonical(compiled) + b'\n')
    put(root, 'semantic_atoms.jsonl', b''.join(canonical(a) + b'\n' for a in atom_records))
    return {r['id']: r for r in compiled}, atom_records, metadata


def atom_summary(rows):
    n = len(rows); a = sum(r['axis_correct'] for r in rows); j = sum(r['joint_correct'] for r in rows)
    return dict(n=n, axis_correct=a, axis_accuracy=a/n, sign_given_axis_correct=j,
                sign_given_axis_denominator=a, conditional_sign_accuracy=j/a if a else None,
                joint_correct=j, joint_accuracy=j/n,
                unconditional_sign_accuracy=sum(r['sign_correct'] for r in rows)/n)


def decisions(root, cc, ss, predictions):
    records = []; cache = {}; rng = np.random.default_rng(7)
    invariance = {m: dict(comparisons=0, numeric_comparisons=0, unknown_comparisons=0,
                          score_mismatches=0, winner_mismatches=0, maximum_score_error=0.) for m in METHODS}
    for c in cc:
        for state in ss:
            if state['source_split'] == 'train': continue
            facts = [parse(t) for t in state['candidates']]
            gold = np.asarray(facts, dtype=np.int64) @ np.asarray(c['weights'], dtype=np.int64)
            teacher = winner(gold, facts); a, b = np.triu_indices(state['K'], 1)
            orders = [np.arange(state['K'])[::-1], rng.permutation(state['K'])]
            row = dict(criterion_id=c['id'], semantic_id=c['semantic_id'], stratum=c['stratum'],
                       scenario_id=state['id'], source_split=state['source_split'], K=state['K'],
                       teacher=teacher, teacher_scores=gold.tolist(),
                       teacher_top_set=np.flatnonzero(gold == gold.max()).tolist(), methods={})
            for method in METHODS:
                vector = predictions[c['id']]['methods'][method]['vector']; values = score(vector, state['candidates'])
                chosen = winner(values, facts)
                row['methods'][method] = dict(winner=chosen, scores=values.tolist() if values is not None else None,
                                              top1=int(chosen is not None and gold[chosen] == gold.max()),
                                              exact_winner=int(chosen == teacher), covered=chosen is not None,
                                              pairwise=float(np.mean(np.sign(values[a]-values[b]) == np.sign(gold[a]-gold[b]))) if values is not None else 0.)
                for order in orders:
                    other = score(vector, [state['candidates'][j] for j in order])
                    selected = winner(other, [facts[j] for j in order]); restored_win = int(order[selected]) if selected is not None else None
                    check = invariance[method]; check['comparisons'] += 1
                    check['winner_mismatches'] += int(restored_win != chosen)
                    if other is None: check['unknown_comparisons'] += 1
                    else:
                        restored = other[np.argsort(order)]; check['numeric_comparisons'] += 1
                        check['score_mismatches'] += int(not np.array_equal(restored, values))
                        check['maximum_score_error'] = max(check['maximum_score_error'], float(np.max(np.abs(restored-values))))
            records.append(row); cache[(c['id'], state['id'])] = row
    put(root, 'decisions.jsonl', b''.join(canonical(r) + b'\n' for r in records))
    return records, cache, invariance


def decision_summary(rows):
    return {m: dict(n=len(rows), top1=float(np.mean([r['methods'][m]['top1'] for r in rows])),
                    exact_winner=float(np.mean([r['methods'][m]['exact_winner'] for r in rows])),
                    pairwise=float(np.mean([r['methods'][m]['pairwise'] for r in rows])),
                    coverage=float(np.mean([r['methods'][m]['covered'] for r in rows]))) for m in METHODS}


def intervals(cc, atoms, rows, protocol):
    alias = [c for c in cc if c['stratum'] == 'alias']; ids = sorted({c['semantic_id'] for c in alias})
    cfg = protocol['gates']['bootstrap']; rng = np.random.default_rng(cfg['seed'])
    draws = rng.integers(0, len(ids), size=(cfg['repetitions'], len(ids)))
    source = Path(protocol['source_root']); old = {}
    for line in (source / 'score-corrected/decisions.jsonl').read_text().splitlines():
        r = json.loads(line)
        if r['stratum'] == 'alias': old[(r['criterion_id'], r['scenario_id'])] = r
    primary = [r for r in rows if r['stratum'] == 'alias' and r['source_split'] == 'unseen_wording']
    assert len(primary) == len(old) == 512
    atom_means, decision_means, baseline = {}, {}, []
    for m in METHODS:
        atom_means[m] = np.array([np.mean([a['joint_correct'] for a in atoms if a['method'] == m and a['stratum'] == 'alias' and a['semantic_id'] == key]) for key in ids])
        decision_means[m] = np.array([np.mean([r['methods'][m]['top1'] for r in primary if r['semantic_id'] == key]) for key in ids])
    for key in ids:
        members = [r for r in primary if r['semantic_id'] == key]
        baseline.append(np.mean([old[(r['criterion_id'], r['scenario_id'])]['arms']['layer1_span']['top1'] for r in members]))
    baseline = np.array(baseline); assert float(baseline.mean()) == protocol['CBF4_alias_decision_target']
    def bounds(values):
        samples = values[draws].mean(1); alpha = .05 / cfg['family_size']
        return dict(point=float(values.mean()), CI95=np.quantile(samples, [.025, .975]).tolist(),
                    simultaneous_CI=np.quantile(samples, [alpha/2, 1-alpha/2]).tolist())
    versus_CBF4 = {m: dict(absolute=bounds(decision_means[m]), delta=bounds(decision_means[m]-baseline)) for m in METHODS}
    mechanism = {}
    for asg, controls in (('asg', ('cosine_positive','cosine_signed')),
                          ('asg_projected', ('cosine_positive_projected','cosine_signed_projected'))):
        mechanism[asg] = {c: dict(joint_atoms=bounds(atom_means[asg]-atom_means[c]),
                                  decisions=bounds(decision_means[asg]-decision_means[c])) for c in controls}
    return dict(clusters=ids, n_clusters=len(ids), repetitions=cfg['repetitions'], family_size=cfg['family_size'],
                CBF4_baseline=float(baseline.mean()), versus_CBF4=versus_CBF4, mechanism=mechanism)


def causal(root, cc, ss, cache):
    representatives = {}
    for c in sorted((c for c in cc if c['stratum'] != 'literal'), key=lambda c: c['text']):
        representatives.setdefault(tuple(c['ray']), c)
    chosen = list(representatives.values()); stats = {m: dict(n=0, correct_new=0, changed_to_new=0, both_endpoints=0, top_set_new=0) for m in METHODS}
    eligible = 0; path = root / 'teacher_changing_swaps.jsonl'
    if path.exists(): raise RuntimeError('Refusing causal evidence overwrite')
    with path.open('wb') as out:
        for state in ss:
            if state['source_split'] == 'train': continue
            for first, second in itertools.permutations(chosen, 2):
                eligible += 1; before = cache[(first['id'],state['id'])]; after = cache[(second['id'],state['id'])]
                if before['teacher'] == after['teacher']: continue
                row = dict(scenario_id=state['id'], before=first['id'], after=second['id'],
                           teacher_before=before['teacher'], teacher_after=after['teacher'], methods={})
                for m in METHODS:
                    a, b = before['methods'][m]['winner'], after['methods'][m]['winner']
                    correct = int(b == after['teacher']); changed = a is not None and b is not None and a != b
                    flags = dict(correct_new=correct, changed_to_new=int(correct and changed),
                                 both_endpoints=int(a == before['teacher'] and correct), top_set_new=after['methods'][m]['top1'])
                    row['methods'][m] = flags; stats[m]['n'] += 1
                    for key, value in flags.items(): stats[m][key] += value
                out.write(canonical(row) + b'\n')
    summary = {m: dict(n=v['n'], **{key+'_rate': v[key]/v['n'] for key in ('correct_new','changed_to_new','both_endpoints','top_set_new')}) for m,v in stats.items()}
    put(root, 'causal_representatives.json', canonical([dict(id=c['id'],text=c['text'],ray=c['ray']) for c in chosen]) + b'\n')
    return dict(representatives=len(chosen), eligible_pairs=eligible, teacher_changing_pairs=next(iter(stats.values()))['n'], summary=summary)


def diagnosis(axis, conditional_sign):
    high_axis = axis >= .8; high_sign = conditional_sign is not None and conditional_sign >= .9
    if high_axis and high_sign: return 'ATOM_GROUNDING_HIGH; task gates determine many-axis expansion'
    if not high_axis and high_sign: return 'AXIS_GROUNDING_BOTTLENECK; tiny axis-specific learned metric justified next'
    if high_axis and not high_sign: return 'ORIENTATION_BOTTLENECK'
    return 'BOTH_LOW; tested frozen span/prototype retrieval insufficient, stronger encoder is a next candidate, not an information-absence proof'


def main():
    protocol = json.loads(PROTOCOL.read_text()); root = Path(protocol['output_root'])
    if (root / 'results.json').exists(): raise RuntimeError('Refusing to overwrite a measured CBF-5 experiment')
    corpus, features, cc, ss, tt, specs, H = load(root)
    predictions, atoms, proto = compile_cases(root, cc, tt, specs, H)
    atom_stats = {s: {m: atom_summary([a for a in atoms if a['stratum']==s and a['method']==m]) for m in METHODS} for s in ('alias','alias_composition')}
    rows, cache, invariance = decisions(root, cc, ss, predictions)
    summaries = dict(alias=decision_summary([r for r in rows if r['stratum']=='alias' and r['source_split']=='unseen_wording']),
                     alias_composition=decision_summary([r for r in rows if r['stratum']=='alias_composition' and r['source_split']=='unseen_criterion']),
                     literal=decision_summary([r for r in rows if r['stratum']=='literal']))
    per_K = {s: {str(k): decision_summary([r for r in rows if r['stratum']==s and r['K']==k and
                                         (s=='literal' or r['source_split']==('unseen_wording' if s=='alias' else 'unseen_criterion'))]) for k in (2,4,8,16)} for s in summaries}
    compiler_accuracy = {s: {m: dict(n=sum(c['stratum']==s for c in cc),
                                     correct=sum(predictions[c['id']]['methods'][m]['coefficient_exact'] for c in cc if c['stratum']==s),
                                     accuracy=float(np.mean([predictions[c['id']]['methods'][m]['coefficient_exact'] for c in cc if c['stratum']==s])))
                               for m in METHODS} for s in summaries}
    paired = intervals(cc, atoms, rows, protocol); swaps = causal(root, cc, ss, cache); gates = {}
    for m in METHODS:
        pair = paired['versus_CBF4'][m]; inv = invariance[m]; lit = summaries['literal'][m]
        gates[m] = dict(G1_joint_atoms=atom_stats['alias'][m]['joint_accuracy'] >= .8,
                        G2_alias_decision=summaries['alias'][m]['top1'] >= .8 and pair['delta']['simultaneous_CI'][0] > 0 and pair['absolute']['simultaneous_CI'][0] > protocol['CBF4_alias_decision_target'],
                        G3_exact_literal=lit['top1']==lit['pairwise']==lit['exact_winner']==1. and all(predictions[c['id']]['methods'][m]['coefficient_exact'] for c in cc if c['stratum']=='literal'),
                        G4_alias_composition=summaries['alias_composition'][m]['top1'] >= .8,
                        G5_correct_new=swaps['summary'][m]['correct_new_rate'] >= .8,
                        G6_permutation=inv['score_mismatches']==inv['winner_mismatches']==0)
        gates[m]['task_pass'] = all(gates[m].values())
    mechanism = {m: all(v[metric]['simultaneous_CI'][0] > 0 for v in controls.values() for metric in ('joint_atoms','decisions')) for m,controls in paired['mechanism'].items()}
    earned_asg = [m for m in mechanism if gates[m]['task_pass'] and mechanism[m]]
    retrieval = [m for m in METHODS if m.startswith('cosine_') and gates[m]['task_pass']]
    verdict = 'ASG_EARNED; expand schema before B-STEF' if earned_asg else ('RETRIEVAL_SUFFICIENT; retain simpler retrieval and expand schema before B-STEF' if retrieval else 'SEMANTIC_SCHEMA_GROUNDING_NOT_EARNED')
    result = dict(experiment='CBF-5', protocol_sha256=sha(PROTOCOL), corpus=corpus, feature_capture=features,
                  prototype_metadata=proto, atom_diagnostics=atom_stats, summary=summaries, per_K=per_K,
                  compiler_accuracy=compiler_accuracy,
                  paired_intervals=paired, causal=swaps, permutation=invariance, decision_rows=len(rows),
                  gates=gates, ASG_mechanism_pass=mechanism, earned_ASG_variants=earned_asg,
                  passing_retrieval_controls=retrieval, verdict=verdict, B_STEF_allowed=False,
                  diagnoses={m: diagnosis(atom_stats['alias'][m]['axis_accuracy'],atom_stats['alias'][m]['conditional_sign_accuracy']) for m in METHODS})
    put(root, 'results.json', canonical(result) + b'\n')
    print(json.dumps({k: result[k] for k in ('atom_diagnostics','summary','compiler_accuracy','gates','ASG_mechanism_pass','verdict','diagnoses')},indent=2))


if __name__ == '__main__': main()
