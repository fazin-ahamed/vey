"""SRE probability decoding and frozen exact-compiler downstream gates."""
import itertools
import json
from collections import Counter

import numpy as np

from audit import put
from build import canonical
from exact_state_build import parse
from exact_state_run import score, winner
from schema_grounding_compiler import compile_criterion, parse_criterion
from schema_relation_prepare import PROTOCOL


def context(text,term):
    lo,hi=term['context_span']; return text[lo:hi].rstrip('.')+'.'


def semantic_contexts(atoms,compositions):
    texts={c['text'] for c in atoms}
    for c in compositions:
        for t in parse_criterion(c['text']):
            if t['literal_axis'] is None: texts.add(context(c['text'],t))
    return sorted(texts)


def decode(matrix):
    p=np.asarray(matrix,dtype=np.float64)
    assert p.shape==(4,3) and np.isfinite(p).all() and np.all(p>=0) and np.allclose(p.sum(1),1,atol=1e-6)
    classes=np.argmax(p,axis=1)-1; active=np.flatnonzero(classes)
    if not len(active): return None
    evidence=np.maximum(p[:,0],p[:,2])-p[:,1]
    axis=int(active[np.argmax(evidence[active])])
    return dict(axis=axis,sign=int(classes[axis]),relation_matrix=p.tolist(),row_classes=classes.tolist(),nonzero_fields=int(len(active)))


def literal_relation_summary(atoms,matrices):
    gold=np.array([[c['sign'] if j==c['axis'] else 0 for j in range(4)] for c in atoms]); pred=np.array([np.argmax(matrices[c['text']],axis=1)-1 for c in atoms])
    correct=gold==pred; matched=gold!=0
    return dict(n=int(gold.size),correct=int(correct.sum()),accuracy=float(correct.mean()),
                matched_n=int(matched.sum()),matched_accuracy=float(correct[matched].mean()),
                per_class={str(k):dict(n=int((gold==k).sum()),accuracy=float(correct[gold==k].mean())) for k in (-1,0,1)})


def atom_summary(atoms,resolutions):
    axis=sum(resolutions[c['text']] is not None and resolutions[c['text']]['axis']==c['axis'] for c in atoms)
    joint=sum(resolutions[c['text']] is not None and resolutions[c['text']]['axis']==c['axis'] and resolutions[c['text']]['sign']==c['sign'] for c in atoms)
    return dict(n=len(atoms),axis_correct=axis,axis_accuracy=axis/len(atoms),sign_given_axis_correct=joint,
                sign_given_axis_denominator=axis,conditional_sign_accuracy=joint/axis if axis else None,
                joint_correct=joint,joint_accuracy=joint/len(atoms),unknown=sum(resolutions[c['text']] is None for c in atoms))


def evaluate(root,tag,atoms,compositions,literals,states,matrices,g0):
    output=root/tag; output.mkdir()
    texts=semantic_contexts(atoms,compositions); resolutions={q:decode(matrices[q]) for q in texts}
    put(output,'relation_matrices.json',canonical([dict(text=q,fields=['reliability','purchase expense','operating expense','convenience'],classes=[-1,0,1],matrix=np.asarray(matrices[q]).tolist(),resolution=resolutions[q]) for q in texts])+b'\n')
    atomic=atom_summary(atoms,resolutions); relation=literal_relation_summary(g0,matrices)
    queries=[dict(c,stratum='alias') for c in atoms]+[dict(c,stratum='alias_composition') for c in compositions]+literals
    compiled=[]; records=[]; cached={}; rng=np.random.default_rng(7)
    invariance=dict(comparisons=0,numeric_comparisons=0,unknown_comparisons=0,score_mismatches=0,winner_mismatches=0)
    semantic_callbacks=0; literal_callbacks=0
    for c in queries:
        calls=0
        def resolve(t):
            nonlocal calls
            calls+=1; return resolutions[context(c['text'],t)]
        vector,ir=compile_criterion(c['text'],resolve)
        if c['stratum']=='literal':
            literal_callbacks+=calls; assert calls==0 and vector==c['weights']
        else: semantic_callbacks+=calls
        compiled.append(dict(id=c['id'],stratum=c['stratum'],vector=vector,coefficient_exact=vector==c['weights'],atoms=ir))
        for state in states:
            if state['source_split']=='train': continue
            facts=[parse(t) for t in state['candidates']]; gold=np.array(facts,dtype=np.int64)@np.array(c['weights'],dtype=np.int64)
            teacher=winner(gold,facts); values=score(vector,state['candidates']); chosen=winner(values,facts); a,b=np.triu_indices(state['K'],1)
            row=dict(criterion_id=c['id'],stratum=c['stratum'],scenario_id=state['id'],source_split=state['source_split'],K=state['K'],
                     teacher=teacher,teacher_scores=gold.tolist(),teacher_top_set=np.flatnonzero(gold==gold.max()).tolist(),winner=chosen,
                     scores=values.tolist() if values is not None else None,covered=chosen is not None,
                     top1=int(chosen is not None and gold[chosen]==gold.max()),exact_winner=int(chosen==teacher),
                     pairwise=float(np.mean(np.sign(values[a]-values[b])==np.sign(gold[a]-gold[b]))) if values is not None else 0.)
            records.append(row); cached[(c['id'],state['id'])]=row
            for order in (np.arange(state['K'])[::-1],rng.permutation(state['K'])):
                other=score(vector,[state['candidates'][j] for j in order]); selected=winner(other,[facts[j] for j in order])
                restored=int(order[selected]) if selected is not None else None
                invariance['comparisons']+=1; invariance['winner_mismatches']+=int(restored!=chosen)
                if other is None: invariance['unknown_comparisons']+=1
                else:
                    invariance['numeric_comparisons']+=1; invariance['score_mismatches']+=int(not np.array_equal(other[np.argsort(order)],values))
    put(output,'compiled_queries.json',canonical(compiled)+b'\n'); put(output,'decisions.jsonl',b''.join(canonical(r)+b'\n' for r in records))
    def summary(rows):
        return dict(n=len(rows),top1=float(np.mean([r['top1'] for r in rows])),exact_winner=float(np.mean([r['exact_winner'] for r in rows])),
                    coverage=float(np.mean([r['covered'] for r in rows])),pairwise=float(np.mean([r['pairwise'] for r in rows])))
    summaries={s:summary([r for r in records if r['stratum']==s and (s=='literal' or r['source_split']==('unseen_wording' if s=='alias' else 'unseen_criterion'))]) for s in ('alias','alias_composition','literal')}
    per_K={s:{str(k):summary([r for r in records if r['stratum']==s and r['K']==k and (s=='literal' or r['source_split']==('unseen_wording' if s=='alias' else 'unseen_criterion'))]) for k in (2,4,8,16)} for s in summaries}
    representatives={}
    for c in sorted((c for c in queries if c['stratum']!='literal'),key=lambda c:c['text']):
        from schema_grounding_build import ray
        representatives.setdefault(tuple(ray(c['weights'])),c)
    reps=list(representatives.values()); eligible=0; n=0; counts=Counter()
    with (output/'teacher_changing_swaps.jsonl').open('wb') as stream:
        for state in states:
            if state['source_split']=='train': continue
            for first,second in itertools.permutations(reps,2):
                eligible+=1; before=cached[(first['id'],state['id'])]; after=cached[(second['id'],state['id'])]
                if before['teacher']==after['teacher']: continue
                n+=1; a,b=before['winner'],after['winner']; correct=int(b==after['teacher'])
                flags=dict(correct_new=correct,changed_to_new=int(correct and a is not None and b is not None and a!=b),
                           both_endpoints=int(a==before['teacher'] and correct),top_set_new=after['top1'])
                counts.update(flags); stream.write(canonical(dict(scenario_id=state['id'],before=first['id'],after=second['id'],teacher_before=before['teacher'],teacher_after=after['teacher'],**flags))+b'\n')
    causal=dict(representatives=len(reps),eligible_pairs=eligible,n=n,**{key+'_rate':counts[key]/n for key in ('correct_new','changed_to_new','both_endpoints','top_set_new')})
    put(output,'causal_representatives.json',canonical([dict(id=c['id'],text=c['text'],weights=c['weights']) for c in reps])+b'\n')
    cfg=json.loads(PROTOCOL.read_text())['gates']
    gates=dict(G0=relation['accuracy']>=cfg['G0_literal_relation'],G1=atomic['axis_accuracy']>=cfg['G1_axis'],
               G2=atomic['conditional_sign_accuracy'] is not None and atomic['conditional_sign_accuracy']>=cfg['G2_sign_given_axis'],
               G3=atomic['joint_accuracy']>=cfg['G3_joint_atom'],G4=summaries['alias']['top1']>=cfg['G4_alias_decision'],
               G5=summaries['alias_composition']['top1']>=cfg['G5_alias_composition'],G6=causal['correct_new_rate']>=cfg['G6_correct_new'],
               G7=summaries['literal']['top1']==summaries['literal']['exact_winner']==summaries['literal']['pairwise']==1 and literal_callbacks==0 and invariance['score_mismatches']==invariance['winner_mismatches']==0)
    result=dict(literal_relation=relation,atoms=atomic,summary=summaries,per_K=per_K,causal=causal,permutation=invariance,
                compiler_accuracy={s:dict(n=sum(c['stratum']==s for c in compiled),correct=sum(c['coefficient_exact'] for c in compiled if c['stratum']==s)) for s in summaries},
                semantic_callbacks=semantic_callbacks,literal_callbacks=literal_callbacks,gates=gates,passed=all(gates.values()),decision_rows=len(records))
    put(output,'results.json',canonical(result)+b'\n'); return result


def uncertainty(root,tag,atoms,compositions):
    output=root/tag; matrices=json.loads((output/'relation_matrices.json').read_text()); resolved={m['text']:m['resolution'] for m in matrices}
    rows=[json.loads(line) for line in (output/'decisions.jsonl').read_text().splitlines()]; pairs=sorted({c['pair_id'] for c in atoms})
    n_axis=[];n_joint=[];a_decisions=[]
    for key in pairs:
        group=[c for c in atoms if c['pair_id']==key]; assert len(group)==2
        stats=atom_summary(group,resolved); n_axis.append(stats['axis_correct']); n_joint.append(stats['joint_correct'])
        ids={c['id'] for c in group}; a_decisions.append(np.mean([r['top1'] for r in rows if r['criterion_id'] in ids and r['source_split']=='unseen_wording']))
    rng=np.random.default_rng(0); draws=rng.integers(0,len(pairs),(10000,len(pairs))); axes=np.array(n_axis)[draws].sum(1); joints=np.array(n_joint)[draws].sum(1)
    def interval(values): return np.quantile(values,[.025,.975]).tolist()
    result=dict(repetitions=10000,seed=0,opposite_atom_pairs=len(pairs),axis_CI95=interval(axes/(2*len(pairs))),
                joint_CI95=interval(joints/(2*len(pairs))),sign_given_axis_CI95=interval(joints[axes>0]/axes[axes>0]) if np.any(axes>0) else None,
                alias_decision_CI95=interval(np.array(a_decisions)[draws].mean(1)),point_gates_only=True)
    cpairs=sorted({c['pair_id'] for c in compositions}); means=[]
    for key in cpairs:
        group=[c for c in compositions if c['pair_id']==key]; assert len(group)==2
        ids={c['id'] for c in group}; means.append(np.mean([r['top1'] for r in rows if r['criterion_id'] in ids and r['source_split']=='unseen_criterion']))
    cdraws=rng.integers(0,len(cpairs),(10000,len(cpairs))); result.update(opposite_composition_pairs=len(cpairs),composition_decision_CI95=interval(np.array(means)[cdraws].mean(1)))
    put(output,'intervals.json',canonical(result)+b'\n'); return result
