"""Independently reconstruct persisted CBF-5 features, atoms and exact decisions."""
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np

from audit import put
from build import canonical
from schema_grounding_capture import PROTOCOL, atom_input, sha, source_inputs
from schema_grounding_compiler import compile_criterion, parse_criterion
from schema_grounding_resolver import METHODS
from schema_grounding_run import atom_summary, decision_summary, intervals


def reference(h, prototypes, basis, method):
    h = h.astype(np.float64); h /= np.linalg.norm(h)
    p = prototypes.copy(); p /= np.linalg.norm(p, axis=1, keepdims=True)
    if method.endswith('_projected'):
        h -= (h @ basis.T) @ basis; p -= (p @ basis.T) @ basis
        if np.linalg.norm(h) <= np.finfo(float).eps*384 or np.any(np.linalg.norm(p, axis=1) <= np.finfo(float).eps*384): return None
        h /= np.linalg.norm(h); p /= np.linalg.norm(p, axis=1, keepdims=True)
    cos = p @ h; e = cos[::2]-cos[1::2]
    if method.startswith('cosine_signed'):
        index = int(np.argmax(cos)); axis = index//2; sign = 1 if index%2==0 else -1
    else:
        axis = int(np.argmax(cos[::2] if method.startswith('cosine_positive') else abs(e)))
        if e[axis] == 0: return None
        sign = 1 if e[axis] > 0 else -1
    return dict(axis=axis,sign=sign,cosines=cos.tolist(),evidence=e.tolist())


def verify_features(root, source, specs, H, fm):
    old_texts = np.load(source / 'texts.npy', allow_pickle=False).tolist(); old = np.load(source / 'layer1_span.npy', allow_pickle=False)
    old_lookup = {t:i for i,t in enumerate(old_texts)}
    fresh = root / 'atoms'; new_texts = np.load(fresh / 'texts.npy',allow_pickle=False).tolist()
    reconstructed = {}; largest = 0.; offsets_verified = 0
    encoder = json.loads((fresh / 'encoder_manifest.json').read_text())
    assert encoder['revision']=='eb2d654bf0a5b628c8be6c4be7d29118fbef95b8'
    assert encoder['weight_sha256']=='964ceb3612da6cfdb45997d380fdb95f92c7499ffcabb50cbeea55e04756cafd'
    assert encoder['dtype']=='float32' and encoder['stock_not_Vey2']
    assert not encoder['loading']['missing_keys'] and not encoder['loading']['mismatched_keys']
    assert encoder['criteria']==4 and encoder['forwards']==1 and encoder['candidate_forwards']==0
    for shard in encoder['raw_shards']:
        data = np.load(fresh / shard['file'],allow_pickle=False); start=0
        for index, length in zip(data['indices'],data['lengths']):
            stop=start+int(length); text=new_texts[int(index)]
            spec = next(s for s in specs if s['text']==text); lo,hi=spec['span']
            off=data['offsets'][start:stop]; mask=(~data['special'][start:stop]) & (off[:,0]<hi) & (off[:,1]>lo)
            assert np.array_equal(mask,data['span'][start:stop]) and mask.any()
            values=data['hidden'][start:stop,0][mask].astype(np.float64).mean(0)
            reconstructed[text]=values/np.linalg.norm(values); offsets_verified+=1; start=stop
        assert start==len(data['hidden'])
    reused=0; fresh_count=0
    for i,s in enumerate(specs):
        if s['text'] in old_lookup:
            assert np.array_equal(H[i],old[old_lookup[s['text']]]); reused+=1
        else:
            error=float(np.max(np.abs(H[i]-reconstructed[s['text']]))); assert error<1e-6
            largest=max(largest,error); fresh_count+=1
    assert reused==80 and fresh_count==offsets_verified==4
    assert fm['candidate_forwards']==fm['full_composition_forwards']==0
    return dict(reused_feature_rows_exact=reused,new_features_from_raw_tokens=fresh_count,
                new_token_masks_verified=offsets_verified,maximum_pooling_error=largest,
                new_encoder_forwards=1,candidate_forwards=0,full_composition_forwards=0)


def main():
    protocol,source,source_cc,source_ss,lineage=source_inputs(); root=Path(protocol['output_root'])
    result=json.loads((root/'results.json').read_text()); corpus=json.loads((root/'corpus_manifest.json').read_text())
    assert result['protocol_sha256']==corpus['protocol_sha256']==sha(PROTOCOL)
    assert corpus['source_lineage']==lineage
    for name,h in corpus['files'].items(): assert sha(root/name)==h
    assert (root/'states.json').read_bytes()==(source/'states.json').read_bytes()
    cc=json.loads((root/'criteria.json').read_text()); ss=json.loads((root/'states.json').read_text())
    tt=json.loads((root/'prototype_queries.json').read_text()); specs=json.loads((root/'atom_inputs.json').read_text())
    terms=json.loads((root/'query_terms.json').read_text()); H=np.load(root/'features.npy',allow_pickle=False)
    fm=json.loads((root/'feature_manifest.json').read_text()); assert sha(root/'features.npy')==fm['features_sha256']
    feature_checks=verify_features(root,source,specs,H,fm); lookup={s['id']:i for i,s in enumerate(specs)}
    parameters=np.load(root/'prototypes.npz',allow_pickle=False); proto=parameters['prototypes']; basis=parameters['nuisance_basis']
    members=np.stack([H[lookup[atom_input(c['text'],parse_criterion(c['text'])[0])['id']]] for c in tt]).astype(np.float64)
    assert np.array_equal(members,parameters['member_features']) and parameters['member_ids'].tolist()==[c['id'] for c in tt]
    assert len(tt)==48 and {c['id'] for c in tt}.isdisjoint({c['id'] for c in cc})
    source_by_id={c['id']:c for c in source_cc}
    assert all(c==source_by_id[c['id']] and c['stratum']=='train' for c in tt)
    members/=np.linalg.norm(members,axis=1,keepdims=True)
    labels=[(next(i for i,w in enumerate(c['weights']) if w),next(w for w in c['weights'] if w)) for c in tt]
    reconstructed=[]
    for axis in range(4):
        for sign in (1,-1):
            rows=[i for i,label in enumerate(labels) if label==(axis,sign)]; assert len(rows)==6
            assert {tt[i]['template'] for i in rows}=={f'atomic/{j}' for j in range(6)}
            p=members[rows].mean(0); reconstructed.append(p/np.linalg.norm(p))
    prototype_error=float(np.max(np.abs(np.array(reconstructed)-proto))); assert prototype_error<1e-12
    means=np.array([members[[i for i,c in enumerate(tt) if c['template']==f'atomic/{t}']].mean(0) for t in range(6)])
    contrasts=means-means.mean(0); _,sv,vt=np.linalg.svd(contrasts,full_matrices=False)
    expected=vt[sv>np.finfo(float).eps*max(contrasts.shape)*sv[0]]
    projector_error=float(np.max(np.abs(expected.T@expected-basis.T@basis))); assert projector_error<1e-12
    saved=json.loads((root/'compiled_queries.json').read_text()); predictions={r['id']:r for r in saved}; atoms=[]; semantic_count=0; cosine_error=0.
    for c in cc:
        parsed=parse_criterion(c['text']); assert len(parsed)==len(terms[c['id']])
        for t,saved_t in zip(parsed,terms[c['id']]):
            assert t=={k:v for k,v in saved_t.items() if k!='input_id'}
            if t['literal_axis'] is None: assert saved_t['input_id']==atom_input(c['text'],t)['id']
            else: assert saved_t['input_id'] is None
        for m in METHODS:
            predicted=predictions[c['id']]['methods'][m]; exact=[0]*4; unknown=False
            for index,(t,record,gold) in enumerate(zip(parsed,predicted['atoms'],c['gold_atoms'])):
                assert record['term']==c['text'][t['span'][0]:t['span'][1]]
                if t['literal_axis'] is None:
                    ref=reference(H[lookup[atom_input(c['text'],t)['id']]],proto,basis,m); semantic_count+=1
                    found=record['resolution']; assert (ref is None)==(found is None)
                    if ref is not None:
                        assert (ref['axis'],ref['sign'])==(found['axis'],found['sign'])
                        error=float(np.max(np.abs(np.array(ref['cosines'])-found['cosines']))); cosine_error=max(cosine_error,error); assert error<1e-12
                    axis=found is not None and found['axis']==gold['axis']; sign=found is not None and found['sign']==gold['sign']
                    atoms.append(dict(criterion_id=c['id'],semantic_id=c['semantic_id'],stratum=c['stratum'],component=index,method=m,
                                      gold=gold,prediction=found,factor=t['factor'],axis_correct=int(axis),sign_correct=int(sign),joint_correct=int(axis and sign)))
                else:
                    found=dict(axis=t['literal_axis'],sign=t['literal_sign']); assert record['resolution']==found
                if found is None: unknown=True
                else: exact[found['axis']]+=t['factor']*found['sign']
            assert predicted['vector']==(None if unknown else exact)
            assert predicted['coefficient_exact']==(predicted['vector']==c['weights'])
            if c['stratum']=='literal': assert exact==c['weights'] and not unknown
    assert atoms==[json.loads(line) for line in (root/'semantic_atoms.jsonl').read_text().splitlines()]
    for s in ('alias','alias_composition'):
        for m in METHODS: assert atom_summary([a for a in atoms if a['stratum']==s and a['method']==m])==result['atom_diagnostics'][s][m]
    rows=[json.loads(line) for line in (root/'decisions.jsonl').read_text().splitlines()]
    state_lookup={s['id']:s for s in ss}; query_lookup={c['id']:c for c in cc}; cache={}; rng=np.random.default_rng(7); permutation_count=0
    assert len(rows)==237*32==result['decision_rows']
    for r in rows:
        c=query_lookup[r['criterion_id']]; state=state_lookup[r['scenario_id']]; facts=np.array(state['parsed_facts'],dtype=np.int64)
        gold=facts@np.array(c['weights'],dtype=np.int64); choose=lambda v:max(range(len(v)),key=lambda i:(int(v[i]),tuple(facts[i])))
        teacher=choose(gold); assert teacher==r['teacher'] and gold.tolist()==r['teacher_scores']
        assert np.flatnonzero(gold==gold.max()).tolist()==r['teacher_top_set']; a,b=np.triu_indices(len(facts),1)
        orders=[np.arange(len(facts))[::-1],rng.permutation(len(facts))]
        for m in METHODS:
            v=predictions[c['id']]['methods'][m]['vector']; found=r['methods'][m]
            values=None if v is None else facts@np.array(v,dtype=np.int64)
            if values is None: assert found['winner'] is None and not found['covered'] and found['top1']==found['exact_winner']==found['pairwise']==0
            else:
                win=choose(values); assert win==found['winner'] and (values/100).tolist()==found['scores']
                assert found['top1']==int(gold[win]==gold.max()) and found['exact_winner']==int(win==teacher)
                assert found['pairwise']==float(np.mean(np.sign(values[a]-values[b])==np.sign(gold[a]-gold[b])))
                for order in orders:
                    reordered=facts[order]@np.array(v,dtype=np.int64)
                    selected=max(range(len(order)),key=lambda i:(int(reordered[i]),tuple(facts[order[i]])))
                    assert int(order[selected])==win and np.array_equal(reordered[np.argsort(order)],values)
            permutation_count+=2
        cache[(c['id'],state['id'])]=r
    for s in ('alias','alias_composition','literal'):
        subset=[r for r in rows if r['stratum']==s and (s=='literal' or r['source_split']==('unseen_wording' if s=='alias' else 'unseen_criterion'))]
        assert decision_summary(subset)==result['summary'][s]
        for m in METHODS:
            cases=[c for c in cc if c['stratum']==s]; correct=sum(predictions[c['id']]['methods'][m]['coefficient_exact'] for c in cases)
            assert result['compiler_accuracy'][s][m]==dict(n=len(cases),correct=correct,accuracy=correct/len(cases))
    assert intervals(cc,atoms,rows,protocol)==result['paired_intervals']
    representatives={}
    for c in sorted((c for c in cc if c['stratum']!='literal'),key=lambda c:c['text']): representatives.setdefault(tuple(c['ray']),c)
    reps=list(representatives.values()); expected_keys=set()
    for s in ss:
        if s['source_split']=='train': continue
        for first,second in itertools.permutations(reps,2):
            before=cache[(first['id'],s['id'])]; after=cache[(second['id'],s['id'])]
            if before['teacher']!=after['teacher']: expected_keys.add((s['id'],first['id'],second['id']))
    swap_count=0; seen=set(); counts={m:Counter() for m in METHODS}
    with (root/'teacher_changing_swaps.jsonl').open() as stream:
        for line in stream:
            r=json.loads(line); key=(r['scenario_id'],r['before'],r['after']); assert key in expected_keys and key not in seen; seen.add(key)
            before=cache[(r['before'],r['scenario_id'])]; after=cache[(r['after'],r['scenario_id'])]
            assert r['teacher_before']==before['teacher'] and r['teacher_after']==after['teacher'] and before['teacher']!=after['teacher']; swap_count+=1
            for m in METHODS:
                a,b=before['methods'][m]['winner'],after['methods'][m]['winner']; correct=int(b==after['teacher'])
                flags=dict(correct_new=correct,changed_to_new=int(correct and a is not None and b is not None and a!=b),
                           both_endpoints=int(a==before['teacher'] and correct),top_set_new=after['methods'][m]['top1'])
                assert flags==r['methods'][m]; counts[m].update(flags)
    assert seen==expected_keys and swap_count==result['causal']['teacher_changing_pairs']
    assert len(reps)==56 and result['causal']['eligible_pairs']==56*55*32
    for m in METHODS:
        assert result['causal']['summary'][m]==dict(n=swap_count,**{k+'_rate':counts[m][k]/swap_count for k in ('correct_new','changed_to_new','both_endpoints','top_set_new')})
        g=result['gates'][m]; ci=result['paired_intervals']['versus_CBF4'][m]
        assert g['G1_joint_atoms']==(result['atom_diagnostics']['alias'][m]['joint_accuracy']>=.8)
        assert g['G2_alias_decision']==(result['summary']['alias'][m]['top1']>=.8 and ci['delta']['simultaneous_CI'][0]>0 and ci['absolute']['simultaneous_CI'][0]>.3203125)
        assert g['G3_exact_literal'] and g['G6_permutation']
        assert g['G4_alias_composition']==(result['summary']['alias_composition'][m]['top1']>=.8)
        assert g['G5_correct_new']==(counts[m]['correct_new']/swap_count>=.8)
        assert g['task_pass']==all(v for k,v in g.items() if k!='task_pass')
        inv=result['permutation'][m]; assert inv['comparisons']==237*32*2 and inv['score_mismatches']==inv['winner_mismatches']==0
    for m,controls in result['paired_intervals']['mechanism'].items():
        assert result['ASG_mechanism_pass'][m]==all(v[key]['simultaneous_CI'][0]>0 for v in controls.values() for key in ('joint_atoms','decisions'))
    assert result['B_STEF_allowed'] is False
    checks=dict(source_states_byte_identical=True,criteria=len(cc),literal_coefficient_exact=155,
                prototype_training_queries=48,prototype_error=prototype_error,projector_error=projector_error,
                feature_checks=feature_checks,semantic_atoms_reconstructed=semantic_count,maximum_cosine_error=cosine_error,
                decision_rows_reconstructed=len(rows),permutation_comparisons=permutation_count,
                teacher_changing_swaps_reconstructed=swap_count,paired_interval_family_reconstructed=20,
                source_lineage=lineage,all_checks_pass=True)
    put(root,'verification.json',canonical(checks)+b'\n'); print(json.dumps(checks,indent=2))


if __name__=='__main__': main()
