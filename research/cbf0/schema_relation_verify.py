"""Reconstruct SRE decisions from saved head arrays and integer facts."""
import re
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.special import erf
from transformers import AutoTokenizer

from audit import put
from build import canonical
from schema_grounding_compiler import compile_criterion, parse_criterion
from schema_relation_encoder import MODEL, REVISION, SCHEMA
from schema_relation_prepare import PROTOCOL, sha


def probabilities(H,normalizer,checkpoint):
    values=(H.astype(np.float64)-normalizer['mean'])/normalizer['std']; weights={k:v.numpy().astype(np.float64) for k,v in checkpoint['head'].items()}
    if checkpoint['kind']=='linear': logits=values@weights['weight'].T+weights['bias']
    else:
        hidden=values@weights['0.weight'].T+weights['0.bias']; hidden=.5*hidden*(1+erf(hidden/np.sqrt(2)))
        logits=hidden@weights['2.weight'].T+weights['2.bias']
    exponent=np.exp(logits-logits.max(1,keepdims=True)); return exponent/exponent.sum(1,keepdims=True)


def signed_atom(matrix):
    values=np.array(matrix); classes=values.argmax(1)-1; active=[j for j in range(4) if classes[j]!=0]
    if not active: return None
    axis=max(active,key=lambda j:(max(values[j,0],values[j,2])-values[j,1],-j))
    return axis,int(classes[axis])


def verify_tokens(evidence,tokenizer):
    assert evidence['truncation'] is False and evidence['candidate_forwards']==0
    counts=0;tokens=0
    for batch in evidence['batches']:
        pairs=evidence['pairs'][batch['start']:batch['start']+batch['count']]
        assert all(p['field'] in SCHEMA for p in pairs)
        expected=tokenizer([p['criterion'] for p in pairs],text_pair=[p['field'] for p in pairs],padding=True,truncation=False,return_tensors='np')
        assert np.array_equal(expected['input_ids'],batch['input_ids']) and np.array_equal(expected['attention_mask'],batch['attention_mask'])
        if 'token_type_ids' in expected: assert np.array_equal(expected['token_type_ids'],batch['token_type_ids'])
        assert expected['input_ids'].shape[1]<=128;counts+=len(pairs);tokens+=int(expected['attention_mask'].sum())
    assert counts==len(evidence['pairs']) and tokens==evidence['tokens'] and len(evidence['batches'])==evidence['forwards']
    return counts


def verify_stage(root,tag,atoms,compositions,literals,states,H,evidence,checkpoint,normalizer,g0):
    output=root/tag; result=json.loads((output/'results.json').read_text()); p=probabilities(H,normalizer,checkpoint)
    assert len(p)==len(evidence['pairs']); by_text={}
    for i,pair in enumerate(evidence['pairs']):
        by_text.setdefault(pair['criterion'],np.empty((4,3)))[SCHEMA.index(pair['field'])]=p[i]
    saved=json.loads((output/'relation_matrices.json').read_text()); resolutions={}; maximum=0
    for record in saved:
        error=float(np.max(np.abs(by_text[record['text']]-record['matrix']))); maximum=max(maximum,error); assert error<2e-5,error
        truth=signed_atom(record['matrix']); other=signed_atom(by_text[record['text']]); assert truth==other
        if truth is None: assert record['resolution'] is None
        else: assert truth==(record['resolution']['axis'],record['resolution']['sign'])
        resolutions[record['text']]=truth
    axis=sum(resolutions[c['text']] is not None and resolutions[c['text']][0]==c['axis'] for c in atoms)
    joint=sum(resolutions[c['text']] is not None and resolutions[c['text']]==(c['axis'],c['sign']) for c in atoms)
    stats=result['atoms']; assert stats['n']==len(atoms) and stats['axis_correct']==axis and stats['joint_correct']==joint
    assert stats['axis_accuracy']==axis/len(atoms) and stats['joint_accuracy']==joint/len(atoms) and stats['conditional_sign_accuracy']==(joint/axis if axis else None)
    # G0 uses held literal templates, never runtime neural callbacks.
    correct=0;total=0
    for c in g0:
        classes=np.argmax(by_text[c['text']],axis=1)-1
        for j in range(4): correct+=int(classes[j]==(c['sign'] if j==c['axis'] else 0));total+=1
    assert result['literal_relation']['n']==total and result['literal_relation']['correct']==correct
    queries=[dict(c,stratum='alias') for c in atoms]+[dict(c,stratum='alias_composition') for c in compositions]+literals
    compiled={r['id']:r for r in json.loads((output/'compiled_queries.json').read_text())}; vectors={}; literal_calls=0
    for c in queries:
        vector=[0]*4;unknown=False
        for term in parse_criterion(c['text']):
            if term['literal_axis'] is None:
                lo,hi=term['context_span']; atom=resolutions[c['text'][lo:hi].rstrip('.')+'.']
            else: atom=(term['literal_axis'],term['literal_sign'])
            if atom is None: unknown=True
            else: vector[atom[0]]+=term['factor']*atom[1]
        expected=None if unknown else vector; assert compiled[c['id']]['vector']==expected
        assert compiled[c['id']]['coefficient_exact']==(expected==c['weights']);vectors[c['id']]=expected
        if c['stratum']=='literal':
            def forbidden(term):
                nonlocal literal_calls
                literal_calls+=1; raise AssertionError('Neural literal callback')
            exact,_=compile_criterion(c['text'],forbidden);assert exact==c['weights']==expected
    assert literal_calls==result['literal_callbacks']==0
    states_by_id={s['id']:s for s in states}; queries_by_id={c['id']:c for c in queries}; cache={}; sums={s:Counter() for s in ('alias','alias_composition','literal')}; nrows=0;permutations=0;rng=np.random.default_rng(7)
    with (output/'decisions.jsonl').open() as stream:
        for line in stream:
            r=json.loads(line);c=queries_by_id[r['criterion_id']];state=states_by_id[r['scenario_id']];facts=np.array(state['parsed_facts'],dtype=np.int64);gold=facts@np.array(c['weights'],dtype=np.int64)
            choose=lambda values:max(range(len(values)),key=lambda i:(int(values[i]),tuple(facts[i])))
            teacher=choose(gold);assert teacher==r['teacher'] and gold.tolist()==r['teacher_scores'];a,b=np.triu_indices(len(facts),1)
            direction=vectors[c['id']];values=None if direction is None else facts@np.array(direction,dtype=np.int64);win=None if values is None else choose(values)
            assert win==r['winner'] and r['top1']==int(win is not None and gold[win]==gold.max()) and r['exact_winner']==int(win==teacher)
            if values is None: assert r['scores'] is None and not r['covered'] and r['pairwise']==0
            else:
                assert r['scores']==(values/100).tolist() and r['covered']
                assert r['pairwise']==float(np.mean(np.sign(values[a]-values[b])==np.sign(gold[a]-gold[b])))
            for order in (np.arange(len(facts))[::-1],rng.permutation(len(facts))):
                if values is not None:
                    ordered=facts[order]@np.array(direction,dtype=np.int64)
                    chosen=max(range(len(order)),key=lambda i:(int(ordered[i]),tuple(facts[order[i]])))
                    assert int(order[chosen])==win and np.array_equal(ordered[np.argsort(order)],values)
                permutations+=1
            if r['stratum']=='literal' or r['source_split']==('unseen_wording' if r['stratum']=='alias' else 'unseen_criterion'):
                sums[r['stratum']].update(n=1,top1=r['top1'],exact_winner=r['exact_winner'],covered=int(r['covered']))
            cache[(c['id'],state['id'])]=r;nrows+=1
    assert nrows==len(queries)*32==result['decision_rows'] and permutations==result['permutation']['comparisons']
    assert result['permutation']['score_mismatches']==result['permutation']['winner_mismatches']==0
    for key,v in sums.items():
        assert result['summary'][key]['n']==v['n'] and result['summary'][key]['top1']==v['top1']/v['n'] and result['summary'][key]['exact_winner']==v['exact_winner']/v['n']
    representatives={}
    from math import gcd
    for c in sorted((c for c in queries if c['stratum']!='literal'),key=lambda c:c['text']):
        divisor=gcd(*c['weights']);representatives.setdefault(tuple(w//divisor for w in c['weights']),c)
    reps=list(representatives.values());expected=set()
    for state in states:
        if state['source_split']=='train':continue
        for first,second in itertools.permutations(reps,2):
            before=cache[(first['id'],state['id'])];after=cache[(second['id'],state['id'])]
            if before['teacher']!=after['teacher']:expected.add((state['id'],first['id'],second['id']))
    seen=set();counter=Counter()
    with (output/'teacher_changing_swaps.jsonl').open() as stream:
        for line in stream:
            r=json.loads(line);key=(r['scenario_id'],r['before'],r['after']);assert key in expected and key not in seen;seen.add(key)
            before=cache[(r['before'],r['scenario_id'])];after=cache[(r['after'],r['scenario_id'])];a,b=before['winner'],after['winner'];correct_new=int(b==after['teacher'])
            flags=dict(correct_new=correct_new,student_changed=int(a!=b),changed_to_new=int(correct_new and a is not None and b is not None and a!=b),both_endpoints=int(a==before['teacher'] and correct_new),top_set_new=after['top1'])
            assert all(r[k]==v for k,v in flags.items());counter.update(flags)
    assert seen==expected and len(seen)==result['causal']['n']
    for k,v in counter.items():assert result['causal'][k+'_rate']==v/len(seen)
    gates=dict(G0=correct/total>=.95,G1=axis/len(atoms)>=.85,G2=axis>0 and joint/axis>=.90,G3=joint/len(atoms)>=.8,
               G4=result['summary']['alias']['top1']>=.8,G5=result['summary']['alias_composition']['top1']>=.8,G6=counter['correct_new']/len(seen)>=.8,
               G7=result['summary']['literal']['top1']==result['summary']['literal']['exact_winner']==result['summary']['literal']['pairwise']==1 and literal_calls==0 and permutations==result['permutation']['comparisons'])
    assert gates==result['gates'] and all(gates.values())==result['passed']
    return dict(head_probability_maximum_error=maximum,atomic_queries=len(atoms),decision_rows=nrows,permutation_comparisons=permutations,causal_rows=len(seen),gates_reconstructed=True)


def main():
    torch.set_num_threads(4);cfg=json.loads(PROTOCOL.read_text());root=Path(cfg['output_root']);corpus=json.loads((root/'corpus_manifest.json').read_text())
    assert corpus['protocol_sha256']==sha(PROTOCOL)
    for name,h in corpus['files'].items():assert sha(root/name)==h,name
    assert (root/'states.json').read_bytes()==(Path(cfg['source_root'])/'states.json').read_bytes()
    meaning=json.loads((root/'meaning_audit.json').read_text());assert meaning['all_meanings_accepted']
    training=json.loads((root/'training_atoms.json').read_text());validation=json.loads((root/'validation_atoms.json').read_text());g0=json.loads((root/'literal_holdout_atoms.json').read_text())
    final_atoms=json.loads((root/'final_atoms.json').read_text());final_comps=json.loads((root/'final_compositions.json').read_text())
    texts=[{c['text'] for c in group} for group in (training,validation,g0,final_atoms)]
    assert not any(texts[i]&texts[j] for i in range(4) for j in range(i+1,4))
    assert Counter((c['axis'],c['sign']) for c in final_atoms)==Counter({(j,s):16 for j in range(4) for s in (-1,1)})
    evidence=json.loads((root/'frozen_pair_inputs.json').read_text());H=np.load(root/'frozen_pair_features.npy',allow_pickle=False,mmap_mode='r')
    normalizer=np.load(root/'normalizer.npz',allow_pickle=False);rows=[i for i,p in enumerate(evidence['pairs']) if p['criterion'] in texts[0]]
    assert len(rows)==len(training)*4 and np.array_equal(H[rows].mean(0),normalizer['mean']) and np.array_equal(np.maximum(H[rows].std(0),.01),normalizer['std'])
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,use_fast=True);token_count=verify_tokens(evidence,tokenizer)
    assert {p['criterion'] for p in evidence['pairs']}.isdisjoint(texts[3])
    env=json.loads((root/'environment.json').read_text());assert env['corpus_sha256']==sha(root/'corpus_manifest.json') and env['protocol_sha256']==sha(PROTOCOL) and not env['paid_resources']
    line=json.loads((root/'encoder_lineage.json').read_text());assert line['revision']==REVISION and not line['loading']['missing_keys'] and not line['loading']['mismatched_keys']
    results=json.loads((root/'results.json').read_text());selection=json.loads((root/'selection.json').read_text());assert results['selection']==selection and selection['final_openings']==1
    dev=json.loads((root/'development_atoms.json').read_text());comps=json.loads((root/'development_compositions.json').read_text());literals=json.loads((root/'literal_cases.json').read_text());states=json.loads((root/'states.json').read_text());checks={}
    for state in states:
        facts=[dict((key.strip(),int(value)) for key,value in re.findall(r'([^:;]+): (\d+) percent',text)) for text in state['candidates']]
        assert [[f[field] for field in SCHEMA] for f in facts]==state['parsed_facts']
    for name in selection['stages']:
        checkpoint=torch.load(root/(name+'_checkpoint.pt'),map_location='cpu',weights_only=True);details=json.loads((root/(name+'_training.json')).read_text())
        assert details['selected_epoch']==min(range(len(details['trace'])),key=lambda i:details['trace'][i]['validation_CE'])
        if name.startswith('C'):
            assert details['gradient_proof']['encoder']>0 and details['gradient_proof']['head']>0 and details['gradient_proof']['frozen_gradients']==0
            assert details['frozen_prefix_before_sha256']==details['frozen_prefix_after_sha256'] and details['final_layer_before_sha256']!=details['final_layer_after_sha256']
            dh=np.load(root/(name+'_development_features.npy'),allow_pickle=False);de=json.loads((root/(name+'_development_inputs.json')).read_text());token_count+=verify_tokens(de,tokenizer)
        else:
            assert details['encoder_trainable_parameters']==0 and details['first_gradient_L1']>0 and details['frozen_encoder_before_sha256']==details['frozen_encoder_after_sha256'];dh=H;de=evidence
        checks[name]=verify_stage(root,'development_'+name,dev,comps,literals,states,dh,de,checkpoint,normalizer,g0)
    selected=selection['selected'];checkpoint=torch.load(root/(selected+'_checkpoint.pt'),map_location='cpu',weights_only=True)
    final_h=np.load(root/'selected_final_pair_features.npy',allow_pickle=False);final_e=json.loads((root/'selected_final_pair_inputs.json').read_text());token_count+=verify_tokens(final_e,tokenizer)
    checks['final']=verify_stage(root,'final',final_atoms,final_comps,literals,states,final_h,final_e,checkpoint,normalizer,g0)
    assert results['B_STEF_allowed'] is False and results['ASG_closed_permanently'] is True and results['candidate_neural_forwards']==0
    proof=dict(all_checks_pass=True,source_states_byte_identical=True,training_final_texts_disjoint=True,final_atoms=128,final_compositions=128,
               pair_inputs_retokenized=token_count,stages=checks,selected=selected,final_outcome_never_selects_arm=True,exact_literal_neural_calls=0,
               uncalibrated_softmax_not_certificate=True,protocol_sha256=sha(PROTOCOL))
    put(root,'verification.json',canonical(proof)+b'\n');print(json.dumps(proof,indent=2))


if __name__=='__main__':main()
