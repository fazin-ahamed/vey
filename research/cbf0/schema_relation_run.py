"""CBF-6 staged frozen heads, one-layer adaptation, and one final opening."""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from audit import put
from build import canonical
from schema_relation_encoder import SCHEMA, encode_pairs, head, load_encoder, set_adaptation
from schema_relation_evaluate import evaluate, semantic_contexts, uncertainty
from schema_relation_prepare import PROTOCOL, sha


def load(root,name): return json.loads((root/name).read_text())


def resource_snapshot():
    fields={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line}
    return dict(available_RAM_MiB=fields['MemAvailable']/1024,swap_used_MiB=(fields['SwapTotal']-fields['SwapFree'])/1024,load1=os.getloadavg()[0])


def guard_resources():
    snapshot=resource_snapshot()
    while snapshot['available_RAM_MiB']<4096 or snapshot['swap_used_MiB']>INITIAL_SWAP+256:
        torch.set_num_threads(2)
        print(json.dumps(dict(resource_pause=snapshot)),flush=True)
        time.sleep(5); snapshot=resource_snapshot()
    if snapshot['load1']>max(4,os.cpu_count()*.75):
        torch.set_num_threads(2); time.sleep(1)


INITIAL_SWAP=resource_snapshot()['swap_used_MiB']


def pairs_for(atoms):
    return [(c['text'],field) for c in atoms for field in SCHEMA],torch.tensor([c['sign']+1 if j==c['axis'] else 1 for c in atoms for j in range(4)],dtype=torch.long)


def state_copy(module): return {k:v.detach().cpu().clone() for k,v in module.state_dict().items()}


def tensor_hash(parameters):
    digest=hashlib.sha256()
    for name,p in parameters:
        digest.update(name.encode()); digest.update(p.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def reversal_indices(atoms):
    # No semantic label or alias lookup: literal templates stripped of their
    # polarity cue define training-only positive/negative symmetry partners.
    import re
    groups={}
    for i,c in enumerate(atoms):
        text=c['text'].lower()
        text=re.sub(r'\b(?:maximize|minimize|higher|lower|reward|penalize|more|less|positive|negative|adding|subtracting)\b','<polarity>',text)
        groups.setdefault((c['axis'],text),{})[c['sign']]=i
    result=[(members[1]*4+j,members[-1]*4+j) for members in groups.values() if set(members)=={-1,1} for j in range(4)]
    assert result
    return torch.tensor(result,dtype=torch.long)


def frozen_train(kind,X,y,V,vy,config,rev=None):
    torch.manual_seed(config['head_seed']); module=head(kind); optimizer=torch.optim.AdamW(module.parameters(),lr=config['head_lr'],weight_decay=config['head_weight_decay'])
    weights=torch.tensor(config['class_weights']); best=float('inf'); best_state=None; trace=[]; gradient=None
    for epoch in range(config['head_epochs']):
        if epoch%25==0: guard_resources()
        module.train(); optimizer.zero_grad(set_to_none=True); logits=module(X)
        ce=nn.functional.cross_entropy(logits,y,weight=weights); symmetry=torch.tensor(0.)
        if rev is not None:
            probabilities=logits.softmax(-1); symmetry=((probabilities[rev[:,0]]-probabilities[rev[:,1]][:,[2,1,0]])**2).mean()
        loss=ce+.1*symmetry; assert torch.isfinite(loss); loss.backward()
        if epoch==0:
            gradient=float(sum(p.grad.abs().sum() for p in module.parameters())); assert gradient>0
        optimizer.step(); module.eval()
        with torch.no_grad():
            val_logits=module(V); validation=float(nn.functional.cross_entropy(val_logits,vy)); accuracy=float((val_logits.argmax(1)==vy).float().mean())
        trace.append(dict(epoch=epoch,loss=float(loss.detach()),CE=float(ce.detach()),reversal=float(symmetry.detach()),validation_CE=validation,validation_accuracy=accuracy))
        if validation<best: best=validation; best_state=state_copy(module); selected_epoch=epoch
    module.load_state_dict(best_state,strict=True); module.eval()
    return module,dict(kind=kind,trainable_parameters=sum(p.numel() for p in module.parameters()),epochs=config['head_epochs'],selected_epoch=selected_epoch,validation_CE=best,first_gradient_L1=gradient,trace=trace,encoder_trainable_parameters=0,loss='CE+0.1*reversal' if rev is not None else 'CE')


def matrices_for(module,H,mean,std,texts):
    with torch.no_grad(): p=module(torch.tensor((H-mean)/std,dtype=torch.float32)).softmax(-1).numpy()
    assert p.shape==(4*len(texts),3)
    return {q:p[4*i:4*i+4] for i,q in enumerate(texts)}


def eligible_reversal(result):
    a=result['atoms']; s=result['summary']; sign=a['conditional_sign_accuracy']
    return result['literal_relation']['accuracy']>=.95 and a['axis_accuracy']>=.85 and sign is not None and .75<=sign<.90 and a['joint_accuracy']>=.65 and s['alias']['top1']>=.65 and s['alias_composition']['top1']>=.65


def adaptation_train(module,model,tokenizer,training,validation,mean,std,recipe,rev=False):
    torch.manual_seed(recipe['seed']); set_adaptation(model); module=module.to('cuda'); model.eval(); module.train()
    trainable=[(name,p) for name,p in model.named_parameters() if p.requires_grad]; frozen=[(name,p) for name,p in model.named_parameters() if not p.requires_grad]
    assert trainable and all(name.startswith(f'encoder.layer.{len(model.encoder.layer)-1}.') for name,p in trainable)
    before=tensor_hash(frozen); before_last=tensor_hash(trainable); before_head=tensor_hash(module.named_parameters())
    mean_gpu=torch.tensor(mean,device='cuda'); std_gpu=torch.tensor(std,device='cuda'); weights=torch.tensor([4.,2/3,4.],device='cuda')
    pairs,labels=pairs_for(training); val_pairs,val_labels=pairs_for(validation)
    optimizer=torch.optim.AdamW([dict(params=[p for _,p in trainable],lr=recipe['encoder_lr']),dict(params=module.parameters(),lr=recipe['head_lr'])],weight_decay=recipe['weight_decay'])
    generator=torch.Generator().manual_seed(recipe['seed']); best=float('inf'); best_head=None; best_layer=None; trace=[]; grad_proof=None
    rev_pairs=reversal_indices(training) if rev else None
    def logits_for(indices):
        pp=[pairs[i] for i in indices]; encoded=tokenizer([q for q,f in pp],text_pair=[f for q,f in pp],padding=True,truncation=False,return_tensors='pt')
        assert encoded['input_ids'].shape[1]<=128
        H=model(**{k:v.to('cuda') for k,v in encoded.items()}).last_hidden_state[:,0,:]
        return module((H-mean_gpu)/std_gpu)
    for epoch in range(recipe['epochs']):
        guard_resources()
        order=torch.randperm(len(pairs),generator=generator).tolist(); losses=[]
        for start in range(0,len(order),recipe['batch_size']):
            indices=order[start:start+recipe['batch_size']]; optimizer.zero_grad(set_to_none=True)
            logits=logits_for(indices); loss=nn.functional.cross_entropy(logits,labels[indices].to('cuda'),weight=weights)
            if rev:
                # Fixed complete literal symmetry population, independent of
                # minibatch membership; both endpoints receive gradients.
                endpoints=torch.unique(rev_pairs.flatten()).tolist(); mapping={k:i for i,k in enumerate(endpoints)}
                probs=logits_for(endpoints).softmax(-1)
                left=torch.tensor([mapping[int(i)] for i in rev_pairs[:,0]],device='cuda'); right=torch.tensor([mapping[int(i)] for i in rev_pairs[:,1]],device='cuda')
                loss=loss+.1*((probs[left]-probs[right][:,[2,1,0]])**2).mean()
            assert torch.isfinite(loss); loss.backward()
            if grad_proof is None:
                grad_proof=dict(encoder=float(sum(p.grad.abs().sum() for _,p in trainable if p.grad is not None)),head=float(sum(p.grad.abs().sum() for p in module.parameters() if p.grad is not None)),frozen_gradients=sum(p.grad is not None for _,p in frozen))
                assert grad_proof['encoder']>0 and grad_proof['head']>0 and grad_proof['frozen_gradients']==0
            nn.utils.clip_grad_norm_([p for _,p in trainable]+list(module.parameters()),recipe['gradient_clip']); optimizer.step(); losses.append(float(loss.detach()))
        module.eval(); vh,_=encode_pairs(val_pairs,tokenizer,model)
        with torch.no_grad(): vl=module((torch.tensor(vh,device='cuda')-mean_gpu)/std_gpu).cpu(); vce=float(nn.functional.cross_entropy(vl,val_labels)); vacc=float((vl.argmax(1)==val_labels).float().mean())
        trace.append(dict(epoch=epoch,loss=float(np.mean(losses)),validation_CE=vce,validation_accuracy=vacc)); module.train()
        if vce<best: best=vce; best_head=state_copy(module); best_layer=state_copy(model.encoder.layer[-1]); selected_epoch=epoch
    model.encoder.layer[-1].load_state_dict(best_layer,strict=True); module.load_state_dict(best_head,strict=True); module=module.cpu().eval(); model.eval()
    after=tensor_hash(frozen); assert before==after
    assert before_last!=tensor_hash(trainable) and before_head!=tensor_hash(module.named_parameters())
    details=dict(kind='linear',selected_epoch=selected_epoch,validation_CE=best,epochs=recipe['epochs'],trace=trace,gradient_proof=grad_proof,
                 encoder_trainable_parameters=sum(p.numel() for _,p in trainable),head_trainable_parameters=sum(p.numel() for p in module.parameters()),
                 trainable_encoder_names=[name for name,p in trainable],frozen_prefix_before_sha256=before,frozen_prefix_after_sha256=after,
                 final_layer_before_sha256=before_last,final_layer_after_sha256=tensor_hash(trainable),loss='CE+0.1*reversal' if rev else 'CE')
    return module,best_layer,details


def smoke():
    torch.set_num_threads(4); torch.manual_seed(7)
    for kind in ('linear','mlp'):
        m=head(kind); X=torch.randn(12,384); y=torch.arange(12)%3; loss=nn.functional.cross_entropy(m(X),y); loss.backward()
        assert torch.isfinite(loss) and sum(float(p.grad.abs().sum()) for p in m.parameters())>0
    from schema_relation_evaluate import decode
    for j in range(4):
        for sign in (-1,1):
            p=np.zeros((4,3));p[:,1]=1;p[j]=0;p[j,sign+1]=1
            assert (decode(p)['axis'],decode(p)['sign'])==(j,sign)
    assert decode(np.tile([0.,1.,0.],(4,1))) is None
    rows=[dict(text=op+' '+field+'.',axis=j,sign=s) for j,field in enumerate(SCHEMA) for s,op in ((1,'Maximize'),(-1,'Minimize'))]
    rev=reversal_indices(rows); assert rev.shape==(16,2)
    p=torch.tensor([[.1,.2,.7],[.7,.2,.1]])
    assert ((p[0]-p[1][[2,1,0]])**2).sum()==0
    print('Runtime smoke PASS: both head gradients, ternary decoder, UNKNOWN, literal-only reversal partners.')


def main():
    torch.set_num_threads(4); torch.manual_seed(7)
    cfg=json.loads(PROTOCOL.read_text()); root=Path(cfg['output_root']); corpus=load(root,'corpus_manifest.json')
    assert corpus['protocol_sha256']==sha(PROTOCOL)
    for name,h in corpus['files'].items(): assert sha(root/name)==h,name
    audit_path=root/'meaning_audit.json'; audit=load(root,'meaning_audit.json'); assert audit['all_meanings_accepted'] and audit['final_atoms_sha256']==sha(root/'final_atoms.json')
    if (root/'selection.json').exists(): raise RuntimeError('Refusing stage/final evidence overwrite')
    training=load(root,'training_atoms.json'); validation=load(root,'validation_atoms.json'); g0=load(root,'literal_holdout_atoms.json')
    dev=load(root,'development_atoms.json'); comps=load(root,'development_compositions.json'); literals=load(root,'literal_cases.json'); states=load(root,'states.json')
    dev_texts=semantic_contexts(dev,comps); score_texts=sorted(set(dev_texts)|{c['text'] for c in g0})
    all_texts=sorted({c['text'] for c in training+validation}|set(score_texts)); all_pairs=[(q,f) for q in all_texts for f in SCHEMA]
    guard_resources()
    environment=dict(git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                     python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),
                     cuda=torch.version.cuda,device=torch.cuda.get_device_name(0),threads=torch.get_num_threads(),
                     resource_before=resource_snapshot(),protocol_sha256=sha(PROTOCOL),corpus_sha256=sha(root/'corpus_manifest.json'),
                     seed=7,paid_resources=False,meaning_audit_sha256=sha(audit_path))
    put(root,'environment.json',canonical(environment)+b'\n')
    tokenizer,model,lineage=load_encoder(); put(root,'encoder_lineage.json',canonical(lineage)+b'\n')
    H,evidence=encode_pairs(all_pairs,tokenizer,model)
    np.save(root/'frozen_pair_features.npy',H); put(root,'frozen_pair_inputs.json',canonical(evidence)+b'\n')
    lookup={q:i for i,q in enumerate(all_texts)}
    def features(texts): return np.concatenate([H[4*lookup[q]:4*lookup[q]+4] for q in texts],axis=0)
    train_h=features([c['text'] for c in training]); val_h=features([c['text'] for c in validation]); score_h=features(score_texts)
    mean=train_h.mean(0); std=np.maximum(train_h.std(0),.01); np.savez(root/'normalizer.npz',mean=mean,std=std)
    X=torch.tensor((train_h-mean)/std); V=torch.tensor((val_h-mean)/std); _,y=pairs_for(training); _,vy=pairs_for(validation)
    outcomes={}; checkpoints={}; selected=None; reversal_used=False; a_state=None
    def record(name,module,details,layer=None,score_features=None):
        torch.save(dict(head=state_copy(module),last_layer=layer,kind=details['kind']),root/(name+'_checkpoint.pt'))
        put(root,name+'_training.json',canonical(details)+b'\n'); matrices=matrices_for(module,score_h if score_features is None else score_features,mean,std,score_texts)
        outcome=evaluate(root,'development_'+name,dev,comps,literals,states,matrices,g0); outcomes[name]=outcome; checkpoints[name]=(details['kind'],state_copy(module),layer)
        print(json.dumps(dict(stage=name,literal_relation=outcome['literal_relation'],atoms=outcome['atoms'],summary=outcome['summary'],gates=outcome['gates']),indent=2),flush=True)
        return outcome
    for name,kind in (('A','linear'),('B','mlp')):
        module,details=frozen_train(kind,X,y,V,vy,cfg['training']); outcome=record(name,module,details)
        if name=='A': a_state=state_copy(module)
        if outcome['passed']: selected=name; break
        if not reversal_used and eligible_reversal(outcome):
            reversal_used=True; rev_module,rev_details=frozen_train(kind,X,y,V,vy,cfg['training'],reversal_indices(training))
            rev_outcome=record(name+'_reversal',rev_module,rev_details)
            if rev_outcome['passed']: selected=name+'_reversal'; break
    if selected is None:
        # A and B have both failed development; adaptation keeps A's head shape.
        model.encoder.layer[-1].requires_grad_(False); module=head('linear'); module.load_state_dict(a_state,strict=True)
        module,layer,details=adaptation_train(module,model,tokenizer,training,validation,mean,std,cfg['progression']['C_recipe'])
        adapted_h,adapted_evidence=encode_pairs([(q,f) for q in score_texts for f in SCHEMA],tokenizer,model)
        np.save(root/'C_development_features.npy',adapted_h); put(root,'C_development_inputs.json',canonical(adapted_evidence)+b'\n')
        outcome=record('C',module,details,layer,adapted_h)
        if outcome['passed']: selected='C'
        elif not reversal_used and eligible_reversal(outcome):
            reversal_used=True; tokenizer,rev_model,_=load_encoder(); rev_module=head('linear'); rev_module.load_state_dict(a_state,strict=True)
            rev_module,rev_layer,rev_details=adaptation_train(rev_module,rev_model,tokenizer,training,validation,mean,std,cfg['progression']['C_recipe'],True)
            rev_h,rev_evidence=encode_pairs([(q,f) for q in score_texts for f in SCHEMA],tokenizer,rev_model)
            np.save(root/'C_reversal_development_features.npy',rev_h); put(root,'C_reversal_development_inputs.json',canonical(rev_evidence)+b'\n')
            rev_outcome=record('C_reversal',rev_module,rev_details,rev_layer,rev_h)
            if rev_outcome['passed']: selected='C_reversal'
    if selected is None:
        selected=max(outcomes,key=lambda name:(outcomes[name]['atoms']['joint_accuracy'],outcomes[name]['summary']['alias']['top1'],-list(outcomes).index(name)))
    selection=dict(selected=selected,development_passed=outcomes[selected]['passed'],stages=list(outcomes),reversal_triggered=reversal_used,
                   final_openings=1,final_outcomes_never_select_arm=True,meaning_audit_sha256=sha(audit_path),protocol_sha256=sha(PROTOCOL))
    put(root,'selection.json',canonical(selection)+b'\n')
    # Only after selection is immutable do final text/labels enter evaluation.
    final_atoms=load(root,'final_atoms.json'); final_comps=load(root,'final_compositions.json')
    final_texts=sorted(set(semantic_contexts(final_atoms,final_comps))|{c['text'] for c in g0})
    kind,head_state,layer=checkpoints[selected]
    if layer is None:
        del model; torch.cuda.empty_cache(); tokenizer,model,_=load_encoder()
    else: model.encoder.layer[-1].load_state_dict(layer,strict=True)
    module=head(kind); module.load_state_dict(head_state,strict=True); module.eval()
    final_h,final_evidence=encode_pairs([(q,f) for q in final_texts for f in SCHEMA],tokenizer,model)
    np.save(root/'selected_final_pair_features.npy',final_h); put(root,'selected_final_pair_inputs.json',canonical(final_evidence)+b'\n')
    matrices=matrices_for(module,final_h,mean,std,final_texts)
    final=evaluate(root,'final',final_atoms,final_comps,literals,states,matrices,g0); ci=uncertainty(root,'final',final_atoms,final_comps)
    result=dict(experiment='CBF-6',selection=selection,development={name:dict(literal_relation=o['literal_relation'],atoms=o['atoms'],summary=o['summary'],gates=o['gates']) for name,o in outcomes.items()},
                final=final,final_intervals=ci,verdict='SRE_SINGLE_SEED_SCREEN_PASS_REPLICATION_REQUIRED' if final['passed'] else 'SRE_NOT_EARNED_ON_FRESH_CORPUS',
                ASG_closed_permanently=True,B_STEF_allowed=False,B_STEF_next_requirement='Separate architecture-specific deployment foldability study, never automatic old B-STEF launch',candidate_neural_forwards=0,exact_compiler_unchanged=True)
    put(root,'results.json',canonical(result)+b'\n'); print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--smoke',action='store_true'); args=parser.parse_args()
    if args.smoke: smoke()
    else: main()
