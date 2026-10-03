"""Frozen embeddings, shared dot-product probes, criterion-grouped inference."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from scipy.stats import kendalltau, pearsonr, spearmanr
from safetensors.torch import save_file
from torch import nn

from build import ROOT, canonical, catalog, validate

ENCODER='microsoft/deberta-v3-xsmall'
REVISION='eb2d654bf0a5b628c8be6c4be7d29118fbef95b8'
CONFIG=dict(encoder=ENCODER, revision=REVISION, pooling='normalized mean',
            dimensions=[32,64],seed=7,epochs=200,batch_size=128,lr=0.001,
            weight_decay=0.0001,loss='squared_error',bootstrap_reps=10000,
            bootstrap_seed=0,token_limit=96)


def load(root):
    manifest=json.loads((root/'manifest.json').read_text())
    for name,h in manifest['files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h, name
    criteria=json.loads((root/'criteria.json').read_text())
    rows=[json.loads(line) for line in (root/'rows.jsonl').read_text().splitlines()]
    validate(criteria,rows)
    return manifest,criteria,rows


def embed(root,device):
    from transformers import AutoModel, AutoTokenizer
    manifest,criteria,rows=load(root)
    texts=sorted({t for r in rows for t in [r['criterion'],*r['candidates']]})
    metadata=dict(encoder=ENCODER,revision=REVISION,pooling=CONFIG['pooling'],
                  texts_sha256=hashlib.sha256(canonical(texts)).hexdigest(),
                  corpus_sha256=manifest['files']['rows.jsonl'])
    cache=root/'embeddings.npz'
    if cache.exists():
        assert json.loads((root/'embeddings_meta.json').read_text())==metadata
        print('Reusing verified embedding cache',flush=True)
        return
    torch.set_num_threads(4)
    torch.manual_seed(7)
    tokenizer=AutoTokenizer.from_pretrained(ENCODER,revision=REVISION)
    encoder=AutoModel.from_pretrained(ENCODER,revision=REVISION,use_safetensors=True).to(device).eval()
    encoder.requires_grad_(False)
    result=[]
    with torch.inference_mode():
        for offset in range(0,len(texts),64):
            tokens=tokenizer(texts[offset:offset+64],padding=True,return_tensors='pt')
            assert tokens['input_ids'].shape[1]<=CONFIG['token_limit'], 'Unexpected truncation'
            tokens={k:v.to(device) for k,v in tokens.items()}
            hidden=encoder(**tokens).last_hidden_state
            mask=tokens['attention_mask'].unsqueeze(-1)
            pooled=(hidden*mask).sum(1)/mask.sum(1)
            pooled=nn.functional.normalize(pooled,p=2,dim=1)
            result.append(pooled.cpu().numpy())
            if offset%1024==0:
                print(f'Embedding {offset}/{len(texts)}',flush=True)
    np.savez_compressed(cache,texts=np.array(texts),values=np.concatenate(result))
    (root/'embeddings_meta.json').write_bytes(canonical(metadata)+b'\n')
    print(f'Embedded {len(texts)} independent texts',flush=True)


class Field(nn.Module):
    def __init__(self,dim,m):
        super().__init__()
        self.criterion=nn.Linear(dim,m,bias=False)
        self.candidate=nn.Linear(dim,m,bias=False)

    def forward(self,q,x):
        return (self.criterion(q)*self.candidate(x)).sum(-1)


def fit(train,E,m):
    torch.manual_seed(CONFIG['seed'])
    model=Field(len(next(iter(E.values()))),m)
    optimizer=torch.optim.AdamW(model.parameters(),lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay'])
    q=[];x=[];y=[]
    for r in train:
        for cand,target in zip(r['candidates'],r['scores']):
            q.append(E[r['criterion']]);x.append(E[cand]);y.append(target)
    q=torch.tensor(np.stack(q));x=torch.tensor(np.stack(x));y=torch.tensor(y,dtype=torch.float32)
    generator=torch.Generator().manual_seed(CONFIG['seed'])
    losses=[]
    for epoch in range(CONFIG['epochs']):
        order=torch.randperm(len(y),generator=generator)
        total=0.0
        for start in range(0,len(y),CONFIG['batch_size']):
            take=order[start:start+CONFIG['batch_size']]
            pred=model(q[take],x[take])
            loss=(pred-y[take]).square().mean()
            optimizer.zero_grad();loss.backward()
            if epoch==0 and start==0:
                assert all(p.grad is not None and p.grad.abs().max()>0 for p in model.parameters())
            optimizer.step()
            total+=float(loss.detach())*len(take)
        losses.append(total/len(y))
        if epoch%50==0 or epoch==CONFIG['epochs']-1:
            print(f'm={m} epoch={epoch} mse={losses[-1]:.6f}',flush=True)
    return model.eval(),losses


def score_one(A,B,q,x):
    # Fixed single-vector operations: no candidate ordering or set reduction.
    cq=A@q
    zx=B@x
    return float(np.sum(cq*zx,dtype=np.float32))


def corr(fn,s,t):
    if np.ptp(s)==0 or np.ptp(t)==0:
        return None
    value=float(fn(s,t).statistic)
    return value if np.isfinite(value) else None


def metrics(scores,gold):
    s=np.array(scores);t=np.array(gold)
    pred=int(np.argmax(s))
    i,j=np.triu_indices(len(s),1)
    agreement=float(np.mean(np.sign(s[i]-s[j])==np.sign(t[i]-t[j])))
    return dict(top1=int(t[pred]==t.max()),pairwise=agreement,
                spearman=corr(spearmanr,s,t),kendall=corr(kendalltau,s,t),
                pearson=corr(pearsonr,s,t))


def paired_ci(records,left,right):
    groups={}
    for r in records:
        groups.setdefault(r['criterion_id'],[]).append(r[left]['top1']-r[right]['top1'])
    means=np.array([np.mean(groups[k]) for k in sorted(groups)])
    rng=np.random.default_rng(CONFIG['bootstrap_seed'])
    draws=rng.integers(len(means),size=(CONFIG['bootstrap_reps'],len(means)))
    diffs=means[draws].mean(1)
    lo,hi=np.quantile(diffs,[.025,.975])
    return dict(delta=float(means.mean()),ci95=[float(lo),float(hi)],criterion_groups=len(means))


def aggregate(records):
    out={}
    for arm in ('cbf32','cbf64','cosine','blind32','blind64'):
        out[arm]={}
        for key in ('top1','pairwise','spearman','kendall','pearson'):
            vals=[r[arm][key] for r in records if r[arm][key] is not None]
            out[arm][key]=float(np.mean(vals)) if vals else None
            out[arm][key+'_undefined']=len(records)-len(vals)
    out['n']=len(records)
    out['cbf32_vs_cosine']=paired_ci(records,'cbf32','cosine')
    out['cbf64_vs_cosine']=paired_ci(records,'cbf64','cosine')
    out['cbf64_vs_cbf32']=paired_ci(records,'cbf64','cbf32')
    return out


def measure(root):
    torch.set_num_threads(1)
    manifest,criteria,rows=load(root)
    cache=np.load(root/'embeddings.npz',allow_pickle=False)
    E=dict(zip(cache['texts'].tolist(),cache['values']))
    train=[r for r in rows if r['split']=='train']
    models={};heads={};losses={};blind={}
    for m in CONFIG['dimensions']:
        model,loss=fit(train,E,m)
        models[m]=model
        A=model.criterion.weight.detach().numpy().copy()
        B=model.candidate.weight.detach().numpy().copy()
        heads[m]=(A,B)
        losses[str(m)]=loss
        unique=sorted({r['criterion'] for r in train})
        blind[m]=np.mean([A@E[q] for q in unique],axis=0)
        save_file(model.state_dict(),str(root/f'm{m}.safetensors'))
    records=[];invariance={32:True,64:True}
    for r in rows:
        if r['split']=='train':
            continue
        q=E[r['criterion']];xs=[E[x] for x in r['candidates']]
        rec={k:r[k] for k in ('id','scenario_id','split','criterion_id','family','K')}
        rec['gold']=r['scores'];rec['scores']={}
        cosine=[float(q@x) for x in xs]
        rec['cosine']=metrics(cosine,r['scores']);rec['scores']['cosine']=cosine
        for m,(A,B) in heads.items():
            sc=[score_one(A,B,q,x) for x in xs]
            control=[float(np.sum(blind[m]*(B@x),dtype=np.float32)) for x in xs]
            rec[f'cbf{m}']=metrics(sc,r['scores'])
            rec[f'blind{m}']=metrics(control,r['scores'])
            rec['scores'][f'cbf{m}']=sc
            rec['scores'][f'blind{m}']=control
            for perm in (list(reversed(range(r['K']))),np.random.default_rng(0).permutation(r['K']).tolist()):
                moved=[score_one(A,B,q,xs[i]) for i in perm]
                restored=[0.0]*r['K']
                for i,v in zip(perm,moved):restored[i]=v
                assert restored==sc, (r['id'],m,'permutation score changed')
                assert int(np.argmax(restored))==int(np.argmax(sc))
        records.append(rec)
    (root/'predictions.jsonl').write_bytes(b''.join(canonical(r)+b'\n' for r in records))
    summaries={}
    for split in ('unseen_wording','unseen_criterion','held_family'):
        rr=[r for r in records if r['split']==split]
        summaries[split]=aggregate(rr)
        summaries[split]['per_K']={str(k):aggregate([r for r in rr if r['K']==k]) for k in (2,4,8,16)}
        summaries[split]['per_family']={f:aggregate([r for r in rr if r['family']==f]) for f in sorted({r['family'] for r in rr})}
    gates={}
    for m in (32,64):
        hard=summaries['held_family']
        positive=sum(v[f'cbf{m}_vs_cosine']['delta']>0 for v in hard['per_family'].values())
        gates[str(m)]=dict(G1=all(summaries[s][f'cbf{m}_vs_cosine']['ci95'][0]>0 for s in ('unseen_criterion','held_family')),
                           G2=positive>=2 and hard[f'cbf{m}_vs_cosine']['delta']>0,
                           positive_held_families=positive,G3=invariance[m])
    dim=summaries['unseen_criterion']['cbf64_vs_cbf32']
    earns64=dim['delta']>=.01 and dim['ci95'][0]>0 and summaries['held_family']['cbf64_vs_cbf32']['delta']>=0
    selected=64 if earns64 else 32
    directions={}
    for m,(A,B) in heads.items():
        cs={c['id']:np.mean([A@E[f] for f in c['forms']],axis=0) for c in criteria}
        diagnostics=[]
        for i,c1 in enumerate(criteria):
            for c2 in criteria[i+1:]:
                a,b=cs[c1['id']],cs[c2['id']]
                cosine=float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))
                diagnostics.append(dict(first=c1['id'],second=c2['id'],same_family=c1['family']==c2['family'],cosine=cosine))
        directions[str(m)]=diagnostics
    (root/'criterion_directions.json').write_bytes(canonical(directions)+b'\n')
    result=dict(config=CONFIG,split_manifest=manifest,splits=summaries,gates=gates,
                G4=dict(selected_dimension=selected,dimension64_earns_cost=earns64,comparison=dim),
                promotion=all(gates[str(selected)][g] for g in ('G1','G2','G3')),
                projection_parameters={str(m):sum(p.numel() for p in models[m].parameters()) for m in models},
                loss_curves=losses,
                artifact_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file() and p.name!='results.json'})
    (root/'results.json').write_bytes(canonical(result)+b'\n')
    for split,v in summaries.items():
        print(split, {a:v[a]['top1'] for a in ('cbf32','cbf64','cosine')},v['cbf32_vs_cosine'],flush=True)
    print(json.dumps(dict(gates=gates,G4=result['G4'],promotion=result['promotion']),indent=2),flush=True)


def smoke():
    perfect=metrics([1,3,2],[1,3,2])
    assert perfect['top1']==perfect['pairwise']==perfect['spearman']==perfect['kendall']==1
    tied=metrics([2,2,1],[2,2,1])
    assert tied['pairwise']==1 and tied['top1']==1
    assert metrics([1,1],[0,1])['pearson'] is None
    rec=[dict(criterion_id='a',cbf32=dict(top1=1),cosine=dict(top1=0)),
         dict(criterion_id='b',cbf32=dict(top1=1),cosine=dict(top1=0))]
    assert paired_ci(rec,'cbf32','cosine')['ci95']==[1.0,1.0]
    model=Field(4,32)
    model(torch.ones(2,4),torch.ones(2,4)).sum().backward()
    assert all(p.grad.abs().max()>0 for p in model.parameters())
    print('Metric, paired-bootstrap, tie, and forward/backward smoke passed',flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--stage',choices=['embed','fit','smoke'],required=True)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=p.parse_args()
    if args.stage=='smoke':smoke()
    elif args.stage=='embed':embed(args.root,args.device)
    else:
        manifest,_,_=load(args.root)
        meta=json.loads((args.root/'embeddings_meta.json').read_text())
        assert meta['corpus_sha256']==manifest['files']['rows.jsonl'] and meta['revision']==REVISION
        protocol=dict(config=CONFIG,manifest=manifest,commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
        (args.root/'run_manifest.json').write_bytes(canonical(protocol)+b'\n')
        measure(args.root)


if __name__=='__main__':main()
