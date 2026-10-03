"""Deterministic numeric criterion-conditional CBF-0 experiment."""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path

ROOT = Path('/home/fazinahamed/Documents/vey-data/decisionmix/d3/cbf/corrected-v1')
# attribute label, direction, four equivalent preference descriptions
CATALOG = {
 'preference': [('comfort',1),('convenience',1),('flexibility',1),('simplicity',1)],
 'risk': [('financial loss',-1),('reputational damage',-1),('security exposure',-1),('operational disruption',-1)],
 'urgency': [('response delay',-1),('deadline lateness',-1),('time sensitivity',1),('escalation priority',1)],
 'relevance': [('topic relevance',1),('request alignment',1),('evidence pertinence',1),('context relevance',1)],
 'suitability': [('beginner suitability',1),('expert suitability',1),('outdoor suitability',1),('team suitability',1)],
 'quality': [('durability',1),('reliability',1),('precision',1),('craftsmanship',1)],
 'cost': [('purchase expense',-1),('operating expense',-1),('maintenance expense',-1),('delivery expense',-1)],
 'safety': [('injury likelihood',-1),('fire hazard',-1),('chemical hazard',-1),('accident likelihood',-1)],
 'plausibility': [('claim credibility',1),('plan feasibility',1),('causal plausibility',1),('estimate realism',1)],
 'sentiment': [('approval',1),('hostility',-1),('optimism',1),('dissatisfaction',-1)],
 'compliance': [('legal conformity',1),('policy adherence',1),('regulatory conformity',1),('contract adherence',1)],
 'similarity': [('visual resemblance',1),('functional resemblance',1),('stylistic resemblance',1),('semantic resemblance',1)],
 'intent-routing': [('support channel fit',1),('sales channel fit',1),('billing channel fit',1),('technical channel fit',1)],
}
HELD = {'compliance','similarity','intent-routing'}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',',':')).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def catalog():
    out = []
    for family, attrs in CATALOG.items():
        for i, (attribute, direction) in enumerate(attrs):
            if direction > 0:
                forms = [f'Prefer greater {attribute}.', f'Maximize {attribute}.',
                         f'Choose the candidate with the highest {attribute}.', f'More {attribute} is better.']
            else:
                forms = [f'Prefer lower {attribute}.', f'Minimize {attribute}.',
                         f'Choose the candidate with the least {attribute}.', f'Less {attribute} is better.']
            out.append(dict(id=f'{family}/{attribute}', family=family, attribute=attribute,
                            direction=direction, forms=forms,
                            role='held_family' if family in HELD else ('train' if i<2 else 'unseen')))
    return out


def oracle(facts, criterion):
    vals = [f[criterion['attribute']] / 100 for f in facts]
    return vals if criterion['direction'] == 1 else [1-v for v in vals]


def build():
    criteria = catalog()
    rows = []
    for split, seed, reps in [('train',10,8),('unseen_wording',20,4),
                              ('unseen_criterion',30,4),('held_family',40,4)]:
        for family, attrs in CATALOG.items():
            if (family in HELD) != (split == 'held_family'):
                continue
            chosen = [c for c in criteria if c['family']==family and
                      (c['role']=='held_family' if split=='held_family' else
                       c['role']==('unseen' if split=='unseen_criterion' else 'train'))]
            for K in (2,4,8,16):
                for rep in range(reps):
                    sid = f'{split}/{family}/{K}/{rep}'
                    rng = random.Random(int(digest([seed,sid])[:16],16))
                    cols = {attr:rng.sample(range(1,100),K) for attr,_ in attrs}
                    facts = [{attr:cols[attr][i] for attr,_ in attrs} for i in range(K)]
                    texts = []
                    for f in facts:
                        keys = list(f)
                        rng.shuffle(keys)
                        texts.append('; '.join(f'{key}: {f[key]} percent' for key in keys)+'.')
                    for c in chosen:
                        forms = c['forms'][:2] if split=='train' else (c['forms'][2:] if split=='unseen_wording' else c['forms'])
                        for form in forms:
                            rows.append(dict(id=digest([sid,c['id'],form]), scenario_id=sid,
                                             split=split, criterion_id=c['id'], family=family,
                                             criterion=form, K=K, candidates=texts, facts=facts,
                                             scores=oracle(facts,c)))
    validate(criteria,rows)
    return criteria,rows


def validate(criteria,rows):
    by_id = {c['id']:c for c in criteria}
    train = {r['criterion_id'] for r in rows if r['split']=='train'}
    train_forms = {r['criterion'] for r in rows if r['split']=='train'}
    seen_inputs = {}
    winners = {}
    for r in rows:
        c = by_id[r['criterion_id']]
        parsed = [{a: int(v) for a, v in re.findall(r'([^;]+): (\d+) percent', text.rstrip('.'))}
                  for text in r['candidates']]
        parsed = [{a.strip(): v for a, v in f.items()} for f in parsed]
        assert parsed == r['facts'], 'Rendered facts disagree with latent facts'
        assert r['scores']==oracle(r['facts'],c)
        assert len(set(r['candidates']))==r['K']
        assert len(set(r['scores']))==r['K']
        assert r['criterion'] in c['forms']
        if r['split'] in ('unseen_criterion','held_family'):
            assert c['id'] not in train and r['criterion'] not in train_forms
        if r['split']=='held_family':
            assert c['family'] in HELD
        else:
            assert c['family'] not in HELD
        assert oracle(list(reversed(r['facts'])),c)==list(reversed(r['scores']))
        for x,y in zip(r['candidates'],r['scores']):
            key=(r['criterion'],x)
            assert key not in seen_inputs or seen_inputs[key]==y
            seen_inputs[key]=y
        win=max(range(r['K']),key=lambda i:r['scores'][i])
        winners.setdefault(r['scenario_id'],set()).add(win)
    assert len({r['id'] for r in rows})==len(rows)
    for split in ('train','unseen_wording','unseen_criterion','held_family'):
        ws=[v for sid,v in winners.items() if sid.startswith(split+'/')]
        assert sum(len(v)>1 for v in ws)>len(ws)//2, (split,'criterion-blind targets')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args()
    criteria,rows=build()
    args.root.mkdir(parents=True,exist_ok=True)
    payloads={'criteria.json':canonical(criteria)+b'\n',
              'rows.jsonl':b''.join(canonical(r)+b'\n' for r in rows)}
    for name,content in payloads.items():
        path=args.root/name
        if path.exists() and path.read_bytes()!=content:
            raise RuntimeError(f'Refusing to overwrite a changed artifact: {path}')
        path.write_bytes(content)
    manifest=dict(source='project-generated numeric multiattribute oracle',
                  license_class='shipping-train', scope='synthetic numeric preferences; not natural-domain parity',
                  seed_recipe={'train':10,'unseen_wording':20,'unseen_criterion':30,'held_family':40},
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  files={name:hashlib.sha256(content).hexdigest() for name,content in payloads.items()},
                  counts={s:sum(r['split']==s for r in rows) for s in sorted({r['split'] for r in rows})},
                  criteria={role:sum(c['role']==role for c in criteria) for role in ('train','unseen','held_family')})
    (args.root/'manifest.json').write_bytes(canonical(manifest)+b'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()
