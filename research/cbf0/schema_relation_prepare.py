"""CBF-6 literal atomization and immutable, label-separated study cohorts."""
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from audit import put
from build import canonical, digest
from schema_grounding_capture import source_inputs
from schema_grounding_compiler import AXES, compile_criterion

PROTOCOL = Path(__file__).with_name('schema_relation_protocol.json')


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def literal_atoms(c):
    text=c['text']; template=int(c['template'].split('/')[-1])
    if c['template'].startswith('atomic/'):
        terms=[text.split(' Assign double weight')[0].rstrip('.')]
    elif template==0:
        match=re.fullmatch(r'(.+?) and (.+?)\. Use weights (?:one|two) and (?:one|two) respectively\.',text,re.I)
        assert match,text; terms=[match[1],match[2]]
    elif template==1:
        match=re.fullmatch(r'(reward|penalize) (.+?) with weight (?:one|two), and (reward|penalize) (.+?) with weight (?:one|two)\.',text,re.I)
        assert match,text; terms=[match[1]+' '+match[2],match[3]+' '+match[4]]
    elif template==2:
        match=re.fullmatch(r'Prefer (.+?) \(weight (?:one|two)\) together with (.+?) \(weight (?:one|two)\)\.',text,re.I)
        assert match,text; terms=['Prefer '+match[1],'Prefer '+match[2]]
    elif template==3:
        match=re.fullmatch(r'Score options by (adding|subtracting) (?:one|two) times (.+?) and (adding|subtracting) (?:one|two) times (.+?)\.',text,re.I)
        assert match,text; terms=[match[1]+' '+match[2],match[3]+' '+match[4]]
    elif template==4:
        match=re.fullmatch(r'For (.+?), (more|less) is better with weight (?:one|two)\. For (.+?), (more|less) is better with weight (?:one|two)\.',text,re.I)
        assert match,text; terms=[match[2]+' '+match[1],match[4]+' '+match[3]]
    elif template==5:
        terms=text.split('. Give the first preference')[0].split('; ')
    else: raise ValueError(c)
    output=[]
    for term in terms:
        term=term[:1].upper()+term[1:].rstrip('.')+'.'
        coefficients,records=compile_criterion(term,lambda t: (_ for _ in ()).throw(AssertionError('Literal training atom unresolved')))
        indices=[i for i,w in enumerate(coefficients) if w]
        assert len(indices)==1 and len(records)==1,term
        axis=indices[0]; sign=1 if coefficients[axis]>0 else -1
        assert sign==(1 if c['weights'][axis]>0 else -1) and c['weights'][axis]!=0,(term,c)
        assert not re.search(r'\b(?:one|two|1|2)\b',term,re.I),term
        output.append(dict(text=term,axis=axis,sign=sign,source_id=c['id'],source_template=c['template'],source_weight=c['weights'][axis]))
    assert len(output)==sum(w!=0 for w in c['weights'])
    return output


def training_cohorts(criteria):
    grouped=defaultdict(list)
    for c in criteria:
        if c['stratum']=='train':
            for atom in literal_atoms(c): grouped[atom['text']].append(atom)
    cohorts={'training':[],'validation':[]}
    for text,rows in sorted(grouped.items()):
        labels={(r['axis'],r['sign']) for r in rows}; assert len(labels)==1,(text,labels)
        held=any(int(r['source_template'].split('/')[-1]) in (4,5) for r in rows)
        axis,sign=next(iter(labels)); split='validation' if held else 'training'
        cohorts[split].append(dict(id=digest(text),text=text,axis=axis,sign=sign,provenance=rows))
    g0=[]
    for c in criteria:
        if c['stratum']=='polarity':
            axis=next(i for i,w in enumerate(c['weights']) if w); sign=c['weights'][axis]
            g0.append(dict(id=c['id'],text=c['text'],axis=axis,sign=sign,provenance=dict(source='held_literal_polarity',template=c['template'])))
    assert len(g0)==32
    sets=[{r['text'] for r in group} for group in (cohorts['training'],cohorts['validation'],g0)]
    assert not sets[0]&sets[1] and not (sets[0]|sets[1])&sets[2]
    return cohorts['training'],cohorts['validation'],g0


def prepare():
    protocol=json.loads(PROTOCOL.read_text()); root=Path(protocol['output_root']); source=Path(protocol['source_root'])
    if root.exists(): raise RuntimeError('Refusing prepared SRE evidence overwrite')
    inventory_path=Path(__file__).with_name('schema_grounding_result_manifest.json'); inventory=json.loads(inventory_path.read_text())
    hashes={a['file']:a['sha256'] for a in inventory['artifacts']}
    for name in ('criteria.json','states.json','results.json'): assert sha(source/name)==hashes[name],name
    _,old,old_cc,old_ss,lineage=source_inputs()
    training,validation,g0=training_cohorts(old_cc)
    cc=json.loads((source/'criteria.json').read_text()); dev_atoms=[c for c in cc if c['stratum']=='alias']
    for c in dev_atoms:
        axis=next(i for i,w in enumerate(c['weights']) if w); c.update(axis=axis,sign=c['weights'][axis],pair_id=digest([axis,c['template']]))
    assert len(dev_atoms)==32
    development_compositions=[c for c in cc if c['stratum']=='alias_composition']
    literals=[c for c in cc if c['stratum']=='literal']; assert len(literals)==155
    root.mkdir()
    payloads={'training_atoms.json':training,'validation_atoms.json':validation,'literal_holdout_atoms.json':g0,
              'development_atoms.json':dev_atoms,'development_compositions.json':development_compositions,'literal_cases.json':literals}
    for name,rows in payloads.items(): put(root,name,canonical(rows)+b'\n')
    put(root,'states.json',(source/'states.json').read_bytes())
    from schema_relation_corpus import prepare as prepare_final
    prepare_final(root)
    final=json.loads((root/'final_atoms.json').read_text())
    assert len(final)==128 and not {c['text'] for c in final}&{c['text'] for c in training+validation+g0+dev_atoms}
    files=list(payloads)+['states.json','final_atoms.json','final_compositions.json','final_corpus_audit.json']
    manifest=dict(experiment='CBF-6',protocol_sha256=sha(PROTOCOL),license_class='shipping-train',
                  source_inventory_sha256=sha(inventory_path),source_CBF4_lineage=lineage,
                  source_CBF5_files={n:hashes[n] for n in ('criteria.json','states.json','results.json')},
                  training_source_criteria=sum(c['stratum']=='train' for c in old_cc),
                  counts={n:len(rows) for n,rows in payloads.items()},final_atoms=128,final_compositions=128,
                  relation_class_order=[-1,0,1],schema=list(AXES),files={n:sha(root/n) for n in files},
                  training_criterion_weights_never_in_model_input=True,final_labels_never_in_training=True)
    put(root,'corpus_manifest.json',canonical(manifest)+b'\n'); print(json.dumps(manifest,indent=2))


if __name__=='__main__': prepare()
