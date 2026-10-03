"""Freeze new CBF7 cohorts without encoding or exposing final to selection."""
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from audit import put
from build import canonical
from schema_grounding_compiler import compile_criterion, parse_criterion

PROTOCOL=Path(__file__).with_name('relation_pretrained_protocol.json')
SOURCE_INVENTORY=Path(__file__).with_name('schema_relation_result_manifest.json')
COHORTS=('training_atoms.json','validation_atoms.json','literal_holdout_atoms.json','development_atoms.json','development_compositions.json','literal_cases.json','states.json')
BASELINE=('A_checkpoint.pt','A_training.json','frozen_pair_features.npy','frozen_pair_inputs.json','normalizer.npz','encoder_lineage.json')


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def prepare():
    cfg=json.loads(PROTOCOL.read_text());source=Path(cfg['source_root']);root=Path(cfg['output_root']);inventory=json.loads(SOURCE_INVENTORY.read_text())
    assert inventory['data_root']==str(source) and inventory['verdict']=='SRE_NOT_EARNED_ON_FRESH_CORPUS'
    assert json.loads((source/'verification.json').read_text())['all_checks_pass']
    for name in COHORTS+BASELINE+('corpus_manifest.json','final_atoms.json','final_compositions.json','verification.json'):
        assert sha(source/name)==inventory['files'][name]['sha256'],name
    if root.exists():raise RuntimeError('Refusing existing CBF7 evidence root')
    root.mkdir(parents=True)
    for name in COHORTS:put(root,name,(source/name).read_bytes())
    # Import/build only after the committed protocol and generator exist.
    from relation_pretrained_corpus import build_final
    atoms,compositions,audit=build_final()
    assert len(atoms)==len(compositions)==128
    assert len({c['id'] for c in atoms+compositions})==256
    assert len({c['text'] for c in atoms})==128
    assert Counter((c['axis'],c['sign']) for c in atoms)==Counter({(a,s):16 for a in range(4) for s in (-1,1)})
    for group in (atoms,compositions):
        counts=Counter(c['pair_id'] for c in group);assert len(counts)==64 and set(counts.values())=={2}
    atomic={c['text'].rstrip('.')+'.':c for c in atoms};uses=Counter();grammar=[]
    for c in atoms+compositions:
        calls=[]
        def teacher(term):
            lo,hi=term['context_span'];q=c['text'][lo:hi].rstrip('.')+'.';label=atomic[q];calls.append(label['id'])
            return dict(axis=label['axis'],sign=label['sign'])
        vector,ir=compile_criterion(c['text'],teacher)
        assert vector==c['weights'] and all(t['literal_axis'] is None for t in parse_criterion(c['text'])),c['id']
        assert len(calls)==(1 if c in atoms else 2)
        if c in compositions:uses.update(calls)
        grammar.append(dict(id=c['id'],weights=vector,atomic_ids=calls,terms=ir))
    assert set(uses)=={c['id'] for c in atoms} and set(uses.values())=={2}
    train=json.loads((root/'training_atoms.json').read_text());validation=json.loads((root/'validation_atoms.json').read_text());g0=json.loads((root/'literal_holdout_atoms.json').read_text())
    groups=[{c['text'] for c in group} for group in (train,validation,g0,atoms)]
    assert not any(groups[i]&groups[j] for i in range(4) for j in range(i+1,4))
    audit['parent_exact_grammar_checks']=grammar;audit['parent_each_atom_composition_uses']=dict(uses)
    prepared=Path(cfg['prepared_corpus_source_root'])
    original_manifest=json.loads((prepared/'corpus_manifest.json').read_text())
    assert not (prepared/'selection.json').exists() and not (prepared/'results.json').exists()
    for name,expected in original_manifest['files'].items():assert sha(prepared/name)==expected,name
    assert canonical(atoms)+b'\n'==(prepared/'final_atoms.json').read_bytes()
    assert canonical(compositions)+b'\n'==(prepared/'final_compositions.json').read_bytes()
    original_audit=json.loads((prepared/'final_corpus_audit.json').read_text())
    for key,value in audit.items():
        if key not in ('provenance','parent_exact_grammar_checks','parent_each_atom_composition_uses'):
            assert original_audit[key]==value,key
    original_provenance=dict(original_audit['provenance'])
    original_provenance.update(output_root=str(root),protocol_sha256=sha(PROTOCOL))
    assert original_provenance==audit['provenance']
    for name in ('corpus_manifest.json','final_corpus_audit.json'):
        put(root,'original_pre_execution_'+name,(prepared/name).read_bytes())
    blind_files=('blind_meaning_predictions.json','blind_meaning_questions.json','blind_meaning_key.json')
    for name in blind_files:put(root,name,(prepared/name).read_bytes())
    audit['blind_meaning_audit']=original_audit['blind_meaning_audit']
    replay=dict(first_preparation_generator_uncommitted=True,first_preparation_git_not_fabricated=True,
                generator_committed_before_replay=True,
                replay_git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).parents[2],text=True).strip(),
                prepared_source_root=str(prepared),atoms_byte_identical=True,compositions_byte_identical=True,
                original_manifest_sha256=sha(prepared/'corpus_manifest.json'),original_audit_sha256=sha(prepared/'final_corpus_audit.json'),
                protocol_sha256=sha(PROTOCOL),final_atoms_sha256=sha(prepared/'final_atoms.json'),final_compositions_sha256=sha(prepared/'final_compositions.json'),
                reason='Immutable evidence writer rejected in-place metadata amendment before any model load; original failure and files retained.')
    audit['preparation_replay']=replay
    for name,values in (('final_atoms.json',atoms),('final_compositions.json',compositions),('final_corpus_audit.json',audit),('preparation_replay.json',replay)):put(root,name,canonical(values)+b'\n')
    meaning=json.loads((prepared/'meaning_audit.json').read_text())
    assert meaning['all_meanings_accepted'] and meaning['correct']==meaning['n']==128 and meaning['ambiguous']==0
    assert meaning['final_atoms_sha256']==sha(root/'final_atoms.json') and meaning['final_compositions_sha256']==sha(root/'final_compositions.json')
    assert meaning['blind_audit_sha256']==sha(prepared/'final_corpus_audit.json')
    meaning['blind_audit_sha256']=sha(root/'final_corpus_audit.json')
    meaning['blind_predictions_sha256']=sha(root/'blind_meaning_predictions.json')
    put(root,'meaning_audit.json',canonical(meaning)+b'\n')
    manifest=dict(experiment='CBF-7',protocol_sha256=sha(PROTOCOL),source_inventory_sha256=sha(SOURCE_INVENTORY),source_measurement_git_revision=inventory['measurement_git_revision'],
                  source_verified_files={name:sha(source/name) for name in COHORTS+BASELINE},data_license_class='shipping-train: authored criteria only',pretrained_model_license_class='conditional/review, not shipping clearance',
                  counts={name:len(json.loads((root/name).read_text())) for name in COHORTS},final_atoms=128,final_compositions=128,final_closed_until_selection=True,
                  preparation_replay=replay,meaning_audit=dict(accepted=True,before_outcomes=True,n=128,correct=128,blind_predictions_sha256=sha(root/'blind_meaning_predictions.json')),
                  files={name:sha(root/name) for name in COHORTS+blind_files+('final_atoms.json','final_compositions.json','final_corpus_audit.json','meaning_audit.json','preparation_replay.json')})
    put(root,'corpus_manifest.json',canonical(manifest)+b'\n');print(json.dumps(manifest,indent=2));return manifest


if __name__=='__main__':prepare()
