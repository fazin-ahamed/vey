"""CBF-5 immutable corpus and isolated-atom features; reuse old encoder evidence."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from audit import put
from build import canonical, digest
from exact_state_encode import encode, semantic_span
from schema_grounding_build import build_cases
from schema_grounding_compiler import parse_criterion

PROTOCOL = Path(__file__).with_name('schema_grounding_protocol.json')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_inputs():
    protocol = json.loads(PROTOCOL.read_text()); source = Path(protocol['source_root'])
    inventory_path = Path(__file__).with_name('exact_state_result_manifest.json')
    inventory = json.loads(inventory_path.read_text()); base = Path(inventory['output_root'])
    hashes = {a['file']: a['sha256'] for a in inventory['artifacts']}
    names = ('criteria.json', 'states.json', 'texts.npy', 'layer1_span.npy', 'encoder_manifest.json',
             'score-corrected/decisions.jsonl', 'score-corrected/results.json')
    verified = {}
    for name in names:
        path = source / name; relative = str(path.relative_to(base))
        assert sha(path) == hashes[relative], name
        verified[relative] = hashes[relative]
    cc = json.loads((source / 'criteria.json').read_text()); ss = json.loads((source / 'states.json').read_text())
    return protocol, source, cc, ss, dict(inventory_sha256=sha(inventory_path), verified_files=verified)


def atom_input(text, atom):
    lo, hi = atom['context_span']; a, b = atom['span']
    context = text[lo:hi].rstrip('.') + '.'
    span = [a - lo, b - lo]
    assert 0 <= span[0] < span[1] <= len(context)
    assert tuple(span) == semantic_span(context), (text, atom, context, span)
    return dict(id=digest([context, span]), text=context, span=span)


def prepare():
    protocol, source, cc, ss, lineage = source_inputs(); root = Path(protocol['output_root'])
    if root.exists(): raise RuntimeError('Refusing to replace a prepared/captured CBF-5 experiment')
    cases = build_cases(cc)
    training = [c for c in cc if c['stratum'] == 'train' and c['template'].startswith('atomic/')
                and sum(w != 0 for w in c['weights']) == 1 and sum(abs(w) for w in c['weights']) == 1]
    assert len(training) == 48
    inputs = {}; query_terms = {}
    for c in training + cases:
        terms = parse_criterion(c['text']); saved = []
        for atom in terms:
            if c in training or atom['literal_axis'] is None:
                spec = atom_input(c['text'], atom); inputs[spec['id']] = spec
                saved.append(dict(atom, input_id=spec['id']))
            else:
                saved.append(dict(atom, input_id=None))
        query_terms[c['id']] = saved
    specs = sorted(inputs.values(), key=lambda x: (x['text'], x['span']))
    texts = np.load(source / 'texts.npy', allow_pickle=False).tolist(); old_lookup = {t: i for i, t in enumerate(texts)}
    old_features = np.load(source / 'layer1_span.npy', allow_pickle=False)
    missing = [s for s in specs if s['text'] not in old_lookup]
    assert len(specs) == 84 and len(missing) == 4, (len(specs), len(missing))
    root.mkdir(); payloads = {'criteria.json': canonical(cases) + b'\n',
                            'prototype_queries.json': canonical(training) + b'\n',
                            'states.json': (source / 'states.json').read_bytes(),
                            'atom_inputs.json': canonical(specs) + b'\n',
                            'query_terms.json': canonical(query_terms) + b'\n'}
    for name, content in payloads.items(): put(root, name, content)
    manifest = dict(experiment='CBF-5', license_class='shipping-train',
                    source='Unchanged CBF-4 exact states/aliases and project-authored compositional criteria',
                    scope='Four-axis controlled semantic-atom screen; reused aliases, not independent replication',
                    protocol_sha256=sha(PROTOCOL), source_lineage=lineage, source_root=str(source),
                    counts=dict(Counter(c['stratum'] for c in cases)), prototype_queries=48,
                    atomic_inputs=len(specs), reused_atomic_inputs=len(specs) - len(missing), new_atomic_inputs=len(missing),
                    files={name: hashlib.sha256(content).hexdigest() for name, content in payloads.items()})
    put(root, 'corpus_manifest.json', canonical(manifest) + b'\n')
    fresh = root / 'atoms'; fresh.mkdir()
    new_bytes = canonical([dict(text=s['text']) for s in missing]) + b'\n'
    put(fresh, 'criteria.json', new_bytes)
    put(fresh, 'corpus_manifest.json', canonical(dict(files={'criteria.json': hashlib.sha256(new_bytes).hexdigest()},
                                                   CBF5_protocol_sha256=sha(PROTOCOL))) + b'\n')
    encode(fresh)
    new_texts = np.load(fresh / 'texts.npy', allow_pickle=False).tolist(); new_lookup = {t: i for i, t in enumerate(new_texts)}
    new_features = np.load(fresh / 'layer1_span.npy', allow_pickle=False)
    H = np.stack([old_features[old_lookup[s['text']]] if s['text'] in old_lookup
                  else new_features[new_lookup[s['text']]] for s in specs])
    np.save(root / 'features.npy', H)
    feature_manifest = dict(inputs=specs, source_feature_sha256=sha(source / 'layer1_span.npy'),
                            source_texts_sha256=sha(source / 'texts.npy'), features_sha256=sha(root / 'features.npy'),
                            new_encoder=json.loads((fresh / 'encoder_manifest.json').read_text()),
                            reused_inputs=80, new_inputs=4, full_composition_forwards=0, candidate_forwards=0,
                            feature='Unchanged CBF-4 normalized layer1 semantic-span readout on isolated atoms')
    put(root, 'feature_manifest.json', canonical(feature_manifest) + b'\n')
    print(json.dumps({k: manifest[k] for k in ('counts', 'prototype_queries', 'atomic_inputs', 'reused_atomic_inputs', 'new_atomic_inputs')}, indent=2))


if __name__ == '__main__': prepare()
