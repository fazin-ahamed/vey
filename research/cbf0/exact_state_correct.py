"""Apply pinned alias grammar fixes; encode only new text, never refit a map."""
import hashlib
import json
from pathlib import Path

import numpy as np

from audit import put
from build import canonical
from exact_state_build import PROTOCOL, criteria, validate
from exact_state_encode import ARMS, encode
from exact_state_run import lexical, read_corpus

AMENDMENT = Path(__file__).with_name('exact_state_amendment.json')


def main():
    amendment = json.loads(AMENDMENT.read_text())
    source = Path(json.loads(PROTOCOL.read_text())['output_root']); target = Path(amendment['output_root'])
    if target.exists(): raise RuntimeError('Refusing to overwrite a grammar correction')
    old, states = read_corpus(source); new = criteria(); old_ids = {c['id']: c for c in old}
    repairs = {c['criterion_id']: c for c in amendment['changes']}
    assert len(repairs) == 6 and [c['id'] for c in old] == [c['id'] for c in new]
    for c in new:
        previous = old_ids[c['id']]
        assert {k: v for k, v in c.items() if k != 'text'} == {k: v for k, v in previous.items() if k != 'text'}
        if c['id'] in repairs:
            assert c['text'] == repairs[c['id']]['new'] and previous['text'] == repairs[c['id']]['old']
        else: assert c == previous
    checks = validate(new, states); target.mkdir(); patch = target / 'patch'; patch.mkdir()
    data = canonical(new) + b'\n'; put(target, 'criteria.json', data)
    state_bytes = (source / 'states.json').read_bytes(); put(target, 'states.json', state_bytes)
    corpus = json.loads((source / 'corpus_manifest.json').read_text())
    corpus.update(checks=checks, builder_sha256=hashlib.sha256(Path(__file__).with_name('exact_state_build.py').read_bytes()).hexdigest(),
                  amendment_sha256=hashlib.sha256(AMENDMENT.read_bytes()).hexdigest(), grammar_correction_only=True)
    corpus['files'] = {'criteria.json': hashlib.sha256(data).hexdigest(), 'states.json': hashlib.sha256(state_bytes).hexdigest()}
    put(target, 'corpus_manifest.json', canonical(corpus) + b'\n')
    changed = [c for c in new if c['id'] in repairs]; patch_bytes = canonical(changed) + b'\n'
    put(patch, 'criteria.json', patch_bytes)
    put(patch, 'corpus_manifest.json', canonical(dict(files={'criteria.json': hashlib.sha256(patch_bytes).hexdigest()},
                                                     amendment_sha256=corpus['amendment_sha256'])) + b'\n')
    encode(patch)
    original_texts = np.load(source / 'texts.npy', allow_pickle=False).tolist()
    patch_texts = np.load(patch / 'texts.npy', allow_pickle=False).tolist()
    final_texts = sorted(c['text'] for c in new); np.save(target / 'texts.npy', np.array(final_texts))
    old_index = {t: i for i, t in enumerate(original_texts)}; patch_index = {t: i for i, t in enumerate(patch_texts)}
    final_index = {t: i for i, t in enumerate(final_texts)}; vectors = {}
    for arm in ARMS:
        original = np.load(source / f'{arm}.npy', allow_pickle=False)
        replacements = np.load(patch / f'{arm}.npy', allow_pickle=False)
        features = np.stack([replacements[patch_index[t]] if t in patch_index else original[old_index[t]] for t in final_texts])
        for c in new:
            if c['stratum'] == 'train': assert np.array_equal(features[final_index[c['text']]], original[old_index[c['text']]])
        np.save(target / f'{arm}.npy', features)
        weight_bytes = (source / f'{arm}_weights.npy').read_bytes(); put(target, f'{arm}_weights.npy', weight_bytes)
        W = np.load(target / f'{arm}_weights.npy', allow_pickle=False)
        vectors[arm] = features[[final_index[c['text']] for c in new]].astype(np.float64) @ W
    put(target, 'fit.json', (source / 'fit.json').read_bytes())
    vv = [lexical(c['text']) for c in new]
    vectors['lexical'] = np.array([v if v is not None else np.full(4, np.nan) for v in vv])
    np.savez(target / 'criterion_vectors.npz', ids=np.array([c['id'] for c in new]),
             gold=np.array([c['weights'] for c in new], dtype=np.float64), **vectors)
    obsolete = {c['old'] for c in amendment['changes']}; removed_tokens = 0
    initial = json.loads((source / 'encoder_manifest.json').read_text()); extra = json.loads((patch / 'encoder_manifest.json').read_text())
    for shard in initial['raw_shards']:
        with np.load(source / shard['file'], allow_pickle=False) as raw:
            removed_tokens += sum(int(n) for i, n in zip(raw['indices'], raw['lengths']) if original_texts[int(i)] in obsolete)
    combined = dict(initial)
    combined.update(raw_shards=[dict(s, file='../' + s['file'], texts_file='../texts.npy') for s in initial['raw_shards']] +
                                [dict(s, file='patch/' + s['file'], texts_file='patch/texts.npy') for s in extra['raw_shards']],
                    forwards=initial['forwards'] + extra['forwards'], forwarded_criteria_total=initial['criteria'] + extra['criteria'],
                    tokens=initial['tokens'] - removed_tokens + extra['tokens'], captured_tokens=initial['tokens'] + extra['tokens'],
                    superseded_texts=sorted(obsolete), corpus_sha256=hashlib.sha256((target / 'corpus_manifest.json').read_bytes()).hexdigest(),
                    amendment_sha256=corpus['amendment_sha256'])
    put(target, 'encoder_manifest.json', canonical(combined) + b'\n')
    proof = dict(changed_questions=len(changed), new_encoder_texts=len(patch_texts), encoder_forwards=extra['forwards'],
                 refits=0, training_features_unchanged=True, weights_and_ridge_choices_byte_identical=True,
                 original_literal_questions_unchanged=True, final_questions=len(final_texts), final_valid_tokens=combined['tokens'])
    put(target, 'correction.json', canonical(proof) + b'\n'); print(json.dumps(proof, indent=2))


if __name__ == '__main__': main()
