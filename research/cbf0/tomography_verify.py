"""Verify all persisted readouts and decoders without another encoder forward."""
import hashlib
import json
from pathlib import Path

import numpy as np

from audit import put
from build import canonical
from tomography_encode import POOLS, spans

ROOT = Path('/home/fazinahamed/Documents/vey-data/decisionmix/d3/cbf/tomography-v1')


def main():
    root = ROOT
    meta = json.loads((root / 'encoder_manifest.json').read_text())
    texts = np.load(root / 'texts.npy', allow_pickle=False).tolist()
    indices = {t: i for i, t in enumerate(texts)}
    layers = meta['layers']
    features = {(layer, pool): np.load(root / f'L{layer:02d}_{pool}.npy', mmap_mode='r')
                for layer in range(layers) for pool in POOLS}
    maximum, seen, token_count = 0., set(), 0
    for shard in meta['raw_shards']:
        with np.load(root / shard['file'], allow_pickle=False) as raw:
            states = raw['hidden']; lens = raw['lengths']; start = 0
            assert states.dtype == np.float32 and states.shape[1:] == (layers, 384)
            for index, length in zip(raw['indices'], lens):
                index, length = int(index), int(length)
                assert index not in seen; seen.add(index)
                part = states[start:start + length]
                offset = raw['offsets'][start:start + length]
                special = raw['special'][start:start + length].astype(bool)
                info = spans(texts[index])
                for pool in POOLS:
                    mask = raw[pool][start:start + length].astype(bool)
                    if pool == 'shipped':
                        assert mask.all()
                    elif pool == 'cls':
                        assert mask[0] and not mask[1:].any() and special[0]
                    elif pool == 'content':
                        assert np.array_equal(mask, ~special)
                    else:
                        expected = np.zeros(length, dtype=bool)
                        for lo, hi in info[pool]:
                            expected |= (offset[:, 0] < hi) & (offset[:, 1] > lo)
                        expected &= ~special
                        assert np.array_equal(mask, expected), (index, pool)
                    if mask.any():
                        pooled = part[mask].mean(0, dtype=np.float32)
                        pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
                    else:
                        pooled = np.zeros((layers, 384), dtype=np.float32)
                    for layer in range(layers):
                        error = float(np.max(np.abs(pooled[layer] - features[layer, pool][index])))
                        maximum = max(maximum, error)
                start += length
            assert start == len(states)
            token_count += start
    assert seen == set(range(len(texts))) and maximum <= 1e-6
    suite = json.loads((root / 'layer_results.json').read_text())
    candidate_checks = 0
    for name in suite['candidate_interfaces']:
        layer, pool = int(name[1:3]), name[4:]
        with np.load(root / f'candidate_{name}.npz', allow_pickle=False) as saved:
            inds = [indices[t] for t in saved['texts'].tolist()]
            H = features[layer, pool][inds].astype(np.float64)
            for arm in ('absolute', 'difference'):
                assert np.array_equal(H @ saved[arm + '_weights'], saved[arm + '_attributes'])
                candidate_checks += len(H)
    query_checks = 0
    for name in suite['criterion_interfaces']:
        layer, pool = int(name[1:3]), name[4:]
        with np.load(root / f'criterion_{name}.npz', allow_pickle=False) as weights:
            for q in map(json.loads, (root / f'criterion_{name}_queries.jsonl').read_text().splitlines()):
                fold = -1 if q['fold'] == 'original' else int(q['fold'])
                h = features[layer, pool][indices[q['query']]].astype(np.float64)
                B = weights['projection_' + str(q['group'])]
                h = h - (h @ B.T) @ B
                pred = h @ weights[f'weights_{q["group"]}_{fold}']
                assert pred.tolist() == q['predicted']
                query_checks += 1
    assert len(suite['candidate_interfaces']) == layers * 5
    assert len(suite['criterion_interfaces']) == layers * 4
    assert not suite['B_STEF_allowed'] and not suite['full_unseen_criterion_transfer_proven']
    proof = dict(texts=len(texts), valid_tokens=token_count, layers=layers,
                 fixed_readouts_reconstructed_from_raw=True, maximum_raw_readout_error=maximum,
                 visible_text_span_masks_verified=True, candidate_attribute_vectors_exact=candidate_checks,
                 criterion_query_vectors_exact=query_checks, encoder_extra_forwards=0,
                 candidate_interfaces=len(suite['candidate_interfaces']), criterion_interfaces=len(suite['criterion_interfaces']))
    put(root, 'frozen_smoke.json', canonical(proof) + b'\n')
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
