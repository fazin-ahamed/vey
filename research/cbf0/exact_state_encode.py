"""CBF-4 frozen criterion-only capture; no candidate neural processing."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer

from audit import put
from build import canonical
from exact_state_build import PROTOCOL

ARMS = ('final_mean', 'layer1_mean', 'layer1_span')


def semantic_span(text):
    prefix = re.match(r'(?:Choose the option with the |Choose using |Prefer |Aim for )', text, re.I)
    lo = prefix.end() if prefix else 0
    hi = len(text.rstrip('.'))
    assert lo < hi
    return lo, hi


def masks(texts, encoded):
    offsets = encoded['offset_mapping'].numpy()
    valid = encoded['attention_mask'].numpy().astype(bool)
    special = encoded['special_tokens_mask'].numpy().astype(bool)
    span = np.zeros_like(valid)
    for i, text in enumerate(texts):
        lo, hi = semantic_span(text)
        span[i] = valid[i] & ~special[i] & (offsets[i, :, 0] < hi) & (offsets[i, :, 1] > lo)
    assert np.all(span.sum(1) > 0)
    return valid, special, span


def encode(root):
    protocol = json.loads(PROTOCOL.read_text()); spec = protocol['encoder']
    corpus = json.loads((root / 'corpus_manifest.json').read_text())
    for name, expected in corpus['files'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
    texts = sorted(c['text'] for c in json.loads((root / 'criteria.json').read_text()))
    assert len(texts) == len(set(texts))
    if (root / 'encoder_manifest.json').exists() or (root / 'hidden').exists():
        raise RuntimeError('Refusing another forward over a captured/partial criterion corpus')
    tokenizer = AutoTokenizer.from_pretrained(spec['model'], revision=spec['revision'], use_fast=True)
    assert tokenizer.is_fast
    torch.set_num_threads(4); torch.manual_seed(7)
    model, loading = AutoModel.from_pretrained(spec['model'], revision=spec['revision'], use_safetensors=True,
                                              dtype=torch.float32, output_loading_info=True)
    assert not loading['missing_keys'] and not loading['mismatched_keys'], loading
    assert model.config._commit_hash == spec['revision'] and model.config.hidden_size == 384
    weight_path = Path(hf_hub_download(spec['model'], 'model.safetensors', revision=spec['revision']))
    weight_hash = hashlib.sha256(weight_path.read_bytes()).hexdigest()
    assert weight_hash == spec['weight_sha256']
    lineage = dict(encoder=spec['model'], revision=spec['revision'], stock_not_Vey2=True,
                   weight_sha256=weight_hash, config=model.config.to_dict(),
                   loading={k: sorted(v) if isinstance(v, set) else v for k, v in loading.items()},
                   tokenizer_sha256=hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
                   dtype='float32', criteria=len(texts), candidate_forwards=0,
                   protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
                   corpus_sha256=hashlib.sha256((root / 'corpus_manifest.json').read_bytes()).hexdigest())
    put(root, 'encoder_lineage.json', canonical(lineage) + b'\n')
    model = model.to('cuda').eval(); model.requires_grad_(False)
    raw = root / 'hidden'; raw.mkdir()
    features = {arm: np.empty((len(texts), 384), np.float32) for arm in ARMS}
    shards = []; maximum_tokens = 0
    with torch.inference_mode():
        for start in range(0, len(texts), spec['batch_size']):
            tt = texts[start:start + spec['batch_size']]
            encoded = tokenizer(tt, padding=True, return_offsets_mapping=True,
                                return_special_tokens_mask=True, return_tensors='pt')
            length = encoded['input_ids'].shape[1]; maximum_tokens = max(maximum_tokens, length)
            assert length <= spec['token_limit'], 'Never silently truncate a criterion'
            valid, special, span = masks(tt, encoded)
            inputs = {k: v.to('cuda') for k, v in encoded.items() if k not in ('offset_mapping', 'special_tokens_mask')}
            hidden = model(**inputs, output_hidden_states=True).hidden_states
            assert len(hidden) == 13 and hidden[1].dtype == hidden[-1].dtype == torch.float32
            for arm, layer, mask in (('final_mean', -1, valid), ('layer1_mean', 1, valid), ('layer1_span', 1, span)):
                mm = torch.tensor(mask, device='cuda').unsqueeze(-1)
                pooled = (hidden[layer] * mm).sum(1) / mm.sum(1)
                features[arm][start:start + len(tt)] = torch.nn.functional.normalize(pooled, dim=1).cpu().numpy()
            vv = torch.tensor(valid, device='cuda')
            name = f'hidden/{start:05d}.npz'
            np.savez(root / name, hidden=torch.stack((hidden[1], hidden[-1]), dim=2)[vv].cpu().numpy(),
                     indices=np.arange(start, start + len(tt)), lengths=valid.sum(1),
                     input_ids=encoded['input_ids'].numpy()[valid], offsets=encoded['offset_mapping'].numpy()[valid],
                     special=special[valid], span=span[valid])
            shards.append(dict(file=name, tokens=int(valid.sum()), texts=len(tt)))
            print(f'Captured criteria {start + len(tt)}/{len(texts)}; candidate forwards 0', flush=True)
    np.save(root / 'texts.npy', np.array(texts))
    for arm, values in features.items(): np.save(root / f'{arm}.npy', values)
    meta = dict(**lineage, forwards=len(shards), text_layers_saved=[1, 12], maximum_tokens=maximum_tokens,
                raw_shards=shards, features=[f'{arm}.npy' for arm in ARMS], tokens=sum(s['tokens'] for s in shards))
    put(root, 'encoder_manifest.json', canonical(meta) + b'\n')
    print(json.dumps({k: meta[k] for k in ('criteria', 'forwards', 'candidate_forwards', 'maximum_tokens', 'tokens')}, indent=2))


if __name__ == '__main__':
    encode(Path(json.loads(PROTOCOL.read_text())['output_root']))
