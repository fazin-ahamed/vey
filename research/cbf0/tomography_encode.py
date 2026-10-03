"""Single frozen pass; grammar-assisted spans and every-layer token evidence."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer

from build import canonical
from audit import put
from run import ENCODER, REVISION

POOLS = ('shipped', 'cls', 'content', 'numeric', 'namevalue', 'keyword')


def spans(text):
    if ': ' in text:
        fields = list(re.finditer(r'([^;]+): (\d+) percent', text.rstrip('.')))
        assert len(fields) == 4, text
        numbers = [m.span(2) for m in fields]
        names = [(m.start(1) + len(m.group(1)) - len(m.group(1).lstrip()), m.end(1)) for m in fields]
        return dict(candidate=True, numeric=numbers, namevalue=names + numbers, keyword=[])
    patterns = (r'(?:Prefer (?:greater|lower) |(?:Maximize|Minimize) |Choose the candidate with the (?:highest|least) )(.+)\.',
                r'(?:More|Less) (.+) is better\.')
    match = next((m for p in patterns if (m := re.fullmatch(p, text))), None)
    assert match is not None, text
    return dict(candidate=False, numeric=[], namevalue=[], keyword=[match.span(1)])


def token_masks(texts, encoded):
    offsets = encoded['offset_mapping'].numpy()
    valid = encoded['attention_mask'].numpy().astype(bool)
    special = encoded['special_tokens_mask'].numpy().astype(bool)
    masks = dict(shipped=valid, content=valid & ~special)
    masks['cls'] = np.zeros_like(valid); masks['cls'][:, 0] = True
    assert np.all(valid[:, 0] & special[:, 0])
    for pool in ('numeric', 'namevalue', 'keyword'):
        mask = np.zeros_like(valid)
        for i, text in enumerate(texts):
            info = spans(text)
            for lo, hi in info[pool]:
                mask[i] |= (offsets[i, :, 0] < hi) & (offsets[i, :, 1] > lo)
            mask[i] &= valid[i] & ~special[i]
            if (info['candidate'] and pool != 'keyword') or (not info['candidate'] and pool == 'keyword'):
                assert mask[i].sum() > 0, (text, pool)
        masks[pool] = mask
    return masks


def tokenize_smoke(texts):
    tokenizer = AutoTokenizer.from_pretrained(ENCODER, revision=REVISION, use_fast=True)
    assert tokenizer.is_fast
    examples = [next(t for t in texts if ': ' in t), next(t for t in texts if ': ' not in t)]
    encoded = tokenizer(examples, padding=True, return_offsets_mapping=True,
                        return_special_tokens_mask=True, return_tensors='pt')
    masks = token_masks(examples, encoded)
    return dict(examples=examples, tokens=[tokenizer.convert_ids_to_tokens(r) for r in encoded['input_ids'].tolist()],
                offsets=encoded['offset_mapping'].tolist(), masks={k: v.tolist() for k, v in masks.items()})


def encode(root, source, protocol):
    cache = np.load(source / 'embeddings.npz', allow_pickle=False)
    texts = cache['texts'].tolist()
    smoke = tokenize_smoke(texts)
    put(root, 'span_smoke.json', canonical(smoke) + b'\n')
    tokenizer = AutoTokenizer.from_pretrained(ENCODER, revision=REVISION, use_fast=True)
    weight_path = Path(hf_hub_download(ENCODER, 'model.safetensors', revision=REVISION))
    torch.set_num_threads(4)
    torch.manual_seed(7)
    model, loading = AutoModel.from_pretrained(ENCODER, revision=REVISION, use_safetensors=True,
                                              dtype=torch.float32, output_loading_info=True)
    assert not loading['missing_keys'] and not loading['mismatched_keys'], loading
    assert model.config._commit_hash == REVISION
    model = model.to('cuda').eval(); model.requires_grad_(False)
    layers = model.config.num_hidden_layers + 1
    width = model.config.hidden_size
    assert width == 384
    raw = root / 'hidden'; raw.mkdir()
    features = {k: np.zeros((layers, len(texts), width), dtype=np.float32) for k in POOLS}
    shards = []
    with torch.inference_mode():
        for start in range(0, len(texts), 64):
            tt = texts[start:start + 64]
            encoded = tokenizer(tt, padding=True, return_offsets_mapping=True,
                                return_special_tokens_mask=True, return_tensors='pt')
            assert encoded['input_ids'].shape[1] <= 96
            masks = token_masks(tt, encoded)
            inputs = {k: v.to('cuda') for k, v in encoded.items() if k not in ('offset_mapping', 'special_tokens_mask')}
            result = model(**inputs, output_hidden_states=True)
            hidden = result.hidden_states
            assert len(hidden) == layers and all(h.dtype == torch.float32 for h in hidden)
            gpu_masks = {k: torch.tensor(v, device='cuda').unsqueeze(-1) for k, v in masks.items()}
            for layer, h in enumerate(hidden):
                for pool, mask in gpu_masks.items():
                    pooled = (h * mask).sum(1) / mask.sum(1).clamp_min(1)
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                    features[pool][layer, start:start + len(tt)] = pooled.cpu().numpy()
            valid = encoded['attention_mask'].bool()
            token_states = torch.stack(hidden, dim=2)[valid.to('cuda')].cpu().numpy()
            name = f'{start:05d}.npz'
            np.savez(raw / name, hidden=token_states, indices=np.arange(start, start + len(tt)),
                     lengths=valid.sum(1).numpy(), input_ids=encoded['input_ids'][valid].numpy(),
                     offsets=encoded['offset_mapping'][valid].numpy(),
                     special=encoded['special_tokens_mask'][valid].numpy(),
                     **{k: v[valid.numpy()] for k, v in masks.items()})
            shards.append(dict(file='hidden/' + name, tokens=len(token_states), shape=list(token_states.shape)))
            if start % 1024 == 0:
                print(f'Captured {start}/{len(texts)} texts, all {layers} layers', flush=True)
    error = float(np.max(np.abs(features['shipped'][-1] - cache['values'])))
    assert error <= 1e-6, ('Original-cache parity failed', error)
    names = []
    for pool, values in features.items():
        for layer in range(layers):
            name = f'L{layer:02d}_{pool}.npy'
            np.save(root / name, values[layer]); names.append(name)
    np.save(root / 'texts.npy', np.array(texts))
    meta = dict(encoder=ENCODER, revision=REVISION, stock_not_Vey2=True,
                encoder_weight_sha256=hashlib.sha256(weight_path.read_bytes()).hexdigest(),
                config=model.config.to_dict(), loading={k: sorted(v) if isinstance(v, set) else v for k, v in loading.items()},
                layers=layers, width=width,
                texts=len(texts), forwards=len(shards), raw_shards=shards,
                pools=list(POOLS), features=names, maximum_original_cache_error=error,
                dtype='float32', token_states_persisted=True)
    put(root, 'encoder_manifest.json', canonical(meta) + b'\n')
    print('Frozen encoder capture complete; original cache max error', error, flush=True)


def recover_manifest(root, source):
    """Recover metadata from completed capture; never run another forward."""
    import zipfile
    cache = np.load(source / 'embeddings.npz', allow_pickle=False)
    texts = np.load(root / 'texts.npy', allow_pickle=False)
    assert np.array_equal(texts, cache['texts'])
    model, loading = AutoModel.from_pretrained(ENCODER, revision=REVISION, use_safetensors=True,
                                              dtype=torch.float32, output_loading_info=True)
    assert not loading['missing_keys'] and not loading['mismatched_keys']
    assert model.config._commit_hash == REVISION
    layers, width = model.config.num_hidden_layers + 1, model.config.hidden_size
    shards = []
    for start in range(0, len(texts), 64):
        name = f'hidden/{start:05d}.npz'
        with np.load(root / name, allow_pickle=False) as data:
            indices = data['indices']; tokens = int(data['lengths'].sum())
            assert np.array_equal(indices, np.arange(start, min(start + 64, len(texts))))
        with zipfile.ZipFile(root / name) as archive:
            with archive.open('hidden.npy') as stream:
                version = np.lib.format.read_magic(stream)
                header = (np.lib.format.read_array_header_1_0(stream) if version == (1, 0)
                          else np.lib.format.read_array_header_2_0(stream))
        shape, _, dtype = header
        assert shape == (tokens, layers, width) and dtype == np.float32
        shards.append(dict(file=name, tokens=tokens, shape=list(shape)))
    names = [f'L{layer:02d}_{pool}.npy' for pool in POOLS for layer in range(layers)]
    for name in names:
        values = np.load(root / name, mmap_mode='r')
        assert values.shape == (len(texts), width) and values.dtype == np.float32
    error = float(np.max(np.abs(np.load(root / f'L{layers-1:02d}_shipped.npy') - cache['values'])))
    assert error <= 1e-6
    weight_path = Path(hf_hub_download(ENCODER, 'model.safetensors', revision=REVISION))
    meta = dict(encoder=ENCODER, revision=REVISION, stock_not_Vey2=True,
                encoder_weight_sha256=hashlib.sha256(weight_path.read_bytes()).hexdigest(),
                config=model.config.to_dict(), loading={k: sorted(v) if isinstance(v, set) else v for k, v in loading.items()},
                layers=layers, width=width, texts=len(texts), forwards=len(shards),
                raw_shards=shards, pools=list(POOLS), features=names,
                maximum_original_cache_error=error, dtype='float32', token_states_persisted=True,
                recovery='Original capture completed; loading-info set serialization failed. CPU-only weight reload, zero additional forwards.')
    put(root, 'encoder_manifest.json', canonical(meta) + b'\n')
    print('Recovered complete encoder manifest without another forward; parity', error, flush=True)
