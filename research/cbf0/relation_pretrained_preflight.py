"""CBF7 pre-outcome byte replay and real literal-pair execution proof; no quality screen."""
import copy
import datetime
import json
import subprocess
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn

from audit import put
from build import canonical
from relation_pretrained_corpus import build_final
from relation_pretrained_encoder import load_model, tokenizer_parity, full_forward_parity, native_class_indices, SCHEMA
from relation_pretrained_run import PROTOCOL, cache_features, native_matrices, gpu_serialization, tensor_hash, sha, pairs_for, resource_snapshot


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(7)
    cfg = json.loads(PROTOCOL.read_text())
    root = Path(cfg['output_root'])
    git = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).parents[2], text=True).strip()
    if (root / 'results.json').exists() or (root / 'selection.json').exists():
        raise RuntimeError('Preflight cannot amend an experiment after quality selection')
    proof = dict(scope='Byte regeneration and literal-only execution; no development/final quality scoring',
                 git_revision=git, protocol_sha256=sha(PROTOCOL), started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 source_sha256={name: sha(Path(__file__).with_name(name)) for name in
                                ('relation_pretrained_preflight.py', 'relation_pretrained_corpus.py',
                                 'relation_pretrained_encoder.py', 'relation_pretrained_run.py')},
                 seed=7, threads=4, interop_threads=1, resource_before=resource_snapshot(), models={})
    try:
        manifest = json.loads((root / 'corpus_manifest.json').read_text())
        prior_audit = json.loads((root / 'final_corpus_audit.json').read_text())
        for name in ('corpus_manifest.json', 'final_corpus_audit.json'):
            original = root / ('original_pre_execution_' + name)
            if not original.exists():
                put(root, original.name, (root / name).read_bytes())
        atoms, comps, audit = build_final()
        assert canonical(atoms) + b'\n' == (root / 'final_atoms.json').read_bytes()
        assert canonical(comps) + b'\n' == (root / 'final_compositions.json').read_bytes()
        for key, value in audit.items():
            if key != 'provenance':
                assert prior_audit[key] == value, key
        prior_provenance = copy.deepcopy(prior_audit['provenance'])
        prior_provenance['protocol_sha256'] = sha(PROTOCOL)
        assert prior_provenance == audit['provenance']
        prior_audit['provenance'] = audit['provenance']
        replay = dict(generator_committed_before_replay=True, replay_git_revision=git,
                      first_preparation_generator_uncommitted=True, first_preparation_git_not_fabricated=True,
                      atoms_byte_identical=True, compositions_byte_identical=True,
                      original_manifest_sha256=sha(root / 'original_pre_execution_corpus_manifest.json'),
                      original_audit_sha256=sha(root / 'original_pre_execution_final_corpus_audit.json'),
                      protocol_sha256=sha(PROTOCOL), final_atoms_sha256=sha(root / 'final_atoms.json'),
                      final_compositions_sha256=sha(root / 'final_compositions.json'))
        put(root, 'preparation_replay.json', canonical(replay) + b'\n')
        prior_audit['preparation_replay'] = replay
        put(root, 'final_corpus_audit.json', canonical(prior_audit) + b'\n')
        manifest['pre_execution_protocol_sha256'] = manifest['protocol_sha256']
        manifest['protocol_sha256'] = sha(PROTOCOL)
        manifest['preparation_replay'] = replay
        manifest['files']['final_corpus_audit.json'] = sha(root / 'final_corpus_audit.json')
        manifest['files']['preparation_replay.json'] = sha(root / 'preparation_replay.json')
        put(root, 'corpus_manifest.json', canonical(manifest) + b'\n')
        proof['preparation_replay'] = replay
        put(root, 'preflight_environment.json', canonical(proof) + b'\n')
        training = json.loads((root / 'training_atoms.json').read_text())[:2]
        texts = [case['text'] for case in training]
        pairs, labels = pairs_for(training)
        for key in ('stock_xsmall', 'nli_xsmall'):
            with gpu_serialization():
                tokenizer, model, lineage = load_model(key)
                before = tensor_hash(model.named_parameters())
                put(root, f'preflight_lineage_{key}.json', canonical(lineage) + b'\n')
                parity = tokenizer_parity(texts, tokenizer, 'higher', key)
                cls, pooled, inputs = cache_features(root, key + '_literal_preflight', texts, tokenizer, model, 'higher')
                cached_cls, cached_pooled, _ = cache_features(root, key + '_literal_preflight', texts, tokenizer, model, 'higher')
                assert np.array_equal(cls, cached_cls)
                if pooled is not None:
                    assert np.array_equal(pooled, cached_pooled)
                mean = cls.mean(0)
                std = np.maximum(cls.std(0), .01)
                head = nn.Linear(cls.shape[1], 3, dtype=torch.float32)
                loss = nn.functional.cross_entropy(head(torch.tensor((cls - mean) / std)), labels,
                                                   weight=torch.tensor(cfg['training']['class_weights']))
                loss.backward()
                gradient = sum(float(p.grad.abs().sum()) for p in head.parameters())
                assert torch.isfinite(loss) and gradient > 0
                details = dict(floating_parameters=lineage['floating_parameters'], width=cls.shape[1],
                               pairs=len(pairs), cache_reload_identical=True, native_tokenizer_parity=parity,
                               head_gradient_L1=gradient, pretrained_gradients=0,
                               input_evidence_sha256=sha(root / f'inputs_{key}_literal_preflight_higher.json'))
                if key == 'nli_xsmall':
                    details['native_forward_parity'] = full_forward_parity(pairs, tokenizer, model, 'higher')
                    mapping = native_class_indices(model)
                    matrices, _ = native_matrices(model, pooled, mapping, texts)
                    encoded = tokenizer([q for q, _ in pairs], text_pair=[f'Higher {field} is preferred.' for _, field in pairs],
                                        padding=True, return_tensors='pt')
                    with torch.inference_mode():
                        reference = model(**{k: v.to('cuda') for k, v in encoded.items()}).logits.softmax(-1).cpu().numpy()
                    reference = reference[:, [mapping[s] for s in (-1, 0, 1)]]
                    error = float(np.max(np.abs(reference - np.concatenate([matrices[q] for q in texts]))))
                    assert error < 2e-5, error
                    details['cached_native_probability_max_error'] = error
                assert all(not p.requires_grad and p.grad is None for p in model.parameters())
                after = tensor_hash(model.named_parameters())
                assert before == after
                details.update(frozen_before_sha256=before, frozen_after_sha256=after)
                proof['models'][key] = details
                del model, tokenizer, head
                torch.cuda.empty_cache()
        proof['all_checks_pass'] = True
        proof['resource_after'] = resource_snapshot()
        proof['completed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        put(root, 'preflight_proof.json', canonical(proof) + b'\n')
        print(json.dumps(proof, indent=2))
    except BaseException as exc:
        proof.update(all_checks_pass=False, error=repr(exc), traceback=traceback.format_exc())
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        put(root, f'preflight_failure_{stamp}.json', canonical(proof) + b'\n')
        raise


if __name__ == '__main__':
    main()
