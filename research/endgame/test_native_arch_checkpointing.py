import copy

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from research.endgame.native_arch_model import canonical_modules


def test_nonreentrant_checkpointing_preserves_stochastic_cross_gradients():
    config = transformers.DebertaV2Config(
        vocab_size=32, hidden_size=24, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4,
        max_position_embeddings=64, max_relative_positions=16,
        relative_attention=True, position_biased_input=False,
        pos_att_type=["p2c", "c2p"],
        hidden_dropout_prob=.2, attention_probs_dropout_prob=.2,
    )
    semantic, _, _ = canonical_modules()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        ordinary = semantic.CrossEncoderScorer(transformers.DebertaV2Model(config), hidden=16)
        checkpointed = copy.deepcopy(ordinary)
        checkpointed.pool.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False, "preserve_rng_state": True})
        ordinary.train()
        checkpointed.train()
        ids = torch.randint(1, 32, (2, 3, 12))
        mask = torch.ones_like(ids)
        mask[0, :, -3:] = 0
        targets = torch.tensor([[.25, .75, 0.], [0., 0., 1.]])
        outputs, gradients, rng_after = [], [], []
        for model in (ordinary, checkpointed):
            torch.manual_seed(101)
            values = model(ids, mask)
            loss = -(targets * values.log_softmax(dim=-1)).sum() / len(targets)
            loss.backward()
            outputs.append(values.detach())
            gradients.append({name: None if param.grad is None else param.grad.detach().clone()
                              for name, param in model.named_parameters()})
            rng_after.append(torch.get_rng_state())
    torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-6, atol=1e-7)
    assert torch.equal(rng_after[0], rng_after[1])
    assert gradients[0].keys() == gradients[1].keys()
    for name, expected in gradients[0].items():
        actual = gradients[1][name]
        if expected is None:
            assert actual is None
        else:
            assert actual is not None
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
    encoder_gradient = gradients[1]["pool.encoder.encoder.layer.0.output.dense.weight"]
    assert torch.isfinite(encoder_gradient).all()
    assert torch.count_nonzero(encoder_gradient) > 0
    assert torch.count_nonzero(gradients[1]["head.0.weight"]) > 0
