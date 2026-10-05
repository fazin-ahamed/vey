import math

import pytest

torch = pytest.importorskip("torch")

from research.endgame.native_field_model import native_readout
from research.endgame.native_field_train import decision_loss


def test_sparse_soft_targets_normalize_by_decisions_not_states():
    first = torch.tensor([[math.log(.8), math.log(.2)],
                          [math.log(.1), math.log(.9)]], dtype=torch.float64, requires_grad=True)
    second = torch.tensor([[math.log(.7), math.log(.3)],
                           [math.log(.4), math.log(.6)]], dtype=torch.float64, requires_grad=True)
    records = [
        {"decisions": [{"endpoint": "first", "target_distribution": [1., 0.]},
                       {"endpoint": "second", "target_distribution": [.25, .75]}]},
        {"decisions": [{"endpoint": "first", "target_distribution": [0., 1.]}]},
    ]
    loss, count = decision_loss({"first": first, "second": second}, records)
    expected = -math.log(.8) - .25 * math.log(.7) - .75 * math.log(.3) - math.log(.9)
    assert count == 3
    assert float(loss.detach()) == pytest.approx(expected)
    (loss / count).backward()
    assert first.grad[0].tolist() == pytest.approx([-.2 / 3, .2 / 3])
    assert first.grad[1].tolist() == pytest.approx([.1 / 3, -.1 / 3])
    assert second.grad[0].tolist() == pytest.approx([.45 / 3, -.45 / 3])
    assert second.grad[1].tolist() == [0., 0.]


def test_native_readout_uses_semantic_coordinates_and_stable_ties():
    tied = native_readout(["zeta", "alpha"], [.5, .5], "choice", ["alpha", "zeta"])
    assert tied["answer"] == "alpha"
    score = native_readout(["2", "0", "1"], [.3, .1, .6], "score", ["0", "1", "2"], 2)
    assert score["expected_level"] == pytest.approx(1.2)
    assert score["cdf"] == pytest.approx([.1, .7, 1.])
