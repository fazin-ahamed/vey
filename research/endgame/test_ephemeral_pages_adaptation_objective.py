import copy

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("scipy")
pytest.importorskip("transformers")

from research.endgame import ephemeral_pages_adaptation as adaptation


@pytest.mark.parametrize("chunk_size", [1, 2, 3])
def test_cached_state_gradients_match_whole_batch_with_repeated_pages(monkeypatch, chunk_size):
    monkeypatch.setattr(adaptation.trainer, "_resource_guard", lambda *args: None)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        encoder = torch.nn.Linear(adaptation.WIDTH, adaptation.WIDTH)
        reader = adaptation.readers.PageReader()
        reference_encoder, reference_reader = copy.deepcopy(encoder), copy.deepcopy(reader)
        inputs = torch.randn(4, adaptation.WIDTH)
        tables = {"q": encoder(inputs), "pages": encoder(inputs + .1)}
        reference_tables = {"q": reference_encoder(inputs), "pages": reference_encoder(inputs + .1)}
        mask = np.ones((4, 2), dtype=bool)
        grade_mask, orientation_mask = mask.copy(), mask.copy()
        grade_mask[2, 1] = False
        orientation_mask[0, 1] = False
        arrays = {
            "page_mask": mask,
            "relevance_target": np.asarray([[1, 0], [0, 1], [1, 0], [0, 0]], dtype=np.float32),
            "grade_target": np.asarray([[.1, .8], [.7, .2], [.1, .7], [.2, .8]], dtype=np.float32),
            "directed_grade_target": np.zeros((4, 2), dtype=np.float32),
            "grade_mask": grade_mask,
            "orientation_target": np.asarray([[1, -1], [-1, 1], [1, 1], [-1, -1]], dtype=np.float32),
            "orientation_mask": orientation_mask,
            "known_target": np.asarray([1, 1, 0, 1], dtype=np.float32),
            "_page_ix": np.asarray([[0, 1], [2, 3], [0, 2], [3, 1]]),
            "_query_ix": np.asarray([0, 1, 0, 1]),
        }
        normalizer = {key: {"mean": np.zeros(adaptation.WIDTH, dtype=np.float32),
                            "std": np.ones(adaptation.WIDTH, dtype=np.float32)}
                      for key in ("q", "pages")}
        indices = np.arange(4)
        denominators = adaptation.trainer._denominators(arrays, indices, "pages", chunk_size)
        actual = adaptation._objective(
            reader, tables, arrays, indices, denominators, chunk_size, 0, normalizer,
            backward=True, counters={"backward_calls": 0})
        targets = adaptation.trainer._batch(arrays, indices, "cpu")
        output = adaptation._reader_forward(
            reference_reader, reference_tables, arrays, indices, normalizer, "cpu")
        expected = adaptation.readers.eca_loss(output, targets, directed_grade=False)
        expected["total"].backward()
        for component in adaptation.trainer.COMPONENTS:
            assert actual[component] == pytest.approx(float(expected[component].detach()), abs=1e-6)
        for module, reference in ((encoder, reference_encoder), (reader, reference_reader)):
            for (name, parameter), (reference_name, reference_parameter) in zip(
                    module.named_parameters(), reference.named_parameters()):
                assert name == reference_name
                assert parameter.grad is not None
                assert reference_parameter.grad is not None
                torch.testing.assert_close(parameter.grad, reference_parameter.grad,
                                           rtol=1e-5, atol=1e-7)
