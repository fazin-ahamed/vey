"""IR ranking metric tests using synthetic rankings."""
import math

from vey.evaluation.metrics import mrr_at_k, ndcg_at_k, recall_at_k


def test_ndcg_perfect_and_partial():
    rel = {"a": 1.0, "b": 1.0}
    assert ndcg_at_k(["a", "b", "c"], rel, 10) == 1.0
    # one relevant doc pushed to rank 2 -> discounted
    got = ndcg_at_k(["c", "a"], {"a": 1.0}, 10)
    assert abs(got - (1 / math.log2(3))) < 1e-9


def test_recall_at_k_counts_hits_over_relevant():
    assert recall_at_k(["a", "x", "y"], {"a", "b"}, 10) == 0.5
    assert recall_at_k(["x", "y"], {"a"}, 10) == 0.0
    assert recall_at_k(["a", "b"], {"a", "b"}, 1) == 0.5


def test_mrr_uses_first_relevant_rank():
    assert mrr_at_k(["x", "a", "b"], {"a"}, 10) == 0.5
    assert mrr_at_k(["a"], {"a"}, 10) == 1.0
    assert mrr_at_k(["x", "y"], {"a"}, 10) == 0.0
