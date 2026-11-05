"""Unit tests for recommendation metrics, NDCG calculation, and data leakage prevention."""

import math
import pytest
from recommendation.evaluator import RecommendationEvaluator

class DummyRepo:
    def __init__(self):
        self.db_path = None
    def get_catalog_statistics(self):
        return {"total_books": 20, "total_readers": 5, "total_interactions": 20, "dataset_mode": "development_sample"}
    def get_all_interactions(self):
        return [
            {"user_id": "u1", "book_id": "b1", "rating": 5},
            {"user_id": "u1", "book_id": "b2", "rating": 5},
            {"user_id": "u1", "book_id": "b3", "rating": 4},
            {"user_id": "u1", "book_id": "b4", "rating": 5},
            {"user_id": "u2", "book_id": "b5", "rating": 5},
            {"user_id": "u2", "book_id": "b6", "rating": 4},
            {"user_id": "u2", "book_id": "b7", "rating": 3},
        ]

@pytest.fixture
def evaluator():
    return RecommendationEvaluator(DummyRepo())

def test_dcg_calculation():
    """Verify DCG calculation against manually derived values."""
    # hits = [1, 0, 1] at k=3
    # rank 1: 1 / log2(2) = 1.0
    # rank 2: 0 / log2(3) = 0.0
    # rank 3: 1 / log2(4) = 0.5
    # Total DCG = 1.5
    hits = [1, 0, 1]
    dcg = RecommendationEvaluator.compute_dcg(hits, k=3)
    assert pytest.approx(dcg, rel=1e-5) == 1.5

def test_idcg_and_ndcg_mathematical_correctness():
    """Verify IDCG and NDCG for partial and perfect rankings."""
    # Case 1: Perfect ranking of 2 relevant items at k=3: [1, 1, 0]
    # DCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.63092975 = 1.63092975
    # IDCG for 2 relevant items at k=3 = 1.63092975
    # NDCG = 1.0
    idcg_2 = RecommendationEvaluator.compute_idcg(num_relevant=2, k=3)
    expected_idcg = 1.0 + (1.0 / math.log2(3))
    assert pytest.approx(idcg_2, rel=1e-5) == expected_idcg

    hits_perfect = [1, 1, 0]
    dcg_perfect = RecommendationEvaluator.compute_dcg(hits_perfect, k=3)
    ndcg_perfect = dcg_perfect / idcg_2
    assert pytest.approx(ndcg_perfect, rel=1e-5) == 1.0

    # Case 2: Suboptimal ranking: [0, 1, 0] when 2 relevant items exist in test set
    # DCG = 0/log2(2) + 1/log2(3) = 0.63092975
    # IDCG = 1.0 + 0.63092975 = 1.63092975
    # NDCG = 0.63092975 / 1.63092975 = 0.38685
    hits_sub = [0, 1, 0]
    dcg_sub = RecommendationEvaluator.compute_dcg(hits_sub, k=3)
    ndcg_sub = dcg_sub / idcg_2
    assert pytest.approx(ndcg_sub, rel=1e-4) == 0.38685

    # Case 3: Zero hits
    hits_zero = [0, 0, 0]
    dcg_zero = RecommendationEvaluator.compute_dcg(hits_zero, k=3)
    assert dcg_zero == 0.0

def test_user_aware_split_disjointness(evaluator: RecommendationEvaluator):
    """Verify train and test partitions have zero overlap."""
    train, test = evaluator.create_user_aware_split(test_ratio=0.25, random_seed=42)
    assert len(train) > 0
    assert len(test) > 0

    train_pairs = {(r["user_id"], r["book_id"]) for r in train}
    test_pairs = {(r["user_id"], r["book_id"]) for r in test}
    assert train_pairs.isdisjoint(test_pairs)
