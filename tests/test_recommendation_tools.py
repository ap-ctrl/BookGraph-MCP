"""Tests for Recommendation Engine (Similar, Personalized, Cold-Start, Popular)."""

import pytest
from services.recommendation_service import RecommendationService

def test_recommend_similar_books(rec_service: RecommendationService):
    """Test 5: Retrieve similar books using content similarity."""
    res = rec_service.recommend_similar_books(book_id="56732449", limit=5)
    assert res["count"] >= 1
    assert "reference_book" in res
    assert res["reference_book"]["title"] == "The Love Hypothesis"
    recs = res["recommendations"]
    assert len(recs) <= 5
    for r in recs:
        assert "similarity_score" in r
        assert r["similarity_score"] > 0.0
        assert "explanation" in r
        assert r["book_id"] != "56732449"

def test_recommendations_for_known_reader(rec_service: RecommendationService):
    """Test 6: Retrieve personalized recommendations for a known reader."""
    res = rec_service.get_personalized_recommendations(user_id="usr_workplace_01", limit=5)
    assert res["is_cold_start"] is False
    assert len(res["recommendations"]) >= 1
    for r in res["recommendations"]:
        assert "hybrid_score" in r
        assert r["hybrid_score"] >= 0.0
        assert "explanation" in r

def test_handle_reader_with_insufficient_history(rec_service: RecommendationService):
    """Test 7: Handle a reader with insufficient history (cold start)."""
    # Test user with only 1 interaction
    res_cold = rec_service.get_personalized_recommendations(user_id="usr_cold_05", limit=5)
    assert res_cold["is_cold_start"] is True
    assert "cold_start_reason" in res_cold
    assert len(res_cold["recommendations"]) >= 1

    # Test unknown reader ID
    res_unknown = rec_service.get_personalized_recommendations(user_id="usr_unknown_9999", limit=5)
    assert res_unknown["is_cold_start"] is True
    assert "cold_start_reason" in res_unknown

def test_popular_books_ranking(rec_service: RecommendationService):
    """Test popularity-based Bayesian ranking."""
    res = rec_service.get_popular_books(limit=5, min_ratings=50)
    assert res["count"] >= 1
    for b in res["books"]:
        assert "bayesian_rating" in b
        assert b["bayesian_rating"] > 0.0
