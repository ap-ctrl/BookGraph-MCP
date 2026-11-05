"""Tests for Review Analysis, Reader Insights, and Catalogue Analytics."""

import pytest
from services.book_service import BookService
from services.nlp_service import NLPService

def test_analyse_existing_book_reviews(nlp_service: NLPService):
    """Test 9: Analyse an existing book's reviews with sentiment and themes."""
    res = nlp_service.analyse_book_reviews("56732449")
    assert res["found"] is not False
    assert res["total_reviews_analyzed"] >= 1
    assert "sentiment_overview" in res
    assert "distribution" in res["sentiment_overview"]
    assert "thematic_feedback" in res
    assert "character" in res["thematic_feedback"]
    assert "pacing" in res["thematic_feedback"]
    assert "plot" in res["thematic_feedback"]

def test_handle_books_without_review_data(nlp_service: NLPService):
    """Test 10: Handle books without review data or nonexistent books."""
    res = nlp_service.analyse_book_reviews("nonexistent_book_id_9999")
    assert res["found"] is False
    assert "error" in res

def test_retrieve_catalogue_statistics(book_service: BookService):
    """Test 12: Retrieve catalogue statistics."""
    stats = book_service.get_catalogue_statistics()
    assert stats["total_books"] >= 10
    assert stats["total_authors"] >= 5
    assert stats["total_interactions"] >= 10
    assert stats["total_trope_classifications"] >= 10
    assert stats["catalog_average_rating"] > 0.0

def test_get_author_insights(book_service: BookService):
    """Test author analytics retrieval."""
    res = book_service.get_author_insights("Emily Henry")
    assert res["found"] is True
    assert res["total_books_in_catalog"] >= 2
    assert res["average_author_rating"] > 3.5

def test_get_reader_insights(nlp_service: NLPService):
    """Test reader insight profiling."""
    res = nlp_service.get_reader_insights("usr_workplace_01")
    assert res["found"] is True
    assert res["total_books_shelved"] >= 3
    assert len(res["favorite_genres"]) >= 1
