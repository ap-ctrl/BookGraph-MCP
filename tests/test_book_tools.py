"""Tests for Book Discovery, Details, Trope Filtering, and Comparison."""

import pytest
from services.book_service import BookService
from services.nlp_service import NLPService

def test_search_existing_book(book_service: BookService):
    """Test 1: Search an existing book."""
    res = book_service.search_books(query="Love Hypothesis")
    assert res["pagination"]["total_matches"] >= 1
    found_titles = [b["title"] for b in res["results"]]
    assert any("The Love Hypothesis" in t for t in found_titles)

def test_search_nonexistent_book(book_service: BookService):
    """Test 2: Search for a nonexistent book."""
    res = book_service.search_books(query="NonExistentBookTitleXyZ99999")
    assert res["pagination"]["total_matches"] == 0
    assert len(res["results"]) == 0

def test_get_book_details(book_service: BookService):
    """Test 3: Retrieve book details by ID."""
    details = book_service.get_book_details("56732449")
    assert details is not None
    assert details["book_id"] == "56732449"
    assert "Ali Hazelwood" in details["author"]
    assert details["average_rating"] > 4.0
    assert "confirmed_tropes" in details
    assert len(details["confirmed_tropes"]) >= 1

def test_filter_books_by_rating(book_service: BookService):
    """Test 4: Filter books by minimum rating."""
    res = book_service.search_books(min_rating=4.20)
    assert res["pagination"]["total_matches"] >= 1
    for b in res["results"]:
        assert b["average_rating"] >= 4.20

def test_retrieve_trope_based_results(nlp_service: NLPService):
    """Test 8: Retrieve trope-based results with evidence."""
    res = nlp_service.find_books_by_tropes(tropes=["Enemies to Lovers"])
    assert res["total_matches"] >= 1
    book = res["books"][0]
    assert len(book["matched_tropes_with_evidence"]) >= 1
    trope_item = book["matched_tropes_with_evidence"][0]
    assert "evidence" in trope_item
    assert trope_item["evidence"] != ""
    assert trope_item["method"] in ["rule_based_keyword", "tfidf_matcher", "hybrid"]

def test_compare_two_books(book_service: BookService):
    """Test 11: Compare two books side by side."""
    res = book_service.compare_books("56732449", "25614492")
    assert "book_1" in res
    assert "book_2" in res
    assert "trope_overlap" in res
    assert res["book_1"]["title"] == "The Love Hypothesis"
    assert res["book_2"]["title"] == "The Hating Game"
    assert isinstance(res["trope_overlap"]["shared_tropes"], list)

def test_validate_invalid_tool_inputs(book_service: BookService, nlp_service: NLPService):
    """Test 13: Validate invalid tool inputs gracefully."""
    # Negative pagination limit
    res_neg = book_service.search_books(limit=-5)
    assert res_neg["pagination"]["limit"] == 1  # Clamped to minimum 1

    # Empty tropes list
    res_trope_err = nlp_service.find_books_by_tropes(tropes=[])
    assert "error" in res_trope_err

    # Invalid book ID for comparison
    comp_err = book_service.compare_books("invalid_id_1", "invalid_id_2")
    assert "error" in comp_err
