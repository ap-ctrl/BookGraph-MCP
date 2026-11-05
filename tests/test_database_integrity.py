"""Unit tests for database integrity, constraints, idempotency, and empty database behavior."""

import json
import sqlite3
import pytest
from pathlib import Path

from database.connection import get_db_connection, init_db
from database.repository import BookRepository
from preprocessing.preprocessor import DataPreprocessor

@pytest.fixture
def temp_db(tmp_path: Path):
    """Isolated temporary SQLite database."""
    db_file = tmp_path / "test_integrity.db"
    init_db(db_file)
    return db_file

@pytest.fixture
def temp_repo(temp_db: Path):
    return BookRepository(temp_db)

def test_empty_database_behavior(temp_repo: BookRepository):
    """Verify repository methods behave gracefully on an empty database."""
    stats = temp_repo.get_catalog_statistics()
    assert stats["total_books"] == 0
    assert stats["total_interactions"] == 0
    assert stats["total_reviews"] == 0
    assert stats["catalog_average_rating"] == 0.0

    books, total = temp_repo.search_books(query="Romance")
    assert books == []
    assert total == 0

    assert temp_repo.get_book_by_id("nonexistent_id") is None
    assert temp_repo.get_interactions_for_user("unknown_user") == []
    assert temp_repo.get_reviews_for_book("nonexistent_book") == []
    assert temp_repo.get_top_popular_books() == []

def test_duplicate_book_id_upsert(temp_repo: BookRepository, temp_db: Path):
    """Verify duplicate book_id updates attributes without deleting child relations."""
    book_record = {
        "book_id": "test_bk_01",
        "title": "Initial Title",
        "title_without_series": "Initial Title",
        "author_name": "Author A",
        "author_id": "auth_1",
        "average_rating": 4.0,
        "ratings_count": 100,
        "text_reviews_count": 10,
        "publication_year": 2020,
        "publisher": "Imprint",
        "num_pages": 300,
        "description": "A love story.",
        "genres": '["Contemporary Romance"]',
        "popular_shelves": '[]',
        "similar_books": '[]',
        "url": "",
        "image_url": "",
        "bayesian_rating": 3.95,
    }
    temp_repo.insert_or_replace_books([book_record])

    # Insert child interaction & review
    temp_repo.insert_or_replace_interactions([
        {"user_id": "u1", "book_id": "test_bk_01", "rating": 5, "is_read": 1, "date_added": "2021", "read_at": "2021"}
    ])
    temp_repo.insert_or_replace_reviews([
        {"review_id": "r1", "user_id": "u1", "book_id": "test_bk_01", "rating": 5, "review_text": "Great book", "sentiment_score": 0.8, "sentiment_label": "positive", "date_added": "2021"}
    ])

    # Update book record (same book_id, modified title)
    book_record_updated = dict(book_record)
    book_record_updated["title"] = "Updated Title"
    temp_repo.insert_or_replace_books([book_record_updated])

    # Book title must be updated
    fetched = temp_repo.get_book_by_id("test_bk_01")
    assert fetched is not None
    assert fetched["title"] == "Updated Title"

    # Child interactions and reviews must NOT be deleted
    interactions = temp_repo.get_interactions_for_user("u1")
    assert len(interactions) == 1
    assert interactions[0]["book_id"] == "test_bk_01"

    reviews = temp_repo.get_reviews_for_book("test_bk_01")
    assert len(reviews) == 1
    assert reviews[0]["review_id"] == "r1"

def test_foreign_key_enforcement_in_database(temp_db: Path):
    """Verify database raises IntegrityError when inserting interaction with nonexistent book."""
    with pytest.raises(sqlite3.IntegrityError):
        with get_db_connection(temp_db) as conn:
            conn.execute(
                "INSERT INTO interactions (user_id, book_id, rating, is_read) VALUES (?, ?, ?, ?);",
                ("user_x", "nonexistent_book_id", 5, 1),
            )

    with pytest.raises(sqlite3.IntegrityError):
        with get_db_connection(temp_db) as conn:
            conn.execute(
                "INSERT INTO reviews (review_id, user_id, book_id, rating, review_text) VALUES (?, ?, ?, ?, ?);",
                ("rev_x", "user_x", "nonexistent_book_id", 4, "Invalid reference test"),
            )

def test_preprocessor_filters_invalid_foreign_keys():
    """Verify preprocessor discards interactions and reviews pointing to nonexistent books."""
    preprocessor = DataPreprocessor()
    valid_ids = {"book_valid_1"}

    invalid_interaction = {"user_id": "u1", "book_id": "book_nonexistent", "rating": 5}
    valid_interaction = {"user_id": "u1", "book_id": "book_valid_1", "rating": 5}

    clean_invalid = preprocessor.clean_interaction_record(invalid_interaction, valid_ids)
    clean_valid = preprocessor.clean_interaction_record(valid_interaction, valid_ids)

    assert clean_invalid is None
    assert clean_valid is not None
    assert preprocessor.validation_stats["interactions"]["invalid_book_ref"] == 1

def test_repeated_pipeline_execution_idempotency(temp_repo: BookRepository):
    """Verify repeated inserts produce stable counts and do not duplicate rows."""
    from ingestion.seed_data import CANONICAL_BOOKS, CANONICAL_INTERACTIONS, CANONICAL_REVIEWS
    preprocessor = DataPreprocessor()
    b_df, i_df, r_df = preprocessor.process_all(CANONICAL_BOOKS, CANONICAL_INTERACTIONS, CANONICAL_REVIEWS)

    # First load
    temp_repo.insert_or_replace_books(b_df.to_dict("records"))
    temp_repo.insert_or_replace_interactions(i_df.to_dict("records"))
    temp_repo.insert_or_replace_reviews(r_df.to_dict("records"))

    stats1 = temp_repo.get_catalog_statistics()
    count_books_1 = stats1["total_books"]
    count_inter_1 = stats1["total_interactions"]
    count_revs_1 = stats1["total_reviews"]

    # Second load (identical data)
    temp_repo.insert_or_replace_books(b_df.to_dict("records"))
    temp_repo.insert_or_replace_interactions(i_df.to_dict("records"))
    temp_repo.insert_or_replace_reviews(r_df.to_dict("records"))

    stats2 = temp_repo.get_catalog_statistics()
    assert stats2["total_books"] == count_books_1
    assert stats2["total_interactions"] == count_inter_1
    assert stats2["total_reviews"] == count_revs_1
