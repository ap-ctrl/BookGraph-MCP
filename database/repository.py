"""Database repository layer providing parameterized data operations."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from config.settings import logger
from database.connection import get_db_connection

class BookRepository:
    """Repository handling all database interactions with parameterized queries."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path

    # =========================================================================
    # Book Operations
    # =========================================================================

    def search_books(
        self,
        query: Optional[str] = None,
        author: Optional[str] = None,
        genre: Optional[str] = None,
        min_rating: Optional[float] = None,
        publication_year: Optional[int] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Search books with optional filters, parameterized conditions, and pagination.
        Returns (list_of_books, total_count).
        """
        conditions = []
        params: List[Any] = []

        if query and query.strip():
            conditions.append("(title LIKE ? OR description LIKE ?)")
            term = f"%{query.strip()}%"
            params.extend([term, term])

        if author and author.strip():
            conditions.append("author_name LIKE ?")
            params.append(f"%{author.strip()}%")

        if genre and genre.strip():
            conditions.append("genres LIKE ?")
            params.append(f"%{genre.strip()}%")

        if min_rating is not None and min_rating > 0:
            conditions.append("average_rating >= ?")
            params.append(float(min_rating))

        if publication_year is not None:
            conditions.append("publication_year = ?")
            params.append(int(publication_year))

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        count_sql = f"SELECT COUNT(*) as total FROM books {where_clause};"
        select_sql = f"""
            SELECT book_id, title, title_without_series, author_name, author_id,
                   average_rating, ratings_count, text_reviews_count, publication_year,
                   publisher, num_pages, description, genres, popular_shelves,
                   similar_books, url, image_url, bayesian_rating
            FROM books
            {where_clause}
            ORDER BY bayesian_rating DESC, ratings_count DESC
            LIMIT ? OFFSET ?;
        """

        with get_db_connection(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(count_sql, params)
            total = cur.fetchone()["total"]

            cur.execute(select_sql, params + [limit, offset])
            rows = cur.fetchall()
            results = [dict(row) for row in rows]

            # Decode JSON fields
            for r in results:
                r["genres"] = json.loads(r.get("genres") or "[]")
                r["popular_shelves"] = json.loads(r.get("popular_shelves") or "[]")
                r["similar_books"] = json.loads(r.get("similar_books") or "[]")

        return results, total

    def get_book_by_id(self, book_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve single book record by Goodreads book_id."""
        sql = """
            SELECT book_id, title, title_without_series, author_name, author_id,
                   average_rating, ratings_count, text_reviews_count, publication_year,
                   publisher, num_pages, description, genres, popular_shelves,
                   similar_books, url, image_url, bayesian_rating
            FROM books
            WHERE book_id = ?;
        """
        with get_db_connection(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(sql, (str(book_id).strip(),))
            row = cur.fetchone()
            if not row:
                return None
            result = dict(row)
            result["genres"] = json.loads(result.get("genres") or "[]")
            result["popular_shelves"] = json.loads(result.get("popular_shelves") or "[]")
            result["similar_books"] = json.loads(result.get("similar_books") or "[]")
            return result

    def get_books_by_ids(self, book_ids: List[str]) -> List[Dict[str, Any]]:
        """Retrieve multiple books by ID list."""
        if not book_ids:
            return []
        placeholders = ",".join("?" for _ in book_ids)
        sql = f"""
            SELECT book_id, title, title_without_series, author_name, author_id,
                   average_rating, ratings_count, text_reviews_count, publication_year,
                   publisher, num_pages, description, genres, popular_shelves,
                   similar_books, url, image_url, bayesian_rating
            FROM books
            WHERE book_id IN ({placeholders});
        """
        with get_db_connection(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(sql, [str(b).strip() for b in book_ids])
            rows = cur.fetchall()
            results = []
            for row in rows:
                r = dict(row)
                r["genres"] = json.loads(r.get("genres") or "[]")
                r["popular_shelves"] = json.loads(r.get("popular_shelves") or "[]")
                r["similar_books"] = json.loads(r.get("similar_books") or "[]")
                results.append(r)
        return results

    def get_top_popular_books(self, limit: int = 10, min_ratings: int = 100) -> List[Dict[str, Any]]:
        """Retrieve top ranked romance books using Bayesian weighted rating."""
        sql = """
            SELECT book_id, title, author_name, average_rating, ratings_count,
                   text_reviews_count, publication_year, genres, bayesian_rating, url
            FROM books
            WHERE ratings_count >= ?
            ORDER BY bayesian_rating DESC, ratings_count DESC
            LIMIT ?;
        """
        with get_db_connection(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(sql, (min_ratings, limit))
            rows = [dict(r) for r in cur.fetchall()]
            for r in rows:
                r["genres"] = json.loads(r.get("genres") or "[]")
        return rows

    def get_author_insights(self, author_name: str) -> Dict[str, Any]:
        """Aggregate author statistics and list works in catalogue."""
        sql = """
            SELECT book_id, title, author_name, average_rating, ratings_count, publication_year,
                   bayesian_rating, genres
            FROM books
            WHERE author_name LIKE ?
            ORDER BY publication_year ASC, bayesian_rating DESC;
        """
        with get_db_connection(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(sql, (f"%{author_name.strip()}%",))
            books = [dict(r) for r in cur.fetchall()]
            for b in books:
                b["genres"] = json.loads(b.get("genres") or "[]")

        if not books:
            return {"author": author_name, "found": False, "books": [], "total_books": 0}

        ratings = [b["average_rating"] for b in books]
        total_ratings = sum(b["ratings_count"] for b in books)
        avg_rating = round(sum(ratings) / len(ratings), 2) if ratings else 0.0

        return {
            "author": books[0]["author_name"] if books else author_name,
            "found": True,
            "total_books_in_catalog": len(books),
            "average_author_rating": avg_rating,
            "total_reader_ratings": total_ratings,
            "catalog_works": books,
        }

    # =========================================================================
    # Trope Operations
    # =========================================================================

    def get_tropes_for_book(self, book_id: str) -> List[Dict[str, Any]]:
        """Retrieve extracted romance tropes for a book."""
        sql = """
            SELECT trope_name, confidence, method, evidence
            FROM tropes
            WHERE book_id = ?
            ORDER BY confidence DESC;
        """
        with get_db_connection(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(sql, (str(book_id).strip(),))
            return [dict(r) for r in cur.fetchall()]

    def find_books_by_tropes(
        self, trope_names: List[str], match_all: bool = False, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Find books matching specified romance tropes."""
        if not trope_names:
            return []

        placeholders = ",".join("?" for _ in trope_names)
        cleaned_tropes = [t.strip() for t in trope_names]

        if match_all:
            sql = f"""
                SELECT b.book_id, b.title, b.author_name, b.average_rating,
                       b.ratings_count, b.bayesian_rating, b.genres,
                       GROUP_CONCAT(t.trope_name, '; ') as matched_tropes,
                       COUNT(DISTINCT t.trope_name) as trope_count
                FROM books b
                JOIN tropes t ON b.book_id = t.book_id
                WHERE t.trope_name COLLATE NOCASE IN ({placeholders})
                GROUP BY b.book_id
                HAVING trope_count = ?
                ORDER BY b.bayesian_rating DESC
                LIMIT ?;
            """
            params = cleaned_tropes + [len(cleaned_tropes), limit]
        else:
            sql = f"""
                SELECT b.book_id, b.title, b.author_name, b.average_rating,
                       b.ratings_count, b.bayesian_rating, b.genres,
                       GROUP_CONCAT(t.trope_name, '; ') as matched_tropes,
                       COUNT(DISTINCT t.trope_name) as trope_count
                FROM books b
                JOIN tropes t ON b.book_id = t.book_id
                WHERE t.trope_name COLLATE NOCASE IN ({placeholders})
                GROUP BY b.book_id
                ORDER BY trope_count DESC, b.bayesian_rating DESC
                LIMIT ?;
            """
            params = cleaned_tropes + [limit]

        with get_db_connection(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(sql, params)
            rows = [dict(r) for r in cur.fetchall()]
            for r in rows:
                r["genres"] = json.loads(r.get("genres") or "[]")
                r["matched_tropes"] = [t.strip() for t in (r.get("matched_tropes") or "").split("; ") if t.strip()]
        return rows

    # =========================================================================
    # Review & Sentiment Operations
    # =========================================================================

    def get_reviews_for_book(self, book_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve reviews for a book."""
        sql = """
            SELECT review_id, rating, review_text, sentiment_score, sentiment_label, date_added
            FROM reviews
            WHERE book_id = ?
            ORDER BY date_added DESC
            LIMIT ?;
        """
        with get_db_connection(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(sql, (str(book_id).strip(), limit))
            return [dict(r) for r in cur.fetchall()]

    def get_review_sentiment_summary(self, book_id: str) -> Dict[str, Any]:
        """Aggregate sentiment distribution and reader themes for a book."""
        sql = """
            SELECT
                COUNT(*) as total_reviews,
                AVG(sentiment_score) as avg_sentiment,
                SUM(CASE WHEN sentiment_label = 'positive' THEN 1 ELSE 0 END) as positive_count,
                SUM(CASE WHEN sentiment_label = 'neutral' THEN 1 ELSE 0 END) as neutral_count,
                SUM(CASE WHEN sentiment_label = 'negative' THEN 1 ELSE 0 END) as negative_count
            FROM reviews
            WHERE book_id = ?;
        """
        with get_db_connection(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(sql, (str(book_id).strip(),))
            summary_row = cur.fetchone()
            summary = dict(summary_row) if summary_row else {}

            # Fetch themes
            cur.execute(
                "SELECT theme_type, theme_key, score, evidence FROM review_themes WHERE book_id = ? ORDER BY score DESC;",
                (str(book_id).strip(),),
            )
            themes = [dict(r) for r in cur.fetchall()]

        return {
            "book_id": book_id,
            "total_reviews_analyzed": summary.get("total_reviews", 0) or 0,
            "average_sentiment_polarity": round(summary.get("avg_sentiment") or 0.0, 3),
            "sentiment_distribution": {
                "positive": summary.get("positive_count", 0) or 0,
                "neutral": summary.get("neutral_count", 0) or 0,
                "negative": summary.get("negative_count", 0) or 0,
            },
            "themes": themes,
        }

    # =========================================================================
    # Reader Interaction Operations
    # =========================================================================

    def get_interactions_for_user(self, user_id: str) -> List[Dict[str, Any]]:
        """Retrieve interaction history for an anonymized user."""
        sql = """
            SELECT i.user_id, i.book_id, i.rating, i.is_read, i.date_added,
                   b.title, b.author_name, b.genres
            FROM interactions i
            JOIN books b ON i.book_id = b.book_id
            WHERE i.user_id = ?
            ORDER BY i.rating DESC, i.date_added DESC;
        """
        with get_db_connection(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(sql, (str(user_id).strip(),))
            rows = [dict(r) for r in cur.fetchall()]
            for r in rows:
                r["genres"] = json.loads(r.get("genres") or "[]")
        return rows

    def get_user_read_book_ids(self, user_id: str) -> Set[str]:
        """Get set of book IDs read by a user."""
        sql = "SELECT book_id FROM interactions WHERE user_id = ?;"
        with get_db_connection(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(sql, (str(user_id).strip(),))
            return {r["book_id"] for r in cur.fetchall()}

    def get_all_interactions(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve all interactions for recommendation matrix building."""
        sql = "SELECT user_id, book_id, rating FROM interactions"
        if limit:
            sql += f" LIMIT {int(limit)}"
        sql += ";"
        with get_db_connection(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(sql)
            return [dict(r) for r in cur.fetchall()]

    # =========================================================================
    # Catalogue & Analytics Operations
    # =========================================================================

    def get_catalog_statistics(self) -> Dict[str, Any]:
        """Return catalogue coverage, counts, and statistical aggregates."""
        with get_db_connection(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) as c FROM books;")
            total_books = cur.fetchone()["c"]

            cur.execute("SELECT COUNT(DISTINCT author_name) as c FROM books;")
            total_authors = cur.fetchone()["c"]

            cur.execute("SELECT COUNT(*) as c FROM interactions;")
            total_interactions = cur.fetchone()["c"]

            cur.execute("SELECT COUNT(DISTINCT user_id) as c FROM interactions;")
            total_readers = cur.fetchone()["c"]

            cur.execute("SELECT COUNT(*) as c FROM reviews;")
            total_reviews = cur.fetchone()["c"]

            cur.execute("SELECT COUNT(*) as c FROM tropes;")
            total_trope_links = cur.fetchone()["c"]

            cur.execute("SELECT AVG(average_rating) as avg_r, MIN(publication_year) as min_y, MAX(publication_year) as max_y FROM books;")
            agg = cur.fetchone()

        return {
            "total_books": total_books,
            "total_authors": total_authors,
            "total_readers": total_readers,
            "total_interactions": total_interactions,
            "total_reviews": total_reviews,
            "total_trope_classifications": total_trope_links,
            "catalog_average_rating": round(agg["avg_r"] or 0.0, 2),
            "publication_year_range": {"min": agg["min_y"], "max": agg["max_y"]},
            "dataset_provenance": "UCSD Goodreads Book Graph (Romance Subset, Wan & McAuley)",
        }

    def get_genre_distribution(self) -> Dict[str, int]:
        """Aggregate genre frequency across catalogue."""
        with get_db_connection(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT genres FROM books;")
            rows = cur.fetchall()

        genre_counts: Dict[str, int] = {}
        for r in rows:
            genres = json.loads(r["genres"] or "[]")
            for g in genres:
                genre_counts[g] = genre_counts.get(g, 0) + 1
        return dict(sorted(genre_counts.items(), key=lambda x: x[1], reverse=True))

    def get_top_rated_authors(self, limit: int = 10, min_books: int = 1) -> List[Dict[str, Any]]:
        """Return top rated authors with catalogue counts."""
        sql = """
            SELECT author_name, COUNT(*) as book_count,
                   AVG(average_rating) as avg_rating,
                   SUM(ratings_count) as total_ratings
            FROM books
            GROUP BY author_name
            HAVING book_count >= ?
            ORDER BY avg_rating DESC, total_ratings DESC
            LIMIT ?;
        """
        with get_db_connection(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(sql, (min_books, limit))
            return [
                {
                    "author_name": r["author_name"],
                    "book_count": r["book_count"],
                    "average_rating": round(r["avg_rating"], 2),
                    "total_ratings": r["total_ratings"],
                }
                for r in cur.fetchall()
            ]

    # =========================================================================
    # Bulk Ingestion Helpers
    # =========================================================================

    def insert_or_replace_books(self, records: List[Dict[str, Any]]) -> int:
        """Insert or replace book records in batch."""
        if not records:
            return 0
        sql = """
            INSERT OR REPLACE INTO books (
                book_id, title, title_without_series, author_name, author_id,
                average_rating, ratings_count, text_reviews_count, publication_year,
                publisher, num_pages, description, genres, popular_shelves,
                similar_books, url, image_url, bayesian_rating
            ) VALUES (
                :book_id, :title, :title_without_series, :author_name, :author_id,
                :average_rating, :ratings_count, :text_reviews_count, :publication_year,
                :publisher, :num_pages, :description, :genres, :popular_shelves,
                :similar_books, :url, :image_url, :bayesian_rating
            );
        """
        with get_db_connection(self.db_path) as conn:
            cur = conn.cursor()
            cur.executemany(sql, records)
        return len(records)

    def insert_or_replace_interactions(self, records: List[Dict[str, Any]]) -> int:
        """Insert or replace interaction records in batch."""
        if not records:
            return 0
        sql = """
            INSERT OR REPLACE INTO interactions (
                user_id, book_id, rating, is_read, date_added, read_at
            ) VALUES (
                :user_id, :book_id, :rating, :is_read, :date_added, :read_at
            );
        """
        with get_db_connection(self.db_path) as conn:
            cur = conn.cursor()
            cur.executemany(sql, records)
        return len(records)

    def insert_or_replace_reviews(self, records: List[Dict[str, Any]]) -> int:
        """Insert or replace review records in batch."""
        if not records:
            return 0
        sql = """
            INSERT OR REPLACE INTO reviews (
                review_id, user_id, book_id, rating, review_text, sentiment_score, sentiment_label, date_added
            ) VALUES (
                :review_id, :user_id, :book_id, :rating, :review_text, :sentiment_score, :sentiment_label, :date_added
            );
        """
        with get_db_connection(self.db_path) as conn:
            cur = conn.cursor()
            cur.executemany(sql, records)
        return len(records)

    def insert_or_replace_tropes(self, records: List[Dict[str, Any]]) -> int:
        """Insert or replace trope records in batch."""
        if not records:
            return 0
        sql = """
            INSERT OR REPLACE INTO tropes (
                book_id, trope_name, confidence, method, evidence
            ) VALUES (
                :book_id, :trope_name, :confidence, :method, :evidence
            );
        """
        with get_db_connection(self.db_path) as conn:
            cur = conn.cursor()
            cur.executemany(sql, records)
        return len(records)

    def insert_or_replace_review_themes(self, records: List[Dict[str, Any]]) -> int:
        """Insert review theme records in batch."""
        if not records:
            return 0
        sql = """
            INSERT INTO review_themes (
                book_id, theme_type, theme_key, score, evidence
            ) VALUES (
                :book_id, :theme_type, :theme_key, :score, :evidence
            );
        """
        with get_db_connection(self.db_path) as conn:
            cur = conn.cursor()
            cur.executemany(sql, records)
        return len(records)
