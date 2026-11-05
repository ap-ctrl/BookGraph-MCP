"""Book service handling search, detail retrieval, author analytics, and book comparison."""

from typing import Any, Dict, List, Optional
from config.settings import logger
from database.repository import BookRepository

class BookService:
    """Domain service for book discovery, metadata inspection, and comparisons."""

    def __init__(self, repo: BookRepository):
        self.repo = repo

    def search_books(
        self,
        query: Optional[str] = None,
        author: Optional[str] = None,
        genre: Optional[str] = None,
        min_rating: Optional[float] = None,
        publication_year: Optional[int] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Search books with structured pagination and validation."""
        limit = max(1, min(limit, 50))  # Apply sensible limits
        offset = max(0, offset)

        books, total = self.repo.search_books(
            query=query,
            author=author,
            genre=genre,
            min_rating=min_rating,
            publication_year=publication_year,
            limit=limit,
            offset=offset,
        )

        return {
            "query": query,
            "filters": {
                "author": author,
                "genre": genre,
                "min_rating": min_rating,
                "publication_year": publication_year,
            },
            "pagination": {"limit": limit, "offset": offset, "total_matches": total},
            "results": [
                {
                    "book_id": b["book_id"],
                    "title": b["title"],
                    "author": b["author_name"],
                    "average_rating": b["average_rating"],
                    "bayesian_rating": b["bayesian_rating"],
                    "ratings_count": b["ratings_count"],
                    "publication_year": b["publication_year"],
                    "genres": b["genres"],
                    "description": (b["description"][:280] + "...") if b["description"] and len(b["description"]) > 280 else b["description"],
                }
                for b in books
            ],
        }

    def get_book_details(self, book_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve full book metadata, tropes, and sentiment summary."""
        book = self.repo.get_book_by_id(book_id)
        if not book:
            return None

        tropes = self.repo.get_tropes_for_book(book_id)
        sentiment_summary = self.repo.get_review_sentiment_summary(book_id)

        return {
            "book_id": book["book_id"],
            "title": book["title"],
            "title_without_series": book["title_without_series"],
            "author": book["author_name"],
            "author_id": book["author_id"],
            "average_rating": book["average_rating"],
            "bayesian_rating": book["bayesian_rating"],
            "ratings_count": book["ratings_count"],
            "text_reviews_count": book["text_reviews_count"],
            "publication_year": book["publication_year"],
            "publisher": book["publisher"],
            "num_pages": book["num_pages"],
            "description": book["description"],
            "genres": book["genres"],
            "popular_shelves": [s.get("name") for s in book["popular_shelves"] if isinstance(s, dict)][:8],
            "confirmed_tropes": tropes,
            "review_sentiment": sentiment_summary,
            "similar_book_ids": book["similar_books"][:5],
            "url": book["url"],
            "image_url": book["image_url"],
        }

    def compare_books(self, book_id_1: str, book_id_2: str) -> Dict[str, Any]:
        """Perform side-by-side comparison between two romance titles."""
        b1 = self.get_book_details(book_id_1)
        b2 = self.get_book_details(book_id_2)

        if not b1 or not b2:
            return {
                "error": "One or both books could not be found in the database catalogue.",
                "book_1_found": b1 is not None,
                "book_2_found": b2 is not None,
            }

        # Compare tropes
        b1_tropes = {t["trope_name"] for t in b1["confirmed_tropes"]}
        b2_tropes = {t["trope_name"] for t in b2["confirmed_tropes"]}

        shared_tropes = sorted(list(b1_tropes & b2_tropes))
        b1_exclusive = sorted(list(b1_tropes - b2_tropes))
        b2_exclusive = sorted(list(b2_tropes - b1_tropes))

        # Compare genres
        b1_genres = set(b1["genres"])
        b2_genres = set(b2["genres"])
        shared_genres = sorted(list(b1_genres & b2_genres))

        return {
            "comparison_summary": f"Comparison between '{b1['title']}' and '{b2['title']}'",
            "book_1": {
                "book_id": b1["book_id"],
                "title": b1["title"],
                "author": b1["author"],
                "average_rating": b1["average_rating"],
                "bayesian_rating": b1["bayesian_rating"],
                "ratings_count": b1["ratings_count"],
                "num_pages": b1["num_pages"],
                "publication_year": b1["publication_year"],
                "tropes": [t["trope_name"] for t in b1["confirmed_tropes"]],
            },
            "book_2": {
                "book_id": b2["book_id"],
                "title": b2["title"],
                "author": b2["author"],
                "average_rating": b2["average_rating"],
                "bayesian_rating": b2["bayesian_rating"],
                "ratings_count": b2["ratings_count"],
                "num_pages": b2["num_pages"],
                "publication_year": b2["publication_year"],
                "tropes": [t["trope_name"] for t in b2["confirmed_tropes"]],
            },
            "trope_overlap": {
                "shared_tropes": shared_tropes,
                "book_1_exclusive": b1_exclusive,
                "book_2_exclusive": b2_exclusive,
            },
            "genre_overlap": shared_genres,
        }

    def get_author_insights(self, author_name: str) -> Dict[str, Any]:
        """Retrieve aggregated author statistics and catalogue bibliography."""
        return self.repo.get_author_insights(author_name)

    def get_catalogue_statistics(self) -> Dict[str, Any]:
        """Retrieve catalogue size, counts, and date freshness."""
        return self.repo.get_catalog_statistics()

    def get_genre_insights(self) -> Dict[str, Any]:
        """Retrieve breakdown of genres and top titles per genre."""
        dist = self.repo.get_genre_distribution()
        return {
            "total_distinct_genres": len(dist),
            "genre_frequencies": dist,
            "primary_romance_subgenres": list(dist.keys())[:6],
        }
