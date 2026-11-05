"""Popularity-based recommendation engine using Bayesian Weighted Rating."""

from typing import Any, Dict, List, Optional
import pandas as pd
from database.repository import BookRepository

class PopularityRecommender:
    """
    Bayesian Weighted Rating Recommender:
    WR = (v / (v + m)) * R + (m / (v + m)) * C
    where:
      R = average rating of book
      v = number of ratings for book
      m = minimum votes required to be listed (smoothing threshold)
      C = mean rating across the entire catalogue
    """

    def __init__(self, repo: BookRepository):
        self.repo = repo

    def recommend(
        self,
        limit: int = 10,
        min_ratings: int = 100,
        genre: Optional[str] = None,
        exclude_book_ids: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Return top ranked books by Bayesian shrinkage, optionally filtering excluded books."""
        excluded = set(exclude_book_ids or [])
        fetch_limit = limit * 3 + len(excluded)
        books = self.repo.get_top_popular_books(limit=fetch_limit, min_ratings=min_ratings)

        # Fallback if strict min_ratings yields fewer books than limit
        if len(books) < limit and min_ratings > 0:
            books = self.repo.get_top_popular_books(limit=fetch_limit, min_ratings=0)

        if genre:
            genre_lower = genre.lower()
            books = [
                b for b in books
                if any(genre_lower in g.lower() for g in b.get("genres", []))
            ]

        if excluded:
            books = [b for b in books if b["book_id"] not in excluded]

        results = []
        for rank, b in enumerate(books[:limit], start=1):
            results.append({
                "rank": rank,
                "book_id": b["book_id"],
                "title": b["title"],
                "author": b["author_name"],
                "average_rating": b["average_rating"],
                "bayesian_rating": b["bayesian_rating"],
                "ratings_count": b["ratings_count"],
                "genres": b.get("genres", []),
                "recommendation_rationale": (
                    f"Rank #{rank} by Bayesian weighted rating ({b['bayesian_rating']:.2f}) "
                    f"across {b['ratings_count']:,} reader reviews."
                ),
            })
        return results
