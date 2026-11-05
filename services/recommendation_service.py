"""Recommendation service coordinating Content, Collaborative, Popularity, and Hybrid models."""

from typing import Any, Dict, List, Optional
from config.settings import logger
from database.repository import BookRepository
from recommendation.collaborative import CollaborativeRecommender
from recommendation.content_based import ContentBasedRecommender
from recommendation.hybrid import HybridRecommender
from recommendation.popularity import PopularityRecommender

class RecommendationService:
    """Provides high-level recommendation endpoints to MCP tools."""

    def __init__(self, repo: BookRepository):
        self.repo = repo
        self.popularity_rec = PopularityRecommender(repo)
        self.content_rec = ContentBasedRecommender(repo)
        self.collab_rec = CollaborativeRecommender(repo)
        self.hybrid_rec = HybridRecommender(
            repo, self.content_rec, self.collab_rec, self.popularity_rec, content_weight=0.5
        )

        # Lazy initialize models
        self._ensure_models_fitted()

    def _ensure_models_fitted(self) -> None:
        """Fit content and collaborative models if needed."""
        if not self.content_rec.is_fitted:
            self.content_rec.fit()
        if not self.collab_rec.is_fitted:
            self.collab_rec.fit()

    def recommend_similar_books(
        self,
        book_id: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 5,
        target_tropes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Recommend books similar to a supplied book ID or query title.
        Includes similarity scores and supporting feature rationale.
        """
        self._ensure_models_fitted()
        limit = max(1, min(limit, 25))

        target_id = book_id
        if not target_id and query:
            target_id = self.content_rec.resolve_book_id(query)

        if not target_id:
            return {
                "error": "Could not identify target reference book. Please supply a valid book_id or title query.",
                "recommendations": [],
            }

        ref_book = self.repo.get_book_by_id(target_id)
        if not ref_book:
            return {
                "error": f"Book ID '{target_id}' not found in the romance database.",
                "recommendations": [],
            }

        recs = self.content_rec.recommend_similar(
            target_id, limit=limit, target_tropes=target_tropes
        )

        return {
            "reference_book": {
                "book_id": ref_book["book_id"],
                "title": ref_book["title"],
                "author": ref_book["author_name"],
                "average_rating": ref_book["average_rating"],
                "genres": ref_book.get("genres", []),
            },
            "requested_tropes": target_tropes or [],
            "count": len(recs),
            "recommendations": recs,
        }

    def get_personalized_recommendations(
        self,
        user_id: Optional[str] = None,
        favorite_genres: Optional[List[str]] = None,
        desired_tropes: Optional[List[str]] = None,
        limit: int = 5,
    ) -> Dict[str, Any]:
        """
        Personalized recommendations using Hybrid / Collaborative engine.
        Handles cold-start readers gracefully.
        """
        self._ensure_models_fitted()
        limit = max(1, min(limit, 25))

        return self.hybrid_rec.recommend_for_reader(
            user_id=user_id,
            favorite_genres=favorite_genres,
            desired_tropes=desired_tropes,
            limit=limit,
        )

    def get_popular_books(
        self,
        limit: int = 10,
        min_ratings: int = 100,
        genre: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retrieve popular romance books ranked by Bayesian weighted rating."""
        limit = max(1, min(limit, 50))
        recs = self.popularity_rec.recommend(limit=limit, min_ratings=min_ratings, genre=genre)
        return {
            "ranking_method": "Bayesian Weighted Rating (shrinkage smoothed with global catalogue mean)",
            "genre_filter": genre,
            "min_ratings_threshold": min_ratings,
            "count": len(recs),
            "books": recs,
        }
