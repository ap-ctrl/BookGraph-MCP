"""Hybrid recommendation combining Content-Based and Collaborative Filtering."""

from typing import Any, Dict, List, Optional
import numpy as np

from database.repository import BookRepository
from recommendation.collaborative import CollaborativeRecommender
from recommendation.content_based import ContentBasedRecommender
from recommendation.popularity import PopularityRecommender

class HybridRecommender:
    """
    Combines Collaborative Filtering and Content-Based Features with configurable weights.
    Explicitly handles cold-start users and missing signals.
    """

    def __init__(
        self,
        repo: BookRepository,
        content_recommender: ContentBasedRecommender,
        collab_recommender: CollaborativeRecommender,
        popularity_recommender: PopularityRecommender,
        content_weight: float = 0.5,
    ):
        self.repo = repo
        self.content_recommender = content_recommender
        self.collab_recommender = collab_recommender
        self.popularity_recommender = popularity_recommender
        self.content_weight = content_weight

    def recommend_for_reader(
        self,
        user_id: Optional[str] = None,
        favorite_genres: Optional[List[str]] = None,
        desired_tropes: Optional[List[str]] = None,
        limit: int = 5,
    ) -> Dict[str, Any]:
        """
        Generate recommendations for a dataset reader ID or preference profile.
        Handles cold-start cases explicitly.
        """
        is_cold_start = False
        cold_start_reason = ""
        user_history = []

        if user_id:
            user_history = self.repo.get_interactions_for_user(user_id)

        # Case 1: Known reader with collaborative history
        if user_id and len(user_history) >= 2:
            collab_recs, cold = self.collab_recommender.recommend_for_user(user_id, limit=limit * 2)
            if not cold and collab_recs:
                # Augment with content signals from user's top-rated book
                top_book = max(user_history, key=lambda x: x["rating"])
                content_recs = self.content_recommender.recommend_similar(
                    top_book["book_id"], limit=limit * 2, target_tropes=desired_tropes
                )

                # Merge scores
                scores: Dict[str, Dict[str, Any]] = {}
                raw_max = max([r["predicted_rating"] for r in collab_recs] or [5.0])
                collab_max = max(1e-6, raw_max)

                for r in collab_recs:
                    bid = r["book_id"]
                    norm_collab = r["predicted_rating"] / collab_max
                    scores[bid] = {
                        "book_id": bid,
                        "title": r["title"],
                        "author": r["author"],
                        "collab_score": norm_collab,
                        "content_score": 0.0,
                        "explanation": r["explanation"],
                    }

                for r in content_recs:
                    bid = r["book_id"]
                    if bid in scores:
                        scores[bid]["content_score"] = r["similarity_score"]
                        scores[bid]["explanation"] += f" Also {r['explanation']}"
                    else:
                        scores[bid] = {
                            "book_id": bid,
                            "title": r["title"],
                            "author": r["author"],
                            "collab_score": 0.0,
                            "content_score": r["similarity_score"],
                            "explanation": r["explanation"],
                        }

                # Compute hybrid weighted score
                final_ranked = []
                for bid, item in scores.items():
                    hybrid_score = (
                        self.content_weight * item["content_score"]
                        + (1 - self.content_weight) * item["collab_score"]
                    )
                    book = self.repo.get_book_by_id(bid)
                    if not book:
                        continue
                    final_ranked.append({
                        "book_id": bid,
                        "title": item["title"],
                        "author": item["author"],
                        "hybrid_score": round(hybrid_score, 3),
                        "average_rating": book["average_rating"],
                        "bayesian_rating": book["bayesian_rating"],
                        "genres": book.get("genres", []),
                        "explanation": (
                            f"Hybrid recommendation (weights: {self.content_weight:.1f} content / "
                            f"{1-self.content_weight:.1f} collaborative). {item['explanation']}"
                        ),
                    })

                final_ranked.sort(key=lambda x: x["hybrid_score"], reverse=True)
                return {
                    "user_id": user_id,
                    "is_cold_start": False,
                    "recommendations": final_ranked[:limit],
                    "model_weights": {"content": self.content_weight, "collaborative": 1 - self.content_weight},
                }

        # Case 2: Cold start reader (Unknown user, 0 or 1 interaction)
        is_cold_start = True
        if not user_id:
            cold_start_reason = "No reader ID supplied; preference profile or popularity fallback utilized."
        elif len(user_history) == 0:
            cold_start_reason = f"Reader '{user_id}' has no prior interactions in the UCSD romance dataset."
        else:
            cold_start_reason = f"Reader '{user_id}' has only {len(user_history)} interaction (insufficient for collaborative filtering)."

        # Fallback to trope/genre matching or popularity baseline
        recs = []
        if desired_tropes:
            trope_books = self.repo.find_books_by_tropes(desired_tropes, limit=limit)
            for b in trope_books:
                recs.append({
                    "book_id": b["book_id"],
                    "title": b["title"],
                    "author": b["author_name"],
                    "average_rating": b["average_rating"],
                    "bayesian_rating": b["bayesian_rating"],
                    "genres": b.get("genres", []),
                    "matched_tropes": b.get("matched_tropes", []),
                    "explanation": (
                        f"Selected based on requested tropes ({', '.join(desired_tropes)}) "
                        f"and high Bayesian rating ({b['bayesian_rating']:.2f})."
                    ),
                })
        elif favorite_genres:
            genre_name = favorite_genres[0]
            books, _ = self.repo.search_books(genre=genre_name, limit=limit)
            for b in books:
                recs.append({
                    "book_id": b["book_id"],
                    "title": b["title"],
                    "author": b["author_name"],
                    "average_rating": b["average_rating"],
                    "bayesian_rating": b["bayesian_rating"],
                    "genres": b.get("genres", []),
                    "explanation": f"Top ranked {genre_name} book by reader acclaim ({b['bayesian_rating']:.2f}).",
                })
        else:
            # Popularity baseline fallback
            pop_books = self.popularity_recommender.recommend(limit=limit)
            recs = [
                {
                    "book_id": b["book_id"],
                    "title": b["title"],
                    "author": b["author"],
                    "average_rating": b["average_rating"],
                    "bayesian_rating": b["bayesian_rating"],
                    "genres": b.get("genres", []),
                    "explanation": f"Popularity baseline fallback: {b['recommendation_rationale']}",
                }
                for b in pop_books
            ]

        return {
            "user_id": user_id,
            "is_cold_start": True,
            "cold_start_reason": cold_start_reason,
            "recommendations": recs[:limit],
        }
