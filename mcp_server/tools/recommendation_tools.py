"""Recommendation MCP tools exposing Popularity, Content-Based, and Hybrid engines."""

from typing import Any, Dict, List, Optional
from mcp.server.mcpserver import MCPServer
from services.recommendation_service import RecommendationService

def register_recommendation_tools(
    server: MCPServer, rec_service: RecommendationService
) -> None:
    """Register recommendation tools."""

    @server.tool(
        name="recommend_similar_books",
        description=(
            "Recommend romance books most similar to a supplied book ID or title query using "
            "TF-IDF content vectorization and cosine similarity. Allows filtering by target romance tropes "
            "and provides mathematical similarity scores and explanatory rationale."
        ),
    )
    def recommend_similar_books(
        book_id: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 5,
        target_tropes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Recommend books similar to a target title or book_id."""
        return rec_service.recommend_similar_books(
            book_id=book_id, query=query, limit=limit, target_tropes=target_tropes
        )

    @server.tool(
        name="get_personalized_recommendations",
        description=(
            "Generate personalized romance book recommendations for an anonymized dataset reader ID "
            "or an explicit preference profile (favorite genres or desired tropes). Uses hybrid "
            "collaborative filtering and content-based scoring, with explicit cold-start handling."
        ),
    )
    def get_personalized_recommendations(
        user_id: Optional[str] = None,
        favorite_genres: Optional[List[str]] = None,
        desired_tropes: Optional[List[str]] = None,
        limit: int = 5,
    ) -> Dict[str, Any]:
        """Generate personalized recommendations for a reader profile."""
        return rec_service.get_personalized_recommendations(
            user_id=user_id,
            favorite_genres=favorite_genres,
            desired_tropes=desired_tropes,
            limit=limit,
        )

    @server.tool(
        name="get_popular_books",
        description=(
            "Retrieve top-rated, highly popular romance books ranked by Bayesian Weighted Rating "
            "(smoothing raw star averages against global catalogue volume to prevent low-sample bias). "
            "Supports optional subgenre filtering."
        ),
    )
    def get_popular_books(
        limit: int = 10,
        min_ratings: int = 100,
        genre: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return popular books ranked by Bayesian shrinkage."""
        return rec_service.get_popular_books(
            limit=limit, min_ratings=min_ratings, genre=genre
        )
