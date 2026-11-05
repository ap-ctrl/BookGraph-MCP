"""NLP and Reader Intelligence service for Tropes, Sentiments, and Themes."""

from typing import Any, Dict, List, Optional
from config.settings import logger
from database.repository import BookRepository

class NLPService:
    """Service handling romance trope matching, review sentiment, and reader analytics."""

    def __init__(self, repo: BookRepository):
        self.repo = repo

    def find_books_by_tropes(
        self, tropes: List[str], match_all: bool = False, limit: int = 10
    ) -> Dict[str, Any]:
        """
        Search books exhibiting one or more romance tropes.
        Includes classification confidence, evidence, and detection methods.
        """
        if not tropes:
            return {"error": "Please provide at least one trope name.", "books": []}

        limit = max(1, min(limit, 50))
        books = self.repo.find_books_by_tropes(tropes, match_all=match_all, limit=limit)

        results = []
        for b in books:
            book_tropes = self.repo.get_tropes_for_book(b["book_id"])
            matched_tropes_info = [
                t for t in book_tropes
                if any(req.strip().lower() in t["trope_name"].lower() for req in tropes)
            ]

            results.append({
                "book_id": b["book_id"],
                "title": b["title"],
                "author": b["author_name"],
                "average_rating": b["average_rating"],
                "bayesian_rating": b["bayesian_rating"],
                "genres": b.get("genres", []),
                "matched_tropes_with_evidence": matched_tropes_info,
            })

        return {
            "requested_tropes": tropes,
            "match_mode": "all" if match_all else "any",
            "total_matches": len(results),
            "books": results,
        }

    def analyse_book_reviews(self, book_id: str) -> Dict[str, Any]:
        """
        Retrieve stored sentiment statistics and thematic breakdown (pacing, character, plot).
        Strictly reports processed data without fabricating statistics.
        """
        book = self.repo.get_book_by_id(book_id)
        if not book:
            return {"error": f"Book with ID '{book_id}' not found in database.", "found": False}

        summary = self.repo.get_review_sentiment_summary(book_id)
        raw_reviews = self.repo.get_reviews_for_book(book_id, limit=3)

        # Do not expose lengthy copyrighted review text; provide short summaries
        sample_quotes = []
        for r in raw_reviews:
            quote = r.get("review_text", "")
            if len(quote) > 140:
                quote = quote[:137] + "..."
            sample_quotes.append({
                "rating": r["rating"],
                "sentiment": r.get("sentiment_label"),
                "excerpt": quote,
            })

        themes_by_category = {"character": [], "pacing": [], "plot": []}
        for t in summary.get("themes", []):
            cat = t.get("theme_type")
            if cat in themes_by_category:
                themes_by_category[cat].append({
                    "theme": t["theme_key"],
                    "frequency_score": t["score"],
                    "evidence_snippet": t.get("evidence"),
                })

        return {
            "book_id": book_id,
            "found": True,
            "title": book["title"],
            "author": book["author_name"],
            "average_rating": book["average_rating"],
            "total_reviews_analyzed": summary["total_reviews_analyzed"],
            "sentiment_overview": {
                "average_polarity": summary["average_sentiment_polarity"],
                "distribution": summary["sentiment_distribution"],
            },
            "thematic_feedback": themes_by_category,
            "representative_review_excerpts": sample_quotes,
        }

    def get_reader_insights(self, user_id: str) -> Dict[str, Any]:
        """
        Summarise reader interaction profile with anonymisation.
        Does not expose raw user metadata or unnecessary identifiers.
        """
        interactions = self.repo.get_interactions_for_user(user_id)
        if not interactions:
            return {
                "user_id": user_id,
                "found": False,
                "message": f"No interaction history found for reader '{user_id}'.",
            }

        ratings = [i["rating"] for i in interactions if i["rating"] > 0]
        avg_user_rating = round(sum(ratings) / len(ratings), 2) if ratings else 0.0

        # Extract top genres read
        genre_counts: Dict[str, int] = {}
        for i in interactions:
            for g in i.get("genres", []):
                genre_counts[g] = genre_counts.get(g, 0) + 1

        favorite_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)

        return {
            "reader_id": user_id,
            "found": True,
            "total_books_shelved": len(interactions),
            "total_books_read": sum(1 for i in interactions if i["is_read"] == 1),
            "average_star_rating_given": avg_user_rating,
            "favorite_genres": [g[0] for g in favorite_genres[:3]],
            "reading_history_summary": [
                {
                    "book_id": i["book_id"],
                    "title": i["title"],
                    "author": i["author_name"],
                    "rating_given": i["rating"],
                }
                for i in interactions[:5]
            ],
        }
