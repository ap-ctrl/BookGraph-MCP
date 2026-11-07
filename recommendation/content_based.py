"""Content-based book recommendation engine using TF-IDF and Cosine Similarity."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import logger
from database.repository import BookRepository

class ContentBasedRecommender:
    """TF-IDF Content-Based Filtering on romance book metadata, descriptions, and tropes."""

    def __init__(self, repo: BookRepository):
        self.repo = repo
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
        )
        self.book_ids: List[str] = []
        self.tfidf_matrix = None
        self.is_fitted = False

    def build_feature_corpus(self, books: List[Dict[str, Any]]) -> List[str]:
        """Combine metadata into rich textual representation for vectorization."""
        corpus = []
        for b in books:
            genres_str = " ".join(b.get("genres", []))
            desc = b.get("description", "")
            title = b.get("title", "")
            author = b.get("author_name", "")
            # Weight author and genres heavily by repeating
            combined = f"{title} {author} {author} {genres_str} {genres_str} {desc}"
            corpus.append(combined)
        return corpus

    def fit(self, books: Optional[List[Dict[str, Any]]] = None) -> None:
        """Fit vectorizer on the provided list of book metadata.
        The caller may supply an explicit list of book dictionaries (as returned by
        ``BookRepository.search_books``). If None, fetches up to 10,000 books from
        the repository catalogue.
        """
        if books is None:
            books, _ = self.repo.search_books(limit=10000)

        if not books:
            logger.warning("No books provided to fit ContentBasedRecommender.")
            return

        self.book_ids = [b["book_id"] for b in books]
        corpus = self.build_feature_corpus(books)
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.is_fitted = True
        logger.info("ContentBasedRecommender fitted on %d books.", len(self.book_ids))

    def resolve_book_id(self, query: str) -> Optional[str]:
        """Resolve a freeform title/author search query to a canonical book_id."""
        books, _ = self.repo.search_books(query=query, limit=1)
        if books:
            return books[0]["book_id"]
        # Try author search
        books, _ = self.repo.search_books(author=query, limit=1)
        if books:
            return books[0]["book_id"]
        return None

    def recommend_similar(
        self,
        book_id: str,
        limit: int = 5,
        target_tropes: Optional[List[str]] = None,
        exclude_book_ids: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Recommend books most similar to target book_id using cosine similarity.
        Optionally filter or boost candidates containing specified target tropes.
        """
        excluded = set(exclude_book_ids or [])
        if not self.is_fitted:
            self.fit()

        if book_id not in self.book_ids:
            # Fallback to book's similar_books field if present
            target = self.repo.get_book_by_id(book_id)
            if not target:
                return []
            sim_ids = [b for b in target.get("similar_books", []) if b not in excluded]
            sim_books = self.repo.get_books_by_ids(sim_ids[:limit])
            return [
                {
                    "book_id": b["book_id"],
                    "title": b["title"],
                    "author": b["author_name"],
                    "similarity_score": 0.85,
                    "average_rating": b["average_rating"],
                    "genres": b.get("genres", []),
                    "explanation": "Derived from publisher-associated similar works.",
                }
                for b in sim_books
            ]

        target_idx = self.book_ids.index(book_id)
        target_vector = self.tfidf_matrix[target_idx]
        similarities = cosine_similarity(target_vector, self.tfidf_matrix)[0]

        # Get sorted indices descending (excluding self)
        ranked_indices = np.argsort(similarities)[::-1]
        ranked_indices = [i for i in ranked_indices if i != target_idx]

        target_book = self.repo.get_book_by_id(book_id) or {}
        target_title = target_book.get("title", "Reference Book")

        candidates = []
        for idx in ranked_indices:
            cid = self.book_ids[idx]
            if cid in excluded:
                continue
            sim_score = float(similarities[idx])
            cand_book = self.repo.get_book_by_id(cid)
            if not cand_book:
                continue

            # Fetch tropes for this candidate
            cand_tropes = [t["trope_name"] for t in self.repo.get_tropes_for_book(cid)]

            # Check if trope filtering is requested
            if target_tropes:
                matched_tropes = [t for t in target_tropes if t.strip().title() in cand_tropes]
                if not matched_tropes:
                    continue
            else:
                matched_tropes = cand_tropes

            # Identify overlapping genres
            overlap_genres = list(set(target_book.get("genres", [])) & set(cand_book.get("genres", [])))

            explanation = (
                f"Shares {sim_score:.2f} content similarity with '{target_title}' "
                f"based on overlapping themes in {', '.join(overlap_genres or cand_book.get('genres', []))}."
            )
            if matched_tropes:
                explanation += f" Features confirmed tropes: {', '.join(matched_tropes[:3])}."

            candidates.append({
                "book_id": cid,
                "title": cand_book["title"],
                "author": cand_book["author_name"],
                "similarity_score": round(sim_score, 3),
                "average_rating": cand_book["average_rating"],
                "bayesian_rating": cand_book["bayesian_rating"],
                "genres": cand_book.get("genres", []),
                "tropes": cand_tropes,
                "explanation": explanation,
            })

            if len(candidates) >= limit:
                break

        return candidates
