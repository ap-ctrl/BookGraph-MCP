"""Collaborative filtering recommender using sparse matrices and item-item similarity."""

from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import logger
from database.repository import BookRepository

class CollaborativeRecommender:
    """Memory-efficient Item-Item Collaborative Filtering using Sparse CSR representation."""

    def __init__(self, repo: BookRepository):
        self.repo = repo
        self.user_to_idx: Dict[str, int] = {}
        self.idx_to_user: Dict[int, str] = {}
        self.book_to_idx: Dict[str, int] = {}
        self.idx_to_book: Dict[int, str] = {}
        self.user_item_matrix: Optional[csr_matrix] = None
        self.item_similarity: Optional[np.ndarray] = None
        self.is_fitted = False

    def fit(self, interactions: Optional[List[Dict[str, Any]]] = None) -> None:
        """Construct sparse user-item interaction matrix and compute item similarities."""
        data = interactions if interactions is not None else self.repo.get_all_interactions()
        if not data:
            logger.warning("No interactions available to fit CollaborativeRecommender.")
            return

        df = pd.DataFrame(data)
        users = df["user_id"].unique()
        books = df["book_id"].unique()

        self.user_to_idx = {u: i for i, u in enumerate(users)}
        self.idx_to_user = {i: u for i, u in enumerate(users)}
        self.book_to_idx = {b: i for i, b in enumerate(books)}
        self.idx_to_book = {i: b for i, b in enumerate(books)}

        row_indices = df["user_id"].map(self.user_to_idx).values
        col_indices = df["book_id"].map(self.book_to_idx).values
        ratings = df["rating"].astype(float).values

        # Build sparse CSR matrix of shape (n_users, n_books)
        n_users = len(users)
        n_books = len(books)
        self.user_item_matrix = csr_matrix((ratings, (row_indices, col_indices)), shape=(n_users, n_books))

        # Compute item-item cosine similarity on sparse column vectors (shape: n_books x n_books)
        # Transpose user_item_matrix to get item x user matrix
        item_user_matrix = self.user_item_matrix.T
        self.item_similarity = cosine_similarity(item_user_matrix, dense_output=True)
        # Zero out diagonal
        np.fill_diagonal(self.item_similarity, 0.0)

        self.is_fitted = True
        logger.info(
            "CollaborativeRecommender fitted: %d users, %d books, %d interactions.",
            n_users,
            n_books,
            len(df),
        )

    def recommend_for_user(
        self, user_id: str, limit: int = 5
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Recommend books for user based on item-item collaborative filtering.
        Returns (recommendations, is_cold_start).
        """
        if not self.is_fitted:
            self.fit()

        if user_id not in self.user_to_idx or self.user_item_matrix is None:
            # Cold start: unknown user
            return [], True

        u_idx = self.user_to_idx[user_id]
        user_ratings = self.user_item_matrix.getrow(u_idx).toarray().flatten()

        rated_indices = np.where(user_ratings > 0)[0]
        if len(rated_indices) < 2:
            # Cold start: insufficient interaction history
            return [], True

        # Predict score for unrated items
        # score(i) = sum(similarity(i, j) * rating(j)) / (sum(abs(similarity(i, j))) + 1e-9)
        unrated_indices = np.where(user_ratings == 0)[0]
        if len(unrated_indices) == 0:
            return [], False

        sim_sub = self.item_similarity[unrated_indices, :][:, rated_indices]
        ratings_sub = user_ratings[rated_indices]

        numerator = np.dot(sim_sub, ratings_sub)
        denominator = np.sum(np.abs(sim_sub), axis=1) + 1e-9
        pred_scores = numerator / denominator

        top_local_idx = np.argsort(pred_scores)[::-1][:limit]
        recs = []

        for local_idx in top_local_idx:
            item_idx = unrated_indices[local_idx]
            book_id = self.idx_to_book[item_idx]
            score = float(pred_scores[local_idx])
            book = self.repo.get_book_by_id(book_id)
            if not book:
                continue

            recs.append({
                "book_id": book_id,
                "title": book["title"],
                "author": book["author_name"],
                "predicted_rating": round(score, 2),
                "average_rating": book["average_rating"],
                "genres": book.get("genres", []),
                "method": "collaborative_item_item",
                "explanation": (
                    f"Highly correlated with reader's past high ratings in your reading profile "
                    f"(Predicted score: {score:.2f})."
                ),
            })

        return recs, False
