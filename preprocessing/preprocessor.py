"""Data cleaning, validation, and preprocessing for Romance datasets."""

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from config.settings import logger, settings

GENRE_KEYWORDS = {
    "contemporary": "Contemporary Romance",
    "historical": "Historical Romance",
    "fantasy": "Fantasy Romance / Romantasy",
    "paranormal": "Paranormal Romance",
    "dark": "Dark Romance",
    "sports": "Sports Romance",
    "erotic": "Erotic Romance",
    "ya": "Young Adult Romance",
    "new-adult": "New Adult Romance",
    "lgbtq": "Queer / LGBTQ+ Romance",
    "chick-lit": "Romantic Comedy",
}

class DataPreprocessor:
    """Independent cleaners and pipeline orchestrator for books, interactions, and reviews."""

    def __init__(self, processed_dir: Optional[Path] = None):
        self.processed_dir = processed_dir or settings.processed_data_dir
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.validation_stats: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset_mode": "development_sample",
            "processing_time_seconds": 0.0,
            "books": {"input": 0, "output": 0, "missing_desc": 0, "missing_year": 0, "duplicates": 0, "failures": 0},
            "interactions": {"input": 0, "output": 0, "duplicates": 0, "invalid_book_ref": 0, "failures": 0},
            "reviews": {"input": 0, "output": 0, "empty_text": 0, "invalid_book_ref": 0, "duplicates": 0, "failures": 0},
        }

    @staticmethod
    def normalize_text(text: Optional[str]) -> str:
        """Strip HTML tags, extraneous whitespace, and normalize unicode."""
        if not text:
            return ""
        # Remove HTML tags if present
        clean = re.sub(r"<[^>]+>", " ", text)
        # Normalize whitespace
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean

    @staticmethod
    def extract_genres_from_shelves(shelves: Any) -> List[str]:
        """Infer categorized romance subgenres from shelf tags."""
        genres = set()
        if not shelves:
            return ["Romance"]

        shelf_list = shelves if isinstance(shelves, list) else []
        for shelf in shelf_list:
            if isinstance(shelf, dict):
                name = shelf.get("name", "").lower()
            elif isinstance(shelf, str):
                name = shelf.lower()
            else:
                continue

            for keyword, genre_label in GENRE_KEYWORDS.items():
                if keyword in name:
                    genres.add(genre_label)

        if not genres:
            genres.add("Contemporary Romance")
        return sorted(list(genres))

    def clean_book_record(self, raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate and clean a single book record."""
        self.validation_stats["books"]["input"] += 1
        book_id = str(raw.get("book_id", "")).strip()
        if not book_id:
            self.validation_stats["books"]["failures"] += 1
            return None

        title = self.normalize_text(raw.get("title") or raw.get("title_without_series") or "Untitled")
        title_without_series = self.normalize_text(raw.get("title_without_series") or title)

        # Author extraction
        author_name = raw.get("author_name")
        authors_raw = raw.get("authors", [])
        author_id = ""
        if not author_name and authors_raw and isinstance(authors_raw, list):
            first_auth = authors_raw[0]
            if isinstance(first_auth, dict):
                author_id = str(first_auth.get("author_id", "")).strip()
                author_name = first_auth.get("name") or f"Author {author_id}"
        if not author_name:
            author_name = "Unknown Author"

        # Ratings
        try:
            avg_rating = float(raw.get("average_rating") or 0.0)
            avg_rating = max(0.0, min(5.0, avg_rating))
        except (ValueError, TypeError):
            avg_rating = 0.0

        try:
            ratings_count = int(raw.get("ratings_count") or 0)
        except (ValueError, TypeError):
            ratings_count = 0

        try:
            reviews_count = int(raw.get("text_reviews_count") or 0)
        except (ValueError, TypeError):
            reviews_count = 0

        # Publication year
        pub_year_raw = raw.get("publication_year")
        try:
            pub_year = int(pub_year_raw) if pub_year_raw and str(pub_year_raw).isdigit() else None
        except (ValueError, TypeError):
            pub_year = None
        if pub_year is None or pub_year < 1500 or pub_year > 2030:
            pub_year = None
            self.validation_stats["books"]["missing_year"] += 1

        desc = self.normalize_text(raw.get("description", ""))
        if not desc:
            self.validation_stats["books"]["missing_desc"] += 1

        popular_shelves = raw.get("popular_shelves", [])
        genres = self.extract_genres_from_shelves(popular_shelves)

        similar_books = raw.get("similar_books", [])
        if isinstance(similar_books, list):
            similar_books_clean = [str(x).strip() for x in similar_books if str(x).strip()]
        else:
            similar_books_clean = []

        cleaned = {
            "book_id": book_id,
            "title": title,
            "title_without_series": title_without_series,
            "author_name": str(author_name).strip(),
            "author_id": str(author_id).strip(),
            "average_rating": round(avg_rating, 2),
            "ratings_count": ratings_count,
            "text_reviews_count": reviews_count,
            "publication_year": pub_year,
            "publisher": str(raw.get("publisher") or "").strip(),
            "num_pages": int(raw.get("num_pages")) if str(raw.get("num_pages", "")).isdigit() else None,
            "description": desc,
            "genres": json.dumps(genres),
            "popular_shelves": json.dumps(popular_shelves if isinstance(popular_shelves, list) else []),
            "similar_books": json.dumps(similar_books_clean),
            "url": str(raw.get("url") or raw.get("link") or "").strip(),
            "image_url": str(raw.get("image_url") or "").strip(),
            "bayesian_rating": round(avg_rating, 2),  # shrinkage computed in batch
        }
        self.validation_stats["books"]["output"] += 1
        return cleaned

    def clean_interaction_record(
        self, raw: Dict[str, Any], valid_book_ids: Set[str]
    ) -> Optional[Dict[str, Any]]:
        """Validate and clean a single reader-book interaction record."""
        self.validation_stats["interactions"]["input"] += 1
        user_id = str(raw.get("user_id", "")).strip()
        book_id = str(raw.get("book_id", "")).strip()

        if not user_id or not book_id:
            self.validation_stats["interactions"]["failures"] += 1
            return None

        if book_id not in valid_book_ids:
            self.validation_stats["interactions"]["invalid_book_ref"] += 1
            return None

        try:
            rating = int(raw.get("rating") or 0)
            rating = max(0, min(5, rating))
        except (ValueError, TypeError):
            rating = 0

        is_read = int(raw.get("is_read", 1 if rating > 0 else 0))

        cleaned = {
            "user_id": user_id,
            "book_id": book_id,
            "rating": rating,
            "is_read": is_read,
            "date_added": str(raw.get("date_added") or "").strip(),
            "read_at": str(raw.get("read_at") or "").strip(),
        }
        self.validation_stats["interactions"]["output"] += 1
        return cleaned

    def clean_review_record(
        self, raw: Dict[str, Any], valid_book_ids: Set[str]
    ) -> Optional[Dict[str, Any]]:
        """Validate and clean a single review record."""
        self.validation_stats["reviews"]["input"] += 1
        review_id = str(raw.get("review_id", "")).strip()
        user_id = str(raw.get("user_id", "")).strip()
        book_id = str(raw.get("book_id", "")).strip()

        if not review_id or not user_id or not book_id:
            self.validation_stats["reviews"]["failures"] += 1
            return None

        if book_id not in valid_book_ids:
            self.validation_stats["reviews"]["invalid_book_ref"] += 1
            return None

        review_text = self.normalize_text(raw.get("review_text", ""))
        if not review_text:
            self.validation_stats["reviews"]["empty_text"] += 1
            return None

        try:
            rating = int(raw.get("rating") or 0)
            rating = max(0, min(5, rating))
        except (ValueError, TypeError):
            rating = 0

        cleaned = {
            "review_id": review_id,
            "user_id": user_id,
            "book_id": book_id,
            "rating": rating,
            "review_text": review_text,
            "date_added": str(raw.get("date_added") or "").strip(),
        }
        self.validation_stats["reviews"]["output"] += 1
        return cleaned

    def compute_bayesian_ratings(self, books_df: pd.DataFrame, m_quantile: float = 0.25) -> pd.DataFrame:
        """
        Compute Bayesian weighted rating:
        WR = (v / (v + m)) * R + (m / (v + m)) * C
        """
        if books_df.empty:
            return books_df

        C = books_df["average_rating"].mean()
        # Minimum votes threshold m (e.g. 25th percentile or min 50)
        m = max(50.0, books_df["ratings_count"].quantile(m_quantile)) if len(books_df) > 1 else 10.0

        v = books_df["ratings_count"].astype(float)
        R = books_df["average_rating"].astype(float)

        books_df["bayesian_rating"] = ((v / (v + m)) * R + (m / (v + m)) * C).round(3)
        return books_df

    def process_all(
        self,
        raw_books: Iterable[Dict[str, Any]],
        raw_interactions: Iterable[Dict[str, Any]],
        raw_reviews: Iterable[Dict[str, Any]],
        dataset_mode: str = "development_sample",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Process books, interactions, and reviews and return clean DataFrames."""
        start_time = time.time()
        self.validation_stats["dataset_mode"] = dataset_mode

        # 1. Clean Books
        cleaned_books = []
        seen_books = set()
        for b in raw_books:
            c = self.clean_book_record(b)
            if c:
                if c["book_id"] in seen_books:
                    self.validation_stats["books"]["duplicates"] += 1
                    continue
                seen_books.add(c["book_id"])
                cleaned_books.append(c)

        books_df = pd.DataFrame(cleaned_books)
        if not books_df.empty:
            books_df = self.compute_bayesian_ratings(books_df)

        valid_book_ids = set(books_df["book_id"].unique()) if not books_df.empty else set()

        # 2. Clean Interactions
        cleaned_interactions = []
        seen_interactions = set()
        for i in raw_interactions:
            c = self.clean_interaction_record(i, valid_book_ids)
            if c:
                pair = (c["user_id"], c["book_id"])
                if pair in seen_interactions:
                    self.validation_stats["interactions"]["duplicates"] += 1
                    continue
                seen_interactions.add(pair)
                cleaned_interactions.append(c)

        interactions_df = pd.DataFrame(cleaned_interactions)

        # 3. Clean Reviews
        cleaned_reviews = []
        seen_reviews = set()
        for r in raw_reviews:
            c = self.clean_review_record(r, valid_book_ids)
            if c:
                if c["review_id"] in seen_reviews:
                    self.validation_stats["reviews"]["duplicates"] += 1
                    continue
                seen_reviews.add(c["review_id"])
                cleaned_reviews.append(c)

        reviews_df = pd.DataFrame(cleaned_reviews)

        # Calculate processing duration
        self.validation_stats["processing_time_seconds"] = round(time.time() - start_time, 3)

        # Save to Parquet
        books_df.to_parquet(self.processed_dir / "processed_books.parquet", index=False)
        interactions_df.to_parquet(self.processed_dir / "processed_interactions.parquet", index=False)
        reviews_df.to_parquet(self.processed_dir / "processed_reviews.parquet", index=False)

        # Generate validation report and data dictionary
        self.generate_validation_report()
        self.generate_data_dictionary()

        return books_df, interactions_df, reviews_df

    def generate_validation_report(self) -> Path:
        """Write a comprehensive data validation report markdown document based on actual execution."""
        report_path = settings.reports_dir / "data_validation_report.md"
        stats = self.validation_stats
        mode = stats.get("dataset_mode", "development_sample")
        duration = stats.get("processing_time_seconds", 0.0)

        content = f"""# Romance Reader Intelligence: Data Validation Report

**Generated:** {stats['timestamp']}  
**Execution Environment:** Python 3.13 / UCSD Goodreads Romance Graph Pipeline  
**Dataset Operational Mode:** `{mode}`  
**Processing Duration:** {duration:.3f} seconds  

---

## 1. Provenance & Dataset Classification

- **Operational Mode:** `{mode}`
- **Source Classification:** {"Curated demonstration seed fixture with synthetic reader interaction traces." if mode == "development_sample" else "Full UCSD Goodreads Romance dataset (Wan & McAuley, 2018)."}
- **Integrity Rule:** Seed dataset record counts originate strictly from pipeline execution and must not be conflated with the published 335,449-book UCSD research dataset.

---

## 2. Summary Validation Statistics

| Dataset | Input Records | Cleaned Output | Invalid Records | Duplicates Filtered | Missing Fields | Invalid Foreign Keys |
|---|---|---|---|---|---|---|
| **Books** | {stats['books']['input']} | {stats['books']['output']} | {stats['books']['failures']} | {stats['books']['duplicates']} | Desc: {stats['books']['missing_desc']}, Year: {stats['books']['missing_year']} | N/A |
| **Interactions** | {stats['interactions']['input']} | {stats['interactions']['output']} | {stats['interactions']['failures']} | {stats['interactions']['duplicates']} | 0 | {stats['interactions']['invalid_book_ref']} |
| **Reviews** | {stats['reviews']['input']} | {stats['reviews']['output']} | {stats['reviews']['failures']} | {stats['reviews']['duplicates']} | Empty Text: {stats['reviews']['empty_text']} | {stats['reviews']['invalid_book_ref']} |

---

## 3. Detailed Metric Descriptions

| Metric | Pipeline Description | Actual Value |
|---|---|---|
| **Input Records** | Total records read from raw source files or seed collections | Books: {stats['books']['input']}, Interactions: {stats['interactions']['input']}, Reviews: {stats['reviews']['input']} |
| **Processed Records** | Successfully validated and cleaned records persisted to Parquet/SQLite | Books: {stats['books']['output']}, Interactions: {stats['interactions']['output']}, Reviews: {stats['reviews']['output']} |
| **Invalid Records** | Records rejected due to missing primary identifiers or fatal parse errors | Books: {stats['books']['failures']}, Interactions: {stats['interactions']['failures']}, Reviews: {stats['reviews']['failures']} |
| **Duplicate Records** | Duplicate entities discarded (duplicate `book_id`, `(user_id, book_id)`, or `review_id`) | Books: {stats['books']['duplicates']}, Interactions: {stats['interactions']['duplicates']}, Reviews: {stats['reviews']['duplicates']} |
| **Missing Fields** | Optional attributes missing or defaulted (e.g. publication year, empty review text) | Books (Desc/Year): {stats['books']['missing_desc']}/{stats['books']['missing_year']}, Reviews (Empty): {stats['reviews']['empty_text']} |
| **Invalid Foreign Keys** | Reader interactions or reviews pointing to unindexed `book_id` references | Interactions: {stats['interactions']['invalid_book_ref']}, Reviews: {stats['reviews']['invalid_book_ref']} |
| **Processing Time** | Total wall-clock time elapsed for validation, cleaning, and Parquet serialization | {duration:.3f} seconds |
| **Dataset Mode** | Active execution mode | `{mode}` |

---

## 4. Referential & Structural Integrity Verification

1. **Books Pipeline**:
   - `book_id` must be a non-empty string.
   - Text fields (`title`, `description`) are stripped of HTML tags and normalized.
   - Reader ratings are clamped to the valid domain `[0.0, 5.0]`.
   - Shelf tags are categorized into standard romance subgenres.
   - Bayesian weighted shrinkage ratings are computed dynamically via $WR = \\frac{{v}}{{v+m}}R + \\frac{{m}}{{v+m}}C$.

2. **Interactions Pipeline**:
   - Both `user_id` and `book_id` must be present.
   - Strict foreign key integrity: interaction `book_id` must resolve against the verified book catalogue.
   - Exact `(user_id, book_id)` duplicates are pruned.
   - Star ratings are constrained to integers `[0, 5]`.

3. **Reviews Pipeline**:
   - `review_id` must be unique across all records.
   - Foreign key integrity is strictly enforced against the cleaned book catalogue.
   - Reviews with empty or whitespace-only bodies are discarded.
"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(content)
        return report_path

    def generate_data_dictionary(self) -> Path:
        """Generate official data dictionary markdown document."""
        doc_path = settings.reports_dir / "data_dictionary.md"
        content = """# Romance Reader Intelligence: Data Dictionary

Derived from the UCSD Goodreads Book Graph (Romance Subset) by Mengting Wan & Julian McAuley (RecSys 2018).

## Table: `books`

| Column | Type | Nullable | Description | Source / Provenance |
|---|---|---|---|---|
| `book_id` | TEXT | NO | Unique Goodreads book identifier (Primary Key) | `book_id` |
| `title` | TEXT | NO | Book title with edition/series suffixes cleaned | `title` |
| `title_without_series` | TEXT | NO | Pure book title without series bracket notation | `title_without_series` |
| `author_name` | TEXT | NO | Primary author's full name | Inferred from `authors` / metadata |
| `author_id` | TEXT | YES | Goodreads author identifier | `authors[0].author_id` |
| `average_rating` | REAL | NO | Average reader rating (0.00 to 5.00) | `average_rating` |
| `ratings_count` | INTEGER | NO | Total number of user ratings | `ratings_count` |
| `text_reviews_count` | INTEGER | NO | Total number of written text reviews | `text_reviews_count` |
| `publication_year` | INTEGER | YES | Year of first or primary publication | `publication_year` |
| `publisher` | TEXT | YES | Publishing imprint or house | `publisher` |
| `num_pages` | INTEGER | YES | Page count | `num_pages` |
| `description` | TEXT | YES | Cleaned synopsis text without HTML tags | `description` |
| `genres` | TEXT | NO | JSON array of inferred romance subgenres | Inferred from `popular_shelves` |
| `popular_shelves` | TEXT | YES | JSON array of raw shelf tags and counts | `popular_shelves` |
| `similar_books` | TEXT | YES | JSON array of related book IDs | `similar_books` |
| `url` | TEXT | YES | Goodreads URL | `url` / `link` |
| `image_url` | TEXT | YES | Book cover thumbnail URL | `image_url` |
| `bayesian_rating` | REAL | NO | Bayesian shrinkage weighted rating | Calculated mathematically |

## Table: `interactions`

| Column | Type | Nullable | Description | Source / Provenance |
|---|---|---|---|---|
| `user_id` | TEXT | NO | Anonymized user identifier (Composite PK) | `user_id` |
| `book_id` | TEXT | NO | Goodreads book identifier (Composite PK, FK to `books`) | `book_id` |
| `rating` | INTEGER | NO | Integer rating from 0 (unrated/shelved) to 5 | `rating` |
| `is_read` | INTEGER | NO | Binary flag (1 if read, 0 if shelved only) | `is_read` |
| `date_added` | TEXT | YES | Timestamp when book was added to user shelf | `date_added` |
| `read_at` | TEXT | YES | Timestamp when book was marked finished | `read_at` |

## Table: `reviews`

| Column | Type | Nullable | Description | Source / Provenance |
|---|---|---|---|---|
| `review_id` | TEXT | NO | Unique review identifier (Primary Key) | `review_id` |
| `user_id` | TEXT | NO | Anonymized user identifier | `user_id` |
| `book_id` | TEXT | NO | Goodreads book identifier (FK to `books`) | `book_id` |
| `rating` | INTEGER | NO | Star rating associated with the review | `rating` |
| `review_text` | TEXT | NO | Normalized review text | `review_text` |
| `sentiment_score` | REAL | YES | Compound polarity score [-1.0, 1.0] | NLP Sentiment Engine |
| `sentiment_label` | TEXT | YES | 'positive', 'neutral', 'negative' | NLP Sentiment Engine |
| `date_added` | TEXT | YES | Timestamp of review posting | `date_added` |

## Table: `tropes`

| Column | Type | Nullable | Description | Source / Provenance |
|---|---|---|---|---|
| `book_id` | TEXT | NO | Book ID (Composite PK, FK to `books`) | `book_id` |
| `trope_name` | TEXT | NO | Canonical romance trope name (Composite PK) | Standardized Trope Ontology |
| `confidence` | REAL | NO | Classification confidence score (0.0 - 1.0) | Trope Engine |
| `method` | TEXT | NO | `rule_based_keyword`, `tfidf_matcher`, `hybrid` | Trope Engine |
| `evidence` | TEXT | NO | Excerpt or matched keywords justifying extraction | Trope Engine |

## Table: `review_themes`

| Column | Type | Nullable | Description | Source / Provenance |
|---|---|---|---|---|
| `book_id` | TEXT | NO | Book ID (FK to `books`) | `book_id` |
| `theme_type` | TEXT | NO | 'character', 'pacing', or 'plot' | NLP Theme Extractor |
| `theme_key` | TEXT | NO | Specific thematic aspect (e.g. 'grumpy_hero') | NLP Theme Extractor |
| `score` | REAL | NO | Prominence or polarity score | NLP Theme Extractor |
| `evidence` | TEXT | YES | Contextual excerpt from reader reviews | NLP Theme Extractor |
"""
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(content)
        return doc_path
