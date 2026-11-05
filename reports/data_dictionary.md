# Romance Reader Intelligence: Data Dictionary

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
