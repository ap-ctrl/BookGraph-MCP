# Technical Audit Report: Romance Reader Intelligence (BookGraph-MCP)

**Audit Date:** 2026-11-05  
**Auditor:** Antigravity AI Engineering Team  
**Repository Branch:** `audit-fix`  
**Execution Environment:** Windows, Python 3.13.5, `mcp==2.3.0`, `scikit-learn==1.3.0+`, `pandas==2.0.0+`

---

## 1. Executive Summary

This engineering audit examines the BookGraph-MCP codebase to verify data provenance, ingestion pipelines, database integrity, recommendation model evaluation (specifically auditing for data leakage and mathematical metric flaws), NLP trope/sentiment engines, MCP server capabilities, automated test coverage, and documentation accuracy.

The audit revealed that while the core modular architecture and MCP protocol integration are structurally clean and functional (21/21 pytest tests currently pass on sample fixtures), there are significant technical discrepancies between documentation claims and executable reality, including:
1. **Dataset Provenance & Scale:** The repository claims processing of the full UCSD Goodreads Romance dataset (335,449 books, 42.8M interactions, 3.56M reviews), but the local repository contains only a 20-book, 20-interaction, 10-review curated seed fixture. Furthermore, several seed books (e.g., *The Love Hypothesis* [2021], *Book Lovers* [2022]) postdate the 2017/2018 UCSD crawl and are modern curated/synthetic records.
2. **Evaluation Metric Defect & Data Leakage:** In `recommendation/evaluator.py`, NDCG@K calculation incorrectly uses the observed hit list sorted descending as the ideal DCG (IDCG), artificially inflating NDCG whenever any hit occurs. Additionally, the `HybridRecommender` queries the live SQLite database directly for user interaction history during evaluation rather than using isolated training interactions, introducing test-set leakage.
3. **Mathematical Cause of Zero Performance:** The 0.0000 Precision/Recall reported for Popularity and Collaborative Filtering was traced to the sparsity and disjoint nature of the 20-record seed dataset: each held-out test book has zero interactions in the training set (item cold-start in test), rendering collaborative filtering unable to index or recommend it, while popularity recommends global high-volume titles not present in the small test partition.
4. **SQLite Cascading Delete Hazard:** `insert_or_replace_books` uses `INSERT OR REPLACE INTO books`. In SQLite with `PRAGMA foreign_keys = ON`, `REPLACE` executes a `DELETE` followed by an `INSERT`, which cascades and deletes all child records in `interactions`, `reviews`, `tropes`, and `review_themes`.
5. **NLP Evidence Traceability:** In `nlp/trope_extractor.py`, TF-IDF matching assigns arbitrary first-sentence excerpts as evidence rather than tracing actual thematic keywords or sentence-level cosine relevance.

---

## 2. Existing Functionality vs. Verified Functionality

| Subsystem / Module | Claimed Functionality | Actual Verified Functionality | Status / Verdict |
|---|---|---|---|
| **Data Ingestion** (`ingestion/`) | Streaming ingestion of 335k books and 42.8M interactions from UCSD Romance subset. | Streams gzipped JSON lines; creates 20-record sample dataset. Full UCSD raw files are not present locally. | **Functional for development sample only; lacks full raw data.** |
| **Data Preprocessing** (`preprocessing/`) | Validation, cleaning, HTML removal, Bayesian rating shrinkage, Parquet intermediate exports. | Cleans records, clamps ratings [0, 5], calculates Bayesian weighted ratings, outputs Parquet and validation report. | **Verified working as designed.** |
| **Database Layer** (`database/`) | SQLite repository with parameterized queries, indexes, and foreign key constraints. | Schema and queries work; connection handles WAL mode and row factory. However, `INSERT OR REPLACE` creates cascading delete hazard. | **Verified working, but requires upsert fix (`ON CONFLICT DO UPDATE`).** |
| **Popularity Recommender** (`recommendation/popularity.py`) | Bayesian shrinkage ranking with minimum vote thresholds. | Queries precomputed `bayesian_rating` from SQLite. Does not compute shrinkage dynamically from train splits. | **Verified working; relies on static table column.** |
| **Content-Based Recommender** (`recommendation/content_based.py`) | TF-IDF and cosine similarity on combined book metadata and descriptions. | TF-IDF vectorization and cosine similarity work reliably. Fallback uses hardcoded 0.85 similarity score. | **Verified working; lexical matching.** |
| **Collaborative Filtering** (`recommendation/collaborative.py`) | Item-Item CF using sparse SciPy CSR matrices and cosine similarity. | Sparse matrix construction and cosine similarity work. Fails on cold-start items with 0 training interactions. | **Verified working; zero score explained by fixture sparsity.** |
| **Hybrid Recommender** (`recommendation/hybrid.py`) | Combines collaborative and content scores with fallback mechanisms. | Combines scores; transparent cold-start fallback. Direct DB queries during evaluation cause leakage. | **Verified working; leakage fix required.** |
| **Recommendation Evaluation** (`recommendation/evaluator.py`) | User-aware train/test split without leakage; Precision@K, Recall@K, NDCG@K, Coverage. | Creates user split, but NDCG calculation is mathematically flawed; hybrid recommender leaks test data through DB. | **Defective; requires immediate correction.** |
| **Trope Extraction** (`nlp/trope_extractor.py`) | 10 romance tropes with rule-based regex, TF-IDF prototypes, and confidence scores. | Rule-based regex works well. TF-IDF evidence defaults to first sentence regardless of trope relevance. | **Functional; evidence extraction needs improvement.** |
| **Sentiment Analysis** (`nlp/sentiment_analyzer.py`) | Domain-specific lexicon with boosters and negations. | Lexicon scoring with tanh normalization works predictably on review text. Not an ML/LLM model. | **Verified working; accurately described as lexicon-based.** |
| **Review Theme Extractor** (`nlp/theme_extractor.py`) | Extracts pacing, character, and plot themes with text evidence. | Regex pattern matching across 3 dimensions extracts quotes and frequencies. | **Verified working.** |
| **MCP Server** (`server.py`, `mcp_server/`) | 12 Tools, 5 Resources, 5 Prompts over STDIO/Streamable HTTP using official MCP Python SDK v2. | All 12 tools, 5 resources, and 5 prompts are properly registered on `MCPServer`. Tested over STDIO. | **Verified working.** |

---

## 3. Issues Identified (By Priority)

### Critical Defects (Priority 1)

1. **NDCG@K Metric Implementation Error (`recommendation/evaluator.py`):**
   - **Current code:** `ideal_hits = sorted(hits, reverse=True); idcg = self.compute_dcg(ideal_hits, k)`.
   - **Flaw:** If a model returns 1 hit at rank 3 out of 5 relevant test items, `hits = [0, 0, 1, 0, 0]`. `ideal_hits` becomes `[1, 0, 0, 0, 0]`. Then `dcg / idcg` is calculated against a single hit rather than against the true ideal ranking of all relevant items (`min(n_relevant, k)` ones). This produces invalid NDCG scores.
   - **Fix:** Construct true ideal vector: `ideal_relevance = [1] * min(len(relevant_books), k) + [0] * max(0, k - len(relevant_books))`.

2. **Evaluation Data Leakage in Hybrid Recommender (`recommendation/hybrid.py`, `evaluator.py`):**
   - **Flaw:** In `run_benchmark()`, `hybrid.recommend_for_reader(user_id=u)` calls `self.repo.get_interactions_for_user(u)`. This pulls interactions directly from the database where both train and test interactions reside. The recommender selects the top-rated book from the user's *entire* history (including held-out test ratings) to seed content recommendations.
   - **Fix:** Pass training interactions explicitly or inject a training-only interaction store into the recommender during evaluation.

3. **Database Cascading Delete Risk (`database/repository.py`):**
   - **Flaw:** `insert_or_replace_books` executes `INSERT OR REPLACE INTO books`. In SQLite, `REPLACE` is implemented as an internal `DELETE` followed by `INSERT`. Because `interactions`, `reviews`, `tropes`, and `review_themes` define `FOREIGN KEY (book_id) REFERENCES books(book_id) ON DELETE CASCADE`, re-running ingestion or upserting books silently wipes all related interactions, reviews, and tropes!
   - **Fix:** Replace with standard SQLite upsert syntax: `INSERT INTO books (...) VALUES (...) ON CONFLICT(book_id) DO UPDATE SET ...`.

### High-Priority Defects (Priority 2)

4. **Misrepresented Dataset Provenance & Scale:**
   - **Flaw:** Documentation repeatedly suggests the application is actively running on 335,449 books and 42.8M interactions. In reality, `data/raw/` only contains 20 books, 20 interactions, and 10 reviews. Books like *The Love Hypothesis* (2021) were not part of the 2017 UCSD crawl.
   - **Fix:** Transparently partition data into:
     - Curated Modern Demonstration Fixture (20 books, 20 interactions, 10 reviews).
     - Full UCSD Goodreads Romance Dataset (external download, streamed when present).
   - Update `dataset_manager.py`, `seed_data.py`, and `repository.py` to tag provenance accurately.

5. **Investigation & Explanation of 0.0000 Benchmark Results:**
   - **Flaw:** Previous report simply printed 0.0000 for Popularity and Collaborative filtering without technical investigation.
   - **Finding:**
     - Collaborative: The 4 held-out test items have 0 interactions in the training split. Item-item CF cannot calculate similarities for items absent from training.
     - Popularity: Baseline recommends top 5 global books, which happen not to overlap with the 4 specific held-out items.
   - **Fix:** Explicitly document this sparsity artifact in `recommendation_evaluation.md` and ensure the evaluator reports evaluation conditions, random seeds, and eligibility thresholds.

6. **Trope Extraction Evidence Fabrication in Pure TF-IDF Matches (`nlp/trope_extractor.py`):**
   - **Flaw:** When TF-IDF similarity exceeds the threshold without a regex match, the extractor attaches `sentences[0]` (the book's opening sentence) as evidence.
   - **Fix:** Compute sentence-level TF-IDF similarity against the trope prototype and extract the sentence with the highest cosine score as genuine evidence.

### Medium-Priority Improvements (Priority 3)

7. **Data Validation Reporting Automation:**
   - `reports/data_validation_report.md` was previously static. It should be dynamically generated with exact runtime metrics (input records, processed records, invalid records, duplicates, foreign key drops, execution duration, and dataset mode).

8. **Test Suite Depth:**
   - Existing 21 tests cover high-level MCP tool invocations but lack unit tests for metric math (Precision@K, Recall@K, NDCG@K edge cases), duplicate handling, foreign key enforcement, and empty database behaviors.

9. **Documentation Alignment:**
   - Remove unsupported claims such as "enterprise-grade", "production-ready", and "zero-hallucination" in `README.md`, `PROJECT_CONTEXT.md`, and `PROJECT_STATUS.md`.

---

## 4. Files Requiring Modification

| File | Proposed Modifications |
|---|---|
| `ingestion/seed_data.py` | Add provenance classification (curated demonstration vs UCSD crawl); clarify synthetic interactions and reviews. |
| `ingestion/dataset_manager.py` | Distinguish between development fixture mode and full UCSD mode; improve error messaging and streaming safety. |
| `database/schema.sql` | Retain schema; verify indexing and foreign key constraints. |
| `database/repository.py` | Fix `INSERT OR REPLACE` to `ON CONFLICT DO UPDATE` to prevent cascading deletion; support training interaction queries. |
| `recommendation/evaluator.py` | Fix NDCG@K ideal vector math; eliminate DB leakage; record eligible users, seed, and data mode. |
| `recommendation/hybrid.py` | Allow supplying user history directly instead of querying full DB during evaluation. |
| `nlp/trope_extractor.py` | Improve TF-IDF evidence sentence selection using sentence-level cosine scoring; tighten confidence scoring. |
| `reports/data_validation_report.md` | Regenerate with actual runtime execution metrics. |
| `reports/recommendation_evaluation.md` | Regenerate with leakage-free evaluation and mathematical explanation of results. |
| `tests/` | Add unit tests for evaluation metrics, database constraints, trope evidence, and cold-start edge cases. |
| `README.md`, `PROJECT_STATUS.md`, `PROJECT_CONTEXT.md` | Align claims with executable facts, document exact dataset status, and describe system honestly. |

---

## 5. Missing Dependencies and Environment Requirements

- **Python Version:** Python 3.13.5 (Compatible).
- **Core Dependencies:** `mcp>=2.3.0`, `pandas>=2.0.0`, `numpy>=1.24.0`, `scipy>=1.10.0`, `scikit-learn>=1.3.0`, `pyarrow>=12.0.0`. All installed and operational.
- **External Dataset:** The full UCSD Romance dataset (`goodreads_books_romance.json.gz`, `goodreads_interactions_romance.json.gz`, `goodreads_reviews_romance.json.gz`) is **not present** locally due to licensing and size restrictions (academic research distribution). The project must operate cleanly in labelled development mode and document external acquisition steps.

---

## 6. Dataset Availability & Current Processing Scale

- **Current Local Dataset:** Curated demonstration fixture.
  - Books: 20
  - Reader Interactions: 20 (across 5 simulated readers)
  - Reviews: 10
- **Source Dataset (UCSD Romance Subset):**
  - Published scale: 335,449 books, 42,792,856 interactions, 3,565,378 reviews.
  - Availability: Not downloaded; download instructions are documented in `ingestion/dataset_manager.py`.

---

## 7. Recommended Fixes in Priority Order

1. **Phase 1:** Update `ingestion/seed_data.py` and `dataset_manager.py` to establish verified provenance and streaming validation.
2. **Phase 2:** Fix `database/repository.py` upsert query to eliminate SQLite cascading deletion risk; add database integrity tests.
3. **Phase 3 & 4:** Fix `evaluator.py` NDCG@K formula and eliminate test-set leakage in `hybrid.py`. Re-run benchmark with documented random seed and explain metrics.
4. **Phase 5:** Upgrade `nlp/trope_extractor.py` sentence-level evidence extraction; add unit tests for sentiment and theme edge cases.
5. **Phase 6 & 7:** Validate MCP tools, resources, prompts, and transport; extend test suite to cover all newly identified edge cases.
6. **Phase 8 & 9:** Update documentation (`README.md`, `PROJECT_STATUS.md`) and compile final technical validation report.
