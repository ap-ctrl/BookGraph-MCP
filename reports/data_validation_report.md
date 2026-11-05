# Romance Reader Intelligence: Data Validation Report

**Generated:** 2026-11-05T10:24:05.178343+00:00  
**Execution Environment:** Python 3.13 / UCSD Goodreads Romance Graph Pipeline  
**Dataset Operational Mode:** `development_sample`  
**Processing Duration:** 0.018 seconds  

---

## 1. Provenance & Dataset Classification

- **Operational Mode:** `development_sample`
- **Source Classification:** Curated demonstration seed fixture with synthetic reader interaction traces.
- **Integrity Rule:** Seed dataset record counts originate strictly from pipeline execution and must not be conflated with the published 335,449-book UCSD research dataset.

---

## 2. Summary Validation Statistics

| Dataset | Input Records | Cleaned Output | Invalid Records | Duplicates Filtered | Missing Fields | Invalid Foreign Keys |
|---|---|---|---|---|---|---|
| **Books** | 20 | 20 | 0 | 0 | Desc: 0, Year: 0 | N/A |
| **Interactions** | 20 | 20 | 0 | 0 | 0 | 0 |
| **Reviews** | 10 | 10 | 0 | 0 | Empty Text: 0 | 0 |

---

## 3. Detailed Metric Descriptions

| Metric | Pipeline Description | Actual Value |
|---|---|---|
| **Input Records** | Total records read from raw source files or seed collections | Books: 20, Interactions: 20, Reviews: 10 |
| **Processed Records** | Successfully validated and cleaned records persisted to Parquet/SQLite | Books: 20, Interactions: 20, Reviews: 10 |
| **Invalid Records** | Records rejected due to missing primary identifiers or fatal parse errors | Books: 0, Interactions: 0, Reviews: 0 |
| **Duplicate Records** | Duplicate entities discarded (duplicate `book_id`, `(user_id, book_id)`, or `review_id`) | Books: 0, Interactions: 0, Reviews: 0 |
| **Missing Fields** | Optional attributes missing or defaulted (e.g. publication year, empty review text) | Books (Desc/Year): 0/0, Reviews (Empty): 0 |
| **Invalid Foreign Keys** | Reader interactions or reviews pointing to unindexed `book_id` references | Interactions: 0, Reviews: 0 |
| **Processing Time** | Total wall-clock time elapsed for validation, cleaning, and Parquet serialization | 0.018 seconds |
| **Dataset Mode** | Active execution mode | `development_sample` |

---

## 4. Referential & Structural Integrity Verification

1. **Books Pipeline**:
   - `book_id` must be a non-empty string.
   - Text fields (`title`, `description`) are stripped of HTML tags and normalized.
   - Reader ratings are clamped to the valid domain `[0.0, 5.0]`.
   - Shelf tags are categorized into standard romance subgenres.
   - Bayesian weighted shrinkage ratings are computed dynamically via $WR = \frac{v}{v+m}R + \frac{m}{v+m}C$.

2. **Interactions Pipeline**:
   - Both `user_id` and `book_id` must be present.
   - Strict foreign key integrity: interaction `book_id` must resolve against the verified book catalogue.
   - Exact `(user_id, book_id)` duplicates are pruned.
   - Star ratings are constrained to integers `[0, 5]`.

3. **Reviews Pipeline**:
   - `review_id` must be unique across all records.
   - Foreign key integrity is strictly enforced against the cleaned book catalogue.
   - Reviews with empty or whitespace-only bodies are discarded.
