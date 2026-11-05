# Romance Reader Intelligence: Data Validation Report

Generated: 2026-11-05T08:35:43.483786
Environment: Python 3.13 / UCSD Goodreads Romance Graph Pipeline

## Summary Statistics

| Dataset | Input Records | Cleaned Output | Missing Fields | Duplicates Filtered | Invalid Foreign Keys | Failures |
|---|---|---|---|---|---|---|
| **Books** | 20 | 20 | Desc: 0, Year: 0 | 0 | N/A | 0 |
| **Interactions** | 20 | 20 | 0 | 0 | 0 | 0 |
| **Reviews** | 10 | 10 | Empty Text: 0 | 0 | 0 | 0 |

## Validation Rules Applied

1. **Books Pipeline**:
   - `book_id` must be non-empty string.
   - Text fields (`title`, `description`) stripped of HTML tags and normalized.
   - Ratings clamped to range `[0.0, 5.0]`.
   - Shelf tags mapped to canonical romance subgenres (`Contemporary Romance`, `Historical Romance`, `Fantasy Romance / Romantasy`, etc.).
   - Bayesian weighted shrinkage rating computed via $WR = \frac{v}{v+m}R + \frac{m}{v+m}C$.

2. **Interactions Pipeline**:
   - Both `user_id` and `book_id` validated.
   - Foreign key integrity checked: interaction book IDs must exist in the clean book catalogue.
   - Exact `(user_id, book_id)` duplicates removed.
   - Ratings constrained to valid integer domain `[0, 5]`.

3. **Reviews Pipeline**:
   - Review IDs must be distinct.
   - Foreign key integrity strictly enforced against valid book catalogue.
   - Reviews with empty or whitespace-only bodies discarded.
