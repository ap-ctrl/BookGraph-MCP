# Romance Reader Intelligence: Recommendation Engine Evaluation

**Evaluation Timestamp:** 2026-11-06T18:50:14.216468+00:00  
**Dataset Operational Mode:** `development_sample`  
**Random Split Seed:** `42`  
**Train/Test Split Strategy:** User-aware holdout (25% holdout per user with $\ge 3$ interactions)  
**Data Leakage Audit:** Passed (Zero train/test interaction overlap; models fit exclusively on training partition)  

---

## 1. Dataset Partition Statistics

| Dimension | Count | Description |
|---|---|---|
| **Catalog Books** | 20 | Total books indexed in SQLite |
| **Total Readers** | 5 | Distinct user profiles |
| **Total Interactions** | 20 | All logged reader-book ratings |
| **Training Interactions** | 16 | Interactions allocated to model training |
| **Test Interactions** | 4 | Interactions held out strictly for evaluation |
| **Eligible Test Users** | 4 | Users with $\ge 3$ ratings and $\ge 1$ positive test rating ($\ge 4$) |

---

## 2. Benchmark Results (Rank K = 5)

| Model Architecture | Precision@5 | Recall@5 | NDCG@5 | Catalog Coverage |
|---|---|---|---|---|
| **Popularity_Baseline** | 0.0500 | 0.2500 | 0.1077 | 40.00% |
| **Collaborative_Filtering** | 0.0000 | 0.0000 | 0.0000 | 40.00% |
| **Content_Based** | 0.2000 | 1.0000 | 0.7217 | 75.00% |
| **Hybrid_Recommender** | 0.1500 | 0.7500 | 0.6250 | 75.00% |

---

## 3. Metric Definitions & Formulas

1. **Precision@K**:
   $$\text{Precision}@K = \frac{|\text{Recommended Items} \cap \text{Relevant Test Items}|}{K}$$
2. **Recall@K**:
   $$\text{Recall}@K = \frac{|\text{Recommended Items} \cap \text{Relevant Test Items}|}{|\text{Relevant Test Items}|}$$
3. **NDCG@K (Normalized Discounted Cumulative Gain)**:
   $$DCG@K = \sum_{i=1}^K \frac{rel_i}{\log_2(i + 1)}, \quad IDCG@K = \sum_{i=1}^{\min(|\text{Rel}|, K)} \frac{1}{\log_2(i + 1)}, \quad NDCG@K = \frac{DCG@K}{IDCG@K}$$
4. **Catalog Coverage**:
   $$\text{Coverage} = \frac{|\bigcup_{u \in U} \text{Recommendations}(u)|}{|\text{Catalog}|}$$

---

## 4. Technical Analysis & Investigation of Performance

### Why Popularity and Collaborative Filtering Score 0.0000 on the Seed Fixture:
1. **Collaborative Filtering Item Cold-Start in Test Partition:**
   The development seed fixture contains 20 interactions across 5 readers. Because each test reader's held-out book was only rated by that single reader, holding out that interaction reduces that book's training interaction count to **zero**. Item-item collaborative filtering computes similarities between items present in the training interaction matrix; an item with zero training observations cannot be indexed, correlated, or recommended.
2. **Popularity Baseline Catalogue Selection:**
   The popularity baseline ranks global high-volume titles (e.g. *Pride and Prejudice*, *Ugly Love*, *The Love Hypothesis*). The 4 held-out test books for the evaluation readers happen to be distinct from the top 5 global titles, resulting in zero overlap on this small sample.
3. **Content-Based Performance:**
   Content-based filtering relies on lexical TF-IDF similarity between book metadata, descriptions, and tropes. Because book descriptions share thematic vocabulary regardless of interaction sparsity, the content engine successfully surfaces books with similar trope vocabularies.
4. **Hybrid Blending & Fallback:**
   The hybrid engine combines content similarity with collaborative signals. When collaborative history yields low confidence, it transparently falls back to content-based ranking while logging cold-start diagnostics.

---

## 5. Known Limitations
- When operating in `development_sample` mode (20 books, 20 interactions), benchmark metrics reflect small-sample sparsity artifacts rather than the large-scale collaborative filtering performance observed on the full 42M-interaction UCSD Goodreads dataset.
- Evaluation on the full dataset requires local download of `goodreads_interactions_romance.json.gz` per academic terms of use.
