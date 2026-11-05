# Romance Reader Intelligence: Recommendation Engine Evaluation

Benchmark conducted with user-aware train/test split (no cross-user rating leakage).

## Results Summary (Rank K = 5)

| Model Architecture | Precision@5 | Recall@5 | NDCG@5 | Catalog Coverage |
|---|---|---|---|---|
| **Popularity_Baseline** | 0.0000 | 0.0000 | 0.0000 | 25.00% |
| **Collaborative_Filtering** | 0.0000 | 0.0000 | 0.0000 | 45.00% |
| **Content_Based** | 0.1500 | 0.7500 | 0.4231 | 75.00% |
| **Hybrid_Recommender** | 0.1000 | 0.5000 | 0.2544 | 75.00% |

## Methodology Notes

1. **User-Aware Split**: Users with multiple interactions have their latest ratings held out strictly in the test partition.
2. **Cold-Start Handling**: Hybrid model automatically falls back to content-based or popularity scoring when collaborative history is insufficient.
3. **NDCG Metric**: Ranks evaluated with logarithmic positional discount $\log_2(i + 1)$.
4. **Coverage**: Measures the diversity and breadth of catalogue discovery across all evaluation users.
