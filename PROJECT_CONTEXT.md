# Romance Reader Intelligence: Project Context

## Executive Summary
**Romance Reader Intelligence** is an enterprise-grade, modular Model Context Protocol (MCP) server built with Python 3.13 and the official MCP Python SDK v2 (`mcp.server.mcpserver.MCPServer`). The system provides AI assistants (such as Claude Desktop, Cursor, Antigravity, and other MCP-compliant hosts) with deep analytical and discovery capabilities over large-scale romance literature datasets.

Rather than being a conventional web app with MCP added as a shim, this project is designed from the protocol layer up to expose book discovery, trope intelligence, content-collaborative hybrid recommendations, reader interaction analytics, and sentiment-thematic review mining.

## Architecture & Data Flow
```
+-------------------------------------------------------------------+
|                        MCP Host (AI Client)                       |
+-------------------------------------------------------------------+
                                  |
                                  | STDIO / Streamable HTTP (JSON-RPC)
                                  v
+-------------------------------------------------------------------+
|              MCP Server (mcp.server.mcpserver.MCPServer)           |
|                                                                   |
|   Tools (12)            Resources (5)             Prompts (5)     |
|   - search_books        - catalog/stats           - discovery     |
|   - get_book_details    - genres                  - trope_explorer|
|   - find_by_tropes      - tropes                  - review_analys.|
|   - recommend_similar   - authors/top-rated       - reader_profile|
|   - get_personalized    - analytics/overview      - compare_books |
|   - review_analysis                                               |
|   - analytics & compare                                           |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                          Service Layer                            |
|       (BookService, RecommendationService, NLPService, etc.)      |
+-------------------------------------------------------------------+
         |                        |                       |
         v                        v                       v
+------------------+    +-------------------+   +--------------------+
| SQLite Database  |    | Recommendation    |   | NLP & Trope        |
| (Indexed Books,  |    | Engines (Popular, |   | Engine (Trope regex|
| Interactions,    |    | TF-IDF Content,   |   | & embeddings, VADER|
| Precomputed NLP) |    | Sparse Item-CF)   |   | /Lexicon Sentiment)|
+------------------+    +-------------------+   +--------------------+
         ^                        ^                       ^
         |                        |                       |
+-------------------------------------------------------------------+
|               Data Pipeline (Ingestion & Preprocessing)           |
| - UCSD Goodreads Romance Graph (Books, Interactions, Reviews)     |
| - Streaming parser, Parquet intermediates, SQLite Loader          |
| - Validation Reports & Data Dictionaries                          |
+-------------------------------------------------------------------+
```

## Primary Datasets & Sources
1. **UCSD Goodreads Book Graph (Romance Subset)**:
   - `goodreads_books_romance.json.gz` (335,449 book records)
   - `goodreads_interactions_romance.json.gz` (42,792,856 reader-book interactions)
   - `goodreads_reviews_romance.json.gz` (3,565,378 reviews)
   - Researchers: Mengting Wan, Julian McAuley (UCSD)
   - Strictly academic use; data is not committed to public VCS; samples used for reproducible testing and development.
2. **Open Library API**:
   - Secondary enrichment for missing author/work keys and metadata verification with respectful User-Agent and caching.

## Key Design Principles
1. **Protocol-Native**: Strictly follows MCP Python SDK v2 using `MCPServer`, registering tools via `@server.tool()`, resources via `@server.resource()`, and prompts via `@server.prompt()`.
2. **Zero-Hallucination & Provenance**: Recommendations, trope extractions, and review sentiments provide explicit confidence, mathematical similarity, and provenance tags (e.g., `rule_based_keyword`, `tfidf_matcher`, `bayesian_weighted`).
3. **Resource & Memory Discipline**: Never loads 42M interactions into dense memory. Uses chunked generators, sparse CSR matrices for collaborative filtering, and indexed SQLite stores.
4. **Stderr Logging**: Adheres to STDIO transport hygiene — `sys.stderr` is used for all logging to avoid corrupting the JSON-RPC communication stream on `stdout`.
5. **Cold-Start Resilience**: Hybrid recommendation model explicitly checks reader history, falling back gracefully to content-based or popularity scoring with diagnostic explanations.

## Verification & Status
- **Test Suite**: 21/21 tests passing in 0.77s (`pytest -v`)
- **Inspector Test**: 100% passed over STDIO JSON-RPC 2.0 with zero stdout corruption (`python scripts/test_inspector.py`)
- **Completion Scenario**: Verified in `tests/test_e2e_scenario.py`
