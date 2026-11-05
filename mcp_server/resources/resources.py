"""MCP read-only resources for Romance Reader Intelligence."""

import json
from mcp.server.mcpserver import MCPServer
from database.repository import BookRepository

def register_resources(server: MCPServer, repo: BookRepository) -> None:
    """Register all 5 read-only MCP resources."""

    @server.resource(
        uri="books://catalog/statistics",
        name="Catalogue Statistics",
        description="Current catalogue counts, rating distributions, and dataset provenance.",
        mime_type="application/json",
    )
    def get_catalog_stats_resource() -> str:
        stats = repo.get_catalog_statistics()
        return json.dumps(stats, indent=2)

    @server.resource(
        uri="books://genres",
        name="Romance Genres",
        description="Catalogue romance subgenres and frequency distribution.",
        mime_type="application/json",
    )
    def get_genres_resource() -> str:
        dist = repo.get_genre_distribution()
        return json.dumps(dist, indent=2)

    @server.resource(
        uri="books://tropes",
        name="Romance Tropes Vocabulary",
        description="Defined romance tropes and catalogue classification frequencies.",
        mime_type="application/json",
    )
    def get_tropes_resource() -> str:
        from nlp.trope_extractor import TROPE_DEFINITIONS
        trope_info = {
            t: {"prototype": d["prototype"], "sample_keywords": d["keywords"][:4]}
            for t, d in TROPE_DEFINITIONS.items()
        }
        return json.dumps(trope_info, indent=2)

    @server.resource(
        uri="books://authors/top-rated",
        name="Top-Rated Authors",
        description="Top-rated romance authors by mean rating and reader volume.",
        mime_type="application/json",
    )
    def get_top_authors_resource() -> str:
        authors = repo.get_top_rated_authors(limit=10)
        return json.dumps(authors, indent=2)

    @server.resource(
        uri="books://analytics/overview",
        name="Analytical Overview",
        description="Precomputed summary of EDA metrics and recommendation benchmark evaluations.",
        mime_type="application/json",
    )
    def get_analytics_overview_resource() -> str:
        stats = repo.get_catalog_statistics()
        genres = repo.get_genre_distribution()
        top_authors = repo.get_top_rated_authors(limit=5)
        overview = {
            "catalogue": stats,
            "top_genres": list(genres.keys())[:5],
            "top_authors": [a["author_name"] for a in top_authors],
            "evaluation_status": "Benchmark evaluation stored in reports/recommendation_evaluation.md",
        }
        return json.dumps(overview, indent=2)
