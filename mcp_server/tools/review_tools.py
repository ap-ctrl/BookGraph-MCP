"""Review and reader analytics MCP tools."""

from typing import Any, Dict
from mcp.server.mcpserver import MCPServer
from services.nlp_service import NLPService

def register_review_tools(server: MCPServer, nlp_service: NLPService) -> None:
    """Register review sentiment and reader insight tools."""

    @server.tool(
        name="analyse_book_reviews",
        description=(
            "Retrieve sentiment polarity distributions and extracted reader feedback across narrative dimensions "
            "(character dynamics, pacing flow, plot construction) for a specific romance book. Strictly reports "
            "empirically processed review data without inventing statistics."
        ),
    )
    def analyse_book_reviews(book_id: str) -> Dict[str, Any]:
        """Analyze reviews for a book."""
        return nlp_service.analyse_book_reviews(book_id=book_id)

    @server.tool(
        name="get_reader_insights",
        description=(
            "Summarise historical reading behavior and shelf patterns for an anonymized dataset reader ID. "
            "Returns reading volume, average given star ratings, and favorite subgenres while preserving privacy."
        ),
    )
    def get_reader_insights(user_id: str) -> Dict[str, Any]:
        """Summarize reader profile."""
        return nlp_service.get_reader_insights(user_id=user_id)
