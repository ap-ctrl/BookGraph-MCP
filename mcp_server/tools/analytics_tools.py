"""Catalogue and subgenre statistical analytics MCP tools."""

from typing import Any, Dict
from mcp.server.mcpserver import MCPServer
from services.book_service import BookService

def register_analytics_tools(server: MCPServer, book_service: BookService) -> None:
    """Register catalogue and genre analytics tools."""

    @server.tool(
        name="get_genre_insights",
        description=(
            "Retrieve statistical breakdown of romance subgenres present across the catalogue "
            "(Contemporary, Historical, Fantasy/Romantasy, Dark, Sports, Queer/LGBTQ+, etc.)."
        ),
    )
    def get_genre_insights() -> Dict[str, Any]:
        """Return genre breakdown and primary subgenres."""
        return book_service.get_genre_insights()

    @server.tool(
        name="get_catalogue_statistics",
        description=(
            "Retrieve catalogue coverage metrics, including total indexed books, distinct authors, "
            "reader interaction counts, review volumes, trope classification counts, and dataset provenance."
        ),
    )
    def get_catalogue_statistics() -> Dict[str, Any]:
        """Return catalogue coverage and freshness statistics."""
        return book_service.get_catalogue_statistics()
