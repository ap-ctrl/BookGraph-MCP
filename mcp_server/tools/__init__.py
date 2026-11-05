"""Tool registration orchestrator for Romance Reader Intelligence."""

from mcp.server.mcpserver import MCPServer
from mcp_server.tools.analytics_tools import register_analytics_tools
from mcp_server.tools.book_tools import register_book_tools
from mcp_server.tools.recommendation_tools import register_recommendation_tools
from mcp_server.tools.review_tools import register_review_tools
from services.book_service import BookService
from services.nlp_service import NLPService
from services.recommendation_service import RecommendationService

def register_all_tools(
    server: MCPServer,
    book_service: BookService,
    rec_service: RecommendationService,
    nlp_service: NLPService,
) -> None:
    """Register all 12 MCP tools."""
    register_book_tools(server, book_service, nlp_service)
    register_recommendation_tools(server, rec_service)
    register_review_tools(server, nlp_service)
    register_analytics_tools(server, book_service)
