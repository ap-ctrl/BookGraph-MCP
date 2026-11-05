"""Romance Reader Intelligence MCP Server Entrypoint.

Built with Python 3.13 and the official MCP Python SDK v2.
Exposes Tools, Resources, and Prompts over STDIO or Streamable HTTP.
"""

import argparse
import sys
from typing import Optional

from mcp.server.mcpserver import MCPServer

from config.settings import logger, settings
from database.connection import init_db
from database.repository import BookRepository
from mcp_server.prompts import register_prompts
from mcp_server.resources import register_resources
from mcp_server.tools import register_all_tools
from services.book_service import BookService
from services.nlp_service import NLPService
from services.recommendation_service import RecommendationService

def create_server(db_path: Optional[str] = None) -> MCPServer:
    """Create and configure Romance Reader Intelligence MCPServer instance."""
    logger.info("Initializing Romance Reader Intelligence MCP Server (SDK v2)...")

    # Ensure directories and schema exist
    settings.ensure_directories()
    init_db(settings.db_path)

    # Initialize repository and domain services
    repo = BookRepository(settings.db_path)
    book_service = BookService(repo)
    rec_service = RecommendationService(repo)
    nlp_service = NLPService(repo)

    # Instantiate MCPServer
    server = MCPServer(
        name=settings.server_name,
        instructions=(
            "Romance Reader Intelligence: An MCP-powered book discovery, recommendation, "
            "trope intelligence, and reader review analytics server."
        ),
    )

    # Register MCP capabilities
    register_all_tools(server, book_service, rec_service, nlp_service)
    register_resources(server, repo)
    register_prompts(server)

    logger.info(
        "MCPServer '%s' configured with %d tools, %d resources, %d prompts.",
        settings.server_name,
        len(server._tool_manager.list_tools()),
        len(server._resource_manager.list_resources()),
        len(server._prompt_manager.list_prompts()),
    )

    return server

def main() -> None:
    """Run MCP server according to configured transport."""
    parser = argparse.ArgumentParser(description="Romance Reader Intelligence MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default=settings.transport,
        help="MCP transport protocol (default: stdio)",
    )
    parser.add_argument("--port", type=int, default=settings.port, help="Port for HTTP transport")
    parser.add_argument("--host", default=settings.host, help="Host for HTTP transport")
    args = parser.parse_args()

    server = create_server()

    logger.info("Starting server on transport: %s", args.transport)
    if args.transport == "stdio":
        server.run(transport="stdio")
    elif args.transport == "streamable-http":
        server.run(transport="streamable-http", host=args.host, port=args.port)
    elif args.transport == "sse":
        server.run(transport="sse", host=args.host, port=args.port)
    else:
        server.run(transport="stdio")

if __name__ == "__main__":
    main()
