"""Book discovery, inspection, trope filtering, and comparison MCP tools."""

from typing import Any, Dict, List, Optional
from mcp.server.mcpserver import MCPServer
from services.book_service import BookService
from services.nlp_service import NLPService

def register_book_tools(server: MCPServer, book_service: BookService, nlp_service: NLPService) -> None:
    """Register core book search and comparison tools."""

    @server.tool(
        name="search_books",
        description=(
            "Search for romance books by title keyword, author name, subgenre, minimum star rating, "
            "or publication year. Supports result pagination and limits."
        ),
    )
    def search_books(
        query: Optional[str] = None,
        author: Optional[str] = None,
        genre: Optional[str] = None,
        publication_year: Optional[int] = None,
        min_rating: Optional[float] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Search books with optional filters and pagination."""
        return book_service.search_books(
            query=query,
            author=author,
            genre=genre,
            publication_year=publication_year,
            min_rating=min_rating,
            limit=limit,
            offset=offset,
        )

    @server.tool(
        name="get_book_details",
        description=(
            "Retrieve comprehensive metadata, synopsis, confirmed romance tropes, and reader review "
            "sentiment summary for a specific Goodreads book_id."
        ),
    )
    def get_book_details(book_id: str) -> Dict[str, Any]:
        """Retrieve full details for a book by book_id."""
        details = book_service.get_book_details(book_id)
        if not details:
            return {"error": f"Book with ID '{book_id}' not found in the catalogue.", "found": False}
        return {"found": True, "book": details}

    @server.tool(
        name="find_books_by_tropes",
        description=(
            "Find romance books matching one or more specific romance tropes "
            "(e.g., 'Workplace Romance', 'Slow Burn', 'Enemies to Lovers', 'Fake Dating', 'Marriage of Convenience'). "
            "Returns classification confidence, detection method, and verbatim evidence snippets."
        ),
    )
    def find_books_by_tropes(
        tropes: List[str],
        match_all: bool = False,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Search books matching specified trope labels."""
        return nlp_service.find_books_by_tropes(tropes=tropes, match_all=match_all, limit=limit)

    @server.tool(
        name="compare_books",
        description=(
            "Compare two romance book records side-by-side using reader rating statistics, "
            "page counts, shared subgenres, and overlapping vs. unique romance tropes."
        ),
    )
    def compare_books(book_id_1: str, book_id_2: str) -> Dict[str, Any]:
        """Compare two book records."""
        return book_service.compare_books(book_id_1=book_id_1, book_id_2=book_id_2)

    @server.tool(
        name="get_author_insights",
        description=(
            "Retrieve aggregated author statistics, catalogue bibliography, and mean reader ratings "
            "for a romance author."
        ),
    )
    def get_author_insights(author_name: str) -> Dict[str, Any]:
        """Retrieve author bibliography and reader metrics."""
        return book_service.get_author_insights(author_name=author_name)
