"""End-to-End Test for the Definition of Completion Scenario (Section 11).

Scenario:
User query to AI Host:
"Recommend five romance books similar to The Love Hypothesis that contain workplace romance
and slow-burn themes. Explain why each book was selected."
"""

import pytest
from mcp.server.mcpserver import MCPServer
from database.repository import BookRepository

def test_complete_user_query_scenario(mcp_server: MCPServer, repo: BookRepository):
    """
    Execute full end-to-end user query scenario through the MCP tool layer.
    Verifies that:
    1. Reference book 'The Love Hypothesis' is searched and found via search_books tool.
    2. Candidate books similar to the reference book are retrieved via recommend_similar_books.
    3. Similarity and trope filtering (Workplace Romance & Slow Burn) are applied.
    4. Exact metadata matches database records (no hallucinated books).
    5. Explanations cite verified backend features and evidence.
    """
    # Step 1: Identify reference book via search_books tool
    search_tool = mcp_server._tool_manager.get_tool("search_books")
    search_res = search_tool.fn(query="The Love Hypothesis")
    assert search_res["pagination"]["total_matches"] >= 1
    ref_book_record = search_res["results"][0]
    ref_book_id = ref_book_record["book_id"]
    assert ref_book_record["title"] == "The Love Hypothesis"

    # Step 2: Retrieve similar books matching workplace romance and slow burn
    rec_tool = mcp_server._tool_manager.get_tool("recommend_similar_books")
    rec_res = rec_tool.fn(
        book_id=ref_book_id,
        limit=5,
        target_tropes=["Workplace Romance", "Slow Burn"],
    )

    assert "recommendations" in rec_res
    recs = rec_res["recommendations"]
    assert len(recs) >= 1
    assert len(recs) <= 5

    # Step 3: Verify each recommended book exists in database and has legitimate rationale
    for book in recs:
        cand_id = book["book_id"]
        db_book = repo.get_book_by_id(cand_id)
        assert db_book is not None, f"Recommended book {cand_id} does not exist in database!"
        assert db_book["title"] == book["title"]
        assert db_book["author_name"] == book["author"]

        # Verify similarity score is positive
        assert book["similarity_score"] > 0.0

        # Verify tropes include at least one of requested tropes
        cand_tropes = book.get("tropes", [])
        has_workplace = any("workplace" in t.lower() for t in cand_tropes)
        has_slowburn = any("slow" in t.lower() for t in cand_tropes)
        assert has_workplace or has_slowburn, f"Book {book['title']} lacks requested tropes!"

        # Verify explanation is present and cites backend similarity
        assert "explanation" in book
        assert len(book["explanation"]) > 20
        assert "similarity" in book["explanation"].lower() or "tropes" in book["explanation"].lower()

    # Step 4: Verify detail retrieval and review analysis for the top recommendation
    top_rec = recs[0]
    detail_tool = mcp_server._tool_manager.get_tool("get_book_details")
    details = detail_tool.fn(book_id=top_rec["book_id"])
    assert details["found"] is True
    assert details["book"]["title"] == top_rec["title"]

    # Step 5: Verify trope evidence is genuine and verifiable in database
    trope_tool = mcp_server._tool_manager.get_tool("find_books_by_tropes")
    trope_res = trope_tool.fn(tropes=["Workplace Romance", "Slow Burn"])
    assert trope_res["total_matches"] >= 1
    for match in trope_res["books"]:
        for t_info in match["matched_tropes_with_evidence"]:
            assert "evidence" in t_info
            assert t_info["evidence"] != ""
            assert t_info["confidence"] > 0.0
