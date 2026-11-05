"""Tests for MCP Protocol Compliance (Tool Discovery, Resources, Prompts, Host Integration)."""

import pytest
from mcp.server.mcpserver import MCPServer

EXPECTED_TOOLS = [
    "search_books",
    "get_book_details",
    "find_books_by_tropes",
    "compare_books",
    "get_author_insights",
    "recommend_similar_books",
    "get_personalized_recommendations",
    "get_popular_books",
    "analyse_book_reviews",
    "get_reader_insights",
    "get_genre_insights",
    "get_catalogue_statistics",
]

EXPECTED_RESOURCES = [
    "books://catalog/statistics",
    "books://genres",
    "books://tropes",
    "books://authors/top-rated",
    "books://analytics/overview",
]

EXPECTED_PROMPTS = [
    "book_discovery_guide",
    "romance_trope_explorer",
    "book_review_analysis",
    "reading_profile_analysis",
    "compare_romance_books",
]

def test_mcp_tool_discovery(mcp_server: MCPServer):
    """Test 14: Verify MCP tool discovery returns all 12 tools."""
    tools = mcp_server._tool_manager.list_tools()
    tool_names = [t.name for t in tools]
    assert len(tool_names) == 12
    for exp in EXPECTED_TOOLS:
        assert exp in tool_names

def test_mcp_resource_access(mcp_server: MCPServer):
    """Test 15: Verify MCP resource discovery and execution."""
    resources = mcp_server._resource_manager.list_resources()
    resource_uris = [r.uri for r in resources]
    assert len(resource_uris) == 5
    for exp_uri in EXPECTED_RESOURCES:
        assert exp_uri in resource_uris

    # Verify resource content can be read via registered function
    target_res = next((r for r in resources if r.uri == "books://catalog/statistics"), None)
    assert target_res is not None
    content = target_res.fn()
    assert "total_books" in content

def test_mcp_prompt_retrieval(mcp_server: MCPServer):
    """Test 16: Verify MCP prompt discovery and invocation."""
    prompts = mcp_server._prompt_manager.list_prompts()
    prompt_names = [p.name for p in prompts]
    assert len(prompt_names) == 5
    for exp_prompt in EXPECTED_PROMPTS:
        assert exp_prompt in prompt_names

    # Test prompt invocation
    p_obj = mcp_server._prompt_manager.get_prompt("book_discovery_guide")
    assert p_obj is not None
    prompt_text = p_obj.fn(favorite_book="Beach Read", preferred_trope="Enemies to Lovers")
    assert "Beach Read" in prompt_text
    assert "Enemies to Lovers" in prompt_text

def test_mcp_host_integration(mcp_server: MCPServer):
    """Test 17: Verify MCP host integration: tool invocation via tool manager."""
    tool = mcp_server._tool_manager.get_tool("get_catalogue_statistics")
    assert tool is not None
    stats = tool.fn()
    assert stats["total_books"] >= 10
    assert "dataset_provenance" in stats
