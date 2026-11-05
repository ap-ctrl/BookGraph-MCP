"""Interactive demo script allowing direct query testing from the VS Code terminal."""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import create_server

def run_demo():
    print("================================================================================")
    print("ROMANCE READER INTELLIGENCE: INTERACTIVE DEMO (VS CODE TERMINAL)")
    print("================================================================================")
    
    server = create_server()
    
    print("\n[Step 1] Searching for 'The Love Hypothesis' using tool 'search_books'...")
    search_tool = server._tool_manager.get_tool("search_books")
    search_res = search_tool.fn(query="The Love Hypothesis")
    
    if not search_res["results"]:
        print("No books found. Make sure you ran 'python scripts/run_pipeline.py' first.")
        return
        
    ref_book = search_res["results"][0]
    print(f"-> Found: '{ref_book['title']}' by {ref_book['author']} (ID: {ref_book['book_id']})")
    print(f"   Average Rating: {ref_book['average_rating']} | Bayesian Rating: {ref_book['bayesian_rating']}")
    print(f"   Genres: {', '.join(ref_book['genres'])}")

    print("\n[Step 2] Recommending 5 similar books with 'Workplace Romance' & 'Slow Burn' tropes...")
    rec_tool = server._tool_manager.get_tool("recommend_similar_books")
    rec_res = rec_tool.fn(
        book_id=ref_book["book_id"],
        limit=5,
        target_tropes=["Workplace Romance", "Slow Burn"]
    )

    for i, book in enumerate(rec_res["recommendations"], 1):
        print(f"\n{i}. {book['title']} by {book['author']}")
        print(f"   Similarity Score : {book['similarity_score']:.2f}")
        print(f"   Average Rating   : {book['average_rating']} / 5.0")
        print(f"   Verified Tropes  : {', '.join(book.get('tropes', []))}")
        print(f"   Why Selected     : {book['explanation']}")

    print("\n[Step 3] Deep review sentiment & thematic analysis for 'The Love Hypothesis'...")
    rev_tool = server._tool_manager.get_tool("analyse_book_reviews")
    rev_res = rev_tool.fn(book_id=ref_book["book_id"])
    
    print(f"-> Total Reviews Analyzed: {rev_res['total_reviews_analyzed']}")
    print(f"-> Sentiment Polarity    : {rev_res['sentiment_overview']['average_polarity']} (Positive: {rev_res['sentiment_overview']['distribution']['positive']})")
    print("-> Extracted Feedback Themes:")
    for cat, themes in rev_res["thematic_feedback"].items():
        if themes:
            theme_names = [t["theme"] for t in themes]
            print(f"   • {cat.capitalize():10}: {', '.join(theme_names)}")

    print("\n================================================================================")
    print("DEMO COMPLETE! To run as an MCP server in VS Code, run: python server.py")
    print("================================================================================")

if __name__ == "__main__":
    run_demo()
