"""MCP Prompts guiding AI host interaction and evidence-based analysis."""

from mcp.server.mcpserver import MCPServer

def register_prompts(server: MCPServer) -> None:
    """Register all 5 MCP prompts."""

    @server.prompt(
        name="book_discovery_guide",
        description="Guide an AI assistant through exploring romance books based on reader taste, tropes, and ratings.",
    )
    def book_discovery_guide(
        favorite_book: str = "The Love Hypothesis",
        preferred_trope: str = "Workplace Romance",
    ) -> str:
        return f"""You are the Romance Reader Intelligence Advisor.
The reader has mentioned enjoying '{favorite_book}' and has a preference for the '{preferred_trope}' trope.

Follow this systematic procedure using your MCP tools:
1. First, call `recommend_similar_books` or `search_books` to locate '{favorite_book}' and retrieve its confirmed tropes and metadata.
2. Next, call `find_books_by_tropes` with the trope '{preferred_trope}' to discover candidate titles verified to contain this dynamic.
3. For the top candidates, call `get_book_details` or `recommend_similar_books` to obtain similarity scores and evidence excerpts.
4. Synthesize your final recommendations for the reader:
   - Clearly state why each book was chosen.
   - Reference concrete evidence from the backend rather than fabricating descriptions.
   - If a requested trope cannot be confirmed in the dataset, explicitly declare it as unverified.
"""

    @server.prompt(
        name="romance_trope_explorer",
        description="Explore specific romance tropes (e.g. Enemies to Lovers, Slow Burn, Fake Dating) with textual evidence.",
    )
    def romance_trope_explorer(
        primary_trope: str = "Enemies to Lovers",
        secondary_trope: str = "Slow Burn",
    ) -> str:
        return f"""You are an expert literary trope analyst specializing in romance literature.
Your task is to analyze titles featuring '{primary_trope}' and '{secondary_trope}'.

Instructions:
1. Read the resource `books://tropes` to review the canonical definitions and keywords.
2. Call `find_books_by_tropes` with tropes=['{primary_trope}', '{secondary_trope}'] (try both match_all=True and match_all=False).
3. For each book returned, inspect the `matched_tropes_with_evidence` field:
   - Identify the classification method (`rule_based_keyword`, `tfidf_matcher`, or `hybrid`).
   - Quote the verified evidence snippet returned by the server.
4. Present your findings to the user, highlighting how the author executes both tropes together.
"""

    @server.prompt(
        name="book_review_analysis",
        description="Analyze reader sentiment and thematic feedback (character, pacing, plot) for a romance book.",
    )
    def book_review_analysis(book_id: str = "56732449") -> str:
        return f"""You are a reader analytics specialist.
You are tasked with providing an empirical review analysis for book ID '{book_id}'.

Instructions:
1. Call `analyse_book_reviews` with book_id='{book_id}'.
2. Review the sentiment distribution (positive, neutral, negative percentage).
3. Examine the three thematic feedback dimensions:
   - Character dynamics (e.g. hero appeal, banter, heroine authenticity).
   - Narrative pacing (e.g. brisk flow, slow burn buildup, middle drag).
   - Plot construction (e.g. third-act conflict plausibility, payoff satisfaction).
4. Provide a balanced, objective report. Do not invent reader quotes; use only the representative snippets and themes supplied by the backend.
"""

    @server.prompt(
        name="reading_profile_analysis",
        description="Analyze a reader's historical interaction profile and provide personalized next reads.",
    )
    def reading_profile_analysis(user_id: str = "usr_workplace_01") -> str:
        return f"""You are a personalized reading concierge.
Your goal is to evaluate the reading history of user '{user_id}' and recommend their next reads.

Instructions:
1. Call `get_reader_insights` with user_id='{user_id}' to examine reading volume, average given star rating, and favorite subgenres.
2. Call `get_personalized_recommendations` with user_id='{user_id}'.
3. Note whether the backend flags this reader as a cold start or provides collaborative recommendations:
   - If collaborative: explain how their past ratings influenced the recommendations.
   - If cold-start: explain that recommendations are based on popular subgenres or specified tropes.
4. Formulate an engaging, personalized reading roadmap for the reader.
"""

    @server.prompt(
        name="compare_romance_books",
        description="Perform a side-by-side comparative analysis between two romance titles.",
    )
    def compare_romance_books(
        book_id_1: str = "56732449",
        book_id_2: str = "25614492",
    ) -> str:
        return f"""You are conducting a comparative literary analysis of two romance titles.
Book 1 ID: '{book_id_1}'
Book 2 ID: '{book_id_2}'

Instructions:
1. Call `compare_books` with book_id_1='{book_id_1}' and book_id_2='{book_id_2}'.
2. Compare the two works across:
   - Reader acclaim: average ratings and ratings volume.
   - Romance subgenres and setting.
   - Trope overlap: which tropes are shared versus unique to each book.
3. Call `analyse_book_reviews` for each book if deeper sentiment comparison on pacing or character dynamics is requested.
4. Conclude with guidance on which readers would prefer Book 1 versus Book 2.
"""
