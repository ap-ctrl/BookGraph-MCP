"""Thematic feedback extraction from reader reviews across pacing, character, and plot."""

import re
from typing import Any, Dict, List

THEME_PATTERNS = {
    "pacing": {
        "brisk_pacing": {
            "patterns": [r"\bbrisk\b", r"\bfast[- ]paced\b", r"\bpage[- ]turner\b", r"\bread in one sitting\b", r"\bflew through\b"],
            "description": "Fast, engaging pacing that keeps readers hooked",
        },
        "slow_pacing": {
            "patterns": [r"\bslow[- ]burn\b", r"\bunhurried\b", r"\btakes time to build\b", r"\bgradual\b"],
            "description": "Deliberate, slow-building romantic tension",
        },
        "dragged_middle": {
            "patterns": [r"\bdragged\b", r"\bstalled\b", r"\bslow in the middle\b", r"\bmeandering\b", r"\bmiddle chapters\b"],
            "description": "Reader perception of pacing slump or narrative dragging in the middle",
        },
        "rushed_ending": {
            "patterns": [r"\brushed ending\b", r"\babrupt (?:ending|resolution)\b", r"\bwrapped up too quickly\b"],
            "description": "Reader perception that climax and resolution were rushed",
        },
    },
    "character": {
        "grumpy_hero": {
            "patterns": [r"\bgrumpy hero\b", r"\bsoft for (?:her|the heroine)\b", r"\bbrooding\b", r"\bgold standard\b"],
            "description": "Beloved stern or grumpy male lead who melts for the protagonist",
        },
        "relatable_heroine": {
            "patterns": [r"\brelatable\b", r"\blovable heroine\b", r"\bwitty banter\b", r"\bsharp\b", r"\bambitious\b"],
            "description": "Engaging, emotionally authentic, or sharp female protagonist",
        },
        "childish_behavior": {
            "patterns": [r"\bchildish\b", r"\bimmature\b", r"\bpetty\b", r"\bunlikable\b", r"\bcringeworthy\b"],
            "description": "Critique of juvenile or unconvincing adult protagonist actions",
        },
        "electric_chemistry": {
            "patterns": [r"\belectric chemistry\b", r"\bspark\b", r"\birresistible\b", r"\bswoon\b", r"\bbanter was clever\b"],
            "description": "High praise for romantic spark, banter, and mutual attraction",
        },
    },
    "plot": {
        "third_act_misunderstanding": {
            "patterns": [r"\bthird[- ]act\b", r"\bmisunderstanding\b", r"\blame miscommunication\b", r"\bforced conflict\b"],
            "description": "Feedback regarding the third-act breakup or conflict plausibility",
        },
        "satisfying_payoff": {
            "patterns": [r"\bsatisfying\b", r"\bpayoff\b", r"\bheartwarming\b", r"\bemotional resolution\b", r"\bdeeply satisfying\b"],
            "description": "Strong reader satisfaction with romantic climax and happily-ever-after",
        },
        "emotional_depth": {
            "patterns": [r"\bemotional depth\b", r"\bgrief\b", r"\bvulnerability\b", r"\bhealing\b", r"\bprofound heart\b"],
            "description": "Appreciation for nuanced exploration of grief, family, and personal growth",
        },
        "authentic_setting": {
            "patterns": [r"\bauthentic\b", r"\bacademic setting\b", r"\bworkplace setting\b", r"\bsetting was gorgeous\b"],
            "description": "Reader praise for realistic world-building and atmospheric setting",
        },
    },
}

class ReviewThemeExtractor:
    """Extracts thematic sentiment dimensions (character, pacing, plot) from reviews."""

    def extract_themes_for_book(
        self, book_id: str, reviews: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract recurring themes and evidence quotes from a book's reviews."""
        if not reviews:
            return []

        theme_scores: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for rev in reviews:
            text = rev.get("review_text", "")
            if not text:
                continue

            for theme_type, subthemes in THEME_PATTERNS.items():
                for theme_key, data in subthemes.items():
                    for pattern in data["patterns"]:
                        match = re.search(pattern, text, re.IGNORECASE)
                        if match:
                            pair = (theme_type, theme_key)
                            # Extract short sentence excerpt (under 160 chars) as evidence
                            sentences = re.split(r"(?<=[.!?])\s+", text)
                            evidence_quote = ""
                            for s in sentences:
                                if re.search(pattern, s, re.IGNORECASE):
                                    evidence_quote = s.strip()[:180]
                                    break

                            if pair not in theme_scores:
                                theme_scores[pair] = {
                                    "book_id": book_id,
                                    "theme_type": theme_type,
                                    "theme_key": theme_key,
                                    "score": 0.0,
                                    "evidence": evidence_quote,
                                    "count": 0,
                                }
                            theme_scores[pair]["count"] += 1
                            theme_scores[pair]["score"] += 1.0
                            if not theme_scores[pair]["evidence"]:
                                theme_scores[pair]["evidence"] = evidence_quote

        results = []
        for (theme_type, theme_key), item in theme_scores.items():
            results.append({
                "book_id": book_id,
                "theme_type": theme_type,
                "theme_key": theme_key,
                "score": round(item["score"], 2),
                "evidence": item["evidence"],
            })

        return sorted(results, key=lambda x: x["score"], reverse=True)
