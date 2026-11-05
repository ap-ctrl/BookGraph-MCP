"""Sentiment analysis engine for Romance Reader Intelligence."""

import math
import re
from typing import Any, Dict, List

# Core romance and literary review sentiment lexicon
POSITIVE_WORDS = {
    "love": 3.0, "loved": 3.2, "adoring": 2.8, "adored": 3.0, "amazing": 2.9,
    "brilliant": 3.0, "swoon": 2.9, "swoon-worthy": 3.4, "swoony": 3.1, "masterpiece": 3.5,
    "perfection": 3.4, "perfect": 3.0, "heartwarming": 2.7, "hilarious": 2.5,
    "chemistry": 2.5, "electric": 2.8, "delightful": 2.6, "captivating": 3.0,
    "unforgettable": 3.0, "satisfying": 2.6, "charming": 2.4, "witty": 2.5,
    "steamy": 2.2, "sweet": 2.0, "authentic": 2.3, "recommend": 2.2, "favorite": 3.0,
    "crying": 1.5, "beautiful": 2.8, "gorgeous": 2.7, "treasured": 2.9,
}

NEGATIVE_WORDS = {
    "hate": -3.0, "hated": -3.2, "awful": -3.0, "terrible": -3.0, "boring": -2.8,
    "bored": -2.6, "bland": -2.2, "disappointed": -2.5, "disappointing": -2.6,
    "forced": -2.0, "cringe": -2.4, "cringeworthy": -2.7, "childish": -2.2,
    "toxic": -3.0, "dragged": -2.3, "unlikable": -2.5, "flat": -2.0,
    "insta-love": -2.2, "cliché": -1.8, "cliche": -1.8, "tedious": -2.5,
    "annoying": -2.3, "stalled": -2.0, "cheated": -3.0, "ridiculous": -2.2,
    "poor": -2.0, "waste": -3.0, "shallow": -2.2, "abrupt": -1.8,
}

BOOSTERS = {"very": 1.3, "really": 1.25, "extremely": 1.4, "so": 1.2, "incredibly": 1.35}
NEGATORS = {"not", "never", "didn't", "wasn't", "barely", "hardly", "no"}

class SentimentAnalyzer:
    """Robust lexicon-based sentiment analyzer with negation and booster handling."""

    def __init__(self):
        self.positive_lexicon = POSITIVE_WORDS
        self.negative_lexicon = NEGATIVE_WORDS

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of a single review text.
        Returns compound polarity in [-1.0, 1.0] and categorical label.
        """
        if not text or not text.strip():
            return {"compound": 0.0, "label": "neutral", "score": 0.0}

        words = re.findall(r"\b[a-zA-Z'-]+\b", text.lower())
        total_score = 0.0
        word_count = len(words)

        if word_count == 0:
            return {"compound": 0.0, "label": "neutral", "score": 0.0}

        for i, word in enumerate(words):
            val = 0.0
            if word in self.positive_lexicon:
                val = self.positive_lexicon[word]
            elif word in self.negative_lexicon:
                val = self.negative_lexicon[word]

            if val != 0.0:
                # Check for preceding booster
                if i > 0 and words[i - 1] in BOOSTERS:
                    val *= BOOSTERS[words[i - 1]]

                # Check for preceding negation in 3-word window
                window = words[max(0, i - 3):i]
                if any(neg in window for neg in NEGATORS):
                    val = -val * 0.8

                total_score += val

        # Normalize score into [-1.0, 1.0] using hyperbolic tangent scaling
        normalized_compound = math.tanh(total_score / 4.0)
        normalized_compound = round(normalized_compound, 3)

        if normalized_compound >= 0.08:
            label = "positive"
        elif normalized_compound <= -0.08:
            label = "negative"
        else:
            label = "neutral"

        return {
            "compound": normalized_compound,
            "label": label,
            "score": normalized_compound,
        }

    def analyze_batch(self, reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of reviews efficiently."""
        results = []
        for r in reviews:
            res = self.analyze_text(r.get("review_text", ""))
            augmented = dict(r)
            augmented["sentiment_score"] = res["compound"]
            augmented["sentiment_label"] = res["label"]
            results.append(augmented)
        return results
