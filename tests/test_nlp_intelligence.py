"""Unit tests for NLP components: Romance Trope Extractor, Sentiment Analyzer, Theme Extractor."""

import pytest
from nlp.sentiment_analyzer import SentimentAnalyzer
from nlp.theme_extractor import ReviewThemeExtractor
from nlp.trope_extractor import RomanceTropeExtractor

@pytest.fixture
def sentiment_analyzer():
    return SentimentAnalyzer()

@pytest.fixture
def trope_extractor():
    return RomanceTropeExtractor()

@pytest.fixture
def theme_extractor():
    return ReviewThemeExtractor()

def test_sentiment_positive_review(sentiment_analyzer: SentimentAnalyzer):
    text = "I absolutely loved this book! Adam Carlsen is a swoon-worthy hero with electric chemistry."
    res = sentiment_analyzer.analyze_text(text)
    assert res["label"] == "positive"
    assert res["compound"] > 0.3

def test_sentiment_negative_review(sentiment_analyzer: SentimentAnalyzer):
    text = "Awful and terribly boring. The characters were toxic, childish, and unlikable. Complete waste of time."
    res = sentiment_analyzer.analyze_text(text)
    assert res["label"] == "negative"
    assert res["compound"] < -0.3

def test_sentiment_neutral_review(sentiment_analyzer: SentimentAnalyzer):
    text = "The book was released as a paperback edition with 384 pages in October 2021."
    res = sentiment_analyzer.analyze_text(text)
    assert res["label"] == "neutral"
    assert abs(res["compound"]) < 0.1

def test_sentiment_negation_handling(sentiment_analyzer: SentimentAnalyzer):
    # 'boring' is negative, but 'not boring' should flip polarity
    negated = sentiment_analyzer.analyze_text("This book was not boring at all, it was captivating.")
    assert negated["label"] == "positive"
    assert negated["compound"] > 0.0

def test_sentiment_booster_handling(sentiment_analyzer: SentimentAnalyzer):
    baseline = sentiment_analyzer.analyze_text("The humor was hilarious.")
    boosted = sentiment_analyzer.analyze_text("The humor was extremely hilarious.")
    assert boosted["compound"] >= baseline["compound"]

def test_sentiment_empty_and_whitespace(sentiment_analyzer: SentimentAnalyzer):
    assert sentiment_analyzer.analyze_text("")["label"] == "neutral"
    assert sentiment_analyzer.analyze_text("   \n\t  ")["compound"] == 0.0

def test_trope_extraction_evidence_traceability(trope_extractor: RomanceTropeExtractor):
    desc = (
        "Olive Smith panics and kisses Adam Carlsen to convince her friend she is dating. "
        "Their fake dating arrangement turns into an office romance with intense slow burn tension."
    )
    tropes = trope_extractor.extract_tropes_from_text(desc)
    assert len(tropes) > 0

    trope_names = [t["trope_name"] for t in tropes]
    assert "Fake Dating" in trope_names
    assert "Slow Burn" in trope_names

    fake_dating = next(t for t in tropes if t["trope_name"] == "Fake Dating")
    assert fake_dating["confidence"] > 0.5
    assert "fake dating" in fake_dating["evidence"].lower()

def test_review_theme_extraction(theme_extractor: ReviewThemeExtractor):
    reviews = [
        {
            "review_text": "The pacing was brisk and page-turning! The grumpy hero melted for the heroine.",
        },
        {
            "review_text": "The third-act misunderstanding felt forced, but the emotional depth of the ending was deeply satisfying.",
        },
    ]
    themes = theme_extractor.extract_themes_for_book("book_test", reviews)
    assert len(themes) > 0

    types = {t["theme_type"] for t in themes}
    assert "pacing" in types
    assert "character" in types
    assert "plot" in types

    for t in themes:
        assert t["evidence"] != ""
        assert t["score"] > 0
