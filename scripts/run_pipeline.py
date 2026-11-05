"""End-to-End reproducible pipeline script for Romance Reader Intelligence."""

import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.eda import ExploratoryDataAnalysis
from config.settings import logger, settings
from database.connection import init_db
from database.repository import BookRepository
from ingestion.dataset_manager import DatasetManager
from nlp.sentiment_analyzer import SentimentAnalyzer
from nlp.theme_extractor import ReviewThemeExtractor
from nlp.trope_extractor import RomanceTropeExtractor
from preprocessing.preprocessor import DataPreprocessor
from recommendation.evaluator import RecommendationEvaluator

def run_pipeline() -> None:
    """Execute complete end-to-end ingestion, preprocessing, NLP, and modeling pipeline."""
    start_time = time.time()
    logger.info("Starting Romance Reader Intelligence Pipeline...")
    settings.ensure_directories()

    # Step 1: Database Initialization
    logger.info("Step 1: Initializing SQLite database schema...")
    init_db()
    repo = BookRepository()

    # Step 2: Data Acquisition / Ingestion
    logger.info("Step 2: Checking source datasets and determining operational mode...")
    dm = DatasetManager()
    books_file, inter_file, rev_file, dataset_mode = dm.get_source_paths()
    logger.info("Dataset mode established: '%s'", dataset_mode)

    if dataset_mode == "full_dataset":
        logger.info("Streaming from full UCSD raw dataset files: %s", [f.name for f in [books_file, inter_file, rev_file]])
        raw_books = dm.stream_records(books_file, max_records=5000)
        raw_interactions = dm.stream_records(inter_file, max_records=25000)
        raw_reviews = dm.stream_records(rev_file, max_records=5000)
    else:
        logger.info("Full UCSD files not detected locally; streaming from curated development fixture (%s)", books_file.name)
        raw_books = dm.stream_records(books_file)
        raw_interactions = dm.stream_records(inter_file)
        raw_reviews = dm.stream_records(rev_file)

    # Step 3: Data Preprocessing & Validation
    logger.info("Step 3: Preprocessing and validating datasets...")
    preprocessor = DataPreprocessor()
    books_df, interactions_df, reviews_df = preprocessor.process_all(
        raw_books, raw_interactions, raw_reviews, dataset_mode=dataset_mode
    )

    # Insert into database
    logger.info("Step 4: Populating database tables...")
    repo.insert_or_replace_books(books_df.to_dict("records"))
    repo.insert_or_replace_interactions(interactions_df.to_dict("records"))

    # Step 5: Trope Extraction (NLP)
    logger.info("Step 5: Extracting Romance Tropes with evidence tracking...")
    trope_extractor = RomanceTropeExtractor()
    all_tropes = []
    for _, b in books_df.iterrows():
        book_id = b["book_id"]
        desc = b["description"]
        shelves = b.get("popular_shelves")
        extracted = trope_extractor.extract_tropes_from_text(desc, shelves)
        for t in extracted:
            all_tropes.append({
                "book_id": book_id,
                "trope_name": t["trope_name"],
                "confidence": t["confidence"],
                "method": t["method"],
                "evidence": t["evidence"],
            })
    repo.insert_or_replace_tropes(all_tropes)
    logger.info("Extracted %d trope classifications across books.", len(all_tropes))

    # Step 6: Review Sentiment & Theme Mining (NLP)
    logger.info("Step 6: Analyzing review sentiments and narrative themes...")
    sentiment_analyzer = SentimentAnalyzer()
    analyzed_reviews = sentiment_analyzer.analyze_batch(reviews_df.to_dict("records"))
    repo.insert_or_replace_reviews(analyzed_reviews)

    theme_extractor = ReviewThemeExtractor()
    all_themes = []
    for book_id in books_df["book_id"].unique():
        book_revs = [r for r in analyzed_reviews if r["book_id"] == book_id]
        themes = theme_extractor.extract_themes_for_book(book_id, book_revs)
        all_themes.extend(themes)
    repo.insert_or_replace_review_themes(all_themes)
    logger.info("Extracted %d review themes across books.", len(all_themes))

    # Step 7: Exploratory Data Analysis (EDA)
    logger.info("Step 7: Generating Exploratory Data Analysis and charts...")
    eda = ExploratoryDataAnalysis(repo)
    eda.generate_all_analyses()

    # Step 8: Recommendation Benchmark & Evaluation
    logger.info("Step 8: Running Recommendation Benchmarks (User-Aware Train/Test Split)...")
    evaluator = RecommendationEvaluator(repo)
    evaluator.run_benchmark(k=5)

    duration = round(time.time() - start_time, 2)
    cat_stats = repo.get_catalog_statistics()
    logger.info(
        "Pipeline completed successfully in %s seconds! Catalogue: %d books, %d interactions, %d reviews.",
        duration,
        cat_stats["total_books"],
        cat_stats["total_interactions"],
        cat_stats["total_reviews"],
    )

if __name__ == "__main__":
    run_pipeline()
