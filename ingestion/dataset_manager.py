"""Dataset Manager for acquiring, streaming, inspecting, and sampling Romance datasets."""

import gzip
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

from config.settings import logger, settings

UCSD_HOMEPAGE = "https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home"
UCSD_DOWNLOADS = "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/UCSD%20Book%20Graph.html"

REQUIRED_BOOK_FILES = [
    "goodreads_books_romance.json.gz",
    "goodreads_interactions_romance.json.gz",
    "goodreads_reviews_romance.json.gz",
]

class DatasetManager:
    """Manages raw dataset acquisition, streaming ingestion, and development sampling."""

    def __init__(self, raw_dir: Optional[Path] = None, processed_dir: Optional[Path] = None):
        self.raw_dir = raw_dir or settings.raw_data_dir
        self.processed_dir = processed_dir or settings.processed_data_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_download_instructions() -> str:
        """Return clear, academic-attribution compliant download instructions."""
        return f"""================================================================================
UCSD GOODREADS ROMANCE BOOK GRAPH DATASET ACQUISITION GUIDE
================================================================================
Source Attribution:
  Wan, Mengting, and Julian McAuley. "Item Recommendation on Monotonic Behavior Chains."
  RecSys 2018.
  UCSD Book Graph Project: {UCSD_HOMEPAGE}
  Dataset Portal: {UCSD_DOWNLOADS}

Terms of Use:
  - strictly for academic research and personal educational use.
  - Commercial use, public redistribution, and direct scraping are strictly prohibited.

Download Links (Romance subset):
  1. Books:        goodreads_books_romance.json.gz (approx 120MB compressed)
  2. Interactions: goodreads_interactions_romance.json.gz (approx 1.2GB compressed)
  3. Reviews:      goodreads_reviews_romance.json.gz (approx 1.5GB compressed)

Instructions:
  1. Download the above 3 files from the official UCSD Book Graph page.
  2. Place them into: {settings.raw_data_dir.resolve()}
  3. Do not commit or push the raw dataset files to Git.
================================================================================
"""

    def check_raw_files(self) -> Dict[str, bool]:
        """Check whether the full raw UCSD files are present."""
        status = {}
        for filename in REQUIRED_BOOK_FILES:
            filepath = self.raw_dir / filename
            status[filename] = filepath.exists() and filepath.stat().st_size > 0
        return status

    def stream_records(
        self, filepath: Path, max_records: Optional[int] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream records one-by-one from a gzipped or plain JSON file.
        Memory safe: avoids loading complete 42M interaction arrays into RAM.
        """
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        is_gzipped = filepath.suffix == ".gz" or str(filepath).endswith(".json.gz")
        open_fn: Callable = gzip.open if is_gzipped else open

        count = 0
        with open_fn(filepath, mode="rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    record = json.loads(line_str)
                    yield record
                    count += 1
                    if max_records and count >= max_records:
                        break
                except json.JSONDecodeError as e:
                    logger.warning("Skipping malformed JSON line in %s: %s", filepath.name, e)
                    continue

    def inspect_record_structure(
        self, filepath: Path, sample_size: int = 5
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Inspect record fields and sample records from a dataset file."""
        fields = set()
        samples = []
        for record in self.stream_records(filepath, max_records=sample_size):
            fields.update(record.keys())
            samples.append(record)
        return sorted(list(fields)), samples

    def create_development_sample(
        self,
        output_books_path: Optional[Path] = None,
        output_interactions_path: Optional[Path] = None,
        output_reviews_path: Optional[Path] = None,
    ) -> Dict[str, int]:
        """
        Generate a curated, authentic development dataset based on canonical romance books,
        rich interactions, and multi-thematic reviews adhering strictly to UCSD schema format.
        Ensures the pipeline is 100% reproducible and testable immediately.
        """
        b_path = output_books_path or (self.raw_dir / "sample_goodreads_books_romance.json.gz")
        i_path = output_interactions_path or (self.raw_dir / "sample_goodreads_interactions_romance.json.gz")
        r_path = output_reviews_path or (self.raw_dir / "sample_goodreads_reviews_romance.json.gz")

        from ingestion.seed_data import CANONICAL_BOOKS, CANONICAL_INTERACTIONS, CANONICAL_REVIEWS

        # Write Books
        with gzip.open(b_path, "wt", encoding="utf-8") as f:
            for book in CANONICAL_BOOKS:
                f.write(json.dumps(book) + "\n")

        # Write Interactions
        with gzip.open(i_path, "wt", encoding="utf-8") as f:
            for interaction in CANONICAL_INTERACTIONS:
                f.write(json.dumps(interaction) + "\n")

        # Write Reviews
        with gzip.open(r_path, "wt", encoding="utf-8") as f:
            for review in CANONICAL_REVIEWS:
                f.write(json.dumps(review) + "\n")

        logger.info(
            "Created development dataset: %d books, %d interactions, %d reviews",
            len(CANONICAL_BOOKS),
            len(CANONICAL_INTERACTIONS),
            len(CANONICAL_REVIEWS),
        )
        return {
            "books": len(CANONICAL_BOOKS),
            "interactions": len(CANONICAL_INTERACTIONS),
            "reviews": len(CANONICAL_REVIEWS),
        }
