"""Application configuration and centralized settings."""

import logging
import os
import sys
from pathlib import Path
from pydantic import BaseModel, Field

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Set safe MPLCONFIGDIR to avoid permission warnings on Windows
os.environ["MPLCONFIGDIR"] = str(BASE_DIR / ".matplotlib_cache")

class Settings(BaseModel):
    """Central configuration class."""
    server_name: str = "romance-reader-intelligence"
    server_version: str = "0.1.0"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    transport: str = os.getenv("MCP_TRANSPORT", "stdio")
    port: int = int(os.getenv("PORT", "8000"))
    host: str = os.getenv("HOST", "127.0.0.1")

    # Storage paths
    base_dir: Path = BASE_DIR
    raw_data_dir: Path = BASE_DIR / "data" / "raw"
    processed_data_dir: Path = BASE_DIR / "data" / "processed"
    external_data_dir: Path = BASE_DIR / "data" / "external"
    models_dir: Path = BASE_DIR / "models"
    reports_dir: Path = BASE_DIR / "reports"
    figures_dir: Path = BASE_DIR / "reports" / "figures"
    db_path: Path = BASE_DIR / "data" / "processed" / "romance_books.db"

    # Open Library API settings
    openlibrary_user_agent: str = os.getenv(
        "OPENLIBRARY_USER_AGENT",
        "RomanceReaderIntelligence/0.1.0 (academic-research@example.com)"
    )
    openlibrary_delay: float = float(os.getenv("OPENLIBRARY_RATE_LIMIT_DELAY", "1.0"))

    # Optional LLM integration settings
    llm_provider: str = os.getenv("LLM_PROVIDER", "none")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "")

    def ensure_directories(self) -> None:
        """Create necessary directories if they do not exist."""
        for directory in [
            self.raw_data_dir,
            self.processed_data_dir,
            self.external_data_dir,
            self.models_dir,
            self.reports_dir,
            self.figures_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

settings = Settings()

def setup_logging(level: str | None = None) -> logging.Logger:
    """
    Configure logging strictly directed to sys.stderr.
    STDIO transport requires stdout to be pure JSON-RPC messages.
    """
    log_level_name = level or settings.log_level
    numeric_level = getattr(logging, log_level_name.upper(), logging.INFO)

    logger = logging.getLogger("romance_reader_intelligence")
    logger.setLevel(numeric_level)

    # Avoid duplicate handlers if setup_logging is called multiple times
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Prevent propagation to root logger if root has stdout handlers
    logger.propagate = False
    return logger

logger = setup_logging()
