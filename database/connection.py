"""Database connection manager and schema initialization."""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

from config.settings import logger, settings

SCHEMA_PATH = Path(__file__).parent / "schema.sql"

def get_raw_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Create a raw SQLite connection with row factory and WAL mode."""
    target_path = db_path or settings.db_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn

@contextmanager
def get_db_connection(db_path: Optional[Path] = None) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for SQLite connections with auto-commit/rollback."""
    conn = get_raw_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error("Database error occurred: %s", e)
        raise
    finally:
        conn.close()

def init_db(db_path: Optional[Path] = None) -> None:
    """Initialize SQLite database using schema.sql."""
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema_sql = f.read()

    with get_db_connection(db_path) as conn:
        conn.executescript(schema_sql)
    logger.info("Database schema initialized at %s", db_path or settings.db_path)
