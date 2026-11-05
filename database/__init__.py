"""Database package for Romance Reader Intelligence."""

from database.connection import get_db_connection, init_db
from database.repository import BookRepository

__all__ = ["get_db_connection", "init_db", "BookRepository"]
