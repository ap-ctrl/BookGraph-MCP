"""Test fixtures for Romance Reader Intelligence."""

import os
import pytest
from pathlib import Path

from config.settings import settings
from database.connection import init_db
from database.repository import BookRepository
from server import create_server
from services.book_service import BookService
from services.nlp_service import NLPService
from services.recommendation_service import RecommendationService

@pytest.fixture(scope="session")
def repo():
    """Session repository pointing to initialized database."""
    init_db(settings.db_path)
    return BookRepository(settings.db_path)

@pytest.fixture(scope="session")
def book_service(repo):
    return BookService(repo)

@pytest.fixture(scope="session")
def rec_service(repo):
    return RecommendationService(repo)

@pytest.fixture(scope="session")
def nlp_service(repo):
    return NLPService(repo)

@pytest.fixture(scope="session")
def mcp_server():
    """Configured MCPServer instance."""
    return create_server()
