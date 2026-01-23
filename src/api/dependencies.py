"""
Dependency injection for FastAPI.
Provides dependencies for database, classifier, and repository.
"""

from functools import lru_cache
from typing import Generator

from src.database.connection import DatabaseConnection
from src.services.classifier import CLIPClassifier
from src.services.embedding import EmbeddingService
from src.repositories.painting_repository import PaintingRepository


def get_db() -> Generator:
    """
    Dependency for database connection.
    Yields a connection and ensures proper cleanup.
    """
    with DatabaseConnection.get_connection() as conn:
        yield conn


@lru_cache()
def get_classifier() -> CLIPClassifier:
    """
    Dependency for CLIP classifier (singleton).
    Returns the same instance across requests.
    """
    classifier = CLIPClassifier.get_instance()
    if not classifier.is_loaded:
        classifier.load_model()
    return classifier


@lru_cache()
def get_embedding_service() -> EmbeddingService:
    """
    Dependency for embedding service (singleton).
    Returns the same instance across requests.
    """
    service = EmbeddingService.get_instance()
    if not service.is_loaded:
        service.load_model()
    return service


def get_repository() -> PaintingRepository:
    """
    Dependency for painting repository.
    Creates a new instance per request.
    """
    return PaintingRepository()
