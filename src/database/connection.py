"""
Database connection management for Franlince API.
Provides connection pooling and context managers for database access.
"""

from contextlib import contextmanager
from typing import Generator, Any, Optional

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import pool

from src.core.config import get_settings


class DatabaseConnection:
    """Manages database connections with connection pooling."""

    _pool: Optional[pool.ThreadedConnectionPool] = None

    @classmethod
    def initialize_pool(cls, minconn: int = 1, maxconn: int = 10) -> None:
        """Initializes the connection pool."""
        if cls._pool is None:
            settings = get_settings()
            cls._pool = pool.ThreadedConnectionPool(
                minconn=minconn,
                maxconn=maxconn,
                **settings.db_config
            )

    @classmethod
    def close_pool(cls) -> None:
        """Closes all connections in the pool."""
        if cls._pool is not None:
            cls._pool.closeall()
            cls._pool = None

    @classmethod
    @contextmanager
    def get_connection(cls) -> Generator[Any, None, None]:
        """
        Context manager for getting a database connection from the pool.

        Usage:
            with DatabaseConnection.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM pinturas")
        """
        if cls._pool is None:
            cls.initialize_pool()

        conn = cls._pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cls._pool.putconn(conn)

    @classmethod
    @contextmanager
    def get_cursor(cls, cursor_factory=RealDictCursor) -> Generator[Any, None, None]:
        """
        Context manager for getting a cursor with automatic connection handling.

        Usage:
            with DatabaseConnection.get_cursor() as cursor:
                cursor.execute("SELECT * FROM pinturas")
                results = cursor.fetchall()
        """
        with cls.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=cursor_factory)
            try:
                yield cursor
            finally:
                cursor.close()


def get_db_connection():
    """
    Simple function to get a database connection.
    For use without connection pooling (e.g., in scripts).
    """
    settings = get_settings()
    return psycopg2.connect(
        **settings.db_config,
        cursor_factory=RealDictCursor
    )
