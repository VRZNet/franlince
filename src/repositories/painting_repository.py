"""
Repository for painting data access operations.
"""

import json
from typing import Optional, List, Tuple, Dict

import psycopg2

from src.database.connection import DatabaseConnection
from src.services.embedding import EmbeddingService
from src.core.constants import MIN_SIMILARITY_THRESHOLD


class PaintingRepository:
    """Repository for CRUD operations on paintings."""

    def save(
        self,
        filename: str,
        image_bytes: bytes,
        classification: dict
    ) -> str:
        """
        Save a new painting to the database.

        Args:
            filename: Name of the image file.
            image_bytes: Raw image bytes.
            classification: Classification result with embedding.

        Returns:
            UUID of the created painting.
        """
        embedding_str = EmbeddingService.embedding_to_pg_format(
            classification["embedding"]
        )

        top = classification["top_estilos"]

        with DatabaseConnection.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO pinturas (
                    archivo, imagen, estilo_principal, confianza,
                    estilo_2, confianza_2, estilo_3, confianza_3,
                    todos_estilos, embedding
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector
                )
                RETURNING id
            """, (
                filename,
                psycopg2.Binary(image_bytes),
                classification["estilo_principal"],
                classification["confianza"],
                top[1]["estilo"] if len(top) > 1 else None,
                top[1]["confianza"] if len(top) > 1 else None,
                top[2]["estilo"] if len(top) > 2 else None,
                top[2]["confianza"] if len(top) > 2 else None,
                json.dumps(classification["todos_estilos"]),
                embedding_str
            ))

            result = cursor.fetchone()
            return str(result["id"])

    def get_by_id(self, painting_id: str) -> Optional[Dict]:
        """
        Get a painting by its ID.

        Args:
            painting_id: UUID of the painting.

        Returns:
            Painting data dict or None if not found.
        """
        with DatabaseConnection.get_cursor() as cursor:
            cursor.execute("""
                SELECT id, archivo, ruta, estilo_principal, confianza,
                       estilo_2, confianza_2, estilo_3, confianza_3,
                       todos_estilos, created_at, updated_at
                FROM pinturas
                WHERE id = %s
            """, (painting_id,))

            result = cursor.fetchone()
            return dict(result) if result else None

    def get_image(self, painting_id: str) -> Optional[Dict]:
        """
        Get painting image data.

        Args:
            painting_id: UUID of the painting.

        Returns:
            Dict with archivo and imagen or None.
        """
        with DatabaseConnection.get_cursor() as cursor:
            cursor.execute("""
                SELECT archivo, imagen
                FROM pinturas
                WHERE id = %s
            """, (painting_id,))

            result = cursor.fetchone()
            return dict(result) if result else None

    def list_all(
        self,
        estilo: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Tuple[List[Dict], int]:
        """
        List all paintings with optional filtering.

        Args:
            estilo: Filter by style.
            limit: Maximum results.
            offset: Pagination offset.

        Returns:
            Tuple of (list of paintings, total count).
        """
        with DatabaseConnection.get_cursor() as cursor:
            if estilo:
                cursor.execute("""
                    SELECT id, archivo, ruta, estilo_principal, confianza,
                           estilo_2, confianza_2, estilo_3, confianza_3, created_at
                    FROM pinturas
                    WHERE estilo_principal = %s
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                """, (estilo, limit, offset))
            else:
                cursor.execute("""
                    SELECT id, archivo, ruta, estilo_principal, confianza,
                           estilo_2, confianza_2, estilo_3, confianza_3, created_at
                    FROM pinturas
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                """, (limit, offset))

            paintings = [dict(row) for row in cursor.fetchall()]

            # Get total count
            if estilo:
                cursor.execute(
                    "SELECT COUNT(*) as total FROM pinturas WHERE estilo_principal = %s",
                    (estilo,)
                )
            else:
                cursor.execute("SELECT COUNT(*) as total FROM pinturas")

            total = cursor.fetchone()["total"]

            return paintings, total

    def search_by_style(
        self,
        estilo: str,
        min_confianza: float = 0.0
    ) -> List[dict]:
        """
        Search paintings by style with minimum confidence.

        Args:
            estilo: Style name.
            min_confianza: Minimum confidence threshold.

        Returns:
            List of matching paintings.
        """
        with DatabaseConnection.get_cursor() as cursor:
            cursor.execute("""
                SELECT id, archivo, ruta, estilo_principal, confianza, created_at
                FROM pinturas
                WHERE estilo_principal = %s AND confianza >= %s
                ORDER BY confianza DESC
            """, (estilo, min_confianza))

            return [dict(row) for row in cursor.fetchall()]

    def semantic_search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        min_similarity: float = MIN_SIMILARITY_THRESHOLD
    ) -> List[dict]:
        """
        Search paintings by semantic similarity.

        Args:
            query_embedding: Text query embedding.
            limit: Maximum results.
            min_similarity: Minimum similarity threshold (default 0.18 = 18%).

        Returns:
            List of matching paintings with similarity scores.
        """
        embedding_str = EmbeddingService.embedding_to_pg_format(query_embedding)

        with DatabaseConnection.get_cursor() as cursor:
            # Debug: ver todas las similitudes sin filtro para diagnóstico
            cursor.execute("""
                SELECT
                    archivo, estilo_principal,
                    (embedding <#> %s::vector) * -1 as similitud
                FROM pinturas
                WHERE embedding IS NOT NULL
                ORDER BY embedding <#> %s::vector
                LIMIT 5
            """, (embedding_str, embedding_str))
            top_5 = cursor.fetchall()
            print(f"[DEBUG] Top 5 similitudes (sin filtro):")
            for row in top_5:
                print(f"  - {row['archivo']}: {row['similitud']:.4f} ({row['estilo_principal']})")

            # Búsqueda real con filtro
            cursor.execute("""
                SELECT
                    id, archivo, ruta, estilo_principal, confianza,
                    (embedding <#> %s::vector) * -1 as similitud
                FROM pinturas
                WHERE embedding IS NOT NULL
                  AND (embedding <#> %s::vector) * -1 >= %s
                ORDER BY embedding <#> %s::vector
                LIMIT %s
            """, (embedding_str, embedding_str, min_similarity, embedding_str, limit))

            return [dict(row) for row in cursor.fetchall()]

    def get_similarity_for_painting(
        self,
        query_embedding: List[float],
        painting_id: str
    ) -> Optional[dict]:
        """
        Get similarity score for a specific painting.

        Args:
            query_embedding: Text query embedding.
            painting_id: UUID of the painting.

        Returns:
            Dict with painting info and similarity, or None.
        """
        embedding_str = EmbeddingService.embedding_to_pg_format(query_embedding)

        with DatabaseConnection.get_cursor() as cursor:
            # Get similarity for specific painting
            cursor.execute("""
                SELECT
                    id, archivo, estilo_principal,
                    (embedding <#> %s::vector) * -1 as similitud
                FROM pinturas
                WHERE id = %s AND embedding IS NOT NULL
            """, (embedding_str, painting_id))

            result = cursor.fetchone()
            if not result:
                return None

            # Get rank (position) among all paintings
            cursor.execute("""
                SELECT COUNT(*) + 1 as rank
                FROM pinturas
                WHERE embedding IS NOT NULL
                  AND (embedding <#> %s::vector) < (
                      SELECT embedding <#> %s::vector
                      FROM pinturas
                      WHERE id = %s
                  )
            """, (embedding_str, embedding_str, painting_id))

            rank_result = cursor.fetchone()

            return {
                **dict(result),
                "rank": rank_result["rank"] if rank_result else None
            }

    def delete(self, painting_id: str) -> Optional[dict]:
        """
        Delete a painting.

        Args:
            painting_id: UUID of the painting.

        Returns:
            Deleted painting info or None if not found.
        """
        with DatabaseConnection.get_cursor() as cursor:
            # Get info before deleting
            cursor.execute(
                "SELECT archivo, ruta FROM pinturas WHERE id = %s",
                (painting_id,)
            )
            painting = cursor.fetchone()

            if not painting:
                return None

            cursor.execute(
                "DELETE FROM pinturas WHERE id = %s",
                (painting_id,)
            )

            return dict(painting)

    def get_stats(self) -> dict:
        """
        Get catalog statistics.

        Returns:
            Dict with total, embeddings count, last update, and by style stats.
        """
        with DatabaseConnection.get_cursor() as cursor:
            # Total paintings
            cursor.execute("SELECT COUNT(*) as total FROM pinturas")
            total = cursor.fetchone()["total"]

            # By style
            cursor.execute("""
                SELECT estilo_principal, COUNT(*) as cantidad,
                       ROUND(AVG(confianza)::numeric, 3) as confianza_promedio
                FROM pinturas
                GROUP BY estilo_principal
                ORDER BY cantidad DESC
            """)
            por_estilo = [dict(row) for row in cursor.fetchall()]

            # With embeddings
            cursor.execute(
                "SELECT COUNT(*) as total FROM pinturas WHERE embedding IS NOT NULL"
            )
            con_embeddings = cursor.fetchone()["total"]

            # Last update
            cursor.execute("SELECT MAX(created_at) as ultima FROM pinturas")
            ultima = cursor.fetchone()["ultima"]

            return {
                "total_pinturas": total,
                "con_embeddings": con_embeddings,
                "ultima_actualizacion": ultima.isoformat() if ultima else None,
                "por_estilo": por_estilo
            }

    def update_embedding(
        self,
        filename: str,
        embedding: List[float]
    ) -> bool:
        """
        Update embedding for a painting.

        Args:
            filename: File name of the painting.
            embedding: New embedding values.

        Returns:
            True if updated, False if not found.
        """
        embedding_str = EmbeddingService.embedding_to_pg_format(embedding)

        with DatabaseConnection.get_cursor() as cursor:
            cursor.execute("""
                UPDATE pinturas
                SET embedding = %s::vector
                WHERE archivo = %s
            """, (embedding_str, filename))

            return cursor.rowcount > 0
