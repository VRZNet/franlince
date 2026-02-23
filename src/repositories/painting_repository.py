"""
Repository for painting data access operations.
"""

import json
from typing import Optional, List, Tuple, Dict

import psycopg2

from src.database.connection import DatabaseConnection
from src.services.embedding import EmbeddingService
from src.core.constants import (
    MIN_SIMILARITY_THRESHOLD,
    MIN_VISUAL_SIMILARITY,
    MIN_EMOTION_SIMILARITY
)


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
            classification: Classification result with embedding and emotions.

        Returns:
            UUID of the created painting.
        """
        embedding_str = EmbeddingService.embedding_to_pg_format(
            classification["embedding"]
        )

        # Handle emotional embedding if present
        embedding_emocional_str = None
        if classification.get("embedding_emocional"):
            embedding_emocional_str = EmbeddingService.embedding_to_pg_format(
                classification["embedding_emocional"]
            )

        top = classification["top_estilos"]

        # Get top emotions if present
        top_emociones = classification.get("top_emociones", [])

        with DatabaseConnection.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO pinturas (
                    archivo, imagen, estilo_principal, confianza,
                    estilo_2, confianza_2, estilo_3, confianza_3,
                    todos_estilos, embedding,
                    emocion_principal, emocion_confianza,
                    emocion_2, emocion_2_confianza,
                    emocion_3, emocion_3_confianza,
                    todas_emociones, embedding_emocional
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector,
                    %s, %s, %s, %s, %s, %s, %s, %s::vector
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
                embedding_str,
                top_emociones[0]["emocion"] if len(top_emociones) > 0 else None,
                top_emociones[0]["confianza"] if len(top_emociones) > 0 else None,
                top_emociones[1]["emocion"] if len(top_emociones) > 1 else None,
                top_emociones[1]["confianza"] if len(top_emociones) > 1 else None,
                top_emociones[2]["emocion"] if len(top_emociones) > 2 else None,
                top_emociones[2]["confianza"] if len(top_emociones) > 2 else None,
                json.dumps(top_emociones) if top_emociones else None,
                embedding_emocional_str
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
                           estilo_2, confianza_2, estilo_3, confianza_3,
                           emocion_principal, created_at
                    FROM pinturas
                    WHERE estilo_principal = %s
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                """, (estilo, limit, offset))
            else:
                cursor.execute("""
                    SELECT id, archivo, ruta, estilo_principal, confianza,
                           estilo_2, confianza_2, estilo_3, confianza_3,
                           emocion_principal, created_at
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

    def search_by_content_then_emotion(
        self,
        content_embedding: List[float],
        emotion_embedding: List[float],
        content_limit: int = 50,
        final_limit: int = 20,
        min_content_similarity: float = MIN_VISUAL_SIMILARITY,
        min_emotion_similarity: float = MIN_EMOTION_SIMILARITY,
        content_weight: float = 0.6,
        emotion_weight: float = 0.4,
        emotion_name: Optional[str] = None
    ) -> List[dict]:
        """
        Two-step hybrid search: first filter by content, then rank by emotion.

        This ensures that content requirements are met FIRST, and then
        results are ordered by emotional similarity.

        Args:
            content_embedding: Visual content embedding from query.
            emotion_embedding: Emotional embedding from query.
            content_limit: Max candidates from content search.
            final_limit: Final number of results to return.
            min_content_similarity: Minimum content similarity (0-1).
            min_emotion_similarity: Minimum emotion similarity (0-1).
            content_weight: Weight for content score in final ranking.
            emotion_weight: Weight for emotion score in final ranking.
            emotion_name: If provided, filter to paintings classified with this emotion.

        Returns:
            List of paintings with combined scores.
        """
        content_emb_str = EmbeddingService.embedding_to_pg_format(content_embedding)
        emotion_emb_str = EmbeddingService.embedding_to_pg_format(emotion_embedding)

        # Build optional emotion classification filter
        if emotion_name:
            emotion_filter = """
                  AND (
                      p.emocion_principal = %s
                      OR p.emocion_2 = %s
                      OR p.emocion_3 = %s
                  )
            """
            params = (
                content_emb_str, content_emb_str, min_content_similarity,
                content_emb_str, content_limit,
                emotion_emb_str, emotion_emb_str, content_weight,
                emotion_emb_str, emotion_emb_str, emotion_weight,
                emotion_emb_str, emotion_emb_str, min_emotion_similarity,
                emotion_name, emotion_name, emotion_name,
                final_limit
            )
        else:
            emotion_filter = ""
            params = (
                content_emb_str, content_emb_str, min_content_similarity,
                content_emb_str, content_limit,
                emotion_emb_str, emotion_emb_str, content_weight,
                emotion_emb_str, emotion_emb_str, emotion_weight,
                emotion_emb_str, emotion_emb_str, min_emotion_similarity,
                final_limit
            )

        with DatabaseConnection.get_cursor() as cursor:
            cursor.execute(f"""
                WITH content_candidates AS (
                    SELECT
                        id, archivo, ruta, estilo_principal, confianza,
                        emocion_principal, emocion_confianza,
                        (embedding <#> %s::vector) * -1 as similitud_contenido
                    FROM pinturas
                    WHERE embedding IS NOT NULL
                      AND (embedding <#> %s::vector) * -1 >= %s
                    ORDER BY embedding <#> %s::vector
                    LIMIT %s
                )
                SELECT
                    cc.*,
                    COALESCE(
                        (p.embedding_emocional <#> %s::vector) * -1,
                        (p.embedding <#> %s::vector) * -1
                    ) as similitud_emocion,
                    (cc.similitud_contenido * %s) +
                    (COALESCE(
                        (p.embedding_emocional <#> %s::vector) * -1,
                        (p.embedding <#> %s::vector) * -1
                    ) * %s) as score_combinado
                FROM content_candidates cc
                JOIN pinturas p ON cc.id = p.id
                WHERE COALESCE(
                    (p.embedding_emocional <#> %s::vector) * -1,
                    (p.embedding <#> %s::vector) * -1
                ) >= %s
                {emotion_filter}
                ORDER BY score_combinado DESC
                LIMIT %s
            """, params)

            results = []
            for row in cursor.fetchall():
                results.append({
                    "id": row["id"],
                    "archivo": row["archivo"],
                    "ruta": row["ruta"],
                    "estilo_principal": row["estilo_principal"],
                    "confianza": row["confianza"],
                    "emocion_principal": row["emocion_principal"],
                    "emocion_confianza": row["emocion_confianza"],
                    "similitud_contenido": row["similitud_contenido"],
                    "similitud_emocion": row["similitud_emocion"],
                    "score_combinado": row["score_combinado"]
                })

            return results

    def search_by_emotion(
        self,
        emotion_embedding: List[float],
        limit: int = 20,
        min_similarity: float = MIN_EMOTION_SIMILARITY,
        emotion_name: Optional[str] = None
    ) -> List[dict]:
        """
        Search paintings by emotional similarity only.

        When emotion_name matches a known category, results are filtered
        to only include paintings classified with that emotion (in their
        top 3 emotions), ensuring relevant results.

        Args:
            emotion_embedding: Emotional query embedding.
            limit: Maximum results.
            min_similarity: Minimum similarity threshold.
            emotion_name: If provided, filter to paintings classified with this emotion.

        Returns:
            List of paintings ordered by emotional similarity.
        """
        emotion_emb_str = EmbeddingService.embedding_to_pg_format(emotion_embedding)

        with DatabaseConnection.get_cursor() as cursor:
            # Build emotion classification filter
            if emotion_name:
                emotion_filter = """
                  AND (
                      emocion_principal = %s
                      OR emocion_2 = %s
                      OR emocion_3 = %s
                  )
                """
                params = (
                    emotion_emb_str, emotion_emb_str,
                    emotion_emb_str, emotion_emb_str, min_similarity,
                    emotion_name, emotion_name, emotion_name,
                    emotion_emb_str, emotion_emb_str,
                    limit
                )
            else:
                emotion_filter = ""
                params = (
                    emotion_emb_str, emotion_emb_str,
                    emotion_emb_str, emotion_emb_str, min_similarity,
                    emotion_emb_str, emotion_emb_str,
                    limit
                )

            cursor.execute(f"""
                SELECT
                    id, archivo, ruta, estilo_principal, confianza,
                    emocion_principal, emocion_confianza,
                    COALESCE(
                        (embedding_emocional <#> %s::vector) * -1,
                        (embedding <#> %s::vector) * -1
                    ) as similitud_emocion
                FROM pinturas
                WHERE embedding IS NOT NULL
                  AND COALESCE(
                      (embedding_emocional <#> %s::vector) * -1,
                      (embedding <#> %s::vector) * -1
                  ) >= %s
                  {emotion_filter}
                ORDER BY COALESCE(
                    embedding_emocional <#> %s::vector,
                    embedding <#> %s::vector
                )
                LIMIT %s
            """, params)

            return [dict(row) for row in cursor.fetchall()]

    def update_emotion_data(
        self,
        painting_id: str,
        emotion_embedding: List[float],
        emotions: List[dict]
    ) -> bool:
        """
        Update emotion embedding and classification for a painting.

        Args:
            painting_id: UUID of the painting.
            emotion_embedding: New emotional embedding.
            emotions: List of emotion classifications.

        Returns:
            True if updated, False if not found.
        """
        emotion_emb_str = EmbeddingService.embedding_to_pg_format(emotion_embedding)

        with DatabaseConnection.get_cursor() as cursor:
            cursor.execute("""
                UPDATE pinturas
                SET embedding_emocional = %s::vector,
                    emocion_principal = %s,
                    emocion_confianza = %s,
                    emocion_2 = %s,
                    emocion_2_confianza = %s,
                    emocion_3 = %s,
                    emocion_3_confianza = %s,
                    todas_emociones = %s
                WHERE id = %s
            """, (
                emotion_emb_str,
                emotions[0]["emocion"] if len(emotions) > 0 else None,
                emotions[0]["confianza"] if len(emotions) > 0 else None,
                emotions[1]["emocion"] if len(emotions) > 1 else None,
                emotions[1]["confianza"] if len(emotions) > 1 else None,
                emotions[2]["emocion"] if len(emotions) > 2 else None,
                emotions[2]["confianza"] if len(emotions) > 2 else None,
                json.dumps(emotions),
                painting_id
            ))

            return cursor.rowcount > 0

    def get_emotion_stats(self) -> List[dict]:
        """
        Get statistics about emotion classifications.

        Returns:
            List of emotion stats with count and average confidence.
        """
        with DatabaseConnection.get_cursor() as cursor:
            cursor.execute("""
                SELECT emocion_principal, COUNT(*) as cantidad,
                       ROUND(AVG(emocion_confianza)::numeric, 3) as confianza_promedio
                FROM pinturas
                WHERE emocion_principal IS NOT NULL
                GROUP BY emocion_principal
                ORDER BY cantidad DESC
            """)

            return [dict(row) for row in cursor.fetchall()]

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
