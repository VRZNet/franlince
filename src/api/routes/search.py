"""
Search routes for finding paintings.
Supports visual content search, emotional search, and hybrid search.
"""

from typing import Optional

from fastapi import APIRouter, Query, Depends

from src.api.dependencies import get_embedding_service, get_repository
from src.services.embedding import EmbeddingService
from src.repositories.painting_repository import PaintingRepository
from src.core.constants import (
    MIN_SIMILARITY_THRESHOLD,
    MIN_VISUAL_SIMILARITY,
    MIN_EMOTION_SIMILARITY,
    EMOTION_CATEGORIES,
    HARD_MIN_SIMILARITY
)


router = APIRouter(prefix="/catalog", tags=["Busqueda"])

# Lowercase lookup for matching queries to emotion category names
_EMOTION_NAME_LOOKUP = {name.lower(): name for name in EMOTION_CATEGORIES}


def _match_emotion_category(query: str) -> Optional[str]:
    """Return the emotion category name if the query matches one, else None."""
    return _EMOTION_NAME_LOOKUP.get(query.strip().lower())


@router.get("/search")
async def search_by_style(
    estilo: str = Query(..., description="Style to search"),
    min_confianza: float = Query(0.0, description="Minimum confidence (0-1)"),
    repository: PaintingRepository = Depends(get_repository)
):
    """
    Search paintings by style.

    - **estilo**: Style name (Paisaje, Marino, Abstracto, etc.)
    - **min_confianza**: Filter by minimum confidence
    """
    paintings = repository.search_by_style(estilo, min_confianza)

    return {
        "estilo": estilo,
        "total": len(paintings),
        "pinturas": paintings
    }


@router.get("/debug-similarity")
async def debug_similarity(
    query: str = Query(..., description="Text query"),
    painting_id: str = Query(..., description="Painting ID to check"),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    repository: PaintingRepository = Depends(get_repository)
):
    """
    Debug endpoint: check similarity between a query and specific painting.
    """
    query_embedding = embedding_service.get_text_embedding(query)
    result = repository.get_similarity_for_painting(query_embedding, painting_id)

    if not result:
        return {"error": "Painting not found or has no embedding"}

    return {
        "query": query,
        "painting_id": painting_id,
        "archivo": result["archivo"],
        "estilo": result["estilo_principal"],
        "similitud": round(result["similitud"] * 100, 2),
        "rank": result.get("rank")
    }


@router.get("/semantic-search")
async def semantic_search(
    query: str = Query(..., description="Description of what you're looking for"),
    limit: int = Query(100, description="Maximum number of results"),
    min_similarity: float = Query(MIN_SIMILARITY_THRESHOLD, description="Minimum similarity threshold (0-1, default 28%)"),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    repository: PaintingRepository = Depends(get_repository)
):
    """
    Semantic search by natural language description.

    Examples:
    - "flores coloridas"
    - "paisaje con montanas y atardecer"
    - "arte abstracto azul y dorado"
    - "retrato de mujer elegante"
    """
    # Enforce hard minimum - frontend cannot bypass this floor
    min_similarity = max(min_similarity, HARD_MIN_SIMILARITY)

    query_embedding = embedding_service.get_text_embedding(query)

    # Debug: verificar norma del embedding (debe ser ~1.0 si está normalizado)
    embedding_norm = sum(x**2 for x in query_embedding) ** 0.5
    print(f"[semantic-search] Query: '{query}' | Embedding norm: {embedding_norm:.4f} | Threshold: {min_similarity}")

    results = repository.semantic_search(query_embedding, limit, min_similarity)
    print(f"[semantic-search] Resultados encontrados: {len(results)}")

    return {
        "query": query,
        "total": len(results),
        "resultados": [
            {
                "id": str(r["id"]),
                "archivo": r["archivo"],
                "ruta": r["ruta"],
                "estilo": r["estilo_principal"],
                "confianza_estilo": round(r["confianza"] * 100, 1),
                "similitud_busqueda": round(r["similitud"] * 100, 1)
            }
            for r in results
        ]
    }


@router.get("/smart-search")
async def smart_search(
    query: str = Query(..., description="Natural language query (content + emotion)"),
    limit: int = Query(20, description="Maximum number of results"),
    content_weight: float = Query(0.6, description="Weight for content similarity (0-1)"),
    emotion_weight: float = Query(0.4, description="Weight for emotion similarity (0-1)"),
    min_content_similarity: float = Query(MIN_VISUAL_SIMILARITY, description="Minimum content similarity"),
    min_emotion_similarity: float = Query(MIN_EMOTION_SIMILARITY, description="Minimum emotion similarity"),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    repository: PaintingRepository = Depends(get_repository)
):
    """
    Smart hybrid search that automatically detects content vs emotion.

    Parses queries like:
    - "caballo que inspire libertad" → content: caballo, emotion: libertad
    - "flores coloridas con energía" → content: flores coloridas, emotion: energía
    - "paisaje montañoso" → content only (no emotion detected)
    - "libertad y aventura" → emotion only (no specific content)

    The search FIRST filters by visual content, THEN ranks by emotional similarity.
    This ensures you get relevant content that also matches the desired emotion.

    Examples:
    - "vocho azul vintage que inspire libertad y aventura"
    - "caballo corriendo con energía"
    - "flores que transmitan paz"
    - "paisaje que evoque nostalgia"
    """
    # Enforce hard minimum - frontend cannot bypass this floor
    min_content_similarity = max(min_content_similarity, HARD_MIN_SIMILARITY)

    # Parse query to separate content from emotion
    content_query, emotion_query, has_emotion = embedding_service.parse_hybrid_query(query)

    print(f"[smart-search] Query: '{query}'")
    print(f"[smart-search] Contenido: '{content_query}' | Emoción: '{emotion_query}' | Has emotion: {has_emotion}")

    if has_emotion and content_query:
        # HYBRID: Content + Emotion (two-step search)
        content_embedding = embedding_service.get_text_embedding(content_query)
        emotion_embedding = embedding_service.get_emotion_text_embedding(emotion_query)

        results = repository.search_by_content_then_emotion(
            content_embedding=content_embedding,
            emotion_embedding=emotion_embedding,
            content_limit=50,
            final_limit=limit,
            min_content_similarity=min_content_similarity,
            min_emotion_similarity=min_emotion_similarity,
            content_weight=content_weight,
            emotion_weight=emotion_weight,
            emotion_name=_match_emotion_category(emotion_query)
        )

        return {
            "query": query,
            "tipo_busqueda": "hibrida",
            "contenido_buscado": content_query,
            "emocion_buscada": emotion_query,
            "total": len(results),
            "resultados": [
                {
                    "id": str(r["id"]),
                    "archivo": r["archivo"],
                    "ruta": r["ruta"],
                    "estilo": r["estilo_principal"],
                    "emocion": r.get("emocion_principal"),
                    "similitud_contenido": round(r["similitud_contenido"] * 100, 1),
                    "similitud_emocion": round(r["similitud_emocion"] * 100, 1),
                    "score_combinado": round(r["score_combinado"] * 100, 1)
                }
                for r in results
            ]
        }

    elif has_emotion and not content_query:
        # EMOTION ONLY search
        resolved_emotion = emotion_query or query
        emotion_embedding = embedding_service.get_emotion_text_embedding(resolved_emotion)

        results = repository.search_by_emotion(
            emotion_embedding=emotion_embedding,
            limit=limit,
            min_similarity=min_emotion_similarity,
            emotion_name=_match_emotion_category(resolved_emotion)
        )

        return {
            "query": query,
            "tipo_busqueda": "emocion",
            "contenido_buscado": "",
            "emocion_buscada": emotion_query or query,
            "total": len(results),
            "resultados": [
                {
                    "id": str(r["id"]),
                    "archivo": r["archivo"],
                    "ruta": r["ruta"],
                    "estilo": r["estilo_principal"],
                    "emocion": r.get("emocion_principal"),
                    "similitud_contenido": 0,
                    "similitud_emocion": round(r["similitud_emocion"] * 100, 1),
                    "score_combinado": round(r["similitud_emocion"] * 100, 1)
                }
                for r in results
            ]
        }

    else:
        # CONTENT ONLY search (fallback to semantic search)
        query_embedding = embedding_service.get_text_embedding(query)
        results = repository.semantic_search(
            query_embedding,
            limit=limit,
            min_similarity=min_content_similarity
        )

        return {
            "query": query,
            "tipo_busqueda": "contenido",
            "contenido_buscado": query,
            "emocion_buscada": "",
            "total": len(results),
            "resultados": [
                {
                    "id": str(r["id"]),
                    "archivo": r["archivo"],
                    "ruta": r["ruta"],
                    "estilo": r["estilo_principal"],
                    "emocion": None,
                    "similitud_contenido": round(r["similitud"] * 100, 1),
                    "similitud_emocion": 0,
                    "score_combinado": round(r["similitud"] * 100, 1)
                }
                for r in results
            ]
        }


@router.get("/emotion-search")
async def emotion_search(
    query: str = Query(..., description="Emotional description (e.g., 'libertad', 'paz y tranquilidad')"),
    limit: int = Query(20, description="Maximum number of results"),
    min_similarity: float = Query(MIN_EMOTION_SIMILARITY, description="Minimum emotion similarity"),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    repository: PaintingRepository = Depends(get_repository)
):
    """
    Search paintings by the emotion they evoke.

    Examples:
    - "libertad y aventura"
    - "paz y tranquilidad"
    - "energía y dinamismo"
    - "nostalgia"
    - "alegría"
    """
    # Enforce hard minimum for emotion search
    min_similarity = max(min_similarity, HARD_MIN_SIMILARITY)

    emotion_embedding = embedding_service.get_emotion_text_embedding(query)

    results = repository.search_by_emotion(
        emotion_embedding=emotion_embedding,
        limit=limit,
        min_similarity=min_similarity,
        emotion_name=_match_emotion_category(query)
    )

    return {
        "query": query,
        "total": len(results),
        "resultados": [
            {
                "id": str(r["id"]),
                "archivo": r["archivo"],
                "ruta": r["ruta"],
                "estilo": r["estilo_principal"],
                "emocion_principal": r.get("emocion_principal"),
                "similitud_emocion": round(r["similitud_emocion"] * 100, 1)
            }
            for r in results
        ]
    }


@router.get("/emociones")
async def list_emotions():
    """
    List all available emotion categories.
    """
    return {
        "emociones": list(EMOTION_CATEGORIES.keys()),
        "total": len(EMOTION_CATEGORIES)
    }
