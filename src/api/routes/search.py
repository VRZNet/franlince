"""
Search routes for finding paintings.
"""

from fastapi import APIRouter, Query, Depends

from src.api.dependencies import get_embedding_service, get_repository
from src.services.embedding import EmbeddingService
from src.repositories.painting_repository import PaintingRepository
from src.core.constants import MIN_SIMILARITY_THRESHOLD


router = APIRouter(prefix="/catalog", tags=["Busqueda"])


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
