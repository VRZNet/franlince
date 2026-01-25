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


@router.get("/semantic-search")
async def semantic_search(
    query: str = Query(..., description="Description of what you're looking for"),
    limit: int = Query(10, description="Number of results"),
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

    results = repository.semantic_search(query_embedding, limit, min_similarity)

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
