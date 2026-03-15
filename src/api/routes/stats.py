"""
Statistics and utility routes.
"""

from fastapi import APIRouter, Depends

from src.api.dependencies import get_repository
from src.repositories.painting_repository import PaintingRepository


router = APIRouter(tags=["Estadisticas"])


@router.get("/catalog/stats")
async def get_stats(
    repository: PaintingRepository = Depends(get_repository)
):
    """
    Get catalog statistics.
    """
    return repository.get_stats()


