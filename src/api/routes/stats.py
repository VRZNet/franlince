"""
Statistics and utility routes.
"""

from datetime import datetime

from fastapi import APIRouter, Depends

from src.api.dependencies import get_classifier, get_repository
from src.core.constants import STYLE_CATEGORIES
from src.services.classifier import CLIPClassifier
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


@router.get("/catalog/estilos")
async def list_estilos():
    """
    List all available styles.
    """
    return {
        "estilos": list(STYLE_CATEGORIES.keys())
    }


@router.get("/health", tags=["Sistema"])
async def health_check(
    classifier: CLIPClassifier = Depends(get_classifier)
):
    """
    Verify that the service is running.
    """
    return {
        "status": "ok",
        "model_loaded": classifier.is_loaded,
        "timestamp": datetime.now().isoformat()
    }
