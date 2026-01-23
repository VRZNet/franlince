"""
Catalog routes for uploading and managing paintings.
"""

import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, Query, Depends, Response

from src.api.dependencies import get_classifier, get_repository
from src.core.constants import ALLOWED_IMAGE_TYPES, MEDIA_TYPES
from src.services.classifier import CLIPClassifier
from src.repositories.painting_repository import PaintingRepository


router = APIRouter(prefix="/catalog", tags=["Catalogacion"])


@router.post("/upload")
async def upload_painting(
    file: UploadFile = File(...),
    classifier: CLIPClassifier = Depends(get_classifier),
    repository: PaintingRepository = Depends(get_repository)
):
    """
    Upload an image, classify it automatically, and save to catalog.

    - **file**: Painting image (JPG, PNG, etc.)

    Returns the painting ID and classification.
    """
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed: {file.content_type}"
        )

    try:
        image_bytes = await file.read()

        extension = Path(file.filename).suffix or ".jpg"
        unique_filename = f"{uuid.uuid4()}{extension}"

        classification = classifier.classify_from_bytes(image_bytes)

        painting_id = repository.save(unique_filename, image_bytes, classification)

        return {
            "success": True,
            "message": "Pintura catalogada exitosamente",
            "data": {
                "id": painting_id,
                "archivo": unique_filename,
                "estilo_principal": classification["estilo_principal"],
                "confianza": round(classification["confianza"] * 100, 1),
                "top_estilos": [
                    {
                        "estilo": e["estilo"],
                        "confianza": round(e["confianza"] * 100, 1)
                    }
                    for e in classification["top_estilos"]
                ]
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-batch")
async def upload_paintings_batch(
    files: List[UploadFile] = File(...),
    classifier: CLIPClassifier = Depends(get_classifier),
    repository: PaintingRepository = Depends(get_repository)
):
    """
    Upload multiple images and catalog them all.

    - **files**: List of painting images

    Returns summary of all classifications.
    """
    results = []
    errors = []

    for file in files:
        try:
            if file.content_type not in ALLOWED_IMAGE_TYPES:
                errors.append({
                    "archivo": file.filename,
                    "error": "Tipo no permitido"
                })
                continue

            image_bytes = await file.read()

            extension = Path(file.filename).suffix or ".jpg"
            unique_filename = f"{uuid.uuid4()}{extension}"

            classification = classifier.classify_from_bytes(image_bytes)

            painting_id = repository.save(
                unique_filename, image_bytes, classification
            )

            results.append({
                "id": painting_id,
                "archivo_original": file.filename,
                "archivo_guardado": unique_filename,
                "estilo": classification["estilo_principal"],
                "confianza": round(classification["confianza"] * 100, 1)
            })

        except Exception as e:
            errors.append({"archivo": file.filename, "error": str(e)})

    return {
        "success": True,
        "total_procesadas": len(results),
        "total_errores": len(errors),
        "pinturas": results,
        "errores": errors
    }


@router.get("/paintings")
async def list_paintings(
    estilo: Optional[str] = Query(None, description="Filter by style"),
    limit: int = Query(50, description="Result limit"),
    offset: int = Query(0, description="Pagination offset"),
    repository: PaintingRepository = Depends(get_repository)
):
    """
    List all paintings in the catalog.

    - **estilo**: Filter by style (optional)
    - **limit**: Maximum results (default: 50)
    - **offset**: For pagination
    """
    paintings, total = repository.list_all(estilo=estilo, limit=limit, offset=offset)

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "pinturas": paintings
    }


@router.get("/painting/{painting_id}")
async def get_painting(
    painting_id: str,
    repository: PaintingRepository = Depends(get_repository)
):
    """
    Get details of a specific painting.

    - **painting_id**: UUID of the painting
    """
    painting = repository.get_by_id(painting_id)

    if not painting:
        raise HTTPException(status_code=404, detail="Pintura no encontrada")

    return painting


@router.get("/painting/{painting_id}/image")
async def get_painting_image(
    painting_id: str,
    repository: PaintingRepository = Depends(get_repository)
):
    """
    Get the image of a painting from the database.

    - **painting_id**: UUID of the painting

    Returns the image file.
    """
    result = repository.get_image(painting_id)

    if not result or not result.get("imagen"):
        raise HTTPException(status_code=404, detail="Imagen no encontrada")

    extension = Path(result["archivo"]).suffix.lower()
    media_type = MEDIA_TYPES.get(extension, "image/jpeg")

    return Response(
        content=bytes(result["imagen"]),
        media_type=media_type,
        headers={
            "Content-Disposition": f"inline; filename={result['archivo']}"
        }
    )


@router.delete("/painting/{painting_id}")
async def delete_painting(
    painting_id: str,
    repository: PaintingRepository = Depends(get_repository)
):
    """
    Delete a painting from the catalog.

    - **painting_id**: UUID of the painting to delete
    """
    painting = repository.delete(painting_id)

    if not painting:
        raise HTTPException(status_code=404, detail="Pintura no encontrada")

    return {
        "success": True,
        "message": f"Pintura {painting_id} eliminada",
        "archivo": painting["archivo"]
    }
