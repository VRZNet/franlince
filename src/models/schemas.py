"""
Pydantic schemas for API request/response validation.
"""

from datetime import datetime
from typing import Optional, List
from uuid import UUID

from pydantic import BaseModel, Field


# Style classification schemas
class StyleScoreSchema(BaseModel):
    """Style classification score."""
    estilo: str
    confianza: float


class ClassificationResult(BaseModel):
    """Result of image classification."""
    estilo_principal: str
    confianza: float
    top_estilos: List[StyleScoreSchema]
    todos_estilos: List[StyleScoreSchema]
    embedding: List[float]


# Painting schemas
class PaintingBase(BaseModel):
    """Base painting schema."""
    archivo: str
    estilo_principal: str
    confianza: float


class PaintingCreate(PaintingBase):
    """Schema for creating a painting."""
    ruta: Optional[str] = None
    estilo_2: Optional[str] = None
    confianza_2: Optional[float] = None
    estilo_3: Optional[str] = None
    confianza_3: Optional[float] = None
    todos_estilos: Optional[List[StyleScoreSchema]] = None
    embedding: Optional[List[float]] = None


class PaintingResponse(BaseModel):
    """Schema for painting response."""
    id: UUID
    archivo: str
    ruta: Optional[str] = None
    estilo_principal: str
    confianza: float
    estilo_2: Optional[str] = None
    confianza_2: Optional[float] = None
    estilo_3: Optional[str] = None
    confianza_3: Optional[float] = None
    todos_estilos: Optional[List[StyleScoreSchema]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class PaintingListItem(BaseModel):
    """Schema for painting list item (without all styles)."""
    id: UUID
    archivo: str
    ruta: Optional[str] = None
    estilo_principal: str
    confianza: float
    estilo_2: Optional[str] = None
    confianza_2: Optional[float] = None
    estilo_3: Optional[str] = None
    confianza_3: Optional[float] = None
    created_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


# Upload response schemas
class UploadResultData(BaseModel):
    """Data from a successful upload."""
    id: str
    archivo: str
    estilo_principal: str
    confianza: float = Field(description="Confidence as percentage (0-100)")
    top_estilos: List[StyleScoreSchema]


class UploadResponse(BaseModel):
    """Response for single image upload."""
    success: bool
    message: str
    data: UploadResultData


class BatchUploadItem(BaseModel):
    """Single item in batch upload results."""
    id: str
    archivo_original: str
    archivo_guardado: str
    estilo: str
    confianza: float


class BatchUploadError(BaseModel):
    """Error in batch upload."""
    archivo: str
    error: str


class BatchUploadResponse(BaseModel):
    """Response for batch upload."""
    success: bool
    total_procesadas: int
    total_errores: int
    pinturas: List[BatchUploadItem]
    errores: List[BatchUploadError]


# Search schemas
class SearchRequest(BaseModel):
    """Request for search."""
    estilo: str
    min_confianza: float = 0.0


class SearchResultItem(BaseModel):
    """Single search result item."""
    id: str
    archivo: str
    ruta: Optional[str] = None
    estilo: str
    confianza_estilo: float
    similitud_busqueda: Optional[float] = None


class SearchResponse(BaseModel):
    """Response for search."""
    estilo: Optional[str] = None
    query: Optional[str] = None
    total: int
    pinturas: Optional[List[PaintingListItem]] = None
    resultados: Optional[List[SearchResultItem]] = None


class SemanticSearchResponse(BaseModel):
    """Response for semantic search."""
    query: str
    total: int
    resultados: List[SearchResultItem]


# Stats schemas
class StyleStats(BaseModel):
    """Statistics for a single style."""
    estilo_principal: str
    cantidad: int
    confianza_promedio: float


class StatsResponse(BaseModel):
    """Response for catalog statistics."""
    total_pinturas: int
    con_embeddings: int
    ultima_actualizacion: Optional[str] = None
    por_estilo: List[StyleStats]


# List response schemas
class PaintingListResponse(BaseModel):
    """Response for painting list."""
    total: int
    limit: int
    offset: int
    pinturas: List[PaintingListItem]


# Styles list response
class StylesResponse(BaseModel):
    """Response for available styles."""
    estilos: List[str]


# Health check response
class HealthResponse(BaseModel):
    """Response for health check."""
    status: str
    model_loaded: bool
    timestamp: str


# Delete response
class DeleteResponse(BaseModel):
    """Response for delete operation."""
    success: bool
    message: str
    archivo: str
