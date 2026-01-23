#!/usr/bin/env python3
"""
Integration tests for the API.

Run with: pytest tests/test_api.py -v
"""

import sys
from pathlib import Path
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def test_image_bytes():
    """Create test image bytes."""
    img = Image.new("RGB", (100, 100), color="red")
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self, client):
        """Test health endpoint returns ok status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "model_loaded" in data
        assert "timestamp" in data


class TestStylesEndpoint:
    """Tests for styles listing endpoint."""

    def test_list_estilos(self, client):
        """Test listing available styles."""
        response = client.get("/catalog/estilos")

        assert response.status_code == 200
        data = response.json()
        assert "estilos" in data
        assert len(data["estilos"]) > 0
        assert "Paisaje" in data["estilos"]
        assert "Abstracto" in data["estilos"]


class TestStatsEndpoint:
    """Tests for statistics endpoint."""

    def test_get_stats(self, client):
        """Test getting catalog statistics."""
        response = client.get("/catalog/stats")

        assert response.status_code == 200
        data = response.json()
        assert "total_pinturas" in data
        assert "con_embeddings" in data
        assert "por_estilo" in data


class TestPaintingsEndpoint:
    """Tests for paintings list endpoint."""

    def test_list_paintings(self, client):
        """Test listing paintings."""
        response = client.get("/catalog/paintings")

        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "limit" in data
        assert "offset" in data
        assert "pinturas" in data

    def test_list_paintings_with_limit(self, client):
        """Test listing paintings with limit."""
        response = client.get("/catalog/paintings?limit=10&offset=0")

        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 10
        assert data["offset"] == 0


class TestSearchEndpoint:
    """Tests for search endpoint."""

    def test_search_by_style(self, client):
        """Test searching by style."""
        response = client.get("/catalog/search?estilo=Paisaje")

        assert response.status_code == 200
        data = response.json()
        assert data["estilo"] == "Paisaje"
        assert "total" in data
        assert "pinturas" in data


class TestSemanticSearchEndpoint:
    """Tests for semantic search endpoint."""

    @pytest.mark.slow
    def test_semantic_search(self, client):
        """Test semantic search."""
        response = client.get("/catalog/semantic-search?query=flores%20coloridas")

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "flores coloridas"
        assert "total" in data
        assert "resultados" in data


class TestUploadEndpoint:
    """Tests for upload endpoint."""

    @pytest.mark.slow
    def test_upload_painting(self, client, test_image_bytes):
        """Test uploading a painting."""
        response = client.post(
            "/catalog/upload",
            files={"file": ("test.jpg", test_image_bytes, "image/jpeg")}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert "id" in data["data"]
        assert "estilo_principal" in data["data"]

    def test_upload_invalid_type(self, client):
        """Test uploading invalid file type."""
        response = client.post(
            "/catalog/upload",
            files={"file": ("test.txt", b"not an image", "text/plain")}
        )

        assert response.status_code == 400


class TestPaintingDetailEndpoint:
    """Tests for painting detail endpoint."""

    def test_get_painting_not_found(self, client):
        """Test getting non-existent painting."""
        response = client.get("/catalog/painting/00000000-0000-0000-0000-000000000000")

        assert response.status_code == 404


class TestDeleteEndpoint:
    """Tests for delete endpoint."""

    def test_delete_painting_not_found(self, client):
        """Test deleting non-existent painting."""
        response = client.delete("/catalog/painting/00000000-0000-0000-0000-000000000000")

        assert response.status_code == 404
