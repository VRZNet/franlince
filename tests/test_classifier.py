#!/usr/bin/env python3
"""
Tests for the CLIP classifier service.

Run with: pytest tests/test_classifier.py -v
"""

import sys
from pathlib import Path

import pytest
from PIL import Image
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.constants import STYLE_CATEGORIES
from src.services.classifier import CLIPClassifier
from src.services.image_processor import ImageProcessor


class TestImageProcessor:
    """Tests for ImageProcessor."""

    def test_resize_large_image(self):
        """Test that large images are resized."""
        processor = ImageProcessor(max_size=512)

        # Create a large test image
        large_image = Image.new("RGB", (2000, 1500), color="red")

        resized = processor.resize_image(large_image)

        assert max(resized.size) <= 512
        assert resized.size[0] == 512 or resized.size[1] == 512

    def test_small_image_unchanged(self):
        """Test that small images are not resized."""
        processor = ImageProcessor(max_size=1024)

        small_image = Image.new("RGB", (500, 400), color="blue")

        result = processor.resize_image(small_image)

        assert result.size == (500, 400)

    def test_load_from_bytes(self):
        """Test loading image from bytes."""
        processor = ImageProcessor()

        # Create a test image and convert to bytes
        test_image = Image.new("RGB", (100, 100), color="green")
        from io import BytesIO
        buffer = BytesIO()
        test_image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        loaded = processor.load_from_bytes(image_bytes)

        assert loaded.mode == "RGB"
        assert loaded.size == (100, 100)


class TestCLIPClassifier:
    """Tests for CLIPClassifier."""

    @pytest.fixture
    def classifier(self):
        """Get classifier instance."""
        return CLIPClassifier.get_instance()

    def test_singleton_pattern(self):
        """Test that classifier is a singleton."""
        c1 = CLIPClassifier.get_instance()
        c2 = CLIPClassifier.get_instance()

        assert c1 is c2

    def test_style_categories_exist(self):
        """Test that style categories are defined."""
        assert len(STYLE_CATEGORIES) > 0
        assert "Paisaje" in STYLE_CATEGORIES
        assert "Abstracto" in STYLE_CATEGORIES

    @pytest.mark.slow
    def test_model_loads(self, classifier):
        """Test that model loads correctly."""
        classifier.load_model()

        assert classifier.is_loaded
        assert classifier.model is not None
        assert classifier.processor is not None
        assert len(classifier.category_embeddings) == len(STYLE_CATEGORIES)

    @pytest.mark.slow
    def test_classify_image(self, classifier):
        """Test image classification."""
        classifier.load_model()

        # Create a test image
        test_image = Image.new("RGB", (224, 224), color="blue")

        result = classifier.classify_image(test_image)

        assert "estilo_principal" in result
        assert "confianza" in result
        assert "top_estilos" in result
        assert "todos_estilos" in result
        assert "embedding" in result

        assert result["estilo_principal"] in STYLE_CATEGORIES
        assert 0 <= result["confianza"] <= 1
        assert len(result["top_estilos"]) == 3
        assert len(result["todos_estilos"]) == len(STYLE_CATEGORIES)
        assert len(result["embedding"]) == 512

    @pytest.mark.slow
    def test_get_image_embedding(self, classifier):
        """Test image embedding generation."""
        classifier.load_model()

        test_image = Image.new("RGB", (224, 224), color="red")

        embedding = classifier.get_image_embedding(test_image)

        assert len(embedding) == 512
        assert all(isinstance(v, float) for v in embedding)


class TestDebugClassifier:
    """Debug tests for classifier (optional manual testing)."""

    @pytest.mark.skip(reason="Manual test - requires images folder")
    def test_classify_real_image(self):
        """Classify a real image for manual verification."""
        images_folder = Path("./pinturas")

        if not images_folder.exists():
            pytest.skip("Images folder not found")

        images = list(images_folder.glob("*.jpg")) + list(images_folder.glob("*.png"))

        if not images:
            pytest.skip("No images found")

        classifier = CLIPClassifier.get_instance()
        classifier.load_model()

        result = classifier.classify_from_path(str(images[0]))

        print(f"\nImage: {images[0].name}")
        print(f"Style: {result['estilo_principal']} ({result['confianza']*100:.1f}%)")
        print("Top 3:")
        for style in result["top_estilos"]:
            print(f"  - {style['estilo']}: {style['confianza']*100:.1f}%")
