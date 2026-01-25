"""
Embedding service for generating image and text embeddings.
"""

import re
from typing import Optional, List

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from deep_translator import GoogleTranslator

from src.core.config import get_settings
from src.services.image_processor import ImageProcessor


class EmbeddingService:
    """
    Service for generating embeddings using CLIP.
    Used for semantic search functionality.
    """

    _instance: Optional["EmbeddingService"] = None

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize embedding service.

        Args:
            model_name: HuggingFace model name. Defaults to settings value.
        """
        settings = get_settings()
        self.model_name = model_name or settings.clip_model_name
        self.model: Optional[CLIPModel] = None
        self.processor: Optional[CLIPProcessor] = None
        self.image_processor = ImageProcessor()
        self._is_loaded = False

    @classmethod
    def get_instance(cls) -> "EmbeddingService":
        """Get singleton instance of embedding service."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    def load_model(self) -> None:
        """Load CLIP model for embedding generation."""
        if self._is_loaded:
            return

        print(f"Loading CLIP model for embeddings: {self.model_name}...")
        self.model = CLIPModel.from_pretrained(self.model_name)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self._is_loaded = True
        print("Embedding model loaded successfully")

    def get_image_embedding(self, image: Image.Image) -> List[float]:
        """
        Generate embedding for an image.

        Args:
            image: PIL Image to embed.

        Returns:
            List of embedding values (512 dimensions).
        """
        if not self._is_loaded:
            self.load_model()

        prepared_image = self.image_processor.prepare_for_model(image)
        inputs = self.processor(images=prepared_image, return_tensors="pt")

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(
                p=2, dim=-1, keepdim=True
            )

        return image_features.squeeze().tolist()

    def get_image_embedding_from_path(self, image_path: str) -> List[float]:
        """
        Generate embedding for an image from file path.

        Args:
            image_path: Path to image file.

        Returns:
            List of embedding values.
        """
        image = self.image_processor.load_from_path(image_path)
        return self.get_image_embedding(image)

    def get_image_embedding_from_bytes(self, image_bytes: bytes) -> List[float]:
        """
        Generate embedding for an image from bytes.

        Args:
            image_bytes: Raw image bytes.

        Returns:
            List of embedding values.
        """
        image = self.image_processor.load_from_bytes(image_bytes)
        return self.get_image_embedding(image)

    def _is_spanish(self, text: str) -> bool:
        """
        Detect if text contains Spanish characters or common Spanish words.

        Args:
            text: Text to analyze.

        Returns:
            True if text appears to be in Spanish.
        """
        spanish_chars = re.compile(r'[áéíóúüñ¿¡]', re.IGNORECASE)
        if spanish_chars.search(text):
            return True

        spanish_words = [
            'con', 'del', 'las', 'los', 'una', 'uno', 'para', 'por',
            'sobre', 'entre', 'hacia', 'desde', 'hasta', 'durante',
            'mediante', 'según', 'sin', 'bajo', 'contra', 'ante',
            'flores', 'paisaje', 'montañas', 'caballos', 'personas',
            'casa', 'árbol', 'cielo', 'mar', 'río', 'bosque', 'campo',
            'noche', 'día', 'sol', 'luna', 'estrellas', 'nubes',
            'colores', 'azul', 'rojo', 'verde', 'amarillo', 'blanco',
            'negro', 'abstracto', 'moderno', 'antiguo', 'clásico'
        ]

        text_lower = text.lower()
        words = text_lower.split()

        for word in words:
            if word in spanish_words:
                return True

        return False

    def _translate_to_english(self, text: str) -> str:
        """
        Translate Spanish text to English.

        Args:
            text: Text to translate.

        Returns:
            Translated text in English.
        """
        try:
            translator = GoogleTranslator(source='es', target='en')
            translated = translator.translate(text)
            return translated if translated else text
        except Exception:
            return text

    def get_text_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text query.
        Automatically translates Spanish to English for better CLIP results.

        Args:
            text: Text to embed (Spanish or English).

        Returns:
            List of embedding values (512 dimensions).
        """
        if not self._is_loaded:
            self.load_model()

        # Traducir español a inglés si es necesario
        if self._is_spanish(text):
            text_en = self._translate_to_english(text)
            print(f"Traducción: '{text}' → '{text_en}'")
        else:
            text_en = text

        # Usar un prompt que le de más contexto al modelo
        prompt = f"a painting of {text_en}"

        inputs = self.processor(
            text=[prompt],
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(
                p=2, dim=-1, keepdim=True
            )

        return text_features.squeeze().tolist()

    @staticmethod
    def embedding_to_pg_format(embedding: List[float]) -> str:
        """
        Convert embedding to PostgreSQL vector format.

        Args:
            embedding: List of float values.

        Returns:
            String in PostgreSQL vector format "[x,y,z,...]"
        """
        return "[" + ",".join(map(str, embedding)) + "]"
