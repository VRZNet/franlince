"""
CLIP-based art style classifier service.
"""

import math
from typing import Optional, Dict, List

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from src.core.config import get_settings
from src.core.constants import STYLE_CATEGORIES, CLASSIFICATION_TEMPERATURE
from src.services.image_processor import ImageProcessor


class CLIPClassifier:
    """
    Art style classifier using CLIP model.
    Pre-computes category embeddings for efficient classification.
    """

    _instance: Optional["CLIPClassifier"] = None

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize classifier with CLIP model.

        Args:
            model_name: HuggingFace model name. Defaults to settings value.
        """
        settings = get_settings()
        self.model_name = model_name or settings.clip_model_name
        self.model: Optional[CLIPModel] = None
        self.processor: Optional[CLIPProcessor] = None
        self.category_embeddings: Dict[str, torch.Tensor] = {}
        self.image_processor = ImageProcessor()
        self._is_loaded = False

    @classmethod
    def get_instance(cls) -> "CLIPClassifier":
        """Get singleton instance of classifier."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    def load_model(self) -> None:
        """Load CLIP model and pre-compute category embeddings."""
        if self._is_loaded:
            return

        print(f"Loading CLIP model: {self.model_name}...")
        self.model = CLIPModel.from_pretrained(self.model_name)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)

        self._precompute_category_embeddings()
        self._is_loaded = True
        print("CLIP model loaded successfully")

    def _precompute_category_embeddings(self) -> None:
        """Pre-compute text embeddings for all style categories."""
        for style_name, prompts in STYLE_CATEGORIES.items():
            text_inputs = self.processor(
                text=prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            )

            with torch.no_grad():
                text_features = self.model.get_text_features(**text_inputs)
                text_features = text_features / text_features.norm(
                    p=2, dim=-1, keepdim=True
                )
                avg_embedding = text_features.mean(dim=0, keepdim=True)
                avg_embedding = avg_embedding / avg_embedding.norm(
                    p=2, dim=-1, keepdim=True
                )
                self.category_embeddings[style_name] = avg_embedding

    def get_image_embedding(self, image: Image.Image) -> List[float]:
        """
        Get normalized embedding for an image.

        Args:
            image: PIL Image to embed.

        Returns:
            List of embedding values (512 dimensions).
        """
        if not self._is_loaded:
            self.load_model()

        prepared_image = self.image_processor.prepare_for_model(image)
        image_inputs = self.processor(images=prepared_image, return_tensors="pt")

        with torch.no_grad():
            image_features = self.model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(
                p=2, dim=-1, keepdim=True
            )

        return image_features.squeeze().tolist()

    def classify_image(self, image: Image.Image) -> dict:
        """
        Classify an image and return style with embedding.

        Args:
            image: PIL Image to classify.

        Returns:
            Dict with estilo_principal, confianza, top_estilos,
            todos_estilos, and embedding.
        """
        if not self._is_loaded:
            self.load_model()

        prepared_image = self.image_processor.prepare_for_model(image)
        image_inputs = self.processor(images=prepared_image, return_tensors="pt")

        with torch.no_grad():
            image_features = self.model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(
                p=2, dim=-1, keepdim=True
            )

        # Calculate similarity with each category
        similarities: Dict[str, float] = {}
        for style_name, text_embedding in self.category_embeddings.items():
            similarity = (image_features @ text_embedding.T).squeeze().item()
            similarities[style_name] = similarity

        # Convert to probabilities with softmax
        exp_sims = {
            k: math.exp(v / CLASSIFICATION_TEMPERATURE)
            for k, v in similarities.items()
        }
        total = sum(exp_sims.values())
        probs = {k: v / total for k, v in exp_sims.items()}

        # Sort results
        results = [
            {"estilo": k, "confianza": v}
            for k, v in probs.items()
        ]
        results.sort(key=lambda x: x["confianza"], reverse=True)

        embedding = image_features.squeeze().tolist()

        return {
            "estilo_principal": results[0]["estilo"],
            "confianza": results[0]["confianza"],
            "top_estilos": results[:3],
            "todos_estilos": results,
            "embedding": embedding
        }

    def classify_from_path(self, image_path: str) -> dict:
        """
        Classify an image from file path.

        Args:
            image_path: Path to image file.

        Returns:
            Classification result dict.
        """
        image = self.image_processor.load_from_path(image_path)
        return self.classify_image(image)

    def classify_from_bytes(self, image_bytes: bytes) -> dict:
        """
        Classify an image from bytes.

        Args:
            image_bytes: Raw image bytes.

        Returns:
            Classification result dict.
        """
        image = self.image_processor.load_from_bytes(image_bytes)
        return self.classify_image(image)
