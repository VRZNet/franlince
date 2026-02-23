"""
CLIP-based art style classifier service.
"""

import math
from typing import Optional, Dict, List

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from src.core.config import get_settings
from src.core.constants import (
    STYLE_CATEGORIES,
    EMOTION_CATEGORIES,
    CLASSIFICATION_TEMPERATURE,
    EMOTION_AGGREGATION_METHOD,
    EMOTION_TOP_K,
    EMOTION_CLASSIFICATION_TEMPERATURE
)
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
        # Store individual prompt embeddings for emotions (not averaged)
        self.emotion_prompt_embeddings: Dict[str, torch.Tensor] = {}
        # Store averaged embeddings for backward compatibility
        self.emotion_embeddings: Dict[str, torch.Tensor] = {}
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
        """Load CLIP model and pre-compute category and emotion embeddings."""
        if self._is_loaded:
            return

        print(f"Loading CLIP model: {self.model_name}...")
        self.model = CLIPModel.from_pretrained(self.model_name)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)

        self._precompute_category_embeddings()
        self._precompute_emotion_embeddings()
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

    def _precompute_emotion_embeddings(self) -> None:
        """Pre-compute text embeddings for all emotion categories.

        Stores both individual prompt embeddings (for top-k scoring)
        and averaged embeddings (for backward compatibility).
        """
        for emotion_name, prompts in EMOTION_CATEGORIES.items():
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

                # Store individual prompt embeddings (N x 512)
                self.emotion_prompt_embeddings[emotion_name] = text_features

                # Also compute and store averaged embedding for compatibility
                avg_embedding = text_features.mean(dim=0, keepdim=True)
                avg_embedding = avg_embedding / avg_embedding.norm(
                    p=2, dim=-1, keepdim=True
                )
                self.emotion_embeddings[emotion_name] = avg_embedding

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

    def classify_emotions(self, image_features: torch.Tensor) -> List[dict]:
        """
        Classify emotions for an image given its features.

        Uses configurable aggregation method (top_k_max by default) to
        compute category scores from individual prompt similarities.

        Args:
            image_features: Normalized image features from CLIP.

        Returns:
            List of dicts with emocion, confianza, and raw_score, sorted by confidence.
        """
        emotion_similarities: Dict[str, float] = {}

        for emotion_name, prompt_embeddings in self.emotion_prompt_embeddings.items():
            # Calculate similarity with each prompt in this category
            # image_features: (1, 512), prompt_embeddings: (N, 512)
            similarities = (image_features @ prompt_embeddings.T).squeeze()

            # Handle single prompt case
            if similarities.dim() == 0:
                similarities = similarities.unsqueeze(0)

            # Aggregate based on method
            if EMOTION_AGGREGATION_METHOD == "max":
                score = similarities.max().item()

            elif EMOTION_AGGREGATION_METHOD == "mean":
                score = similarities.mean().item()

            elif EMOTION_AGGREGATION_METHOD == "top_k_mean":
                k = min(EMOTION_TOP_K, len(similarities))
                top_k_sims = torch.topk(similarities, k).values
                score = top_k_sims.mean().item()

            elif EMOTION_AGGREGATION_METHOD == "top_k_max":
                # Use max of top-k (most discriminative)
                k = min(EMOTION_TOP_K, len(similarities))
                top_k_sims = torch.topk(similarities, k).values
                # Weight: first prompt counts more
                weights = torch.tensor([1.0, 0.7, 0.5][:k])
                weights = weights / weights.sum()
                score = (top_k_sims * weights).sum().item()

            else:
                # Fallback to max
                score = similarities.max().item()

            emotion_similarities[emotion_name] = score

        # Convert to probabilities with softmax
        # Use configurable temperature for emotion classification
        emotion_temperature = EMOTION_CLASSIFICATION_TEMPERATURE
        exp_sims = {
            k: math.exp(v / emotion_temperature)
            for k, v in emotion_similarities.items()
        }
        total = sum(exp_sims.values())
        probs = {k: v / total for k, v in exp_sims.items()}

        # Sort results
        results = [
            {
                "emocion": k,
                "confianza": v,
                "raw_score": emotion_similarities[k]
            }
            for k, v in probs.items()
        ]
        results.sort(key=lambda x: x["confianza"], reverse=True)

        return results

    def get_emotion_embedding(self, image_features: torch.Tensor) -> List[float]:
        """
        Generate an emotion-weighted embedding for an image.

        Takes the image features and creates a weighted combination
        of the top emotion embeddings based on their similarity scores.

        Args:
            image_features: Normalized image features from CLIP.

        Returns:
            List of embedding values (512 dimensions).
        """
        # Get emotion classifications
        emotion_results = self.classify_emotions(image_features)
        top_emotions = emotion_results[:5]

        # Create weighted combination of emotion embeddings
        weighted_embedding = torch.zeros(1, 512)
        total_weight = 0

        for emotion_data in top_emotions:
            emotion_name = emotion_data["emocion"]
            weight = emotion_data["confianza"]

            if weight > 0 and emotion_name in self.emotion_embeddings:
                weighted_embedding += self.emotion_embeddings[emotion_name] * weight
                total_weight += weight

        if total_weight > 0:
            weighted_embedding = weighted_embedding / total_weight
            weighted_embedding = weighted_embedding / weighted_embedding.norm(
                p=2, dim=-1, keepdim=True
            )

        return weighted_embedding.squeeze().tolist()

    def classify_image(self, image: Image.Image) -> dict:
        """
        Classify an image and return style, emotions, and embeddings.

        Args:
            image: PIL Image to classify.

        Returns:
            Dict with estilo_principal, confianza, top_estilos,
            todos_estilos, embedding, top_emociones, todas_emociones,
            and embedding_emocional.
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

        # Calculate similarity with each style category
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

        # Sort style results
        style_results = [
            {"estilo": k, "confianza": v}
            for k, v in probs.items()
        ]
        style_results.sort(key=lambda x: x["confianza"], reverse=True)

        # Classify emotions
        emotion_results = self.classify_emotions(image_features)

        # Get embeddings
        visual_embedding = image_features.squeeze().tolist()
        emotion_embedding = self.get_emotion_embedding(image_features)

        return {
            "estilo_principal": style_results[0]["estilo"],
            "confianza": style_results[0]["confianza"],
            "top_estilos": style_results[:3],
            "todos_estilos": style_results,
            "embedding": visual_embedding,
            "top_emociones": emotion_results[:5],
            "todas_emociones": emotion_results,
            "embedding_emocional": emotion_embedding
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
