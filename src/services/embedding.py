"""
Embedding service for generating image and text embeddings.
Supports both visual content embeddings and emotional embeddings.
"""

import re
from typing import Optional, List, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from deep_translator import GoogleTranslator

from src.core.config import get_settings
from src.services.image_processor import ImageProcessor
from src.core.constants import EMOTION_CATEGORIES, EMOTION_KEYWORDS_ES


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

    # Patrones conversacionales en español que no aportan contenido visual
    # Nota: (?:\s+(?:de\s+)?)? hace opcional el espacio y "de" al final del patrón
    _QUERY_PREFIXES = [
        r"^busco\s+una?\s+pintura(?:\s+(?:de\s+)?)?",
        r"^busco\s+un\s+cuadro(?:\s+(?:de\s+)?)?",
        r"^busco\s+algo\s+(?:de|sobre|con)\s+",
        r"^busco\s+",
        r"^quiero\s+una?\s+pintura(?:\s+(?:de\s+)?)?",
        r"^quiero\s+un\s+cuadro(?:\s+(?:de\s+)?)?",
        r"^quiero\s+",
        r"^necesito\s+una?\s+pintura(?:\s+(?:de\s+)?)?",
        r"^necesito\s+",
        r"^estoy\s+buscando\s+una?\s+pintura(?:\s+(?:de\s+)?)?",
        r"^estoy\s+buscando\s+",
        r"^quisiera\s+una?\s+pintura(?:\s+(?:de\s+)?)?",
        r"^quisiera\s+",
        r"^me\s+gustaría\s+una?\s+pintura(?:\s+(?:de\s+)?)?",
        r"^me\s+gustaría\s+",
        r"^dame\s+una?\s+pintura(?:\s+(?:de\s+)?)?",
        r"^muéstrame\s+una?\s+pintura(?:\s+(?:de\s+)?)?",
        r"^una?\s+pintura(?:\s+(?:de\s+)?)?",
        r"^un\s+cuadro(?:\s+(?:de\s+)?)?",
        r"\s+para\s+el\s+cuarto\s+de\s+mis\s+hijos?\s*$",
        r"\s+para\s+la\s+habitación\s+de\s+mis\s+hijos?\s*$",
        r"\s+para\s+decorar\s+.*$",
    ]

    def _normalize_query(self, text: str) -> str:
        """
        Elimina frases conversacionales del inicio/fin de la query,
        dejando solo el contenido descriptivo para CLIP.

        Ejemplo: "Busco una pintura de personajes de videojuegos para el cuarto de mis hijos"
                → "personajes de videojuegos"
        """
        normalized = text.strip()
        for pattern in self._QUERY_PREFIXES:
            normalized = re.sub(pattern, "", normalized, flags=re.IGNORECASE).strip()
        result = normalized.strip()
        if result:
            print(f"[normalize_query] '{text}' → '{result}'")
        return result if result else text

    def _translate_to_english(self, text: str) -> str:
        """
        Translate text to English, trying Spanish first then auto-detection.

        Args:
            text: Text to translate (any language).

        Returns:
            Translated text in English.
        """
        try:
            # Primero intentar con español forzado
            translator_es = GoogleTranslator(source='es', target='en')
            translated = translator_es.translate(text)

            # Si la traducción es diferente, usarla
            if translated and translated.lower() != text.lower():
                return translated

            # Si no cambió, intentar con auto-detect
            translator_auto = GoogleTranslator(source='auto', target='en')
            translated = translator_auto.translate(text)
            return translated if translated else text
        except Exception:
            return text

    def get_text_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text query.
        Automatically translates any language to English for better CLIP results.
        Uses multiple prompts and averages them for better semantic matching.

        Args:
            text: Text to embed (any language).

        Returns:
            List of embedding values (512 dimensions).
        """
        if not self._is_loaded:
            self.load_model()

        # Eliminar frases conversacionales antes de embeddear
        text = self._normalize_query(text)

        # Siempre traducir a inglés (auto-detect idioma)
        text_en = self._translate_to_english(text)
        if text_en.lower() != text.lower():
            print(f"Traducción: '{text}' → '{text_en}'")

        # Usar múltiples prompts para capturar tanto contenido como estilo
        prompts = [
            text_en,  # Búsqueda directa del contenido
            f"a {text_en}",  # Con artículo
            f"an image of {text_en}",  # Contexto de imagen
            f"a painting of {text_en}",  # Contexto artístico
        ]

        inputs = self.processor(
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            # Normalizar cada embedding
            text_features = text_features / text_features.norm(
                p=2, dim=-1, keepdim=True
            )
            # Promediar todos los prompts
            avg_features = text_features.mean(dim=0, keepdim=True)
            # Re-normalizar el promedio
            avg_features = avg_features / avg_features.norm(
                p=2, dim=-1, keepdim=True
            )

        return avg_features.squeeze().tolist()

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

    def get_raw_text_embedding(self, text: str) -> List[float]:
        """
        Generate raw embedding for text without prompt variations.
        Used internally for emotion classification.

        Args:
            text: Text to embed (should be in English).

        Returns:
            List of embedding values (512 dimensions).
        """
        if not self._is_loaded:
            self.load_model()

        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(
                p=2, dim=-1, keepdim=True
            )

        return text_features.squeeze().tolist()

    def get_emotion_text_embedding(self, emotion_text: str) -> List[float]:
        """
        Generate embedding optimized for emotional/mood queries.
        Uses prompts that emphasize emotional aspects.

        Args:
            emotion_text: Emotional description (any language).

        Returns:
            List of embedding values (512 dimensions).
        """
        if not self._is_loaded:
            self.load_model()

        text_en = self._translate_to_english(emotion_text)
        if text_en.lower() != emotion_text.lower():
            print(f"Traducción emocional: '{emotion_text}' → '{text_en}'")

        # Prompts optimized for emotional/mood search
        prompts = [
            f"artwork that evokes {text_en}",
            f"painting with {text_en} mood and atmosphere",
            f"art expressing the feeling of {text_en}",
            f"image that inspires {text_en}",
            f"visual art with emotional tone of {text_en}"
        ]

        inputs = self.processor(
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(
                p=2, dim=-1, keepdim=True
            )
            avg_features = text_features.mean(dim=0, keepdim=True)
            avg_features = avg_features / avg_features.norm(
                p=2, dim=-1, keepdim=True
            )

        return avg_features.squeeze().tolist()

    def get_image_emotion_embedding(self, image: Image.Image) -> List[float]:
        """
        Generate emotion-focused embedding for an image.
        Compares the image against emotional prompts to create
        an embedding that captures its emotional essence.

        Args:
            image: PIL Image to analyze.

        Returns:
            List of embedding values (512 dimensions).
        """
        if not self._is_loaded:
            self.load_model()

        # Get image embedding
        image_embedding = np.array(self.get_image_embedding(image))

        # Calculate similarity with each emotion category
        emotion_scores = []
        for emotion_name, prompts in EMOTION_CATEGORIES.items():
            # Get average embedding for this emotion's prompts
            prompt_embeddings = []
            for prompt in prompts:
                emb = self.get_raw_text_embedding(prompt)
                prompt_embeddings.append(emb)

            emotion_embedding = np.mean(prompt_embeddings, axis=0)
            emotion_embedding = emotion_embedding / np.linalg.norm(emotion_embedding)

            # Calculate similarity
            similarity = np.dot(image_embedding, emotion_embedding)
            emotion_scores.append((emotion_name, similarity))

        # Create weighted combination of top emotion embeddings
        emotion_scores.sort(key=lambda x: x[1], reverse=True)
        top_emotions = emotion_scores[:5]  # Top 5 emotions

        # Weight embeddings by their similarity scores
        weighted_embedding = np.zeros(512)
        total_weight = 0

        for emotion_name, score in top_emotions:
            if score > 0:
                prompts = EMOTION_CATEGORIES[emotion_name]
                prompt_embeddings = [
                    self.get_raw_text_embedding(p) for p in prompts
                ]
                emotion_emb = np.mean(prompt_embeddings, axis=0)
                weighted_embedding += emotion_emb * score
                total_weight += score

        if total_weight > 0:
            weighted_embedding = weighted_embedding / total_weight
            weighted_embedding = weighted_embedding / np.linalg.norm(weighted_embedding)

        return weighted_embedding.tolist()

    def classify_emotions(
        self,
        image: Image.Image,
        top_k: int = 5
    ) -> List[dict]:
        """
        Classify the emotions an image evokes.

        Args:
            image: PIL Image to classify.
            top_k: Number of top emotions to return.

        Returns:
            List of dicts with emotion name and confidence score.
        """
        if not self._is_loaded:
            self.load_model()

        image_embedding = np.array(self.get_image_embedding(image))

        emotion_scores = []
        for emotion_name, prompts in EMOTION_CATEGORIES.items():
            prompt_embeddings = []
            for prompt in prompts:
                emb = self.get_raw_text_embedding(prompt)
                prompt_embeddings.append(emb)

            emotion_embedding = np.mean(prompt_embeddings, axis=0)
            emotion_embedding = emotion_embedding / np.linalg.norm(emotion_embedding)

            similarity = np.dot(image_embedding, emotion_embedding)
            emotion_scores.append({
                "emocion": emotion_name,
                "confianza": float(similarity)
            })

        # Sort by confidence and return top_k
        emotion_scores.sort(key=lambda x: x["confianza"], reverse=True)
        return emotion_scores[:top_k]

    def parse_hybrid_query(self, query: str) -> Tuple[str, str, bool]:
        """
        Parse a query to separate visual content from emotional content.

        Handles queries like:
        - "caballo que inspire libertad" → ("caballo", "libertad", True)
        - "flores coloridas con energía" → ("flores coloridas", "energía", True)
        - "paisaje montañoso" → ("paisaje montañoso", "", False)

        Args:
            query: Natural language query.

        Returns:
            Tuple of (content_query, emotion_query, has_emotion).
        """
        query_lower = query.lower()

        # Check if query contains emotional keywords
        has_emotion = any(kw in query_lower for kw in EMOTION_KEYWORDS_ES)

        if not has_emotion:
            return (query, "", False)

        # Patterns to split content from emotion
        # Verbs include singular and plural subjunctive forms, with optional "o" between verbs
        _EMOTION_VERBS = (
            r"(?:inspire[n]?|inspira|evoque[n]?|evoca|transmita[n]?|transmite|exprese[n]?|expresa)"
        )
        _EMOTION_VERBS_WITH_OR = (
            rf"{_EMOTION_VERBS}(?:\s+o\s+{_EMOTION_VERBS})?"
        )
        patterns = [
            rf"(.+?)\s+que\s+{_EMOTION_VERBS_WITH_OR}\s+(.+)",
            r"(.+?)\s+con\s+(?:sensación|sentimiento|emoción|ambiente|atmósfera)\s+(?:de\s+)?(.+)",
            r"(.+?)\s+(?:inspirando|evocando|transmitiendo|expresando)\s+(.+)",
            r"(.+?)\s+que\s+(?:de[n]?|genere[n]?|provoque[n]?)\s+(.+)",
        ]

        # Generic words that don't add visual meaning
        _GENERIC_CONTENT = {
            "pinturas", "pintura", "cuadros", "cuadro", "obras", "obra",
            "arte", "imágenes", "imagenes", "imagen", "dibujos", "dibujo",
        }

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                emotion = match.group(2).strip()
                # Eliminar prefijos conversacionales del contenido (sin fallback al original)
                stripped = content
                for prefix in self._QUERY_PREFIXES:
                    stripped = re.sub(prefix, "", stripped, flags=re.IGNORECASE).strip()
                # Si tras limpiar queda vacío o solo palabra genérica → solo emoción
                if not stripped or stripped.lower() in _GENERIC_CONTENT:
                    return ("", emotion, True)
                return (stripped, emotion, True)

        # Check for emotion keywords at the end
        emotion_found = ""
        content = query

        for emotion in EMOTION_CATEGORIES.keys():
            emotion_lower = emotion.lower()
            if emotion_lower in query_lower:
                emotion_found = emotion
                # Remove emotion from content
                content = re.sub(
                    rf"\b{emotion_lower}\b",
                    "",
                    query,
                    flags=re.IGNORECASE
                ).strip()
                # Clean up connectors
                content = re.sub(
                    r"\s+(y|con|que|de)\s*$",
                    "",
                    content,
                    flags=re.IGNORECASE
                ).strip()
                break

        if emotion_found:
            return (content, emotion_found, True)

        return (query, "", False)
