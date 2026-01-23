"""
Image processing service for Franlince API.
Handles image resizing and preparation.
"""

from io import BytesIO
from typing import Optional
from PIL import Image

from src.core.config import get_settings


class ImageProcessor:
    """Handles image processing operations."""

    def __init__(self, max_size: Optional[int] = None):
        """
        Initialize image processor.

        Args:
            max_size: Maximum dimension for images. Defaults to settings value.
        """
        settings = get_settings()
        self.max_size = max_size or settings.max_image_size

    def resize_image(self, image: Image.Image) -> Image.Image:
        """
        Resize image if it exceeds max size while maintaining aspect ratio.

        Args:
            image: PIL Image to resize.

        Returns:
            Resized PIL Image.
        """
        if max(image.size) > self.max_size:
            ratio = self.max_size / max(image.size)
            new_size = (
                int(image.size[0] * ratio),
                int(image.size[1] * ratio)
            )
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        return image

    def load_from_bytes(self, image_bytes: bytes) -> Image.Image:
        """
        Load image from bytes and convert to RGB.

        Args:
            image_bytes: Raw image bytes.

        Returns:
            PIL Image in RGB mode.
        """
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        return image

    def load_from_path(self, image_path: str) -> Image.Image:
        """
        Load image from file path and convert to RGB.

        Args:
            image_path: Path to image file.

        Returns:
            PIL Image in RGB mode.
        """
        image = Image.open(image_path).convert("RGB")
        return image

    def prepare_for_model(self, image: Image.Image) -> Image.Image:
        """
        Prepare image for model input by resizing if needed.

        Args:
            image: PIL Image.

        Returns:
            Prepared PIL Image.
        """
        return self.resize_image(image)
