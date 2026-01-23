"""
Constants for Franlince API.
Contains style categories and other constant values.
"""

from typing import Dict, List, Set

# Style categories with multiple prompts for CLIP classification
STYLE_CATEGORIES: Dict[str, List[str]] = {
    "Paisaje": [
        "a painting of mountains and valleys",
        "a painting of countryside with trees and fields",
        "a painting of sunset or sunrise over hills",
        "a landscape painting with sky and nature",
        "a painting of forests and rivers in nature"
    ],
    "Marino": [
        "a painting of ocean waves and sea",
        "a painting of boats and ships on water",
        "a painting of beach with sand and waves",
        "a seascape painting with blue water",
        "a painting of lighthouse by the sea"
    ],
    "Abstracto": [
        "an abstract painting with swirling colors and shapes",
        "a modern abstract painting with gold and blue textures",
        "an abstract art with fluid flowing patterns",
        "a contemporary abstract painting with metallic colors",
        "an abstract painting with geometric shapes and bold colors"
    ],
    "Retrato": [
        "a portrait painting of a person face",
        "an artistic portrait of a woman with decorative elements",
        "a stylized portrait with gold accents",
        "a portrait painting showing human face and shoulders",
        "a decorative portrait with artistic background"
    ],
    "Naturaleza Muerta": [
        "a still life painting with fruits on table",
        "a still life painting with vases and flowers",
        "a painting of wine bottles and glasses",
        "a still life with food and kitchen objects",
        "a painting of objects arranged on a surface"
    ],
    "Urbano": [
        "a street art style painting with graffiti",
        "a pop art painting with bold colors and icons",
        "a painting of classical sculpture with modern graffiti background",
        "an urban art painting mixing classical and street style",
        "a painting with hearts and graffiti street art style"
    ],
    "Floral": [
        "a painting of roses and flowers",
        "a floral painting with colorful blooms",
        "a painting of sunflowers",
        "a decorative painting with flower arrangements",
        "a painting featuring flowers as main subject"
    ],
    "Fauna": [
        "a painting of wild animals",
        "a painting of elephants or lions",
        "a painting of horses running",
        "a painting of birds in nature",
        "an animal portrait painting"
    ],
    "Religioso": [
        "a religious painting with Jesus Christ",
        "a painting of Virgin Mary",
        "a painting with angels and saints",
        "a biblical scene painting",
        "a painting of the last supper or crucifixion"
    ]
}

# Allowed image types for upload
ALLOWED_IMAGE_TYPES: List[str] = [
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp"
]

# Image file extensions
IMAGE_EXTENSIONS: Set[str] = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"
}

# Media type mapping for serving images
MEDIA_TYPES: Dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp"
}

# Classification temperature for softmax
CLASSIFICATION_TEMPERATURE: float = 0.5
