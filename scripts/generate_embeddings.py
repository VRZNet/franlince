#!/usr/bin/env python3
"""
Generate embeddings and save to PostgreSQL.

This script:
1. Loads each image from the paintings folder
2. Generates its embedding (512-dimension vector) with CLIP
3. Saves the embedding to PostgreSQL for semantic search

Usage:
    python -m scripts.generate_embeddings ./pinturas           # Generate embeddings
    python -m scripts.generate_embeddings search "red flowers" # Search
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.constants import IMAGE_EXTENSIONS
from src.database.connection import get_db_connection
from src.services.embedding import EmbeddingService
from src.repositories.painting_repository import PaintingRepository


def update_embeddings(folder_path: str):
    """Generate and save embeddings for all images."""

    folder = Path(folder_path)
    images = [
        f for f in folder.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    ]

    print(f"Found {len(images)} images in {folder_path}\n")

    conn = get_db_connection()
    if not conn:
        return

    cursor = conn.cursor()

    embedding_service = EmbeddingService()
    embedding_service.load_model()

    repository = PaintingRepository()

    updated = 0
    errors = 0

    for i, img_path in enumerate(images, 1):
        filename = img_path.name
        print(f"[{i}/{len(images)}] {filename}...", end=" ")

        try:
            embedding = embedding_service.get_image_embedding_from_path(str(img_path))

            if repository.update_embedding(filename, embedding):
                print("OK")
                updated += 1
            else:
                print("Not found in DB")

        except Exception as e:
            print(f"Error: {e}")
            errors += 1

    conn.commit()

    print("\nCreating vector search index...")
    try:
        cursor.execute("""
            DROP INDEX IF EXISTS idx_pinturas_embedding;
            CREATE INDEX idx_pinturas_embedding
            ON pinturas USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 10);
        """)
        conn.commit()
        print("Index created")
    except Exception as e:
        print(f"Index not created (can be created later): {e}")

    cursor.close()
    conn.close()

    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"  Updated: {updated}")
    print(f"  Errors: {errors}")
    print("\nEmbeddings saved to PostgreSQL")


def test_semantic_search(query: str, limit: int = 5):
    """Test semantic search."""

    print(f"\nSearching: \"{query}\"")
    print("-" * 50)

    embedding_service = EmbeddingService()
    embedding_service.load_model()

    query_embedding = embedding_service.get_text_embedding(query)

    repository = PaintingRepository()
    results = repository.semantic_search(query_embedding, limit)

    if results:
        print(f"\nTop {limit} results:\n")
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r['archivo']}")
            print(f"     Style: {r['estilo_principal']} | Similarity: {r['similitud']:.3f}")
            print()
    else:
        print("  No results found")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("Usage:")
        print("  python -m scripts.generate_embeddings ./pinturas           # Generate embeddings")
        print("  python -m scripts.generate_embeddings search \"flores rojas\" # Search")
        sys.exit(1)

    if sys.argv[1] == "search":
        if len(sys.argv) < 3:
            print("Specify what to search: python -m scripts.generate_embeddings search \"your search\"")
            sys.exit(1)
        test_semantic_search(sys.argv[2])
    else:
        update_embeddings(sys.argv[1])
        print("\nTo test semantic search:")
        print("   python -m scripts.generate_embeddings search \"colorful flowers\"")
        print("   python -m scripts.generate_embeddings search \"mountain landscape\"")
        print("   python -m scripts.generate_embeddings search \"blue and gold abstract art\"")
