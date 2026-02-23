#!/usr/bin/env python3
"""
Generate emotional embeddings for existing paintings.

This script:
1. Fetches all paintings from the database that don't have emotional embeddings
2. Loads each image and generates emotional embeddings + classification
3. Updates the database with the emotional data

Usage:
    python -m scripts.generate_emotion_embeddings              # Process all paintings
    python -m scripts.generate_emotion_embeddings --force      # Reprocess all (including existing)
    python -m scripts.generate_emotion_embeddings --test 5     # Process only 5 paintings (for testing)
"""

import sys
import argparse
from pathlib import Path
from io import BytesIO

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image

from src.database.connection import DatabaseConnection
from src.services.classifier import CLIPClassifier
from src.repositories.painting_repository import PaintingRepository


def get_paintings_to_process(force: bool = False, limit: int = None):
    """Get list of paintings that need emotional embedding processing."""
    with DatabaseConnection.get_cursor() as cursor:
        if force:
            query = """
                SELECT id, archivo
                FROM pinturas
                WHERE embedding IS NOT NULL
            """
        else:
            query = """
                SELECT id, archivo
                FROM pinturas
                WHERE embedding IS NOT NULL
                  AND embedding_emocional IS NULL
            """

        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query)
        return [dict(row) for row in cursor.fetchall()]


def get_painting_image(painting_id: str) -> bytes:
    """Get image bytes for a painting."""
    with DatabaseConnection.get_cursor() as cursor:
        cursor.execute(
            "SELECT imagen FROM pinturas WHERE id = %s",
            (painting_id,)
        )
        result = cursor.fetchone()
        if result and result["imagen"]:
            return bytes(result["imagen"])
    return None


def process_painting(
    painting_id: str,
    classifier: CLIPClassifier,
    repository: PaintingRepository
) -> bool:
    """Process a single painting and update its emotional data."""
    try:
        # Get image bytes
        image_bytes = get_painting_image(painting_id)
        if not image_bytes:
            print("No image data")
            return False

        # Load image
        image = Image.open(BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Get classification (includes emotions)
        classification = classifier.classify_image(image)

        # Update database with emotional data
        emotion_embedding = classification.get("embedding_emocional", [])
        emotions = classification.get("top_emociones", [])

        if not emotion_embedding:
            print("No emotional embedding generated")
            return False

        return repository.update_emotion_data(
            painting_id=painting_id,
            emotion_embedding=emotion_embedding,
            emotions=emotions
        )

    except Exception as e:
        print(f"Error: {e}")
        return False


def generate_emotion_embeddings(force: bool = False, limit: int = None):
    """Generate emotional embeddings for all paintings."""
    print("=" * 60)
    print("GENERATING EMOTIONAL EMBEDDINGS")
    print("=" * 60)

    # Get paintings to process
    paintings = get_paintings_to_process(force=force, limit=limit)

    if not paintings:
        print("\nNo paintings to process.")
        if not force:
            print("All paintings already have emotional embeddings.")
            print("Use --force to reprocess all paintings.")
        return

    print(f"\nFound {len(paintings)} paintings to process\n")

    # Initialize classifier
    print("Loading CLIP model...")
    classifier = CLIPClassifier()
    classifier.load_model()
    print("Model loaded\n")

    repository = PaintingRepository()

    # Process each painting
    processed = 0
    errors = 0

    for i, painting in enumerate(paintings, 1):
        painting_id = str(painting["id"])
        archivo = painting["archivo"]

        print(f"[{i}/{len(paintings)}] {archivo}...", end=" ")

        if process_painting(painting_id, classifier, repository):
            print("OK")
            processed += 1
        else:
            errors += 1

    # Create index for emotional embeddings if we have processed enough
    if processed > 10:
        print("\nCreating vector search index for emotional embeddings...")
        try:
            with DatabaseConnection.get_cursor() as cursor:
                cursor.execute("""
                    DROP INDEX IF EXISTS idx_pinturas_embedding_emocional;
                    CREATE INDEX idx_pinturas_embedding_emocional
                    ON pinturas USING ivfflat (embedding_emocional vector_cosine_ops)
                    WITH (lists = 10);
                """)
            print("Index created")
        except Exception as e:
            print(f"Index not created (can be created later): {e}")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total to process: {len(paintings)}")
    print(f"  Processed: {processed}")
    print(f"  Errors: {errors}")


def test_emotion_search(query: str, limit: int = 5):
    """Test emotional search."""
    from src.services.embedding import EmbeddingService

    print(f"\nSearching emotions: \"{query}\"")
    print("-" * 50)

    embedding_service = EmbeddingService()
    embedding_service.load_model()

    emotion_embedding = embedding_service.get_emotion_text_embedding(query)

    repository = PaintingRepository()
    results = repository.search_by_emotion(emotion_embedding, limit)

    if results:
        print(f"\nTop {limit} results:\n")
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r['archivo']}")
            print(f"     Style: {r['estilo_principal']}")
            print(f"     Emotion: {r.get('emocion_principal', 'N/A')}")
            print(f"     Similarity: {r['similitud_emocion']:.3f}")
            print()
    else:
        print("  No results found")


def test_hybrid_search(query: str, limit: int = 5):
    """Test hybrid search (content + emotion)."""
    from src.services.embedding import EmbeddingService

    embedding_service = EmbeddingService()
    embedding_service.load_model()

    # Parse query
    content_query, emotion_query, has_emotion = embedding_service.parse_hybrid_query(query)

    print(f"\nHybrid Search: \"{query}\"")
    print(f"  Content: \"{content_query}\"")
    print(f"  Emotion: \"{emotion_query}\"")
    print(f"  Has emotion: {has_emotion}")
    print("-" * 50)

    if not has_emotion:
        print("No emotion detected in query. Use emotion-search for emotion-only queries.")
        return

    content_embedding = embedding_service.get_text_embedding(content_query)
    emotion_embedding = embedding_service.get_emotion_text_embedding(emotion_query)

    repository = PaintingRepository()
    results = repository.search_by_content_then_emotion(
        content_embedding=content_embedding,
        emotion_embedding=emotion_embedding,
        final_limit=limit
    )

    if results:
        print(f"\nTop {limit} results:\n")
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r['archivo']}")
            print(f"     Style: {r['estilo_principal']}")
            print(f"     Emotion: {r.get('emocion_principal', 'N/A')}")
            print(f"     Content sim: {r['similitud_contenido']:.3f}")
            print(f"     Emotion sim: {r['similitud_emocion']:.3f}")
            print(f"     Combined: {r['score_combinado']:.3f}")
            print()
    else:
        print("  No results found")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate emotional embeddings for paintings"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess all paintings, even those with existing emotional embeddings"
    )
    parser.add_argument(
        "--test",
        type=int,
        metavar="N",
        help="Process only N paintings (for testing)"
    )
    parser.add_argument(
        "--search-emotion",
        type=str,
        metavar="QUERY",
        help="Test emotional search with a query"
    )
    parser.add_argument(
        "--search-hybrid",
        type=str,
        metavar="QUERY",
        help="Test hybrid search (content + emotion) with a query"
    )

    args = parser.parse_args()

    if args.search_emotion:
        test_emotion_search(args.search_emotion)
    elif args.search_hybrid:
        test_hybrid_search(args.search_hybrid)
    else:
        generate_emotion_embeddings(force=args.force, limit=args.test)
        print("\nTo test emotional search:")
        print('   python -m scripts.generate_emotion_embeddings --search-emotion "libertad y aventura"')
        print('   python -m scripts.generate_emotion_embeddings --search-emotion "paz y tranquilidad"')
        print("\nTo test hybrid search:")
        print('   python -m scripts.generate_emotion_embeddings --search-hybrid "caballo que inspire libertad"')
        print('   python -m scripts.generate_emotion_embeddings --search-hybrid "flores con energía"')
