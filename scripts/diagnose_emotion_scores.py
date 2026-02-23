#!/usr/bin/env python3
"""
Diagnóstico de scores emocionales para imágenes.

Este script te permite:
1. Ver los scores raw de cada prompt individual
2. Comparar diferentes métodos de agregación
3. Identificar qué prompts son más discriminativos

Uso:
    python -m scripts.diagnose_emotion_scores path/to/image.jpg
    python -m scripts.diagnose_emotion_scores path/to/image.jpg --top-emotions 5
    python -m scripts.diagnose_emotion_scores path/to/image.jpg --show-prompts
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from PIL import Image

from src.services.classifier import CLIPClassifier
from src.core.constants import (
    EMOTION_CATEGORIES,
    EMOTION_AGGREGATION_METHOD,
    EMOTION_TOP_K
)


def diagnose_image(image_path: str, top_emotions: int = 5, show_prompts: bool = False):
    """Analiza una imagen y muestra diagnóstico detallado."""

    print("=" * 70)
    print(f"DIAGNÓSTICO DE EMOCIONES: {Path(image_path).name}")
    print("=" * 70)

    # Load image
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Initialize classifier
    print("\nCargando modelo CLIP...")
    classifier = CLIPClassifier()
    classifier.load_model()

    # Get image features
    prepared_image = classifier.image_processor.prepare_for_model(image)
    image_inputs = classifier.processor(images=prepared_image, return_tensors="pt")

    with torch.no_grad():
        image_features = classifier.model.get_image_features(**image_inputs)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

    print("\n" + "=" * 70)
    print("SCORES POR MÉTODO DE AGREGACIÓN")
    print("=" * 70)

    # Calculate scores for each method
    methods = {
        "max": {},
        "mean": {},
        "top_3_mean": {},
        "top_3_weighted": {}
    }

    all_prompt_scores = {}

    for emotion_name, prompt_embeddings in classifier.emotion_prompt_embeddings.items():
        # Calculate similarity with each prompt
        similarities = (image_features @ prompt_embeddings.T).squeeze()
        if similarities.dim() == 0:
            similarities = similarities.unsqueeze(0)

        sims_list = similarities.tolist()
        all_prompt_scores[emotion_name] = sims_list

        # Different aggregation methods
        methods["max"][emotion_name] = max(sims_list)
        methods["mean"][emotion_name] = sum(sims_list) / len(sims_list)

        top_3 = sorted(sims_list, reverse=True)[:3]
        methods["top_3_mean"][emotion_name] = sum(top_3) / len(top_3)

        weights = [1.0, 0.7, 0.5][:len(top_3)]
        weight_sum = sum(weights)
        methods["top_3_weighted"][emotion_name] = sum(
            s * w for s, w in zip(top_3, weights)
        ) / weight_sum

    # Print comparison table
    print("\n{:<15} {:>10} {:>10} {:>12} {:>14}".format(
        "Emoción", "MAX", "MEAN", "TOP3_MEAN", "TOP3_WEIGHTED"
    ))
    print("-" * 65)

    # Sort by top_3_weighted for display
    sorted_emotions = sorted(
        methods["top_3_weighted"].items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_emotions]

    for emotion, _ in sorted_emotions:
        print("{:<15} {:>10.4f} {:>10.4f} {:>12.4f} {:>14.4f}".format(
            emotion,
            methods["max"][emotion],
            methods["mean"][emotion],
            methods["top_3_mean"][emotion],
            methods["top_3_weighted"][emotion]
        ))

    # Print spread analysis
    print("\n" + "=" * 70)
    print("ANÁLISIS DE DISCRIMINACIÓN")
    print("=" * 70)

    for method_name, scores in methods.items():
        sorted_scores = sorted(scores.values(), reverse=True)
        top1 = sorted_scores[0]
        top2 = sorted_scores[1]
        spread = top1 - top2

        print(f"\n{method_name}:")
        print(f"  Top 1: {top1:.4f}")
        print(f"  Top 2: {top2:.4f}")
        print(f"  Spread (diferencia): {spread:.4f}")
        print(f"  Ratio (top1/top2): {top1/top2:.3f}")

    # Show individual prompts if requested
    if show_prompts:
        print("\n" + "=" * 70)
        print("SCORES POR PROMPT INDIVIDUAL")
        print("=" * 70)

        for emotion, _ in sorted_emotions:
            print(f"\n{emotion}:")
            prompts = EMOTION_CATEGORIES[emotion]
            scores = all_prompt_scores[emotion]

            # Sort prompts by score
            prompt_scores = list(zip(prompts, scores))
            prompt_scores.sort(key=lambda x: x[1], reverse=True)

            for i, (prompt, score) in enumerate(prompt_scores[:5]):
                # Truncate long prompts
                prompt_short = prompt[:60] + "..." if len(prompt) > 60 else prompt
                marker = "★" if i == 0 else " "
                print(f"  {marker} {score:.4f} | {prompt_short}")

    # Final classification with current method
    print("\n" + "=" * 70)
    print(f"CLASIFICACIÓN FINAL (método: {EMOTION_AGGREGATION_METHOD}, k={EMOTION_TOP_K})")
    print("=" * 70)

    results = classifier.classify_emotions(image_features)

    print("\n{:<15} {:>12} {:>12}".format("Emoción", "Confianza", "Raw Score"))
    print("-" * 40)
    for r in results[:top_emotions]:
        print("{:<15} {:>11.2%} {:>12.4f}".format(
            r["emocion"],
            r["confianza"],
            r.get("raw_score", 0)
        ))


def compare_methods_recommendation():
    """Print recommendation based on use case."""
    print("\n" + "=" * 70)
    print("RECOMENDACIONES")
    print("=" * 70)
    print("""
    MÉTODO          | CUÁNDO USAR
    ----------------|--------------------------------------------------
    max             | Imágenes con una emoción dominante clara
    mean            | NO RECOMENDADO - diluye los scores
    top_3_mean      | Balance entre especificidad y robustez
    top_3_weighted  | RECOMENDADO - da más peso al mejor match

    Si tus imágenes son ambiguas (como Rivera), usa top_3_weighted o max.
    Si los scores siguen muy juntos, considera:
    1. Reducir el número de prompts por categoría
    2. Hacer prompts más específicos y únicos por categoría
    3. Usar un threshold mínimo de diferencia
    """)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diagnóstico de clasificación emocional"
    )
    parser.add_argument(
        "image_path",
        help="Ruta a la imagen a analizar"
    )
    parser.add_argument(
        "--top-emotions", "-n",
        type=int,
        default=5,
        help="Número de emociones top a mostrar (default: 5)"
    )
    parser.add_argument(
        "--show-prompts", "-p",
        action="store_true",
        help="Mostrar scores de prompts individuales"
    )

    args = parser.parse_args()

    if not Path(args.image_path).exists():
        print(f"Error: No se encontró la imagen: {args.image_path}")
        sys.exit(1)

    diagnose_image(
        args.image_path,
        top_emotions=args.top_emotions,
        show_prompts=args.show_prompts
    )
    compare_methods_recommendation()
