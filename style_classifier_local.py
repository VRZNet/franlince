"""
=============================================================================
CLASIFICADOR DE ESTILO ARTÍSTICO - FRANLINCE (v3 - Prompts Mejorados)
=============================================================================

INSTRUCCIONES DE INSTALACIÓN:
    pip install torch torchvision transformers pillow

USO:
    python style_classifier_local_v3.py /ruta/a/carpeta/con/pinturas

=============================================================================
"""

import sys
from pathlib import Path
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel


# ============================================================================
# CATEGORÍAS CON PROMPTS OPTIMIZADOS PARA FRANLINCE
# ============================================================================

STYLE_CATEGORIES = {
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


class ArtStyleClassifier:
    """Clasificador de estilo artístico usando CLIP con múltiples prompts por categoría"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        print(f"🔄 Cargando modelo {model_name}...")
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.categories = STYLE_CATEGORIES
        self.style_names = list(self.categories.keys())
        
        # Pre-calcular embeddings de texto para todas las categorías
        self._precompute_text_embeddings()
        
        print("✅ Modelo cargado correctamente\n")
    
    def _precompute_text_embeddings(self):
        """Pre-calcula los embeddings de texto para mayor velocidad"""
        self.category_embeddings = {}
        
        for style_name, prompts in self.categories.items():
            # Procesar todos los prompts de esta categoría
            text_inputs = self.processor(
                text=prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            with torch.no_grad():
                text_features = self.model.get_text_features(**text_inputs)
                # Normalizar
                text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
                # Promediar los embeddings de todos los prompts
                avg_embedding = text_features.mean(dim=0, keepdim=True)
                avg_embedding = avg_embedding / avg_embedding.norm(p=2, dim=-1, keepdim=True)
                self.category_embeddings[style_name] = avg_embedding
    
    def classify(self, image_path: str, top_k: int = 3) -> dict:
        """Clasifica el estilo de una pintura."""
        # Cargar imagen
        image = Image.open(image_path).convert("RGB")
        
        # Redimensionar si es muy grande
        max_size = 1024
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Obtener embedding de la imagen
        image_inputs = self.processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        
        # Calcular similitud con cada categoría
        similarities = {}
        for style_name, text_embedding in self.category_embeddings.items():
            similarity = (image_features @ text_embedding.T).squeeze().item()
            similarities[style_name] = similarity
        
        # Convertir a probabilidades con softmax
        import math
        # Usar temperatura para hacer las diferencias más pronunciadas
        temperature = 0.5
        exp_sims = {k: math.exp(v / temperature) for k, v in similarities.items()}
        total = sum(exp_sims.values())
        probs = {k: v / total for k, v in exp_sims.items()}
        
        # Ordenar resultados
        results = [{"estilo": k, "confianza": v} for k, v in probs.items()]
        results.sort(key=lambda x: x["confianza"], reverse=True)
        
        return {
            "archivo": Path(image_path).name,
            "ruta": str(image_path),
            "estilo_principal": results[0]["estilo"],
            "confianza": results[0]["confianza"],
            "top_estilos": results[:top_k],
            "todos_los_estilos": results
        }
    
    def classify_batch(self, folder_path: str) -> list:
        """Clasifica todas las imágenes en una carpeta."""
        folder = Path(folder_path)
        extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
        images = [f for f in folder.iterdir() if f.suffix.lower() in extensions]
        
        print(f"📂 Encontradas {len(images)} imágenes en {folder_path}\n")
        
        results = []
        for i, img_path in enumerate(images, 1):
            print(f"[{i}/{len(images)}] Procesando: {img_path.name}")
            try:
                result = self.classify(str(img_path))
                results.append(result)
                self._print_result(result)
            except Exception as e:
                print(f"  ❌ Error: {e}\n")
                results.append({"archivo": img_path.name, "error": str(e)})
        
        return results
    
    def _print_result(self, result: dict):
        """Imprime resultado de forma visual"""
        print(f"  🎨 Estilo: {result['estilo_principal']} ({result['confianza']*100:.1f}%)")
        print(f"  📊 Top 3:")
        for style in result["top_estilos"]:
            bar = "█" * int(style["confianza"] * 30)
            print(f"     {style['estilo']:18} {bar} {style['confianza']*100:.1f}%")
        print()


def export_to_csv(results: list, output_path: str = "catalogo_estilos.csv"):
    """Exporta resultados a CSV"""
    import csv
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["archivo", "estilo_principal", "confianza", "estilo_2", "conf_2", "estilo_3", "conf_3"])
        
        for r in results:
            if "error" in r:
                writer.writerow([r["archivo"], "ERROR", 0, "", 0, "", 0])
            else:
                row = [r["archivo"], r["estilo_principal"], f"{r['confianza']:.3f}"]
                for style in r["top_estilos"][1:3]:
                    row.extend([style["estilo"], f"{style['confianza']:.3f}"])
                writer.writerow(row)
    
    print(f"\n📁 Resultados exportados a: {output_path}")


def export_to_json(results: list, output_path: str = "catalogo_estilos.json"):
    """Exporta resultados a JSON"""
    import json
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"📁 Resultados exportados a: {output_path}")


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nUso: python style_classifier_local_v3.py <carpeta_con_imagenes>")
        print("\nEjemplo:")
        print("  python style_classifier_local_v3.py ./mis_pinturas")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    
    if not Path(folder_path).exists():
        print(f"❌ La carpeta '{folder_path}' no existe")
        sys.exit(1)
    
    # Crear clasificador
    classifier = ArtStyleClassifier()
    
    # Clasificar todas las imágenes
    results = classifier.classify_batch(folder_path)
    
    # Exportar resultados
    export_to_csv(results)
    export_to_json(results)
    
    # Resumen
    print("\n" + "="*60)
    print("📊 RESUMEN DEL CATÁLOGO")
    print("="*60)
    
    from collections import Counter
    styles = Counter(r.get("estilo_principal", "Error") for r in results)
    
    print(f"\nTotal de pinturas: {len(results)}")
    print("\nDistribución por estilo:")
    for style, count in styles.most_common():
        pct = count / len(results) * 100
        bar = "█" * int(pct / 2)
        print(f"  {style:18} {bar} {count} ({pct:.0f}%)")