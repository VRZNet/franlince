"""
Script para validar si el modelo CLIP normaliza la salida automáticamente.
"""

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from pathlib import Path

def check_clip_normalization(image_path: str = None):
    print("=" * 60)
    print("VALIDACIÓN DE NORMALIZACIÓN EN CLIP")
    print("=" * 60)
    
    # Cargar modelo
    print("\n🔄 Cargando modelo CLIP...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("✅ Modelo cargado")
    
    # Buscar una imagen si no se especifica
    if not image_path:
        pinturas_dir = Path("./pinturas")
        if pinturas_dir.exists():
            images = list(pinturas_dir.glob("*.jpg")) + list(pinturas_dir.glob("*.JPG")) + list(pinturas_dir.glob("*.png"))
            if images:
                image_path = str(images[0])
    
    if not image_path:
        print("\n❌ No se encontró ninguna imagen para probar")
        print("   Uso: python check_normalization.py ./ruta/a/imagen.jpg")
        return
    
    print(f"\n📸 Imagen de prueba: {image_path}")
    
    # Cargar imagen
    image = Image.open(image_path).convert("RGB")
    
    # === PRUEBA 1: Embedding de imagen SIN normalizar ===
    print("\n" + "-" * 60)
    print("PRUEBA 1: Embedding de IMAGEN")
    print("-" * 60)
    
    image_inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        image_features_raw = model.get_image_features(**image_inputs)
    
    norma_imagen_raw = torch.norm(image_features_raw, p=2, dim=-1).item()
    
    print(f"\n   Sin normalizar manualmente:")
    print(f"   - Norma del vector: {norma_imagen_raw:.4f}")
    print(f"   - Dimensiones: {image_features_raw.shape}")
    print(f"   - ¿Normalizado (norma ≈ 1)? {'SÍ ✅' if abs(norma_imagen_raw - 1.0) < 0.01 else 'NO ❌'}")
    
    # Normalizar manualmente
    image_features_norm = image_features_raw / image_features_raw.norm(p=2, dim=-1, keepdim=True)
    norma_imagen_norm = torch.norm(image_features_norm, p=2, dim=-1).item()
    
    print(f"\n   Después de normalizar manualmente:")
    print(f"   - Norma del vector: {norma_imagen_norm:.4f}")
    print(f"   - ¿Normalizado (norma ≈ 1)? {'SÍ ✅' if abs(norma_imagen_norm - 1.0) < 0.01 else 'NO ❌'}")
    
    # === PRUEBA 2: Embedding de texto SIN normalizar ===
    print("\n" + "-" * 60)
    print("PRUEBA 2: Embedding de TEXTO")
    print("-" * 60)
    
    texto = "a beautiful landscape painting"
    text_inputs = processor(text=[texto], return_tensors="pt", padding=True)
    
    with torch.no_grad():
        text_features_raw = model.get_text_features(**text_inputs)
    
    norma_texto_raw = torch.norm(text_features_raw, p=2, dim=-1).item()
    
    print(f"\n   Texto: \"{texto}\"")
    print(f"\n   Sin normalizar manualmente:")
    print(f"   - Norma del vector: {norma_texto_raw:.4f}")
    print(f"   - Dimensiones: {text_features_raw.shape}")
    print(f"   - ¿Normalizado (norma ≈ 1)? {'SÍ ✅' if abs(norma_texto_raw - 1.0) < 0.01 else 'NO ❌'}")
    
    # Normalizar manualmente
    text_features_norm = text_features_raw / text_features_raw.norm(p=2, dim=-1, keepdim=True)
    norma_texto_norm = torch.norm(text_features_norm, p=2, dim=-1).item()
    
    print(f"\n   Después de normalizar manualmente:")
    print(f"   - Norma del vector: {norma_texto_norm:.4f}")
    print(f"   - ¿Normalizado (norma ≈ 1)? {'SÍ ✅' if abs(norma_texto_norm - 1.0) < 0.01 else 'NO ❌'}")
    
    # === PRUEBA 3: Similitud coseno ===
    print("\n" + "-" * 60)
    print("PRUEBA 3: Similitud Coseno")
    print("-" * 60)
    
    # Sin normalizar
    sim_raw = (image_features_raw @ text_features_raw.T).item()
    
    # Con normalización
    sim_norm = (image_features_norm @ text_features_norm.T).item()
    
    print(f"\n   Similitud imagen-texto SIN normalizar: {sim_raw:.4f}")
    print(f"   Similitud imagen-texto CON normalizar: {sim_norm:.4f}")
    print(f"\n   ⚠️  Sin normalización, la similitud puede estar fuera de [-1, 1]")
    print(f"   ✅ Con normalización, la similitud está en rango [-1, 1]")
    
    # === CONCLUSIÓN ===
    print("\n" + "=" * 60)
    print("CONCLUSIÓN")
    print("=" * 60)
    
    if abs(norma_imagen_raw - 1.0) < 0.01 and abs(norma_texto_raw - 1.0) < 0.01:
        print("""
   ✅ El modelo CLIP YA NORMALIZA automáticamente.
   
   No necesitas normalizar manualmente en tu código.
        """)
    else:
        print(f"""
   ❌ El modelo CLIP NO NORMALIZA automáticamente.
   
   - Norma de imagen: {norma_imagen_raw:.2f} (debería ser ~1.0)
   - Norma de texto:  {norma_texto_raw:.2f} (debería ser ~1.0)
   
   👉 DEBES normalizar manualmente para búsquedas correctas:
   
      features = features / features.norm(p=2, dim=-1, keepdim=True)
        """)


if __name__ == "__main__":
    import sys
    
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    check_clip_normalization(image_path)
