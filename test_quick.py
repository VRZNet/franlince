#!/usr/bin/env python3
"""
Script rápido para probar el clasificador
Ejecuta: python test_quick.py
"""

from pathlib import Path
from style_classifier_local import ArtStyleClassifier

if __name__ == "__main__":
    print("🎨 Test rápido del clasificador\n")
    
    # Verificar que hay imágenes
    folder = Path("./pinturas")
    if not folder.exists():
        print("❌ Carpeta './pinturas' no encontrada")
        exit(1)
    
    # Buscar imágenes
    images = list(folder.glob("*.jpg")) + list(folder.glob("*.JPG")) + \
             list(folder.glob("*.png")) + list(folder.glob("*.PNG"))
    
    if not images:
        print("❌ No hay imágenes .jpg o .png en ./pinturas")
        exit(1)
    
    print(f"✅ Encontradas {len(images)} imágenes\n")
    
    # Inicializar clasificador
    print("⏳ Cargando modelo CLIP...")
    classifier = ArtStyleClassifier()
    
    # Procesar solo la primera imagen
    img = images[0]
    print(f"\n🖼️  Procesando: {img.name}")
    
    try:
        result = classifier.classify(str(img))
        print(f"✅ Estilo: {result['estilo_principal']} ({result['confianza']*100:.1f}%)")
        print(f"\nTop 3:")
        for style in result['top_estilos']:
            print(f"  - {style['estilo']}: {style['confianza']*100:.1f}%")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
