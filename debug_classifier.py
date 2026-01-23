"""
Script de debug interactivo para el clasificador de estilos
Ejecuta: python debug_classifier.py
"""

import sys
from pathlib import Path
from PIL import Image
import torch
from style_classifier_local import ArtStyleClassifier, STYLE_CATEGORIES


def debug_single_image():
    """Debug de una imagen individual"""
    print("\n" + "="*60)
    print("🔍 DEBUG: Clasificar imagen individual")
    print("="*60)
    
    # Buscar imágenes disponibles
    folder = Path("./pinturas")
    if not folder.exists():
        print("❌ Carpeta './pinturas' no encontrada")
        return
    
    images = list(folder.glob("*.[Jj][Pp][Gg]")) + list(folder.glob("*.[Pp][Nn][Gg]"))
    
    if not images:
        print("❌ No hay imágenes en ./pinturas")
        return
    
    print(f"\n📁 Imágenes disponibles: {len(images)}")
    for i, img in enumerate(images[:5], 1):
        print(f"  {i}. {img.name}")
    
    if len(images) > 5:
        print(f"  ... y {len(images) - 5} más")
    
    # Seleccionar primera imagen para debug
    img_path = images[0]
    print(f"\n✅ Usando: {img_path.name}\n")
    
    # Inicializar clasificador
    print("⏳ Inicializando modelo CLIP...")
    classifier = ArtStyleClassifier()
    
    # Debug: Información de modelo
    print("\n📊 INFO DEL MODELO:")
    print(f"  Modelo: openai/clip-vit-base-patch32")
    print(f"  Dispositivo: {torch.device('cpu')}")
    print(f"  Categorías definidas: {len(STYLE_CATEGORIES)}")
    
    # Debug: Información de imagen
    print("\n🖼️  INFO DE LA IMAGEN:")
    img = Image.open(img_path)
    print(f"  Ruta: {img_path}")
    print(f"  Tamaño original: {img.size}")
    print(f"  Formato: {img.format}")
    print(f"  Modo: {img.mode}")
    
    # Clasificar
    print("\n⏳ Procesando clasificación...")
    try:
        result = classifier.classify(str(img_path), top_k=5)
        
        print("\n✅ RESULTADO:")
        print(f"  Estilo principal: {result['estilo_principal']}")
        print(f"  Confianza: {result['confianza']*100:.2f}%")
        
        print("\n📊 TOP 5 ESTILOS:")
        for i, style in enumerate(result['top_estilos'], 1):
            bar = "█" * int(style['confianza'] * 50)
            print(f"  {i}. {style['estilo']:20} {bar} {style['confianza']*100:.2f}%")
        
        print("\n💾 TODOS LOS ESTILOS:")
        for style in result['todos_los_estilos']:
            print(f"  {style['estilo']:20} {style['confianza']*100:.2f}%")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


def debug_categories():
    """Debug de las categorías definidas"""
    print("\n" + "="*60)
    print("📋 DEBUG: Categorías de estilos")
    print("="*60)
    
    print(f"\nTotal de categorías: {len(STYLE_CATEGORIES)}\n")
    
    for i, (style, description) in enumerate(STYLE_CATEGORIES.items(), 1):
        print(f"{i}. {style}")
        print(f"   Descripción: {description}\n")


def debug_batch():
    """Debug de procesamiento por lotes"""
    print("\n" + "="*60)
    print("📦 DEBUG: Procesamiento por lotes")
    print("="*60)
    
    folder = Path("./pinturas")
    if not folder.exists():
        print("❌ Carpeta './pinturas' no encontrada")
        return
    
    extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
    images = [f for f in folder.iterdir() if f.suffix.lower() in extensions]
    
    print(f"\n📁 Total de imágenes: {len(images)}")
    
    if len(images) == 0:
        print("❌ No hay imágenes para procesar")
        return
    
    # Mostrar info de archivos
    print("\n📄 Primeras 10 imágenes:")
    for img in images[:10]:
        try:
            pil_img = Image.open(img)
            print(f"  {img.name:50} {pil_img.size[0]}x{pil_img.size[1]}")
        except Exception as e:
            print(f"  {img.name:50} ❌ Error: {e}")
    
    # Inicializar y procesar
    print("\n⏳ Inicializando modelo...")
    classifier = ArtStyleClassifier()
    
    print(f"\n⏳ Procesando {len(images)} imágenes...")
    results = classifier.classify_batch(str(folder))
    
    # Estadísticas
    successful = len([r for r in results if "error" not in r])
    errors = len([r for r in results if "error" in r])
    
    print("\n" + "="*60)
    print("📊 ESTADÍSTICAS")
    print("="*60)
    print(f"✅ Exitosas: {successful}")
    print(f"❌ Errores: {errors}")
    print(f"📊 Tasa de éxito: {successful/len(results)*100:.1f}%")
    
    if errors > 0:
        print("\n🔴 Errores encontrados:")
        for r in results:
            if "error" in r:
                print(f"  {r['archivo']}: {r['error']}")


if __name__ == "__main__":
    print("\n🎨 DEBUG INTERACTIVO - CLASIFICADOR DE ESTILOS")
    print("="*60)
    print("\nOpciones:")
    print("  1. Clasificar imagen individual")
    print("  2. Ver categorías de estilos")
    print("  3. Procesar lote (carpeta)")
    print("  4. Todas las anteriores")
    
    choice = input("\nSelecciona opción (1-4): ").strip()
    
    if choice in ["1", "4"]:
        debug_single_image()
    
    if choice in ["2", "4"]:
        debug_categories()
    
    if choice in ["3", "4"]:
        debug_batch()
    
    print("\n✅ Debug finalizado\n")
