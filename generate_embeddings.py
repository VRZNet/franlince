"""
=============================================================================
GENERAR EMBEDDINGS Y GUARDAR EN POSTGRESQL - FRANLINCE
=============================================================================

Este script:
1. Carga cada imagen de la carpeta pinturas
2. Genera su embedding (vector 512 dimensiones) con CLIP
3. Guarda el embedding en PostgreSQL para búsqueda semántica

Requisitos:
    pip install torch torchvision transformers pillow psycopg2-binary

Uso:
    python generate_embeddings.py ./pinturas

=============================================================================
"""

import sys
from pathlib import Path
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import psycopg2
import json


# Configuración de la base de datos
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "franlince_catalog",
    "user": "franlince",
    "password": "franlince123"
}


class EmbeddingGenerator:
    """Genera embeddings de imágenes usando CLIP"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        print(f"🔄 Cargando modelo {model_name}...")
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        print("✅ Modelo cargado\n")
    
    def get_image_embedding(self, image_path: str) -> list:
        """Genera embedding de una imagen"""
        image = Image.open(image_path).convert("RGB")
        
        # Redimensionar si es muy grande
        max_size = 1024
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Generar embedding
        inputs = self.processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            # Normalizar
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        
        # Convertir a lista
        return image_features.squeeze().tolist()
    
    def get_text_embedding(self, text: str) -> list:
        """Genera embedding de un texto (para búsquedas)"""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        
        return text_features.squeeze().tolist()


def connect_db():
    """Conecta a PostgreSQL"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        return None


def update_embeddings(folder_path: str):
    """Genera y guarda embeddings para todas las imágenes"""
    
    folder = Path(folder_path)
    extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
    images = [f for f in folder.iterdir() if f.suffix.lower() in extensions]
    
    print(f"📂 Encontradas {len(images)} imágenes en {folder_path}\n")
    
    # Conectar a DB
    conn = connect_db()
    if not conn:
        return
    
    cursor = conn.cursor()
    
    # Inicializar generador de embeddings
    generator = EmbeddingGenerator()
    
    # Procesar cada imagen
    updated = 0
    errors = 0
    
    for i, img_path in enumerate(images, 1):
        filename = img_path.name
        print(f"[{i}/{len(images)}] {filename}...", end=" ")
        
        try:
            # Generar embedding
            embedding = generator.get_image_embedding(str(img_path))
            
            # Convertir a formato PostgreSQL vector
            embedding_str = "[" + ",".join(map(str, embedding)) + "]"
            
            # Actualizar en DB
            cursor.execute("""
                UPDATE pinturas 
                SET embedding = %s::vector 
                WHERE archivo = %s
            """, (embedding_str, filename))
            
            if cursor.rowcount > 0:
                print("✅")
                updated += 1
            else:
                print("⚠️ No encontrado en DB")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            errors += 1
    
    conn.commit()
    
    # Crear índice para búsqueda rápida
    print("\n🔧 Creando índice de búsqueda vectorial...")
    try:
        cursor.execute("""
            DROP INDEX IF EXISTS idx_pinturas_embedding;
            CREATE INDEX idx_pinturas_embedding 
            ON pinturas USING ivfflat (embedding vector_cosine_ops) 
            WITH (lists = 10);
        """)
        conn.commit()
        print("✅ Índice creado")
    except Exception as e:
        print(f"⚠️ Índice no creado (se puede crear después): {e}")
    
    cursor.close()
    conn.close()
    
    print(f"\n{'='*50}")
    print(f"📊 RESUMEN")
    print(f"{'='*50}")
    print(f"  ✅ Actualizados: {updated}")
    print(f"  ❌ Errores: {errors}")
    print(f"\n✅ Embeddings guardados en PostgreSQL")


def test_semantic_search(query: str, limit: int = 5):
    """Prueba búsqueda semántica"""
    
    print(f"\n🔍 Buscando: \"{query}\"")
    print("-" * 50)
    
    # Generar embedding del texto de búsqueda
    generator = EmbeddingGenerator()
    query_embedding = generator.get_text_embedding(query)
    embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
    
    # Buscar en DB
    conn = connect_db()
    if not conn:
        return
    
    cursor = conn.cursor()
    
    # Búsqueda por similitud coseno
    cursor.execute("""
        SELECT 
            archivo, 
            estilo_principal, 
            confianza,
            1 - (embedding <=> %s::vector) as similitud
        FROM pinturas 
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, (embedding_str, embedding_str, limit))
    
    results = cursor.fetchall()
    
    if results:
        print(f"\n📋 Top {limit} resultados:\n")
        for i, (archivo, estilo, confianza, similitud) in enumerate(results, 1):
            print(f"  {i}. {archivo}")
            print(f"     Estilo: {estilo} | Similitud: {similitud:.3f}")
            print()
    else:
        print("  No se encontraron resultados")
    
    cursor.close()
    conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("Uso:")
        print("  python generate_embeddings.py ./pinturas           # Generar embeddings")
        print("  python generate_embeddings.py search \"flores rojas\" # Buscar")
        sys.exit(1)
    
    if sys.argv[1] == "search":
        if len(sys.argv) < 3:
            print("❌ Especifica qué buscar: python generate_embeddings.py search \"tu búsqueda\"")
            sys.exit(1)
        test_semantic_search(sys.argv[2])
    else:
        update_embeddings(sys.argv[1])
        print("\n💡 Para probar búsqueda semántica:")
        print("   python generate_embeddings.py search \"flores coloridas\"")
        print("   python generate_embeddings.py search \"paisaje con montañas\"")
        print("   python generate_embeddings.py search \"arte abstracto azul y dorado\"")
