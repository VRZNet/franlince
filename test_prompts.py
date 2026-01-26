"""
This script is used to test different prompts for the semantic search.
"""

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from pathlib import Path
import psycopg2

# Configuración de la base de datos
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "franlince_catalog",
    "user": "franlince",
    "password": "franlince123"
}

def connect_db():
    """Conecta a PostgreSQL"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        return None

def test_prompts(query: str, limit: int = 5):
    """Prueba diferentes prompts para una consulta de búsqueda"""

    print(f"\n🔍 Buscando: \"{query}\"")
    print("-" * 50)

    # Generar embedding del texto de búsqueda
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    prompts = [
        f"a painting of {query}",
        f"a photo of {query}",
        f"an artistic image of {query}",
        f"a drawing of {query}",
        f"{query}",
    ]

    for prompt in prompts:
        print(f"\n--- Prompt: '{prompt}' ---")

        inputs = processor(text=[prompt], return_tensors="pt", padding=True)
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        
        embedding_str = "[" + ",".join(map(str, text_features.squeeze().tolist())) + "]"

        # Buscar en DB
        conn = connect_db()
        if not conn:
            return

        cursor = conn.cursor()

        # Búsqueda por producto interno
        cursor.execute("""
            SELECT 
                archivo, 
                estilo_principal, 
                confianza,
                (embedding <#> %s::vector) * -1 as similitud
            FROM pinturas 
            WHERE embedding IS NOT NULL
            ORDER BY embedding <#> %s::vector
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
    import sys
    if len(sys.argv) < 2:
        print("Uso: python test_prompts.py \"<consulta>\"")
        sys.exit(1)
    
    test_prompts(sys.argv[1])
