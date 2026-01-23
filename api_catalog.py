"""
=============================================================================
API DE CATALOGACIÓN - FRANLINCE
=============================================================================

API para que operaciones suba imágenes y se cataloguen automáticamente.

Requisitos:
    pip install fastapi uvicorn python-multipart psycopg2-binary torch transformers pillow

Ejecutar:
    uvicorn api_catalog:app --reload --host 0.0.0.0 --port 8000

Endpoints:
    POST /catalog/upload      - Subir y catalogar una imagen
    POST /catalog/upload-batch - Subir múltiples imágenes
    GET  /catalog/paintings    - Listar todas las pinturas
    GET  /catalog/painting/{id} - Ver detalle de una pintura
    GET  /catalog/search       - Buscar por estilo
    GET  /catalog/semantic-search - Búsqueda semántica por texto
    GET  /catalog/stats        - Estadísticas del catálogo
    DELETE /catalog/painting/{id} - Eliminar una pintura

=============================================================================
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
import uuid
import os
from pathlib import Path
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import psycopg2
from psycopg2.extras import RealDictCursor
import json
from datetime import datetime
import shutil


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "franlince_catalog",
    "user": "franlince",
    "password": "franlince123"
}

# Carpeta para guardar imágenes subidas
UPLOAD_DIR = Path("./pinturas_catalogo")
UPLOAD_DIR.mkdir(exist_ok=True)

# Categorías de estilo
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


# =============================================================================
# INICIALIZACIÓN
# =============================================================================

app = FastAPI(
    title="Franlince - API de Catalogación",
    description="API para catalogar pinturas automáticamente usando IA",
    version="1.0.0"
)

# CORS para permitir requests desde cualquier origen (tablet, web, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales para el modelo (se cargan una vez)
model = None
processor = None
category_embeddings = {}


def load_model():
    """Carga el modelo CLIP una sola vez al iniciar"""
    global model, processor, category_embeddings
    
    if model is None:
        print("🔄 Cargando modelo CLIP...")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Pre-calcular embeddings de categorías
        for style_name, prompts in STYLE_CATEGORIES.items():
            text_inputs = processor(
                text=prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            with torch.no_grad():
                text_features = model.get_text_features(**text_inputs)
                text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
                avg_embedding = text_features.mean(dim=0, keepdim=True)
                avg_embedding = avg_embedding / avg_embedding.norm(p=2, dim=-1, keepdim=True)
                category_embeddings[style_name] = avg_embedding
        
        print("✅ Modelo cargado")


@app.on_event("startup")
async def startup_event():
    """Carga el modelo al iniciar el servidor"""
    load_model()


def get_db():
    """Obtiene conexión a la base de datos"""
    return psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)


# =============================================================================
# FUNCIONES DE CLASIFICACIÓN
# =============================================================================

def classify_image(image: Image.Image) -> dict:
    """Clasifica una imagen y retorna estilo + embedding"""
    global model, processor, category_embeddings
    
    # Redimensionar si es muy grande
    max_size = 1024
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    # Obtener embedding de la imagen
    image_inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        image_features = model.get_image_features(**image_inputs)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    
    # Calcular similitud con cada categoría
    import math
    similarities = {}
    for style_name, text_embedding in category_embeddings.items():
        similarity = (image_features @ text_embedding.T).squeeze().item()
        similarities[style_name] = similarity
    
    # Convertir a probabilidades
    temperature = 0.5
    exp_sims = {k: math.exp(v / temperature) for k, v in similarities.items()}
    total = sum(exp_sims.values())
    probs = {k: v / total for k, v in exp_sims.items()}
    
    # Ordenar resultados
    results = [{"estilo": k, "confianza": v} for k, v in probs.items()]
    results.sort(key=lambda x: x["confianza"], reverse=True)
    
    # Embedding como lista
    embedding = image_features.squeeze().tolist()
    
    return {
        "estilo_principal": results[0]["estilo"],
        "confianza": results[0]["confianza"],
        "top_estilos": results[:3],
        "todos_estilos": results,
        "embedding": embedding
    }


def save_to_db(filename: str, image_bytes: bytes, classification: dict) -> str:
    """Guarda la pintura clasificada en la base de datos (incluyendo la imagen)"""
    conn = get_db()
    cursor = conn.cursor()
    
    embedding_str = "[" + ",".join(map(str, classification["embedding"])) + "]"
    
    top = classification["top_estilos"]
    
    cursor.execute("""
        INSERT INTO pinturas (
            archivo, imagen, estilo_principal, confianza,
            estilo_2, confianza_2, estilo_3, confianza_3,
            todos_estilos, embedding
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector
        )
        RETURNING id
    """, (
        filename,
        psycopg2.Binary(image_bytes),
        classification["estilo_principal"],
        classification["confianza"],
        top[1]["estilo"] if len(top) > 1 else None,
        top[1]["confianza"] if len(top) > 1 else None,
        top[2]["estilo"] if len(top) > 2 else None,
        top[2]["confianza"] if len(top) > 2 else None,
        json.dumps(classification["todos_estilos"]),
        embedding_str
    ))
    
    painting_id = cursor.fetchone()["id"]
    conn.commit()
    cursor.close()
    conn.close()
    
    return str(painting_id)


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.post("/catalog/upload", tags=["Catalogación"])
async def upload_painting(file: UploadFile = File(...)):
    """
    Sube una imagen, la clasifica automáticamente y la guarda en el catálogo.
    
    - **file**: Imagen de la pintura (JPG, PNG, etc.)
    
    Retorna el ID de la pintura y su clasificación.
    """
    # Validar tipo de archivo
    allowed_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Tipo de archivo no permitido: {file.content_type}")
    
    try:
        # Leer bytes de la imagen
        image_bytes = await file.read()
        
        # Generar nombre único
        extension = Path(file.filename).suffix or ".jpg"
        unique_filename = f"{uuid.uuid4()}{extension}"
        
        # Abrir imagen desde bytes y clasificar
        from io import BytesIO
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        classification = classify_image(image)
        
        # Guardar en DB (imagen como bytes)
        painting_id = save_to_db(unique_filename, image_bytes, classification)
        
        return {
            "success": True,
            "message": "Pintura catalogada exitosamente",
            "data": {
                "id": painting_id,
                "archivo": unique_filename,
                "estilo_principal": classification["estilo_principal"],
                "confianza": round(classification["confianza"] * 100, 1),
                "top_estilos": [
                    {"estilo": e["estilo"], "confianza": round(e["confianza"] * 100, 1)}
                    for e in classification["top_estilos"]
                ]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/catalog/upload-batch", tags=["Catalogación"])
async def upload_paintings_batch(files: List[UploadFile] = File(...)):
    """
    Sube múltiples imágenes y las cataloga todas.
    
    - **files**: Lista de imágenes de pinturas
    
    Retorna el resumen de todas las clasificaciones.
    """
    from io import BytesIO
    
    results = []
    errors = []
    
    for file in files:
        try:
            # Validar tipo
            allowed_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
            if file.content_type not in allowed_types:
                errors.append({"archivo": file.filename, "error": "Tipo no permitido"})
                continue
            
            # Leer bytes de la imagen
            image_bytes = await file.read()
            
            # Generar nombre único
            extension = Path(file.filename).suffix or ".jpg"
            unique_filename = f"{uuid.uuid4()}{extension}"
            
            # Clasificar
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            classification = classify_image(image)
            
            # Guardar en DB (imagen como bytes)
            painting_id = save_to_db(unique_filename, image_bytes, classification)
            
            results.append({
                "id": painting_id,
                "archivo_original": file.filename,
                "archivo_guardado": unique_filename,
                "estilo": classification["estilo_principal"],
                "confianza": round(classification["confianza"] * 100, 1)
            })
            
        except Exception as e:
            errors.append({"archivo": file.filename, "error": str(e)})
    
    return {
        "success": True,
        "total_procesadas": len(results),
        "total_errores": len(errors),
        "pinturas": results,
        "errores": errors
    }


@app.get("/catalog/paintings", tags=["Consultas"])
async def list_paintings(
    estilo: Optional[str] = Query(None, description="Filtrar por estilo"),
    limit: int = Query(50, description="Límite de resultados"),
    offset: int = Query(0, description="Offset para paginación")
):
    """
    Lista todas las pinturas del catálogo.
    
    - **estilo**: Filtrar por estilo (opcional)
    - **limit**: Máximo de resultados (default: 50)
    - **offset**: Para paginación
    """
    conn = get_db()
    cursor = conn.cursor()
    
    if estilo:
        cursor.execute("""
            SELECT id, archivo, ruta, estilo_principal, confianza, 
                   estilo_2, confianza_2, estilo_3, confianza_3, created_at
            FROM pinturas 
            WHERE estilo_principal = %s
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
        """, (estilo, limit, offset))
    else:
        cursor.execute("""
            SELECT id, archivo, ruta, estilo_principal, confianza,
                   estilo_2, confianza_2, estilo_3, confianza_3, created_at
            FROM pinturas 
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
        """, (limit, offset))
    
    paintings = cursor.fetchall()
    
    # Contar total
    if estilo:
        cursor.execute("SELECT COUNT(*) as total FROM pinturas WHERE estilo_principal = %s", (estilo,))
    else:
        cursor.execute("SELECT COUNT(*) as total FROM pinturas")
    
    total = cursor.fetchone()["total"]
    
    cursor.close()
    conn.close()
    
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "pinturas": [dict(p) for p in paintings]
    }


@app.get("/catalog/painting/{painting_id}", tags=["Consultas"])
async def get_painting(painting_id: str):
    """
    Obtiene el detalle de una pintura específica.
    
    - **painting_id**: UUID de la pintura
    """
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, archivo, ruta, estilo_principal, confianza,
               estilo_2, confianza_2, estilo_3, confianza_3,
               todos_estilos, created_at, updated_at
        FROM pinturas 
        WHERE id = %s
    """, (painting_id,))
    
    painting = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if not painting:
        raise HTTPException(status_code=404, detail="Pintura no encontrada")
    
    return dict(painting)


@app.get("/catalog/search", tags=["Búsqueda"])
async def search_by_style(
    estilo: str = Query(..., description="Estilo a buscar"),
    min_confianza: float = Query(0.0, description="Confianza mínima (0-1)")
):
    """
    Busca pinturas por estilo.
    
    - **estilo**: Nombre del estilo (Paisaje, Marino, Abstracto, etc.)
    - **min_confianza**: Filtrar por confianza mínima
    """
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, archivo, ruta, estilo_principal, confianza, created_at
        FROM pinturas 
        WHERE estilo_principal = %s AND confianza >= %s
        ORDER BY confianza DESC
    """, (estilo, min_confianza))
    
    paintings = cursor.fetchall()
    cursor.close()
    conn.close()
    
    return {
        "estilo": estilo,
        "total": len(paintings),
        "pinturas": [dict(p) for p in paintings]
    }


@app.get("/catalog/semantic-search", tags=["Búsqueda"])
async def semantic_search(
    query: str = Query(..., description="Descripción de lo que buscas"),
    limit: int = Query(10, description="Número de resultados")
):
    """
    Búsqueda semántica por descripción en lenguaje natural.
    
    Ejemplos:
    - "flores coloridas"
    - "paisaje con montañas y atardecer"
    - "arte abstracto azul y dorado"
    - "retrato de mujer elegante"
    """
    global model, processor
    
    # Generar embedding del texto de búsqueda
    text_inputs = processor(text=[query], return_tensors="pt", padding=True)
    
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    
    query_embedding = text_features.squeeze().tolist()
    embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
    
    # Buscar en DB
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            id, archivo, ruta, estilo_principal, confianza,
            1 - (embedding <=> %s::vector) as similitud
        FROM pinturas 
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, (embedding_str, embedding_str, limit))
    
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    
    return {
        "query": query,
        "total": len(results),
        "resultados": [
            {
                "id": str(r["id"]),
                "archivo": r["archivo"],
                "ruta": r["ruta"],
                "estilo": r["estilo_principal"],
                "confianza_estilo": round(r["confianza"] * 100, 1),
                "similitud_busqueda": round(r["similitud"] * 100, 1)
            }
            for r in results
        ]
    }


@app.get("/catalog/stats", tags=["Estadísticas"])
async def get_stats():
    """
    Obtiene estadísticas del catálogo.
    """
    conn = get_db()
    cursor = conn.cursor()
    
    # Total de pinturas
    cursor.execute("SELECT COUNT(*) as total FROM pinturas")
    total = cursor.fetchone()["total"]
    
    # Por estilo
    cursor.execute("""
        SELECT estilo_principal, COUNT(*) as cantidad, 
               ROUND(AVG(confianza)::numeric, 3) as confianza_promedio
        FROM pinturas 
        GROUP BY estilo_principal 
        ORDER BY cantidad DESC
    """)
    por_estilo = cursor.fetchall()
    
    # Con embeddings
    cursor.execute("SELECT COUNT(*) as total FROM pinturas WHERE embedding IS NOT NULL")
    con_embeddings = cursor.fetchone()["total"]
    
    # Última actualización
    cursor.execute("SELECT MAX(created_at) as ultima FROM pinturas")
    ultima = cursor.fetchone()["ultima"]
    
    cursor.close()
    conn.close()
    
    return {
        "total_pinturas": total,
        "con_embeddings": con_embeddings,
        "ultima_actualizacion": ultima.isoformat() if ultima else None,
        "por_estilo": [dict(e) for e in por_estilo]
    }


@app.delete("/catalog/painting/{painting_id}", tags=["Administración"])
async def delete_painting(painting_id: str):
    """
    Elimina una pintura del catálogo.
    
    - **painting_id**: UUID de la pintura a eliminar
    """
    conn = get_db()
    cursor = conn.cursor()
    
    # Obtener info antes de borrar
    cursor.execute("SELECT archivo, ruta FROM pinturas WHERE id = %s", (painting_id,))
    painting = cursor.fetchone()
    
    if not painting:
        cursor.close()
        conn.close()
        raise HTTPException(status_code=404, detail="Pintura no encontrada")
    
    # Borrar de DB
    cursor.execute("DELETE FROM pinturas WHERE id = %s", (painting_id,))
    conn.commit()
    
    # Borrar archivo físico (opcional)
    try:
        filepath = Path(painting["ruta"])
        if filepath.exists():
            filepath.unlink()
    except:
        pass
    
    cursor.close()
    conn.close()
    
    return {
        "success": True,
        "message": f"Pintura {painting_id} eliminada",
        "archivo": painting["archivo"]
    }


@app.get("/catalog/estilos", tags=["Consultas"])
async def list_estilos():
    """
    Lista todos los estilos disponibles.
    """
    return {
        "estilos": list(STYLE_CATEGORIES.keys())
    }


@app.get("/catalog/painting/{painting_id}/image", tags=["Consultas"])
async def get_painting_image(painting_id: str):
    """
    Obtiene la imagen de una pintura desde la base de datos.
    
    - **painting_id**: UUID de la pintura
    
    Retorna la imagen como archivo.
    """
    from fastapi.responses import Response
    
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT archivo, imagen 
        FROM pinturas 
        WHERE id = %s
    """, (painting_id,))
    
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if not result or not result["imagen"]:
        raise HTTPException(status_code=404, detail="Imagen no encontrada")
    
    # Detectar tipo de imagen por extensión
    extension = Path(result["archivo"]).suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp"
    }
    media_type = media_types.get(extension, "image/jpeg")
    
    return Response(
        content=bytes(result["imagen"]),
        media_type=media_type,
        headers={"Content-Disposition": f"inline; filename={result['archivo']}"}
    )


# =============================================================================
# HEALTH CHECK
# =============================================================================

@app.get("/health", tags=["Sistema"])
async def health_check():
    """Verifica que el servicio esté funcionando"""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
