# Franlince - API de Catalogacion de Pinturas

API para catalogar pinturas automaticamente usando IA (CLIP de OpenAI).

## Tabla de Contenidos

- [Descripcion](#descripcion)
- [Tecnologias](#tecnologias)
- [Arquitectura](#arquitectura)
- [Instalacion](#instalacion)
- [Uso de la API](#uso-de-la-api)
- [Ejemplos Detallados](#ejemplos-detallados)
- [Scripts de Utilidad](#scripts-de-utilidad)
- [Base de Datos](#base-de-datos)
- [Tests](#tests)
- [Configuracion](#configuracion)
- [Solucion de Problemas](#solucion-de-problemas)

## Descripcion

Franlince es una API REST que permite:

- **Catalogacion Automatica**: Sube imagenes de pinturas y el sistema las clasifica automaticamente en 9 estilos artisticos usando inteligencia artificial.
- **Busqueda por Estilo**: Filtra pinturas por categoria artistica.
- **Busqueda Semantica**: Encuentra pinturas usando descripciones en lenguaje natural (ej: "flores coloridas", "paisaje con montanas").
- **Gestion de Catalogo**: CRUD completo para administrar el catalogo de pinturas.

### Estilos Soportados

| Estilo | Descripcion |
|--------|-------------|
| **Paisaje** | Montanas, campos, cielos, naturaleza |
| **Marino** | Mar, oceano, playas, barcos |
| **Abstracto** | Formas geometricas, patrones, colores |
| **Retrato** | Personas, rostros, figuras humanas |
| **Naturaleza Muerta** | Frutas, flores en jarrones, objetos |
| **Urbano** | Ciudades, edificios, arte callejero |
| **Floral** | Flores como tema principal |
| **Fauna** | Animales, vida silvestre |
| **Religioso** | Escenas biblicas, santos, iconografia |

## Tecnologias

- **FastAPI** - Framework web moderno y de alto rendimiento
- **CLIP (OpenAI)** - Modelo de IA para clasificacion de imagenes
- **PostgreSQL + pgvector** - Base de datos con soporte para busqueda vectorial
- **Pydantic** - Validacion de datos y configuracion
- **Docker** - Contenedorizacion de servicios

## Arquitectura

El proyecto sigue una arquitectura Clean Architecture con separacion de capas:

```
franlince/
├── src/
│   ├── api/                    # Capa de Presentacion (FastAPI)
│   │   ├── main.py             # Configuracion de la app FastAPI
│   │   ├── dependencies.py     # Inyeccion de dependencias
│   │   └── routes/
│   │       ├── catalog.py      # Endpoints: upload, paintings, delete
│   │       ├── search.py       # Endpoints: search, semantic-search
│   │       └── stats.py        # Endpoints: stats, estilos, health
│   │
│   ├── core/                   # Configuracion Central
│   │   ├── config.py           # Settings con variables de entorno
│   │   └── constants.py        # Categorias de estilos, constantes
│   │
│   ├── models/                 # Modelos de Dominio
│   │   ├── painting.py         # Entidad Painting (dataclass)
│   │   └── schemas.py          # Schemas Pydantic (request/response)
│   │
│   ├── services/               # Capa de Logica de Negocio
│   │   ├── classifier.py       # CLIPClassifier - clasificacion IA
│   │   ├── embedding.py        # EmbeddingService - vectores semanticos
│   │   └── image_processor.py  # Redimension y procesamiento
│   │
│   ├── repositories/           # Capa de Acceso a Datos
│   │   └── painting_repository.py  # CRUD PostgreSQL
│   │
│   └── database/               # Conexion a Base de Datos
│       └── connection.py       # Pool de conexiones
│
├── scripts/                    # Scripts de Utilidad
│   ├── load_initial_data.py    # Cargar catalogo inicial
│   └── generate_embeddings.py  # Generar embeddings masivamente
│
├── tests/                      # Tests Automatizados
│   ├── conftest.py             # Configuracion pytest
│   ├── test_classifier.py      # Tests del clasificador
│   └── test_api.py             # Tests de integracion API
│
├── docker/                     # Configuracion Docker
│   ├── docker-compose.yml      # PostgreSQL + pgvector
│   └── init.sql                # Schema de base de datos
│
├── run.py                      # Punto de entrada principal
├── requirements.txt            # Dependencias Python
└── .env.example                # Template de configuracion
```

### Flujo de Clasificacion

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Imagen    │────>│ ImageProcessor│────>│  CLIPClassifier │
│   (upload)  │     │  (resize)    │     │  (clasificar)   │
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                   │
                    ┌──────────────┐     ┌─────────▼────────┐
                    │  PostgreSQL  │<────│ PaintingRepository│
                    │  (guardar)   │     │  (save + embed)  │
                    └──────────────┘     └──────────────────┘
```

## Instalacion

### Requisitos Previos

- Python 3.9+
- Docker y Docker Compose
- 4GB RAM minimo (para el modelo CLIP)

### 1. Clonar y Configurar Entorno

```bash
cd franlince
python -m venv env
source env/bin/activate  # Linux/Mac
# o en Windows: env\Scripts\activate

pip install -r requirements.txt
```

### 2. Configurar Variables de Entorno

```bash
cp .env.example .env
# Editar .env si es necesario
```

### 3. Iniciar Base de Datos

```bash
cd docker
docker-compose up -d
cd ..

# Verificar que PostgreSQL esta corriendo
docker ps
```

### 4. Ejecutar la API

```bash
python run.py
```
source venv/bin/activate && python run.py  

La API estara disponible en `http://localhost:8000`

**Primera ejecucion**: El modelo CLIP se descargara automaticamente (~600MB).

## Uso de la API

### Documentacion Interactiva

Con la API corriendo, accede a:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Endpoints Disponibles

#### Sistema
| Metodo | Endpoint | Descripcion |
|--------|----------|-------------|
| GET | `/health` | Estado del servicio y modelo |

#### Catalogacion
| Metodo | Endpoint | Descripcion |
|--------|----------|-------------|
| POST | `/catalog/upload` | Subir y clasificar una imagen |
| POST | `/catalog/upload-batch` | Subir multiples imagenes |
| GET | `/catalog/paintings` | Listar pinturas (con paginacion) |
| GET | `/catalog/painting/{id}` | Detalle de una pintura |
| GET | `/catalog/painting/{id}/image` | Obtener imagen binaria |
| DELETE | `/catalog/painting/{id}` | Eliminar pintura |

#### Busqueda
| Metodo | Endpoint | Descripcion |
|--------|----------|-------------|
| GET | `/catalog/search` | Buscar por estilo |
| GET | `/catalog/semantic-search` | Busqueda por texto natural |

#### Estadisticas
| Metodo | Endpoint | Descripcion |
|--------|----------|-------------|
| GET | `/catalog/stats` | Estadisticas del catalogo |
| GET | `/catalog/estilos` | Lista de estilos disponibles |

## Ejemplos Detallados

### 1. Health Check

```bash
curl http://localhost:8000/health
```

**Respuesta:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "timestamp": "2026-01-23T10:30:00.000000"
}
```

### 2. Listar Estilos Disponibles

```bash
curl http://localhost:8000/catalog/estilos
```

**Respuesta:**
```json
{
  "estilos": [
    "Paisaje", "Marino", "Abstracto", "Retrato",
    "Naturaleza Muerta", "Urbano", "Floral", "Fauna", "Religioso"
  ]
}
```

### 3. Subir y Clasificar una Imagen

```bash
curl -X POST \
  -F "file=@mi_pintura.jpg" \
  http://localhost:8000/catalog/upload
```

**Respuesta:**
```json
{
  "success": true,
  "message": "Pintura catalogada exitosamente",
  "data": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "archivo": "a1b2c3d4-e5f6-7890-abcd-ef1234567890.jpg",
    "estilo_principal": "Paisaje",
    "confianza": 87.5,
    "top_estilos": [
      {"estilo": "Paisaje", "confianza": 87.5},
      {"estilo": "Marino", "confianza": 6.2},
      {"estilo": "Floral", "confianza": 3.1}
    ]
  }
}
```

### 4. Subir Multiples Imagenes

```bash
curl -X POST \
  -F "files=@pintura1.jpg" \
  -F "files=@pintura2.jpg" \
  -F "files=@pintura3.png" \
  http://localhost:8000/catalog/upload-batch
```

**Respuesta:**
```json
{
  "success": true,
  "total_procesadas": 3,
  "total_errores": 0,
  "pinturas": [
    {"id": "...", "archivo_original": "pintura1.jpg", "estilo": "Abstracto", "confianza": 92.3},
    {"id": "...", "archivo_original": "pintura2.jpg", "estilo": "Retrato", "confianza": 78.1},
    {"id": "...", "archivo_original": "pintura3.png", "estilo": "Floral", "confianza": 85.7}
  ],
  "errores": []
}
```

### 5. Listar Pinturas

```bash
# Todas las pinturas (paginado)
curl "http://localhost:8000/catalog/paintings?limit=10&offset=0"

# Filtrar por estilo
curl "http://localhost:8000/catalog/paintings?estilo=Abstracto&limit=20"
```

**Respuesta:**
```json
{
  "total": 150,
  "limit": 10,
  "offset": 0,
  "pinturas": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "archivo": "imagen1.jpg",
      "estilo_principal": "Paisaje",
      "confianza": 0.875,
      "estilo_2": "Marino",
      "confianza_2": 0.062,
      "created_at": "2026-01-23T10:30:00"
    }
  ]
}
```

### 6. Buscar por Estilo

```bash
curl "http://localhost:8000/catalog/search?estilo=Floral&min_confianza=0.5"
```

**Respuesta:**
```json
{
  "estilo": "Floral",
  "total": 25,
  "pinturas": [
    {
      "id": "...",
      "archivo": "flores_rojas.jpg",
      "estilo_principal": "Floral",
      "confianza": 0.92,
      "created_at": "2026-01-20T15:45:00"
    }
  ]
}
```

### 7. Busqueda Semantica

```bash
# Buscar por descripcion en lenguaje natural
curl "http://localhost:8000/catalog/semantic-search?query=flores%20coloridas%20en%20jarron"

curl "http://localhost:8000/catalog/semantic-search?query=atardecer%20en%20la%20playa"

curl "http://localhost:8000/catalog/semantic-search?query=retrato%20de%20mujer%20elegante"
```

**Respuesta:**
```json
{
  "query": "flores coloridas en jarron",
  "total": 10,
  "resultados": [
    {
      "id": "...",
      "archivo": "bodegon_flores.jpg",
      "estilo": "Floral",
      "confianza_estilo": 89.2,
      "similitud_busqueda": 78.5
    },
    {
      "id": "...",
      "archivo": "naturaleza_muerta.jpg",
      "estilo": "Naturaleza Muerta",
      "confianza_estilo": 85.1,
      "similitud_busqueda": 72.3
    }
  ]
}
```

### 8. Ver Detalle de Pintura

```bash
curl http://localhost:8000/catalog/painting/550e8400-e29b-41d4-a716-446655440000
```

### 9. Obtener Imagen

```bash
# Descargar imagen
curl -o pintura.jpg http://localhost:8000/catalog/painting/{id}/image

# Ver en navegador
open "http://localhost:8000/catalog/painting/{id}/image"
```

### 10. Eliminar Pintura

```bash
curl -X DELETE http://localhost:8000/catalog/painting/550e8400-e29b-41d4-a716-446655440000
```

**Respuesta:**
```json
{
  "success": true,
  "message": "Pintura 550e8400-e29b-41d4-a716-446655440000 eliminada",
  "archivo": "imagen1.jpg"
}
```

### 11. Estadisticas del Catalogo

```bash
curl http://localhost:8000/catalog/stats
```

**Respuesta:**
```json
{
  "total_pinturas": 150,
  "con_embeddings": 150,
  "ultima_actualizacion": "2026-01-23T10:30:00",
  "por_estilo": [
    {"estilo_principal": "Paisaje", "cantidad": 35, "confianza_promedio": 0.823},
    {"estilo_principal": "Abstracto", "cantidad": 28, "confianza_promedio": 0.756},
    {"estilo_principal": "Floral", "cantidad": 25, "confianza_promedio": 0.891}
  ]
}
```

## Scripts de Utilidad

### Cargar Datos Iniciales

Si tienes un archivo `catalogo_estilos.json` generado previamente:

```bash
# Cargar catalogo
python -m scripts.load_initial_data catalogo_estilos.json

# Ejecutar consultas de prueba
python -m scripts.load_initial_data test
```

### Generar Embeddings Masivamente

Para actualizar embeddings de imagenes existentes:

```bash
# Generar embeddings para todas las imagenes en una carpeta
python -m scripts.generate_embeddings ./pinturas

# Probar busqueda semantica
python -m scripts.generate_embeddings search "paisaje con montanas"
```

## Base de Datos

### Schema

```sql
CREATE TABLE pinturas (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    archivo VARCHAR(255) NOT NULL,
    ruta VARCHAR(500),
    imagen BYTEA,                    -- Imagen binaria
    estilo_principal VARCHAR(50) NOT NULL,
    confianza FLOAT NOT NULL,
    estilo_2 VARCHAR(50),
    confianza_2 FLOAT,
    estilo_3 VARCHAR(50),
    confianza_3 FLOAT,
    todos_estilos JSONB,
    embedding vector(512),           -- Vector para busqueda semantica
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Indices

- `idx_pinturas_estilo` - Busqueda rapida por estilo
- `idx_pinturas_archivo` - Busqueda por nombre de archivo
- `idx_pinturas_embedding` - Busqueda vectorial (IVFFlat)

### Comandos Utiles

```bash
# Conectar a PostgreSQL
docker exec -it franlince_db psql -U franlince -d franlince_catalog

# Ver pinturas
SELECT archivo, estilo_principal, confianza FROM pinturas LIMIT 10;

# Contar por estilo
SELECT * FROM resumen_estilos;

# Ver pinturas con embeddings
SELECT COUNT(*) FROM pinturas WHERE embedding IS NOT NULL;
```

## Tests

### Ejecutar Tests

```bash
# Tests rapidos (sin cargar modelo CLIP)
pytest tests/ -v

# Todos los tests incluyendo lentos
pytest tests/ -v -m "slow"

# Test especifico
pytest tests/test_api.py -v

# Con coverage
pytest tests/ --cov=src --cov-report=html
```

### Estructura de Tests

- `test_classifier.py` - Tests unitarios del clasificador CLIP
- `test_api.py` - Tests de integracion de endpoints
- `conftest.py` - Fixtures y configuracion de pytest

## Configuracion

### Variables de Entorno

Crea un archivo `.env` basado en `.env.example`:

```env
# Base de Datos
DB_HOST=localhost
DB_PORT=5432
DB_NAME=franlince_catalog
DB_USER=franlince
DB_PASSWORD=franlince123

# Directorio de Uploads
UPLOAD_DIR=./pinturas_catalogo

# Modelo CLIP
CLIP_MODEL_NAME=openai/clip-vit-base-patch32

# API
API_HOST=0.0.0.0
API_PORT=8000

# Procesamiento de Imagenes
MAX_IMAGE_SIZE=1024
```

### Descripcion de Variables

| Variable | Descripcion | Default |
|----------|-------------|---------|
| `DB_HOST` | Host de PostgreSQL | localhost |
| `DB_PORT` | Puerto de PostgreSQL | 5432 |
| `DB_NAME` | Nombre de la base de datos | franlince_catalog |
| `DB_USER` | Usuario de BD | franlince |
| `DB_PASSWORD` | Contrasena de BD | franlince123 |
| `UPLOAD_DIR` | Directorio para imagenes | ./pinturas_catalogo |
| `CLIP_MODEL_NAME` | Modelo HuggingFace | openai/clip-vit-base-patch32 |
| `API_HOST` | Host de la API | 0.0.0.0 |
| `API_PORT` | Puerto de la API | 8000 |
| `MAX_IMAGE_SIZE` | Tamano maximo de imagen (px) | 1024 |

## Solucion de Problemas

### Error: "Address already in use"

```bash
# Encontrar proceso usando el puerto
lsof -i :8000

# Matar el proceso
kill -9 <PID>
```

### Error: "Connection refused" (PostgreSQL)

```bash
# Verificar que Docker esta corriendo
docker ps

# Reiniciar contenedor
cd docker
docker-compose down
docker-compose up -d
```

### Error: "Module not found"

```bash
# Activar entorno virtual
source env/bin/activate

# Reinstalar dependencias
pip install -r requirements.txt
```

### Modelo CLIP no carga

```bash
# Verificar espacio en disco (necesita ~600MB)
df -h

# Limpiar cache de HuggingFace si es necesario
rm -rf ~/.cache/huggingface
```

### Busqueda semantica sin resultados

Verificar que las pinturas tienen embeddings:

```bash
curl http://localhost:8000/catalog/stats
# Verificar que "con_embeddings" > 0
```

Si no hay embeddings, regenerarlos:

```bash
python -m scripts.generate_embeddings ./pinturas
```

## Desarrollo

### Agregar Nuevo Estilo

1. Editar `src/core/constants.py`
2. Agregar categoria a `STYLE_CATEGORIES` con prompts descriptivos
3. Reiniciar la API para recargar el modelo

### Estructura de Prompts

Cada estilo usa multiples prompts para mejorar la clasificacion:

```python
"Paisaje": [
    "a painting of mountains and valleys",
    "a painting of countryside with trees and fields",
    "a landscape painting with sky and nature",
    # ... mas prompts
]
```

## Licencia

Proyecto de certificacion - FRANLINCE

---

**Desarrollado con FastAPI + CLIP + PostgreSQL**
