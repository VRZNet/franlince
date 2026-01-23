"""
FastAPI application entry point.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import catalog, search, stats
from src.api.dependencies import get_classifier
from src.database.connection import DatabaseConnection
from src.core.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Initializes resources on startup and cleans up on shutdown.
    """
    # Startup
    settings = get_settings()
    settings.ensure_upload_dir()

    # Initialize database connection pool
    DatabaseConnection.initialize_pool()

    # Load CLIP model
    classifier = get_classifier()
    if not classifier.is_loaded:
        classifier.load_model()

    yield

    # Shutdown
    DatabaseConnection.close_pool()


app = FastAPI(
    title="Franlince - API de Catalogacion",
    description="API para catalogar pinturas automaticamente usando IA",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(catalog.router)
app.include_router(search.router)
app.include_router(stats.router)
