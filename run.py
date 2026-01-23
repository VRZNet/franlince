#!/usr/bin/env python3
"""
Entry point for running the Franlince API.

Usage:
    python run.py
"""

import uvicorn
from src.api.main import app
from src.core.config import get_settings


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
