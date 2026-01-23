"""
Configuration settings for Franlince API.
Uses pydantic-settings for environment variable support.
"""

import os
from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database configuration
    db_host: str = Field(default="localhost", alias="DB_HOST")
    db_port: int = Field(default=5432, alias="DB_PORT")
    db_name: str = Field(default="franlince_catalog", alias="DB_NAME")
    db_user: str = Field(default="franlince", alias="DB_USER")
    db_password: str = Field(default="franlince123", alias="DB_PASSWORD")

    # Upload directory
    upload_dir: Path = Field(default=Path("./pinturas_catalogo"), alias="UPLOAD_DIR")

    # Model configuration
    clip_model_name: str = Field(
        default="openai/clip-vit-base-patch32",
        alias="CLIP_MODEL_NAME"
    )

    # API configuration
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    # Image processing
    max_image_size: int = Field(default=1024, alias="MAX_IMAGE_SIZE")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True
    }

    @property
    def db_config(self) -> dict:
        """Returns database configuration as a dictionary."""
        return {
            "host": self.db_host,
            "port": self.db_port,
            "database": self.db_name,
            "user": self.db_user,
            "password": self.db_password
        }

    def ensure_upload_dir(self) -> None:
        """Creates upload directory if it doesn't exist."""
        self.upload_dir.mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Returns cached settings instance."""
    return Settings()
