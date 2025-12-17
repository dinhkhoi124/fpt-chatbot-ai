"""Configuration management for the application"""
import os
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # API Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    MODEL_PATH: Path = BASE_DIR / "models"
    VECTORSTORE_PATH: Path = BASE_DIR / "vectorstore"
    
    # LLM Settings
    MODEL_NAME: str = "Phi-3-mini-4k-instruct-q4.gguf"
    MAX_TOKENS: int = 512
    TEMPERATURE: float = 0.3
    TOP_P: float = 0.9
    GPU_LAYERS: int = 32  # For GPU ≤ 4GB (Phi-3 mini optimized)
    
    # RAG Settings
    TOP_K: int = 5
    SCORE_THRESHOLD: float = 0.7
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
