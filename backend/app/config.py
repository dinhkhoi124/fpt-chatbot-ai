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
    VECTORSTORE_PATH: Path = BASE_DIR / "vectorstore"
    CHROMA_DB_PATH: Path = BASE_DIR / "app" / "vector_db"
    CHROMA_COLLECTION_NAME: str = "fpt_university"
    
    # Cloud LLM Settings
    GEMINI_API_KEY: str = ""
    LLM_MODEL: str = "gemini-2.0-flash-exp"
    EMBEDDING_MODEL: str = "text-embedding-004"
    MAX_TOKENS: int = 2048
    TEMPERATURE: float = 0.7
    
    # RAG Settings
    TOP_K: int = 5
    SCORE_THRESHOLD: float = 0.7
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = str(Path(__file__).parent.parent / ".env")
        case_sensitive = True


settings = Settings()
