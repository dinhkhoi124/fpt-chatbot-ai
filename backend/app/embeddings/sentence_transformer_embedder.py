"""Sentence Transformers embedder implementation"""
from typing import List
from chromadb.utils import embedding_functions
import torch

from app.embeddings.base import BaseEmbedder
from app.utils.logger import get_logger

logger = get_logger(__name__)


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Sentence Transformers embedding model (Local/Offline)
    
    Features:
    - Local embedding generation (no API calls)
    - Optimized for multilingual support (especially Vietnamese)
    - GPU acceleration support
    - Same model as used in ai/ directory for consistency
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    ):
        """
        Initialize Sentence Transformers embedder
        
        Args:
            model_name: Model name from sentence-transformers library
        """
        self.model_name = model_name
        
        # Auto-select device (GPU if available, otherwise CPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Sentence Transformer model on {self.device.upper()}...")
        
        # Initialize the embedding function from ChromaDB utils
        # This matches exactly what's used in ai/rag_system.py
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.model_name,
            device=self.device
        )
        
        logger.info(f"Initialized SentenceTransformerEmbedder with model: {self.model_name}")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            # Return zero vector with correct dimension
            return [0.0] * self.get_dimension()
        
        try:
            # Use the ChromaDB embedding function
            # It expects a list and returns a list of embeddings
            embeddings = self.embedding_fn([text])
            return embeddings[0]
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        try:
            # Filter out empty texts
            valid_texts = [t if t and t.strip() else " " for t in texts]
            embeddings = self.embedding_fn(valid_texts)
            return embeddings
            
        except Exception as e:
            logger.error(f"Error embedding batch: {e}")
            # Fallback to individual embedding
            return [self.embed_text(text) for text in texts]
    
    def get_dimension(self) -> int:
        """
        Get embedding dimension
        
        Returns:
            Embedding vector dimension (768 for mpnet-base-v2)
        """
        return 768
    
    @property
    def model(self) -> str:
        """Get model name for compatibility"""
        return self.model_name
