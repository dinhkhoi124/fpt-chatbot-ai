"""Base embedder interface"""
from abc import ABC, abstractmethod
from typing import List, Union


class BaseEmbedder(ABC):
    """
    Abstract base class for text embedding models
    
    Design: Support different embedding services (Google, OpenAI, etc.)
    """
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """
        Get embedding dimension
        
        Returns:
            Dimension of embedding vectors
        """
        pass
