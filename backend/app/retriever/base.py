"""Base retriever interface"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseRetriever(ABC):
    """
    Abstract base class for document retrievers
    
    Design: Support different vector stores (FAISS, Qdrant, etc.)
    """
    
    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents
        
        Args:
            query: Search query
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            
        Returns:
            List of documents with content and metadata
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get retriever statistics
        
        Returns:
            Dict with index size, dimension, etc.
        """
        pass
