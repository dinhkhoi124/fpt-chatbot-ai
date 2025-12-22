"""Google Gemini embedder implementation"""
from typing import List
from google import genai

from app.embeddings.base import BaseEmbedder
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class GeminiEmbedder(BaseEmbedder):
    """
    Gemini embedding model using Google GenAI SDK
    
    Features:
    - Cloud-based embedding via API
    - Support for batch processing
    - Vietnamese text support
    """
    
    def __init__(
        self,
        api_key: str = None,
        model: str = None
    ):
        """
        Initialize Gemini embedder
        
        Args:
            api_key: Google API key (default from settings)
            model: Model name (default from settings)
        """
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model = model or settings.EMBEDDING_MODEL
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required. Set it in .env or pass to constructor")
        
        # Initialize GenAI client
        self.client = genai.Client(api_key=self.api_key)
        logger.info(f"Initialized GeminiEmbedder with model: {self.model}")
    
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
            return [0.0] * self.get_dimension()
        
        try:
            result = self.client.models.embed_content(
                model=self.model,
                contents=text
            )
            
            # Extract embedding from response
            if hasattr(result, 'embeddings') and result.embeddings:
                embedding = result.embeddings[0]
                if hasattr(embedding, 'values'):
                    return list(embedding.values)
            
            logger.error("Failed to extract embedding from response")
            return [0.0] * self.get_dimension()
            
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
        
        embeddings = []
        for text in texts:
            try:
                embedding = self.embed_text(text)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Error embedding text in batch: {e}")
                # Use zero vector as fallback
                embeddings.append([0.0] * self.get_dimension())
        
        return embeddings
    
    def get_dimension(self) -> int:
        """
        Get embedding dimension for the model
        
        Returns:
            Dimension of embedding vectors
        """
        # Gemini text-embedding-004 produces 768-dimensional embeddings
        dimension_map = {
            "text-embedding-004": 768,
            "embedding-001": 768,
        }
        return dimension_map.get(self.model, 768)
