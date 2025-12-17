"""Chat service - orchestrates RAG pipeline"""
from typing import Optional

from app.core.rag_engine import RAGEngine
from app.schemas.chat import ChatResponse
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ChatService:
    """
    Chat service orchestrates the full RAG pipeline:
    1. Retrieve relevant documents
    2. Build context-aware prompt
    3. Generate response with LLM
    """
    
    def __init__(self):
        """Initialize RAG engine (lazy loading)"""
        self._rag_engine: Optional[RAGEngine] = None
    
    @property
    def rag_engine(self) -> RAGEngine:
        """Lazy initialization of RAG engine"""
        if self._rag_engine is None:
            logger.info("Initializing RAG engine...")
            self._rag_engine = RAGEngine()
        return self._rag_engine
    
    async def process_message(self, message: str) -> ChatResponse:
        """
        Process user message through RAG pipeline
        
        Args:
            message: User's question
            
        Returns:
            ChatResponse with answer and sources
        """
        logger.info(f"Processing message: {message[:50]}...")
        
        # TODO: Implement full RAG pipeline
        # For now, return a placeholder response
        return ChatResponse(
            answer="Xin chào! Hệ thống đang được khởi tạo. Vui lòng quay lại sau.",
            sources=[],
            metadata={
                "model": "placeholder",
                "tokens_used": 0
            }
        )
