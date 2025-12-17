"""RAG Engine - orchestrates retrieval and generation"""
from typing import List, Dict, Any, Optional

from app.llm.base import BaseLLM
from app.retriever.base import BaseRetriever
from app.core.prompt_builder import PromptBuilder
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RAGEngine:
    """
    RAG Engine combines:
    - Document retrieval
    - Prompt building
    - LLM generation
    
    Design: LLM-agnostic, can swap implementations
    """
    
    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        retriever: Optional[BaseRetriever] = None
    ):
        """
        Initialize RAG engine
        
        Args:
            llm: LLM implementation (lazy load if None)
            retriever: Retriever implementation (lazy load if None)
        """
        self._llm = llm
        self._retriever = retriever
        self.prompt_builder = PromptBuilder()
    
    @property
    def llm(self) -> BaseLLM:
        """Lazy load LLM"""
        if self._llm is None:
            from app.llm.llama_cpp_llm import LlamaCppLLM
            logger.info("Lazy loading LLM...")
            self._llm = LlamaCppLLM()
        return self._llm
    
    @property
    def retriever(self) -> BaseRetriever:
        """Lazy load retriever"""
        if self._retriever is None:
            from app.retriever.faiss_retriever import FAISSRetriever
            logger.info("Lazy loading retriever...")
            self._retriever = FAISSRetriever()
        return self._retriever
    
    def generate(self, query: str) -> Dict[str, Any]:
        """
        Full RAG pipeline
        
        Args:
            query: User's question
            
        Returns:
            Dict with answer, sources, and metadata
        """
        logger.info(f"RAG query: {query[:50]}...")
        
        # Step 1: Retrieve relevant documents
        documents = self.retriever.search(query)
        logger.info(f"Retrieved {len(documents)} documents")
        
        # Step 2: Build prompt with context
        prompt = self.prompt_builder.build(query, documents)
        
        # Step 3: Generate answer with LLM
        answer = self.llm.generate(prompt)
        
        return {
            "answer": answer,
            "sources": [doc["metadata"] for doc in documents],
            "num_sources": len(documents)
        }
