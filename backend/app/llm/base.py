"""Base LLM interface"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BaseLLM(ABC):
    """
    Abstract base class for LLM implementations
    
    Design: Allow easy swap between llama.cpp, Ollama, etc.
    """
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate text from prompt
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information
        
        Returns:
            Dict with model name, parameters, etc.
        """
        pass
