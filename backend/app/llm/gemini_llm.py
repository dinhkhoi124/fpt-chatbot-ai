"""Google Gemini Cloud LLM implementation"""
from typing import Optional, Dict, Any
from google import genai

from app.llm.base import BaseLLM
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class GeminiLLM(BaseLLM):
    """
    Google Gemini Cloud LLM implementation
    
    Uses Google's Generative AI API for text generation.
    No local inference - fully cloud-based.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        default_max_tokens: int = None,
        default_temperature: float = None
    ):
        """
        Initialize Gemini LLM client
        
        Args:
            api_key: Google API key (defaults from settings)
            model_name: Model name (defaults from settings)
            default_max_tokens: Default maximum tokens to generate
            default_temperature: Default sampling temperature
        """
        self.api_key = api_key or settings.GEMINI_API_KEY
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        self.model_name = model_name or settings.LLM_MODEL
        self.default_max_tokens = default_max_tokens or settings.MAX_TOKENS
        self.default_temperature = default_temperature or settings.TEMPERATURE
        
        # Initialize the client with new google.genai package
        self.client = genai.Client(api_key=self.api_key)
        
        logger.info(f"Initialized Gemini LLM with model: {self.model_name}")
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate text from prompt using Gemini API
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate (overrides default)
            temperature: Sampling temperature (overrides default)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        try:
            # Prepare generation config
            config = {
                "max_output_tokens": max_tokens or self.default_max_tokens,
                "temperature": temperature or self.default_temperature,
            }
            
            # Add any additional kwargs to config
            config.update(kwargs)
            
            logger.debug(f"Generating with prompt length: {len(prompt)} chars")
            
            # Generate response using new API
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )
            
            # Extract text from response
            if response.text:
                logger.debug(f"Generated response length: {len(response.text)} chars")
                return response.text
            else:
                logger.warning("Empty response from Gemini API")
                return ""
                
        except Exception as e:
            logger.error(f"Error generating text with Gemini: {e}")
            raise RuntimeError(f"Gemini generation failed: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information
        
        Returns:
            Dict with model name and configuration
        """
        return {
            "type": "gemini",
            "model_name": self.model_name,
            "provider": "Google Generative AI",
            "max_tokens": self.default_max_tokens,
            "temperature": self.default_temperature,
            "is_cloud": True
        }
