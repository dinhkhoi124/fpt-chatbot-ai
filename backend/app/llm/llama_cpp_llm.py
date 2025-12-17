"""Llama.cpp LLM implementation"""
from typing import Optional, Dict, Any
import time

from app.llm.base import BaseLLM
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class LlamaCppLLM(BaseLLM):
    """
    LLM implementation using llama-cpp-python
    
    Optimized for GPU ≤ 4GB
    """
    
    def __init__(self):
        """Initialize llama.cpp model"""
        self._model = None
        logger.info("LlamaCppLLM initialized (lazy loading)")
    
    def _load_model(self):
        """Lazy load model on first use"""
        if self._model is not None:
            return
        
        try:
            from llama_cpp import Llama
            
            model_path = settings.MODEL_PATH / settings.MODEL_NAME
            logger.info(f"Loading model from: {model_path}")
            
            self._model = Llama(
                model_path=str(model_path),
                n_gpu_layers=settings.GPU_LAYERS,
                n_ctx=2048,  # Context window
                verbose=False
            )
            
            logger.info("Model loaded successfully")
        except ImportError:
            logger.warning("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
            self._model = None
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            self._model = None
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate text using llama.cpp
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens (default from settings)
            temperature: Sampling temperature (default from settings)
            **kwargs: Additional parameters
            
        Returns:
            Generated text
        """
        self._load_model()
        
        if self._model is None:
            error_msg = "Error: Model not loaded. Please check configuration and model file exists."
            logger.error(error_msg)
            return error_msg
        
        max_tokens = max_tokens or settings.MAX_TOKENS
        temperature = temperature or settings.TEMPERATURE
        
        # Start timing
        start_time = time.time()
        
        try:
            logger.info(f"Generating response (max_tokens={max_tokens}, temperature={temperature})...")
            
            response = self._model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=settings.TOP_P,
                echo=False,
                stop=["CÂU HỎI:", "CONTEXT:", "\n\n", "<|", "|>", "support>", "<|end|>", "<|assistant|>"]
            )
            
            answer = response["choices"][0]["text"].strip()
            
            # Clean up special tokens that might slip through
            for token in ["<|assistant|>", "<|end|>", "<|user|>", "support>", "<|", "|>"]:
                answer = answer.replace(token, "").strip()
            
            # Log timing
            elapsed_time = time.time() - start_time
            tokens_generated = len(answer.split())  # Rough estimate
            logger.info(
                f"Generation complete: {len(answer)} chars, "
                f"~{tokens_generated} tokens, {elapsed_time:.2f}s "
                f"({tokens_generated/elapsed_time:.1f} tokens/s)"
            )
            
            return answer
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"Generation error after {elapsed_time:.2f}s: {str(e)}")
            return f"Error during generation: {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "type": "llama.cpp",
            "model_name": settings.MODEL_NAME,
            "gpu_layers": settings.GPU_LAYERS,
            "max_tokens": settings.MAX_TOKENS,
            "temperature": settings.TEMPERATURE,
            "loaded": self._model is not None
        }
