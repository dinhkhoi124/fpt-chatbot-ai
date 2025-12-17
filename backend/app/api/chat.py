"""Chat API endpoint"""
from fastapi import APIRouter, HTTPException

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Initialize chat service (singleton pattern)
chat_service = ChatService()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint - main entry point for user queries
    
    Args:
        request: ChatRequest with user message
        
    Returns:
        ChatResponse with bot answer and sources
    """
    try:
        logger.info(f"Received chat request: {request.message[:50]}...")
        response = await chat_service.process_message(request.message)
        return response
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
