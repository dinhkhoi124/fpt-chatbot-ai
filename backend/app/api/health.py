"""Health check endpoint"""
from fastapi import APIRouter
from datetime import datetime

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint to verify service is running"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "FPT Chatbot AI Backend",
        "version": "0.1.0"
    }
