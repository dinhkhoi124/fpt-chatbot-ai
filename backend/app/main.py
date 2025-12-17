"""FastAPI entrypoint for FPT Chatbot Backend"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat, health
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="FPT Chatbot AI Backend",
    description="Inference service for admission chatbot with Local LLM + RAG",
    version="0.1.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(chat.router, prefix="/api", tags=["chat"])


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting FPT Chatbot AI Backend...")
    logger.info(f"Model path: {settings.MODEL_PATH}")
    logger.info(f"Vector store path: {settings.VECTORSTORE_PATH}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down FPT Chatbot AI Backend...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
