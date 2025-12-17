"""Chat request and response schemas"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request schema for chat endpoint"""
    
    message: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User's question or message"
    )
    
    session_id: Optional[str] = Field(
        None,
        description="Optional session ID for conversation tracking"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Học phí ngành Công nghệ thông tin là bao nhiêu?",
                "session_id": "user_123_session_456"
            }
        }


class Source(BaseModel):
    """Source document information"""
    
    title: Optional[str] = None
    url: Optional[str] = None
    excerpt: Optional[str] = None


class ChatResponse(BaseModel):
    """Response schema for chat endpoint"""
    
    answer: str = Field(
        ...,
        description="Generated answer from the chatbot"
    )
    
    sources: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of source documents used"
    )
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (model, tokens, etc.)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Học phí ngành Công nghệ thông tin năm 2024 là 29.000.000 VNĐ/năm.",
                "sources": [
                    {
                        "title": "Bảng học phí 2024",
                        "url": "https://example.com/tuition"
                    }
                ],
                "metadata": {
                    "model": "Llama-3.1-8B",
                    "tokens_used": 150
                }
            }
        }
