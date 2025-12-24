"""Prompt builder for anti-hallucination"""

from typing import List, Dict, Any
from app.utils.logger import get_logger

logger = get_logger(__name__)


class PromptBuilder:
    """
    Builds prompts that prevent hallucination

    Key principles:
    - Explicit instruction to use ONLY provided context
    - Clear format for context injection
    - Vietnamese language instruction
    """

    SYSTEM_PROMPT = """
Bạn là trợ lý tư vấn tuyển sinh của Đại học FPT (FPT University AI).

YÊU CẦU QUAN TRỌNG:
- Chỉ trả lời dựa trên thông tin được cung cấp trong CONTEXT bên dưới
- Nếu không tìm thấy thông tin trong CONTEXT, hãy trả lời: "Xin lỗi, tôi không có thông tin về vấn đề này. Vui lòng liên hệ phòng tuyển sinh để được hỗ trợ tốt hơn."
- TUYỆT ĐỐI KHÔNG bịa đặt thông tin không có trong văn bản
- Trả lời rõ ràng, chi tiết bằng tiếng Việt
- Giọng điệu thân thiện, chuyên nghiệp, khuyến khích học sinh
- Nếu có thông tin về địa chỉ, học phí, điều kiện, hãy liệt kê đầy đủ và rõ ràng
"""

    def build(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """
        Build complete prompt with context
        
        This method matches the logic from ai/demo.py for consistency
        
        Args:
            query: User's question
            documents: Retrieved documents with content and metadata
            
        Returns:
            Complete prompt string
        """
        # Build context from documents (similar to ai/demo.py)
        context_parts = []
        
        if not documents:
            logger.warning("No documents provided for context")
            return self.build_without_context(query)
        
        for i, doc in enumerate(documents, 1):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            
            # Extract metadata info
            title = metadata.get("title", "Nguồn không tên")
            url = metadata.get("url", "#")
            score = doc.get("score", 0.0)
            
            # Format: Include source info and content
            # This matches the format from ai/demo.py
            context_parts.append(
                f"--- Nguồn {i} ({title} - {url}) [Độ liên quan: {score:.2f}] ---\n{content}"
            )
        
        context = "\n\n".join(context_parts)
        
        # Build final prompt matching ai/demo.py structure
        prompt = f"""{self.SYSTEM_PROMPT}

DỮ LIỆU ĐẦU VÀO (CONTEXT):
{context}

CÂU HỎI NGƯỜI DÙNG:
{query}

TRẢ LỜI:"""
        
        logger.debug(f"Built prompt with {len(documents)} documents, total length: {len(prompt)} chars")
        return prompt

    def build_without_context(self, query: str) -> str:
        """
        Build prompt without context (fallback)

        Args:
            query: User's question

        Returns:
            Prompt string
        """
        return f"""{self.SYSTEM_PROMPT}

CÂU HỎI: {query}

TRẢ LỜI: Xin lỗi, tôi không có thông tin về vấn đề này. Vui lòng liên hệ phòng tuyển sinh để được hỗ trợ tốt hơn."""
