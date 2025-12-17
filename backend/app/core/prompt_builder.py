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
Bạn là trợ lý tư vấn tuyển sinh của trường Đại học FPT.

QUAN TRỌNG:
- Chỉ trả lời dựa trên thông tin được cung cấp trong CONTEXT bên dưới
- Nếu không tìm thấy thông tin trong CONTEXT, hãy trả lời: "Xin lỗi, tôi không có thông tin về vấn đề này. Vui lòng liên hệ phòng tuyển sinh để được hỗ trợ tốt hơn."
- KHÔNG đưa ra thông tin bạn không chắc chắn
- Trả lời ngắn gọn, rõ ràng bằng tiếng Việt
"""

    def build(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """
        Build complete prompt with context

        Args:
            query: User's question
            documents: Retrieved documents with content and metadata

        Returns:
            Complete prompt string
        """
        # Build context from documents
        context_parts = []
        for i, doc in enumerate(documents, 1):
            content = doc.get("content", "")
            context_parts.append(f"[Tài liệu {i}]\n{content}")

        context = "\n\n".join(context_parts)

        # Build final prompt
        prompt = f"""{self.SYSTEM_PROMPT}

CONTEXT:
{context}

CÂU HỎI: {query}

TRẢ LỜI:"""

        logger.debug(f"Built prompt with {len(documents)} documents")
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
