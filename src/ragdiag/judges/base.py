"""Base abstract interface and context formatting for LLM judges."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from ragdiag.judges.models import JudgeResult
from ragdiag.models.chunk import RetrievedChunk


def format_context(context: Sequence[RetrievedChunk]) -> str:
    """Format a sequence of retrieved chunks into a structured evidence string.

    Retains chunk IDs to facilitate attribution and diagnosis:
        [doc_17]
        Refunds are processed within 7 business days.

        [doc_21]
        Additional policy text...

    Args:
        context: Sequence of `RetrievedChunk` instances.

    Returns:
        Formatted context string. If empty, returns 'No context retrieved.'.
    """
    if not context:
        return "No context retrieved."

    blocks = [f"[{chunk.id}]\n{chunk.text.strip()}" for chunk in context]
    return "\n\n".join(blocks)


class Judge(ABC):
    """Abstract interface for LLM-powered semantic evaluation judges.

    All LLM judge implementations (OpenAI, Anthropic, Gemini, local, etc.)
    must conform to this interface to keep RAGDiag decoupled from specific providers.
    """

    @abstractmethod
    def evaluate(
        self,
        query: str,
        expected_answer: str,
        generated_answer: str,
        context: Sequence[RetrievedChunk],
    ) -> JudgeResult:
        """Evaluate semantic answer correctness and groundedness for a query.

        Args:
            query: The user query string.
            expected_answer: Ground truth reference answer from the golden dataset.
            generated_answer: The answer synthesized by the RAG pipeline.
            context: The chunks retrieved by the RAG pipeline.

        Returns:
            A validated `JudgeResult` containing boolean judgements, confidence, and reason.

        Raises:
            JudgeError: If the judge encounters an unrecoverable failure.
        """
