"""Data models for semantic LLM judging."""

from pydantic import BaseModel, Field


class JudgeResult(BaseModel):
    """Structured semantic evaluation result produced by an LLM judge.

    Attributes:
        answer_correct: Whether the generated answer is semantically accurate
            relative to the expected answer.
        grounded: Whether claims made in the generated answer are strictly supported
            by the retrieved context chunks.
        confidence: Normalized certainty score of the judge in its evaluation (0.0 to 1.0).
        reason: Concise natural language rationale explaining the evaluation decisions.
    """

    answer_correct: bool = Field(
        description=(
            "True if the generated answer is semantically accurate compared to expected answer."
        )
    )
    grounded: bool = Field(
        description=(
            "True if factual claims in the generated answer are supported by retrieved context."
        )
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score of the evaluation, bounded between 0.0 and 1.0.",
    )
    reason: str = Field(
        description="Concise rationale explaining the correctness and groundedness evaluations."
    )
