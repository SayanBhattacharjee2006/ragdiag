"""Domain models for root-cause failure diagnosis."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class FailureCategory(StrEnum):
    """Controlled taxonomy of primary root-cause failure modes in RAG systems."""

    PASS = "PASS"
    WRONG_CHUNK_RETRIEVED = "WRONG_CHUNK_RETRIEVED"
    WRONG_CHUNK_RANK = "WRONG_CHUNK_RANK"
    INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"
    RETRIEVED_BUT_NOT_GROUNDED = "RETRIEVED_BUT_NOT_GROUNDED"
    ANSWER_INCORRECT = "ANSWER_INCORRECT"
    LATENCY_OUTLIER = "LATENCY_OUTLIER"
    UNKNOWN = "UNKNOWN"


class DiagnosisResult(BaseModel):
    """Structured diagnostic evaluation explaining why a query failed or passed.

    Attributes:
        category: The primary root-cause failure category.
        severity: Impact severity of the diagnosis ('info', 'warning', 'major').
        confidence: Confidence score of the diagnosis (0.0 to 1.0).
        reason: Plain-text explanation of the diagnosis.
        evidence: Concrete factual points justifying the category.
    """

    category: FailureCategory = Field(
        description="The primary root-cause failure category assigned to this query."
    )
    severity: Literal["info", "warning", "major"] = Field(
        description="Impact severity of the diagnosed outcome."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score in the diagnostic attribution (0.0 to 1.0).",
    )
    reason: str = Field(description="Concise rationale explaining the diagnostic verdict.")
    evidence: list[str] = Field(
        default_factory=list,
        description="List of factual evidence strings supporting the verdict.",
    )
