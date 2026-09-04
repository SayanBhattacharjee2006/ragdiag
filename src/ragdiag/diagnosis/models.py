"""Domain models for root-cause failure diagnosis."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


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

    @property
    def action(self) -> str:
        """Actionable recommendation for this failure category."""
        return FAILURE_ACTIONS[self]


FAILURE_ACTIONS: dict[FailureCategory, str] = {
    FailureCategory.PASS: "No action required.",
    FailureCategory.WRONG_CHUNK_RETRIEVED: (
        "Review the retrieval strategy and query formulation; "
        "the pipeline retrieved irrelevant context."
    ),
    FailureCategory.WRONG_CHUNK_RANK: (
        "Improve ranking or reranking so relevant context appears earlier."
    ),
    FailureCategory.INSUFFICIENT_CONTEXT: (
        "Increase retrieval depth or improve retrieval coverage "
        "so all required context is retrieved."
    ),
    FailureCategory.RETRIEVED_BUT_NOT_GROUNDED: (
        "Improve answer grounding so the generated response "
        "stays supported by the retrieved context."
    ),
    FailureCategory.ANSWER_INCORRECT: (
        "Review the generation prompt, model behavior, and context usage for answer correctness."
    ),
    FailureCategory.LATENCY_OUTLIER: (
        "Investigate slow retrieval or generation paths and optimize the latency bottleneck."
    ),
    FailureCategory.UNKNOWN: ("Inspect the pipeline execution error and underlying integration."),
}


def get_action_for_category(category: FailureCategory | str) -> str:
    """Return the deterministic actionable recommendation for a failure category."""
    if isinstance(category, FailureCategory):
        return FAILURE_ACTIONS.get(category, FAILURE_ACTIONS[FailureCategory.UNKNOWN])
    if isinstance(category, str):
        try:
            enum_val = FailureCategory(category)
            return FAILURE_ACTIONS[enum_val]
        except ValueError:
            return FAILURE_ACTIONS[FailureCategory.UNKNOWN]
    return FAILURE_ACTIONS[FailureCategory.UNKNOWN]


class DiagnosisResult(BaseModel):
    """Structured diagnostic evaluation explaining why a query failed or passed.

    Attributes:
        category: The primary root-cause failure category.
        severity: Impact severity of the diagnosis ('info', 'warning', 'major').
        confidence: Confidence score of the diagnosis (0.0 to 1.0).
        reason: Plain-text explanation of the diagnosis.
        action: Actionable recommendation explaining what a developer should investigate or improve.
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
    action: str = Field(
        default="",
        description="Actionable recommendation explaining what to investigate or improve.",
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="List of factual evidence strings supporting the verdict.",
    )

    @model_validator(mode="after")
    def _set_default_action(self) -> "DiagnosisResult":
        if not self.action:
            self.action = get_action_for_category(self.category)
        return self
