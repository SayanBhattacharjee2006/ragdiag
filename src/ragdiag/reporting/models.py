"""Data models for system-level evaluation reports and diagnostic intelligence."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from ragdiag.diagnosis.models import FailureCategory, get_action_for_category
from ragdiag.metrics.models import LatencySummary


class HealthGrade(StrEnum):
    """Categorical performance grades derived deterministically from health score."""

    EXCELLENT = "Excellent"
    GOOD = "Good"
    FAIR = "Fair"
    POOR = "Poor"
    CRITICAL = "Critical"


class HealthStatus(StrEnum):
    """Operational health status indicating system stability and quality."""

    HEALTHY = "Healthy"
    DEGRADED = "Degraded"
    UNHEALTHY = "Unhealthy"
    CRITICAL = "Critical"


class HealthProfile(BaseModel):
    """System-level health assessment of a RAG pipeline evaluation run.

    Attributes:
        score: Overall deterministic health score bounded between 0.0 and 100.0.
        grade: Categorical rating ('Excellent', 'Good', 'Fair', 'Poor', 'Critical').
        status: High-level operational status ('Healthy', 'Degraded', 'Unhealthy', 'Critical').
        strengths: List of empirically observed strengths based on evaluation metrics.
        weaknesses: List of identified bottlenecks and failure modes.
        recommendations: Actionable, deduplicated recommendations derived from failure analysis.
    """

    score: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="Overall deterministic health score bounded between 0.0 and 100.0.",
    )
    grade: str = Field(default="Excellent", description="Categorical rating.")
    status: str = Field(default="Healthy", description="High-level operational health status.")
    strengths: list[str] = Field(default_factory=list, description="Empirical strengths.")
    weaknesses: list[str] = Field(default_factory=list, description="Identified weak areas.")
    recommendations: list[str] = Field(
        default_factory=list, description="Actionable recommendations."
    )


class TopFailure(BaseModel):
    """Structured representation of a high-priority query failure.

    Attributes:
        query_id: Unique identifier of the query sample.
        query: Query text.
        category: Diagnostic failure category.
        severity: Failure severity level ('info', 'warning', 'major').
        confidence: Classification confidence score between 0.0 and 1.0.
        reason: Concise diagnostic verdict explaining the failure.
        action: Actionable recommendation explaining what to investigate or improve.
        evidence: Concrete diagnostic signals supporting the verdict.
    """

    query_id: str
    query: str
    category: FailureCategory
    severity: Literal["info", "warning", "major"]
    confidence: float
    reason: str
    action: str = Field(
        default="",
        description="Actionable recommendation explaining what to investigate or improve.",
    )
    evidence: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _set_default_action(self) -> "TopFailure":
        if not self.action:
            self.action = get_action_for_category(self.category)
        return self


class RetrievalSummary(BaseModel):
    """Aggregate summary of retrieval-quality metrics.

    Attributes:
        mean_precision_at_k: Average Precision@K over completed queries.
        mean_recall_at_k: Average Recall@K over completed queries.
        mrr: Mean Reciprocal Rank over completed queries.
        k: The rank cutoff parameter K used for retrieval evaluation.
    """

    mean_precision_at_k: float = 0.0
    mean_recall_at_k: float = 0.0
    mrr: float = 0.0
    k: int = 5


class SemanticSummary(BaseModel):
    """Aggregate summary of semantic generation quality evaluated by the LLM judge.

    Attributes:
        answer_correctness_rate: Proportion of evaluated queries where answer was correct.
        groundedness_rate: Proportion of evaluated queries where claims were grounded in context.
        mean_judge_confidence: Average judge confidence score across evaluated queries.
    """

    answer_correctness_rate: float | None = None
    groundedness_rate: float | None = None
    mean_judge_confidence: float | None = None


class QueryTypeMetrics(BaseModel):
    """Breakdown of retrieval, semantic, and diagnostic metrics for a query type.

    Attributes:
        query_type: Identifier of the query type (e.g. 'factual', 'reasoning', 'multi-hop').
        total_queries: Total number of queries belonging to this query type.
        completed_queries: Completed query count.
        failed_queries: Infrastructure execution failure count.
        mean_precision_at_k: Average Precision@K over completed queries of this type.
        mean_recall_at_k: Average Recall@K over completed queries of this type.
        mrr: Mean Reciprocal Rank over completed queries of this type.
        answer_correctness_rate: Answer correctness rate when judge results exist.
        groundedness_rate: Groundedness rate when judge results exist.
        diagnosis_counts: Count of queries in each failure category for this query type.
    """

    query_type: str
    total_queries: int = 0
    completed_queries: int = 0
    failed_queries: int = 0
    mean_precision_at_k: float = 0.0
    mean_recall_at_k: float = 0.0
    mrr: float = 0.0
    answer_correctness_rate: float | None = None
    groundedness_rate: float | None = None
    diagnosis_counts: dict[str, int] = Field(default_factory=dict)


class EvaluationReport(BaseModel):
    """System-level diagnostic evaluation report for a RAG pipeline against a golden dataset.

    Contains complete structured evidence for terminal presentation, JSON export,
    and downstream comparison.

    Attributes:
        dataset_name: Identifier of the evaluated golden dataset.
        dataset_version: Version of the golden dataset.
        pipeline_name: Identifier of the evaluated RAG pipeline adapter.
        total_queries: Total count of queries in the evaluation run.
        completed_queries: Queries that successfully finished execution.
        failed_queries: Queries that encountered infrastructure or runtime exceptions.
        judged_queries: Queries successfully evaluated by the LLM judge.
        judge_failures: Queries where the LLM judge threw an exception or timed out.
        retrieval: Retrieval metrics summary across completed queries.
        semantic: Semantic quality summary, or None if no judge was configured.
        latency: Statistical summary of retrieval latencies.
        diagnosis_counts: Complete failure category distribution (all categories represented).
        diagnosis_by_query_type: Failure category counts broken down by query type.
        metrics_by_query_type: Retrieval and semantic metrics broken down by query type.
        top_failures: Deterministically ranked list of the most critical query failures.
        overall_insights: Rule-based deterministic insights highlighting system health.
    """

    dataset_name: str = ""
    dataset_version: str = ""
    pipeline_name: str | None = None
    total_queries: int = 0
    completed_queries: int = 0
    failed_queries: int = 0
    judged_queries: int = 0
    judge_failures: int = 0
    retrieval: RetrievalSummary = Field(default_factory=RetrievalSummary)
    semantic: SemanticSummary | None = None
    latency: LatencySummary = Field(default_factory=LatencySummary)
    diagnosis_counts: dict[str, int] = Field(default_factory=dict)
    diagnosis_by_query_type: dict[str, dict[str, int]] = Field(default_factory=dict)
    metrics_by_query_type: dict[str, QueryTypeMetrics] = Field(default_factory=dict)
    top_failures: list[TopFailure] = Field(default_factory=list)
    overall_insights: list[str] = Field(default_factory=list)
    health_profile: HealthProfile = Field(
        default_factory=HealthProfile,
        description="Overall system health assessment including score, grade, and recommendations.",
    )
