"""Data models for retrieval metrics and latency analysis."""

from pydantic import BaseModel, Field


class RetrievalMetricResult(BaseModel):
    """Retrieval quality metrics computed for a single query.

    Attributes:
        precision_at_k: Ratio of relevant chunks in the top-K retrieved results.
        recall_at_k: Proportion of all relevant chunks found in the top-K retrieved results.
        reciprocal_rank: Reciprocal of the 1-indexed rank of the first relevant chunk (or 0.0).
        k: The cutoff threshold K used for Precision@K and Recall@K.
    """

    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    k: int = 5


class LatencySummary(BaseModel):
    """Summary statistics for pipeline execution latency in milliseconds.

    Attributes:
        count: Number of latency data points analyzed.
        mean_ms: Arithmetic average latency.
        p50_ms: 50th percentile (median) latency.
        p95_ms: 95th percentile latency.
        p99_ms: 99th percentile latency.
        min_ms: Minimum observed latency.
        max_ms: Maximum observed latency.
    """

    count: int = 0
    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0


class AggregateEvaluationReport(BaseModel):
    """Aggregated evaluation metrics across all completed query samples.

    Attributes:
        total_queries: Total number of evaluated queries in the dataset.
        completed_queries: Number of queries that completed successfully.
        failed_queries: Number of queries that encountered execution failures.
        mean_precision_at_k: Average Precision@K over completed queries.
        mean_recall_at_k: Average Recall@K over completed queries.
        mrr: Mean Reciprocal Rank over completed queries.
        k: The cutoff parameter K used for retrieval metrics.
        retrieval_latency: Latency summary statistics for the retrieval stage.
    """

    total_queries: int = 0
    completed_queries: int = 0
    failed_queries: int = 0
    mean_precision_at_k: float = 0.0
    mean_recall_at_k: float = 0.0
    mrr: float = 0.0
    k: int = 5
    retrieval_latency: LatencySummary = Field(default_factory=LatencySummary)
    judged_queries: int = 0
    judge_failures: int = 0
    answer_correctness_rate: float | None = None
    groundedness_rate: float | None = None
    mean_judge_confidence: float | None = None
    judge_latency: LatencySummary | None = None
    diagnosis_counts: dict[str, int] = Field(default_factory=dict)
    diagnosis_by_query_type: dict[str, dict[str, int]] = Field(default_factory=dict)
    failure_count: int = 0
