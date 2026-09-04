"""Data models for multi-pipeline comparison in RAGDiag."""

from typing import Literal

from pydantic import BaseModel, Field

from ragdiag.reporting.models import EvaluationReport


class MetricDeltas(BaseModel):
    """Numerical metric deltas computed as Pipeline B minus Pipeline A.

    Attributes:
        precision_at_k: Difference in mean Precision@K (positive means B is higher).
        recall_at_k: Difference in mean Recall@K (positive means B is higher).
        mrr: Difference in Mean Reciprocal Rank (positive means B is higher).
        answer_correctness: Difference in answer correctness rate (None if unjudged).
        groundedness: Difference in groundedness rate (None if unjudged).
        mean_retrieval_ms: Difference in mean retrieval latency in ms (positive means B is slower).
        p95_retrieval_ms: Difference in P95 retrieval latency in ms (positive means B is slower).
    """

    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    answer_correctness: float | None = None
    groundedness: float | None = None
    mean_retrieval_ms: float = 0.0
    p95_retrieval_ms: float = 0.0


class QueryTypeDeltas(BaseModel):
    """Breakdown of metric and diagnostic deltas for a specific query type.

    Attributes:
        query_type: Identifier of the query type (e.g. 'factual', 'reasoning', 'multi-hop').
        recall_at_k: Difference in mean Recall@K for this query type (B - A).
        mrr: Difference in Mean Reciprocal Rank for this query type (B - A).
        answer_correctness: Difference in answer correctness rate (None if unjudged).
        groundedness: Difference in groundedness rate (None if unjudged).
        total_failure_delta: Difference in total non-PASS failure count (negative is better).
        failure_deltas: Differences in per-category failure counts (B - A).
    """

    query_type: str
    recall_at_k: float = 0.0
    mrr: float = 0.0
    answer_correctness: float | None = None
    groundedness: float | None = None
    total_failure_delta: int = 0
    failure_deltas: dict[str, int] = Field(default_factory=dict)


class QueryOutcomeComparison(BaseModel):
    """Per-query outcome transition between Pipeline A and Pipeline B.

    Attributes:
        query_id: Unique query sample identifier.
        diagnosis_a: Diagnostic failure category for Pipeline A.
        diagnosis_b: Diagnostic failure category for Pipeline B.
        recall_a: Recall@K for Pipeline A on this query.
        recall_b: Recall@K for Pipeline B on this query.
        grounded_a: Groundedness status for Pipeline A (None if unjudged).
        grounded_b: Groundedness status for Pipeline B (None if unjudged).
        answer_correct_a: Answer correctness for Pipeline A (None if unjudged).
        answer_correct_b: Answer correctness for Pipeline B (None if unjudged).
        outcome: Transition classification: 'improved', 'regressed', or 'unchanged'.
    """

    query_id: str
    diagnosis_a: str
    diagnosis_b: str
    recall_a: float = 0.0
    recall_b: float = 0.0
    grounded_a: bool | None = None
    grounded_b: bool | None = None
    answer_correct_a: bool | None = None
    answer_correct_b: bool | None = None
    outcome: Literal["improved", "regressed", "unchanged"] = "unchanged"


class MetricRegression(BaseModel):
    """Details of a single metric that exhibited a meaningful regression.

    Attributes:
        metric_name: Name of the regressed metric.
        baseline_value: Metric value in Pipeline A.
        current_value: Metric value in Pipeline B.
        delta: Value difference (Pipeline B minus Pipeline A).
        threshold: Configured tolerance threshold that was exceeded.
        unit: Optional unit string (e.g. 'ms').
    """

    metric_name: str
    baseline_value: float
    current_value: float
    delta: float
    threshold: float
    unit: str = ""


class DiagnosisTransition(BaseModel):
    """Summary of a specific diagnostic category transition among regressed queries.

    Attributes:
        from_category: Category in Pipeline A (baseline).
        to_category: Category in Pipeline B (current).
        transition: Formatted transition string (e.g. 'PASS -> INSUFFICIENT_CONTEXT').
        count: Number of queries undergoing this transition.
        query_ids: List of query IDs that underwent this transition.
    """

    from_category: str
    to_category: str
    transition: str
    count: int = 0
    query_ids: list[str] = Field(default_factory=list)


class RegressedQuery(BaseModel):
    """Detailed information on a single regressed query.

    Attributes:
        query_id: Unique query identifier.
        baseline_diagnosis: Diagnosis in baseline pipeline (Pipeline A).
        current_diagnosis: Diagnosis in current pipeline (Pipeline B).
        transition: Formatted transition string (e.g. 'PASS -> INSUFFICIENT_CONTEXT').
        recall_a: Recall@K in Pipeline A.
        recall_b: Recall@K in Pipeline B.
        grounded_a: Groundedness in Pipeline A (None if unjudged).
        grounded_b: Groundedness in Pipeline B (None if unjudged).
        answer_correct_a: Correctness in Pipeline A (None if unjudged).
        answer_correct_b: Correctness in Pipeline B (None if unjudged).
        reason: Plain-text explanation of why this query regressed.
    """

    query_id: str
    baseline_diagnosis: str
    current_diagnosis: str
    transition: str
    recall_a: float = 0.0
    recall_b: float = 0.0
    grounded_a: bool | None = None
    grounded_b: bool | None = None
    answer_correct_a: bool | None = None
    answer_correct_b: bool | None = None
    reason: str = ""


class RegressionAnalysis(BaseModel):
    """Structured regression analysis contrasting baseline and candidate pipeline outcomes.

    Attributes:
        overall_regression: True if a meaningful overall regression is detected.
        metric_regressions: List of metrics exceeding negative tolerance thresholds.
        diagnosis_regressions: List of category transitions among regressed queries.
        increased_failures: Mapping of failure categories that saw an increase in failures.
        regressed_queries: Detailed list of individual queries that regressed.
        regressed_query_count: Total count of regressed queries.
        important_regressions: Ranked list of highest-priority regression summary strings.
        summary: Deterministic narrative summary explaining the regression verdict.
    """

    overall_regression: bool = False
    metric_regressions: list[MetricRegression] = Field(default_factory=list)
    diagnosis_regressions: list[DiagnosisTransition] = Field(default_factory=list)
    increased_failures: dict[str, int] = Field(default_factory=dict)
    regressed_queries: list[RegressedQuery] = Field(default_factory=list)
    regressed_query_count: int = 0
    important_regressions: list[str] = Field(default_factory=list)
    summary: str = ""


class ComparisonReport(BaseModel):
    """Comprehensive comparison report between two RAG pipeline configurations.

    Attributes:
        dataset_name: Name identifier of the evaluated golden dataset.
        dataset_version: Version identifier of the dataset.
        pipeline_a_name: Identifier of Pipeline A.
        pipeline_b_name: Identifier of Pipeline B.
        pipeline_a_report: Full system-level EvaluationReport for Pipeline A.
        pipeline_b_report: Full system-level EvaluationReport for Pipeline B.
        metric_deltas: Aggregate metric deltas (Pipeline B - Pipeline A).
        diagnosis_deltas: Differences in per-category failure counts (Pipeline B - Pipeline A).
        query_type_deltas: Per-query-type metric and failure deltas.
        query_outcomes: List of matched per-query outcome transitions.
        queries_improved: Total count of queries where Pipeline B improved outcome.
        queries_regressed: Total count of queries where Pipeline B regressed outcome.
        queries_unchanged: Total count of queries with equivalent outcomes.
        quality_winner: Winner based on quality metrics ('Pipeline A', 'Pipeline B', or 'TIE').
        latency_winner: Winner based on retrieval latency ('Pipeline A', 'Pipeline B', or 'TIE').
        overall_winner: Overall winner considering quality and latency trade-offs.
        trade_off: Qualitative trade-off statement, if applicable.
        summary: Deterministic narrative summary explaining the comparison outcome.
        regression_analysis: Dedicated analysis identifying meaningful quality/latency regressions.
    """

    dataset_name: str
    dataset_version: str
    pipeline_a_name: str
    pipeline_b_name: str
    pipeline_a_report: EvaluationReport
    pipeline_b_report: EvaluationReport
    metric_deltas: MetricDeltas = Field(default_factory=MetricDeltas)
    diagnosis_deltas: dict[str, int] = Field(default_factory=dict)
    query_type_deltas: dict[str, QueryTypeDeltas] = Field(default_factory=dict)
    query_outcomes: list[QueryOutcomeComparison] = Field(default_factory=list)
    queries_improved: int = 0
    queries_regressed: int = 0
    queries_unchanged: int = 0
    quality_winner: str = "TIE"
    latency_winner: str = "TIE"
    overall_winner: str = "TIE"
    winner: str = "TIE"
    trade_off: str | None = None
    summary: str = ""
    regression_analysis: RegressionAnalysis = Field(default_factory=RegressionAnalysis)
