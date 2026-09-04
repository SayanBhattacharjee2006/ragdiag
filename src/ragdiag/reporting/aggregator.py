"""Report aggregation engine turning EvaluationResults into system-level EvaluationReports."""

from ragdiag.diagnosis.models import FailureCategory, get_action_for_category
from ragdiag.metrics.aggregation import mean_reciprocal_rank
from ragdiag.metrics.latency import calculate_latency_summary
from ragdiag.metrics.retrieval import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from ragdiag.models.result import EvaluationResult
from ragdiag.reporting.insights import generate_insights
from ragdiag.reporting.models import (
    EvaluationReport,
    QueryTypeMetrics,
    RetrievalSummary,
    SemanticSummary,
    TopFailure,
)

TAXONOMY_ORDER: list[str] = [cat.value for cat in FailureCategory]

SEVERITY_RANKS: dict[str, int] = {
    "major": 0,
    "warning": 1,
    "info": 2,
}

STANDARD_QUERY_TYPES: list[str] = ["factual", "reasoning", "multi-hop"]


def _get_category_str(result: EvaluationResult) -> str:
    """Extract string failure category from an EvaluationResult."""
    if hasattr(result.diagnosis, "category"):
        cat_val = result.diagnosis.category
        return cat_val.value if hasattr(cat_val, "value") else str(cat_val)
    if isinstance(result.diagnosis, dict) and "category" in result.diagnosis:
        return str(result.diagnosis["category"])
    return FailureCategory.UNKNOWN.value


def _get_severity(result: EvaluationResult) -> str:
    """Extract string severity from an EvaluationResult."""
    if hasattr(result.diagnosis, "severity"):
        return str(result.diagnosis.severity)
    if isinstance(result.diagnosis, dict) and "severity" in result.diagnosis:
        return str(result.diagnosis["severity"])
    return "major"


def _get_confidence(result: EvaluationResult) -> float:
    """Extract float confidence score from an EvaluationResult."""
    if hasattr(result.diagnosis, "confidence"):
        return float(result.diagnosis.confidence)
    if isinstance(result.diagnosis, dict) and "confidence" in result.diagnosis:
        val = result.diagnosis["confidence"]
        return float(val) if isinstance(val, (int, float)) else 1.0
    return 1.0


def _get_reason(result: EvaluationResult) -> str:
    """Extract diagnosis reason string from an EvaluationResult."""
    if hasattr(result.diagnosis, "reason"):
        return str(result.diagnosis.reason)
    if isinstance(result.diagnosis, dict) and "reason" in result.diagnosis:
        return str(result.diagnosis["reason"])
    return result.error or "Evaluation failure"


def _get_action(result: EvaluationResult) -> str:
    """Extract diagnosis action string from an EvaluationResult."""
    if hasattr(result.diagnosis, "action") and result.diagnosis.action:
        return str(result.diagnosis.action)
    if isinstance(result.diagnosis, dict) and "action" in result.diagnosis:
        return str(result.diagnosis["action"])
    cat_val = _get_category_str(result)
    return get_action_for_category(cat_val)


def _get_evidence(result: EvaluationResult) -> list[str]:
    """Extract diagnosis evidence list from an EvaluationResult."""
    if hasattr(result.diagnosis, "evidence"):
        return list(result.diagnosis.evidence)
    if isinstance(result.diagnosis, dict) and "evidence" in result.diagnosis:
        ev = result.diagnosis["evidence"]
        return list(ev) if isinstance(ev, list) else [str(ev)]
    return [result.error] if result.error else []


def build_report(
    results: list[EvaluationResult],
    dataset_name: str = "",
    dataset_version: str = "",
    pipeline_name: str | None = None,
    k: int = 5,
    max_top_failures: int = 5,
) -> EvaluationReport:
    """Aggregate a sequence of EvaluationResults into a complete EvaluationReport.

    Consumes already-captured evidence and metric calculations without introducing
    secondary LLM calls or duplicate metric formulas.

    Args:
        results: Sequence of per-query EvaluationResult instances.
        dataset_name: Name identifier of the dataset.
        dataset_version: Version identifier of the dataset.
        pipeline_name: Name of the pipeline adapter.
        k: Retrieval rank cutoff K.
        max_top_failures: Maximum number of top failures to select.

    Returns:
        Fully populated, validated `EvaluationReport`.
    """
    total = len(results)
    completed = [r for r in results if r.status == "completed"]
    failed = total - len(completed)

    # 1. Failure category counts across all queries (retaining all 8 categories)
    diagnosis_counts: dict[str, int] = {cat: 0 for cat in TAXONOMY_ORDER}
    for r in results:
        cat_name = _get_category_str(r)
        diagnosis_counts[cat_name] = diagnosis_counts.get(cat_name, 0) + 1

    # 2. Overall Retrieval Summary across completed queries
    precisions: list[float] = []
    recalls: list[float] = []
    rrs: list[float] = []
    retrieval_latencies: list[float] = []

    p_key = f"precision_at_{k}"
    r_key = f"recall_at_{k}"

    judged_queries = 0
    judge_failures = 0
    correct_count = 0
    grounded_count = 0
    confidences: list[float] = []

    for r in completed:
        # Precomputed or defensively computed retrieval metrics
        if p_key in r.metrics and isinstance(r.metrics[p_key], (int, float)):
            p_val = float(r.metrics[p_key])
        else:
            p_val = precision_at_k(r.expected_chunk_ids, r.retrieved_chunks, k=k)

        if r_key in r.metrics and isinstance(r.metrics[r_key], (int, float)):
            r_val = float(r.metrics[r_key])
        else:
            r_val = recall_at_k(r.expected_chunk_ids, r.retrieved_chunks, k=k)

        if "reciprocal_rank" in r.metrics and isinstance(
            r.metrics["reciprocal_rank"], (int, float)
        ):
            rr_val = float(r.metrics["reciprocal_rank"])
        else:
            rr_val = reciprocal_rank(r.expected_chunk_ids, r.retrieved_chunks)

        precisions.append(p_val)
        recalls.append(r_val)
        rrs.append(rr_val)

        if "retrieval_ms" in r.latency:
            retrieval_latencies.append(float(r.latency["retrieval_ms"]))

        # Semantic judge metrics
        if r.judge_error is not None:
            judge_failures += 1
        elif "answer_correct" in r.metrics:
            judged_queries += 1
            if r.metrics.get("answer_correct") is True:
                correct_count += 1
            if r.metrics.get("grounded") is True:
                grounded_count += 1
            if "judge_confidence" in r.metrics and isinstance(
                r.metrics["judge_confidence"], (int, float)
            ):
                confidences.append(float(r.metrics["judge_confidence"]))

    mean_prec = (sum(precisions) / len(precisions)) if precisions else 0.0
    mean_rec = (sum(recalls) / len(recalls)) if recalls else 0.0
    mrr_val = mean_reciprocal_rank(rrs) if rrs else 0.0

    retrieval_summary = RetrievalSummary(
        mean_precision_at_k=mean_prec,
        mean_recall_at_k=mean_rec,
        mrr=mrr_val,
        k=k,
    )

    # 3. Overall Semantic Summary
    semantic_summary: SemanticSummary | None = None
    if judged_queries > 0 or judge_failures > 0:
        c_rate = (correct_count / judged_queries) if judged_queries > 0 else None
        g_rate = (grounded_count / judged_queries) if judged_queries > 0 else None
        mean_conf = (sum(confidences) / len(confidences)) if confidences else None
        semantic_summary = SemanticSummary(
            answer_correctness_rate=c_rate,
            groundedness_rate=g_rate,
            mean_judge_confidence=mean_conf,
        )

    # 4. Latency Summary
    latency_summary = calculate_latency_summary(retrieval_latencies)

    # 5. Query-Type Breakdown
    # Discover all present query types, prioritizing standard order
    seen_types = set()
    for r in results:
        if r.query_type:
            seen_types.add(str(r.query_type))

    ordered_types: list[str] = [qt for qt in STANDARD_QUERY_TYPES if qt in seen_types]
    extra_types = sorted(seen_types - set(STANDARD_QUERY_TYPES))
    ordered_types.extend(extra_types)

    diagnosis_by_query_type: dict[str, dict[str, int]] = {}
    metrics_by_query_type: dict[str, QueryTypeMetrics] = {}

    for qt in ordered_types:
        qt_all = [r for r in results if r.query_type == qt]
        qt_completed = [r for r in completed if r.query_type == qt]
        qt_failed = len(qt_all) - len(qt_completed)

        # Diagnosis counts for this query type (all 8 categories represented)
        qt_diag_counts: dict[str, int] = {cat: 0 for cat in TAXONOMY_ORDER}
        for r in qt_all:
            cat_name = _get_category_str(r)
            qt_diag_counts[cat_name] = qt_diag_counts.get(cat_name, 0) + 1
        diagnosis_by_query_type[qt] = qt_diag_counts

        # Retrieval metrics for this query type
        qt_p: list[float] = []
        qt_r: list[float] = []
        qt_rrs: list[float] = []
        qt_correct = 0
        qt_grounded = 0
        qt_judged = 0

        for r in qt_completed:
            if p_key in r.metrics and isinstance(r.metrics[p_key], (int, float)):
                qt_p.append(float(r.metrics[p_key]))
            else:
                qt_p.append(precision_at_k(r.expected_chunk_ids, r.retrieved_chunks, k=k))

            if r_key in r.metrics and isinstance(r.metrics[r_key], (int, float)):
                qt_r.append(float(r.metrics[r_key]))
            else:
                qt_r.append(recall_at_k(r.expected_chunk_ids, r.retrieved_chunks, k=k))

            if "reciprocal_rank" in r.metrics and isinstance(
                r.metrics["reciprocal_rank"], (int, float)
            ):
                qt_rrs.append(float(r.metrics["reciprocal_rank"]))
            else:
                qt_rrs.append(reciprocal_rank(r.expected_chunk_ids, r.retrieved_chunks))

            if r.judge_error is None and "answer_correct" in r.metrics:
                qt_judged += 1
                if r.metrics.get("answer_correct") is True:
                    qt_correct += 1
                if r.metrics.get("grounded") is True:
                    qt_grounded += 1

        qt_mean_p = (sum(qt_p) / len(qt_p)) if qt_p else 0.0
        qt_mean_r = (sum(qt_r) / len(qt_r)) if qt_r else 0.0
        qt_mrr = mean_reciprocal_rank(qt_rrs) if qt_rrs else 0.0
        qt_c_rate = (qt_correct / qt_judged) if qt_judged > 0 else None
        qt_g_rate = (qt_grounded / qt_judged) if qt_judged > 0 else None

        metrics_by_query_type[qt] = QueryTypeMetrics(
            query_type=qt,
            total_queries=len(qt_all),
            completed_queries=len(qt_completed),
            failed_queries=qt_failed,
            mean_precision_at_k=qt_mean_p,
            mean_recall_at_k=qt_mean_r,
            mrr=qt_mrr,
            answer_correctness_rate=qt_c_rate,
            groundedness_rate=qt_g_rate,
            diagnosis_counts=qt_diag_counts,
        )

    # 6. Deterministic Top Failures Selection
    # Non-PASS results ranked by:
    # 1. severity (major < warning < info in SEVERITY_RANKS)
    # 2. confidence (descending, so -confidence)
    # 3. category priority (index in TAXONOMY_ORDER)
    # 4. query_id (lexicographical ascending)
    non_pass_results = [r for r in results if _get_category_str(r) != "PASS"]

    def failure_sort_key(res: EvaluationResult) -> tuple[int, float, int, str]:
        sev = _get_severity(res)
        sev_rank = SEVERITY_RANKS.get(sev, 1)
        conf = _get_confidence(res)
        cat_str = _get_category_str(res)
        cat_index = TAXONOMY_ORDER.index(cat_str) if cat_str in TAXONOMY_ORDER else 99
        return (sev_rank, -conf, cat_index, res.query_id)

    sorted_failures = sorted(non_pass_results, key=failure_sort_key)
    top_failures_list: list[TopFailure] = []

    for r in sorted_failures[:max_top_failures]:
        cat_val = _get_category_str(r)
        try:
            cat_enum = FailureCategory(cat_val)
        except ValueError:
            cat_enum = FailureCategory.UNKNOWN

        sev_val = _get_severity(r)
        severity_typed = sev_val if sev_val in ("info", "warning", "major") else "major"

        top_failures_list.append(
            TopFailure(
                query_id=r.query_id,
                query=r.query,
                category=cat_enum,
                severity=severity_typed,  # type: ignore[arg-type]
                confidence=_get_confidence(r),
                reason=_get_reason(r),
                action=_get_action(r),
                evidence=_get_evidence(r),
            )
        )

    # 7. Construct Initial Report & Generate Insights
    report = EvaluationReport(
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        pipeline_name=pipeline_name,
        total_queries=total,
        completed_queries=len(completed),
        failed_queries=failed,
        judged_queries=judged_queries,
        judge_failures=judge_failures,
        retrieval=retrieval_summary,
        semantic=semantic_summary,
        latency=latency_summary,
        diagnosis_counts=diagnosis_counts,
        diagnosis_by_query_type=diagnosis_by_query_type,
        metrics_by_query_type=metrics_by_query_type,
        top_failures=top_failures_list,
        overall_insights=[],
    )

    report.overall_insights = generate_insights(report)
    return report
