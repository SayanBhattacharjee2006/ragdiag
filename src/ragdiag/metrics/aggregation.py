"""Aggregation of retrieval metrics and latencies across evaluation results."""

from collections.abc import Sequence

from ragdiag.metrics.latency import calculate_latency_summary
from ragdiag.metrics.models import AggregateEvaluationReport
from ragdiag.metrics.retrieval import precision_at_k, recall_at_k, reciprocal_rank
from ragdiag.models.result import EvaluationResult


def mean_reciprocal_rank(reciprocal_ranks: Sequence[float]) -> float:
    """Calculate Mean Reciprocal Rank (MRR) across multiple queries.

    Args:
        reciprocal_ranks: Sequence of individual per-query reciprocal rank values.

    Returns:
        The arithmetic mean of the reciprocal ranks, or 0.0 if empty.
    """
    if not reciprocal_ranks:
        return 0.0
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def aggregate_metrics(
    results: Sequence[EvaluationResult],
    k: int = 5,
) -> AggregateEvaluationReport:
    """Aggregate retrieval quality and latency statistics across evaluation results.

    Failed queries (`status != 'completed'`) are excluded from retrieval metric
    and latency calculations so that pipeline infrastructure errors do not
    contaminate quality measurements. Failed queries are tracked and reported
    in `failed_queries`.

    Args:
        results: Sequence of `EvaluationResult` instances produced by `Evaluator`.
        k: The retrieval rank depth threshold K (default: 5).

    Returns:
        A validated `AggregateEvaluationReport` containing mean retrieval metrics,
        latency percentiles, and completion counts.
    """
    total = len(results)
    completed = [r for r in results if r.status == "completed"]
    failed = total - len(completed)

    # Aggregate diagnosis categories across all results (completed and failed)
    diagnosis_counts: dict[str, int] = {}
    diagnosis_by_query_type: dict[str, dict[str, int]] = {}
    failure_count = 0

    for r in results:
        cat_str: str | None = None
        if hasattr(r.diagnosis, "category"):
            cat_val = r.diagnosis.category
            cat_str = cat_val.value if hasattr(cat_val, "value") else str(cat_val)
        elif isinstance(r.diagnosis, dict) and "category" in r.diagnosis:
            cat_str = str(r.diagnosis["category"])

        if cat_str:
            diagnosis_counts[cat_str] = diagnosis_counts.get(cat_str, 0) + 1
            if cat_str != "PASS":
                failure_count += 1

            if r.query_type:
                q_type = str(r.query_type)
                if q_type not in diagnosis_by_query_type:
                    diagnosis_by_query_type[q_type] = {}
                diagnosis_by_query_type[q_type][cat_str] = (
                    diagnosis_by_query_type[q_type].get(cat_str, 0) + 1
                )

    if not completed:
        return AggregateEvaluationReport(
            total_queries=total,
            completed_queries=0,
            failed_queries=failed,
            mean_precision_at_k=0.0,
            mean_recall_at_k=0.0,
            mrr=0.0,
            k=k,
            diagnosis_counts=diagnosis_counts,
            diagnosis_by_query_type=diagnosis_by_query_type,
            failure_count=failure_count,
        )

    precisions: list[float] = []
    recalls: list[float] = []
    rrs: list[float] = []
    retrieval_latencies: list[float] = []

    judged_queries = 0
    judge_failures = 0
    correct_count = 0
    grounded_count = 0
    confidences: list[float] = []
    judge_latencies: list[float] = []

    p_key = f"precision_at_{k}"
    r_key = f"recall_at_{k}"

    for r in completed:
        # Extract precomputed metric if present, or compute defensively
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
            if "judge_ms" in r.latency:
                judge_latencies.append(float(r.latency["judge_ms"]))
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
            if "judge_ms" in r.latency:
                judge_latencies.append(float(r.latency["judge_ms"]))

    mean_prec = sum(precisions) / len(precisions) if precisions else 0.0
    mean_rec = sum(recalls) / len(recalls) if recalls else 0.0
    mrr_val = mean_reciprocal_rank(rrs)
    latency_summary = calculate_latency_summary(retrieval_latencies)

    correctness_rate = (correct_count / judged_queries) if judged_queries > 0 else None
    grounded_rate = (grounded_count / judged_queries) if judged_queries > 0 else None
    mean_conf = (sum(confidences) / len(confidences)) if confidences else None
    judge_latency_summary = calculate_latency_summary(judge_latencies) if judge_latencies else None

    return AggregateEvaluationReport(
        total_queries=total,
        completed_queries=len(completed),
        failed_queries=failed,
        mean_precision_at_k=mean_prec,
        mean_recall_at_k=mean_rec,
        mrr=mrr_val,
        k=k,
        retrieval_latency=latency_summary,
        judged_queries=judged_queries,
        judge_failures=judge_failures,
        answer_correctness_rate=correctness_rate,
        groundedness_rate=grounded_rate,
        mean_judge_confidence=mean_conf,
        judge_latency=judge_latency_summary,
        diagnosis_counts=diagnosis_counts,
        diagnosis_by_query_type=diagnosis_by_query_type,
        failure_count=failure_count,
    )
