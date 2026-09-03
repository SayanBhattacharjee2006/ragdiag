"""Deterministic rule-based insight generation for system-level evaluation reports."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ragdiag.reporting.models import EvaluationReport


CATEGORY_DISPLAY_NAMES: dict[str, str] = {
    "WRONG_CHUNK_RETRIEVED": "Wrong-chunk retrieval",
    "WRONG_CHUNK_RANK": "Suboptimal chunk ranking",
    "INSUFFICIENT_CONTEXT": "Insufficient context retrieval",
    "RETRIEVED_BUT_NOT_GROUNDED": "Ungrounded generation (hallucination)",
    "ANSWER_INCORRECT": "Incorrect answer generation",
    "LATENCY_OUTLIER": "Retrieval latency outlier",
    "UNKNOWN": "Pipeline execution failure",
}


def generate_insights(report: "EvaluationReport") -> list[str]:
    """Generate deterministic, human-readable insights from an EvaluationReport.

    Rules evaluate measured retrieval, semantic, diagnostic, and latency signals
    using fixed thresholds without invoking an LLM.

    Thresholds:
        - Weakest query type recall gap: difference >= 0.10 between max and min recall.
        - Absolute low recall threshold: < 0.70.
        - Groundedness gap: difference >= 0.10 between max and min groundedness.
        - Latency outlier concern: any query flagged as LATENCY_OUTLIER, or P95 > 1000ms.

    Args:
        report: Fully aggregated `EvaluationReport` instance.

    Returns:
        List of concise, factual insight strings.
    """
    insights: list[str] = []

    # 1. All Passed Check
    total_failures = sum(count for cat, count in report.diagnosis_counts.items() if cat != "PASS")
    if report.completed_queries > 0 and total_failures == 0 and report.failed_queries == 0:
        insights.append("All evaluated queries passed retrieval, context, and quality checks.")
        return insights

    # 2. Dominant Failure Mode
    if total_failures > 0:
        non_pass_counts = {
            cat: cnt for cat, cnt in report.diagnosis_counts.items() if cat != "PASS" and cnt > 0
        }
        if non_pass_counts:
            dominant_cat, max_cnt = max(non_pass_counts.items(), key=lambda item: item[1])
            cat_name = CATEGORY_DISPLAY_NAMES.get(dominant_cat, dominant_cat)
            pct = (max_cnt / total_failures) * 100.0
            insights.append(
                f"{cat_name} is the dominant failure mode "
                f"({max_cnt} queries, {pct:.0f}% of failures)."
            )

    # 3. Weakest Retrieval Query Type
    active_types = [qm for qm in report.metrics_by_query_type.values() if qm.completed_queries > 0]
    if len(active_types) >= 2:
        sorted_by_recall = sorted(active_types, key=lambda qm: qm.mean_recall_at_k)
        lowest = sorted_by_recall[0]
        highest = sorted_by_recall[-1]
        k_val = report.retrieval.k

        gap = highest.mean_recall_at_k - lowest.mean_recall_at_k
        if gap >= 0.10:
            insights.append(
                f"'{lowest.query_type}' queries have the weakest retrieval performance "
                f"(Recall@{k_val}: {lowest.mean_recall_at_k:.2f} "
                f"vs {highest.mean_recall_at_k:.2f})."
            )
        elif lowest.mean_recall_at_k < 0.70:
            insights.append(
                f"'{lowest.query_type}' queries have low retrieval recall "
                f"(Recall@{k_val}: {lowest.mean_recall_at_k:.2f})."
            )

    # 4. Groundedness Gap Across Query Types
    grounded_types = [qm for qm in active_types if qm.groundedness_rate is not None]
    if len(grounded_types) >= 2:
        sorted_by_grounded = sorted(grounded_types, key=lambda qm: qm.groundedness_rate or 0.0)
        low_g = sorted_by_grounded[0]
        high_g = sorted_by_grounded[-1]
        low_rate = low_g.groundedness_rate or 0.0
        high_rate = high_g.groundedness_rate or 0.0

        if (high_rate - low_rate) >= 0.10:
            insights.append(
                f"Groundedness is substantially lower for '{low_g.query_type}' queries "
                f"({low_rate:.2f} vs {high_rate:.2f})."
            )

    # 5. Latency Concerns
    latency_outliers = report.diagnosis_counts.get("LATENCY_OUTLIER", 0)
    if latency_outliers > 0:
        insights.append(
            f"Latency concerns detected: {latency_outliers} queries exceeded latency thresholds "
            f"(P95: {report.latency.p95_ms:.1f} ms)."
        )
    elif report.latency.p95_ms > 1000.0 and report.latency.count > 0:
        insights.append(
            f"High tail latency observed: P95 retrieval latency is {report.latency.p95_ms:.1f} ms."
        )

    # 6. Judge Failures
    if report.judge_failures > 0:
        insights.append(
            f"Judge evaluation failed on {report.judge_failures} queries; "
            "semantic metrics exclude these from denominators."
        )

    # 7. Pipeline Failures
    if report.failed_queries > 0:
        insights.append(
            f"{report.failed_queries} queries suffered pipeline execution failures "
            "(categorized as UNKNOWN)."
        )

    return insights
