"""Deterministic health profile calculation engine for RAGDiag evaluation reports."""

from typing import TYPE_CHECKING

from ragdiag.diagnosis.models import FailureCategory, get_action_for_category
from ragdiag.reporting.models import HealthGrade, HealthProfile, HealthStatus

if TYPE_CHECKING:
    from ragdiag.reporting.models import EvaluationReport

# Priority ordering for evaluating diagnosis failure recommendations
# Major failures take precedence over warnings
DIAGNOSIS_ACTION_PRIORITY: list[FailureCategory] = [
    FailureCategory.WRONG_CHUNK_RETRIEVED,
    FailureCategory.RETRIEVED_BUT_NOT_GROUNDED,
    FailureCategory.ANSWER_INCORRECT,
    FailureCategory.UNKNOWN,
    FailureCategory.INSUFFICIENT_CONTEXT,
    FailureCategory.WRONG_CHUNK_RANK,
    FailureCategory.LATENCY_OUTLIER,
]


def _calculate_latency_score(mean_ms: float) -> float:
    """Calculate a normalized [0.0, 1.0] performance score from mean retrieval latency.

    Piecewise linear interpolation:
        - <= 100ms: 1.0 (optimal)
        - 100ms - 500ms: drops from 1.0 to 0.75 (acceptable)
        - 500ms - 1500ms: drops from 0.75 to 0.25 (degraded)
        - > 1500ms: drops to 0.0 at 3000ms
    """
    if mean_ms <= 100.0:
        return 1.0
    if mean_ms <= 500.0:
        return 1.0 - 0.25 * ((mean_ms - 100.0) / 400.0)
    if mean_ms <= 1500.0:
        return 0.75 - 0.50 * ((mean_ms - 500.0) / 1000.0)
    return max(0.0, 0.25 - 0.25 * ((mean_ms - 1500.0) / 1500.0))


def _derive_grade_and_status(score: float) -> tuple[str, str]:
    """Derive deterministic categorical grade and operational status from health score."""
    if score >= 90.0:
        return HealthGrade.EXCELLENT.value, HealthStatus.HEALTHY.value
    if score >= 75.0:
        return HealthGrade.GOOD.value, HealthStatus.HEALTHY.value
    if score >= 60.0:
        return HealthGrade.FAIR.value, HealthStatus.DEGRADED.value
    if score >= 40.0:
        return HealthGrade.POOR.value, HealthStatus.UNHEALTHY.value
    return HealthGrade.CRITICAL.value, HealthStatus.CRITICAL.value


def compute_health_profile(report: "EvaluationReport") -> HealthProfile:
    """Compute a deterministic, evidence-based HealthProfile from an EvaluationReport.

    Formula:
        Combines retrieval quality (Recall@K, Precision@K, MRR), semantic quality
        (Answer Correctness, Groundedness) when judge results exist, and retrieval
        latency into a bounded 0-100 score, applying controlled penalties for
        infrastructure execution crashes and latency outliers.

    Args:
        report: Aggregated EvaluationReport containing metrics, latencies, and diagnoses.

    Returns:
        Fully populated, validated `HealthProfile`.
    """
    k = report.retrieval.k
    recall = max(0.0, min(1.0, report.retrieval.mean_recall_at_k))
    precision = max(0.0, min(1.0, report.retrieval.mean_precision_at_k))
    mrr = max(0.0, min(1.0, report.retrieval.mrr))
    latency_score = _calculate_latency_score(report.latency.mean_ms)

    has_semantic = (
        report.semantic is not None
        and report.semantic.answer_correctness_rate is not None
        and report.semantic.groundedness_rate is not None
    )

    if has_semantic:
        correctness = max(0.0, min(1.0, report.semantic.answer_correctness_rate or 0.0))
        groundedness = max(0.0, min(1.0, report.semantic.groundedness_rate or 0.0))
        # With Judge: 25% recall, 10% precision, 15% MRR,
        # 25% correctness, 15% groundedness, 10% latency
        base_score = 100.0 * (
            0.25 * recall
            + 0.10 * precision
            + 0.15 * mrr
            + 0.25 * correctness
            + 0.15 * groundedness
            + 0.10 * latency_score
        )
    else:
        # Retrieval Only: 50% recall, 20% precision, 20% MRR, 10% latency
        base_score = 100.0 * (0.50 * recall + 0.20 * precision + 0.20 * mrr + 0.10 * latency_score)

    # Controlled penalties (without double-penalizing standard retrieval misses)
    crash_penalty = 0.0
    outlier_penalty = 0.0
    if report.total_queries > 0:
        crash_ratio = report.failed_queries / report.total_queries
        crash_penalty = crash_ratio * 25.0

        latency_outliers = report.diagnosis_counts.get(FailureCategory.LATENCY_OUTLIER.value, 0)
        outlier_penalty = (latency_outliers / report.total_queries) * 5.0

    score = round(max(0.0, min(100.0, base_score - crash_penalty - outlier_penalty)), 1)
    grade, status = _derive_grade_and_status(score)

    # 1. Strengths Extraction (Strictly supported by measured data)
    strengths: list[str] = []
    if recall >= 0.85:
        strengths.append(f"High recall (mean Recall@{k}: {recall:.2f})")
    if precision >= 0.75:
        strengths.append(f"High precision (mean Precision@{k}: {precision:.2f})")
    if mrr >= 0.80:
        strengths.append(f"Strong ranking quality (MRR: {mrr:.2f})")

    if has_semantic and report.semantic is not None:
        if (
            report.semantic.answer_correctness_rate is not None
            and report.semantic.answer_correctness_rate >= 0.85
        ):
            strengths.append(
                f"High answer correctness ({report.semantic.answer_correctness_rate:.0%})"
            )
        if (
            report.semantic.groundedness_rate is not None
            and report.semantic.groundedness_rate >= 0.85
        ):
            strengths.append(
                f"Strong context groundedness ({report.semantic.groundedness_rate:.0%})"
            )

    if report.completed_queries > 0 and report.latency.mean_ms <= 100.0:
        strengths.append(f"Low retrieval latency (mean: {report.latency.mean_ms:.1f}ms)")

    total_diag_failures = sum(
        cnt for cat, cnt in report.diagnosis_counts.items() if cat != FailureCategory.PASS.value
    )
    if report.completed_queries > 0 and report.failed_queries == 0 and total_diag_failures == 0:
        strengths.append("Zero diagnostic failures across all evaluated queries")

    # 2. Weaknesses Extraction
    weaknesses: list[str] = []
    if recall < 0.70:
        weaknesses.append(f"Low recall (mean Recall@{k}: {recall:.2f})")
    if precision < 0.50:
        weaknesses.append(f"Poor precision (mean Precision@{k}: {precision:.2f})")
    if mrr < 0.60:
        weaknesses.append(f"Suboptimal ranking quality (MRR: {mrr:.2f})")

    if has_semantic and report.semantic is not None:
        if (
            report.semantic.answer_correctness_rate is not None
            and report.semantic.answer_correctness_rate < 0.70
        ):
            weaknesses.append(
                f"Low answer correctness ({report.semantic.answer_correctness_rate:.0%})"
            )
        if (
            report.semantic.groundedness_rate is not None
            and report.semantic.groundedness_rate < 0.70
        ):
            weaknesses.append(
                f"Frequent ungrounded answers ({report.semantic.groundedness_rate:.0%})"
            )

    latency_outliers = report.diagnosis_counts.get(FailureCategory.LATENCY_OUTLIER.value, 0)
    if report.latency.mean_ms > 300.0 or report.latency.p95_ms > 1000.0 or latency_outliers > 0:
        weaknesses.append(
            f"High retrieval latency (mean: {report.latency.mean_ms:.1f}ms, "
            f"P95: {report.latency.p95_ms:.1f}ms)"
        )

    # Diagnosis-specific weaknesses
    for cat in [
        FailureCategory.WRONG_CHUNK_RETRIEVED,
        FailureCategory.INSUFFICIENT_CONTEXT,
        FailureCategory.WRONG_CHUNK_RANK,
        FailureCategory.RETRIEVED_BUT_NOT_GROUNDED,
        FailureCategory.ANSWER_INCORRECT,
    ]:
        cnt = report.diagnosis_counts.get(cat.value, 0)
        if cnt > 0:
            q_word = "query" if cnt == 1 else "queries"
            if cat == FailureCategory.WRONG_CHUNK_RETRIEVED:
                weaknesses.append(
                    f"Complete retrieval misses on {cnt} {q_word} (WRONG_CHUNK_RETRIEVED)"
                )
            elif cat == FailureCategory.INSUFFICIENT_CONTEXT:
                weaknesses.append(
                    f"Partial context retrieved on {cnt} {q_word} (INSUFFICIENT_CONTEXT)"
                )
            elif cat == FailureCategory.WRONG_CHUNK_RANK:
                weaknesses.append(f"Suboptimal chunk ranking on {cnt} {q_word} (WRONG_CHUNK_RANK)")
            elif cat == FailureCategory.RETRIEVED_BUT_NOT_GROUNDED:
                weaknesses.append(
                    f"Ungrounded claims generated on {cnt} {q_word} (RETRIEVED_BUT_NOT_GROUNDED)"
                )
            elif cat == FailureCategory.ANSWER_INCORRECT:
                weaknesses.append(
                    f"Incorrect answers generated on {cnt} {q_word} (ANSWER_INCORRECT)"
                )

    unknown_cnt = report.diagnosis_counts.get(FailureCategory.UNKNOWN.value, 0)
    if report.failed_queries > 0 or unknown_cnt > 0:
        f_count = max(report.failed_queries, unknown_cnt)
        q_word = "query" if f_count == 1 else "queries"
        weaknesses.append(f"Pipeline execution failures ({f_count} {q_word})")

    # 3. Recommendations Generation (Reusing Failure -> Action Mapping)
    recommendations: list[str] = []
    for cat in DIAGNOSIS_ACTION_PRIORITY:
        cnt = report.diagnosis_counts.get(cat.value, 0)
        if cnt > 0:
            act = get_action_for_category(cat)
            if act not in recommendations:
                recommendations.append(act)

    # Fallback recommendations if metrics are low without specific diagnosis triggers
    if recall < 0.70 and not any(
        report.diagnosis_counts.get(c.value, 0) > 0
        for c in (FailureCategory.WRONG_CHUNK_RETRIEVED, FailureCategory.INSUFFICIENT_CONTEXT)
    ):
        act = get_action_for_category(FailureCategory.INSUFFICIENT_CONTEXT)
        if act not in recommendations:
            recommendations.append(act)

    if (
        report.latency.mean_ms > 300.0 or report.latency.p95_ms > 1000.0
    ) and get_action_for_category(FailureCategory.LATENCY_OUTLIER) not in recommendations:
        recommendations.append(get_action_for_category(FailureCategory.LATENCY_OUTLIER))

    # Clean pass fallback
    if not recommendations and score >= 90.0:
        recommendations.append(get_action_for_category(FailureCategory.PASS))

    return HealthProfile(
        score=score,
        grade=grade,
        status=status,
        strengths=strengths,
        weaknesses=weaknesses,
        recommendations=recommendations,
    )
