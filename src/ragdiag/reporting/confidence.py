"""Deterministic evaluation confidence calculation engine for RAGDiag."""

from typing import TYPE_CHECKING

from ragdiag.reporting.models import ConfidenceLevel, EvaluationConfidence

if TYPE_CHECKING:
    from ragdiag.reporting.models import EvaluationReport


def _calculate_sample_size_score(n: int) -> float:
    """Calculate a smooth, saturating [0.0, 1.0] score based on dataset query count.

    Saturating thresholds:
        - 0 queries: 0.0
        - 1-4 queries: 0.20 - 0.35 (very limited evidence)
        - 5-9 queries: 0.40 - 0.56 (limited evidence)
        - 10-24 queries: 0.60 - 0.79 (moderate evidence)
        - 25-49 queries: 0.80 - 0.99 (good evidence)
        - 50+ queries: 1.0 (strong evidence)
    """
    if n <= 0:
        return 0.0
    if n < 5:
        return 0.20 + 0.15 * ((n - 1) / 3.0 if n > 1 else 0.0)
    if n < 10:
        return 0.40 + 0.20 * ((n - 5) / 5.0)
    if n < 25:
        return 0.60 + 0.20 * ((n - 10) / 15.0)
    if n < 50:
        return 0.80 + 0.20 * ((n - 25) / 25.0)
    return 1.0


def _derive_confidence_level(score: float) -> str:
    """Derive deterministic categorical confidence level from score."""
    if score >= 90.0:
        return ConfidenceLevel.HIGH.value
    if score >= 75.0:
        return ConfidenceLevel.GOOD.value
    if score >= 60.0:
        return ConfidenceLevel.MODERATE.value
    if score >= 40.0:
        return ConfidenceLevel.LOW.value
    return ConfidenceLevel.VERY_LOW.value


def compute_confidence(report: "EvaluationReport") -> EvaluationConfidence:
    """Compute a deterministic EvaluationConfidence from an EvaluationReport.

    Assesses the completeness and dependability of the evaluation evidence
    based on query coverage, sample size, judge evidence, and execution failures.

    Args:
        report: Fully populated EvaluationReport.

    Returns:
        Validated `EvaluationConfidence` model.
    """
    if report.total_queries <= 0:
        return EvaluationConfidence(
            score=0.0,
            level=ConfidenceLevel.VERY_LOW.value,
            reasons=["Evaluation contains no queries."],
        )

    # 1. Query Coverage (up to 50 points)
    coverage_ratio = report.completed_queries / report.total_queries
    coverage_points = 50.0 * coverage_ratio

    # 2. Dataset Sample Size Curve (up to 35 points)
    sample_score = _calculate_sample_size_score(report.total_queries)
    sample_points = 35.0 * sample_score

    # 3. Judge Evidence (up to 15 points)
    judge_configured = report.judged_queries > 0 or report.judge_failures > 0
    if judge_configured:
        total_judge_attempts = report.judged_queries + report.judge_failures
        judge_success_ratio = (
            (report.judged_queries / total_judge_attempts) if total_judge_attempts > 0 else 0.0
        )
        judge_points = 15.0 * judge_success_ratio
    else:
        # Retrieval-only baseline: legitimate evidence, no severe penalty
        judge_points = 8.0

    # 4. Controlled Penalties
    fail_ratio = report.failed_queries / report.total_queries
    fail_penalty = fail_ratio * 15.0

    judge_penalty = 0.0
    if judge_configured and report.judge_failures > 0:
        total_judge_attempts = report.judged_queries + report.judge_failures
        judge_fail_ratio = (
            (report.judge_failures / total_judge_attempts) if total_judge_attempts > 0 else 0.0
        )
        judge_penalty = judge_fail_ratio * 15.0

    if report.completed_queries <= 0:
        score = 0.0
    else:
        raw_score = coverage_points + sample_points + judge_points - fail_penalty - judge_penalty
        score = round(max(0.0, min(100.0, raw_score)), 1)

    level = _derive_confidence_level(score)

    # 5. Concise Deterministic Reasons
    reasons: list[str] = []

    # Coverage reasons
    if report.failed_queries == 0 and report.completed_queries > 0:
        reasons.append("All evaluation queries completed successfully.")
    elif report.failed_queries > 0:
        pct = (report.failed_queries / report.total_queries) * 100.0
        q_str = "query" if report.failed_queries == 1 else "queries"
        reasons.append(
            f"{pct:.0f}% of evaluation queries failed ({report.failed_queries} {q_str})."
        )

    # Dataset size reasons
    if report.total_queries >= 50:
        reasons.append(f"Strong dataset sample size ({report.total_queries} queries).")
    elif report.total_queries >= 25:
        reasons.append(f"Good dataset sample size ({report.total_queries} queries).")
    elif report.total_queries >= 10:
        reasons.append(f"Moderate dataset sample size ({report.total_queries} queries).")
    else:
        reasons.append(
            f"Limited dataset sample size ({report.total_queries} queries; "
            "larger sample recommended)."
        )

    # Judge evidence reasons
    if judge_configured:
        if report.judge_failures == 0 and report.judged_queries > 0:
            reasons.append(
                "Semantic judge evaluation completed successfully across all evaluated queries."
            )
        elif report.judge_failures > 0:
            q_str = "evaluation" if report.judge_failures == 1 else "evaluations"
            reasons.append(f"{report.judge_failures} semantic judge {q_str} failed.")
    else:
        reasons.append("Semantic judge was not configured; retrieval-only evidence.")

    return EvaluationConfidence(
        score=score,
        level=level,
        reasons=reasons,
    )
