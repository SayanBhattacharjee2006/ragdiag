"""Regression analysis engine for multi-pipeline comparison."""

from ragdiag.comparison.models import (
    DiagnosisTransition,
    MetricDeltas,
    MetricRegression,
    QueryOutcomeComparison,
    RegressedQuery,
    RegressionAnalysis,
)
from ragdiag.diagnosis.models import FailureCategory
from ragdiag.reporting.models import EvaluationReport

SEVERITY_NAMES: dict[int, str] = {
    0: "info",
    1: "warning",
    2: "major",
}

# Diagnostic severity ranking: 0 is pass, 1 is warning, 2 is major failure
CATEGORY_SEVERITY_RANK: dict[str, int] = {
    FailureCategory.PASS.value: 0,
    FailureCategory.WRONG_CHUNK_RANK.value: 1,
    FailureCategory.INSUFFICIENT_CONTEXT.value: 1,
    FailureCategory.LATENCY_OUTLIER.value: 1,
    FailureCategory.WRONG_CHUNK_RETRIEVED.value: 2,
    FailureCategory.RETRIEVED_BUT_NOT_GROUNDED.value: 2,
    FailureCategory.ANSWER_INCORRECT.value: 2,
    FailureCategory.UNKNOWN.value: 2,
}

CATEGORY_QUALITY_RANK: dict[str, int] = {
    FailureCategory.PASS.value: 0,
    FailureCategory.LATENCY_OUTLIER.value: 1,
    FailureCategory.WRONG_CHUNK_RANK.value: 2,
    FailureCategory.INSUFFICIENT_CONTEXT.value: 3,
    FailureCategory.ANSWER_INCORRECT.value: 4,
    FailureCategory.RETRIEVED_BUT_NOT_GROUNDED.value: 5,
    FailureCategory.WRONG_CHUNK_RETRIEVED.value: 6,
    FailureCategory.UNKNOWN.value: 7,
}


def _determine_regression_reason(qo: QueryOutcomeComparison, k: int = 5) -> str:
    """Generate a concise human-readable explanation of why a query was classified as regressed."""
    sev_a = CATEGORY_SEVERITY_RANK.get(qo.diagnosis_a, 2)
    sev_b = CATEGORY_SEVERITY_RANK.get(qo.diagnosis_b, 2)
    if sev_b > sev_a:
        name_a = SEVERITY_NAMES.get(sev_a, "unknown")
        name_b = SEVERITY_NAMES.get(sev_b, "unknown")
        return f"Severity worsened from {name_a} ({qo.diagnosis_a}) to {name_b} ({qo.diagnosis_b})"

    if qo.recall_b < qo.recall_a - 0.001:
        return f"Recall@{k} dropped from {qo.recall_a:.2f} to {qo.recall_b:.2f}"

    if qo.grounded_a is True and qo.grounded_b is False:
        return "Answer lost context groundedness (hallucination)"

    if qo.answer_correct_a is True and qo.answer_correct_b is False:
        return "Answer became semantically incorrect"

    q_rank_a = CATEGORY_QUALITY_RANK.get(qo.diagnosis_a, 7)
    q_rank_b = CATEGORY_QUALITY_RANK.get(qo.diagnosis_b, 7)
    if q_rank_b > q_rank_a:
        return f"Diagnosis worsened from {qo.diagnosis_a} to {qo.diagnosis_b}"

    return f"Outcome regressed from {qo.diagnosis_a} to {qo.diagnosis_b}"


def _compute_query_importance_key(q: RegressedQuery) -> tuple[int, int, float, str]:
    """Sort key for ranking important regressions deterministically (lower is more critical)."""
    sev_a = CATEGORY_SEVERITY_RANK.get(q.baseline_diagnosis, 2)
    sev_b = CATEGORY_SEVERITY_RANK.get(q.current_diagnosis, 2)

    # Priority 0: Transition from PASS to Major failure
    if q.baseline_diagnosis == FailureCategory.PASS.value and sev_b == 2:
        tier = 0
    # Priority 1: Transition from PASS to Warning failure
    elif q.baseline_diagnosis == FailureCategory.PASS.value and sev_b == 1:
        tier = 1
    # Priority 2: Transition from Warning to Major failure
    elif sev_a == 1 and sev_b == 2:
        tier = 2
    # Priority 3: Correctness or groundedness lost
    elif (q.answer_correct_a is True and q.answer_correct_b is False) or (
        q.grounded_a is True and q.grounded_b is False
    ):
        tier = 3
    # Priority 4: Category quality rank worsened
    elif CATEGORY_QUALITY_RANK.get(q.current_diagnosis, 7) > CATEGORY_QUALITY_RANK.get(
        q.baseline_diagnosis, 7
    ):
        tier = 4
    # Priority 5: Recall dropped
    else:
        tier = 5

    quality_drop = CATEGORY_QUALITY_RANK.get(q.current_diagnosis, 7) - CATEGORY_QUALITY_RANK.get(
        q.baseline_diagnosis, 7
    )
    recall_drop = q.recall_a - q.recall_b
    return (tier, -quality_drop, -recall_drop, q.query_id)


def _is_meaningful_query_regression(
    qo: QueryOutcomeComparison, quality_tolerance: float = 0.02
) -> bool:
    """Check if a query regression represents a meaningful quality or diagnostic regression."""
    sev_a = CATEGORY_SEVERITY_RANK.get(qo.diagnosis_a, 2)
    sev_b = CATEGORY_SEVERITY_RANK.get(qo.diagnosis_b, 2)
    if sev_b > sev_a:
        return True

    if qo.grounded_a is True and qo.grounded_b is False:
        return True

    if qo.answer_correct_a is True and qo.answer_correct_b is False:
        return True

    q_rank_a = CATEGORY_QUALITY_RANK.get(qo.diagnosis_a, 7)
    q_rank_b = CATEGORY_QUALITY_RANK.get(qo.diagnosis_b, 7)
    if q_rank_b > q_rank_a:
        return True

    if (qo.recall_a - qo.recall_b) > quality_tolerance:
        return True

    return False


def analyze_regressions(
    report_a: EvaluationReport,
    report_b: EvaluationReport,
    metric_deltas: MetricDeltas,
    query_outcomes: list[QueryOutcomeComparison],
    quality_winner: str,
    latency_winner: str,
    overall_winner: str,
    queries_improved: int,
    queries_regressed: int,
    quality_tolerance: float = 0.02,
    latency_tolerance_ms: float = 10.0,
    k: int = 5,
    max_important: int = 5,
) -> RegressionAnalysis:
    """Analyze quality and latency regressions between baseline and candidate pipelines."""
    # 1. Overall Metric Regressions
    metric_regressions: list[MetricRegression] = []

    # Recall@K
    if metric_deltas.recall_at_k < -quality_tolerance:
        metric_regressions.append(
            MetricRegression(
                metric_name=f"Recall@{k}",
                baseline_value=report_a.retrieval.mean_recall_at_k,
                current_value=report_b.retrieval.mean_recall_at_k,
                delta=metric_deltas.recall_at_k,
                threshold=quality_tolerance,
            )
        )

    # Precision@K
    if metric_deltas.precision_at_k < -quality_tolerance:
        metric_regressions.append(
            MetricRegression(
                metric_name=f"Precision@{k}",
                baseline_value=report_a.retrieval.mean_precision_at_k,
                current_value=report_b.retrieval.mean_precision_at_k,
                delta=metric_deltas.precision_at_k,
                threshold=quality_tolerance,
            )
        )

    # MRR
    if metric_deltas.mrr < -quality_tolerance:
        metric_regressions.append(
            MetricRegression(
                metric_name="MRR",
                baseline_value=report_a.retrieval.mrr,
                current_value=report_b.retrieval.mrr,
                delta=metric_deltas.mrr,
                threshold=quality_tolerance,
            )
        )

    # Answer Correctness
    if (
        metric_deltas.answer_correctness is not None
        and metric_deltas.answer_correctness < -quality_tolerance
        and report_a.semantic is not None
        and report_a.semantic.answer_correctness_rate is not None
        and report_b.semantic is not None
        and report_b.semantic.answer_correctness_rate is not None
    ):
        metric_regressions.append(
            MetricRegression(
                metric_name="Answer Correctness",
                baseline_value=report_a.semantic.answer_correctness_rate,
                current_value=report_b.semantic.answer_correctness_rate,
                delta=metric_deltas.answer_correctness,
                threshold=quality_tolerance,
            )
        )

    # Groundedness
    if (
        metric_deltas.groundedness is not None
        and metric_deltas.groundedness < -quality_tolerance
        and report_a.semantic is not None
        and report_a.semantic.groundedness_rate is not None
        and report_b.semantic is not None
        and report_b.semantic.groundedness_rate is not None
    ):
        metric_regressions.append(
            MetricRegression(
                metric_name="Groundedness",
                baseline_value=report_a.semantic.groundedness_rate,
                current_value=report_b.semantic.groundedness_rate,
                delta=metric_deltas.groundedness,
                threshold=quality_tolerance,
            )
        )

    # Mean Retrieval Latency
    if metric_deltas.mean_retrieval_ms > latency_tolerance_ms:
        metric_regressions.append(
            MetricRegression(
                metric_name="Mean Retrieval Latency",
                baseline_value=report_a.latency.mean_ms,
                current_value=report_b.latency.mean_ms,
                delta=metric_deltas.mean_retrieval_ms,
                threshold=latency_tolerance_ms,
                unit="ms",
            )
        )

    # 2. Query-Level Regressions
    regressed_query_objects: list[RegressedQuery] = []
    for qo in query_outcomes:
        if qo.outcome == "regressed" and _is_meaningful_query_regression(
            qo, quality_tolerance=quality_tolerance
        ):
            reason = _determine_regression_reason(qo, k=k)
            regressed_query_objects.append(
                RegressedQuery(
                    query_id=qo.query_id,
                    baseline_diagnosis=qo.diagnosis_a,
                    current_diagnosis=qo.diagnosis_b,
                    transition=f"{qo.diagnosis_a} -> {qo.diagnosis_b}",
                    recall_a=qo.recall_a,
                    recall_b=qo.recall_b,
                    grounded_a=qo.grounded_a,
                    grounded_b=qo.grounded_b,
                    answer_correct_a=qo.answer_correct_a,
                    answer_correct_b=qo.answer_correct_b,
                    reason=reason,
                )
            )

    # 3. Diagnosis Regressions (transitions grouped by pair)
    transition_groups: dict[tuple[str, str], list[str]] = {}
    for rq in regressed_query_objects:
        pair = (rq.baseline_diagnosis, rq.current_diagnosis)
        transition_groups.setdefault(pair, []).append(rq.query_id)

    diagnosis_regressions: list[DiagnosisTransition] = []
    for (from_cat, to_cat), qids in transition_groups.items():
        diagnosis_regressions.append(
            DiagnosisTransition(
                from_category=from_cat,
                to_category=to_cat,
                transition=f"{from_cat} -> {to_cat}",
                count=len(qids),
                query_ids=sorted(qids),
            )
        )

    # Sort transitions: transitions from PASS first, then by count descending
    def transition_sort_key(dt: DiagnosisTransition) -> tuple[int, int, str]:
        is_from_pass = 0 if dt.from_category == FailureCategory.PASS.value else 1
        return (is_from_pass, -dt.count, dt.transition)

    diagnosis_regressions.sort(key=transition_sort_key)

    # Failure categories that increased in frequency (B - A > 0, excluding PASS)
    increased_failures: dict[str, int] = {}
    for cat in FailureCategory:
        if cat == FailureCategory.PASS:
            continue
        cnt_a = report_a.diagnosis_counts.get(cat.value, 0)
        cnt_b = report_b.diagnosis_counts.get(cat.value, 0)
        if cnt_b > cnt_a:
            increased_failures[cat.value] = cnt_b - cnt_a

    # 4. Deterministic Important Regressions Selection
    sorted_regressed = sorted(regressed_query_objects, key=_compute_query_importance_key)
    important_regressions: list[str] = []
    for q in sorted_regressed[:max_important]:
        if q.baseline_diagnosis != q.current_diagnosis:
            important_regressions.append(f"{q.query_id}: {q.transition}")
        else:
            important_regressions.append(f"{q.query_id}: {q.baseline_diagnosis} ({q.reason})")

    # 5. Deterministic Overall Decision and Narrative Summary
    name_a = report_a.pipeline_name or "Pipeline A"
    name_b = report_b.pipeline_name or "Pipeline B"
    reg_count = len(regressed_query_objects)

    overall_regression = False
    summary = ""

    if quality_winner == name_a:
        overall_regression = True
        metrics_str = ", ".join(
            f"{mr.metric_name} {mr.delta:+.2f}" for mr in metric_regressions if not mr.unit
        )
        if not metrics_str:
            metrics_str = "retrieval/generation quality"
        q_str = "query" if reg_count == 1 else "queries"
        summary = (
            f"Overall quality regression detected: {name_b} regressed on {metrics_str} "
            f"compared to {name_a} ({reg_count} {q_str} regressed)."
        )
    elif quality_winner == "TIE":
        if reg_count > queries_improved:
            overall_regression = True
            summary = (
                f"Query-level regression detected: {reg_count} queries regressed while "
                f"{queries_improved} improved despite comparable aggregate metrics."
            )
        elif latency_winner == name_a:
            overall_regression = True
            lat_delta = metric_deltas.mean_retrieval_ms
            summary = (
                f"Performance regression detected: Mean retrieval latency increased by "
                f"{lat_delta:.1f} ms with comparable retrieval quality."
            )
        else:
            overall_regression = False
            if reg_count > 0:
                summary = (
                    f"No overall regression: {reg_count} queries regressed but balanced "
                    f"by improvements within comparison tolerance."
                )
            else:
                summary = "No meaningful regressions detected."
    else:
        # quality_winner == name_b (Pipeline B improved primary quality)
        overall_regression = False
        lat_reg = any(mr.metric_name == "Mean Retrieval Latency" for mr in metric_regressions)
        sec_reg = [mr for mr in metric_regressions if mr.metric_name != "Mean Retrieval Latency"]
        if lat_reg:
            summary = (
                f"No overall regression: {name_b} improved overall quality despite "
                f"increased retrieval latency (+{metric_deltas.mean_retrieval_ms:.1f} ms)."
            )
        elif sec_reg:
            reg_names = ", ".join(mr.metric_name for mr in sec_reg)
            summary = (
                f"No overall regression: {name_b} improved primary quality despite "
                f"secondary drops in {reg_names}."
            )
        else:
            summary = "No meaningful regressions detected; current pipeline improved overall."

    return RegressionAnalysis(
        overall_regression=overall_regression,
        metric_regressions=metric_regressions,
        diagnosis_regressions=diagnosis_regressions,
        increased_failures=increased_failures,
        regressed_queries=regressed_query_objects,
        regressed_query_count=reg_count,
        important_regressions=important_regressions,
        summary=summary,
    )
