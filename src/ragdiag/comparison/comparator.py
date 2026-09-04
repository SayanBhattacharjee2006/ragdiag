"""Comparison engine for evaluating and contrasting two RAG pipelines."""

from typing import Literal

from ragdiag.comparison.models import (
    ComparisonReport,
    MetricDeltas,
    QueryOutcomeComparison,
    QueryTypeDeltas,
)
from ragdiag.diagnosis.models import FailureCategory
from ragdiag.judges.base import Judge
from ragdiag.models.dataset import GoldenDataset
from ragdiag.models.result import EvaluationResult
from ragdiag.pipeline.base import Pipeline
from ragdiag.reporting.aggregator import build_report
from ragdiag.reporting.models import EvaluationReport
from ragdiag.runner.evaluator import Evaluator

TAXONOMY_ORDER: list[str] = [cat.value for cat in FailureCategory]
STANDARD_QUERY_TYPES: list[str] = ["factual", "reasoning", "multi-hop"]

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

# Deterministic category quality ranking (lower is healthier/closer to full PASS):
# Reflects pipeline stage health:
# 0: PASS (successful)
# 1: LATENCY_OUTLIER (all retrieval & semantic checks passed, only slow)
# 2: WRONG_CHUNK_RANK (all required context retrieved in top-K, suboptimal rank)
# 3: INSUFFICIENT_CONTEXT (only partial required context retrieved)
# 4: ANSWER_INCORRECT (all context retrieved & grounded, incorrect answer)
# 5: RETRIEVED_BUT_NOT_GROUNDED (context retrieved, but hallucinated)
# 6: WRONG_CHUNK_RETRIEVED (total retrieval miss, 0 relevant chunks)
# 7: UNKNOWN (execution crash / unhandled failure)
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


def _get_category_str(result: EvaluationResult) -> str:
    """Extract string failure category from an EvaluationResult."""
    if hasattr(result.diagnosis, "category"):
        cat_val = result.diagnosis.category
        return cat_val.value if hasattr(cat_val, "value") else str(cat_val)
    if isinstance(result.diagnosis, dict) and "category" in result.diagnosis:
        return str(result.diagnosis["category"])
    return FailureCategory.UNKNOWN.value


def _get_recall(result: EvaluationResult, k: int = 5) -> float:
    """Extract recall metric from an EvaluationResult."""
    k_key = f"recall_at_{k}"
    if k_key in result.metrics and isinstance(result.metrics[k_key], (int, float)):
        return float(result.metrics[k_key])
    return 0.0


def _classify_outcome(
    res_a: EvaluationResult,
    res_b: EvaluationResult,
    k: int = 5,
) -> Literal["improved", "regressed", "unchanged"]:
    """Classify the per-query outcome transition from Pipeline A to Pipeline B.

    Rules:
        1. Compare severity rank:
           - Lower severity for B -> 'improved' (e.g. PASS vs failure, warning vs major).
           - Higher severity for B -> 'regressed' (e.g. failure vs PASS, major vs warning).
        2. If severity is equal:
           - Compare measured recall (difference > 0.001 wins).
           - Compare measured groundedness (if both available).
           - Compare measured answer correctness (if both available).
           - Compare diagnosis category:
             - If categories are identical -> 'unchanged'.
             - If categories differ -> compare deterministic category quality rank:
               - Higher quality category for B -> 'improved'.
               - Lower quality category for B -> 'regressed'.
        3. Return 'unchanged' only when category, recall, and semantic outcomes
           are genuinely equivalent.
    """
    cat_a = _get_category_str(res_a)
    cat_b = _get_category_str(res_b)
    sev_a = CATEGORY_SEVERITY_RANK.get(cat_a, 2)
    sev_b = CATEGORY_SEVERITY_RANK.get(cat_b, 2)

    if sev_b < sev_a:
        return "improved"
    if sev_b > sev_a:
        return "regressed"

    # Same severity rank: check measured retrieval recall
    rec_a = _get_recall(res_a, k=k)
    rec_b = _get_recall(res_b, k=k)
    if rec_b > rec_a + 0.001:
        return "improved"
    if rec_b < rec_a - 0.001:
        return "regressed"

    # Check semantic groundedness and answer correctness
    g_a = res_a.metrics.get("grounded") if isinstance(res_a.metrics.get("grounded"), bool) else None
    g_b = res_b.metrics.get("grounded") if isinstance(res_b.metrics.get("grounded"), bool) else None
    if g_a is False and g_b is True:
        return "improved"
    if g_a is True and g_b is False:
        return "regressed"

    c_a = (
        res_a.metrics.get("answer_correct")
        if isinstance(res_a.metrics.get("answer_correct"), bool)
        else None
    )
    c_b = (
        res_b.metrics.get("answer_correct")
        if isinstance(res_b.metrics.get("answer_correct"), bool)
        else None
    )
    if c_a is False and c_b is True:
        return "improved"
    if c_a is True and c_b is False:
        return "regressed"

    # If identical category and all measured signals are equivalent, outcome is unchanged
    if cat_a == cat_b:
        return "unchanged"

    # Same severity but different category: use deterministic category quality rank
    q_rank_a = CATEGORY_QUALITY_RANK.get(cat_a, 7)
    q_rank_b = CATEGORY_QUALITY_RANK.get(cat_b, 7)
    if q_rank_b < q_rank_a:
        return "improved"
    if q_rank_b > q_rank_a:
        return "regressed"

    return "unchanged"


def compare_reports(
    report_a: EvaluationReport,
    report_b: EvaluationReport,
    results_a: list[EvaluationResult],
    results_b: list[EvaluationResult],
    quality_tolerance: float = 0.02,
    latency_tolerance_ms: float = 10.0,
    k: int = 5,
) -> ComparisonReport:
    """Compare two pre-computed EvaluationReports and their underlying results.

    Deltas are computed as Pipeline B minus Pipeline A.

    Args:
        report_a: EvaluationReport for Pipeline A.
        report_b: EvaluationReport for Pipeline B.
        results_a: List of EvaluationResult items for Pipeline A.
        results_b: List of EvaluationResult items for Pipeline B.
        quality_tolerance: Margin within which quality metrics are considered tied.
        latency_tolerance_ms: Margin in ms within which latencies are considered tied.
        k: Retrieval rank parameter K.

    Returns:
        Structured `ComparisonReport`.
    """
    name_a = report_a.pipeline_name or "Pipeline A"
    name_b = report_b.pipeline_name or "Pipeline B"

    # 1. Overall Metric Deltas (B - A)
    prec_delta = report_b.retrieval.mean_precision_at_k - report_a.retrieval.mean_precision_at_k
    rec_delta = report_b.retrieval.mean_recall_at_k - report_a.retrieval.mean_recall_at_k
    mrr_delta = report_b.retrieval.mrr - report_a.retrieval.mrr

    correctness_delta: float | None = None
    if (
        report_a.semantic is not None
        and report_a.semantic.answer_correctness_rate is not None
        and report_b.semantic is not None
        and report_b.semantic.answer_correctness_rate is not None
    ):
        correctness_delta = (
            report_b.semantic.answer_correctness_rate - report_a.semantic.answer_correctness_rate
        )

    groundedness_delta: float | None = None
    if (
        report_a.semantic is not None
        and report_a.semantic.groundedness_rate is not None
        and report_b.semantic is not None
        and report_b.semantic.groundedness_rate is not None
    ):
        groundedness_delta = (
            report_b.semantic.groundedness_rate - report_a.semantic.groundedness_rate
        )

    mean_lat_delta = report_b.latency.mean_ms - report_a.latency.mean_ms
    p95_lat_delta = report_b.latency.p95_ms - report_a.latency.p95_ms

    metric_deltas = MetricDeltas(
        precision_at_k=round(prec_delta, 4),
        recall_at_k=round(rec_delta, 4),
        mrr=round(mrr_delta, 4),
        answer_correctness=round(correctness_delta, 4) if correctness_delta is not None else None,
        groundedness=round(groundedness_delta, 4) if groundedness_delta is not None else None,
        mean_retrieval_ms=round(mean_lat_delta, 2),
        p95_retrieval_ms=round(p95_lat_delta, 2),
    )

    # 2. Diagnosis Deltas (B - A for all 8 categories)
    diagnosis_deltas: dict[str, int] = {}
    for cat in TAXONOMY_ORDER:
        cnt_a = report_a.diagnosis_counts.get(cat, 0)
        cnt_b = report_b.diagnosis_counts.get(cat, 0)
        diagnosis_deltas[cat] = cnt_b - cnt_a

    # 3. Query-Type Deltas
    all_query_types = list(
        dict.fromkeys(
            [
                qt
                for qt in STANDARD_QUERY_TYPES
                if qt in report_a.metrics_by_query_type or qt in report_b.metrics_by_query_type
            ]
            + list(report_a.metrics_by_query_type.keys())
            + list(report_b.metrics_by_query_type.keys())
        )
    )

    query_type_deltas: dict[str, QueryTypeDeltas] = {}
    for qt in all_query_types:
        qm_a = report_a.metrics_by_query_type.get(qt)
        qm_b = report_b.metrics_by_query_type.get(qt)

        rec_a = qm_a.mean_recall_at_k if qm_a else 0.0
        rec_b = qm_b.mean_recall_at_k if qm_b else 0.0
        mrr_a = qm_a.mrr if qm_a else 0.0
        mrr_b = qm_b.mrr if qm_b else 0.0

        c_delta: float | None = None
        if (
            qm_a
            and qm_b
            and qm_a.answer_correctness_rate is not None
            and qm_b.answer_correctness_rate is not None
        ):
            c_delta = round(qm_b.answer_correctness_rate - qm_a.answer_correctness_rate, 4)

        g_delta: float | None = None
        if (
            qm_a
            and qm_b
            and qm_a.groundedness_rate is not None
            and qm_b.groundedness_rate is not None
        ):
            g_delta = round(qm_b.groundedness_rate - qm_a.groundedness_rate, 4)

        fail_deltas: dict[str, int] = {}
        tot_fail_delta = 0
        for cat in TAXONOMY_ORDER:
            c_cnt_a = qm_a.diagnosis_counts.get(cat, 0) if qm_a else 0
            c_cnt_b = qm_b.diagnosis_counts.get(cat, 0) if qm_b else 0
            diff = c_cnt_b - c_cnt_a
            fail_deltas[cat] = diff
            if cat != "PASS":
                tot_fail_delta += diff

        query_type_deltas[qt] = QueryTypeDeltas(
            query_type=qt,
            recall_at_k=round(rec_b - rec_a, 4),
            mrr=round(mrr_b - mrr_a, 4),
            answer_correctness=c_delta,
            groundedness=g_delta,
            total_failure_delta=tot_fail_delta,
            failure_deltas=fail_deltas,
        )

    # 4. Per-Query Outcome Comparison
    map_a = {r.query_id: r for r in results_a}
    map_b = {r.query_id: r for r in results_b}
    matched_ids = [r.query_id for r in results_a if r.query_id in map_b]
    # In case B has unique queries not in A
    for r in results_b:
        if r.query_id not in map_a and r.query_id not in matched_ids:
            matched_ids.append(r.query_id)

    query_outcomes: list[QueryOutcomeComparison] = []
    improved_cnt = 0
    regressed_cnt = 0
    unchanged_cnt = 0

    for qid in matched_ids:
        r_a = map_a.get(qid)
        r_b = map_b.get(qid)
        if r_a is None or r_b is None:
            continue

        outcome = _classify_outcome(r_a, r_b, k=k)
        if outcome == "improved":
            improved_cnt += 1
        elif outcome == "regressed":
            regressed_cnt += 1
        else:
            unchanged_cnt += 1

        g_a = r_a.metrics.get("grounded") if isinstance(r_a.metrics.get("grounded"), bool) else None
        g_b = r_b.metrics.get("grounded") if isinstance(r_b.metrics.get("grounded"), bool) else None
        c_a = (
            r_a.metrics.get("answer_correct")
            if isinstance(r_a.metrics.get("answer_correct"), bool)
            else None
        )
        c_b = (
            r_b.metrics.get("answer_correct")
            if isinstance(r_b.metrics.get("answer_correct"), bool)
            else None
        )

        query_outcomes.append(
            QueryOutcomeComparison(
                query_id=qid,
                diagnosis_a=_get_category_str(r_a),
                diagnosis_b=_get_category_str(r_b),
                recall_a=round(_get_recall(r_a, k=k), 4),
                recall_b=round(_get_recall(r_b, k=k), 4),
                grounded_a=g_a,
                grounded_b=g_b,
                answer_correct_a=c_a,
                answer_correct_b=c_b,
                outcome=outcome,
            )
        )

    # 5. Winner Determination
    # Quality Winner (primary signals: Recall -> MRR -> Groundedness -> Correctness)
    if rec_delta > quality_tolerance:
        quality_winner = name_b
    elif rec_delta < -quality_tolerance:
        quality_winner = name_a
    elif mrr_delta > quality_tolerance:
        quality_winner = name_b
    elif mrr_delta < -quality_tolerance:
        quality_winner = name_a
    elif groundedness_delta is not None and groundedness_delta > quality_tolerance:
        quality_winner = name_b
    elif groundedness_delta is not None and groundedness_delta < -quality_tolerance:
        quality_winner = name_a
    elif correctness_delta is not None and correctness_delta > quality_tolerance:
        quality_winner = name_b
    elif correctness_delta is not None and correctness_delta < -quality_tolerance:
        quality_winner = name_a
    else:
        quality_winner = "TIE"

    # Latency Winner (negative delta means B is faster)
    if mean_lat_delta < -latency_tolerance_ms:
        latency_winner = name_b
    elif mean_lat_delta > latency_tolerance_ms:
        latency_winner = name_a
    else:
        latency_winner = "TIE"

    # Overall Winner and Trade-Off Detection
    trade_off: str | None = None
    if quality_winner == name_b and latency_winner == name_a:
        overall_winner = name_b
        trade_off = "Higher quality <-> higher latency"
    elif quality_winner == name_a and latency_winner == name_b:
        overall_winner = name_a
        trade_off = "Higher quality <-> higher latency"
    elif quality_winner == name_b and latency_winner == name_b:
        overall_winner = name_b
        trade_off = "Higher quality and lower latency"
    elif quality_winner == name_a and latency_winner == name_a:
        overall_winner = name_a
        trade_off = "Higher quality and lower latency"
    elif quality_winner == name_b and latency_winner == "TIE":
        overall_winner = name_b
        trade_off = "Higher quality with comparable latency"
    elif quality_winner == name_a and latency_winner == "TIE":
        overall_winner = name_a
        trade_off = "Higher quality with comparable latency"
    elif quality_winner == "TIE":
        if latency_winner != "TIE":
            overall_winner = latency_winner
            trade_off = f"Equivalent quality; {latency_winner} is faster"
        else:
            overall_winner = "TIE"
            trade_off = "Roughly equal performance within comparison tolerance"
    else:
        overall_winner = "TIE"

    # 6. Summary Narrative Generation
    summary_parts: list[str] = []
    points_rec = abs(rec_delta) * 100.0
    points_mrr = abs(mrr_delta) * 100.0

    if quality_winner == name_b and latency_winner == name_a:
        summary_parts.append(
            f"{name_b} improves Recall@{k} by {points_rec:.0f} percentage points and MRR by "
            f"{points_mrr:.0f} points, while increasing mean retrieval latency by "
            f"{mean_lat_delta:.1f} ms."
        )
    elif quality_winner == name_b and latency_winner == name_b:
        summary_parts.append(
            f"{name_b} outperforms {name_a} on both retrieval quality "
            f"(Recall@{k} +{points_rec:.0f} points) and retrieval latency "
            f"({abs(mean_lat_delta):.1f} ms faster)."
        )
    elif quality_winner == name_b and latency_winner == "TIE":
        summary_parts.append(
            f"{name_b} outperforms {name_a} on retrieval quality (Recall@{k} +{points_rec:.0f} "
            f"points), while retrieval latency is comparable (delta: {mean_lat_delta:+.1f} ms)."
        )
    elif quality_winner == name_a and latency_winner == name_b:
        summary_parts.append(
            f"{name_b} is faster ({abs(mean_lat_delta):.1f} ms lower mean latency), "
            f"but {name_a} provides substantially higher retrieval quality "
            f"(Recall@{k} +{points_rec:.0f} points)."
        )
    elif quality_winner == name_a and latency_winner == name_a:
        summary_parts.append(
            f"{name_a} outperforms {name_b} on both retrieval quality "
            f"(Recall@{k} +{points_rec:.0f} points) and retrieval latency "
            f"({abs(mean_lat_delta):.1f} ms faster)."
        )
    elif quality_winner == name_a and latency_winner == "TIE":
        summary_parts.append(
            f"{name_a} outperforms {name_b} on retrieval quality (Recall@{k} +{points_rec:.0f} "
            f"points), while retrieval latency is comparable (delta: {mean_lat_delta:+.1f} ms)."
        )
    elif quality_winner == "TIE" and latency_winner != "TIE":
        summary_parts.append(
            f"The pipelines perform similarly on retrieval quality within tolerance, "
            f"with {latency_winner} achieving lower latency "
            f"({abs(mean_lat_delta):.1f} ms difference)."
        )
    else:
        summary_parts.append(
            f"The pipelines perform similarly within the configured comparison tolerance "
            f"(Recall delta: {rec_delta:+.2f}, latency delta: {mean_lat_delta:+.1f} ms)."
        )

    summary = " ".join(summary_parts)

    return ComparisonReport(
        dataset_name=report_a.dataset_name or report_b.dataset_name,
        dataset_version=report_a.dataset_version or report_b.dataset_version,
        pipeline_a_name=name_a,
        pipeline_b_name=name_b,
        pipeline_a_report=report_a,
        pipeline_b_report=report_b,
        metric_deltas=metric_deltas,
        diagnosis_deltas=diagnosis_deltas,
        query_type_deltas=query_type_deltas,
        query_outcomes=query_outcomes,
        queries_improved=improved_cnt,
        queries_regressed=regressed_cnt,
        queries_unchanged=unchanged_cnt,
        quality_winner=quality_winner,
        latency_winner=latency_winner,
        overall_winner=overall_winner,
        winner=overall_winner,
        trade_off=trade_off,
        summary=summary,
    )


class Comparator:
    """Evaluates and compares two RAG pipelines against a shared GoldenDataset."""

    def __init__(
        self,
        k: int = 5,
        judge: Judge | None = None,
        quality_tolerance: float = 0.02,
        latency_tolerance_ms: float = 10.0,
    ) -> None:
        """Initialize the Comparator.

        Args:
            k: Retrieval rank cutoff parameter K.
            judge: Optional LLM Judge instance applied identically to both pipelines.
            quality_tolerance: Threshold delta below which quality metrics are considered tied.
            latency_tolerance_ms: Threshold delta in ms below which latencies are considered tied.
        """
        self.k = k
        self.judge = judge
        self.quality_tolerance = quality_tolerance
        self.latency_tolerance_ms = latency_tolerance_ms

    def compare(
        self,
        pipeline_a: Pipeline,
        pipeline_b: Pipeline,
        dataset: GoldenDataset,
    ) -> ComparisonReport:
        """Run evaluation on both pipelines and produce a comprehensive ComparisonReport.

        Args:
            pipeline_a: Baseline pipeline adapter.
            pipeline_b: Candidate pipeline adapter.
            dataset: GoldenDataset evaluated by both pipelines.

        Returns:
            Fully populated `ComparisonReport`.
        """
        evaluator = Evaluator(k=self.k, judge=self.judge)

        results_a = evaluator.evaluate(pipeline_a, dataset)
        results_b = evaluator.evaluate(pipeline_b, dataset)

        report_a = build_report(
            results_a,
            dataset_name=dataset.name,
            dataset_version=dataset.version,
            pipeline_name=pipeline_a.name,
            k=self.k,
        )
        report_b = build_report(
            results_b,
            dataset_name=dataset.name,
            dataset_version=dataset.version,
            pipeline_name=pipeline_b.name,
            k=self.k,
        )

        return compare_reports(
            report_a=report_a,
            report_b=report_b,
            results_a=results_a,
            results_b=results_b,
            quality_tolerance=self.quality_tolerance,
            latency_tolerance_ms=self.latency_tolerance_ms,
            k=self.k,
        )
