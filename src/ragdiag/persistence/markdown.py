"""Markdown report generators for human-readable persistence."""

from ragdiag.comparison.models import ComparisonReport
from ragdiag.diagnosis.inspector import inspect_report
from ragdiag.diagnosis.models import get_action_for_category
from ragdiag.reporting.models import EvaluationReport


def render_evaluation_markdown(report: EvaluationReport) -> str:
    """Render a comprehensive GitHub-flavored markdown representation of an EvaluationReport."""
    lines: list[str] = [
        "# RAGDiag Evaluation Report",
        "",
        f"- **Pipeline:** `{report.pipeline_name or 'N/A'}`",
        f"- **Dataset:** `{report.dataset_name}` (v{report.dataset_version})",
        f"- **Total Queries:** {report.total_queries}",
        f"- **Completed Queries:** {report.completed_queries}",
        f"- **Failed Queries:** {report.failed_queries}",
        "",
        "## Retrieval Metrics",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| Precision@{report.retrieval.k} | {report.retrieval.mean_precision_at_k:.2f} |",
        f"| Recall@{report.retrieval.k} | {report.retrieval.mean_recall_at_k:.2f} |",
        f"| MRR | {report.retrieval.mrr:.2f} |",
        "",
    ]

    if report.semantic is not None:
        c_rate = (
            f"{report.semantic.answer_correctness_rate:.2f}"
            if report.semantic.answer_correctness_rate is not None
            else "N/A"
        )
        g_rate = (
            f"{report.semantic.groundedness_rate:.2f}"
            if report.semantic.groundedness_rate is not None
            else "N/A"
        )
        lines.extend(
            [
                "## Semantic Quality",
                "",
                "| Metric | Value |",
                "| :--- | :--- |",
                f"| Answer Correctness | {c_rate} |",
                f"| Groundedness | {g_rate} |",
                "",
            ]
        )

    lines.extend(
        [
            "## Retrieval Latency",
            "",
            f"- **Mean:** {report.latency.mean_ms:.1f} ms",
            f"- **P50:** {report.latency.p50_ms:.1f} ms",
            f"- **P95:** {report.latency.p95_ms:.1f} ms",
            f"- **P99:** {report.latency.p99_ms:.1f} ms",
            "",
            "## Failure Analysis",
            "",
            "| Failure Category | Count | Action Recommendation |",
            "| :--- | :--- | :--- |",
        ]
    )

    for cat, count in report.diagnosis_counts.items():
        action = get_action_for_category(cat)
        lines.append(f"| `{cat}` | {count} | {action} |")

    lines.extend(
        [
            "",
            "## Health Profile",
            "",
            f"- **Score:** {report.health_profile.score:.1f}/100",
            f"- **Grade:** {report.health_profile.grade} (Status: {report.health_profile.status})",
            "",
        ]
    )

    if report.health_profile.strengths:
        lines.append("### Strengths")
        for s in report.health_profile.strengths:
            lines.append(f"- ✓ {s}")
        lines.append("")

    if report.health_profile.weaknesses:
        lines.append("### Weaknesses")
        for w in report.health_profile.weaknesses:
            lines.append(f"- ! {w}")
        lines.append("")

    if report.health_profile.recommendations:
        lines.append("### Recommendations")
        for r in report.health_profile.recommendations:
            lines.append(f"- → {r}")
        lines.append("")

    lines.extend(
        [
            "## Evaluation Confidence",
            "",
            f"- **Score:** {report.confidence.score:.1f}/100",
            f"- **Level:** {report.confidence.level}",
            "",
        ]
    )

    if report.confidence.reasons:
        lines.append("### Reasons")
        for reason in report.confidence.reasons:
            lines.append(f"- {reason}")
        lines.append("")

    return "\n".join(lines)


def render_comparison_markdown(report: ComparisonReport) -> str:
    """Render a comprehensive GitHub-flavored markdown representation of a ComparisonReport."""
    k = report.pipeline_a_report.retrieval.k
    lines: list[str] = [
        "# RAGDiag Pipeline Comparison Report",
        "",
        f"- **Pipeline A (Baseline):** `{report.pipeline_a_name}`",
        f"- **Pipeline B (Candidate):** `{report.pipeline_b_name}`",
        f"- **Dataset:** `{report.dataset_name}` (v{report.dataset_version})",
        f"- **Overall Winner:** `{report.overall_winner}`",
    ]
    if report.trade_off:
        lines.append(f"- **Trade-off:** {report.trade_off}")
    if report.summary:
        lines.append(f"- **Summary:** {report.summary}")

    lines.extend(
        [
            "",
            "## Overall Metric Deltas",
            "",
            "| Metric | Delta (B - A) |",
            "| :--- | :--- |",
            f"| Recall@{k} | {report.metric_deltas.recall_at_k:+.2f} |",
            f"| Precision@{k} | {report.metric_deltas.precision_at_k:+.2f} |",
            f"| MRR | {report.metric_deltas.mrr:+.2f} |",
            f"| Mean Retrieval Latency | {report.metric_deltas.mean_retrieval_ms:+.1f} ms |",
            f"| P95 Retrieval Latency | {report.metric_deltas.p95_retrieval_ms:+.1f} ms |",
            "",
            "## Query Outcomes",
            "",
            f"- **Improved:** {report.queries_improved}",
            f"- **Regressed:** {report.queries_regressed}",
            f"- **Unchanged:** {report.queries_unchanged}",
            "",
            "## Regression Analysis",
            "",
            (
                "- **Overall Regression:** "
                f"`{'YES' if report.regression_analysis.overall_regression else 'NO'}`"
            ),
            f"- **Summary:** {report.regression_analysis.summary}",
            f"- **Regressed Queries Count:** {report.regression_analysis.regressed_query_count}",
            "",
        ]
    )

    if report.regression_analysis.metric_regressions:
        lines.extend(
            [
                "### Regressed Metrics",
                "",
                "| Metric | Baseline | Current | Delta |",
                "| :--- | :--- | :--- | :--- |",
            ]
        )
        for mr in report.regression_analysis.metric_regressions:
            lines.append(
                f"| {mr.metric_name} | {mr.baseline_value:.2f} | "
                f"{mr.current_value:.2f} | {mr.delta:+.2f} |"
            )
        lines.append("")

    if report.regression_analysis.important_regressions:
        lines.append("### Important Regressions")
        for imp in report.regression_analysis.important_regressions:
            lines.append(f"- {imp}")
        lines.append("")

    return "\n".join(lines)


def render_diagnosis_markdown(report: EvaluationReport) -> str:
    """Render a clean GitHub-flavored markdown representation of a DiagnosisInspection."""
    diag = inspect_report(report)
    lines: list[str] = [
        "# RAGDiag Pipeline Diagnosis Report",
        "",
        f"- **Pipeline:** `{report.pipeline_name or 'N/A'}`",
        f"- **Dataset:** `{report.dataset_name}` (v{report.dataset_version})",
        f"- **Total Evaluated:** {diag.total_queries}",
        f"- **Passed:** {diag.passed_queries}",
        f"- **Failed:** {diag.failed_queries}",
        "",
        "## Top Failure Modes",
        "",
    ]

    if diag.failure_modes:
        lines.extend(
            [
                "| Rank | Failure Category | Query Count | Recommended Action |",
                "| :--- | :--- | :--- | :--- |",
            ]
        )
        for idx, fm in enumerate(diag.failure_modes, start=1):
            lines.append(f"| {idx} | `{fm.category}` | {fm.count} | {fm.action} |")
        lines.append("")
    else:
        lines.extend(["All queries passed successfully! No failure modes detected.", ""])

    if diag.important_queries:
        lines.extend(["## Important Queries", ""])
        for tf in diag.important_queries:
            cat_name = tf.category.value if hasattr(tf.category, "value") else str(tf.category)
            lines.extend(
                [
                    f"### `{tf.query_id}`: {cat_name}",
                    f"- **Reason:** {tf.reason}",
                    f"- **Action:** {tf.action}",
                ]
            )
            if tf.evidence:
                lines.append(f"- **Evidence:** {tf.evidence[0]}")
            lines.append("")

    lines.extend(
        [
            "## Health & Evaluation Confidence",
            "",
            (
                f"- **Health Score:** {diag.health_score:.1f}/100 "
                f"({diag.health_grade}, {diag.health_status})"
            ),
            f"- **Confidence Score:** {diag.confidence_score:.1f}/100 ({diag.confidence_level})",
            "",
        ]
    )

    return "\n".join(lines)
