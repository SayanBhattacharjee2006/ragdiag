"""DiagnosisEngine orchestrating root-cause failure classification."""

from ragdiag.diagnosis.models import DiagnosisResult, FailureCategory
from ragdiag.diagnosis.rules import (
    classify_answer_failure,
    classify_context_sufficiency,
    classify_grounding_failure,
    classify_latency_outlier,
    classify_pipeline_failure,
    classify_ranking_failure,
    classify_retrieval_failure,
)
from ragdiag.models.result import EvaluationResult


class DiagnosisEngine:
    """Evaluates captured evidence in an EvaluationResult to diagnose root-cause failure modes.

    Follows a strict precedence hierarchy:
    1. Pipeline execution failure (status != 'completed') -> UNKNOWN
    2. Zero retrieval overlap -> WRONG_CHUNK_RETRIEVED
    3. Low rank of first relevant chunk -> WRONG_CHUNK_RANK
    4. Partial context coverage -> INSUFFICIENT_CONTEXT
    5. Hallucination / ungrounded answer -> RETRIEVED_BUT_NOT_GROUNDED
    6. Semantic answer inaccuracy -> ANSWER_INCORRECT
    7. Latency outlier (when quality checks pass) -> LATENCY_OUTLIER
    8. All checks passed -> PASS

    When quality failures coincide with high latency, the quality failure is
    preserved as primary while latency is recorded as secondary evidence.
    """

    def __init__(
        self,
        k: int = 5,
        rank_threshold: int = 3,
        latency_threshold_ms: float = 1000.0,
    ) -> None:
        """Initialize the diagnosis engine.

        Args:
            k: Retrieval rank cutoff K (default: 5).
            rank_threshold: Maximum acceptable rank for the first relevant chunk (default: 3).
            latency_threshold_ms: Milliseconds threshold for latency outliers (default: 1000.0).
        """
        self.k = k
        self.rank_threshold = rank_threshold
        self.latency_threshold_ms = latency_threshold_ms

    def diagnose(self, result: EvaluationResult) -> DiagnosisResult:
        """Diagnose an EvaluationResult and return a structured DiagnosisResult."""
        # 1. Pipeline execution failure
        pipeline_fail = classify_pipeline_failure(result.status, result.error)
        if pipeline_fail is not None:
            return pipeline_fail

        # Check secondary latency signal
        retrieval_ms = float(result.latency.get("retrieval_ms", 0.0))
        is_latency_outlier = retrieval_ms > self.latency_threshold_ms
        sec_latency_note = (
            f"Secondary signal: Retrieval latency was {retrieval_ms:.1f} ms "
            f"(exceeded threshold {self.latency_threshold_ms:.1f} ms)."
        )

        # 2. Retrieval failure (zero overlap)
        retrieval_fail = classify_retrieval_failure(
            result.expected_chunk_ids,
            result.retrieved_chunks,
            k=self.k,
        )
        if retrieval_fail is not None:
            if is_latency_outlier:
                retrieval_fail.evidence.append(sec_latency_note)
            return retrieval_fail

        # 3. Ranking failure (first relevant chunk beyond threshold)
        ranking_fail = classify_ranking_failure(
            result.expected_chunk_ids,
            result.retrieved_chunks,
            rank_threshold=self.rank_threshold,
        )
        if ranking_fail is not None:
            if is_latency_outlier:
                ranking_fail.evidence.append(sec_latency_note)
            return ranking_fail

        # 4. Insufficient context (partial overlap)
        context_fail = classify_context_sufficiency(
            result.expected_chunk_ids,
            result.retrieved_chunks,
            k=self.k,
        )
        if context_fail is not None:
            if is_latency_outlier:
                context_fail.evidence.append(sec_latency_note)
            return context_fail

        # 5. Grounding failure (hallucination)
        grounding_fail = classify_grounding_failure(
            result.metrics,
            judge_error=result.judge_error,
        )
        if grounding_fail is not None:
            if is_latency_outlier:
                grounding_fail.evidence.append(sec_latency_note)
            return grounding_fail

        # 6. Answer correctness failure
        answer_fail = classify_answer_failure(
            result.metrics,
            generated_answer=result.generated_answer,
            judge_error=result.judge_error,
        )
        if answer_fail is not None:
            if is_latency_outlier:
                answer_fail.evidence.append(sec_latency_note)
            return answer_fail

        # 7. Latency outlier (quality checks passed, but slow)
        if is_latency_outlier:
            latency_fail = classify_latency_outlier(
                result.latency,
                latency_threshold_ms=self.latency_threshold_ms,
            )
            if latency_fail is not None:
                return latency_fail

        # 8. PASS
        has_judge = "answer_correct" in result.metrics and result.judge_error is None
        if has_judge:
            reason = "Query passed all retrieval, grounding, correctness, and latency checks."
            evidence = [
                f"Recall@{self.k}: {result.metrics.get(f'recall_at_{self.k}', 1.0)}",
                f"Precision@{self.k}: {result.metrics.get(f'precision_at_{self.k}', 1.0)}",
                f"Grounded: {result.metrics.get('grounded')}",
                f"Answer correct: {result.metrics.get('answer_correct')}",
                f"Retrieval latency: {retrieval_ms:.1f} ms",
            ]
        else:
            reason = (
                "Retrieval checks passed; semantic answer quality was not "
                "evaluated because no judge was configured."
            )
            evidence = [
                f"Recall@{self.k}: {result.metrics.get(f'recall_at_{self.k}', 1.0)}",
                f"Precision@{self.k}: {result.metrics.get(f'precision_at_{self.k}', 1.0)}",
                f"Retrieval latency: {retrieval_ms:.1f} ms",
            ]

        return DiagnosisResult(
            category=FailureCategory.PASS,
            severity="info",
            confidence=1.0,
            reason=reason,
            evidence=evidence,
        )
