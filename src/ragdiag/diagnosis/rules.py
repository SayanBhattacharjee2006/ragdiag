"""Pure, deterministic rule functions for failure mode classification."""

from ragdiag.diagnosis.models import DiagnosisResult, FailureCategory
from ragdiag.models.chunk import RetrievedChunk


def classify_pipeline_failure(
    status: str,
    error: str | None,
) -> DiagnosisResult | None:
    """Classify pipeline infrastructure or execution failures."""
    if status != "completed":
        return DiagnosisResult(
            category=FailureCategory.UNKNOWN,
            severity="major",
            confidence=1.0,
            reason=f"Pipeline execution failed: {error or 'Unknown error'}",
            evidence=[
                f"Execution status: {status}",
                f"Error details: {error or 'None'}",
            ],
        )
    return None


def classify_retrieval_failure(
    expected_chunk_ids: list[str],
    retrieved_chunks: list[RetrievedChunk],
    k: int = 5,
) -> DiagnosisResult | None:
    """Classify total retrieval failure where no expected chunks are found."""
    if not expected_chunk_ids:
        return None

    retrieved_ids = [c.id for c in retrieved_chunks[:k]]
    overlap = set(expected_chunk_ids) & set(retrieved_ids)

    if not overlap:
        return DiagnosisResult(
            category=FailureCategory.WRONG_CHUNK_RETRIEVED,
            severity="major",
            confidence=1.0,
            reason="None of the expected relevant chunks were retrieved in the top results.",
            evidence=[
                f"Expected chunks: {', '.join(expected_chunk_ids)}",
                f"Retrieved chunks: {', '.join(retrieved_ids) if retrieved_ids else 'None'}",
                f"Recall@{k}: 0.0",
            ],
        )
    return None


def classify_ranking_failure(
    expected_chunk_ids: list[str],
    retrieved_chunks: list[RetrievedChunk],
    rank_threshold: int = 3,
) -> DiagnosisResult | None:
    """Classify ranking failure where first relevant chunk appears beyond rank threshold."""
    if not expected_chunk_ids:
        return None

    first_rank: int | None = None
    first_chunk_id: str | None = None

    for idx, chunk in enumerate(retrieved_chunks, start=1):
        if chunk.id in expected_chunk_ids:
            first_rank = idx
            first_chunk_id = chunk.id
            break

    if first_rank is not None and first_rank > rank_threshold:
        return DiagnosisResult(
            category=FailureCategory.WRONG_CHUNK_RANK,
            severity="warning",
            confidence=1.0,
            reason=(
                f"First relevant chunk was retrieved at rank {first_rank}, "
                f"which exceeds the rank threshold ({rank_threshold})."
            ),
            evidence=[
                f"Expected chunks: {', '.join(expected_chunk_ids)}",
                f"First relevant chunk '{first_chunk_id}' retrieved at rank {first_rank}.",
                f"Rank threshold: {rank_threshold}",
            ],
        )
    return None


def classify_context_sufficiency(
    expected_chunk_ids: list[str],
    retrieved_chunks: list[RetrievedChunk],
    k: int = 5,
) -> DiagnosisResult | None:
    """Classify partial context retrieval where only a subset of required chunks was retrieved."""
    if len(expected_chunk_ids) <= 1:
        return None

    retrieved_ids = [c.id for c in retrieved_chunks[:k]]
    expected_set = set(expected_chunk_ids)
    found = expected_set & set(retrieved_ids)
    missing = expected_set - found

    if 0 < len(found) < len(expected_set):
        recall_val = len(found) / len(expected_set)
        return DiagnosisResult(
            category=FailureCategory.INSUFFICIENT_CONTEXT,
            severity="warning",
            confidence=1.0,
            reason=(
                f"Only {len(found)} of {len(expected_set)} required context chunks were retrieved."
            ),
            evidence=[
                f"Expected chunks: {', '.join(expected_chunk_ids)}",
                f"Found chunks: {', '.join(sorted(found))}",
                f"Missing chunks: {', '.join(sorted(missing))}",
                f"Recall@{k}: {recall_val:.2f}",
            ],
        )
    return None


def classify_grounding_failure(
    metrics: dict[str, object],
    judge_error: str | None = None,
) -> DiagnosisResult | None:
    """Classify hallucination where answer is not supported by retrieved context."""
    if judge_error is not None:
        return None

    if metrics.get("grounded") is False:
        conf_val = metrics.get("judge_confidence")
        conf = float(conf_val) if isinstance(conf_val, (int, float)) else 1.0
        evidence = [
            "Relevant context was retrieved.",
            "LLM judge marked grounded=false.",
            f"Answer correctness={metrics.get('answer_correct')}.",
        ]
        if "judge_reason" in metrics and isinstance(metrics["judge_reason"], str):
            evidence.append(f"Judge reason: {metrics['judge_reason']}")

        return DiagnosisResult(
            category=FailureCategory.RETRIEVED_BUT_NOT_GROUNDED,
            severity="major",
            confidence=conf,
            reason=(
                "Relevant context was retrieved, but the generated answer contained claims "
                "unsupported by the retrieved context (hallucination)."
            ),
            evidence=evidence,
        )
    return None


def classify_answer_failure(
    metrics: dict[str, object],
    expected_answer: str | None = None,
    generated_answer: str | None = None,
    judge_error: str | None = None,
) -> DiagnosisResult | None:
    """Classify semantic answer inaccuracy when context was grounded or available."""
    if judge_error is not None:
        return None

    if metrics.get("answer_correct") is False:
        conf_val = metrics.get("judge_confidence")
        conf = float(conf_val) if isinstance(conf_val, (int, float)) else 1.0
        evidence = [
            f"Expected answer: {expected_answer or 'None'}",
            f"Generated answer: {generated_answer or 'None'}",
            f"LLM judge marked answer_correct=false (grounded={metrics.get('grounded')}).",
        ]
        if "judge_reason" in metrics and isinstance(metrics["judge_reason"], str):
            evidence.append(f"Judge reason: {metrics['judge_reason']}")

        return DiagnosisResult(
            category=FailureCategory.ANSWER_INCORRECT,
            severity="major",
            confidence=conf,
            reason=(
                "The generated answer contradicts or fails to accurately answer the query "
                "according to the expected answer."
            ),
            evidence=evidence,
        )
    return None


def classify_latency_outlier(
    latency: dict[str, float],
    latency_threshold_ms: float = 1000.0,
) -> DiagnosisResult | None:
    """Classify retrieval latency outlier when no quality failure is present."""
    retrieval_ms = float(latency.get("retrieval_ms", 0.0))
    if retrieval_ms > latency_threshold_ms:
        return DiagnosisResult(
            category=FailureCategory.LATENCY_OUTLIER,
            severity="warning",
            confidence=1.0,
            reason=(
                f"Retrieval latency ({retrieval_ms:.1f} ms) exceeded threshold "
                f"({latency_threshold_ms:.1f} ms)."
            ),
            evidence=[
                f"Retrieval latency: {retrieval_ms:.1f} ms",
                f"Threshold: {latency_threshold_ms:.1f} ms",
            ],
        )
    return None
