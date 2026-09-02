"""Retrieval quality metrics: Precision@K, Recall@K, and Reciprocal Rank."""

from collections.abc import Sequence

from ragdiag.metrics.models import RetrievalMetricResult
from ragdiag.models.chunk import RetrievedChunk


def precision_at_k(
    relevant_chunk_ids: Sequence[str] | set[str],
    retrieved_chunks: Sequence[RetrievedChunk],
    k: int = 5,
) -> float:
    """Calculate Precision@K for a single query.

    Precision@K measures the proportion of retrieved chunks in the top-K results
    that are relevant to the query.

    Denominator interpretation:
        Standard information retrieval interpretation is used. When the retriever
        returns fewer than K results, the denominator is the number of returned
        results (`min(k, len(retrieved_chunks))`) to avoid artificially penalizing
        a pipeline configured with a smaller result cutoff.
        If zero chunks are retrieved or K <= 0, returns 0.0.
        Duplicate retrieved chunk IDs are counted only once toward the numerator.

    Args:
        relevant_chunk_ids: Collection of ground-truth relevant chunk IDs.
        retrieved_chunks: Ranked sequence of retrieved chunks from the pipeline.
        k: Maximum rank depth to evaluate (default: 5).

    Returns:
        Precision@K as a float in the range [0.0, 1.0].
    """
    if k <= 0 or not retrieved_chunks:
        return 0.0

    relevant_set = set(relevant_chunk_ids)
    if not relevant_set:
        return 0.0

    top_k = retrieved_chunks[:k]
    denominator = min(k, len(retrieved_chunks))
    if denominator == 0:
        return 0.0

    # Count unique relevant documents retrieved in top-K
    relevant_retrieved = {chunk.id for chunk in top_k if chunk.id in relevant_set}
    return len(relevant_retrieved) / denominator


def recall_at_k(
    relevant_chunk_ids: Sequence[str] | set[str],
    retrieved_chunks: Sequence[RetrievedChunk],
    k: int = 5,
) -> float:
    """Calculate Recall@K for a single query.

    Recall@K measures the fraction of all ground-truth relevant chunks that were
    successfully retrieved in the top-K results.

    Args:
        relevant_chunk_ids: Collection of ground-truth relevant chunk IDs.
        retrieved_chunks: Ranked sequence of retrieved chunks from the pipeline.
        k: Maximum rank depth to evaluate (default: 5).

    Returns:
        Recall@K as a float in the range [0.0, 1.0].
    """
    if k <= 0 or not retrieved_chunks:
        return 0.0

    relevant_set = set(relevant_chunk_ids)
    if not relevant_set:
        return 0.0

    top_k = retrieved_chunks[:k]
    relevant_retrieved = {chunk.id for chunk in top_k if chunk.id in relevant_set}
    return len(relevant_retrieved) / len(relevant_set)


def reciprocal_rank(
    relevant_chunk_ids: Sequence[str] | set[str],
    retrieved_chunks: Sequence[RetrievedChunk],
) -> float:
    """Calculate Reciprocal Rank (RR) for a single query.

    Finds the 1-indexed rank of the first relevant retrieved chunk and returns
    its reciprocal (1 / rank). If no relevant chunk is retrieved, returns 0.0.

    Args:
        relevant_chunk_ids: Collection of ground-truth relevant chunk IDs.
        retrieved_chunks: Ranked sequence of retrieved chunks from the pipeline.

    Returns:
        Reciprocal rank as a float in the range [0.0, 1.0].
    """
    if not retrieved_chunks:
        return 0.0

    relevant_set = set(relevant_chunk_ids)
    if not relevant_set:
        return 0.0

    for rank, chunk in enumerate(retrieved_chunks, start=1):
        if chunk.id in relevant_set:
            return 1.0 / rank

    return 0.0


def compute_retrieval_metrics(
    relevant_chunk_ids: Sequence[str] | set[str],
    retrieved_chunks: Sequence[RetrievedChunk],
    k: int = 5,
) -> RetrievalMetricResult:
    """Compute all standard retrieval quality metrics for a single query.

    Args:
        relevant_chunk_ids: Collection of ground-truth relevant chunk IDs.
        retrieved_chunks: Ranked sequence of retrieved chunks from the pipeline.
        k: Rank depth threshold K (default: 5).

    Returns:
        A validated `RetrievalMetricResult` object.
    """
    return RetrievalMetricResult(
        precision_at_k=precision_at_k(relevant_chunk_ids, retrieved_chunks, k=k),
        recall_at_k=recall_at_k(relevant_chunk_ids, retrieved_chunks, k=k),
        reciprocal_rank=reciprocal_rank(relevant_chunk_ids, retrieved_chunks),
        k=k,
    )
