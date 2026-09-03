"""Evaluation execution engine for running RAG pipelines against golden datasets."""

import time

from ragdiag.diagnosis.classifier import DiagnosisEngine
from ragdiag.diagnosis.models import DiagnosisResult, FailureCategory
from ragdiag.judges.base import Judge
from ragdiag.metrics.retrieval import compute_retrieval_metrics
from ragdiag.models.chunk import RetrievedChunk
from ragdiag.models.dataset import GoldenDataset
from ragdiag.models.result import EvaluationResult
from ragdiag.models.sample import QuerySample
from ragdiag.pipeline.base import Pipeline


class Evaluator:
    """Orchestrates execution of a Pipeline against a GoldenDataset.

    Executes each query sample sequentially, records retrieval and generation
    latencies in milliseconds, captures raw evidence (chunks and answers),
    calculates deterministic retrieval metrics (Precision@K, Recall@K, Reciprocal Rank),
    optionally invokes a semantic LLM judge, assigns root-cause failure diagnoses,
    and isolates errors on a per-sample basis so that individual query failures
    never abort the overall evaluation run.
    """

    def __init__(
        self,
        k: int = 5,
        judge: Judge | None = None,
        diagnosis_engine: DiagnosisEngine | None = None,
    ) -> None:
        """Initialize the Evaluator.

        Args:
            k: The rank depth cutoff K for retrieval metrics (default: 5).
            judge: Optional `Judge` instance for semantic evaluation (default: None).
            diagnosis_engine: Optional `DiagnosisEngine` for root-cause diagnosis.
        """
        self.k = k
        self.judge = judge
        self.diagnosis_engine = diagnosis_engine or DiagnosisEngine(k=k)

    def evaluate(
        self,
        pipeline: Pipeline,
        dataset: GoldenDataset,
    ) -> list[EvaluationResult]:
        """Execute all query samples in the dataset through the pipeline.

        Args:
            pipeline: The RAG pipeline instance to evaluate.
            dataset: The validated golden evaluation dataset.

        Returns:
            A list of `EvaluationResult` objects, one for each sample in the dataset.
        """
        results: list[EvaluationResult] = []
        for sample in dataset.samples:
            result = self.execute_sample(sample, pipeline)
            results.append(result)
        return results

    def execute_sample(
        self,
        sample: QuerySample,
        pipeline: Pipeline,
    ) -> EvaluationResult:
        """Execute a single query sample through retrieval and generation stages.

        Args:
            sample: The query sample to execute.
            pipeline: The RAG pipeline instance.

        Returns:
            An `EvaluationResult` containing chunks, answer, latencies (in ms),
            status ('completed' or 'failed'), and error message if any.
        """
        query_type_val = (
            sample.query_type.value
            if hasattr(sample.query_type, "value")
            else str(sample.query_type)
        )
        retrieval_start = time.perf_counter()
        retrieval_ms = 0.0
        retrieved_chunks: list[RetrievedChunk] = []

        # Stage 1: Retrieval
        try:
            raw_chunks = pipeline.retrieve(sample.query)
            retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0

            if not isinstance(raw_chunks, list):
                raise TypeError(
                    f"retrieve() returned {type(raw_chunks).__name__}, "
                    f"expected list[RetrievedChunk]"
                )
            for idx, chunk in enumerate(raw_chunks):
                if not isinstance(chunk, RetrievedChunk):
                    raise TypeError(
                        f"Chunk at index {idx} in retrieve() output is {type(chunk).__name__}, "
                        f"expected RetrievedChunk"
                    )
            retrieved_chunks = raw_chunks
        except Exception as exc:
            if retrieval_ms == 0.0:
                retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0
            return self._diagnose_result(
                EvaluationResult(
                    query_id=sample.id,
                    query=sample.query,
                    expected_chunk_ids=sample.relevant_chunk_ids,
                    retrieved_chunks=[],
                    generated_answer=None,
                    metrics={},
                    diagnosis={},
                    latency={
                        "retrieval_ms": round(retrieval_ms, 3),
                        "generation_ms": 0.0,
                        "total_ms": round(retrieval_ms, 3),
                    },
                    status="failed",
                    error=f"retrieval failed: {type(exc).__name__}: {exc}",
                    query_type=query_type_val,
                )
            )

        # Stage 2: Generation
        generation_start = time.perf_counter()
        generation_ms = 0.0
        try:
            raw_answer = pipeline.generate(sample.query, retrieved_chunks)
            generation_ms = (time.perf_counter() - generation_start) * 1000.0

            if not isinstance(raw_answer, str):
                raise TypeError(f"generate() returned {type(raw_answer).__name__}, expected str")
            generated_answer = raw_answer
        except Exception as exc:
            if generation_ms == 0.0:
                generation_ms = (time.perf_counter() - generation_start) * 1000.0
            total_ms = retrieval_ms + generation_ms
            return self._diagnose_result(
                EvaluationResult(
                    query_id=sample.id,
                    query=sample.query,
                    expected_chunk_ids=sample.relevant_chunk_ids,
                    retrieved_chunks=retrieved_chunks,
                    generated_answer=None,
                    metrics={},
                    diagnosis={},
                    latency={
                        "retrieval_ms": round(retrieval_ms, 3),
                        "generation_ms": round(generation_ms, 3),
                        "total_ms": round(total_ms, 3),
                    },
                    status="failed",
                    error=f"generation failed: {type(exc).__name__}: {exc}",
                    query_type=query_type_val,
                )
            )

        # Stage 3: Completed Execution
        total_ms = retrieval_ms + generation_ms
        retrieval_metrics = compute_retrieval_metrics(
            relevant_chunk_ids=sample.relevant_chunk_ids,
            retrieved_chunks=retrieved_chunks,
            k=self.k,
        )
        metrics: dict[str, object] = {
            f"precision_at_{self.k}": retrieval_metrics.precision_at_k,
            f"recall_at_{self.k}": retrieval_metrics.recall_at_k,
            "reciprocal_rank": retrieval_metrics.reciprocal_rank,
        }

        latency: dict[str, float] = {
            "retrieval_ms": round(retrieval_ms, 3),
            "generation_ms": round(generation_ms, 3),
            "total_ms": round(total_ms, 3),
        }

        judge_error: str | None = None
        if self.judge is not None:
            judge_start = time.perf_counter()
            try:
                judge_result = self.judge.evaluate(
                    query=sample.query,
                    expected_answer=sample.expected_answer,
                    generated_answer=generated_answer,
                    context=retrieved_chunks,
                )
                judge_ms = (time.perf_counter() - judge_start) * 1000.0
                latency["judge_ms"] = round(judge_ms, 3)
                metrics["answer_correct"] = judge_result.answer_correct
                metrics["grounded"] = judge_result.grounded
                metrics["judge_confidence"] = judge_result.confidence
                metrics["judge_reason"] = judge_result.reason
            except Exception as exc:
                judge_ms = (time.perf_counter() - judge_start) * 1000.0
                latency["judge_ms"] = round(judge_ms, 3)
                judge_error = f"judge failed: {type(exc).__name__}: {exc}"

        return self._diagnose_result(
            EvaluationResult(
                query_id=sample.id,
                query=sample.query,
                expected_chunk_ids=sample.relevant_chunk_ids,
                retrieved_chunks=retrieved_chunks,
                generated_answer=generated_answer,
                metrics=metrics,
                diagnosis={},
                latency=latency,
                status="completed",
                error=None,
                query_type=query_type_val,
                judge_error=judge_error,
            )
        )

    def _diagnose_result(self, result: EvaluationResult) -> EvaluationResult:
        """Assign diagnostic classification to an EvaluationResult safely."""
        try:
            result.diagnosis = self.diagnosis_engine.diagnose(result)
        except Exception as exc:
            result.diagnosis = DiagnosisResult(
                category=FailureCategory.UNKNOWN,
                severity="major",
                confidence=0.0,
                reason=f"Diagnosis engine error: {type(exc).__name__}: {exc}",
                evidence=[str(exc)],
            )
        return result
