"""Evaluation execution engine for running RAG pipelines against golden datasets."""

import time

from ragdiag.models.chunk import RetrievedChunk
from ragdiag.models.dataset import GoldenDataset
from ragdiag.models.result import EvaluationResult
from ragdiag.models.sample import QuerySample
from ragdiag.pipeline.base import Pipeline


class Evaluator:
    """Orchestrates execution of a Pipeline against a GoldenDataset.

    Executes each query sample sequentially, records retrieval and generation
    latencies in milliseconds, captures raw evidence (chunks and answers),
    and isolates errors on a per-sample basis so that individual query failures
    never abort the overall evaluation run.
    """

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
            return EvaluationResult(
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
            return EvaluationResult(
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
            )

        # Stage 3: Completed Execution
        total_ms = retrieval_ms + generation_ms
        return EvaluationResult(
            query_id=sample.id,
            query=sample.query,
            expected_chunk_ids=sample.relevant_chunk_ids,
            retrieved_chunks=retrieved_chunks,
            generated_answer=generated_answer,
            metrics={},
            diagnosis={},
            latency={
                "retrieval_ms": round(retrieval_ms, 3),
                "generation_ms": round(generation_ms, 3),
                "total_ms": round(total_ms, 3),
            },
            status="completed",
            error=None,
        )
