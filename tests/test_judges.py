"""Comprehensive unit tests for the LLM Judge system, OpenAI adapter, and Evaluator integration."""

from collections.abc import Sequence
from unittest.mock import MagicMock

import openai
import pytest
from pydantic import ValidationError

from ragdiag.judges.base import Judge, format_context
from ragdiag.judges.exceptions import (
    JudgeAuthenticationError,
    JudgeParseError,
    JudgeProviderError,
)
from ragdiag.judges.models import JudgeResult
from ragdiag.judges.openai import OpenAIJudge
from ragdiag.metrics.aggregation import aggregate_metrics
from ragdiag.models.chunk import RetrievedChunk
from ragdiag.models.result import EvaluationResult
from ragdiag.models.sample import QuerySample, QueryType
from ragdiag.pipeline.base import Pipeline
from ragdiag.runner.evaluator import Evaluator

# ==============================================================================
# Helper Fixtures & Mock Implementations
# ==============================================================================


class FakeDeterministicJudge(Judge):
    """Deterministic mock judge returning pre-configured results."""

    def __init__(
        self,
        answer_correct: bool = True,
        grounded: bool = True,
        confidence: float = 0.95,
        reason: str = "Test reason",
        should_fail: bool = False,
    ) -> None:
        self.answer_correct = answer_correct
        self.grounded = grounded
        self.confidence = confidence
        self.reason = reason
        self.should_fail = should_fail
        self.call_count = 0

    def evaluate(
        self,
        query: str,
        expected_answer: str,
        generated_answer: str,
        context: Sequence[RetrievedChunk],
    ) -> JudgeResult:
        self.call_count += 1
        if self.should_fail:
            raise JudgeProviderError("Simulated LLM provider failure")
        return JudgeResult(
            answer_correct=self.answer_correct,
            grounded=self.grounded,
            confidence=self.confidence,
            reason=self.reason,
        )


class MockPipeline(Pipeline):
    """Simple pipeline for testing Evaluator with judge."""

    name = "mock_pipeline"

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(id="c1", text="Policy details for refunds.", score=0.9),
        ]

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> str:
        return "Refunds are processed in 5-7 business days."


# ==============================================================================
# JudgeResult Model Tests
# ==============================================================================


class TestJudgeResultModel:
    """Tests for the JudgeResult Pydantic model validation."""

    def test_valid_result(self) -> None:
        """Valid JudgeResult creation with expected fields."""
        res = JudgeResult(
            answer_correct=True,
            grounded=True,
            confidence=0.92,
            reason="Accurate and grounded.",
        )
        assert res.answer_correct is True
        assert res.grounded is True
        assert res.confidence == 0.92
        assert res.reason == "Accurate and grounded."

    def test_confidence_boundaries(self) -> None:
        """Confidence boundaries 0.0 and 1.0 are valid."""
        res_zero = JudgeResult(answer_correct=False, grounded=False, confidence=0.0, reason="No.")
        assert res_zero.confidence == 0.0

        res_one = JudgeResult(answer_correct=True, grounded=True, confidence=1.0, reason="Yes.")
        assert res_one.confidence == 1.0

    def test_invalid_confidence_below_zero(self) -> None:
        """Confidence < 0.0 raises ValidationError."""
        with pytest.raises(ValidationError):
            JudgeResult(
                answer_correct=True,
                grounded=True,
                confidence=-0.01,
                reason="Invalid",
            )

    def test_invalid_confidence_above_one(self) -> None:
        """Confidence > 1.0 raises ValidationError."""
        with pytest.raises(ValidationError):
            JudgeResult(
                answer_correct=True,
                grounded=True,
                confidence=1.05,
                reason="Invalid",
            )


# ==============================================================================
# Context Formatting Tests
# ==============================================================================


class TestContextFormatting:
    """Tests for format_context helper."""

    def test_format_multiple_chunks(self) -> None:
        """Preserves chunk IDs and content in deterministic markdown blocks."""
        chunks = [
            RetrievedChunk(id="doc_17", text="Refunds take 7 business days."),
            RetrievedChunk(id="doc_21", text="Late requests are denied."),
        ]
        formatted = format_context(chunks)
        expected = "[doc_17]\nRefunds take 7 business days.\n\n[doc_21]\nLate requests are denied."
        assert formatted == expected

    def test_format_empty_context(self) -> None:
        """Empty chunk sequence produces predictable fallback message."""
        assert format_context([]) == "No context retrieved."


# ==============================================================================
# Evaluator Integration with Judge Tests
# ==============================================================================


class TestEvaluatorJudgeIntegration:
    """Tests evaluating Evaluator interactions with a Judge."""

    @pytest.fixture
    def sample(self) -> QuerySample:
        return QuerySample(
            id="q1",
            query="What is the refund period?",
            expected_answer="Refunds take 5-7 business days.",
            relevant_chunk_ids=["c1"],
            query_type=QueryType.FACTUAL,
        )

    def test_evaluator_without_judge(self, sample: QuerySample) -> None:
        """When no judge is supplied, no semantic metrics are populated."""
        pipeline = MockPipeline()
        evaluator = Evaluator(k=5, judge=None)

        result = evaluator.execute_sample(sample, pipeline)
        assert result.status == "completed"
        assert result.error is None
        assert result.judge_error is None
        assert "precision_at_5" in result.metrics
        assert "answer_correct" not in result.metrics
        assert "grounded" not in result.metrics
        assert "judge_confidence" not in result.metrics
        assert "judge_ms" not in result.latency

    def test_evaluator_with_successful_judge(self, sample: QuerySample) -> None:
        """When judge succeeds, semantic metrics and judge_ms are recorded."""
        pipeline = MockPipeline()
        fake_judge = FakeDeterministicJudge(
            answer_correct=True,
            grounded=True,
            confidence=0.96,
            reason="Matches policy exactly.",
        )
        evaluator = Evaluator(k=5, judge=fake_judge)

        result = evaluator.execute_sample(sample, pipeline)
        assert result.status == "completed"
        assert result.error is None
        assert result.judge_error is None
        assert fake_judge.call_count == 1

        # Semantic metrics populated
        assert result.metrics["answer_correct"] is True
        assert result.metrics["grounded"] is True
        assert result.metrics["judge_confidence"] == 0.96
        assert result.metrics["judge_reason"] == "Matches policy exactly."

        # Retrieval metrics preserved
        assert result.metrics["precision_at_5"] == 1.0
        assert result.metrics["recall_at_5"] == 1.0

        # Judge latency recorded
        assert "judge_ms" in result.latency
        assert result.latency["judge_ms"] >= 0.0

    def test_evaluator_with_judge_failure(self, sample: QuerySample) -> None:
        """Judge failure does NOT fail pipeline execution; records judge_error."""
        pipeline = MockPipeline()
        failing_judge = FakeDeterministicJudge(should_fail=True)
        evaluator = Evaluator(k=5, judge=failing_judge)

        result = evaluator.execute_sample(sample, pipeline)
        # Pipeline execution succeeded
        assert result.status == "completed"
        assert result.error is None
        # Judge error is recorded explicitly
        assert result.judge_error is not None
        assert "JudgeProviderError" in result.judge_error
        # Retrieval metrics preserved
        assert result.metrics["precision_at_5"] == 1.0
        # Does NOT create false answer_correct or grounded values
        assert "answer_correct" not in result.metrics
        assert "grounded" not in result.metrics
        assert "judge_ms" in result.latency

    def test_judge_not_called_on_pipeline_failure(self) -> None:
        """If retrieval fails, judge is never invoked."""

        class FailingRetrievalPipeline(Pipeline):
            name = "fail"

            def retrieve(self, query: str) -> list[RetrievedChunk]:
                raise RuntimeError("Retrieval crash")

            def generate(self, query: str, chunks: list[RetrievedChunk]) -> str:
                return ""

        fake_judge = FakeDeterministicJudge()
        evaluator = Evaluator(k=5, judge=fake_judge)
        sample = QuerySample(
            id="q1",
            query="Query",
            expected_answer="Ans",
            relevant_chunk_ids=["c1"],
            query_type=QueryType.FACTUAL,
        )

        result = evaluator.execute_sample(sample, FailingRetrievalPipeline())
        assert result.status == "failed"
        assert fake_judge.call_count == 0


# ==============================================================================
# Aggregation Tests with Semantic Metrics
# ==============================================================================


class TestSemanticAggregation:
    """Tests for aggregate_metrics when semantic metrics are present."""

    def test_all_judged_success(self) -> None:
        """Aggregates correctness rate, groundedness rate, and mean confidence."""
        r1 = EvaluationResult(
            query_id="q1",
            query="q1",
            status="completed",
            expected_chunk_ids=["c1"],
            retrieved_chunks=[RetrievedChunk(id="c1", text="t")],
            metrics={
                "precision_at_5": 1.0,
                "recall_at_5": 1.0,
                "reciprocal_rank": 1.0,
                "answer_correct": True,
                "grounded": True,
                "judge_confidence": 0.9,
            },
            latency={"retrieval_ms": 5.0, "judge_ms": 100.0},
        )
        r2 = EvaluationResult(
            query_id="q2",
            query="q2",
            status="completed",
            expected_chunk_ids=["c2"],
            retrieved_chunks=[RetrievedChunk(id="c2", text="t")],
            metrics={
                "precision_at_5": 1.0,
                "recall_at_5": 1.0,
                "reciprocal_rank": 1.0,
                "answer_correct": False,
                "grounded": True,
                "judge_confidence": 0.8,
            },
            latency={"retrieval_ms": 6.0, "judge_ms": 120.0},
        )

        report = aggregate_metrics([r1, r2], k=5)
        assert report.total_queries == 2
        assert report.completed_queries == 2
        assert report.judged_queries == 2
        assert report.judge_failures == 0
        assert report.answer_correctness_rate == pytest.approx(0.5)  # 1 of 2
        assert report.groundedness_rate == pytest.approx(1.0)  # 2 of 2
        assert report.mean_judge_confidence == pytest.approx(0.85)  # (0.9 + 0.8) / 2
        assert report.judge_latency is not None
        assert report.judge_latency.count == 2
        assert report.judge_latency.mean_ms == pytest.approx(110.0)

    def test_mixed_judged_and_judge_failure(self) -> None:
        """Judge failures are counted separately and excluded from rates."""
        r_success = EvaluationResult(
            query_id="q1",
            query="q1",
            status="completed",
            expected_chunk_ids=["c1"],
            retrieved_chunks=[RetrievedChunk(id="c1", text="t")],
            metrics={
                "precision_at_5": 1.0,
                "recall_at_5": 1.0,
                "reciprocal_rank": 1.0,
                "answer_correct": True,
                "grounded": True,
                "judge_confidence": 0.95,
            },
            latency={"retrieval_ms": 5.0, "judge_ms": 90.0},
        )
        r_fail = EvaluationResult(
            query_id="q2",
            query="q2",
            status="completed",
            expected_chunk_ids=["c2"],
            retrieved_chunks=[RetrievedChunk(id="c2", text="t")],
            metrics={"precision_at_5": 1.0, "recall_at_5": 1.0, "reciprocal_rank": 1.0},
            latency={"retrieval_ms": 5.0, "judge_ms": 500.0},
            judge_error="TimeoutError: Request timed out",
        )

        report = aggregate_metrics([r_success, r_fail], k=5)
        assert report.total_queries == 2
        assert report.completed_queries == 2
        assert report.judged_queries == 1
        assert report.judge_failures == 1
        # Correctness computed ONLY on judged queries (1/1 = 1.0)
        assert report.answer_correctness_rate == 1.0
        assert report.groundedness_rate == 1.0
        assert report.mean_judge_confidence == 0.95

    def test_no_judged_queries(self) -> None:
        """When evaluation runs without a judge, rates remain None."""
        r = EvaluationResult(
            query_id="q1",
            query="q1",
            status="completed",
            expected_chunk_ids=["c1"],
            retrieved_chunks=[RetrievedChunk(id="c1", text="t")],
            metrics={"precision_at_5": 1.0, "recall_at_5": 1.0, "reciprocal_rank": 1.0},
            latency={"retrieval_ms": 5.0},
        )
        report = aggregate_metrics([r], k=5)
        assert report.judged_queries == 0
        assert report.judge_failures == 0
        assert report.answer_correctness_rate is None
        assert report.groundedness_rate is None
        assert report.mean_judge_confidence is None
        assert report.judge_latency is None


# ==============================================================================
# OpenAIJudge Adapter Tests (Mocked Client)
# ==============================================================================


class TestOpenAIJudgeMocked:
    """Tests for OpenAIJudge verifying request structure, parsing, and exception mapping."""

    def test_missing_api_key_raises_auth_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without API key in args or environment, raises JudgeAuthenticationError."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(JudgeAuthenticationError, match="OpenAI API key is missing"):
            OpenAIJudge(api_key=None)

    def test_successful_evaluation(self) -> None:
        """Verify chat.completions.parse is called with schema and parsed result is returned."""
        mock_client = MagicMock()
        mock_parsed = JudgeResult(
            answer_correct=True,
            grounded=True,
            confidence=0.94,
            reason="Fully grounded and accurate.",
        )
        mock_message = MagicMock()
        mock_message.refusal = None
        mock_message.parsed = mock_parsed

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        mock_client.beta.chat.completions.parse.return_value = mock_completion

        judge = OpenAIJudge(model="gpt-4o", client=mock_client)
        context = [RetrievedChunk(id="c1", text="Refund period is 7 days.")]
        result = judge.evaluate(
            query="What is the refund period?",
            expected_answer="7 days.",
            generated_answer="Refunds take 7 days.",
            context=context,
        )

        assert result == mock_parsed

        # Check call arguments
        call_kwargs = mock_client.beta.chat.completions.parse.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["response_format"] == JudgeResult
        assert call_kwargs["temperature"] == 0.0

        messages = call_kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "Answer Correctness" in messages[0]["content"]
        assert "Groundedness" in messages[0]["content"]

        user_prompt = messages[1]["content"]
        assert "[User Query]\nWhat is the refund period?" in user_prompt
        assert "[Expected Answer (Reference)]\n7 days." in user_prompt
        assert "[c1]\nRefund period is 7 days." in user_prompt
        assert "[Generated Answer (To Evaluate)]\nRefunds take 7 days." in user_prompt

    def test_auth_error_mapping(self) -> None:
        """openai.AuthenticationError is mapped to JudgeAuthenticationError."""
        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.side_effect = openai.AuthenticationError(
            message="Invalid API key", response=MagicMock(), body=None
        )

        judge = OpenAIJudge(client=mock_client)
        with pytest.raises(JudgeAuthenticationError, match="OpenAI authentication failed"):
            judge.evaluate("q", "exp", "gen", [])

    def test_timeout_error_mapping(self) -> None:
        """openai.APITimeoutError is mapped to JudgeProviderError."""
        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.side_effect = openai.APITimeoutError(
            request=MagicMock()
        )

        judge = OpenAIJudge(client=mock_client)
        with pytest.raises(JudgeProviderError, match="OpenAI API provider error"):
            judge.evaluate("q", "exp", "gen", [])

    def test_rate_limit_error_mapping(self) -> None:
        """openai.RateLimitError is mapped to JudgeProviderError."""
        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.side_effect = openai.RateLimitError(
            message="Rate limit exceeded", response=MagicMock(), body=None
        )

        judge = OpenAIJudge(client=mock_client)
        with pytest.raises(JudgeProviderError, match="OpenAI API provider error"):
            judge.evaluate("q", "exp", "gen", [])

    def test_refusal_raises_parse_error(self) -> None:
        """Model refusal raises JudgeParseError."""
        mock_client = MagicMock()
        mock_message = MagicMock()
        mock_message.refusal = "I cannot evaluate this request."
        mock_message.parsed = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_client.beta.chat.completions.parse.return_value = mock_completion

        judge = OpenAIJudge(client=mock_client)
        with pytest.raises(JudgeParseError, match="Model refused evaluation"):
            judge.evaluate("q", "exp", "gen", [])

    def test_parsed_none_raises_parse_error(self) -> None:
        """Parsed message being None raises JudgeParseError."""
        mock_client = MagicMock()
        mock_message = MagicMock()
        mock_message.refusal = None
        mock_message.parsed = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_client.beta.chat.completions.parse.return_value = mock_completion

        judge = OpenAIJudge(client=mock_client)
        with pytest.raises(JudgeParseError, match="response message.parsed returned None"):
            judge.evaluate("q", "exp", "gen", [])
