"""OpenAI-backed LLM judge implementing semantic correctness and groundedness evaluations."""

import os
from collections.abc import Sequence

import openai
from openai import OpenAI
from pydantic import ValidationError

from ragdiag.judges.base import Judge, format_context
from ragdiag.judges.exceptions import (
    JudgeAuthenticationError,
    JudgeError,
    JudgeParseError,
    JudgeProviderError,
)
from ragdiag.judges.models import JudgeResult
from ragdiag.models.chunk import RetrievedChunk

JUDGE_SYSTEM_PROMPT = """\
You are an impartial, expert evaluation judge for Retrieval-Augmented Generation (RAG) systems.
Your task is to evaluate a generated answer on two distinct, independent dimensions:

1. Answer Correctness:
   - Compare the generated answer against the ground-truth expected answer.
   - Evaluate whether the generated answer accurately answers the query
     according to the expected answer.
   - Accept semantic equivalence; exact phrasing or word-for-word matching is NOT required.
   - If the generated answer contradicts, distorts, or omits key facts from
     the expected answer, mark answer_correct as false.

2. Groundedness:
   - Evaluate whether all factual assertions in the generated answer are
     strictly supported by the provided retrieved context.
   - The retrieved context chunks are the SOLE evidence source for groundedness.
   - Do NOT use the expected answer or prior knowledge as evidence for groundedness.
   - If the generated answer makes claims, figures, or inferences not substantiated
     by the context, mark grounded as false (hallucination).
   - If no context is provided and the answer makes factual claims, mark grounded as false.

Provide your evaluation as a structured output with:
- answer_correct: bool
- grounded: bool
- confidence: float (0.0 to 1.0)
- reason: concise explanation detailing both decisions.
"""


class OpenAIJudge(Judge):
    """LLM Judge backed by OpenAI's structured outputs API.

    Evaluates semantic correctness and context groundedness using model-level
    JSON schema enforcement (`client.beta.chat.completions.parse`).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        client: OpenAI | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Initialize the OpenAI judge.

        Args:
            model: Name of the OpenAI model to use (default: 'gpt-4o-mini').
            api_key: OpenAI API key. If omitted, reads from OPENAI_API_KEY env var.
            client: Optional pre-configured OpenAI client instance (for testing).
            timeout: Request timeout in seconds.

        Raises:
            JudgeAuthenticationError: If no API key is provided or found in the environment.
        """
        self.model = model
        self.timeout = timeout

        if client is not None:
            self.client = client
        else:
            resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not resolved_key:
                raise JudgeAuthenticationError(
                    "OpenAI API key is missing. Set the OPENAI_API_KEY environment "
                    "variable or pass api_key explicitly to OpenAIJudge."
                )
            self.client = OpenAI(api_key=resolved_key, timeout=timeout)

    def evaluate(
        self,
        query: str,
        expected_answer: str,
        generated_answer: str,
        context: Sequence[RetrievedChunk],
    ) -> JudgeResult:
        """Evaluate a query sample using OpenAI structured outputs.

        Args:
            query: User query string.
            expected_answer: Ground truth reference answer.
            generated_answer: Synthesized answer from the pipeline.
            context: Retrieved evidence chunks.

        Returns:
            A validated `JudgeResult`.

        Raises:
            JudgeAuthenticationError: On authentication rejection.
            JudgeProviderError: On timeouts, rate limits, or provider 5xx errors.
            JudgeParseError: If the output cannot be parsed into `JudgeResult`.
            JudgeError: On other unhandled failures.
        """
        formatted_context = format_context(context)

        user_content = (
            f"[User Query]\n{query}\n\n"
            f"[Expected Answer (Reference)]\n{expected_answer}\n\n"
            f"[Retrieved Context (Evidence for Groundedness)]\n{formatted_context}\n\n"
            f"[Generated Answer (To Evaluate)]\n{generated_answer}"
        )

        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                response_format=JudgeResult,
                temperature=0.0,
            )
        except openai.AuthenticationError as exc:
            raise JudgeAuthenticationError(f"OpenAI authentication failed: {exc}") from exc
        except (
            openai.APITimeoutError,
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.InternalServerError,
            openai.APIError,
        ) as exc:
            raise JudgeProviderError(
                f"OpenAI API provider error ({type(exc).__name__}): {exc}"
            ) from exc
        except Exception as exc:
            raise JudgeError(f"Unexpected error during OpenAI evaluation: {exc}") from exc

        try:
            choice = completion.choices[0]
            if choice.message.refusal:
                raise JudgeParseError(f"Model refused evaluation: {choice.message.refusal}")
            parsed = choice.message.parsed
            if parsed is None:
                raise JudgeParseError("OpenAI response message.parsed returned None.")
            return parsed
        except (IndexError, AttributeError, ValidationError) as exc:
            raise JudgeParseError(
                f"Failed to parse structured JudgeResult from OpenAI response: {exc}"
            ) from exc
