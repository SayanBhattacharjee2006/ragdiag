"""Domain model representing a golden evaluation query sample."""

from enum import StrEnum

from pydantic import BaseModel, ValidationInfo, field_validator


class QueryType(StrEnum):
    """Supported taxonomy of query types for golden evaluation samples."""

    FACTUAL = "factual"
    REASONING = "reasoning"
    MULTI_HOP = "multi-hop"


class QuerySample(BaseModel):
    """A golden evaluation query sample with ground truth context and answers.

    Attributes:
        id: Unique non-empty identifier for the sample.
        query: Non-empty user query string to be evaluated.
        expected_answer: Non-empty ground truth answer to the query.
        relevant_chunk_ids: Non-empty list of unique ground-truth chunk IDs.
        query_type: Category of query (one of 'factual', 'reasoning', 'multi-hop').
    """

    id: str
    query: str
    expected_answer: str
    relevant_chunk_ids: list[str]
    query_type: QueryType

    @field_validator("id", "query", "expected_answer", mode="after")
    @classmethod
    def validate_non_empty_string(cls, v: str, info: ValidationInfo) -> str:
        stripped = v.strip()
        if not stripped:
            field_name = info.field_name or "field"
            raise ValueError(f"Field '{field_name}' must not be empty or whitespace-only.")
        return stripped

    @field_validator("relevant_chunk_ids", mode="after")
    @classmethod
    def validate_chunk_ids(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("Field 'relevant_chunk_ids' must contain at least one chunk ID.")
        cleaned: list[str] = []
        seen: set[str] = set()
        for idx, chunk_id in enumerate(v):
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError(
                    f"Chunk ID at index {idx} in 'relevant_chunk_ids' must not be empty."
                )
            stripped = chunk_id.strip()
            if stripped in seen:
                raise ValueError(f"Duplicate chunk ID '{stripped}' found in 'relevant_chunk_ids'.")
            seen.add(stripped)
            cleaned.append(stripped)
        return cleaned
