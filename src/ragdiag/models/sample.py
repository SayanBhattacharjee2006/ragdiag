"""Domain model representing a golden evaluation query sample."""

from pydantic import BaseModel, Field


class QuerySample(BaseModel):
    """A golden evaluation query sample with ground truth context and answers.

    Attributes:
        id: Unique identifier for the sample.
        query: The user query string to be evaluated.
        expected_answer: Ground truth answer to the query.
        relevant_chunk_ids: List of ground-truth chunk IDs relevant to the query.
        query_type: Category or type of query (e.g., 'factoid', 'multi-hop', 'summarization').
    """

    id: str
    query: str
    expected_answer: str
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    query_type: str = "general"
