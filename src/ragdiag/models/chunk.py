"""Domain model representing a retrieved document chunk."""

from pydantic import BaseModel


class RetrievedChunk(BaseModel):
    """A normalized chunk of text retrieved by a RAG pipeline.

    Attributes:
        id: Unique identifier for the chunk.
        text: Text content of the retrieved chunk.
        score: Optional relevance or similarity score assigned by the retriever.
        metadata: Optional arbitrary metadata (e.g., source URI, page number).
    """

    id: str
    text: str
    score: float | None = None
    metadata: dict[str, object] | None = None
