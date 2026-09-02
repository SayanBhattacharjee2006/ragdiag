"""Base pipeline interface for RAGDiag adapters."""

from abc import ABC, abstractmethod

from ragdiag.models.chunk import RetrievedChunk


class Pipeline(ABC):
    """Abstract base class defining the RAGDiag pipeline adapter interface.

    Developers wrap their custom RAG application (built with LangChain, LlamaIndex,
    custom vector stores, or bespoke code) by subclassing this class and implementing
    `retrieve()` and `generate()`.

    The evaluator operates exclusively against this interface:
    - `retrieve()` must return normalized `RetrievedChunk` objects.
    - `generate()` receives the query and those normalized chunks to produce a final answer string.

    This design guarantees that RAGDiag's evaluation and diagnosis engine remains
    completely decoupled from the developer's underlying RAG framework or storage layers.
    """

    name: str = "default_pipeline"

    def __init__(self, name: str | None = None) -> None:
        """Initialize the pipeline adapter with an optional display name.

        Args:
            name: Human-readable name identifying this pipeline configuration.
        """
        if name is not None:
            self.name = name

    @abstractmethod
    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """Retrieve relevant context chunks for the given query.

        Args:
            query: The user query string to search context for.

        Returns:
            A list of normalized `RetrievedChunk` instances.
        """
        raise NotImplementedError

    @abstractmethod
    def generate(self, query: str, chunks: list[RetrievedChunk]) -> str:
        """Generate a response given the query and retrieved context chunks.

        Args:
            query: The original user query.
            chunks: The list of `RetrievedChunk` objects returned by `retrieve()`.

        Returns:
            The generated response string.
        """
        raise NotImplementedError
