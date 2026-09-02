"""Tests for the base Pipeline interface and subclass contracts."""

import pytest

from ragdiag import Pipeline, RetrievedChunk


def test_pipeline_cannot_be_instantiated_directly() -> None:
    """Verify that Pipeline is an abstract base class that prevents direct instantiation."""
    with pytest.raises(TypeError):
        Pipeline()  # type: ignore[abstract]


def test_pipeline_subclass_missing_methods_fails() -> None:
    """Verify that subclasses must implement both retrieve and generate."""

    class IncompletePipeline(Pipeline):
        def retrieve(self, query: str) -> list[RetrievedChunk]:
            return []

    with pytest.raises(TypeError):
        IncompletePipeline()  # type: ignore[abstract]


def test_concrete_pipeline_implementation() -> None:
    """Verify that a valid Pipeline subclass satisfies the contract."""

    class DummyRAGPipeline(Pipeline):
        name = "dummy_rag"

        def retrieve(self, query: str) -> list[RetrievedChunk]:
            return [
                RetrievedChunk(id="c1", text=f"Context for {query}", score=0.9),
            ]

        def generate(self, query: str, chunks: list[RetrievedChunk]) -> str:
            context = " ".join(c.text for c in chunks)
            return f"Answer for '{query}' based on: {context}"

    pipeline = DummyRAGPipeline()
    assert pipeline.name == "dummy_rag"

    # Test retrieval contract
    chunks = pipeline.retrieve("What is payment gateway?")
    assert len(chunks) == 1
    assert isinstance(chunks[0], RetrievedChunk)
    assert chunks[0].id == "c1"

    # Test generation contract
    answer = pipeline.generate("What is payment gateway?", chunks)
    assert "Context for What is payment gateway?" in answer
    assert isinstance(answer, str)


def test_pipeline_custom_name_override() -> None:
    """Verify pipeline initialization with custom name."""

    class SimplePipeline(Pipeline):
        def retrieve(self, query: str) -> list[RetrievedChunk]:
            return []

        def generate(self, query: str, chunks: list[RetrievedChunk]) -> str:
            return "ok"

    pipeline_default = SimplePipeline()
    assert pipeline_default.name == "default_pipeline"

    pipeline_custom = SimplePipeline(name="custom_v2")
    assert pipeline_custom.name == "custom_v2"
