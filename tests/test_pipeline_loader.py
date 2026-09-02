"""Unit tests for dynamic pipeline loading."""

from pathlib import Path

import pytest

from ragdiag.pipeline import Pipeline, PipelineLoadError, load_pipeline


def test_load_pipeline_missing_file(tmp_path: Path) -> None:
    """Verify error when pipeline file does not exist."""
    missing = tmp_path / "non_existent.py"
    with pytest.raises(PipelineLoadError, match="does not exist"):
        load_pipeline(missing)


def test_load_pipeline_non_py_extension(tmp_path: Path) -> None:
    """Verify error when file is not a .py file."""
    txt_file = tmp_path / "pipeline.txt"
    txt_file.write_text("class MyPipeline: pass", encoding="utf-8")
    with pytest.raises(PipelineLoadError, match="must have a .py extension"):
        load_pipeline(txt_file)


def test_load_pipeline_syntax_error(tmp_path: Path) -> None:
    """Verify error when pipeline file contains invalid Python syntax."""
    bad_py = tmp_path / "broken.py"
    bad_py.write_text("def broken_func(:\n  pass", encoding="utf-8")
    with pytest.raises(PipelineLoadError, match="Failed to import pipeline module"):
        load_pipeline(bad_py)


def test_load_pipeline_missing_pipeline_variable(tmp_path: Path) -> None:
    """Verify error when pipeline file has no top-level 'pipeline' variable."""
    valid_py = tmp_path / "no_var.py"
    valid_py.write_text(
        "from ragdiag import Pipeline\n\nclass Custom(Pipeline):\n  pass\n",
        encoding="utf-8",
    )
    with pytest.raises(PipelineLoadError, match="must define a top-level 'pipeline' variable"):
        load_pipeline(valid_py)


def test_load_pipeline_wrong_variable_type(tmp_path: Path) -> None:
    """Verify error when top-level 'pipeline' variable is not an instance of Pipeline."""
    wrong_type_py = tmp_path / "wrong_type.py"
    wrong_type_py.write_text("pipeline = 'not a pipeline object'\n", encoding="utf-8")
    with pytest.raises(
        PipelineLoadError,
        match="to be an instance of Pipeline",
    ):
        load_pipeline(wrong_type_py)


def test_load_valid_pipeline(tmp_path: Path) -> None:
    """Verify successfully loading a valid Pipeline instance."""
    pipeline_py = tmp_path / "custom_pipeline.py"
    code = """
from ragdiag import Pipeline, RetrievedChunk

class CustomRAG(Pipeline):
    name = "custom_test"

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        return [RetrievedChunk(id="c1", text="Sample")]

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> str:
        return "Answer"

pipeline = CustomRAG()
"""
    pipeline_py.write_text(code, encoding="utf-8")
    loaded = load_pipeline(pipeline_py)
    assert isinstance(loaded, Pipeline)
    assert loaded.name == "custom_test"
    assert len(loaded.retrieve("q")) == 1


def test_load_examples_basic_pipeline() -> None:
    """Verify bundled examples/basic_pipeline.py loads successfully."""
    example_path = Path("examples/basic_pipeline.py")
    assert example_path.exists()

    loaded = load_pipeline(example_path)
    assert isinstance(loaded, Pipeline)
    assert loaded.name == "basic_pipeline"
    chunks = loaded.retrieve("What is refund settlement?")
    assert len(chunks) > 0
    answer = loaded.generate("What is refund?", chunks)
    assert "Based on available documentation" in answer
