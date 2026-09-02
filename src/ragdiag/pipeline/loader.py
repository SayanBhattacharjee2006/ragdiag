"""Dynamic loader for RAGDiag pipeline adapter files."""

import importlib.util
from pathlib import Path

from ragdiag.pipeline.base import Pipeline
from ragdiag.pipeline.exceptions import PipelineLoadError


def load_pipeline(path: str | Path) -> Pipeline:
    """Load and validate a Pipeline instance from a Python file path.

    The target Python module must define a top-level `pipeline` variable
    that is an instance of `ragdiag.Pipeline`.

    Args:
        path: Path to the Python pipeline file (.py).

    Returns:
        The validated `Pipeline` instance.

    Raises:
        PipelineLoadError: If the file does not exist, is not a .py file, fails
            to import, lacks a `pipeline` variable, or `pipeline` is not a Pipeline instance.
    """
    file_path = Path(path).resolve()

    if not file_path.exists():
        raise PipelineLoadError(f"Pipeline file does not exist: {file_path}")

    if not file_path.is_file():
        raise PipelineLoadError(f"Pipeline path is not a regular file: {file_path}")

    if file_path.suffix.lower() != ".py":
        raise PipelineLoadError(
            f"Pipeline file must have a .py extension, got '{file_path.suffix}'."
        )

    module_name = f"ragdiag_pipeline_{file_path.stem}_{abs(hash(str(file_path)))}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise PipelineLoadError(f"Failed to create module specification for '{file_path}'.")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise PipelineLoadError(
            f"Failed to import pipeline module from '{file_path}': {type(exc).__name__}: {exc}"
        ) from exc

    if not hasattr(module, "pipeline"):
        raise PipelineLoadError(
            f"Pipeline module '{file_path}' must define a top-level 'pipeline' variable "
            f"(e.g., 'pipeline = MyPipeline()')."
        )

    pipeline_obj = module.pipeline
    if not isinstance(pipeline_obj, Pipeline):
        raise PipelineLoadError(
            f"Expected 'pipeline' in '{file_path}' to be an instance of Pipeline, "
            f"got {type(pipeline_obj).__name__}."
        )

    return pipeline_obj
