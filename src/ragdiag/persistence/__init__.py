"""Automatic persistence subsystem for RAGDiag."""

from ragdiag.persistence.manager import PersistenceResult, ResultPersistence
from ragdiag.persistence.markdown import (
    render_comparison_markdown,
    render_diagnosis_markdown,
    render_evaluation_markdown,
)

__all__ = [
    "PersistenceResult",
    "ResultPersistence",
    "render_comparison_markdown",
    "render_diagnosis_markdown",
    "render_evaluation_markdown",
]
