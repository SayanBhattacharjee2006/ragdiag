# RAGDiag

> **RAG evaluation and root-cause diagnosis SDK/CLI for developers.** Built for the Razorpay Buildathon.

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

When Retrieval-Augmented Generation (RAG) systems fail or produce suboptimal outputs, developers often struggle to isolate the root cause:
- Did the retriever miss the relevant context chunks?
- Were the right chunks retrieved but buried among irrelevant noise?
- Did the generator hallucinate despite having the necessary context?
- Where is the latency bottleneck occurring?

**RAGDiag** is designed to solve this problem by executing evaluation datasets against custom RAG pipelines, computing precision/recall/faithfulness metrics, categorizing structured failure modes, and facilitating configuration comparisons.

> [!NOTE]
> **Status: Under Active Development.**
> This repository currently contains the core project foundation, domain models, and pipeline adapter contract. The evaluation engine, metrics, LLM judge, and diagnostic components will be delivered in upcoming phases.

---

## High-Level Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                    Developer RAG Pipeline                   │
│                                                             │
│   class MyPipeline(ragdiag.Pipeline):                       │
│       def retrieve(query) -> list[RetrievedChunk]           │
│       def generate(query, chunks) -> str                    │
└──────────────────────────────┬──────────────────────────────┘
                               │ Adapter Contract
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                           RAGDiag                           │
│                                                             │
│   1. Execution Harness: QuerySample -> Pipeline             │
│   2. Metrics Calculation: Retrieval, Generation, Latency    │
│   3. Root-Cause Diagnosis: Structured Failure Attribution   │
│   4. Output Models: EvaluationResult                        │
└─────────────────────────────────────────────────────────────┘
```

The design maintains strict decoupling between the developer's underlying RAG framework (LangChain, LlamaIndex, custom vector stores, etc.) and RAGDiag's evaluation harness through normalized domain abstractions (`RetrievedChunk`, `QuerySample`, `EvaluationResult`).

---

## Local Development Setup

RAGDiag uses [`uv`](https://docs.astral.sh/uv/) for fast, reliable virtual environment and dependency management.

### Prerequisites

- Python 3.12+
- `uv` installed (`pip install uv` or via your system package manager)

### Installation

1. Clone the repository and navigate into the project:
   ```bash
   cd ragdiag
   ```

2. Create a virtual environment and install dependencies in editable mode:
   ```bash
   uv venv
   uv pip install -e ".[dev]"
   ```

   Alternatively, using standard `uv run`:
   ```bash
   uv run ragdiag --help
   ```

---

## Running Tests and Linting

Run the test suite with `pytest`:

```bash
uv run pytest
```

Check code style and quality with `ruff`:

```bash
uv run ruff check .
uv run ruff format --check .
```

---

## CLI Usage (Preview)

Display CLI help:

```bash
uv run ragdiag --help
```

Display CLI version:

```bash
uv run ragdiag --version
```

Execute placeholder run command:

```bash
uv run ragdiag run --help
```

---

## Roadmap

- [x] **Phase 1: Project Foundation** (Packaging, domain models, pipeline interface, CLI skeleton, test suite)
- [ ] **Phase 2: Evaluation Metrics** (Retrieval precision/recall, context relevance, answer correctness)
- [ ] **Phase 3: Root-Cause Diagnosis Engine** (Automated classification of retrieval vs. generation failure modes)
- [ ] **Phase 4: Multi-Pipeline Comparison** (Side-by-side diagnostic reports and diffs)
