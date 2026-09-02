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
> This repository contains the project foundation, domain models, pipeline adapter contract, and the validated golden dataset system. The evaluation engine, metrics calculation, LLM judge, and diagnostic components will be delivered in subsequent phases.

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
│   1. Golden Dataset: Validated queries + ground truth       │
│   2. Execution Harness: QuerySample -> Pipeline (Upcoming)  │
│   3. Metrics Calculation: Retrieval & Generation (Upcoming) │
│   4. Root-Cause Diagnosis: Structured Attribution (Upcoming)│
│   5. Output Models: EvaluationResult                        │
└─────────────────────────────────────────────────────────────┘
```

The design maintains strict decoupling between the developer's underlying RAG framework (LangChain, LlamaIndex, custom vector stores, etc.) and RAGDiag's evaluation harness through normalized domain abstractions (`RetrievedChunk`, `QuerySample`, `GoldenDataset`, `EvaluationResult`).

---

## Golden Dataset System

RAGDiag requires a golden evaluation dataset containing ground truth answers, relevant chunk references, and categorized query types.

### Query Type Taxonomy

Evaluation samples are categorized using a controlled MVP taxonomy:
- **`factual`**: Single-fact retrieval and direct lookup queries.
- **`reasoning`**: Analytical queries requiring synthesis or inference from retrieved context.
- **`multi-hop`**: Complex queries requiring information joined across multiple disconnected chunks.

### JSON Schema

Datasets are structured as JSON files matching the `GoldenDataset` schema:

```json
{
  "name": "synapse_eval_v1",
  "version": "1.0",
  "samples": [
    {
      "id": "q001",
      "query": "What is the standard turnaround time for card refund settlements?",
      "expected_answer": "Standard card refund settlements are credited within 5 to 7 business days.",
      "relevant_chunk_ids": ["doc_refund_policy_01"],
      "query_type": "factual"
    },
    {
      "id": "q002",
      "query": "Why was the customer's recurring payment declined?",
      "expected_answer": "The auto-debit was declined because the mandate expired.",
      "relevant_chunk_ids": ["doc_mandates_05"],
      "query_type": "reasoning"
    }
  ]
}
```

### Dataset Validation Rules

RAGDiag enforces strict validation to catch dataset mistakes early:
- Unique sample IDs across the entire dataset.
- At least one sample per dataset.
- Non-empty `id`, `query`, and `expected_answer`.
- Non-empty `relevant_chunk_ids` with at least one valid chunk ID and no duplicates within the same sample.
- `query_type` must be one of `factual`, `reasoning`, or `multi-hop`.

### Validating a Dataset via CLI

Validate any dataset file before running evaluation:

```bash
uv run ragdiag validate --dataset examples/basic_dataset.json
```

Output:
```text
Dataset: basic_dataset
Version: 1.0
Samples: 5
Query types:
  factual: 2
  reasoning: 2
  multi-hop: 1

Dataset is valid.
```

### Loading a Dataset in Python

```python
from ragdiag.dataset import load_dataset

dataset = load_dataset("examples/basic_dataset.json")
print(f"Loaded {len(dataset.samples)} samples from {dataset.name} (v{dataset.version})")
```

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

## CLI Commands

```bash
# Show CLI options
uv run ragdiag --help

# Show version
uv run ragdiag --version

# Validate a golden evaluation dataset
uv run ragdiag validate --dataset examples/basic_dataset.json

# Preview pipeline evaluation run (stub)
uv run ragdiag run --help
```

---

## Roadmap

- [x] **Phase 1: Project Foundation** (Packaging, domain models, pipeline interface, CLI skeleton, test suite)
- [x] **Phase 2: Golden Dataset System** (JSON schema, QueryType taxonomy, loader, validator, CLI validate command)
- [ ] **Phase 3: Evaluation Metrics** (Retrieval precision/recall, context relevance, answer correctness)
- [ ] **Phase 4: Root-Cause Diagnosis Engine** (Automated classification of retrieval vs. generation failure modes)
- [ ] **Phase 5: Multi-Pipeline Comparison** (Side-by-side diagnostic reports and diffs)
