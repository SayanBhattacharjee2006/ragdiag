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

**RAGDiag** is designed to solve this problem by executing evaluation datasets against custom RAG pipelines, computing precision/recall/latency metrics, categorizing structured failure modes, and facilitating configuration comparisons.

> [!NOTE]
> **Status: Under Active Development.**
> This repository contains the project foundation, domain models, pipeline adapter contract, validated golden dataset system, evaluation execution engine, and deterministic retrieval metrics & latency analysis (Phase 4). LLM-as-a-judge (groundedness, answer correctness) and diagnostic components will be delivered in subsequent phases.

---

## High-Level Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                    Developer RAG Pipeline                   │
│                                                             │
│   class MyPipeline(ragdiag.Pipeline):                       │
│       def retrieve(query) -> list[RetrievedChunk]           │
│       def generate(query, chunks) -> str                    │
│   pipeline = MyPipeline()                                   │
└──────────────────────────────┬──────────────────────────────┘
                                │ Adapter Contract
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                           RAGDiag                           │
│                                                             │
│   1. Golden Dataset: Validated queries + ground truth       │
│   2. Execution Engine: Evaluator orchestrates queries       │
│   3. Metrics Layer: Precision@K, Recall@K, MRR, Latency     │
│   4. Root-Cause Diagnosis: Structured Attribution (Upcoming)│
│   5. Output Models: EvaluationResult, AggregateReport       │
└─────────────────────────────────────────────────────────────┘
```

The design maintains strict decoupling between the developer's underlying RAG framework (LangChain, LlamaIndex, custom vector stores, etc.) and RAGDiag's evaluation harness through normalized domain abstractions (`RetrievedChunk`, `QuerySample`, `GoldenDataset`, `EvaluationResult`).

---

## Pipeline Adapter Interface

Developers adapt their RAG application by implementing the `ragdiag.Pipeline` contract in a standalone Python file.

### Adapter File Convention

The adapter file must define a top-level instance named `pipeline`:

```python
# my_pipeline.py
from ragdiag import Pipeline, RetrievedChunk


class MyCustomPipeline(Pipeline):
    name = "my_custom_rag"

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        # Connect to your vector DB, hybrid search, or custom retriever
        return [
            RetrievedChunk(
                id="doc_101",
                text="Refunds are processed within 5-7 business days.",
                score=0.94,
            )
        ]

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> str:
        # Pass retrieved context to your LLM
        return "Your refund will take 5-7 business days."


# Expose top-level adapter instance
pipeline = MyCustomPipeline()
```

### In-Memory Example Pipeline

An in-memory, deterministic example pipeline requiring no API keys or external services is provided at `examples/basic_pipeline.py`.

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

---

## Retrieval Metrics & Latency Analysis

RAGDiag calculates deterministic, framework-agnostic retrieval quality metrics and latency benchmarks on the raw evidence captured during evaluation.

### Core Metrics

- **Precision@K (`precision_at_k`)**:
  Measures the fraction of retrieved chunks in the top-$K$ results that are relevant to the query:
  $$\text{Precision@}K = \frac{|\text{Relevant Chunks in Top-}K|}{\min(K, |\text{Retrieved Chunks}|)}$$
  *Note: When a retriever returns fewer than $K$ results, the denominator uses the count of returned chunks to avoid artificially penalizing configurations with smaller cutoffs.*

- **Recall@K (`recall_at_k`)**:
  Measures the proportion of all ground-truth relevant chunks that were successfully retrieved in the top-$K$ results:
  $$\text{Recall@}K = \frac{|\text{Relevant Chunks in Top-}K|}{|\text{Total Ground-Truth Relevant Chunks}|}$$

- **Mean Reciprocal Rank (`mrr`)**:
  Evaluates ranking quality across queries. For each query, the Reciprocal Rank (RR) is $\frac{1}{\text{rank}}$ of the first relevant retrieved chunk (or $0.0$ if no relevant chunk is retrieved). MRR is the mean of RR scores across queries.

- **Retrieval Latency Statistics**:
  Aggregates retrieval execution times in milliseconds using standard linear interpolation percentiles (NIST / NumPy Method 7) without external dependencies:
  - **Mean**: Arithmetic average retrieval latency.
  - **P50**: Median retrieval latency.
  - **P95**: 95th percentile retrieval latency (tail latency).
  - **P99**: 99th percentile retrieval latency (extreme outliers).

*Failed queries are excluded from quality and latency denominators to prevent infrastructure errors from distorting retrieval quality measurements.*

---

## Evaluation Execution Engine

The `Evaluator` runs a pipeline against a `GoldenDataset`:
1. **Retrieval**: Executes `pipeline.retrieve(query)`, validates `list[RetrievedChunk]` output, and records retrieval latency (`retrieval_ms`).
2. **Generation**: If retrieval succeeds, executes `pipeline.generate(query, chunks)`, validates string output, and records generation latency (`generation_ms`).
3. **Metric Calculation**: Computes per-query retrieval metrics (`precision_at_5`, `recall_at_5`, `reciprocal_rank`).
4. **Error Isolation**: Catches exceptions per query without aborting subsequent queries.
5. **Evidence Capture**: Retains retrieved chunks, answers, status (`completed` / `failed`), errors, and latency breakdowns in `EvaluationResult`.

### Running Evaluation via Python API

```python
from ragdiag import Evaluator, aggregate_metrics
from ragdiag.dataset import load_dataset
from ragdiag.pipeline import load_pipeline

pipeline = load_pipeline("examples/basic_pipeline.py")
dataset = load_dataset("examples/basic_dataset.json")

evaluator = Evaluator(k=5)
results = evaluator.evaluate(pipeline, dataset)

# Aggregate report
report = aggregate_metrics(results, k=5)
print(f"Precision@5: {report.mean_precision_at_k:.2f}")
print(f"Recall@5:    {report.mean_recall_at_k:.2f}")
print(f"MRR:         {report.mrr:.2f}")
print(f"P50 Latency: {report.retrieval_latency.p50_ms:.2f} ms")
```

---

## CLI Commands

### 1. Validate a Golden Dataset

```bash
uv run ragdiag validate --dataset examples/basic_dataset.json
```

Output:
```text
+------------------------ Dataset Validation Summary -------------------------+
| Dataset: basic_dataset                                                      |
| Version: 1.0                                                                |
| Samples: 5                                                                  |
| Query types:                                                                |
|   factual: 2                                                                |
|   reasoning: 2                                                              |
|   multi-hop: 1                                                              |
|                                                                             |
| Dataset is valid.                                                           |
+-----------------------------------------------------------------------------+
```

### 2. Run Pipeline Evaluation

```bash
uv run ragdiag run --pipeline examples/basic_pipeline.py --dataset examples/basic_dataset.json
```

Output:
```text
RAGDiag
------------------------
Pipeline: basic_pipeline
Dataset:  basic_dataset
Queries:  5

Running evaluation...

Evaluation complete.

Retrieval Metrics
------------------------
Precision@5:  0.90
Recall@5:     1.00
MRR:          0.87

Retrieval Latency
------------------------
Mean:   0.01 ms
P50:    0.00 ms
P95:    0.01 ms
P99:    0.01 ms

Completed:  5
Failed:     0
Total time: 0.00s
```

---

## Local Development Setup

RAGDiag uses [`uv`](https://docs.astral.sh/uv/) for fast, reliable virtual environment and dependency management.

### Prerequisites

- Python 3.12+
- `uv` installed (`pip install uv` or via your system package manager)

### Installation

```bash
git clone <repo-url>
cd ragdiag
uv venv
uv pip install -e ".[dev]"
```

### Running Tests and Linting

```bash
# Run pytest suite
uv run pytest

# Check style and formatting
uv run ruff check .
uv run ruff format --check .
```

---

## Roadmap

- [x] **Phase 1: Project Foundation** (Packaging, domain models, pipeline interface, CLI skeleton, test suite)
- [x] **Phase 2: Golden Dataset System** (JSON schema, QueryType taxonomy, loader, validator, CLI validate command)
- [x] **Phase 3: Evaluation Execution Engine** (Pipeline dynamic loader, Evaluator lifecycle, latency tracking, error isolation, CLI run command)
- [x] **Phase 4: Retrieval Metrics & Latency Analysis** (Precision@K, Recall@K, Reciprocal Rank, MRR, percentile latency statistics, aggregate report)
- [ ] **Phase 5: LLM Judge** (Groundedness and answer correctness scoring)
- [ ] **Phase 6: Root-Cause Diagnosis Engine** (Automated classification of retrieval vs. generation failure modes)
- [ ] **Phase 7: Multi-Pipeline Comparison** (Side-by-side diagnostic reports and diffs)
