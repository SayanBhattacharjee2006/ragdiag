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

**RAGDiag** is the first developer tool designed to answer **"Why did this RAG query fail?"** by executing evaluation datasets against custom RAG pipelines, computing deterministic retrieval metrics, evaluating semantic quality via an isolated LLM judge, and attributing structured, evidence-based root-cause failure modes.

> [!NOTE]
> **Status: Under Active Development.**
> This repository contains the project foundation, domain models, pipeline adapter contract, validated golden dataset system, evaluation execution engine, deterministic retrieval metrics & latency analysis, provider-independent LLM Judge system, root-cause diagnosis engine, system-level Evaluation Report, and **Multi-Pipeline Comparison** (Phase 8).

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
│   4. LLM Judge: Semantic Answer Correctness & Groundedness  │
│   5. Diagnosis Engine: Deterministic Root-Cause Attribution │
│   6. Output Models: EvaluationResult, AggregateReport       │
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

## LLM Judge: Answer Correctness & Groundedness

RAGDiag includes an isolated, provider-independent LLM Judge system that evaluates generation quality on two strictly separated dimensions:

### Semantic Dimensions

1. **Answer Correctness (`answer_correct: bool`)**:
   - Compares the synthesized answer against the ground-truth `expected_answer`.
   - Checks semantic equivalence rather than lexical string matching.
   - Evaluates whether the user's question was factually and accurately answered.

2. **Groundedness (`grounded: bool`)**:
   - Evaluates whether claims made in the generated answer are strictly supported by the **retrieved context chunks**.
   - The retrieved chunks are the *sole* evidence source. The expected answer is *never* used as evidence for groundedness.
   - Identifies hallucinations where an answer sounds plausible or matches the expected answer but was fabricated without evidence in the retrieved context.

### Structured Output Enforcement

The judge uses model-level JSON schema enforcement (`client.beta.chat.completions.parse`) guaranteeing outputs conform to `JudgeResult`:
- `answer_correct: bool`
- `grounded: bool`
- `confidence: float` (0.0 to 1.0)
- `reason: str`

### Execution vs. Judge Failure Isolation

A judge failure (network timeout, rate limit, authentication error) does **not** fail the pipeline execution. The query remains `status="completed"`, deterministic retrieval metrics are preserved, and `judge_error` records the incident.

---

## Root-Cause Diagnosis Engine

The core differentiator of RAGDiag is its deterministic root-cause diagnosis engine. Instead of asking a second generic LLM to invent an explanation, RAGDiag applies an explainable decision hierarchy over the captured evidence.

> [!NOTE]
> RAGDiag identifies the most likely primary failure category from the concrete evidence captured during evaluation.

### Controlled Taxonomy (`FailureCategory`)

| Category | Description | Severity |
| :--- | :--- | :--- |
| **`PASS`** | Query completed and passed all retrieval, context, semantic, and latency checks. | `info` |
| **`WRONG_CHUNK_RETRIEVED`** | Retriever returned chunks, but none of the expected relevant chunks were present. | `major` |
| **`WRONG_CHUNK_RANK`** | Relevant chunks were retrieved, but the first relevant chunk appeared beyond the acceptable rank threshold (default: rank > 3). | `warning` |
| **`INSUFFICIENT_CONTEXT`** | Multiple context chunks were required, but only a partial subset was retrieved. | `warning` |
| **`RETRIEVED_BUT_NOT_GROUNDED`** | Relevant context was retrieved, but the LLM answer contained claims unsupported by the context (hallucination). | `major` |
| **`ANSWER_INCORRECT`** | Relevant context was retrieved and answer was grounded, but the answer contradicted or failed to answer the expected answer. | `major` |
| **`LATENCY_OUTLIER`** | All retrieval and quality checks passed, but retrieval latency exceeded the threshold (default: 1000ms). | `warning` |
| **`UNKNOWN`** | Pipeline execution failure or unclassifiable error. | `major` |

### Precedence Hierarchy

1. **Pipeline Execution Failure** (`status != 'completed'`) $\to$ `UNKNOWN`
2. **Total Retrieval Miss** (0 overlap) $\to$ `WRONG_CHUNK_RETRIEVED`
3. **Partial Context Retrieval** (subset retrieved) $\to$ `INSUFFICIENT_CONTEXT`
4. **All Context Retrieved but Ranked Too Low** (rank > 3) $\to$ `WRONG_CHUNK_RANK`
5. **Hallucination** (`grounded == False`) $\to$ `RETRIEVED_BUT_NOT_GROUNDED`
6. **Incorrect Answer** (`answer_correct == False`) $\to$ `ANSWER_INCORRECT`
7. **Latency Outlier** (latency > threshold) $\to$ `LATENCY_OUTLIER`
8. **Pass** (all checks passed) $\to$ `PASS`

*When a quality failure coincides with high latency, the quality failure is preserved as primary and the latency outlier is recorded as secondary evidence.*

---

## Evaluation Execution Engine

The `Evaluator` runs a pipeline against a `GoldenDataset`:
1. **Retrieval**: Executes `pipeline.retrieve(query)`, validates output, and records retrieval latency (`retrieval_ms`).
2. **Generation**: Executes `pipeline.generate(query, chunks)`, validates string output, and records generation latency (`generation_ms`).
3. **Metric Calculation**: Computes per-query retrieval metrics (`precision_at_5`, `recall_at_5`, `reciprocal_rank`).
4. **Semantic Evaluation (Optional)**: If a `Judge` is configured, evaluates semantic correctness and groundedness, recording `judge_ms`.
5. **Root-Cause Diagnosis**: Evaluates `DiagnosisEngine.diagnose(result)` to attach a structured `DiagnosisResult` to every query.
6. **Error Isolation**: Catches exceptions per query without aborting subsequent queries.

### Running Evaluation via Python API

#### 1. Deterministic Evaluation & Diagnosis (No API Key Required)

```python
from ragdiag import Evaluator, aggregate_metrics
from ragdiag.dataset import load_dataset
from ragdiag.pipeline import load_pipeline

pipeline = load_pipeline("examples/basic_pipeline.py")
dataset = load_dataset("examples/basic_dataset.json")

evaluator = Evaluator(k=5)
results = evaluator.evaluate(pipeline, dataset)

report = aggregate_metrics(results, k=5)
print(f"Precision@5: {report.mean_precision_at_k:.2f}")
print(f"Recall@5:    {report.mean_recall_at_k:.2f}")
print(f"MRR:         {report.mrr:.2f}")
print(f"Diagnosis:   {report.diagnosis_counts}")
```

#### 2. Evaluation with OpenAI Judge

```python
import os
from ragdiag import Evaluator, OpenAIJudge, aggregate_metrics
from ragdiag.dataset import load_dataset
from ragdiag.pipeline import load_pipeline

judge = OpenAIJudge(model="gpt-4o-mini")
evaluator = Evaluator(k=5, judge=judge)

results = evaluator.evaluate(
    pipeline=load_pipeline("examples/basic_pipeline.py"),
    dataset=load_dataset("examples/basic_dataset.json"),
)

report = aggregate_metrics(results, k=5)
print(f"Answer Correctness: {report.answer_correctness_rate:.2f}")
print(f"Groundedness:       {report.groundedness_rate:.2f}")
print(f"Diagnosis Counts:   {report.diagnosis_counts}")
```

## Diagnostic Intelligence and Evaluation Report

In addition to per-query diagnoses, RAGDiag aggregates results into a structured system-level **`EvaluationReport`** designed for terminal rendering, JSON export, CI gating, and dashboard consumption.

### System-Level Report Features

- **Stable Failure Taxonomy**: All 8 `FailureCategory` values are always present in `diagnosis_counts` (even with count 0) for consistent machine readability and diffing.
- **Query-Type Analysis**: Evaluates performance independently across `factual`, `reasoning`, and `multi-hop` queries (Precision@K, Recall@K, MRR, Groundedness, Correctness, and category breakdown).
- **Deterministic Top Failures**: Queries with failures are deterministically ranked by severity (`major` > `warning` > `info`), confidence, category priority, and query ID.
- **Deterministic Insights**: Factual, rule-based health insights generated without an LLM:
  - Identifies the weakest query type by recall gap ($\ge 0.10$).
  - Identifies dominant failure modes ($\ge 30\%$ of failures).
  - Flags grounding gaps across query types ($\ge 0.10$).
  - Highlights tail latency and outlier concerns.
- **JSON Serialization**: Complete, typed Pydantic serialization for downstream tooling.

### Generating Reports via Python API

```python
from ragdiag import Evaluator, build_report
from ragdiag.dataset import load_dataset
from ragdiag.pipeline import load_pipeline

pipeline = load_pipeline("examples/basic_pipeline.py")
dataset = load_dataset("examples/basic_dataset.json")

evaluator = Evaluator(k=5)
results = evaluator.evaluate(pipeline, dataset)

# Build system-level report
report = build_report(
    results,
    dataset_name=dataset.name,
    dataset_version=dataset.version,
    pipeline_name=pipeline.name,
    k=5,
)

print(f"Overall Recall@5: {report.retrieval.mean_recall_at_k:.2f}")
print(f"Insights: {report.overall_insights}")

# Export to JSON
json_data = report.model_dump_json(indent=2)
```

---

## Multi-Pipeline Comparison

RAG development is inherently iterative: developers frequently test new chunking strategies, embedding models, vector databases, or hybrid search configurations. **RAGDiag Multi-Pipeline Comparison** enables evidence-based A/B testing of exactly two pipeline configurations against the identical `GoldenDataset` and evaluation parameters.

### Why Multi-Pipeline Comparison Exists

Isolated benchmark numbers rarely reveal architectural trade-offs:
- Does hybrid search improve recall at the expense of doubling retrieval latency?
- Did a new retriever fix reasoning queries while regressing multi-hop queries?
- Which specific queries transitioned from failure to pass?

RAGDiag evaluates both pipelines through the same `Evaluator` and produces a structured `ComparisonReport` that captures metric deltas, failure count shifts, per-query-type breakdowns, individual query transitions, and deterministic winner/trade-off decisions.

### Metric Deltas ($\Delta = \text{Pipeline B} - \text{Pipeline A}$)

All numerical deltas follow a consistent direction: **Pipeline B minus Pipeline A**.
- **Quality Metrics** (Precision@K, Recall@K, MRR, Groundedness, Correctness): A positive delta indicates Pipeline B achieved higher quality.
- **Latency Metrics** (Mean, P95): A positive delta indicates Pipeline B is slower; a negative delta indicates Pipeline B is faster.
- **Judge Availability**: If no judge is configured, semantic metrics are explicitly marked as unavailable (`None`) rather than defaulting to zero.

### Diagnosis Failure Deltas

Compares failure counts across all 8 standard taxonomy categories (`B - A`):
- A negative delta indicates Pipeline B had **fewer failures** in that category (improvement).
- A positive delta indicates Pipeline B had **more failures** (regression).

### Deterministic Winner Strategy & Trade-Off Detection

RAGDiag avoids arbitrary weighted scoring formulas in favor of a transparent, deterministic decision rule:
1. **Quality Priority**: Evaluates Recall@K, MRR, Groundedness (if available), and Correctness (if available) against a tolerance threshold ($\epsilon = 0.02$). If Pipeline B improves primary quality beyond tolerance, it is the quality winner.
2. **Latency Analysis**: Compares mean retrieval latency against a tolerance threshold ($\epsilon_{\text{lat}} = 10.0\text{ ms}$).
3. **Trade-Off Classification**:
   - *Higher quality with higher latency*: Overall winner is declared with trade-off `"Higher quality <-> higher latency"`.
   - *Higher quality with equal/better latency*: Overall winner is declared with trade-off `"Higher quality with improved latency"`.
   - *Lower quality with lower latency*: Trade-off identified as `"Faster latency at the expense of lower quality"`.
   - *Roughly equal performance*: Declared as `"TIE"`.
4. **Descriptive Significance**: MVP results use empirical delta thresholds (e.g. $\ge 0.10$ for substantial improvement) without claiming statistical significance.

### Query-by-Query Outcome Transitions

Queries are matched by `query_id` across both pipeline evaluation runs to classify individual query transitions:
- **Improved**: Pipeline B reduced failure severity (e.g. `INSUFFICIENT_CONTEXT` $\to$ `PASS`) or achieved higher recall without adding new quality failures.
- **Regressed**: Pipeline B introduced a higher-severity failure or lower recall.
- **Unchanged**: Both pipelines achieved equivalent diagnostic outcomes.

### Python Comparison API

```python
from ragdiag import Comparator
from ragdiag.dataset import load_dataset
from ragdiag.pipeline import load_pipeline

pipeline_a = load_pipeline("examples/dense_pipeline.py")
pipeline_b = load_pipeline("examples/hybrid_pipeline.py")
dataset = load_dataset("examples/basic_dataset.json")

comparator = Comparator(k=5)
comparison = comparator.compare(pipeline_a, pipeline_b, dataset)

print(f"Overall Winner: {comparison.overall_winner}")
print(f"Trade-off:      {comparison.trade_off}")
print(f"Recall Delta:   {comparison.metric_deltas.recall_at_k:+.2f}")
print(f"Improved:       {comparison.queries_improved} queries")
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

### 2. Run Pipeline Evaluation & Diagnosis

Run evaluation and print the diagnostic terminal report:

```bash
uv run ragdiag run --pipeline examples/basic_pipeline.py --dataset examples/basic_dataset.json
```

Output:
```text
Running evaluation...

RAGDiag
--------------------------------------------
Pipeline: basic_pipeline
Dataset:  basic_dataset (v1.0)
Queries:  5

Evaluation complete.

OVERALL - Retrieval Metrics
--------------------------------------------
Precision@5:        0.90
Recall@5:           1.00
MRR:                 0.87

Retrieval Latency (illustrative)
------------------------
Mean:                0.01 ms
P50:                 0.00 ms
P95:                 0.01 ms
P99:                 0.01 ms

FAILURE ANALYSIS (Root Cause Analysis)
--------------------------------------------
PASS:                        5
WRONG_CHUNK_RETRIEVED:       0
WRONG_CHUNK_RANK:            0
INSUFFICIENT_CONTEXT:        0
RETRIEVED_BUT_NOT_GROUNDED:  0
ANSWER_INCORRECT:            0
LATENCY_OUTLIER:             0
UNKNOWN:                     0

QUERY TYPES
--------------------------------------------
Factual (2 queries)
  Recall@5: 1.00  MRR: 1.00

Reasoning (2 queries)
  Recall@5: 1.00  MRR: 1.00

Multi-hop (1 query)
  Recall@5: 1.00  MRR: 0.33

INSIGHTS
--------------------------------------------
* All evaluated queries passed the available retrieval and context checks. Semantic answer quality was not evaluated because no judge was configured.

Completed:  5
Failed:     0
Total time: 0.00s
```

### 3. Run Evaluation with JSON Output

Export the complete structured `EvaluationReport` to a JSON file:

```bash
uv run ragdiag run --pipeline examples/basic_pipeline.py --dataset examples/basic_dataset.json --output results.json
```

### 4. Run Evaluation with LLM Judge

To enable semantic answer correctness and groundedness evaluation:

```bash
export OPENAI_API_KEY="sk-..."
uv run ragdiag run --pipeline examples/basic_pipeline.py --dataset examples/basic_dataset.json --judge openai --model gpt-4o-mini
```

### 5. Compare Two Pipelines

Compare two pipelines side-by-side on the same golden dataset:

```bash
uv run ragdiag compare \
  --pipeline-a examples/dense_pipeline.py \
  --pipeline-b examples/hybrid_pipeline.py \
  --dataset examples/basic_dataset.json
```

Output:
```text
Running multi-pipeline comparison...

RAGDiag Comparison
==================================================
Dataset:    basic_dataset (v1.0)
Pipeline A: dense_pipeline
Pipeline B: hybrid_pipeline

OVERALL METRICS
--------------------------------------------------
Metric                   dense_pipeline hybrid_pipeline  Delta (B-A)
--------------------------------------------------------------
Precision@5                     0.87        0.90  +0.03
Recall@5                        0.80        1.00  +0.20
MRR                             0.87        0.87       0.00
Mean Retrieval                 5.4ms      25.2ms     +19.84ms
P95 Retrieval                  5.5ms      25.3ms     +19.78ms

FAILURE COUNTS
--------------------------------------------------
Category                     dense_pipeline hybrid_pipeline    Delta
----------------------------------------------------------
PASS                                 3         5 +2
WRONG_CHUNK_RETRIEVED                0         0      0
WRONG_CHUNK_RANK                     0         0      0
INSUFFICIENT_CONTEXT                 2         0 -2
RETRIEVED_BUT_NOT_GROUNDED           0         0      0
ANSWER_INCORRECT                     0         0      0
LATENCY_OUTLIER                      0         0      0
UNKNOWN                              0         0      0

QUERY TYPES
--------------------------------------------------
Factual
  Recall@5:  1.00 -> 1.00 (0.00)    MRR: 1.00 -> 1.00 (0.00)

Reasoning
  Recall@5:  0.75 -> 1.00 (+0.25)    MRR: 1.00 -> 1.00 (0.00)
  Failure count delta: -1

Multi-hop
  Recall@5:  0.50 -> 1.00 (+0.50)    MRR: 0.33 -> 0.33 (0.00)
  Failure count delta: -1

DECISION
--------------------------------------------------
Overall winner: hybrid_pipeline

Why:
hybrid_pipeline improves Recall@5 by 20 percentage points and MRR by 0 points, 
while increasing mean retrieval latency by 19.8 ms.

Trade-off:
Higher quality <-> higher latency

QUERY OUTCOMES
--------------------------------------------------
Improved:  2
Regressed: 0
Unchanged: 3

Total comparison time: 0.16s
```

### 6. Compare Two Pipelines with JSON Output

Export the complete structured `ComparisonReport` (including nested reports for Pipeline A and B) to JSON:

```bash
uv run ragdiag compare \
  --pipeline-a examples/dense_pipeline.py \
  --pipeline-b examples/hybrid_pipeline.py \
  --dataset examples/basic_dataset.json \
  --output comparison.json
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
- [x] **Phase 5: LLM Judge** (Semantic answer correctness and context groundedness using structured outputs, error isolation, provider decoupling)
- [x] **Phase 6: Root-Cause Diagnosis Engine** (Automated classification of retrieval vs. generation failure modes, evidence generation, query-type breakdowns)
- [x] **Phase 7: Diagnostic Intelligence and Final Evaluation Report** (System-level EvaluationReport, query-type analysis, top failures, rule-based insights, JSON export)
- [x] **Phase 8: Multi-Pipeline Comparison** (Side-by-side diagnostic reports and diffs)
