# RAGDiag

> **Developer tool for evaluating RAG pipelines and identifying evidence-backed primary failure categories.** Built for the Razorpay Buildathon.

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Problem

Traditional RAG evaluation frameworks calculate scalar scores (e.g. an aggregate score of 0.72) but leave developers guessing when outputs degrade:
- Did the vector retriever miss the relevant context chunks entirely?
- Were the right chunks retrieved but buried beneath distracting irrelevant chunks?
- Was only partial context retrieved for multi-part questions?
- Did the LLM hallucinate unsupported claims despite having the context?
- Or did retrieval latency spike beyond acceptable production thresholds?

When engineers test an architectural change—such as switching from dense semantic search to hybrid search—isolated numbers cannot explain whether higher recall justifies the additional latency or which specific queries improved or regressed.

---

## What RAGDiag Does

RAGDiag inspects the raw evidence captured during execution across retrieval and generation to classify why individual queries fail, generate system-level diagnostic intelligence, and perform evidence-based comparisons between pipeline architectures:

```text
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│   RAG Pipeline  │ ───>  │   Evaluation    │ ───>  │  Root-Cause     │ ───>  │ Multi-Pipeline  │
│   (Adapter)     │       │   Harness       │       │  Diagnosis      │       │ Comparison      │
└─────────────────┘       └─────────────────┘       └─────────────────┘       └─────────────────┘
  Custom retriever          Precision@K, MRR,         8-Category Decision       Directional deltas,
  and generator             Latency, LLM Judge        Precedence Hierarchy      Transitions, Winners
```

---

## Key Capabilities

- **Framework-Agnostic Adapter**: Wrap any RAG stack (custom vector stores, LangChain, LlamaIndex, BM25) in standard Python methods.
- **Golden Dataset Schema**: Validated ground-truth dataset format with categorized query types (`factual`, `reasoning`, `multi-hop`).
- **Deterministic Retrieval Metrics**: Exact calculations for Precision@K, Recall@K, Reciprocal Rank, and MRR.
- **Latency Distribution Analysis**: Non-parametric percentile statistics (Mean, P50, P95, P99, Min, Max) for retrieval and generation stages.
- **Isolated LLM Judge**: Evaluates answer correctness against ground truth and context groundedness against retrieved chunks via schema-enforced structured outputs.
- **Evidence-Based Root-Cause Diagnosis**: Classifies failures into a deterministic 8-category taxonomy without asking an unconstrained LLM to guess.
- **Diagnostic System Reports**: Aggregates query-type breakdowns, deterministically ranked top failures, and rule-based insights.
- **Multi-Pipeline A/B Comparison**: Compares two pipeline configurations side-by-side on the same dataset, calculating metric deltas, failure shifts, query transitions (`improved`, `regressed`, `unchanged`), and deterministic winner/trade-off decisions.
- **Automated Regression Analysis**: Evaluates whether a candidate pipeline degraded compared to a baseline, identifying metric regressions beyond tolerance, query regressions, critical diagnosis transitions (e.g., `PASS` $\to$ `INSUFFICIENT_CONTEXT`), and explainable regression summaries.
- **Overall Health Profile**: Computes a deterministic 0–100 health score, categorical grade (`Excellent`, `Good`, `Fair`, `Poor`, `Critical`), operational status, empirical strengths, weaknesses, and actionable deduplicated recommendations.
- **Evaluation Confidence**: Quantifies evidence reliability and completeness (0–100 score, `High` to `Very Low` levels) based on query coverage, saturating sample size curves, and judge evidence availability.
- **Diagnose CLI (`ragdiag diagnose`)**: Instantly inspects existing reports to explain pipeline failures, affected queries, actions, health profile, and evaluation confidence without re-executing pipelines or LLMs.
- **Automatic Result Persistence**: Automatically saves machine-readable JSON and human-readable Markdown to `.ragdiag/` (`evaluations/`, `comparisons/`, `diagnoses/`) with safe atomic writes.
- **CLI & Typed JSON Exports**: Formatted Rich terminal reports and complete Pydantic JSON serialization.

---

## Quickstart

### Installation

```bash
pip install ragdiag
```

Or when developing with [`uv`](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/SayanBhattacharjee2006/ragdiag.git
cd ragdiag
uv venv
uv pip install -e ".[dev]"
```

---

## Pipeline Adapter Interface

Developers adapt their existing RAG pipeline by subclassing `ragdiag.Pipeline` and exposing a top-level instance named `pipeline`:

```python
# my_pipeline.py
from ragdiag import Pipeline, RetrievedChunk


class MyCustomPipeline(Pipeline):
    name = "payment_faq_rag"

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        # Connect to your vector DB, dense index, or hybrid retriever
        return [
            RetrievedChunk(
                id="doc_refund_policy_01",
                text="Standard card refunds settle within 5 to 7 business days.",
                score=0.92,
            )
        ]

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> str:
        # Pass context to your LLM generator
        return "Card refund settlements typically take 5 to 7 business days."


pipeline = MyCustomPipeline()
```

---

## Golden Dataset System

Datasets are JSON files matching the `GoldenDataset` schema with ground truth and categorized query types:

```json
{
  "name": "payment_gateway_eval",
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
      "query": "Why was the customer's recurring auto-debit declined?",
      "expected_answer": "The auto-debit was declined because the e-mandate registration expired.",
      "relevant_chunk_ids": ["doc_subscriptions_03", "doc_mandates_05"],
      "query_type": "reasoning"
    },
    {
      "id": "q003",
      "query": "What is the effective net settlement fee considering base interchange and GST?",
      "expected_answer": "The effective fee is 2.5% base fee plus 18% GST on the fee, totaling 2.95%.",
      "relevant_chunk_ids": ["doc_pricing_tier_01", "doc_tax_regulations_03"],
      "query_type": "multi-hop"
    }
  ]
}
```

### Validate a Dataset

```bash
ragdiag validate --dataset examples/demo_dataset.json
```

---

## Run Evaluation

Evaluate a single pipeline configuration and print the diagnostic terminal report:

```bash
ragdiag run --pipeline examples/basic_pipeline.py --dataset examples/basic_dataset.json
```

### Enable Semantic LLM Judge

```bash
export OPENAI_API_KEY="sk-..."
ragdiag run \
  --pipeline examples/basic_pipeline.py \
  --dataset examples/basic_dataset.json \
  --judge openai \
  --model gpt-4o-mini
```

### Export JSON Report

```bash
ragdiag run \
  --pipeline examples/basic_pipeline.py \
  --dataset examples/basic_dataset.json \
  --output evaluation_report.json
```

---

## Compare Two Pipelines

Compare a baseline pipeline (Pipeline A) against a candidate architecture (Pipeline B) on the same dataset:

```bash
ragdiag compare \
  --pipeline-a examples/dense_pipeline.py \
  --pipeline-b examples/hybrid_pipeline.py \
  --dataset examples/demo_dataset.json
```

### Real Deterministic Comparison Output

```text
Running multi-pipeline comparison...

RAGDiag Comparison
==================================================
Dataset:    payment_gateway_demo_eval (v1.0)
Pipeline A: dense_pipeline
Pipeline B: hybrid_pipeline

OVERALL METRICS
--------------------------------------------------
Metric                   dense_pipeline hybrid_pipeline  Delta (B-A)
--------------------------------------------------------------
Precision@5                     0.87        0.90  +0.03
Recall@5                        0.80        1.00  +0.20
MRR                             0.87        0.87       0.00
Mean Retrieval                 5.2ms      25.5ms     +20.25ms
P95 Retrieval                  5.3ms      25.7ms     +20.35ms

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
while increasing mean retrieval latency by 20.3 ms.

Trade-off:
Higher quality <-> higher latency

QUERY OUTCOMES
--------------------------------------------------
Improved:  2
Regressed: 0
Unchanged: 3

REGRESSION ANALYSIS
--------------------------------------------------
Overall regression: NO
No meaningful regressions detected.

Total comparison time: 0.16s
```

Export comparison report to JSON:

```bash
ragdiag compare \
  --pipeline-a examples/dense_pipeline.py \
  --pipeline-b examples/hybrid_pipeline.py \
  --dataset examples/demo_dataset.json \
  --output comparison.json
```

---

## Diagnose Evaluation Reports (`ragdiag diagnose`)

Inspect an existing serialized evaluation report without rerunning the pipeline or invoking LLM inference:

```bash
ragdiag diagnose .ragdiag/evaluations/latest.json
```

Or inspect any previously saved `EvaluationReport` JSON file:

```bash
ragdiag diagnose evaluation_report.json
```

### Purpose
Answers the central developer question:
**"Why is my RAG pipeline failing, and what should I do about it?"**

- **Zero Re-computation**: Operates offline on existing reports without requiring pipeline execution or LLM tokens.
- **Top Failure Modes**: Identifies primary bottlenecks ranked by severity and frequency.
- **Actionable Recommendations**: Directly displays recommendations from the single source of truth Failure $\to$ Action Mapping.
- **Important Query Details**: Shows affected query IDs, diagnostic verdicts, evidence, and actions.
- **Health & Confidence**: Presents the overall health profile (Score, Grade, Status) and evidence reliability confidence score.

Sample terminal output:

```text
RAG DIAGNOSIS
============================================

Overall:
  42 queries evaluated
  31 passed
  11 failed

TOP FAILURE MODES
--------------------------------------------
1. WRONG_CHUNK_RETRIEVED
   Queries: 6
   Action: Review the retrieval strategy and query formulation; the pipeline retrieved irrelevant context.

2. ANSWER_INCORRECT
   Queries: 3
   Action: Review the generation prompt, model behavior, and context usage for answer correctness.

3. INSUFFICIENT_CONTEXT
   Queries: 2
   Action: Increase retrieval depth or improve retrieval coverage so all required context is retrieved.

IMPORTANT QUERIES
--------------------------------------------
q12
  Diagnosis: WRONG_CHUNK_RETRIEVED
  Reason:    Complete retrieval miss.
  Evidence:  None of the expected chunks were retrieved.
  Action:    Review the retrieval strategy and query formulation; the pipeline retrieved irrelevant context.

HEALTH
--------------------------------------------
Score: 68.0/100
Grade: Fair (Status: Degraded)

EVALUATION CONFIDENCE
--------------------------------------------
Score: 94.0/100
Level: High
```

---

## Automatic Result Persistence (`.ragdiag/`)

RAGDiag automatically persists generated results upon every successful run—**no manual save command is required**.

### Persistence Directory Layout

Results are saved to a project-local `.ragdiag/` directory relative to the current working directory:

```text
.ragdiag/
├── evaluations/
│   ├── latest.json           # Machine-readable Pydantic EvaluationReport
│   ├── latest.md             # Human-readable GitHub-flavored Markdown
│   └── history/
│       ├── 20260904_120000_123456.json
│       └── 20260904_120000_123456.md
├── comparisons/
│   ├── latest.json           # Machine-readable Pydantic ComparisonReport
│   ├── latest.md             # Human-readable comparison summary
│   └── history/
│       ├── 20260904_120500_654321.json
│       └── 20260904_120500_654321.md
└── diagnoses/
    ├── latest.json           # Machine-readable evaluated report
    ├── latest.md             # Clean diagnostic summary Markdown
    └── history/
        ├── 20260904_121000_789012.json
        └── 20260904_121000_789012.md
```

### Safety & Resilience
- **Atomic File Writes**: Files are written to temporary files and atomically renamed, avoiding partial or corrupted state.
- **Fail-Safe Warnings**: If disk persistence fails (e.g. permission or disk space issues), the command displays a non-fatal warning without aborting the evaluation.
- **Isolated Namespaces**: Evaluations, comparisons, and diagnoses reside in separate directories and never overwrite each other.

### Inspecting and Managing Artifacts
- **Inspect**: Open `latest.md` in any Markdown viewer or preview `latest.json` with standard JSON tools.
- **Clean**: Remove the `.ragdiag` folder at any time to clear local run history:
  ```bash
  rm -rf .ragdiag
  ```
- **Git Ignore**: Add `.ragdiag/` to your project's `.gitignore` to prevent committing generated local test runs:
  ```gitignore
  # RAGDiag generated evaluation artifacts
  .ragdiag/
  ```

---

## Regression Analysis

RAGDiag features a dedicated regression analysis engine that answers:
**"Did the current RAG system get worse compared with the baseline, and where?"**

It evaluates both system-wide and query-level behavior against configurable tolerances ($\epsilon_{\text{quality}} = 0.02$, $\epsilon_{\text{latency}} = 10.0\text{ ms}$):

- **Overall Metric Regressions**:
  - Higher-is-better quality metrics: Recall@K, Precision@K, MRR, Groundedness, and Answer Correctness. Decreases exceeding tolerance are flagged.
  - Lower-is-better performance metrics: Mean retrieval latency increases exceeding tolerance are flagged as performance regressions.
- **Meaningful Thresholds**: Minor numerical noise within tolerance is filtered out and never reported as a regression.
- **Query-Level Regressions**: Identifies exact queries whose outcomes deteriorated, preserving Comparator classification semantics.
- **Diagnosis Transitions**: Tracks failure category shifts (e.g., `PASS` $\to$ `INSUFFICIENT_CONTEXT`, `PASS` $\to$ `ANSWER_INCORRECT`).
- **Explainable Decision Logic**: Overall regression is `YES` if more queries regressed than improved or if quality degraded without offsetting gains. Latency trade-offs on an improved candidate pipeline are tracked without falsely marking the entire system as regressed.

In Python:

```python
# Access regression analysis programmatically
ra = comparison.regression_analysis
print(f"Overall Regression: {ra.overall_regression}")
print(f"Summary:            {ra.summary}")
print(f"Regressed Queries:  {ra.regressed_query_count}")

for mr in ra.metric_regressions:
    print(
        f"  {mr.metric_name}: {mr.delta:+.4f} (baseline={mr.baseline:.4f}, current={mr.current:.4f})"
    )

for imp in ra.important_regressions:
    print(f"  Important: {imp}")
```

---

## Root-Cause Failure Taxonomy

RAGDiag classifies every completed query into an explainable 8-category hierarchy and provides a deterministic, actionable recommendation:

| Category | Severity | Description | Action Recommendation |
| :--- | :--- | :--- | :--- |
| **`PASS`** | `info` | Query succeeded across retrieval, context completeness, semantic, and latency checks. | No action required. |
| **`WRONG_CHUNK_RETRIEVED`** | `major` | Complete retrieval miss; none of the required context chunks were retrieved in top-$K$. | Review the retrieval strategy and query formulation; the pipeline retrieved irrelevant context. |
| **`INSUFFICIENT_CONTEXT`** | `warning` | Partial context retrieval; query required multiple chunks but only a subset was retrieved. | Increase retrieval depth or improve retrieval coverage so all required context is retrieved. |
| **`WRONG_CHUNK_RANK`** | `warning` | All required context was retrieved, but the first relevant chunk ranked lower than threshold (rank > 3). | Improve ranking or reranking so relevant context appears earlier. |
| **`RETRIEVED_BUT_NOT_GROUNDED`** | `major` | Hallucination; context was retrieved, but the LLM made claims unsupported by the chunks. | Improve answer grounding so the generated response stays supported by the retrieved context. |
| **`ANSWER_INCORRECT`** | `major` | Context was retrieved and answer was grounded, but contradicted or failed the ground truth. | Review the generation prompt, model behavior, and context usage for answer correctness. |
| **`LATENCY_OUTLIER`** | `warning` | Quality passed, but retrieval latency exceeded threshold (default: 1000ms). | Investigate slow retrieval or generation paths and optimize the latency bottleneck. |
| **`UNKNOWN`** | `major` | Pipeline crash or unclassifiable execution exception. | Inspect the pipeline execution error and underlying integration. |

### Decision Precedence

1. **Pipeline Execution Failure** (`status != 'completed'`) $\to$ `UNKNOWN`
2. **Total Retrieval Miss** (0 overlap with expected chunks) $\to$ `WRONG_CHUNK_RETRIEVED`
3. **Partial Context Retrieval** (subset retrieved) $\to$ `INSUFFICIENT_CONTEXT`
4. **All Context Retrieved but Ranked Late** (rank > 3) $\to$ `WRONG_CHUNK_RANK`
5. **Hallucination** (`grounded == False`) $\to$ `RETRIEVED_BUT_NOT_GROUNDED`
6. **Incorrect Answer** (`answer_correct == False`) $\to$ `ANSWER_INCORRECT`
7. **Latency Outlier** (`retrieval_ms > threshold`) $\to$ `LATENCY_OUTLIER`
8. **Pass** (all checks passed) $\to$ `PASS`

---

## Health Profile

RAGDiag includes an automated, evidence-based **Health Profile** engine answering:
**"How healthy is this RAG system, and what are its biggest weaknesses?"**

### Health Score & Bands
The health score is a deterministic calculation bounded between `0.0` and `100.0`:
- **Quality Signals**: Weighted across Recall@K, Precision@K, and MRR. When LLM judge evaluation is available, Answer Correctness and Groundedness are incorporated into the score.
- **Performance**: Retrieval latency is evaluated using piecewise linear thresholds ($\le 100\text{ms} \to 1.0$, $500\text{ms} \to 0.75$, $1500\text{ms} \to 0.25$, $\ge 3000\text{ms} \to 0.0$).
- **Controlled Penalties**: Execution crashes and severe latency outliers apply controlled, proportional penalties without double-penalizing standard retrieval misses.

| Score Range | Grade | Operational Status | Interpretation |
| :--- | :--- | :--- | :--- |
| **90.0 – 100.0** | `Excellent` | `Healthy` | High retrieval precision, strong ranking, grounded answers, fast latency. |
| **75.0 – 89.9** | `Good` | `Healthy` | Solid overall quality with minor non-critical areas for tuning. |
| **60.0 – 74.9** | `Fair` | `Degraded` | Measurable bottlenecks in recall, context coverage, or latency. |
| **40.0 – 59.9** | `Poor` | `Unhealthy` | Significant retrieval misses, ungrounded generation, or frequent timeouts. |
| **0.0 – 39.9** | `Critical` | `Critical` | Severe pipeline crashes, near-zero recall, or pervasive failures. |

### Empirical Strengths & Weaknesses
- **Strengths**: Extracted strictly from measured data (e.g., Recall $\ge 0.85$, MRR $\ge 0.80$, Latency $\le 100\text{ms}$). Semantic strengths are only reported when judge evaluation was conducted.
- **Weaknesses**: Identifies primary bottlenecks (low recall, poor precision, ungrounded answers, high latency, diagnostic misses).
- **Actionable Recommendations**: Reuses the Failure $\to$ Action Mapping, generating deduplicated, prioritized actions.

In Python:

```python
from ragdiag import Evaluator, load_dataset, load_pipeline

pipeline = load_pipeline("examples/basic_pipeline.py")
dataset = load_dataset("examples/basic_dataset.json")

evaluator = Evaluator()
report = evaluator.evaluate(pipeline, dataset)

# Access Health Profile
hp = report.health_profile
print(f"Health Score: {hp.score}/100 ({hp.grade}, {hp.status})")

print("\nStrengths:")
for s in hp.strengths:
    print(f"  ✓ {s}")

print("\nWeaknesses:")
for w in hp.weaknesses:
    print(f"  ! {w}")

print("\nRecommendations:")
for r in hp.recommendations:
    print(f"  → {r}")
```

---

## Evaluation Confidence

Evaluation Confidence answers:
**"How confident should I be in this evaluation result?"**

It measures the completeness and dependability of the evaluation evidence rather than pipeline quality. A failing pipeline tested on 100 queries with complete judge verification has **High** evaluation confidence (the verdict is dependable), whereas a pipeline tested on only 3 queries has **Moderate** confidence.

### Scoring Factors
- **Query Coverage** (up to 50 pts): Proportional to successfully executed queries ($\frac{\text{completed}}{\text{total}}$).
- **Dataset Sample Size** (up to 35 pts): Evaluated via a smooth, saturating curve ($1 \to 0.20$, $5 \to 0.40$, $10 \to 0.60$, $25 \to 0.80$, $\ge 50 \to 1.0$).
- **Judge Evidence Completeness** (up to 15 pts): Successful judge verification provides the final confidence boost. Retrieval-only evaluation (judge not configured) receives a baseline of 8 pts and is never severely penalized.
- **Controlled Penalties**: Deductions for pipeline execution crashes and judge evaluation failures.

| Score Range | Level | Interpretation |
| :--- | :--- | :--- |
| **90.0 – 100.0** | `High` | Extensive sample size ($50+$ queries), full completion, and verified evidence. |
| **75.0 – 89.9** | `Good` | Solid sample coverage ($25+$ queries) with minimal or zero failures. |
| **60.0 – 74.9** | `Moderate` | Moderate sample size ($10–24$ queries) or minor partial execution drops. |
| **40.0 – 59.9** | `Low` | Limited dataset ($< 10$ queries) or noticeable execution failure rates. |
| **0.0 – 39.9** | `Very Low` | Pervasive execution crashes or critically insufficient evaluation evidence. |

In Python:

```python
# Access Evaluation Confidence
conf = report.confidence
print(f"Confidence Score: {conf.score}/100 ({conf.level})")

print("\nReasons:")
for reason in conf.reasons:
    print(f"  • {reason}")
```

---

## Comparison Methodology

### Directional Deltas ($\Delta = \text{Pipeline B} - \text{Pipeline A}$)
- **Quality Metrics**: Positive delta means Pipeline B achieved higher quality.
- **Latency Metrics**: Positive delta means Pipeline B is slower; negative delta means Pipeline B is faster.
- **Failure Counts**: Negative delta means Pipeline B reduced failures in that category (improvement).

### Winner Strategy & Trade-Off Detection
1. **Quality Priority**: Primary signals are evaluated in priority order:
   $$\text{Recall@}K \longrightarrow \text{MRR} \longrightarrow \text{Groundedness} \longrightarrow \text{Answer Correctness}$$
   Using configurable tolerance $\epsilon = 0.02$.
2. **Latency Trade-Off**: Evaluates latency delta against tolerance $\epsilon_{\text{lat}} = 10.0\text{ ms}$.
3. **Synthesis**:
   - Quality improves + latency increases $\to$ Declares winner with `"Higher quality <-> higher latency"`.
   - Quality improves + latency improves $\to$ Declares winner with `"Higher quality with improved latency"`.
   - Quality decreases + latency improves $\to$ Declares winner with `"Faster latency at the expense of lower quality"`.
   - Both roughly equal $\to$ Declares `"TIE"`.

---

## Python SDK API

```python
from ragdiag import Comparator, Evaluator, OpenAIJudge, build_report
from ragdiag.dataset import load_dataset
from ragdiag.pipeline import load_pipeline

# 1. Load pipeline and dataset
pipeline_a = load_pipeline("examples/dense_pipeline.py")
pipeline_b = load_pipeline("examples/hybrid_pipeline.py")
dataset = load_dataset("examples/demo_dataset.json")

# 2. Compare two pipelines
comparator = Comparator(k=5)
comparison = comparator.compare(pipeline_a, pipeline_b, dataset)

print(f"Overall Winner: {comparison.overall_winner}")
print(f"Trade-off:      {comparison.trade_off}")
print(f"Recall Delta:   {comparison.metric_deltas.recall_at_k:+.2f}")
print(f"Improved:       {comparison.queries_improved} queries")

# 3. Export structured JSON
json_output = comparison.model_dump_json(indent=2)
```

---

## Metrics Reference

- **Precision@K**: Fraction of top-$K$ retrieved chunks that are relevant:
  $$\text{Precision@}K = \frac{|\text{Relevant Chunks in Top-}K|}{\min(K, |\text{Retrieved Chunks}|)}$$
- **Recall@K**: Proportion of all ground-truth relevant chunks retrieved in top-$K$:
  $$\text{Recall@}K = \frac{|\text{Relevant Chunks in Top-}K|}{|\text{Total Relevant Chunks}|}$$
- **Mean Reciprocal Rank (MRR)**: Mean reciprocal rank of the first relevant chunk across queries:
  $$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
- **Answer Correctness**: Semantic equivalence of synthesized answer against expected ground truth.
- **Groundedness**: Evaluates whether synthesized claims are strictly supported by retrieved context chunks (the expected answer is never used as evidence for groundedness).
- **Latency Percentiles**: Linear interpolation percentiles (Mean, P50, P95, P99) for retrieval and generation execution.

---

## Limitations

- **Python-First**: Pipelines and adapters are authored in Python.
- **Two-Pipeline Comparison**: MVP currently supports comparing exactly two pipeline configurations (A vs B).
- **JSON Golden Datasets**: Evaluation datasets are currently loaded from structured JSON files.
- **LLM Judge Providers**: OpenAI structured outputs are currently supported out of the box; additional model providers can implement the extensible `Judge` interface.
- **Rule-Based Diagnosis**: The failure diagnosis taxonomy is derived deterministically from captured evidence rather than statistical model inference.

---

## Roadmap

Planned for future releases:
- [ ] Multi-configuration matrix comparison (>2 pipelines)
- [ ] Automated synthetic golden dataset generation
- [ ] Additional LLM judge providers (Anthropic, Gemini, local Ollama)
- [ ] Web dashboard and visual trace inspector
- [ ] CI/CD automation action for regression gating
- [ ] Docker containerized runner

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
