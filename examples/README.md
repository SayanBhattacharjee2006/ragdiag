# RAGDiag Reference Examples

This directory contains deterministic reference pipelines and evaluation datasets for testing and demonstrating RAGDiag's evaluation, root-cause diagnosis, and multi-pipeline comparison workflows.

> [!NOTE]
> These example pipelines are self-contained reference implementations designed for fast, reproducible, and credential-free testing. They simulate retrieval behavior in-memory without requiring external vector databases or network access.

---

## Included Files

| File | Type | Description |
| :--- | :--- | :--- |
| **`basic_pipeline.py`** | Pipeline Adapter | Single-pipeline reference adapter demonstrating the `Pipeline` class interface (`retrieve` and `generate`). |
| **`dense_pipeline.py`** | Pipeline Adapter | Deterministic simulation of a fast dense semantic search pipeline (~5ms retrieval latency) with partial context coverage on complex queries. |
| **`hybrid_pipeline.py`** | Pipeline Adapter | Deterministic simulation of a hybrid (dense + BM25) retrieval pipeline (~25ms retrieval latency) achieving 100% recall at the expense of latency. |
| **`basic_dataset.json`** | Golden Dataset | Baseline 5-query evaluation dataset covering `factual`, `reasoning`, and `multi-hop` queries. |
| **`demo_dataset.json`** | Golden Dataset | Payment gateway operations evaluation dataset covering refund timelines, webhook reliability, cross-border 3DS rules, auto-debit decline analysis, and multi-hop fee calculations. |

---

## 1. Single Pipeline Evaluation

Run evaluation on the reference `basic_pipeline`:

```bash
ragdiag run --pipeline examples/basic_pipeline.py --dataset examples/basic_dataset.json
```

---

## 2. Multi-Pipeline Comparison

Compare the simulated dense retriever (`Pipeline A`) against the candidate hybrid retriever (`Pipeline B`) using the payment gateway demo dataset:

```bash
ragdiag compare \
  --pipeline-a examples/dense_pipeline.py \
  --pipeline-b examples/hybrid_pipeline.py \
  --dataset examples/demo_dataset.json
```

This demonstrates:
- **Quality Improvement**: Recall@5 increases from 0.80 to 1.00 (+20 percentage points).
- **Failure Resolution**: Eliminates two `INSUFFICIENT_CONTEXT` failures.
- **Latency Trade-Off**: Mean retrieval latency increases by ~20ms, triggering RAGDiag's deterministic trade-off detection: `"Higher quality <-> higher latency"`.

---

## 3. Dataset Format

Datasets are structured JSON files matching the `GoldenDataset` schema:

```json
{
  "name": "payment_gateway_demo_eval",
  "version": "1.0",
  "samples": [
    {
      "id": "q001",
      "query": "What is the standard turnaround time for card refund settlements?",
      "expected_answer": "Standard card refund settlements are credited within 5 to 7 business days.",
      "relevant_chunk_ids": ["doc_refund_policy_01"],
      "query_type": "factual"
    }
  ]
}
```

Validate any dataset with:

```bash
ragdiag validate --dataset examples/demo_dataset.json
```

---

## 4. Adapting Your Own RAG Pipeline

To evaluate your real RAG system (using LangChain, LlamaIndex, Chroma, Pinecone, or custom vector stores), implement the minimal `ragdiag.Pipeline` contract:

```python
# custom_pipeline.py
from ragdiag import Pipeline, RetrievedChunk


class MyPipeline(Pipeline):
    name = "my_custom_rag"

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        # Connect to your vector DB or retriever
        return [
            RetrievedChunk(
                id="chunk_01",
                text="Document context text...",
                score=0.95,
            )
        ]

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> str:
        # Call your LLM with retrieved chunks
        return "Generated response based on context."


# Expose top-level adapter instance named 'pipeline'
pipeline = MyPipeline()
```
