"""Example deterministic dense vector retrieval pipeline for comparison demonstrations."""

import time

from ragdiag import Pipeline, RetrievedChunk


class DensePipeline(Pipeline):
    """Simulated dense semantic search pipeline with fast retrieval but partial context coverage."""

    name = "dense_pipeline"

    CHUNKS: dict[str, list[RetrievedChunk]] = {
        "refund": [
            RetrievedChunk(
                id="doc_refund_policy_01",
                text="Standard card refund settlements are credited within 5 to 7 business days.",
                score=0.92,
            ),
        ],
        "webhook": [
            RetrievedChunk(
                id="doc_webhooks_04",
                text="Webhooks are retried with exponential backoff up to 5 times over 24 hours.",
                score=0.88,
            ),
        ],
        "international": [
            RetrievedChunk(
                id="doc_compliance_02",
                text="International transactions require 3D Secure verification under regulations.",
                score=0.91,
            ),
            RetrievedChunk(
                id="doc_auth_flows_09",
                text="Two-factor authentication mitigates cross-border fraud risks.",
                score=0.86,
            ),
        ],
        # Dense index only retrieves 1 of the 2 required chunks for auto-debit
        "auto-debit": [
            RetrievedChunk(
                id="doc_subscriptions_03",
                text="Recurring payments require active e-mandate registration.",
                score=0.84,
            ),
        ],
        # Dense index only retrieves 1 of the 2 required chunks for multi-hop fee
        "fee": [
            RetrievedChunk(
                id="doc_pricing_tier_01",
                text="Commercial card transactions have a base interchange fee of 2.5%.",
                score=0.85,
            ),
        ],
    }

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """Retrieve matching chunks using simulated dense semantic search."""
        time.sleep(0.005)  # Simulated fast dense retrieval
        query_lower = query.lower()
        matched: list[RetrievedChunk] = []
        seen_ids: set[str] = set()

        for keyword, chunks in self.CHUNKS.items():
            if keyword in query_lower:
                for chunk in chunks:
                    if chunk.id not in seen_ids:
                        seen_ids.add(chunk.id)
                        matched.append(chunk)

        return matched

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> str:
        """Generate response by synthesizing retrieved chunk contents."""
        if not chunks:
            return "Unable to answer: insufficient relevant documentation found."

        context_summary = " ".join(chunk.text for chunk in chunks)
        return f"Based on dense retrieval: {context_summary}"


pipeline = DensePipeline()
