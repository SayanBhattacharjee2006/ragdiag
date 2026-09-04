"""Example deterministic hybrid retrieval pipeline combining dense semantic search and BM25."""

import time

from ragdiag import Pipeline, RetrievedChunk


class HybridPipeline(Pipeline):
    """Simulated hybrid retrieval pipeline achieving higher recall at the cost of latency."""

    name = "hybrid_pipeline"

    CHUNKS: dict[str, list[RetrievedChunk]] = {
        "refund": [
            RetrievedChunk(
                id="doc_refund_policy_01",
                text="Standard card refund settlements are credited within 5 to 7 business days.",
                score=0.96,
            ),
        ],
        "webhook": [
            RetrievedChunk(
                id="doc_webhooks_04",
                text="Webhooks are retried with exponential backoff up to 5 times over 24 hours.",
                score=0.94,
            ),
        ],
        "international": [
            RetrievedChunk(
                id="doc_compliance_02",
                text="International transactions require 3D Secure verification under regulations.",
                score=0.93,
            ),
            RetrievedChunk(
                id="doc_auth_flows_09",
                text="Two-factor authentication mitigates cross-border fraud risks.",
                score=0.91,
            ),
        ],
        # Hybrid retrieval captures both necessary chunks for auto-debit
        "auto-debit": [
            RetrievedChunk(
                id="doc_subscriptions_03",
                text="Recurring payments require active e-mandate registration.",
                score=0.92,
            ),
            RetrievedChunk(
                id="doc_mandates_05",
                text="Expired mandates cause auto-debit transactions to be declined.",
                score=0.95,
            ),
        ],
        # Hybrid retrieval captures both necessary chunks for multi-hop fee calculation
        "fee": [
            RetrievedChunk(
                id="doc_pricing_tier_01",
                text="Commercial card transactions have a base interchange fee of 2.5%.",
                score=0.90,
            ),
            RetrievedChunk(
                id="doc_tax_regulations_03",
                text="Applicable GST of 18% is levied on payment gateway fee amounts.",
                score=0.88,
            ),
        ],
    }

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """Retrieve matching chunks using simulated dense + BM25 hybrid search."""
        time.sleep(0.025)  # Simulated two-stage hybrid retrieval latency
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
        return f"Based on hybrid retrieval: {context_summary}"


pipeline = HybridPipeline()
