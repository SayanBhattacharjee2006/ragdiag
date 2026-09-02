"""Example deterministic in-memory RAG pipeline for RAGDiag demonstration."""

from ragdiag import Pipeline, RetrievedChunk


class BasicPipeline(Pipeline):
    """Deterministic in-memory RAG pipeline that matches keywords to document chunks."""

    name = "basic_pipeline"

    # In-memory document chunk store
    CHUNKS: dict[str, list[RetrievedChunk]] = {
        "refund": [
            RetrievedChunk(
                id="doc_refund_policy_01",
                text="Standard card refund settlements are credited within 5 to 7 business days.",
                score=0.95,
            ),
        ],
        "webhook": [
            RetrievedChunk(
                id="doc_webhooks_04",
                text="Webhooks are retried with exponential backoff up to 5 times over 24 hours.",
                score=0.91,
            ),
        ],
        "international": [
            RetrievedChunk(
                id="doc_compliance_02",
                text="International transactions require 3D Secure verification under regulations.",
                score=0.92,
            ),
            RetrievedChunk(
                id="doc_auth_flows_09",
                text="Two-factor authentication mitigates cross-border fraud risks.",
                score=0.88,
            ),
        ],
        "auto-debit": [
            RetrievedChunk(
                id="doc_subscriptions_03",
                text="Recurring payments require active e-mandate registration.",
                score=0.89,
            ),
            RetrievedChunk(
                id="doc_mandates_05",
                text="Expired mandates cause auto-debit transactions to be declined.",
                score=0.93,
            ),
        ],
        "fee": [
            RetrievedChunk(
                id="doc_pricing_tier_01",
                text="Commercial card transactions have a base interchange fee of 2.5%.",
                score=0.87,
            ),
            RetrievedChunk(
                id="doc_tax_regulations_03",
                text="Applicable GST of 18% is levied on payment gateway fee amounts.",
                score=0.85,
            ),
        ],
    }

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """Retrieve matching chunks based on query keywords."""
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
        return f"Based on available documentation: {context_summary}"


pipeline = BasicPipeline()
