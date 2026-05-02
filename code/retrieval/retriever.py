from __future__ import annotations

from typing import List, Optional

import numpy as np

from config import settings
from corpus.models import Chunk
from retrieval.indexer import get_indexer
from utils.errors import RetrievalError
from utils.logger import get_logger

log = get_logger(__name__)


class Retriever:
    """Handles semantic search over the corpus chunks."""

    def __init__(self) -> None:
        self.indexer = get_indexer()

    def search(
        self,
        query: str,
        company_filter: Optional[str] = None,
        top_k: int = settings.FAISS_TOP_K,
    ) -> List[Chunk]:
        """Search the FAISS index for the given query.

        If company_filter is provided, post-filters the results.
        """
        if self.indexer.index is None:
            self.indexer.build_or_load()

        if self.indexer.index is None or not self.indexer.chunks:
            raise RetrievalError("FAISS index is not initialized or empty")

        try:
            query_embedding = self.indexer.model.encode([query], convert_to_numpy=True)
            if not isinstance(query_embedding, np.ndarray):
                query_embedding = np.array(query_embedding)
        except Exception as e:
            raise RetrievalError(f"Failed to embed query: {e}") from e

        # To support post-filtering by company, we retrieve a larger pool of candidates
        fetch_k = min(top_k * 5, len(self.indexer.chunks)) if company_filter else top_k

        distances, indices = self.indexer.index.search(query_embedding, fetch_k)

        results: List[Chunk] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.indexer.chunks[idx]

            # Post-filter by company
            if company_filter and company_filter.lower() != "none" and company_filter.strip() != "":
                if chunk.company.lower() != company_filter.lower():
                    continue

            results.append(chunk)
            if len(results) >= top_k:
                break

        return results


if __name__ == "__main__":
    retriever = Retriever()

    query = "How long do tests stay active"
    company = "HackerRank"

    print(f"\nSearching for: '{query}' [Company: {company}]")
    chunks = retriever.search(query, company_filter=company, top_k=3)

    for i, c in enumerate(chunks):
        print(f"\nResult {i+1}:")
        print(f"  ID: {c.chunk_id}")
        print(f"  Title: {c.title}")
        print(f"  Product Area: {c.product_area}")
        print(f"  Text preview: {c.text[:150]}...")
