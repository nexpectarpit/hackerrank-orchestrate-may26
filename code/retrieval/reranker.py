from __future__ import annotations

from typing import List

from config import settings
from corpus.models import Chunk
from llm.client import get_client
from llm.prompts import RERANK_PROMPT
from utils.logger import get_logger

log = get_logger(__name__)


class Reranker:
    """Scores and re-ranks retrieved candidate chunks using the LLM."""

    def __init__(self) -> None:
        self.client = get_client()

    def rerank(
        self, chunks: List[Chunk], issue: str, subject: str, top_k: int = 3
    ) -> List[Chunk]:
        """Re-rank candidate chunks based on their relevance to the ticket."""
        if not chunks:
            return []

        # If we have less chunks than top_k, or if the LLM client isn't configured,
        # just return the original list up to top_k.
        if len(chunks) <= top_k or not self.client.client:
            return chunks[:top_k]

        chunks_text = ""
        for i, chunk in enumerate(chunks):
            # Limit the text size per chunk sent to the LLM to avoid token limits
            text_preview = chunk.text[:800]
            chunks_text += (
                f"ID: chunk_{i}\nTitle: {chunk.title}\nText: {text_preview}\n---\n"
            )

        prompt = RERANK_PROMPT.format(
            issue=issue, subject=subject, chunks=chunks_text
        )

        messages = [{"role": "user", "content": prompt}]

        try:
            result = self.client.generate_json(messages, model=settings.GROQ_FAST_MODEL)

            # Map the response scores back to the original chunks
            scored_chunks = []
            for i, chunk in enumerate(chunks):
                key = f"chunk_{i}"
                score = result.get(key, 0)
                try:
                    score = int(score)
                except (ValueError, TypeError):
                    score = 0
                scored_chunks.append((score, chunk))

            # Sort by score descending
            scored_chunks.sort(key=lambda x: x[0], reverse=True)

            log.info("Reranked scores: %s", [(s, c.chunk_id) for s, c in scored_chunks])

            top_chunks = [c for s, c in scored_chunks[:top_k]]
            return top_chunks

        except Exception as e:
            log.warning("Reranking failed: %s. Returning original top_k.", e)
            return chunks[:top_k]
