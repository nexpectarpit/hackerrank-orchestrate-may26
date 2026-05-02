from __future__ import annotations

from typing import List

from config import settings
from llm.client import get_client
from llm.prompts import QUERY_EXPANSION_PROMPT
from utils.logger import get_logger

log = get_logger(__name__)


class QueryExpander:
    """Uses the LLM to generate alternative queries for better semantic search."""

    def __init__(self) -> None:
        self.client = get_client()

    def expand(self, issue: str, subject: str, company: str, count: int = 2) -> List[str]:
        """Generate alternative queries to capture vocabulary mismatches."""
        if not self.client.client:
            log.warning("No LLM client available. Returning original query only.")
            return [issue]

        prompt = QUERY_EXPANSION_PROMPT.format(
            issue=issue,
            subject=subject,
            company=company if company else "unknown",
            count=count,
        )

        messages = [{"role": "user", "content": prompt}]

        try:
            result = self.client.generate_json(messages, model=settings.GROQ_FAST_MODEL)
            queries = result.get("queries", [])
            if not isinstance(queries, list):
                queries = []

            # Clean and filter queries
            queries = [str(q).strip() for q in queries if str(q).strip()]

            if not queries:
                log.warning("LLM returned empty query list. Using original issue.")
                return [issue]

            log.info("Expanded queries: %s", queries)
            
            # Always return the original issue plus the new ones
            return [issue] + queries

        except Exception as e:
            log.warning("Query expansion failed: %s. Using original issue.", e)
            return [issue]
