from __future__ import annotations

"""
Generator -- Stage 2 of the agent pipeline.

Takes retrieved context chunks and the ticket, then generates
the response, status, product_area, request_type, and justification
using Chain-of-Thought prompting.
"""

from typing import Any, Dict, List

from config import settings
from corpus.models import Chunk
from llm.client import get_client
from llm.prompts import GENERATOR_PROMPT
from utils.logger import get_logger

log = get_logger(__name__)


def _format_context(chunks: List[Chunk]) -> str:
    """Format retrieved chunks into a readable context block for the LLM."""
    if not chunks:
        return "(No relevant documentation found.)"

    parts = []
    for i, chunk in enumerate(chunks):
        parts.append(
            f"--- Document {i + 1} ---\n"
            f"Title: {chunk.title}\n"
            f"Product Area: {chunk.product_area}\n"
            f"Source: {chunk.source_url}\n"
            f"Content:\n{chunk.text}\n"
        )
    return "\n".join(parts)


def generate_response(
    ticket_id: int,
    issue: str,
    subject: str,
    company: str,
    chunks: List[Chunk],
    request_type_hint: str = "",
) -> Dict[str, Any]:
    """Generate the agent response for a single ticket.

    Returns a dict with keys:
        status, response, product_area, request_type, justification, log_entry
    """
    client = get_client()

    context = _format_context(chunks)

    prompt = GENERATOR_PROMPT.format(
        ticket_id=ticket_id,
        company=company if company else "the relevant company",
        context=context,
        issue=issue,
        subject=subject,
    )

    messages = [{"role": "user", "content": prompt}]

    defaults = {
        "status": "escalated",
        "response": (
            "Thank you for reaching out. Your request has been escalated "
            "to a human support agent who can assist you further."
        ),
        "product_area": "general_support",
        "request_type": request_type_hint or "product_issue",
        "justification": "Unable to generate a grounded response. Escalated for safety.",
    }

    try:
        result = client.generate_json(messages)

        # Validate status
        status = str(result.get("status", "")).strip().lower()
        if status not in settings.ALLOWED_STATUSES:
            status = "escalated"

        # Validate request_type
        request_type = str(result.get("request_type", "")).strip().lower()
        if request_type not in settings.ALLOWED_REQUEST_TYPES:
            request_type = request_type_hint if request_type_hint else "product_issue"

        response_text = str(result.get("response", defaults["response"])).strip()
        if not response_text:
            response_text = defaults["response"]

        product_area = str(result.get("product_area", "")).strip().lower()
        if not product_area:
            # Derive from the top chunk if available
            if chunks:
                product_area = chunks[0].product_area
            else:
                product_area = "general_support"

        justification = str(result.get("justification", "")).strip()
        if not justification:
            justification = "Response generated from retrieved documentation."

        output = {
            "status": status,
            "response": response_text,
            "product_area": product_area,
            "request_type": request_type,
            "justification": justification,
        }

        log.info(
            "Generator: status=%s, type=%s, area=%s",
            output["status"],
            output["request_type"],
            output["product_area"],
        )

        return output

    except Exception as exc:
        log.warning("Generator failed: %s. Returning escalation defaults.", exc)
        return defaults
