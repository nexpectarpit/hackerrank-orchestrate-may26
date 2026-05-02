from __future__ import annotations

"""
Critic -- Stage 4 of the agent pipeline.

Reviews the generated response for hallucinations, safety issues,
and classification correctness. If the quality is "needs_revision",
the pipeline can trigger a re-generation with stricter constraints.
"""

from typing import Any, Dict, List

from corpus.models import Chunk
from llm.client import get_client
from llm.prompts import CRITIC_PROMPT
from utils.logger import get_logger

log = get_logger(__name__)


def _format_context_for_critic(chunks: List[Chunk]) -> str:
    """Format context chunks for the critic prompt."""
    if not chunks:
        return "(No documentation was retrieved.)"

    parts = []
    for i, chunk in enumerate(chunks):
        parts.append(
            f"--- Document {i + 1} ---\n"
            f"Title: {chunk.title}\n"
            f"Content:\n{chunk.text[:600]}\n"
        )
    return "\n".join(parts)


def critique_response(
    issue: str,
    subject: str,
    response: str,
    status: str,
    request_type: str,
    product_area: str,
    chunks: List[Chunk],
) -> Dict[str, Any]:
    """Review the generated response for quality issues.

    Returns a dict with keys:
        has_hallucination, has_safety_issue, classification_correct,
        suggested_fixes, overall_quality, passed
    """
    client = get_client()

    context = _format_context_for_critic(chunks)

    prompt = CRITIC_PROMPT.format(
        issue=issue,
        subject=subject,
        context=context,
        response=response,
        status=status,
        request_type=request_type,
        product_area=product_area,
    )

    messages = [{"role": "user", "content": prompt}]

    # Default: assume everything is fine (fail-open to avoid blocking)
    defaults = {
        "has_hallucination": False,
        "has_safety_issue": False,
        "classification_correct": True,
        "suggested_fixes": "none",
        "overall_quality": "acceptable",
        "passed": True,
    }

    try:
        result = client.generate_json(messages)

        has_hallucination = bool(result.get("has_hallucination", False))
        has_safety_issue = bool(result.get("has_safety_issue", False))
        classification_correct = bool(result.get("classification_correct", True))
        suggested_fixes = str(result.get("suggested_fixes", "none"))
        overall_quality = str(result.get("overall_quality", "acceptable")).lower()

        if overall_quality not in ("good", "acceptable", "needs_revision"):
            overall_quality = "acceptable"

        passed = (
            overall_quality != "needs_revision"
            and not has_hallucination
            and not has_safety_issue
        )

        output = {
            "has_hallucination": has_hallucination,
            "has_safety_issue": has_safety_issue,
            "classification_correct": classification_correct,
            "suggested_fixes": suggested_fixes,
            "overall_quality": overall_quality,
            "passed": passed,
        }

        if not passed:
            log.warning(
                "Critic REJECTED response: hallucination=%s, safety=%s, quality=%s, fixes=%s",
                has_hallucination,
                has_safety_issue,
                overall_quality,
                suggested_fixes,
            )
        else:
            log.info("Critic APPROVED response: quality=%s", overall_quality)

        return output

    except Exception as exc:
        log.warning("Critic failed: %s. Assuming response is acceptable.", exc)
        return defaults
