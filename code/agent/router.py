from __future__ import annotations

"""
Router -- Stage 1 of the agent pipeline.

Classifies the ticket: resolves the company (if missing), determines
request_type, sensitivity level, and confidence score.
"""

from typing import Any, Dict

from llm.client import get_client
from llm.prompts import ROUTER_PROMPT
from utils.logger import get_logger

log = get_logger(__name__)


def route_ticket(issue: str, subject: str, company: str) -> Dict[str, Any]:
    """Classify a single ticket using the LLM.

    Returns a dict with keys:
        resolved_company, request_type, sensitivity, confidence, reasoning
    """
    client = get_client()

    prompt = ROUTER_PROMPT.format(
        issue=issue,
        subject=subject,
        company=company if company else "None",
    )

    messages = [{"role": "user", "content": prompt}]

    defaults = {
        "resolved_company": company if company else "Unknown",
        "request_type": "product_issue",
        "sensitivity": "low",
        "confidence": 0.5,
        "reasoning": "Classification failed, using defaults.",
    }

    try:
        result = client.generate_json(messages)

        # Validate and normalize the result
        from config.settings import SUPPORTED_COMPANIES
        
        # Hard override: If CSV explicitly provided a supported company, do not let LLM change it
        if company and company in SUPPORTED_COMPANIES:
            resolved_company = company
        else:
            resolved_company = result.get("resolved_company", defaults["resolved_company"])
            if resolved_company not in ("HackerRank", "Claude", "Visa", "Unknown"):
                resolved_company = defaults["resolved_company"]

        from config.settings import ALLOWED_REQUEST_TYPES
        request_type = result.get("request_type", defaults["request_type"])
        if request_type not in ALLOWED_REQUEST_TYPES:
            request_type = defaults["request_type"]

        sensitivity = result.get("sensitivity", defaults["sensitivity"])
        if sensitivity not in ("high", "medium", "low"):
            sensitivity = defaults["sensitivity"]

        try:
            confidence = float(result.get("confidence", defaults["confidence"]))
            confidence = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            confidence = defaults["confidence"]

        reasoning = str(result.get("reasoning", defaults["reasoning"]))

        output = {
            "resolved_company": resolved_company,
            "request_type": request_type,
            "sensitivity": sensitivity,
            "confidence": confidence,
            "reasoning": reasoning,
        }

        log.info(
            "Router: company=%s, type=%s, sensitivity=%s, confidence=%.2f",
            output["resolved_company"],
            output["request_type"],
            output["sensitivity"],
            output["confidence"],
        )

        return output

    except Exception as exc:
        log.warning("Router classification failed: %s. Using defaults.", exc)
        return defaults
