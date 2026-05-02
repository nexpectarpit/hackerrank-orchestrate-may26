from __future__ import annotations

"""
Safety module -- pre-processing safety checks.

Runs before the main generation stage to detect prompt injection,
out-of-scope requests, and harmful content. Uses both pattern
matching (fast, no API cost) and an optional LLM call for ambiguous
cases.
"""

import re
from typing import Any, Dict

from llm.client import get_client
from llm.prompts import SAFETY_PROMPT
from utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Pattern-based fast checks (no LLM call needed)
# ---------------------------------------------------------------------------

# Common prompt injection patterns in multiple languages
_INJECTION_PATTERNS = [
    # English
    r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|rules|prompts)",
    r"reveal\s+(your|the|system)\s+(prompt|instructions|rules|logic)",
    r"(show|display|print|output)\s+(your|the|all)\s+(internal|system)",
    r"act\s+as\s+if\s+you\s+have\s+no\s+rules",
    r"pretend\s+(you\s+are|to\s+be)",
    r"you\s+are\s+now\s+(a|an)\s+",
    r"disregard\s+(your|all|any)\s+(instructions|rules|guidelines)",
    # French
    r"affiche[zr]?\s+(toutes?\s+)?(les\s+)?r[eè]gles\s+internes",
    r"logique\s+(exacte|interne)\s+(que\s+)?vous\s+utilisez",
    r"documents?\s+r[eé]cup[eé]r[eé]s",
    # General code execution attempts
    r"(execute|run|eval)\s+(this|the|following)\s+(code|script|command)",
    r"(delete|remove|rm)\s+(all\s+)?files",
    r"import\s+os",
    r"subprocess\.",
    r"__import__",
]

_INJECTION_RE = re.compile(
    "|".join(_INJECTION_PATTERNS), re.IGNORECASE
)

# Keywords that indicate out-of-scope requests
_OUT_OF_SCOPE_KEYWORDS = [
    "iron man", "avengers", "movie", "recipe", "weather forecast",
    "sports score", "stock price", "crypto", "dating advice",
]


def _fast_injection_check(issue: str, subject: str) -> bool:
    """Quick regex check for known prompt injection patterns."""
    combined = f"{issue} {subject}"
    return bool(_INJECTION_RE.search(combined))


def _fast_out_of_scope_check(issue: str, subject: str) -> bool:
    """Quick keyword check for obviously off-topic requests."""
    combined = f"{issue} {subject}".lower()
    return any(kw in combined for kw in _OUT_OF_SCOPE_KEYWORDS)


# ---------------------------------------------------------------------------
# Full safety check (with LLM)
# ---------------------------------------------------------------------------

def check_safety(issue: str, subject: str, company: str) -> Dict[str, Any]:
    """Run safety checks on a ticket.

    Returns a dict with keys:
        is_prompt_injection, is_out_of_scope, is_harmful,
        safety_reasoning, used_llm
    """
    result = {
        "is_prompt_injection": False,
        "is_out_of_scope": False,
        "is_harmful": False,
        "safety_reasoning": "No safety concerns detected.",
        "used_llm": False,
    }

    # Fast pattern checks first
    combined = f"{issue} {subject}"
    if _INJECTION_RE.search(combined):
        stripped = _INJECTION_RE.sub("", combined).strip()
        if len(stripped) < 15:
            result["is_prompt_injection"] = True
            result["safety_reasoning"] = "Prompt injection pattern detected via regex."
            log.warning("SAFETY: Prompt injection detected (fast check)")
            return result
        else:
            log.warning("SAFETY: Prompt injection detected but seems mixed. Proceeding to ignore malicious part.")

    if _fast_out_of_scope_check(issue, subject):
        result["is_out_of_scope"] = True
        result["safety_reasoning"] = "Out-of-scope topic detected via keyword match."
        log.warning("SAFETY: Out-of-scope request detected (fast check)")
        return result

    # For ambiguous cases, use the LLM
    # Heuristic: if the issue is very short or the company is unknown,
    # or the text contains unusual characters, run the LLM check
    combined = f"{issue} {subject}".strip()

    # Fast check for empty inputs: don't flag as out-of-scope, let pipeline escalate
    if not combined:
        return result

    needs_llm_check = (
        len(combined) < 30
        or (not company or company.lower() in ("none", "", "unknown"))
        or any(ord(c) > 127 for c in combined)  # non-ASCII (possible multi-lang injection)
    )

    if needs_llm_check:
        try:
            client = get_client()
            prompt = SAFETY_PROMPT.format(
                issue=issue,
                subject=subject,
                company=company if company and company.lower() != "none" else "Not specified",
            )
            messages = [{"role": "user", "content": prompt}]
            llm_result = client.generate_json(messages)

            result["is_prompt_injection"] = bool(llm_result.get("is_prompt_injection", False))
            result["is_out_of_scope"] = bool(llm_result.get("is_out_of_scope", False))
            result["is_harmful"] = bool(llm_result.get("is_harmful", False))
            result["safety_reasoning"] = str(
                llm_result.get("safety_reasoning", "LLM safety check completed.")
            )
            result["used_llm"] = True

            if result["is_prompt_injection"]:
                log.warning("SAFETY: Prompt injection detected (LLM check)")
            if result["is_out_of_scope"]:
                log.warning("SAFETY: Out-of-scope detected (LLM check)")
            if result["is_harmful"]:
                log.warning("SAFETY: Harmful content detected (LLM check)")

        except Exception as exc:
            log.warning("LLM safety check failed: %s. Proceeding with caution.", exc)
            result["safety_reasoning"] = f"LLM safety check failed: {exc}"

    return result
