from __future__ import annotations

"""
Structured logging and decision tracing for the support triage agent.

Provides:
- get_logger(): returns a configured stdlib logger for any module
- DecisionTracer: records per-ticket decision traces as structured dicts,
  then writes them to a JSON file at the end of the run for debugging
  and interview preparation.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import settings


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured with the project-wide log level.

    All loggers share a single StreamHandler writing to stderr so that
    normal stdout output (progress bars, summaries) stays clean.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
    return logger


class DecisionTracer:
    """Accumulates per-ticket decision traces and writes them to disk.

    Usage::

        tracer = DecisionTracer()
        tracer.start_ticket(ticket_id=1, issue="...", subject="...", company="...")
        tracer.record("router", {"detected_company": "Claude", "confidence": 0.95})
        tracer.record("retrieval", {"chunks_retrieved": 8})
        tracer.end_ticket(duration_ms=2340)
        ...
        tracer.save()  # writes JSON to traces/ directory
    """

    def __init__(self) -> None:
        self._traces: list[dict[str, Any]] = []
        self._current: dict[str, Any] | None = None
        self._run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._logger = get_logger("tracer")

    def start_ticket(
        self,
        ticket_id: int,
        issue: str,
        subject: str,
        company: str,
    ) -> None:
        """Begin recording a new ticket trace."""
        self._current = {
            "ticket_id": ticket_id,
            "issue": issue[:200],  # truncate for readability
            "subject": subject,
            "company": company,
            "trace": {},
        }

    def record(self, stage: str, data: dict[str, Any]) -> None:
        """Record a decision for the current ticket at the given stage."""
        if self._current is None:
            self._logger.warning(
                "record() called without start_ticket() -- ignoring"
            )
            return
        self._current["trace"][stage] = data

    def end_ticket(self, duration_ms: int = 0) -> None:
        """Finalize the current ticket trace."""
        if self._current is None:
            return
        self._current["duration_ms"] = duration_ms
        self._traces.append(self._current)
        self._current = None

    def save(self) -> Path:
        """Write all traces to a JSON file and return the file path."""
        out_dir = settings.TRACES_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"run_{self._run_id}.json"

        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(self._traces, fh, indent=2, ensure_ascii=False)

        self._logger.info("Decision traces saved to %s", out_path)
        return out_path

    @property
    def traces(self) -> list[dict[str, Any]]:
        """Return a copy of all traces collected so far."""
        return list(self._traces)
