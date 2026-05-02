from __future__ import annotations

"""
CSV reading and writing with schema validation.

Ensures the output CSV always matches the required 8-column schema
defined in config.settings.OUTPUT_COLUMNS.
"""

import pandas as pd
from pathlib import Path
from typing import Any

from config import settings
from utils.errors import SchemaValidationError
from utils.logger import get_logger

log = get_logger(__name__)


def read_input_tickets(path: Path | None = None) -> pd.DataFrame:
    """Read the input support_tickets.csv and normalize column names.

    Returns a DataFrame with lowercase column names:
    issue, subject, company
    """
    csv_path = path or settings.INPUT_CSV
    log.info("Reading input tickets from %s", csv_path)

    df = pd.read_csv(csv_path)

    # Normalize column names to lowercase
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    expected = {"issue", "subject", "company"}
    missing = expected - set(df.columns)
    if missing:
        raise SchemaValidationError(
            f"Input CSV is missing required columns: {missing}"
        )

    # Replace NaN with empty strings for text fields
    for col in ["issue", "subject", "company"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    log.info("Loaded %d tickets", len(df))
    return df


def read_sample_tickets(path: Path | None = None) -> pd.DataFrame:
    """Read the sample_support_tickets.csv for evaluation."""
    csv_path = path or settings.SAMPLE_CSV
    log.info("Reading sample tickets from %s", csv_path)

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def validate_output_row(row: dict[str, Any]) -> dict[str, str]:
    """Validate and clean a single output row.

    Raises SchemaValidationError if required fields are missing or
    status/request_type values are not in the allowed sets.
    """
    errors = []

    # Check all required columns exist
    for col in settings.OUTPUT_COLUMNS:
        if col not in row:
            errors.append(f"Missing column: {col}")

    if errors:
        raise SchemaValidationError(
            f"Output row validation failed: {'; '.join(errors)}"
        )

    # Validate allowed values
    status = str(row.get("Status", "")).strip().lower()
    if status not in settings.ALLOWED_STATUSES:
        errors.append(
            f"Invalid status '{status}'. "
            f"Allowed: {settings.ALLOWED_STATUSES}"
        )

    request_type = str(row.get("Request Type", "")).strip().lower()
    if request_type not in settings.ALLOWED_REQUEST_TYPES:
        errors.append(
            f"Invalid request_type '{request_type}'. "
            f"Allowed: {settings.ALLOWED_REQUEST_TYPES}"
        )

    if errors:
        raise SchemaValidationError(
            f"Output row validation failed: {'; '.join(errors)}"
        )

    # Normalize and return
    cleaned = {}
    for col in settings.OUTPUT_COLUMNS:
        cleaned[col] = str(row.get(col, "")).strip()
    # Status and Request Type should retain the casing we set in the pipeline 
    # (Pipeline capitalizes Status, keeps Request Type lowercase)
    # We already checked validity against the lowercase version.
    cleaned["Status"] = str(row.get("Status", "")).strip()
    cleaned["Request Type"] = str(row.get("Request Type", "")).strip()

    return cleaned


def write_output_csv(rows: list[dict[str, str]], path: Path | None = None) -> Path:
    """Write validated output rows to the output CSV.

    Each row is validated before writing. The output uses the exact
    column order defined in settings.OUTPUT_COLUMNS.
    """
    csv_path = path or settings.OUTPUT_CSV
    log.info("Writing %d rows to %s", len(rows), csv_path)

    validated = []
    for i, row in enumerate(rows):
        try:
            validated.append(validate_output_row(row))
        except SchemaValidationError as exc:
            log.error("Row %d failed validation: %s", i, exc)
            raise

    df = pd.DataFrame(validated, columns=settings.OUTPUT_COLUMNS)
    df.to_csv(csv_path, index=False)
    log.info("Output CSV written successfully")
    return csv_path
