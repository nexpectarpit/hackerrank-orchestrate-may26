from __future__ import annotations

"""
Evaluator -- compares agent output against the sample_support_tickets.csv.

Usage:
    python -m evaluation.evaluator
"""

import pandas as pd

from config import settings
from utils.csv_handler import read_sample_tickets
from utils.logger import get_logger

log = get_logger(__name__)


def evaluate() -> None:
    """Compare output.csv against sample tickets and print accuracy."""

    # Read sample tickets (ground truth)
    sample = read_sample_tickets()

    # Read our output
    try:
        output = pd.read_csv(settings.OUTPUT_CSV)
        output.columns = [c.strip().lower().replace(" ", "_") for c in output.columns]
    except FileNotFoundError:
        print("ERROR: output.csv not found. Run main.py first.")
        return

    # Match tickets by issue text (first 100 chars to handle truncation)
    sample["match_key"] = sample["issue"].astype(str).str[:100].str.strip().str.lower()
    output["match_key"] = output["issue"].astype(str).str[:100].str.strip().str.lower()

    # Fields to compare
    fields = ["status", "request_type"]
    if "product_area" in sample.columns:
        fields.append("product_area")

    print()
    print("=" * 60)
    print("  EVALUATION: Output vs Sample Tickets")
    print("=" * 60)
    print()

    total_matched = 0
    field_correct = {f: 0 for f in fields}

    for _, s_row in sample.iterrows():
        key = s_row["match_key"]
        matches = output[output["match_key"] == key]

        if matches.empty:
            print(f"  [MISS] No output match for: {s_row['issue'][:60]}...")
            continue

        o_row = matches.iloc[0]
        total_matched += 1

        row_summary = f"  Ticket: {str(s_row.get('subject', ''))[:40]}"
        mismatches = []

        for field in fields:
            expected = str(s_row.get(field, "")).strip().lower()
            actual = str(o_row.get(field, "")).strip().lower()

            if expected and actual == expected:
                field_correct[field] += 1
            elif expected:
                mismatches.append(f"{field}: expected='{expected}' got='{actual}'")

        if mismatches:
            print(f"{row_summary}")
            for m in mismatches:
                print(f"    MISMATCH: {m}")
        else:
            print(f"{row_summary} -- ALL CORRECT")

    print()
    print("-" * 60)
    print(f"  Matched: {total_matched}/{len(sample)} sample tickets")
    print()
    for field in fields:
        total = total_matched
        correct = field_correct[field]
        pct = (correct / total * 100) if total > 0 else 0
        print(f"  {field:20s}: {correct}/{total} ({pct:.0f}%)")
    print("-" * 60)
    print()


if __name__ == "__main__":
    evaluate()
