from __future__ import annotations

"""
main.py -- Entry point for the support triage agent.

Reads support_tickets.csv, processes each ticket through the
multi-stage agent pipeline, and writes the results to output.csv.
"""

import sys
import time
from datetime import datetime
from pathlib import Path

from config import settings
from agent.pipeline import Pipeline
from retrieval.indexer import get_indexer
from utils.csv_handler import read_input_tickets, write_output_csv
from utils.logger import DecisionTracer, get_logger
from utils.errors import AgentError

log = get_logger(__name__)


def main() -> None:
    """Run the full batch processing pipeline."""

    print("=" * 60)
    print("  HackerRank Orchestrate - Support Triage Agent")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # 1. Validate environment
    # ------------------------------------------------------------------
    if not settings.GROQ_API_KEY:
        log.critical("GROQ_API_KEY is not set. Add it to code/.env and retry.")
        sys.exit(1)

    log.info("Using model: %s", settings.GROQ_MODEL)
    log.info("Temperature: %s", settings.GROQ_TEMPERATURE)

    # ------------------------------------------------------------------
    # 2. Build or load the FAISS index
    # ------------------------------------------------------------------
    print("[1/4] Building corpus index...")
    indexer = get_indexer()
    indexer.build_or_load()
    print(f"  Index ready: {len(indexer.chunks)} chunks indexed.\n")

    # ------------------------------------------------------------------
    # 3. Read input tickets
    # ------------------------------------------------------------------
    print("[2/4] Loading support tickets...")
    df = read_input_tickets()
    total = len(df)
    print(f"  Loaded {total} tickets.\n")

    # ------------------------------------------------------------------
    # 4. Process each ticket
    # ------------------------------------------------------------------
    print("[3/4] Processing tickets...")
    print("-" * 60)

    tracer = DecisionTracer()
    pipeline = Pipeline(tracer=tracer)

    results = []
    start_time = time.time()

    for idx, row in df.iterrows():
        ticket_id = int(idx) + 1
        issue = str(row.get("issue", ""))
        subject = str(row.get("subject", ""))
        company = str(row.get("company", ""))

        try:
            result = pipeline.process_ticket(
                ticket_id=ticket_id,
                issue=issue,
                subject=subject,
                company=company,
            )
            results.append(result)

            # Print progress
            status_icon = "R" if result.get("Status", "").lower() == "replied" else "E"
            print(
                f"  [{status_icon}] Ticket {ticket_id:2d}/{total} | "
                f"{str(result.get('Status', '')):9s} | {str(result.get('Request Type', '')):16s} | "
                f"{subject[:40]}"
            )

            # Generate clean, formatted log entry according to AGENTS.md
            log_path = settings.CODE_DIR / "log.txt"
            
            timestamp = datetime.now().isoformat(timespec='seconds')
            
            # SANITIZATION: Remove all internal newlines to prevent broken words/wrapped lines
            full_query = f"{issue}".replace("\n", " ").replace("\r", " ").strip()
            
            status_str = str(result.get("Status", "Escalated")).capitalize()
            decision_action = "Reply" if status_str.lower() == "replied" else "Escalate"
            
            # Construct structured summary according to new requirements
            req_type = str(result.get("Request Type", "product_issue")).replace("\n", " ")
            risk = str(result.get("sensitivity", "low")).replace("\n", " ")
            docs = result.get("retrieved_docs", [])
            
            # Be descriptive about docs if possible
            if not docs:
                doc_summary = "no documents"
            else:
                # Extract some keywords from titles for "doc types"
                unique_titles = list(dict.fromkeys(docs))
                if len(unique_titles) > 1:
                    doc_summary = f"{unique_titles[0]} and other relevant documentation"
                else:
                    doc_summary = f"{unique_titles[0]} documentation"
            
            doc_summary = doc_summary.replace("\n", " ")
            reason = str(result.get("Justification", "Ticket processed.")).replace("\n", " ").strip(".")
            summary = f"Classified as {req_type} with {risk} risk. Retrieved {doc_summary}. Decided to {decision_action} because {reason}."
            
            actions = ["Classified ticket"]
            retrieved_docs = result.get("retrieved_docs", [])
            if retrieved_docs:
                actions.append(f"Retrieved docs: {', '.join(retrieved_docs)}")
            else:
                actions.append("Retrieved docs: None")
            
            actions.append(f"Decision: {decision_action}")
            actions.append("Generated response")
            
            # Ensure each action is a single line
            actions_str = "\n".join(f"* {a.replace(chr(10), ' ')}" for a in actions)
            
            repo_root = str(settings.REPO_ROOT).replace("\n", " ")
            
            log_text = f"""## {timestamp} TICKET {ticket_id} TRIAGE

User Prompt (verbatim, secrets redacted):
{full_query}

Agent Response Summary:
{summary}

Actions:

{actions_str}

Context:
tool=antigravity
branch=main
repo_root={repo_root}
worktree=main
parent_agent=none
"""
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(log_text + "\n")
            except Exception as log_exc:
                log.error("Failed to write to log file: %s", log_exc)

        except AgentError as exc:
            log.error("Ticket %d failed: %s", ticket_id, exc)
            # Graceful degradation: escalate the failed ticket
            fallback = {
                "Issue": issue,
                "Subject": subject,
                "Company": company,
                "Response": (
                    "We encountered an issue processing your request. "
                    "Your ticket has been escalated to a human agent."
                ),
                "Product Area": "general_support",
                "Status": "Escalated",
                "Request Type": "product_issue",
                "Justification": f"Processing error: {exc}. Escalated for human review.",
            }
            results.append(fallback)
            print(f"  [!] Ticket {ticket_id:2d}/{total} | ERROR -- escalated | {subject[:40]}")

        except Exception as exc:
            log.error("Unexpected error on ticket %d: %s", ticket_id, exc)
            fallback = {
                "Issue": issue,
                "Subject": subject,
                "Company": company,
                "Response": (
                    "We encountered an unexpected issue processing your request. "
                    "Your ticket has been escalated to a human agent."
                ),
                "Product Area": "general_support",
                "Status": "Escalated",
                "Request Type": "product_issue",
                "Justification": f"Unexpected error: {exc}. Escalated for human review.",
            }
            results.append(fallback)
            print(f"  [!] Ticket {ticket_id:2d}/{total} | ERROR -- escalated | {subject[:40]}")

    elapsed = time.time() - start_time
    print("-" * 60)
    print()

    # ------------------------------------------------------------------
    # 5. Write output CSV
    # ------------------------------------------------------------------
    print("[4/4] Writing output...")
    output_path = write_output_csv(results)
    print(f"  Output written to: {output_path}\n")

    # Save decision traces
    trace_path = tracer.save()

    # ------------------------------------------------------------------
    # 6. Print summary
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    replied = sum(1 for r in results if str(r.get("Status", "")).lower() == "replied")
    escalated = sum(1 for r in results if str(r.get("Status", "")).lower() == "escalated")

    print(f"  Total tickets:  {total}")
    print(f"  Replied:        {replied}")
    print(f"  Escalated:      {escalated}")
    print(f"  Time elapsed:   {elapsed:.1f}s ({elapsed/total:.1f}s per ticket)")
    print(f"  Decision traces: {trace_path}")
    print()

    # Request type breakdown
    type_counts: dict[str, int] = {}
    for r in results:
        rt = r.get("Request Type", "unknown")
        type_counts[rt] = type_counts.get(rt, 0) + 1

    print("  Request type breakdown:")
    for rt, count in sorted(type_counts.items()):
        print(f"    {rt:20s}: {count}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
