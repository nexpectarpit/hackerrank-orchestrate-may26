from __future__ import annotations

"""
Corpus parser -- walks data/ and converts every .md file into a Document.

Extracts YAML frontmatter (title, description, breadcrumbs, source_url)
and the markdown body. Determines the company and product_area from
the file path and breadcrumbs.

Usage (standalone test)::

    python -m corpus.parser
"""

import re
from pathlib import Path

import yaml

from config import settings
from corpus.models import Document
from utils.errors import CorpusParsingError
from utils.logger import get_logger

log = get_logger(__name__)

# Pre-compiled pattern for YAML frontmatter delimiters
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _detect_company(file_path: Path) -> str:
    """Determine the company from the file path.

    The data/ directory has top-level folders: hackerrank/, claude/, visa/.
    """
    relative = file_path.relative_to(settings.DATA_DIR)
    top_folder = relative.parts[0].lower()

    mapping = {
        "hackerrank": "HackerRank",
        "claude": "Claude",
        "visa": "Visa",
    }
    return mapping.get(top_folder, "Unknown")


def _derive_product_area(
    breadcrumbs: list[str],
    file_path: Path,
    company: str,
) -> str:
    """Derive a product_area string from breadcrumbs or folder structure.

    Priority:
    1. Last breadcrumb entry (most specific topic)
    2. Subfolder name under the company directory
    3. Fallback to "general"
    """
    if breadcrumbs:
        # Use the last breadcrumb, normalized to snake_case
        raw = breadcrumbs[-1]
        return raw.strip().lower().replace(" ", "_").replace("-", "_")

    # Fall back to the subfolder name
    relative = file_path.relative_to(settings.DATA_DIR)
    parts = relative.parts
    if len(parts) >= 3:
        return parts[1].lower().replace("-", "_")

    return "general"


def _parse_frontmatter(content: str) -> tuple[dict, str]:
    """Split a markdown file into frontmatter dict and body text.

    Returns (metadata_dict, body_string). If no frontmatter is found,
    metadata_dict will be empty and body_string will be the full content.
    """
    match = _FRONTMATTER_RE.match(content)
    if not match:
        return {}, content

    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        meta = {}

    body = content[match.end():]
    return meta, body


def parse_file(file_path: Path) -> Document:
    """Parse a single markdown file into a Document."""
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise CorpusParsingError(
            f"Cannot read {file_path}: {exc}"
        ) from exc

    meta, body = _parse_frontmatter(content)

    company = _detect_company(file_path)

    # Extract title -- prefer frontmatter, fall back to first heading
    title = meta.get("title", "")
    if not title:
        heading_match = re.search(r"^#\s+(.+)", body, re.MULTILINE)
        title = heading_match.group(1).strip() if heading_match else file_path.stem

    # Extract breadcrumbs
    breadcrumbs_raw = meta.get("breadcrumbs", [])
    if isinstance(breadcrumbs_raw, list):
        breadcrumbs = [str(b).strip() for b in breadcrumbs_raw if b]
    else:
        breadcrumbs = []

    product_area = _derive_product_area(breadcrumbs, file_path, company)

    return Document(
        file_path=str(file_path),
        company=company,
        title=title,
        description=meta.get("description", "") or "",
        breadcrumbs=breadcrumbs,
        source_url=meta.get("source_url", "") or "",
        body=body.strip(),
        product_area=product_area,
    )


def parse_corpus(data_dir: Path | None = None) -> list[Document]:
    """Walk the entire data/ directory and parse every .md file.

    Returns a list of Document objects sorted by company then file path.
    """
    root = data_dir or settings.DATA_DIR
    log.info("Parsing corpus from %s", root)

    if not root.is_dir():
        raise CorpusParsingError(f"Data directory does not exist: {root}")

    documents: list[Document] = []
    errors: list[str] = []

    for md_file in sorted(root.rglob("*.md")):
        try:
            doc = parse_file(md_file)
            documents.append(doc)
        except CorpusParsingError as exc:
            errors.append(str(exc))
            log.warning("Skipping file: %s", exc)

    # Summary
    by_company: dict[str, int] = {}
    for doc in documents:
        by_company[doc.company] = by_company.get(doc.company, 0) + 1

    log.info("Parsed %d documents total", len(documents))
    for company, count in sorted(by_company.items()):
        log.info("  %s: %d", company, count)

    if errors:
        log.warning("%d files had parsing errors", len(errors))

    return documents


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    docs = parse_corpus()

    print(f"\nTotal documents: {len(docs)}")
    print("---")

    by_company: dict[str, int] = {}
    for d in docs:
        by_company[d.company] = by_company.get(d.company, 0) + 1
    for company, count in sorted(by_company.items()):
        print(f"  {company}: {count}")

    print("\n--- Sample documents ---")
    for d in docs[:3]:
        print(f"\n  Company:      {d.company}")
        print(f"  Title:        {d.title[:80]}")
        print(f"  Product area: {d.product_area}")
        print(f"  Breadcrumbs:  {d.breadcrumbs}")
        print(f"  Body length:  {len(d.body)} chars")
        print(f"  File:         {d.file_path}")
