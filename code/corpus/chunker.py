from __future__ import annotations

"""
Chunker -- splits parsed Documents into Chunks by heading boundaries.

Each chunk corresponds to one h2 or h3 section of an article, keeping
the section header as metadata. Very short trailing sections are merged
into the previous chunk to avoid noise.

Usage (standalone test)::

    python -m corpus.chunker
"""

import re
from corpus.models import Document, Chunk
from utils.logger import get_logger

log = get_logger(__name__)

# Minimum chunk length (characters) -- sections shorter than this are
# merged into the previous chunk to reduce noise in retrieval.
_MIN_CHUNK_LENGTH = 80

# Pattern that matches markdown headings at level 1, 2, or 3.
_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)


def _normalize_text(text: str) -> str:
    """Light normalization: collapse excessive blank lines."""
    # Replace 3+ consecutive newlines with 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_document(doc: Document, doc_index: int = 0) -> list[Chunk]:
    """Split a single Document into a list of Chunks.

    Strategy:
    1. Find all h1/h2/h3 headings and their positions.
    2. Split the body at each heading boundary.
    3. Each section becomes a chunk with the heading as section_header.
    4. Sections shorter than _MIN_CHUNK_LENGTH are merged into the
       previous chunk to avoid low-signal fragments.
    5. If no headings are found, the entire body becomes one chunk.
    """
    body = doc.body
    if not body.strip():
        return []

    # Find all heading positions
    headings = list(_HEADING_RE.finditer(body))

    if not headings:
        # No headings -- entire body is one chunk
        text = _normalize_text(body)
        if not text:
            return []
        return [
            Chunk(
                chunk_id=f"{doc.company.lower()}_{doc_index:04d}_s00",
                text=f"{doc.title}\n\n{text}",
                company=doc.company,
                title=doc.title,
                section_header=doc.title,
                product_area=doc.product_area,
                source_file=doc.file_path,
                source_url=doc.source_url,
                breadcrumbs=list(doc.breadcrumbs),
            )
        ]

    # Build raw sections: (header_text, body_text) pairs
    raw_sections: list[tuple[str, str]] = []

    # Text before the first heading (preamble)
    preamble = body[: headings[0].start()].strip()
    if preamble:
        raw_sections.append((doc.title, preamble))

    # Each heading section
    for i, match in enumerate(headings):
        header_text = match.group(2).strip()
        start = match.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(body)
        section_body = body[start:end].strip()
        raw_sections.append((header_text, section_body))

    # Merge short sections into previous
    merged: list[tuple[str, str]] = []
    for header, text in raw_sections:
        if merged and len(text) < _MIN_CHUNK_LENGTH:
            prev_header, prev_text = merged[-1]
            merged[-1] = (prev_header, f"{prev_text}\n\n## {header}\n{text}")
        else:
            merged.append((header, text))

    # Convert to Chunk objects
    chunks: list[Chunk] = []
    for section_idx, (header, text) in enumerate(merged):
        normalized = _normalize_text(text)
        if not normalized:
            continue

        # Prepend the article title to give context
        full_text = f"{doc.title} -- {header}\n\n{normalized}"

        chunk = Chunk(
            chunk_id=f"{doc.company.lower()}_{doc_index:04d}_s{section_idx:02d}",
            text=full_text,
            company=doc.company,
            title=doc.title,
            section_header=header,
            product_area=doc.product_area,
            source_file=doc.file_path,
            source_url=doc.source_url,
            breadcrumbs=list(doc.breadcrumbs),
        )
        chunks.append(chunk)

    return chunks


def chunk_corpus(documents: list[Document]) -> list[Chunk]:
    """Chunk all documents in the corpus.

    Returns a flat list of all chunks with unique chunk_ids.
    """
    all_chunks: list[Chunk] = []

    for doc_idx, doc in enumerate(documents):
        try:
            chunks = chunk_document(doc, doc_index=doc_idx)
            all_chunks.extend(chunks)
        except Exception as exc:
            log.warning(
                "Failed to chunk document %s: %s", doc.file_path, exc
            )

    log.info("Created %d chunks from %d documents", len(all_chunks), len(documents))

    # Per-company breakdown
    by_company: dict[str, int] = {}
    for c in all_chunks:
        by_company[c.company] = by_company.get(c.company, 0) + 1
    for company, count in sorted(by_company.items()):
        log.info("  %s: %d chunks", company, count)

    return all_chunks


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from corpus.parser import parse_corpus

    docs = parse_corpus()
    chunks = chunk_corpus(docs)

    print(f"\nTotal chunks: {len(chunks)}")
    print("---")

    by_company: dict[str, int] = {}
    for c in chunks:
        by_company[c.company] = by_company.get(c.company, 0) + 1
    for company, count in sorted(by_company.items()):
        print(f"  {company}: {count}")

    # Show avg chunk length
    if chunks:
        avg_len = sum(len(c.text) for c in chunks) / len(chunks)
        print(f"\nAvg chunk length: {avg_len:.0f} chars")

    print("\n--- Sample chunks ---")
    for c in chunks[:3]:
        print(f"\n  ID:             {c.chunk_id}")
        print(f"  Company:        {c.company}")
        print(f"  Title:          {c.title[:60]}")
        print(f"  Section header: {c.section_header[:60]}")
        print(f"  Product area:   {c.product_area}")
        print(f"  Text preview:   {c.text[:150]}...")
        print(f"  Text length:    {len(c.text)} chars")
