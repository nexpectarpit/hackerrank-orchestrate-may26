from __future__ import annotations

"""
Data models for the corpus pipeline.

Document represents a single parsed markdown article.
Chunk represents a section extracted from a Document, ready for embedding.
"""

from dataclasses import dataclass, field


@dataclass
class Document:
    """A single markdown article from the data/ corpus.

    Attributes:
        file_path: absolute path to the source .md file
        company: one of HackerRank, Claude, Visa
        title: article title from frontmatter or first heading
        description: article description from frontmatter (may be empty)
        breadcrumbs: topic hierarchy from frontmatter (e.g. ["Account Settings", "Manage Account"])
        source_url: original URL from frontmatter
        body: full markdown body text (without frontmatter)
        product_area: derived from breadcrumbs or folder structure
    """

    file_path: str
    company: str
    title: str
    body: str
    description: str = ""
    breadcrumbs: list[str] = field(default_factory=list)
    source_url: str = ""
    product_area: str = ""


@dataclass
class Chunk:
    """A section of a Document, ready for embedding and retrieval.

    Attributes:
        chunk_id: unique identifier (e.g. "hackerrank_0042_s02")
        text: the actual text content of this chunk
        company: inherited from parent Document
        title: inherited from parent Document
        section_header: the h2/h3 heading this chunk falls under
        product_area: inherited from parent Document
        source_file: file_path of the parent Document
        source_url: inherited from parent Document
        breadcrumbs: inherited from parent Document
    """

    chunk_id: str
    text: str
    company: str
    title: str
    section_header: str = ""
    product_area: str = ""
    source_file: str = ""
    source_url: str = ""
    breadcrumbs: list[str] = field(default_factory=list)
