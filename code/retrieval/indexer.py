from __future__ import annotations

import json
from typing import Any, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import settings
from corpus.chunker import chunk_corpus
from corpus.models import Chunk
from corpus.parser import parse_corpus
from utils.errors import IndexBuildError
from utils.logger import get_logger

log = get_logger(__name__)


class Indexer:
    """Manages the FAISS index and sentence embeddings for chunks."""

    def __init__(self) -> None:
        # Load embedding model lazily or immediately
        log.info("Loading embedding model: %s", settings.EMBEDDING_MODEL)
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.index: Optional[faiss.IndexFlatL2] = None
        self.chunks: List[Chunk] = []

    def build_or_load(self, force_rebuild: bool = False) -> None:
        """Load the FAISS index from disk, or build it if missing/forced."""
        index_path = settings.INDEX_DIR / "faiss.index"
        meta_path = settings.INDEX_DIR / "chunks.json"

        if not force_rebuild and index_path.exists() and meta_path.exists():
            log.info("Loading existing FAISS index from %s", settings.INDEX_DIR)
            try:
                self.index = faiss.read_index(str(index_path))
                with open(meta_path, "r", encoding="utf-8") as f:
                    chunk_dicts = json.load(f)
                    self.chunks = [Chunk(**c) for c in chunk_dicts]
                log.info("Loaded index with %d chunks", len(self.chunks))
                return
            except Exception as e:
                log.warning("Failed to load existing index, rebuilding: %s", e)

        log.info("Building new FAISS index...")
        docs = parse_corpus()
        self.chunks = chunk_corpus(docs)

        if not self.chunks:
            raise IndexBuildError("No chunks created from corpus")

        texts = [c.text for c in self.chunks]

        log.info("Computing embeddings for %d chunks...", len(texts))
        try:
            # show_progress_bar is helpful for local execution visibility
            embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)
        except Exception as e:
            raise IndexBuildError(f"Failed to compute embeddings: {e}") from e

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

        settings.INDEX_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump([c.__dict__ for c in self.chunks], f, ensure_ascii=False)

        log.info("FAISS index built and saved to %s", settings.INDEX_DIR)


# Singleton instance
_indexer: Optional[Indexer] = None


def get_indexer() -> Indexer:
    global _indexer
    if _indexer is None:
        _indexer = Indexer()
    return _indexer


if __name__ == "__main__":
    idx = get_indexer()
    idx.build_or_load(force_rebuild=True)
