"""
RAG (Retrieval-Augmented Generation) for machine documentation.

Two backends (chosen automatically at import time):
  1. **FAISS + sentence-transformers**  – proper semantic vector search.
     Activate:  ``pip install predictive-maintenance-mcp[vector-search]``
  2. **TF-IDF** (scikit-learn) – keyword-based fallback, zero extra deps.

Both backends share the same public API:
  - ``get_or_build_index()``  →  ``DocumentIndex``
  - ``search_documents(query)``  →  ``list[dict]``

Design choices:
  - Persistent index cache in CACHE_DIR → re-index only when source changes
  - Paragraph-aware chunking with configurable size
  - Cosine similarity for ranking (both backends)
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import RESOURCES_DIR, CACHE_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional FAISS + sentence-transformers availability
# ---------------------------------------------------------------------------

try:
    import faiss
    from sentence_transformers import SentenceTransformer

    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False


def backend_name() -> str:
    """Return the active retrieval backend name."""
    return "faiss" if _HAS_FAISS else "tfidf"


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 128,
    source: str = "",
) -> list[dict[str, Any]]:
    """Split *text* into overlapping character-level chunks.

    Each chunk dict contains:
      - ``text``  : the chunk content
      - ``source``: originating filename / identifier
      - ``index`` : sequential chunk number
      - ``start`` / ``end`` : character offsets in the original text
    """
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size - 1)
    chunks: list[dict[str, Any]] = []
    start = 0
    idx = 0
    while start < len(text):
        end = start + chunk_size
        chunk_text_str = text[start:end]
        if chunk_text_str.strip():
            chunks.append(
                {
                    "text": chunk_text_str,
                    "source": source,
                    "index": idx,
                    "start": start,
                    "end": min(end, len(text)),
                }
            )
            idx += 1
        start += chunk_size - chunk_overlap
    return chunks


def chunk_text_by_paragraphs(
    text: str,
    max_chunk_chars: int = 1024,
    source: str = "",
) -> list[dict[str, Any]]:
    """Split on paragraph boundaries, merging short paragraphs.

    Produces semantically cleaner chunks than pure character splitting.
    """
    paragraphs = re.split(r"\n{2,}", text)
    chunks: list[dict[str, Any]] = []
    buf = ""
    offset = 0
    idx = 0
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(buf) + len(para) + 2 > max_chunk_chars and buf:
            chunks.append(
                {
                    "text": buf,
                    "source": source,
                    "index": idx,
                    "start": offset,
                    "end": offset + len(buf),
                }
            )
            idx += 1
            offset += len(buf)
            buf = para
        else:
            buf = f"{buf}\n\n{para}" if buf else para
    if buf.strip():
        chunks.append(
            {
                "text": buf,
                "source": source,
                "index": idx,
                "start": offset,
                "end": offset + len(buf),
            }
        )
    return chunks


# ---------------------------------------------------------------------------
# Index — abstract interface + two concrete backends
# ---------------------------------------------------------------------------


class DocumentIndex:
    """Unified document index with FAISS or TF-IDF backend."""

    def __init__(self) -> None:
        self._chunks: list[dict[str, Any]] = []
        self._source_hash: str = ""
        # TF-IDF backend (always available)
        self._tfidf_vectorizer = TfidfVectorizer(
            max_features=10_000,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self._tfidf_matrix = None  # sparse TF-IDF matrix
        # FAISS backend (only when deps available)
        self._faiss_index = None  # faiss.IndexFlatIP
        self._embedder = None  # SentenceTransformer instance
        self._backend: str = "tfidf"

    # -- persistence --

    @staticmethod
    def _cache_path() -> Path:
        suffix = "faiss" if _HAS_FAISS else "tfidf"
        return CACHE_DIR / f"rag_index_{suffix}.pkl"

    def save(self) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {
            "chunks": self._chunks,
            "source_hash": self._source_hash,
            "backend": self._backend,
        }
        if self._backend == "faiss" and self._faiss_index is not None:
            # Serialize FAISS index to bytes
            data["faiss_index_bytes"] = faiss.serialize_index(self._faiss_index)
        else:
            data["vectorizer"] = self._tfidf_vectorizer
            data["matrix"] = self._tfidf_matrix
        with open(self._cache_path(), "wb") as f:
            pickle.dump(data, f)
        logger.info(
            "RAG index saved (%s backend, %d chunks)", self._backend, len(self._chunks)
        )

    @classmethod
    def load(cls) -> Optional["DocumentIndex"]:
        path = cls._cache_path()
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)  # noqa: S301 – trusted local cache
            idx = cls()
            idx._chunks = data["chunks"]
            idx._source_hash = data["source_hash"]
            idx._backend = data.get("backend", "tfidf")
            if idx._backend == "faiss" and _HAS_FAISS and "faiss_index_bytes" in data:
                idx._faiss_index = faiss.deserialize_index(data["faiss_index_bytes"])
                idx._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            elif "vectorizer" in data:
                idx._tfidf_vectorizer = data["vectorizer"]
                idx._tfidf_matrix = data["matrix"]
                idx._backend = "tfidf"
            else:
                return None
            logger.info(
                "RAG index loaded (%s backend, %d chunks)",
                idx._backend,
                len(idx._chunks),
            )
            return idx
        except Exception as exc:
            logger.warning("Failed to load RAG index cache: %s", exc)
            return None

    # -- build --

    @staticmethod
    def _hash_sources(paths: list[Path]) -> str:
        """Deterministic hash of source file sizes + mtimes."""
        parts = []
        for p in sorted(paths):
            parts.append(f"{p.name}:{p.stat().st_size}:{p.stat().st_mtime}")
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]

    def build(self, documents: list[dict[str, Any]]) -> None:
        """Build index from ``{"text": ..., "source": ...}`` dicts."""
        self._chunks = []
        for doc in documents:
            self._chunks.extend(
                chunk_text_by_paragraphs(
                    doc["text"],
                    max_chunk_chars=1024,
                    source=doc.get("source", ""),
                )
            )
        if not self._chunks:
            logger.warning("No chunks produced – index is empty")
            return

        texts = [c["text"] for c in self._chunks]

        if _HAS_FAISS:
            self._backend = "faiss"
            logger.info("Building FAISS index with sentence-transformers …")
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = self._embedder.encode(
                texts, show_progress_bar=False, normalize_embeddings=True
            )
            dim = embeddings.shape[1]
            self._faiss_index = faiss.IndexFlatIP(
                dim
            )  # inner-product on L2-normalised = cosine
            self._faiss_index.add(embeddings.astype(np.float32))
            # Also build TF-IDF as hybrid fallback
            self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(texts)
            logger.info("FAISS index built: %d chunks, dim=%d", len(self._chunks), dim)
        else:
            self._backend = "tfidf"
            self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(texts)
            logger.info(
                "TF-IDF index built: %d chunks, %d features",
                len(self._chunks),
                self._tfidf_matrix.shape[1],
            )

    # -- query --

    def query(
        self, question: str, top_k: int = 5, min_score: float = 0.05
    ) -> list[dict[str, Any]]:
        """Return the *top_k* most relevant chunks for *question*."""
        if not self._chunks:
            return []

        if (
            self._backend == "faiss"
            and self._faiss_index is not None
            and self._embedder is not None
        ):
            return self._query_faiss(question, top_k, min_score)
        return self._query_tfidf(question, top_k, min_score)

    def _query_faiss(
        self, question: str, top_k: int, min_score: float
    ) -> list[dict[str, Any]]:
        q_emb = self._embedder.encode([question], normalize_embeddings=True).astype(
            np.float32
        )
        scores, indices = self._faiss_index.search(q_emb, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score < min_score:
                continue
            results.append(
                {
                    "text": self._chunks[idx]["text"],
                    "source": self._chunks[idx]["source"],
                    "chunk_index": self._chunks[idx]["index"],
                    "score": round(float(score), 4),
                }
            )
        return results

    def _query_tfidf(
        self, question: str, top_k: int, min_score: float
    ) -> list[dict[str, Any]]:
        if self._tfidf_matrix is None:
            return []
        q_vec = self._tfidf_vectorizer.transform([question])
        scores = cosine_similarity(q_vec, self._tfidf_matrix).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in top_idx:
            if scores[i] < min_score:
                continue
            results.append(
                {
                    "text": self._chunks[i]["text"],
                    "source": self._chunks[i]["source"],
                    "chunk_index": self._chunks[i]["index"],
                    "score": round(float(scores[i]), 4),
                }
            )
        return results

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)


# ---------------------------------------------------------------------------
# High-level helpers (used by the MCP server)
# ---------------------------------------------------------------------------

_index: Optional[DocumentIndex] = None


def _collect_documents() -> tuple[list[dict[str, Any]], list[Path]]:
    """Gather all readable documents under ``resources/``."""
    docs: list[dict[str, Any]] = []
    paths: list[Path] = []

    manuals_dir = RESOURCES_DIR / "machine_manuals"
    catalogs_dir = RESOURCES_DIR / "bearing_catalogs"

    for folder in [manuals_dir, catalogs_dir]:
        if not folder.exists():
            continue
        for fpath in folder.iterdir():
            if fpath.suffix.lower() == ".pdf":
                try:
                    from .document_reader import extract_text_from_pdf

                    text = extract_text_from_pdf(fpath, max_pages=50)
                    docs.append({"text": text, "source": fpath.name})
                    paths.append(fpath)
                except Exception as exc:
                    logger.warning("Skipping %s: %s", fpath.name, exc)
            elif fpath.suffix.lower() in {".txt", ".md"}:
                try:
                    text = fpath.read_text(encoding="utf-8")
                    docs.append({"text": text, "source": fpath.name})
                    paths.append(fpath)
                except Exception as exc:
                    logger.warning("Skipping %s: %s", fpath.name, exc)
            elif fpath.suffix.lower() == ".json":
                try:
                    text = fpath.read_text(encoding="utf-8")
                    docs.append({"text": text, "source": fpath.name})
                    paths.append(fpath)
                except Exception as exc:
                    logger.warning("Skipping %s: %s", fpath.name, exc)

    return docs, paths


def get_or_build_index(force_rebuild: bool = False) -> DocumentIndex:
    """Return (and lazily build / cache) the global RAG index."""
    global _index

    docs, paths = _collect_documents()
    current_hash = DocumentIndex._hash_sources(paths) if paths else "empty"

    # Try returning cached index
    if not force_rebuild and _index is not None and _index._source_hash == current_hash:
        return _index

    # Try loading from disk
    if not force_rebuild:
        loaded = DocumentIndex.load()
        if loaded and loaded._source_hash == current_hash:
            _index = loaded
            return _index

    # Rebuild
    logger.info("Building RAG index from %d documents …", len(docs))
    _index = DocumentIndex()
    _index.build(docs)
    _index._source_hash = current_hash
    _index.save()
    return _index


def search_documents(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Public entry point: search all indexed documentation."""
    idx = get_or_build_index()
    return idx.query(query, top_k=top_k)
