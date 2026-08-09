"""
Tests for the RAG (Retrieval-Augmented Generation) module.

Covers:
- Character-level chunking (chunk_text)
- Paragraph-aware chunking (chunk_text_by_paragraphs)
- DocumentIndex (TF-IDF backend — always available)
- Index build, query, save/load round-trip
- Edge cases: empty text, single-chunk docs, min_score filtering
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from predictive_maintenance_mcp.rag import (
    chunk_text,
    chunk_text_by_paragraphs,
    DocumentIndex,
    backend_name,
)

# ── chunk_text ──────────────────────────────────────────────────────────────


class TestChunkText:

    def test_basic_chunking(self):
        text = "A" * 600
        chunks = chunk_text(text, chunk_size=256, chunk_overlap=64)
        assert len(chunks) >= 2
        assert all(c["text"] for c in chunks)
        assert chunks[0]["start"] == 0
        assert chunks[-1]["end"] <= 600

    def test_empty_text(self):
        assert chunk_text("", chunk_size=256) == []

    def test_whitespace_only(self):
        assert chunk_text("   \n\n  ", chunk_size=256) == []

    def test_text_shorter_than_chunk_size(self):
        chunks = chunk_text("short text", chunk_size=256)
        assert len(chunks) == 1
        assert chunks[0]["text"] == "short text"
        assert chunks[0]["start"] == 0
        assert chunks[0]["end"] == 10

    def test_overlap_produces_more_chunks(self):
        text = "X" * 500
        no_overlap = chunk_text(text, chunk_size=200, chunk_overlap=0)
        with_overlap = chunk_text(text, chunk_size=200, chunk_overlap=50)
        assert len(with_overlap) > len(no_overlap)

    def test_chunk_metadata_fields(self):
        chunks = chunk_text("hello world " * 50, chunk_size=100, source="test.pdf")
        for c in chunks:
            assert "text" in c
            assert "source" in c
            assert "index" in c
            assert "start" in c
            assert "end" in c
            assert c["source"] == "test.pdf"

    def test_chunk_indices_sequential(self):
        chunks = chunk_text("A" * 1000, chunk_size=200, chunk_overlap=50)
        indices = [c["index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_overlap_content_correct(self):
        text = "ABCDEFGHIJKLMNOPQRST" * 20  # 400 chars
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
        # End of chunk N should overlap with start of chunk N+1
        for i in range(len(chunks) - 1):
            end_of_current = chunks[i]["text"][-20:]
            start_of_next = chunks[i + 1]["text"][:20]
            assert end_of_current == start_of_next


# ── chunk_text_by_paragraphs ───────────────────────────────────────────────


class TestChunkByParagraphs:

    def test_basic_paragraphs(self):
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = chunk_text_by_paragraphs(text, max_chunk_chars=1024)
        # All three paragraphs fit in one chunk
        assert len(chunks) == 1
        assert "Paragraph one." in chunks[0]["text"]
        assert "Paragraph three." in chunks[0]["text"]

    def test_split_when_exceeds_max(self):
        para = "A" * 100
        text = f"{para}\n\n{para}\n\n{para}"
        chunks = chunk_text_by_paragraphs(text, max_chunk_chars=150)
        assert len(chunks) >= 2

    def test_empty_text(self):
        assert chunk_text_by_paragraphs("") == []

    def test_single_paragraph(self):
        chunks = chunk_text_by_paragraphs(
            "Single paragraph only.", max_chunk_chars=1024
        )
        assert len(chunks) == 1
        assert chunks[0]["text"] == "Single paragraph only."

    def test_source_propagated(self):
        chunks = chunk_text_by_paragraphs("Para 1\n\nPara 2", source="manual.pdf")
        for c in chunks:
            assert c["source"] == "manual.pdf"

    def test_blank_paragraphs_ignored(self):
        text = "Content\n\n\n\n\n\nMore content"
        chunks = chunk_text_by_paragraphs(text, max_chunk_chars=1024)
        assert len(chunks) == 1
        assert "Content" in chunks[0]["text"]
        assert "More content" in chunks[0]["text"]


# ── DocumentIndex (TF-IDF backend) ─────────────────────────────────────────


@pytest.fixture(autouse=True)
def _force_tfidf(monkeypatch):
    """Force TF-IDF backend even if FAISS is installed."""
    monkeypatch.setattr("predictive_maintenance_mcp.rag._HAS_FAISS", False)


class TestDocumentIndexTFIDF:

    def _build_index(self, docs=None):
        idx = DocumentIndex()
        if docs is None:
            docs = [
                {
                    "text": "bearing fault diagnosis vibration analysis",
                    "source": "doc1.pdf",
                },
                {"text": "gearbox oil temperature monitoring", "source": "doc2.pdf"},
                {
                    "text": "motor current signature analysis electrical fault",
                    "source": "doc3.pdf",
                },
            ]
        idx.build(docs)
        return idx

    def test_build_creates_chunks(self):
        idx = self._build_index()
        assert idx.num_chunks > 0

    def test_build_creates_tfidf_matrix(self):
        idx = self._build_index()
        assert idx._tfidf_matrix is not None

    def test_query_returns_results(self):
        idx = self._build_index()
        results = idx.query("bearing fault", top_k=3)
        assert len(results) > 0
        assert all("text" in r for r in results)
        assert all("score" in r for r in results)
        assert all("source" in r for r in results)

    def test_query_relevance_ranking(self):
        idx = self._build_index()
        results = idx.query("bearing vibration", top_k=3)
        # First result should be from doc1 (bearing + vibration)
        assert results[0]["source"] == "doc1.pdf"

    def test_query_scores_descending(self):
        idx = self._build_index()
        results = idx.query("fault diagnosis", top_k=3)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_query_min_score_filtering(self):
        idx = self._build_index()
        # Very high min_score should filter out most results
        results = idx.query("bearing", top_k=3, min_score=0.99)
        assert len(results) <= 1

    def test_query_empty_index(self):
        idx = DocumentIndex()
        idx._chunks = []
        results = idx.query("anything")
        assert results == []

    def test_build_empty_documents(self):
        idx = DocumentIndex()
        idx.build([])
        assert idx.num_chunks == 0
        assert idx.query("test") == []

    def test_build_whitespace_documents(self):
        idx = DocumentIndex()
        idx.build([{"text": "   \n\n  ", "source": "empty.txt"}])
        # Whitespace-only chunks should be skipped
        assert idx.num_chunks == 0

    def test_save_load_roundtrip(self, tmp_path):
        idx = self._build_index()
        idx._source_hash = "test_hash_123"

        # Patch cache path to use tmp_path
        cache_file = tmp_path / "test_index.pkl"
        with patch.object(DocumentIndex, "_cache_path", return_value=cache_file):
            idx.save()
            assert cache_file.exists()

            loaded = DocumentIndex.load()
            assert loaded is not None
            assert loaded._source_hash == "test_hash_123"
            assert loaded.num_chunks == idx.num_chunks

            # Loaded index should still work for queries
            results = loaded.query("bearing fault", top_k=2)
            assert len(results) > 0

    def test_load_nonexistent_cache(self, tmp_path):
        cache_file = tmp_path / "nonexistent.pkl"
        with patch.object(DocumentIndex, "_cache_path", return_value=cache_file):
            assert DocumentIndex.load() is None

    def test_num_chunks_property(self):
        idx = self._build_index()
        assert isinstance(idx.num_chunks, int)
        assert idx.num_chunks > 0


# ── backend_name ───────────────────────────────────────────────────────────


def test_backend_name_returns_string():
    name = backend_name()
    assert name in ("faiss", "tfidf")
