"""Unit tests for text chunker and parsers."""

import pytest
from graph_rag.chunking.chunker import TextChunker
from graph_rag.chunking.parsers import parse_bytes


def test_chunker_basic():
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    text = "Hello world. " * 20
    chunks = chunker.chunk_text(text, doc_id="test_doc")

    assert len(chunks) > 1
    assert chunks[0].doc_id == "test_doc"
    assert chunks[0].id == "test_doc_chunk_0"
    for c in chunks:
        assert len(c.text) > 0


def test_chunker_small_text():
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    text = "Short text."
    chunks = chunker.chunk_text(text, doc_id="short")

    assert len(chunks) == 1
    assert chunks[0].text == "Short text."


def test_chunker_overlap_validation():
    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, chunk_overlap=100)


def test_parse_bytes_text():
    raw_bytes = b"Sample text content from uploaded file."
    result = parse_bytes(raw_bytes, "notes.txt")
    assert "Sample text content" in result
