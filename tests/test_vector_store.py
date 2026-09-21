"""Unit tests for VectorStore."""

import tempfile
from pathlib import Path
from graph_rag.storage.vector_store import VectorStore


def test_vector_store_search_and_filter():
    vs = VectorStore()
    # Add two orthogonal vectors
    vs.add("item_1", [1.0, 0.0, 0.0], metadata={"item_type": "entity", "name": "AI"})
    vs.add("item_2", [0.0, 1.0, 0.0], metadata={"item_type": "chunk", "text": "Biotech"})

    assert len(vs) == 2

    # Query pointing towards item_1
    results = vs.search([0.9, 0.1, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0][0] == "item_1"
    assert results[0][2] > 0.8

    # Query with type filter
    chunk_results = vs.search([1.0, 0.0, 0.0], top_k=2, filter_type="chunk")
    assert len(chunk_results) == 1
    assert chunk_results[0][0] == "item_2"


def test_vector_store_persistence():
    vs = VectorStore()
    vs.add("v1", [0.5, 0.5], metadata={"label": "first"})

    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "vecs.json")
        vs.save(path)

        new_vs = VectorStore()
        new_vs.load(path)
        assert len(new_vs) == 1
        assert new_vs.item_ids[0] == "v1"
        assert new_vs.metadata[0]["label"] == "first"
