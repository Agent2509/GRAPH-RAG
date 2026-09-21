"""Unit tests for GraphStore."""

import pytest
import tempfile
from pathlib import Path
from graph_rag.models import Entity, Relationship
from graph_rag.storage.graph_store import GraphStore


def test_graph_store_add_and_subgraph():
    gs = GraphStore()
    e1 = Entity(name="Alice", type="PERSON", description="Lead scientist")
    e2 = Entity(name="NexusLabs", type="ORGANIZATION", description="AI Lab")
    e3 = Entity(name="DeepBio", type="ORGANIZATION", description="Biotech startup")

    gs.add_entity(e1)
    gs.add_entity(e2)
    gs.add_entity(e3)

    r1 = Relationship(source="Alice", target="NexusLabs", relation_type="FOUNDED", description="Founded in 2021")
    r2 = Relationship(source="NexusLabs", target="DeepBio", relation_type="ACQUIRED", description="Acquired in 2023")

    gs.add_relationship(r1)
    gs.add_relationship(r2)

    stats = gs.get_stats()
    assert stats.node_count == 3
    assert stats.edge_count == 2

    # Test 1-hop subgraph from Alice
    sub_ents, sub_rels = gs.get_subgraph(seed_entities=["Alice"], max_hops=1)
    names = {e.name for e in sub_ents}
    assert "Alice" in names
    assert "NexusLabs" in names
    assert "DeepBio" not in names  # DeepBio is 2 hops away

    # Test 2-hop subgraph from Alice
    sub_ents_2, _ = gs.get_subgraph(seed_entities=["Alice"], max_hops=2)
    names_2 = {e.name for e in sub_ents_2}
    assert "DeepBio" in names_2

    # Shortest path
    path = gs.find_path("Alice", "DeepBio")
    assert path == ["Alice", "NexusLabs", "DeepBio"]


def test_graph_store_serialization():
    gs = GraphStore()
    gs.add_entity(Entity(name="A", type="CONCEPT"))
    gs.add_entity(Entity(name="B", type="CONCEPT"))
    gs.add_relationship(Relationship(source="A", target="B", relation_type="LINKS_TO"))

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "test_graph.json"
        gs.save_json(str(json_path))
        assert json_path.exists()

        new_gs = GraphStore()
        new_gs.load_json(str(json_path))
        assert new_gs.graph.number_of_nodes() == 2
        assert new_gs.graph.number_of_edges() == 1
