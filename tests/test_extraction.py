"""Unit tests for entity and relationship resolution."""

import pytest
from graph_rag.models import Entity, Relationship
from graph_rag.extraction.resolver import EntityResolver


def test_resolver_name_normalization():
    resolver = EntityResolver()
    assert resolver.normalize_name("  Google  ") == "Google"
    assert resolver.normalize_name('"Apple"') == "Apple"
    assert resolver.normalize_name("The Microsoft") == "Microsoft"


def test_resolver_entity_deduplication():
    resolver = EntityResolver()
    raw_entities = [
        Entity(name="OpenAI", type="CONCEPT", description="AI lab", source_chunk_ids=["c1"]),
        Entity(name="openai", type="ORGANIZATION", description="Creator of GPT", source_chunk_ids=["c2"]),
    ]
    resolved = resolver.resolve_entities(raw_entities)
    assert len(resolved) == 1
    openai_ent = list(resolved.values())[0]
    assert openai_ent.type == "ORGANIZATION"
    assert "AI lab" in openai_ent.description
    assert "Creator of GPT" in openai_ent.description
    assert set(openai_ent.source_chunk_ids) == {"c1", "c2"}


def test_resolver_relationship_deduplication():
    resolver = EntityResolver()
    raw_entities = [
        Entity(name="Alice", type="PERSON"),
        Entity(name="NexusLabs", type="ORGANIZATION"),
    ]
    resolved_ents = resolver.resolve_entities(raw_entities)

    raw_rels = [
        Relationship(source="Alice", target="NexusLabs", relation_type="FOUNDED", description="Founded in 2021", weight=1.0, source_chunk_ids=["c1"]),
        Relationship(source="alice", target="nexuslabs", relation_type="FOUNDED", description="Primary founder", weight=1.0, source_chunk_ids=["c2"]),
    ]
    resolved_rels = resolver.resolve_relationships(raw_rels, resolved_ents)

    assert len(resolved_rels) == 1
    assert resolved_rels[0].source == "Alice"
    assert resolved_rels[0].target == "NexusLabs"
    assert resolved_rels[0].weight == 2.0
    assert set(resolved_rels[0].source_chunk_ids) == {"c1", "c2"}


def test_extract_from_batch():
    from graph_rag.llm.mock import MockLLM
    from graph_rag.extraction.extractor import GraphExtractor
    from graph_rag.models import TextChunk

    llm = MockLLM()
    extractor = GraphExtractor(llm)

    chunks = [
        TextChunk(id="c1", text="Alice founded NexusLabs in Berlin.", doc_id="d1", chunk_index=0),
        TextChunk(id="c2", text="NexusLabs partners with DeepBio on clinical genomics.", doc_id="d1", chunk_index=1),
    ]

    ents, rels = extractor.extract_from_batch(chunks)
    assert len(ents) > 0
    assert len(rels) > 0
    # Ensure source_chunk_ids were mapped to batch
    for e in ents:
        assert len(e.source_chunk_ids) > 0

