"""Data models for Graph RAG using Pydantic."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TextChunk(BaseModel):
    """Represents a text chunk extracted from an ingested document."""
    id: str
    text: str
    doc_id: str
    chunk_index: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    token_count: Optional[int] = None


class Entity(BaseModel):
    """Represents a knowledge graph node (entity)."""
    name: str
    type: str = "CONCEPT"
    description: str = ""
    source_chunk_ids: List[str] = Field(default_factory=list)


class Relationship(BaseModel):
    """Represents a directed or undirected knowledge graph edge (relationship)."""
    source: str
    target: str
    relation_type: str = "RELATED_TO"
    description: str = ""
    weight: float = 1.0
    source_chunk_ids: List[str] = Field(default_factory=list)


class Community(BaseModel):
    """Represents a detected entity cluster/community in the knowledge graph."""
    id: int
    level: int = 0
    entity_names: List[str] = Field(default_factory=list)
    title: str = ""
    summary: str = ""
    findings: List[str] = Field(default_factory=list)
    rating: float = 1.0


class RetrievalResult(BaseModel):
    """Represents the complete response from a Graph RAG query."""
    query: str
    answer: str
    search_mode: str = "local"
    cited_entities: List[Entity] = Field(default_factory=list)
    cited_relations: List[Relationship] = Field(default_factory=list)
    cited_chunks: List[TextChunk] = Field(default_factory=list)
    graph_context_text: str = ""


class GraphStats(BaseModel):
    """Summary statistics of the knowledge graph."""
    node_count: int = 0
    edge_count: int = 0
    community_count: int = 0
    density: float = 0.0
    top_entities: List[Dict[str, Any]] = Field(default_factory=list)
