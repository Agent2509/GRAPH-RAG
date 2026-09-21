"""Graph RAG: A 100% free multiplatform Graph RAG framework from scratch."""

from graph_rag.models import (
    TextChunk,
    Entity,
    Relationship,
    Community,
    RetrievalResult,
    GraphStats,
)
from graph_rag.engine import GraphRAG
from graph_rag.llm.base import BaseLLM, BaseEmbedding
from graph_rag.llm.groq_client import GroqLLM
from graph_rag.llm.embeddings import FastEmbedEmbedding
from graph_rag.llm.mock import MockLLM, MockEmbedding

__version__ = "0.1.0"
__all__ = [
    "GraphRAG",
    "TextChunk",
    "Entity",
    "Relationship",
    "Community",
    "RetrievalResult",
    "GraphStats",
    "BaseLLM",
    "BaseEmbedding",
    "GroqLLM",
    "FastEmbedEmbedding",
    "MockLLM",
    "MockEmbedding",
]
