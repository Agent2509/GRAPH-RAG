"""Unified Graph RAG pipeline engine."""

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from graph_rag.models import (
    Entity,
    Relationship,
    TextChunk,
    Community,
    RetrievalResult,
    GraphStats,
)
from graph_rag.llm.base import BaseLLM, BaseEmbedding
from graph_rag.llm.groq_client import GroqLLM
from graph_rag.llm.embeddings import FastEmbedEmbedding
from graph_rag.chunking.chunker import TextChunker
from graph_rag.chunking.parsers import parse_file, parse_bytes
from graph_rag.extraction.extractor import GraphExtractor
from graph_rag.extraction.resolver import EntityResolver
from graph_rag.storage.graph_store import GraphStore
from graph_rag.storage.vector_store import VectorStore
from graph_rag.storage.community_store import CommunityStore
from graph_rag.clustering.communities import CommunityDetector
from graph_rag.clustering.summarizer import CommunitySummarizer
from graph_rag.retrieval.local_search import LocalSearch
from graph_rag.retrieval.global_search import GlobalSearch
from graph_rag.visualization.visualizer import GraphVisualizer


class GraphRAG:
    """End-to-end Knowledge Graph RAG orchestrator."""

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        embedding_model: Optional[BaseEmbedding] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        groq_api_key: Optional[str] = None,
        groq_model: Optional[str] = None,
    ):
        # Initialize LLM
        if llm is not None:
            self.llm = llm
        else:
            self.llm = GroqLLM(api_key=groq_api_key, model=groq_model)

        # Initialize Embedding
        if embedding_model is not None:
            self.embedding = embedding_model
        else:
            self.embedding = FastEmbedEmbedding()

        # Core components
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.extractor = GraphExtractor(self.llm)
        self.resolver = EntityResolver()

        # Storage
        self.graph_store = GraphStore()
        self.vector_store = VectorStore()
        self.community_store = CommunityStore()
        self.chunks_map: Dict[str, TextChunk] = {}

        # Clustering & Search
        self.community_detector = CommunityDetector()
        self.community_summarizer = CommunitySummarizer(self.llm)
        self.visualizer = GraphVisualizer()

        # Search interfaces (lazily instantiated after indexing)
        self._init_search_engines()

    def _init_search_engines(self) -> None:
        """Initialize local and global search engines."""
        self.local_search = LocalSearch(
            graph_store=self.graph_store,
            vector_store=self.vector_store,
            chunks_map=self.chunks_map,
            llm=self.llm,
            embedding_model=self.embedding,
        )
        self.global_search = GlobalSearch(
            community_store=self.community_store,
            llm=self.llm,
            embedding_model=self.embedding,
            graph_store=self.graph_store,
        )

    def add_text(
        self,
        text: str,
        doc_id: str = "doc",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[TextChunk]:
        """Chunk and stage raw text for indexing."""
        chunks = self.chunker.chunk_text(text, doc_id=doc_id, metadata=metadata)
        for c in chunks:
            self.chunks_map[c.id] = c
        return chunks

    def add_file(self, file_path: Union[str, Path]) -> List[TextChunk]:
        """Extract text from file and stage chunks."""
        path = Path(file_path)
        text = parse_file(path)
        return self.add_text(text, doc_id=path.stem, metadata={"source": str(path.name)})

    def add_bytes(self, content: bytes, filename: str) -> List[TextChunk]:
        """Parse in-memory uploaded file bytes and stage chunks."""
        text = parse_bytes(content, filename)
        doc_id = Path(filename).stem
        return self.add_text(text, doc_id=doc_id, metadata={"source": filename})

    def build_index(
        self,
        batch_size: int = 3,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        cache_dir: Optional[str] = "./data/active_index",
    ) -> Dict[str, Any]:
        """Run the complete Graph RAG knowledge extraction and indexing pipeline with disk checkpointing."""
        total_chunks = len(self.chunks_map)
        if total_chunks == 0:
            return {"status": "empty", "message": "No chunks available to index"}

        def report(msg: str, progress: float):
            if progress_callback:
                progress_callback(msg, progress)

        cache_path = None
        batch_cache: Dict[str, Any] = {}
        if cache_dir:
            c_dir = Path(cache_dir)
            c_dir.mkdir(parents=True, exist_ok=True)
            cache_path = c_dir / "extraction_cache.json"
            if cache_path.exists():
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        batch_cache = json.load(f)
                except Exception:
                    batch_cache = {}

        def save_cache():
            if cache_path:
                try:
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(batch_cache, f)
                except Exception:
                    pass

        # 1. Extract Entities & Relations in batches for 3x-4x speedup
        report("Extracting knowledge graph triples in multi-chunk batches...", 0.1)
        raw_entities: List[Entity] = []
        raw_relations: List[Relationship] = []

        chunk_items = list(self.chunks_map.values())
        b_size = max(1, batch_size)
        batches = [chunk_items[i:i + b_size] for i in range(0, len(chunk_items), b_size)]
        total_batches = len(batches)

        extraction_error = None
        completed_batches = 0

        for b_idx, batch in enumerate(batches):
            batch_key = hashlib.sha256("".join(c.id + c.text for c in batch).encode("utf-8")).hexdigest()
            if batch_key in batch_cache:
                cached = batch_cache[batch_key]
                ents = [Entity(**e) for e in cached.get("entities", [])]
                rels = [Relationship(**r) for r in cached.get("relationships", [])]
            else:
                try:
                    ents, rels = self.extractor.extract_from_batch(batch)
                    batch_cache[batch_key] = {
                        "entities": [e.model_dump() for e in ents],
                        "relationships": [r.model_dump() for r in rels],
                    }
                    if (b_idx + 1) % 5 == 0:
                        save_cache()
                except Exception as e:
                    extraction_error = e
                    save_cache()
                    break

            raw_entities.extend(ents)
            raw_relations.extend(rels)
            completed_batches = b_idx + 1
            step_progress = 0.1 + 0.45 * (completed_batches / total_batches)
            processed_chunks = min(completed_batches * b_size, total_chunks)
            report(
                f"Extracted batch {completed_batches}/{total_batches} (chunks 1–{processed_chunks}/{total_chunks} | {len(raw_entities)} entities, {len(raw_relations)} relations)",
                step_progress,
            )

        save_cache()

        if len(raw_entities) == 0:
            if extraction_error:
                raise extraction_error
            return {"status": "empty", "message": "No entities extracted"}

        # 2. Entity Resolution & Deduplication
        report("Resolving and deduplicating entities & relations...", 0.55)
        resolved_entities = self.resolver.resolve_entities(raw_entities)
        resolved_relations = self.resolver.resolve_relationships(raw_relations, resolved_entities)

        # 3. Populate Knowledge Graph
        report("Populating Knowledge Graph...", 0.65)
        for ent in resolved_entities.values():
            self.graph_store.add_entity(ent)
        for rel in resolved_relations:
            self.graph_store.add_relationship(rel)

        # 4. Compute and index Vector Embeddings
        report("Generating vector embeddings...", 0.75)
        processed_chunks_count = min(completed_batches * b_size, total_chunks)
        indexed_chunk_items = chunk_items[:processed_chunks_count]
        chunk_texts = [c.text for c in indexed_chunk_items]
        chunk_ids = [c.id for c in indexed_chunk_items]
        chunk_metas = [{"item_type": "chunk", "doc_id": c.doc_id, "chunk_id": c.id} for c in indexed_chunk_items]
        chunk_vecs = self.embedding.embed_batch(chunk_texts)
        self.vector_store.add_batch(chunk_ids, chunk_vecs, chunk_metas)

        # Embed entities
        ent_list = list(resolved_entities.values())
        if ent_list:
            ent_texts = [f"{e.name}: {e.description}" for e in ent_list]
            ent_ids = [f"entity_{e.name}" for e in ent_list]
            ent_metas = [{"item_type": "entity", "name": e.name, "type": e.type} for e in ent_list]
            ent_vecs = self.embedding.embed_batch(ent_texts)
            self.vector_store.add_batch(ent_ids, ent_vecs, ent_metas)

        # 5. Community Detection & Summarization
        report("Detecting hierarchical graph communities...", 0.85)
        communities = self.community_detector.detect_communities(self.graph_store)

        report("Generating community analytical summaries...", 0.92)
        try:
            self.community_summarizer.summarize_all(
                communities=communities,
                graph_store=self.graph_store,
                community_store=self.community_store,
            )
        except Exception:
            pass

        # Refresh search engines with populated data
        self._init_search_engines()

        stats = self.get_stats()
        if extraction_error:
            report(f"Paused at batch {completed_batches}/{total_batches}. Graph partially indexed!", 1.0)
            return {
                "status": "partial",
                "message": f"Partially indexed {completed_batches}/{total_batches} batches ({stats.node_count} entities, {stats.edge_count} relations). Remaining can be resumed anytime.",
                "completed_batches": completed_batches,
                "total_batches": total_batches,
                "chunks_indexed": processed_chunks_count,
                "entities_count": stats.node_count,
                "relations_count": stats.edge_count,
                "communities_count": stats.community_count,
                "error": str(extraction_error),
            }

        report("Indexing complete!", 1.0)

        return {
            "status": "success",
            "chunks_indexed": total_chunks,
            "entities_count": stats.node_count,
            "relations_count": stats.edge_count,
            "communities_count": stats.community_count,
        }

    def query(self, question: str, mode: str = "local", **kwargs) -> RetrievalResult:
        """Query the Graph RAG engine in 'local' or 'global' search mode."""
        clean_mode = mode.lower().strip()
        if clean_mode == "global":
            return self.global_search.search(question, **kwargs)
        else:
            return self.local_search.search(question, **kwargs)

    def get_graph_html(
        self,
        highlight_nodes: Optional[List[str]] = None,
        filter_community_id: Optional[int] = None,
        min_degree: int = 0,
        color_by: str = "type",
        node_limit: int = 500,
    ) -> str:
        """Render the knowledge graph as interactive HTML."""
        return self.visualizer.generate_html(
            graph_store=self.graph_store,
            highlight_nodes=highlight_nodes,
            filter_community_id=filter_community_id,
            min_degree=min_degree,
            color_by=color_by,
            node_limit=node_limit,
        )

    def get_stats(self) -> GraphStats:
        """Retrieve graph structural and community metrics."""
        stats = self.graph_store.get_stats()
        stats.community_count = len(self.community_store)
        return stats

    def save(self, directory_path: str) -> None:
        """Persist entire indexed Graph RAG pipeline to a directory."""
        dir_p = Path(directory_path)
        dir_p.mkdir(parents=True, exist_ok=True)

        # 1. Save graph
        self.graph_store.save_json(str(dir_p / "graph.json"))

        # 2. Save vectors
        self.vector_store.save(str(dir_p / "vectors.json"))

        # 3. Save communities
        self.community_store.save_json(str(dir_p / "communities.json"))

        # 4. Save chunks
        chunks_data = [c.model_dump() for c in self.chunks_map.values()]
        with open(dir_p / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2)

    def load(self, directory_path: str) -> None:
        """Load an indexed Graph RAG pipeline from a directory."""
        dir_p = Path(directory_path)
        if not dir_p.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        # 1. Load graph
        if (dir_p / "graph.json").exists():
            self.graph_store.load_json(str(dir_p / "graph.json"))

        # 2. Load vectors
        if (dir_p / "vectors.json").exists():
            self.vector_store.load(str(dir_p / "vectors.json"))

        # 3. Load communities
        if (dir_p / "communities.json").exists():
            self.community_store.load_json(str(dir_p / "communities.json"))

        # 4. Load chunks
        if (dir_p / "chunks.json").exists():
            with open(dir_p / "chunks.json", "r", encoding="utf-8") as f:
                raw_chunks = json.load(f)
            self.chunks_map = {c["id"]: TextChunk(**c) for c in raw_chunks}

        self._init_search_engines()
