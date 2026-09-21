"""Local search engine: Multi-hop path-bridged retrieval with cross-encoder reranking and citations."""

from typing import Any, Dict, List, Optional, Set
from graph_rag.models import Entity, Relationship, RetrievalResult, TextChunk
from graph_rag.storage.graph_store import GraphStore
from graph_rag.storage.vector_store import VectorStore
from graph_rag.llm.base import BaseLLM, BaseEmbedding
from graph_rag.retrieval.reranker import CrossEncoderReranker

LOCAL_SEARCH_SYSTEM = """You are an enterprise Knowledge Graph RAG analyst.
Your task is to answer the user query with extreme factual precision, grounded strictly in the provided Knowledge Graph context and source text chunks.

Mandatory Guidelines:
1. CITATIONS: You MUST cite specific evidence for every factual statement using bracketed format:
   - Cite entities: [Entity: Name]
   - Cite graph edges: [Relation: Source -> RELATION_TYPE -> Target]
   - Cite source text chunks: [Chunk: Chunk_ID]
2. CONNECTIVITY: When explaining connections between entities, explicitly describe the intermediate path hops shown in the context.
3. GROUNDING: Never hallucinate connections not verified in the context. If context is missing to answer fully, explicitly declare: "Based on available knowledge, [verified facts]. However, there is no direct evidence confirming [unknown]."
"""

LOCAL_SEARCH_USER = """Question: {query}

--- KNOWLEDGE GRAPH CONTEXT ---

[ENTITIES]
{entities_str}

[RELATIONSHIPS & PATH BRIDGES]
{relations_str}

[SOURCE TEXT CHUNKS]
{chunks_str}
--- END OF CONTEXT ---

Provide an analytical, grounded answer with explicit bracketed citations:
"""


class LocalSearch:
    """Performs entity-centric subgraph retrieval, multi-hop path bridging, reranking, and answer generation."""

    def __init__(
        self,
        graph_store: GraphStore,
        vector_store: VectorStore,
        chunks_map: Dict[str, TextChunk],
        llm: BaseLLM,
        embedding_model: BaseEmbedding,
        parent_map: Optional[Dict[str, str]] = None,
        reranker: Optional[CrossEncoderReranker] = None,
    ):
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.chunks_map = chunks_map
        self.llm = llm
        self.embedding = embedding_model
        self.parent_map = parent_map or {}
        self.reranker = reranker or CrossEncoderReranker()

    def search(
        self,
        query: str,
        top_k_entities: int = 5,
        max_hops: int = 1,
        top_k_chunks: int = 4,
        **kwargs,
    ) -> RetrievalResult:
        q_vec = self.embedding.embed_text(query)

        entity_matches = self.vector_store.search_hybrid(
            query_text=query,
            query_vector=q_vec,
            top_k=top_k_entities,
            filter_type="entity",
        )
        seed_entity_names = [meta.get("name") for _, meta, _ in entity_matches if meta.get("name")]

        query_lower = query.lower()
        for node in self.graph_store.graph.nodes():
            if node.lower() in query_lower:
                if node not in seed_entity_names:
                    seed_entity_names.append(node)

        if not seed_entity_names:
            degrees = sorted(self.graph_store.graph.degree(), key=lambda x: x[1], reverse=True)
            seed_entity_names = [d[0] for d in degrees[:top_k_entities]]

        if len(seed_entity_names) >= 2:
            paths = self.graph_store.find_all_paths_between(seed_entity_names[:6], cutoff=3)
            for path in paths:
                for node in path:
                    if node not in seed_entity_names:
                        seed_entity_names.append(node)

        sub_entities, sub_relations = self.graph_store.get_subgraph(
            seed_entities=seed_entity_names,
            max_hops=max_hops,
            max_nodes=40,
        )

        candidate_chunk_ids: Set[str] = set()
        for ent in sub_entities:
            candidate_chunk_ids.update(ent.source_chunk_ids)
        for rel in sub_relations:
            candidate_chunk_ids.update(rel.source_chunk_ids)

        chunk_matches = self.vector_store.search_hybrid(
            query_text=query,
            query_vector=q_vec,
            top_k=top_k_chunks * 2,
            filter_type="chunk",
        )
        for cid, _, _ in chunk_matches:
            candidate_chunk_ids.add(cid)

        passages = []
        for cid in candidate_chunk_ids:
            if cid in self.chunks_map:
                chunk = self.chunks_map[cid]
                parent_id = chunk.metadata.get("parent_id")
                expanded_text = self.parent_map.get(parent_id, chunk.text) if parent_id else chunk.text
                passages.append({"id": cid, "text": expanded_text})

        reranked = self.reranker.rerank(query=query, passages=passages, top_k=top_k_chunks)
        selected_chunk_ids = [p["id"] for p in reranked]
        retrieved_chunks = [self.chunks_map[cid] for cid in selected_chunk_ids if cid in self.chunks_map]

        entities_lines = [
            f"- **{e.name}** ({e.type}): {e.description or 'No description'}"
            for e in sub_entities
        ]
        entities_str = "\n".join(entities_lines) or "No entities matched."

        relations_lines = [
            f"- ({r.source}) --[{r.relation_type}]--> ({r.target}): {r.description}"
            for r in sub_relations
        ]
        relations_str = "\n".join(relations_lines) or "No direct relationships found."

        chunks_lines = []
        for c in retrieved_chunks:
            parent_id = c.metadata.get("parent_id")
            chunk_content = self.parent_map.get(parent_id, c.text) if parent_id else c.text
            chunks_lines.append(f"[Chunk ID: {c.id}]\n{chunk_content}\n")
        chunks_str = "\n".join(chunks_lines) or "No source text chunks available."

        full_prompt = LOCAL_SEARCH_USER.format(
            query=query,
            entities_str=entities_str,
            relations_str=relations_str,
            chunks_str=chunks_str,
        )

        answer = self.llm.generate(prompt=full_prompt, system_prompt=LOCAL_SEARCH_SYSTEM)

        return RetrievalResult(
            query=query,
            answer=answer,
            search_mode="local",
            cited_entities=sub_entities,
            cited_relations=sub_relations,
            cited_chunks=retrieved_chunks,
            graph_context_text=f"Entities:\n{entities_str}\n\nRelationships:\n{relations_str}",
        )
