"""Local search engine: Entity-grounded subgraph retrieval with source chunks."""

from typing import Dict, List, Optional
from graph_rag.models import Entity, Relationship, RetrievalResult, TextChunk
from graph_rag.storage.graph_store import GraphStore
from graph_rag.storage.vector_store import VectorStore
from graph_rag.llm.base import BaseLLM, BaseEmbedding

LOCAL_SEARCH_SYSTEM = """You are an expert analytical assistant specializing in Knowledge Graph Retrieval-Augmented Generation (Graph RAG).
Your goal is to answer the user's question accurately, grounded strictly in the provided Knowledge Graph context and source text chunks.

Guidelines:
1. Ground your response in the provided Entities, Relationships, and Text Chunks.
2. Explicitly reference connections from the Knowledge Graph when explaining relationships.
3. If the context does not contain enough information to answer the question, state what is known and what cannot be determined. Do not hallucinate connections.
4. Maintain a clear, factual, and well-structured tone.
"""

LOCAL_SEARCH_USER = """Question: {query}

--- KNOWLEDGE GRAPH CONTEXT ---

[ENTITIES]
{entities_str}

[RELATIONSHIPS]
{relations_str}

[SOURCE TEXT CHUNKS]
{chunks_str}
--- END OF CONTEXT ---

Provide a comprehensive, well-grounded answer to the question using the context above.
"""


class LocalSearch:
    """Performs entity-centric subgraph retrieval and answer generation."""

    def __init__(
        self,
        graph_store: GraphStore,
        vector_store: VectorStore,
        chunks_map: Dict[str, TextChunk],
        llm: BaseLLM,
        embedding_model: BaseEmbedding,
    ):
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.chunks_map = chunks_map
        self.llm = llm
        self.embedding = embedding_model

    def search(
        self,
        query: str,
        top_k_entities: int = 5,
        max_hops: int = 1,
        top_k_chunks: int = 3,
        **kwargs,
    ) -> RetrievalResult:
        """Execute local graph-augmented search for a query."""
        # 1. Embed query
        q_vec = self.embedding.embed_text(query)

        # 2. Vector search for closest entities
        entity_matches = self.vector_store.search(q_vec, top_k=top_k_entities, filter_type="entity")
        seed_entity_names = [meta.get("name") for _, meta, _ in entity_matches if meta.get("name")]

        # Also check for exact case-insensitive matches in graph nodes
        query_words = set(query.lower().split())
        for node in self.graph_store.graph.nodes():
            if node.lower() in query.lower() or any(w in query_words for w in node.lower().split()):
                if node not in seed_entity_names:
                    seed_entity_names.append(node)

        # If still no seeds found, pick top degree nodes as fallback
        if not seed_entity_names:
            degrees = sorted(self.graph_store.graph.degree(), key=lambda x: x[1], reverse=True)
            seed_entity_names = [d[0] for d in degrees[:top_k_entities]]

        # 3. Extract k-hop ego subgraph around seed entities
        sub_entities, sub_relations = self.graph_store.get_subgraph(
            seed_entities=seed_entity_names,
            max_hops=max_hops,
            max_nodes=30,
        )

        # 4. Identify linked text chunks from entities & relations
        relevant_chunk_ids = set()
        for ent in sub_entities:
            relevant_chunk_ids.update(ent.source_chunk_ids)
        for rel in sub_relations:
            relevant_chunk_ids.update(rel.source_chunk_ids)

        # Also perform vector search on chunks for direct semantic relevance
        chunk_matches = self.vector_store.search(q_vec, top_k=top_k_chunks, filter_type="chunk")
        for chunk_id, _, _ in chunk_matches:
            relevant_chunk_ids.add(chunk_id)

        # Gather TextChunk objects
        retrieved_chunks = [
            self.chunks_map[cid] for cid in relevant_chunk_ids if cid in self.chunks_map
        ][:top_k_chunks + 3]

        # 5. Format prompt strings
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

        chunks_lines = [
            f"[Chunk ID: {c.id}]\n{c.text}\n"
            for c in retrieved_chunks
        ]
        chunks_str = "\n".join(chunks_lines) or "No source text chunks available."

        full_prompt = LOCAL_SEARCH_USER.format(
            query=query,
            entities_str=entities_str,
            relations_str=relations_str,
            chunks_str=chunks_str,
        )

        # 6. Generate answer
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
