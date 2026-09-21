"""Global search engine: Macro-level synthesis across community reports."""

from typing import List, Optional, Set
import numpy as np
from graph_rag.models import Community, Entity, Relationship, RetrievalResult
from graph_rag.storage.community_store import CommunityStore
from graph_rag.storage.graph_store import GraphStore
from graph_rag.llm.base import BaseLLM, BaseEmbedding

GLOBAL_SEARCH_SYSTEM = """You are a senior executive intelligence analyst. Your role is to answer broad, high-level, and thematic questions about an entire dataset using pre-compiled community reports from a Knowledge Graph.

Guidelines:
1. Provide an executive-level synthesis addressing the question across all relevant communities.
2. Structure your answer with clear thematic headings and bullet points.
3. Explicitly reference the communities and themes that support each observation.
4. Maintain objectivity and ground every finding in the provided community reports.
"""

GLOBAL_SEARCH_USER = """Question: {query}

--- COMMUNITY KNOWLEDGE REPORTS ---
{reports_str}
--- END OF REPORTS ---

Provide a comprehensive, high-level synthesis answering the question based on these community reports:
"""


class GlobalSearch:
    """Performs corpus-wide thematic search across community reports."""

    def __init__(
        self,
        community_store: CommunityStore,
        llm: BaseLLM,
        embedding_model: BaseEmbedding,
        graph_store: Optional[GraphStore] = None,
    ):
        self.community_store = community_store
        self.llm = llm
        self.embedding = embedding_model
        self.graph_store = graph_store

    def search(
        self,
        query: str,
        top_k_communities: int = 5,
        **kwargs,
    ) -> RetrievalResult:
        """Execute global thematic search."""
        communities = self.community_store.get_all_communities()

        if not communities:
            return RetrievalResult(
                query=query,
                answer="No community reports are currently available to perform a global search. Please index documents or run community detection first.",
                search_mode="global",
                cited_entities=[],
                cited_relations=[],
                cited_chunks=[],
                graph_context_text="",
            )

        # Rank communities by semantic similarity to query
        q_vec = np.array(self.embedding.embed_text(query), dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        scored_communities = []
        for comm in communities:
            # Embed community summary + title
            text_rep = f"{comm.title}\n{comm.summary}"
            c_vec = np.array(self.embedding.embed_text(text_rep), dtype=np.float32)
            c_norm = np.linalg.norm(c_vec)
            if c_norm > 0:
                c_vec = c_vec / c_norm
            score = float(np.dot(q_vec, c_vec))
            scored_communities.append((comm, score))

        # Sort by score descending
        scored_communities.sort(key=lambda x: x[1], reverse=True)
        top_communities = [c for c, _ in scored_communities[:top_k_communities]]

        # Format reports
        reports_blocks = []
        for c in top_communities:
            findings_str = "\n".join(f"  * {f}" for f in c.findings) if c.findings else "  * None noted."
            members_str = ", ".join(c.entity_names[:10])
            block = (
                f"[Community {c.id}: {c.title}]\n"
                f"Key Entities: {members_str}\n"
                f"Summary: {c.summary}\n"
                f"Key Findings:\n{findings_str}\n"
            )
            reports_blocks.append(block)

        reports_str = "\n\n".join(reports_blocks)
        full_prompt = GLOBAL_SEARCH_USER.format(query=query, reports_str=reports_str)

        answer = self.llm.generate(prompt=full_prompt, system_prompt=GLOBAL_SEARCH_SYSTEM)

        # Collect cited entities and relations from top communities if graph_store is present
        cited_entities: List[Entity] = []
        cited_relations: List[Relationship] = []
        if self.graph_store:
            seen_nodes: Set[str] = set()
            for c in top_communities:
                for ent_name in c.entity_names:
                    if ent_name not in seen_nodes:
                        ent_obj = self.graph_store.get_entity(ent_name)
                        if ent_obj:
                            cited_entities.append(ent_obj)
                            seen_nodes.add(ent_name)

            for u, v, data in self.graph_store.graph.edges(data=True):
                if u in seen_nodes and v in seen_nodes:
                    cited_relations.append(
                        Relationship(
                            source=u,
                            target=v,
                            relation_type=data.get("relation_type", "RELATED_TO"),
                            description=data.get("description", ""),
                            weight=data.get("weight", 1.0),
                            source_chunk_ids=data.get("source_chunk_ids", []),
                        )
                    )

        return RetrievalResult(
            query=query,
            answer=answer,
            search_mode="global",
            cited_entities=cited_entities[:20],
            cited_relations=cited_relations[:20],
            cited_chunks=[],
            graph_context_text=reports_str,
        )
