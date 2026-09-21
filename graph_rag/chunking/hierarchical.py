"""Hierarchical parent-child chunking for high-precision retrieval with complete context."""

import re
from typing import Dict, List, Tuple
from graph_rag.models import TextChunk


class HierarchicalChunker:
    """Produces fine-grained child chunks mapped to larger parent contextual blocks."""

    def __init__(
        self,
        parent_size: int = 1200,
        child_size: int = 350,
        child_overlap: int = 60,
    ):
        self.parent_size = parent_size
        self.child_size = child_size
        self.child_overlap = child_overlap

    def chunk_document(
        self,
        text: str,
        doc_id: str = "doc",
    ) -> Tuple[List[TextChunk], Dict[str, str]]:
        cleaned = re.sub(r"\r\n", "\n", text).strip()
        if not cleaned:
            return [], {}

        raw_paragraphs = re.split(r"\n\s*\n", cleaned)
        parents: List[str] = []
        curr_p: List[str] = []
        curr_len = 0

        for p in raw_paragraphs:
            p_strip = p.strip()
            if not p_strip:
                continue
            if curr_len + len(p_strip) > self.parent_size and curr_p:
                parents.append("\n\n".join(curr_p))
                curr_p = [p_strip]
                curr_len = len(p_strip)
            else:
                curr_p.append(p_strip)
                curr_len += len(p_strip)
        if curr_p:
            parents.append("\n\n".join(curr_p))

        parent_map: Dict[str, str] = {}
        child_chunks: List[TextChunk] = []
        child_counter = 0

        for p_idx, parent_text in enumerate(parents):
            parent_id = f"{doc_id}_parent_{p_idx}"
            parent_map[parent_id] = parent_text

            sentences = re.split(r"(?<=[.?!])\s+", parent_text)
            curr_c: List[str] = []
            curr_c_len = 0

            for s in sentences:
                s_strip = s.strip()
                if not s_strip:
                    continue
                if curr_c_len + len(s_strip) > self.child_size and curr_c:
                    child_text = " ".join(curr_c)
                    child_chunks.append(
                        TextChunk(
                            id=f"{doc_id}_child_{child_counter}",
                            text=child_text,
                            doc_id=doc_id,
                            chunk_index=child_counter,
                            metadata={"parent_id": parent_id, "is_child": True},
                            token_count=len(child_text.split()),
                        )
                    )
                    child_counter += 1
                    curr_c = [curr_c[-1], s_strip] if len(curr_c) > 1 else [s_strip]
                    curr_c_len = sum(len(x) for x in curr_c)
                else:
                    curr_c.append(s_strip)
                    curr_c_len += len(s_strip)

            if curr_c:
                child_text = " ".join(curr_c)
                child_chunks.append(
                    TextChunk(
                        id=f"{doc_id}_child_{child_counter}",
                        text=child_text,
                        doc_id=doc_id,
                        chunk_index=child_counter,
                        metadata={"parent_id": parent_id, "is_child": True},
                        token_count=len(child_text.split()),
                    )
                )
                child_counter += 1

        return child_chunks, parent_map
