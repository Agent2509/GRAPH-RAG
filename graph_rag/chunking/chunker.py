"""Text chunking utilities for Graph RAG."""

import re
from typing import Any, Dict, List, Optional
from graph_rag.models import TextChunk


class TextChunker:
    """Sliding-window text chunker preserving paragraph and sentence boundaries."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(
        self,
        text: str,
        doc_id: str = "doc",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[TextChunk]:
        """Split text into overlapping chunks."""
        cleaned_text = re.sub(r"\r\n", "\n", text).strip()
        if not cleaned_text:
            return []

        # If text is smaller than chunk_size, return single chunk
        if len(cleaned_text) <= self.chunk_size:
            return [
                TextChunk(
                    id=f"{doc_id}_chunk_0",
                    text=cleaned_text,
                    doc_id=doc_id,
                    chunk_index=0,
                    metadata=metadata or {},
                    token_count=len(cleaned_text.split()),
                )
            ]

        # Break text into paragraphs or sentences
        paragraphs = re.split(r"\n\s*\n", cleaned_text)
        units = []
        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
            if len(p) <= self.chunk_size:
                units.append(p)
            else:
                # If paragraph itself is longer than chunk_size, split by sentences
                sentences = re.split(r"(?<=[.?!])\s+", p)
                for s in sentences:
                    s = s.strip()
                    if not s:
                        continue
                    if len(s) <= self.chunk_size:
                        units.append(s)
                    else:
                        # Split by words if sentence is still too long
                        words = s.split()
                        curr_words = []
                        curr_len = 0
                        for w in words:
                            if curr_len + len(w) + 1 > self.chunk_size and curr_words:
                                units.append(" ".join(curr_words))
                                curr_words = [w]
                                curr_len = len(w)
                            else:
                                curr_words.append(w)
                                curr_len += len(w) + 1
                        if curr_words:
                            units.append(" ".join(curr_words))

        # Assemble units into sliding-window chunks
        chunks: List[TextChunk] = []
        curr_chunk_units = []
        curr_chunk_len = 0
        chunk_idx = 0

        for unit in units:
            unit_len = len(unit)
            if curr_chunk_len + unit_len + 1 > self.chunk_size and curr_chunk_units:
                chunk_str = "\n\n".join(curr_chunk_units)
                chunks.append(
                    TextChunk(
                        id=f"{doc_id}_chunk_{chunk_idx}",
                        text=chunk_str,
                        doc_id=doc_id,
                        chunk_index=chunk_idx,
                        metadata=metadata or {},
                        token_count=len(chunk_str.split()),
                    )
                )
                chunk_idx += 1

                # Carry over overlap from the end of current chunk
                overlap_units = []
                overlap_len = 0
                for prev_unit in reversed(curr_chunk_units):
                    if overlap_len + len(prev_unit) + 1 <= self.chunk_overlap:
                        overlap_units.insert(0, prev_unit)
                        overlap_len += len(prev_unit) + 1
                    else:
                        break

                curr_chunk_units = overlap_units + [unit]
                curr_chunk_len = sum(len(u) + 1 for u in curr_chunk_units)
            else:
                curr_chunk_units.append(unit)
                curr_chunk_len += unit_len + 1

        if curr_chunk_units:
            chunk_str = "\n\n".join(curr_chunk_units)
            chunks.append(
                TextChunk(
                    id=f"{doc_id}_chunk_{chunk_idx}",
                    text=chunk_str,
                    doc_id=doc_id,
                    chunk_index=chunk_idx,
                    metadata=metadata or {},
                    token_count=len(chunk_str.split()),
                )
            )

        return chunks
