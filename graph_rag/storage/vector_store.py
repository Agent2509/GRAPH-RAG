"""Lightweight NumPy-based cosine similarity and BM25 hybrid vector store with persistence."""

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class BM25Index:
    """In-memory BM25 index for sparse keyword retrieval without external dependencies."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_tokens: List[List[str]] = []
        self.doc_lens: List[int] = []
        self.avgdl: float = 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.num_docs: int = 0

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return [w.lower() for w in re.findall(r"\w+", text) if len(w) > 1]

    def build(self, docs: List[str]) -> None:
        self.doc_tokens = [self.tokenize(d) for d in docs]
        self.doc_lens = [len(dt) for dt in self.doc_tokens]
        self.num_docs = len(self.doc_tokens)
        self.avgdl = (sum(self.doc_lens) / self.num_docs) if self.num_docs > 0 else 0.0

        self.doc_freqs = {}
        for dt in self.doc_tokens:
            for term in set(dt):
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

    def score(self, query_tokens: List[str], doc_idx: int) -> float:
        if self.num_docs == 0:
            return 0.0
        doc = self.doc_tokens[doc_idx]
        d_len = self.doc_lens[doc_idx]
        if d_len == 0:
            return 0.0

        score = 0.0
        doc_term_counts: Dict[str, int] = {}
        for t in doc:
            doc_term_counts[t] = doc_term_counts.get(t, 0) + 1

        for q in query_tokens:
            if q not in self.doc_freqs:
                continue
            df = self.doc_freqs[q]
            idf = math.log(1.0 + (self.num_docs - df + 0.5) / (df + 0.5))
            tf = doc_term_counts.get(q, 0)
            if tf > 0:
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (d_len / self.avgdl))
                score += idf * (numerator / denominator)
        return score


class VectorStore:
    """In-memory cosine similarity and BM25 hybrid index with JSON persistence."""

    def __init__(self):
        self.item_ids: List[str] = []
        self.vectors: Optional[np.ndarray] = None
        self.metadata: List[Dict[str, Any]] = []
        self.raw_texts: List[str] = []
        self.bm25: Optional[BM25Index] = None
        self._bm25_dirty: bool = False

    def add(self, item_id: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None) -> None:
        vec = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        meta = metadata or {}
        text_rep = str(meta.get("text") or meta.get("name") or meta.get("title") or item_id)

        if item_id in self.item_ids:
            idx = self.item_ids.index(item_id)
            self.vectors[idx] = vec  # type: ignore
            self.metadata[idx] = meta
            self.raw_texts[idx] = text_rep
            self._bm25_dirty = True
            return

        self.item_ids.append(item_id)
        self.metadata.append(meta)
        self.raw_texts.append(text_rep)
        self._bm25_dirty = True

        if self.vectors is None:
            self.vectors = np.expand_dims(vec, axis=0)
        else:
            self.vectors = np.vstack([self.vectors, vec])

    def add_batch(
        self,
        item_ids: List[str],
        vectors: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if not item_ids:
            return
        meta_list = metadatas or [{} for _ in item_ids]
        for item_id, vec, meta in zip(item_ids, vectors, meta_list):
            self.add(item_id, vec, meta)

    def _ensure_bm25(self) -> None:
        if self.bm25 is None or self._bm25_dirty:
            self.bm25 = BM25Index()
            self.bm25.build(self.raw_texts)
            self._bm25_dirty = False

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter_type: Optional[str] = None,
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        if self.vectors is None or len(self.item_ids) == 0:
            return []

        q_vec = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        scores = np.dot(self.vectors, q_vec)

        valid_indices = []
        for i, meta in enumerate(self.metadata):
            if filter_type is not None:
                if meta.get("item_type") == filter_type or meta.get("type") == filter_type:
                    valid_indices.append(i)
            else:
                valid_indices.append(i)

        if not valid_indices:
            return []

        sorted_indices = sorted(valid_indices, key=lambda idx: float(scores[idx]), reverse=True)
        return [(self.item_ids[idx], self.metadata[idx], float(scores[idx])) for idx in sorted_indices[:top_k]]

    def search_hybrid(
        self,
        query_text: str,
        query_vector: List[float],
        top_k: int = 5,
        filter_type: Optional[str] = None,
        rrf_k: int = 60,
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        if self.vectors is None or len(self.item_ids) == 0:
            return []

        self._ensure_bm25()
        valid_indices = [
            i for i, meta in enumerate(self.metadata)
            if filter_type is None or meta.get("item_type") == filter_type or meta.get("type") == filter_type
        ]
        if not valid_indices:
            return []

        q_vec = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm
        dense_scores = np.dot(self.vectors, q_vec)
        dense_ranked = sorted(valid_indices, key=lambda i: float(dense_scores[i]), reverse=True)
        dense_rank_map = {idx: rank for rank, idx in enumerate(dense_ranked)}

        q_tokens = BM25Index.tokenize(query_text)
        bm25_ranked = sorted(valid_indices, key=lambda i: self.bm25.score(q_tokens, i), reverse=True)
        bm25_rank_map = {idx: rank for rank, idx in enumerate(bm25_ranked)}

        rrf_scores: Dict[int, float] = {}
        for idx in valid_indices:
            d_rank = dense_rank_map[idx]
            s_rank = bm25_rank_map[idx]
            rrf_score = (1.0 / (rrf_k + d_rank + 1)) + (1.0 / (rrf_k + s_rank + 1))
            rrf_scores[idx] = rrf_score

        top_indices = sorted(valid_indices, key=lambda idx: rrf_scores[idx], reverse=True)[:top_k]
        return [(self.item_ids[idx], self.metadata[idx], float(rrf_scores[idx])) for idx in top_indices]

    def save(self, file_path: str) -> None:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "item_ids": self.item_ids,
            "metadata": self.metadata,
            "raw_texts": self.raw_texts,
            "vectors": self.vectors.tolist() if self.vectors is not None else [],
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load(self, file_path: str) -> None:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.item_ids = data.get("item_ids", [])
        self.metadata = data.get("metadata", [])
        self.raw_texts = data.get("raw_texts", [
            str(m.get("text") or m.get("name") or i)
            for i, m in zip(self.item_ids, self.metadata)
        ])
        vec_list = data.get("vectors", [])
        self.vectors = np.array(vec_list, dtype=np.float32) if vec_list else None
        self._bm25_dirty = True

    def __len__(self) -> int:
        return len(self.item_ids)
