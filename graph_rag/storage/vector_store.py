"""Lightweight NumPy-based cosine similarity vector store with persistence."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class VectorStore:
    """In-memory cosine similarity vector index with JSON persistence."""

    def __init__(self):
        self.item_ids: List[str] = []
        self.vectors: Optional[np.ndarray] = None  # shape: (N, D)
        self.metadata: List[Dict[str, Any]] = []

    def add(self, item_id: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a single item vector and associated metadata."""
        vec = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        if item_id in self.item_ids:
            idx = self.item_ids.index(item_id)
            self.vectors[idx] = vec  # type: ignore
            self.metadata[idx] = metadata or {}
            return

        self.item_ids.append(item_id)
        self.metadata.append(metadata or {})

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
        """Add multiple items and vectors efficiently."""
        if not item_ids:
            return
        meta_list = metadatas or [{} for _ in item_ids]
        for i, (item_id, vec, meta) in enumerate(zip(item_ids, vectors, meta_list)):
            self.add(item_id, vec, meta)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter_type: Optional[str] = None,
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """Search top_k nearest items by cosine similarity."""
        if self.vectors is None or len(self.item_ids) == 0:
            return []

        q_vec = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        # Cosine similarity is simply dot product because vectors are normalized
        scores = np.dot(self.vectors, q_vec)

        # Apply filter if provided
        valid_indices = []
        for i, meta in enumerate(self.metadata):
            if filter_type is not None:
                if meta.get("item_type") == filter_type or meta.get("type") == filter_type:
                    valid_indices.append(i)
            else:
                valid_indices.append(i)

        if not valid_indices:
            return []

        # Sort filtered indices by score descending
        sorted_indices = sorted(valid_indices, key=lambda idx: float(scores[idx]), reverse=True)
        top_indices = sorted_indices[:top_k]

        results = []
        for idx in top_indices:
            results.append((self.item_ids[idx], self.metadata[idx], float(scores[idx])))

        return results

    def save(self, file_path: str) -> None:
        """Persist vector index to a single JSON file."""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "item_ids": self.item_ids,
            "metadata": self.metadata,
            "vectors": self.vectors.tolist() if self.vectors is not None else [],
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load(self, file_path: str) -> None:
        """Load vector index from JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.item_ids = data.get("item_ids", [])
        self.metadata = data.get("metadata", [])
        vec_list = data.get("vectors", [])
        if vec_list:
            self.vectors = np.array(vec_list, dtype=np.float32)
        else:
            self.vectors = None

    def __len__(self) -> int:
        return len(self.item_ids)
