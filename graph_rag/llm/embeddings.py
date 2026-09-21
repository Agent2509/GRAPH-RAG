"""FastEmbed ONNX CPU embeddings implementation."""

from typing import Dict, List, Optional
import numpy as np
from fastembed import TextEmbedding

from graph_rag.llm.base import BaseEmbedding


class FastEmbedEmbedding(BaseEmbedding):
    """Local, lightweight ONNX-runtime text embeddings on CPU."""

    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or self.DEFAULT_MODEL
        # First load from local cached files to prevent DNS lookups or network errors
        try:
            self._model = TextEmbedding(model_name=self.model_name, local_files_only=True)
        except Exception:
            self._model = TextEmbedding(model_name=self.model_name)
        self._cache: Dict[str, List[float]] = {}
        self._dim = 384

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        if text in self._cache:
            return self._cache[text]

        embeddings = list(self._model.embed([text]))
        vec = embeddings[0].tolist()
        self._cache[text] = vec
        return vec

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of text strings with caching."""
        uncached_indices = []
        uncached_texts = []
        results = [None] * len(texts)

        for i, t in enumerate(texts):
            if t in self._cache:
                results[i] = self._cache[t]
            else:
                uncached_indices.append(i)
                uncached_texts.append(t)

        if uncached_texts:
            new_embeddings = list(self._model.embed(uncached_texts))
            for idx, emb in zip(uncached_indices, new_embeddings):
                vec = emb.tolist()
                self._cache[texts[idx]] = vec
                results[idx] = vec

        return results

    @property
    def dimension(self) -> int:
        """Return the vector dimensionality."""
        return self._dim
