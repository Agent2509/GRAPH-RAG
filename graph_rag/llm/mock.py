"""Mock LLM and Embedding implementations for deterministic testing."""

import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Union
import numpy as np

from graph_rag.llm.base import BaseLLM, BaseEmbedding


class MockLLM(BaseLLM):
    """Deterministic Mock LLM for offline tests and development."""

    def __init__(self, default_response: Optional[str] = None):
        self.default_response = default_response or "This is a mock answer based on the retrieved graph context."

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        if "summary" in prompt.lower() or "community" in prompt.lower():
            return "This community focuses on technology entities, business partnerships, and scientific innovations."
        return self.default_response

    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> Union[Dict[str, Any], List[Any]]:
        # If this is an entity/relation extraction prompt:
        # Extract capitalized words as sample entities
        capitalized = list(set(re.findall(r"\b[A-Z][a-zA-Z0-9_]+\b", prompt)))
        # Filter out common stop words
        stopwords = {"The", "This", "That", "In", "On", "At", "For", "With", "By", "From", "And", "Or", "An", "A", "JSON", "Extract"}
        valid_words = [w for w in capitalized if w not in stopwords]

        entities = []
        for word in valid_words[:5]:
            entities.append({
                "name": word,
                "type": "ORGANIZATION" if word.endswith("Corp") or word.endswith("Labs") or word.endswith("Inc") else "CONCEPT",
                "description": f"Entity representing {word} in the text.",
            })

        relationships = []
        if len(entities) >= 2:
            for i in range(len(entities) - 1):
                relationships.append({
                    "source": entities[i]["name"],
                    "target": entities[i + 1]["name"],
                    "relation_type": "CONNECTED_TO",
                    "description": f"{entities[i]['name']} is connected to {entities[i + 1]['name']}.",
                    "weight": 1.0,
                })

        return {
            "entities": entities,
            "relationships": relationships,
        }


class MockEmbedding(BaseEmbedding):
    """Deterministic Mock Embedding producing fixed-dimension unit vectors."""

    def __init__(self, dimension: int = 64):
        self._dim = dimension

    def _hash_vector(self, text: str) -> List[float]:
        # Generate pseudo-random deterministic vector from text hash
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # Seed numpy random generator with hash
        seed = int.from_bytes(h[:4], "big")
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self._dim)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_text(self, text: str) -> List[float]:
        return self._hash_vector(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self._hash_vector(t) for t in texts]

    @property
    def dimension(self) -> int:
        return self._dim
