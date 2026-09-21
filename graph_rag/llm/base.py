"""Base interfaces for LLM and Embedding models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union


class BaseLLM(ABC):
    """Abstract interface for Large Language Model clients."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        """Generate text completion from prompt."""
        pass

    @abstractmethod
    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> Union[Dict[str, Any], List[Any]]:
        """Generate structured JSON output from prompt."""
        pass


class BaseEmbedding(ABC):
    """Abstract interface for text embedding models."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding vector for a single string."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for a batch of strings."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the vector dimensionality."""
        pass
