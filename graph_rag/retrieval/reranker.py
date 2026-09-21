"""Local CPU Cross-Encoder Reranker using FlashRank with graceful fallback."""

from typing import Any, Dict, List, Optional


class CrossEncoderReranker:
    """Ultra-fast ONNX-runtime cross-encoder reranker on CPU."""

    def __init__(self, model_name: str = "ms-marco-TinyBERT-L-2-v2"):
        self.model_name = model_name
        self.ranker = None
        self.available = False
        try:
            from flashrank import Ranker
            self.ranker = Ranker(model_name=self.model_name)
            self.available = True
        except Exception:
            self.available = False

    def rerank(
        self,
        query: str,
        passages: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        if not passages:
            return []
        if not self.available or self.ranker is None:
            return passages[:top_k]

        try:
            from flashrank import RerankRequest
            req = RerankRequest(query=query, passages=passages)
            results = self.ranker.rerank(req)
            return results[:top_k]
        except Exception:
            return passages[:top_k]
