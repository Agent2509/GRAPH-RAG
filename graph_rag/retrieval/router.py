"""Agentic Query Router for automatic retrieval mode selection."""

from typing import List, Literal
from pydantic import BaseModel, Field
from graph_rag.llm.base import BaseLLM

ROUTER_SYSTEM_PROMPT = """You are an expert query classifier for an enterprise Knowledge Graph RAG system.
Your job is to analyze the user query and determine the optimal search strategy:

- 'local': Questions focused on specific entities, people, organizations, dates, products, or direct relationships (e.g., "What did Alice Zhao invent?", "How is Company A connected to Company B?").
- 'global': Broad, thematic, summary, or macro-level questions about the entire corpus (e.g., "What are the main themes?", "Summarize the key findings", "What are the primary operational risks?").
- 'hybrid': Questions asking for broad themes combined with specific entity case studies or comparisons.

You MUST return a JSON object strictly conforming to this schema:
{
  "mode": "local" | "global" | "hybrid",
  "reasoning": "Brief explanation of routing decision",
  "extracted_entities": ["Entity1", "Entity2"]
}
"""


class RouteDecision(BaseModel):
    mode: Literal["local", "global", "hybrid"] = "local"
    reasoning: str = ""
    extracted_entities: List[str] = Field(default_factory=list)


class QueryRouter:
    """Classifies user queries into optimal retrieval pipelines."""

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def route(self, query: str) -> RouteDecision:
        """Decide the optimal retrieval mode for a query."""
        q_lower = query.lower().strip()

        thematic_keywords = [
            "main themes", "summarize", "overview", "key findings",
            "summary of", "big picture", "broad trends", "executive summary",
            "all topics", "high level"
        ]
        if any(kw in q_lower for kw in thematic_keywords):
            return RouteDecision(
                mode="global",
                reasoning="Query matches macro-thematic patterns.",
                extracted_entities=[],
            )

        try:
            data = self.llm.generate_json(
                prompt=f"Classify this query: {query}",
                system_prompt=ROUTER_SYSTEM_PROMPT,
            )
            if isinstance(data, dict) and "mode" in data:
                return RouteDecision(**data)
        except Exception:
            pass

        return RouteDecision(
            mode="local",
            reasoning="Default fallback to local entity-centric search.",
            extracted_entities=[],
        )
