"""Knowledge graph entity and relationship extractor."""

import json
from typing import Dict, List, Tuple
from graph_rag.models import Entity, Relationship, TextChunk
from graph_rag.llm.base import BaseLLM

EXTRACTION_SYSTEM_PROMPT = """You are an expert Knowledge Graph architect. Your task is to extract all significant entities and their relationships from the provided text to build an accurate, comprehensive knowledge graph.

Guidelines:
1. ENTITIES:
   - Identify distinct, named entities: PEOPLE, ORGANIZATIONS, TECHNOLOGIES, PRODUCTS, CONCEPTS, LOCATIONS, EVENTS.
   - Entity names MUST be concise and standardized (e.g., "Apple", not "the tech company Apple").
   - Provide a clear, factual 1-2 sentence description summarizing what the text reveals about this entity.
2. RELATIONSHIPS & CONNECTIVITY:
   - Identify direct and meaningful connections linking the extracted entities.
   - Connectivity: Strive to connect every extracted entity by at least one relationship to related entities, document sections, or topics (e.g., PART_OF, DEFINED_IN, REQUIRES, IMPLEMENTS, APPLIES_TO, GOVERNED_BY, CONNECTED_TO). Do not leave entities isolated if a connection is stated or implied in the text.
   - Source and Target MUST match the entity names extracted above.
   - relation_type MUST be an active, uppercase verb phrase (e.g., FOUNDED, ACQUIRED, DEVELOPED, COLLABORATES_WITH, PART_OF, REQUIRES, IMPLEMENTS, APPLIES_TO).
   - Provide a 1-sentence description explaining the relationship as described in the text.
   - Assign a confidence weight between 0.5 and 1.0.

You MUST format your entire response as a valid JSON object matching this schema:
{
  "entities": [
    {
      "name": "Entity Name",
      "type": "ORGANIZATION | PERSON | TECHNOLOGY | PRODUCT | CONCEPT | LOCATION | EVENT",
      "description": "Factual description from text"
    }
  ],
  "relationships": [
    {
      "source": "Entity Name A",
      "target": "Entity Name B",
      "relation_type": "RELATION_NAME",
      "description": "Explanation of how A and B relate",
      "weight": 1.0
    }
  ]
}
"""

EXTRACTION_USER_PROMPT = """Extract all key entities and relationships from the following text:

--- TEXT ---
{text}
--- END TEXT ---
"""


class GraphExtractor:
    """Extracts entities and relationships from text chunks using an LLM."""

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def extract_from_chunk(self, chunk: TextChunk) -> Tuple[List[Entity], List[Relationship]]:
        """Extract entities and relationships from a single text chunk."""
        prompt = EXTRACTION_USER_PROMPT.format(text=chunk.text)
        data = self.llm.generate_json(prompt=prompt, system_prompt=EXTRACTION_SYSTEM_PROMPT)

        if not isinstance(data, dict):
            return [], []

        raw_entities = data.get("entities", [])
        raw_relations = data.get("relationships", [])

        entities: List[Entity] = []
        for e in raw_entities:
            if isinstance(e, dict) and e.get("name"):
                name = str(e["name"]).strip()
                if name:
                    entities.append(
                        Entity(
                            name=name,
                            type=str(e.get("type", "CONCEPT")).strip().upper(),
                            description=str(e.get("description", "")).strip(),
                            source_chunk_ids=[chunk.id],
                        )
                    )

        relationships: List[Relationship] = []
        for r in raw_relations:
            if isinstance(r, dict) and r.get("source") and r.get("target"):
                src = str(r["source"]).strip()
                tgt = str(r["target"]).strip()
                rel_type = str(r.get("relation_type", "RELATED_TO")).strip().upper().replace(" ", "_")
                if src and tgt and src.lower() != tgt.lower():
                    relationships.append(
                        Relationship(
                            source=src,
                            target=tgt,
                            relation_type=rel_type,
                            description=str(r.get("description", "")).strip(),
                            weight=float(r.get("weight", 1.0)),
                            source_chunk_ids=[chunk.id],
                        )
                    )

        return entities, relationships

    def extract_from_batch(self, chunks: List[TextChunk]) -> Tuple[List[Entity], List[Relationship]]:
        """Extract entities and relationships from a batch of adjacent text chunks at once."""
        if not chunks:
            return [], []
        if len(chunks) == 1:
            return self.extract_from_chunk(chunks[0])

        # Combine chunks with clear demarcations
        chunk_parts = []
        batch_chunk_ids = [c.id for c in chunks]
        for idx, c in enumerate(chunks, 1):
            chunk_parts.append(f"[Section {idx} - Chunk {c.id}]:\n{c.text}")

        combined_text = "\n\n".join(chunk_parts)
        prompt = EXTRACTION_USER_PROMPT.format(text=combined_text)
        data = self.llm.generate_json(prompt=prompt, system_prompt=EXTRACTION_SYSTEM_PROMPT)

        if not isinstance(data, dict):
            return [], []

        raw_entities = data.get("entities", [])
        raw_relations = data.get("relationships", [])

        entities: List[Entity] = []
        for e in raw_entities:
            if isinstance(e, dict) and e.get("name"):
                name = str(e["name"]).strip()
                if name:
                    # Associate with chunks that mention this entity
                    matching_cids = [
                        c.id for c in chunks if name.lower() in c.text.lower()
                    ] or batch_chunk_ids
                    entities.append(
                        Entity(
                            name=name,
                            type=str(e.get("type", "CONCEPT")).strip().upper(),
                            description=str(e.get("description", "")).strip(),
                            source_chunk_ids=matching_cids,
                        )
                    )

        relationships: List[Relationship] = []
        for r in raw_relations:
            if isinstance(r, dict) and r.get("source") and r.get("target"):
                src = str(r["source"]).strip()
                tgt = str(r["target"]).strip()
                rel_type = str(r.get("relation_type", "RELATED_TO")).strip().upper().replace(" ", "_")
                if src and tgt and src.lower() != tgt.lower():
                    # Associate with chunks mentioning either endpoint
                    matching_cids = [
                        c.id for c in chunks
                        if src.lower() in c.text.lower() or tgt.lower() in c.text.lower()
                    ] or batch_chunk_ids
                    relationships.append(
                        Relationship(
                            source=src,
                            target=tgt,
                            relation_type=rel_type,
                            description=str(r.get("description", "")).strip(),
                            weight=float(r.get("weight", 1.0)),
                            source_chunk_ids=matching_cids,
                        )
                    )

        return entities, relationships
