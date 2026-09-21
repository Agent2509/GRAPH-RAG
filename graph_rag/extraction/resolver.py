"""Entity resolution and deduplication module."""

import re
from typing import Dict, List, Tuple
from graph_rag.models import Entity, Relationship


class EntityResolver:
    """Normalizes and resolves entities and relationships across chunks."""

    def __init__(self):
        # Suffixes to strip when normalizing corporate entities
        self.corporate_suffixes = [
            r"\bInc\.?\b",
            r"\bLLC\.?\b",
            r"\bCorp\.?\b",
            r"\bCorporation\b",
            r"\bLtd\.?\b",
            r"\bCo\.?\b",
        ]

    def normalize_name(self, name: str) -> str:
        """Standardize an entity name."""
        cleaned = name.strip()
        # Remove surrounding quotes
        cleaned = re.sub(r'^["\']|["\']$', "", cleaned).strip()
        # Remove leading "the "
        cleaned = re.sub(r"^[Tt]he\s+", "", cleaned).strip()
        return cleaned

    def resolve_entities(self, entities: List[Entity]) -> Dict[str, Entity]:
        """Deduplicate entities by canonical normalized name."""
        resolved: Dict[str, Entity] = {}

        for entity in entities:
            canonical = self.normalize_name(entity.name)
            if not canonical:
                continue

            lower_key = canonical.lower()

            if lower_key not in resolved:
                resolved[lower_key] = Entity(
                    name=canonical,
                    type=entity.type,
                    description=entity.description,
                    source_chunk_ids=list(set(entity.source_chunk_ids)),
                )
            else:
                existing = resolved[lower_key]
                # Merge descriptions if distinct
                if entity.description and entity.description not in existing.description:
                    if existing.description:
                        existing.description += f" {entity.description}"
                    else:
                        existing.description = entity.description

                # Union chunk provenance
                existing.source_chunk_ids = list(
                    set(existing.source_chunk_ids + entity.source_chunk_ids)
                )

                # Prioritize specific types over default "CONCEPT"
                if existing.type == "CONCEPT" and entity.type != "CONCEPT":
                    existing.type = entity.type

        return resolved

    def resolve_relationships(
        self,
        relationships: List[Relationship],
        resolved_entities: Dict[str, Entity],
    ) -> List[Relationship]:
        """Align relationship endpoints with canonical entities and deduplicate edges."""
        edge_map: Dict[Tuple[str, str, str], Relationship] = {}

        for rel in relationships:
            src_norm = self.normalize_name(rel.source)
            tgt_norm = self.normalize_name(rel.target)

            src_key = src_norm.lower()
            tgt_key = tgt_norm.lower()

            # Ensure both source and target exist in resolved entities
            if src_key not in resolved_entities or tgt_key not in resolved_entities:
                continue

            canonical_src = resolved_entities[src_key].name
            canonical_tgt = resolved_entities[tgt_key].name

            if canonical_src.lower() == canonical_tgt.lower():
                continue

            edge_key = (canonical_src.lower(), canonical_tgt.lower(), rel.relation_type)

            if edge_key not in edge_map:
                edge_map[edge_key] = Relationship(
                    source=canonical_src,
                    target=canonical_tgt,
                    relation_type=rel.relation_type,
                    description=rel.description,
                    weight=rel.weight,
                    source_chunk_ids=list(set(rel.source_chunk_ids)),
                )
            else:
                existing_rel = edge_map[edge_key]
                # Combine descriptions
                if rel.description and rel.description not in existing_rel.description:
                    if existing_rel.description:
                        existing_rel.description += f" {rel.description}"
                    else:
                        existing_rel.description = rel.description

                # Increase weight to reflect repeated corroboration
                existing_rel.weight += rel.weight
                existing_rel.source_chunk_ids = list(
                    set(existing_rel.source_chunk_ids + rel.source_chunk_ids)
                )

        return list(edge_map.values())
