"""NetworkX-based embedded Knowledge Graph store."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import networkx as nx

from graph_rag.models import Entity, Relationship, GraphStats


class GraphStore:
    """Embedded Knowledge Graph store powered by NetworkX."""

    def __init__(self):
        # MultiDiGraph allows multiple labeled relationships between the same pair of entities
        self.graph = nx.MultiDiGraph()

    def add_entity(self, entity: Entity) -> None:
        """Add or update an entity node."""
        name = entity.name.strip()
        if not name:
            return

        if self.graph.has_node(name):
            # Update attributes
            curr_desc = self.graph.nodes[name].get("description", "")
            if entity.description and entity.description not in curr_desc:
                new_desc = f"{curr_desc} {entity.description}".strip() if curr_desc else entity.description
                self.graph.nodes[name]["description"] = new_desc

            curr_chunks = set(self.graph.nodes[name].get("source_chunk_ids", []))
            curr_chunks.update(entity.source_chunk_ids)
            self.graph.nodes[name]["source_chunk_ids"] = list(curr_chunks)

            if self.graph.nodes[name].get("type") == "CONCEPT" and entity.type != "CONCEPT":
                self.graph.nodes[name]["type"] = entity.type
        else:
            self.graph.add_node(
                name,
                type=entity.type,
                description=entity.description,
                source_chunk_ids=list(set(entity.source_chunk_ids)),
            )

    def add_relationship(self, rel: Relationship) -> None:
        """Add or update a directed relationship edge."""
        src = rel.source.strip()
        tgt = rel.target.strip()
        if not src or not tgt or src.lower() == tgt.lower():
            return

        # Ensure endpoints exist
        if not self.graph.has_node(src):
            self.graph.add_node(src, type="CONCEPT", description="", source_chunk_ids=[])
        if not self.graph.has_node(tgt):
            self.graph.add_node(tgt, type="CONCEPT", description="", source_chunk_ids=[])

        # Check if an edge with this relation_type already exists
        edge_key = None
        if self.graph.has_edge(src, tgt):
            for k, data in self.graph[src][tgt].items():
                if data.get("relation_type") == rel.relation_type:
                    edge_key = k
                    break

        if edge_key is not None:
            data = self.graph[src][tgt][edge_key]
            data["weight"] = data.get("weight", 1.0) + rel.weight
            if rel.description and rel.description not in data.get("description", ""):
                data["description"] = f"{data.get('description', '')} {rel.description}".strip()
            chunks = set(data.get("source_chunk_ids", []))
            chunks.update(rel.source_chunk_ids)
            data["source_chunk_ids"] = list(chunks)
        else:
            self.graph.add_edge(
                src,
                tgt,
                relation_type=rel.relation_type,
                description=rel.description,
                weight=rel.weight,
                source_chunk_ids=list(set(rel.source_chunk_ids)),
            )

    def get_entity(self, name: str) -> Optional[Entity]:
        """Retrieve entity by name."""
        if self.graph.has_node(name):
            data = self.graph.nodes[name]
            return Entity(
                name=name,
                type=data.get("type", "CONCEPT"),
                description=data.get("description", ""),
                source_chunk_ids=data.get("source_chunk_ids", []),
            )
        return None

    def get_all_entities(self) -> List[Entity]:
        """Return all entities in the graph."""
        entities = []
        for node, data in self.graph.nodes(data=True):
            entities.append(
                Entity(
                    name=node,
                    type=data.get("type", "CONCEPT"),
                    description=data.get("description", ""),
                    source_chunk_ids=data.get("source_chunk_ids", []),
                )
            )
        return entities

    def get_all_relationships(self) -> List[Relationship]:
        """Return all relationships in the graph."""
        relations = []
        for u, v, data in self.graph.edges(data=True):
            relations.append(
                Relationship(
                    source=u,
                    target=v,
                    relation_type=data.get("relation_type", "RELATED_TO"),
                    description=data.get("description", ""),
                    weight=data.get("weight", 1.0),
                    source_chunk_ids=data.get("source_chunk_ids", []),
                )
            )
        return relations

    def get_neighbors(self, entity_name: str) -> List[str]:
        """Return all direct in- and out-neighbors of an entity."""
        if not self.graph.has_node(entity_name):
            return []
        successors = list(self.graph.successors(entity_name))
        predecessors = list(self.graph.predecessors(entity_name))
        return list(set(successors + predecessors))

    def get_subgraph(
        self,
        seed_entities: List[str],
        max_hops: int = 1,
        max_nodes: int = 50,
    ) -> Tuple[List[Entity], List[Relationship]]:
        """Extract k-hop ego subgraph around seed entities."""
        visited_nodes: Set[str] = set()
        current_frontier = set(n for n in seed_entities if self.graph.has_node(n))
        visited_nodes.update(current_frontier)

        for _ in range(max_hops):
            next_frontier = set()
            for node in current_frontier:
                neighbors = self.get_neighbors(node)
                for nb in neighbors:
                    if nb not in visited_nodes:
                        next_frontier.add(nb)
                        visited_nodes.add(nb)
                        if len(visited_nodes) >= max_nodes:
                            break
                if len(visited_nodes) >= max_nodes:
                    break
            current_frontier = next_frontier
            if len(visited_nodes) >= max_nodes or not current_frontier:
                break

        sub_entities: List[Entity] = []
        for n in visited_nodes:
            sub_entities.append(self.get_entity(n))  # type: ignore

        sub_relations: List[Relationship] = []
        for u, v, data in self.graph.edges(data=True):
            if u in visited_nodes and v in visited_nodes:
                sub_relations.append(
                    Relationship(
                        source=u,
                        target=v,
                        relation_type=data.get("relation_type", "RELATED_TO"),
                        description=data.get("description", ""),
                        weight=data.get("weight", 1.0),
                        source_chunk_ids=data.get("source_chunk_ids", []),
                    )
                )

        return sub_entities, sub_relations

    def find_path(self, source: str, target: str) -> Optional[List[str]]:
        """Find the shortest path between two entities ignoring direction."""
        undirected = self.graph.to_undirected()
        if undirected.has_node(source) and undirected.has_node(target):
            try:
                return nx.shortest_path(undirected, source, target)
            except nx.NetworkXNoPath:
                return None
        return None

    def get_stats(self) -> GraphStats:
        """Compute graph structural statistics."""
        node_count = self.graph.number_of_nodes()
        edge_count = self.graph.number_of_edges()
        density = nx.density(self.graph) if node_count > 1 else 0.0

        # Degree centrality
        degrees = dict(self.graph.degree())
        top_entities = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
        top_list = [
            {"name": name, "degree": deg, "type": self.graph.nodes[name].get("type", "CONCEPT")}
            for name, deg in top_entities
        ]

        return GraphStats(
            node_count=node_count,
            edge_count=edge_count,
            community_count=0,
            density=round(density, 4),
            top_entities=top_list,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize graph to dictionary."""
        nodes = []
        for n, data in self.graph.nodes(data=True):
            nodes.append({"id": n, **data})
        edges = []
        for u, v, data in self.graph.edges(data=True):
            edges.append({"source": u, "target": v, **data})
        return {"nodes": nodes, "edges": edges}

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Populate graph from serialized dictionary."""
        self.graph.clear()
        for node in data.get("nodes", []):
            node_id = node.get("id")
            attrs = {k: v for k, v in node.items() if k != "id"}
            self.graph.add_node(node_id, **attrs)
        for edge in data.get("edges", []):
            u = edge.get("source")
            v = edge.get("target")
            attrs = {k: v for k, v in edge.items() if k not in ["source", "target"]}
            self.graph.add_edge(u, v, **attrs)

    def save_json(self, file_path: str) -> None:
        """Save graph data to JSON file."""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def load_json(self, file_path: str) -> None:
        """Load graph data from JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.from_dict(data)
