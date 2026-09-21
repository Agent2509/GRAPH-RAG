"""NetworkX-based embedded Knowledge Graph store."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import networkx as nx

from graph_rag.models import Entity, Relationship, GraphStats


class GraphStore:
    """Embedded Knowledge Graph store powered by NetworkX."""

    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def add_entity(self, entity: Entity) -> None:
        name = entity.name.strip()
        if not name:
            return

        if self.graph.has_node(name):
            curr_desc = self.graph.nodes[name].get("description", "")
            if entity.description and entity.description not in curr_desc:
                if not curr_desc:
                    self.graph.nodes[name]["description"] = entity.description[:600].strip()
                elif len(curr_desc) < 600:
                    self.graph.nodes[name]["description"] = f"{curr_desc} {entity.description}"[:600].strip()

            curr_chunks = set(self.graph.nodes[name].get("source_chunk_ids", []))
            curr_chunks.update(entity.source_chunk_ids)
            self.graph.nodes[name]["source_chunk_ids"] = list(curr_chunks)

            if self.graph.nodes[name].get("type") == "CONCEPT" and entity.type != "CONCEPT":
                self.graph.nodes[name]["type"] = entity.type
        else:
            self.graph.add_node(
                name,
                type=entity.type,
                description=entity.description[:600].strip(),
                source_chunk_ids=list(set(entity.source_chunk_ids)),
            )

    def add_relationship(self, rel: Relationship) -> None:
        src = rel.source.strip()
        tgt = rel.target.strip()
        if not src or not tgt or src.lower() == tgt.lower():
            return

        if not self.graph.has_node(src):
            self.graph.add_node(src, type="CONCEPT", description="", source_chunk_ids=[])
        if not self.graph.has_node(tgt):
            self.graph.add_node(tgt, type="CONCEPT", description="", source_chunk_ids=[])

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
                curr = data.get("description", "")
                if len(curr) < 400:
                    data["description"] = f"{curr} {rel.description}"[:400].strip()
            chunks = set(data.get("source_chunk_ids", []))
            chunks.update(rel.source_chunk_ids)
            data["source_chunk_ids"] = list(chunks)
        else:
            self.graph.add_edge(
                src,
                tgt,
                relation_type=rel.relation_type,
                description=rel.description[:400].strip(),
                weight=rel.weight,
                source_chunk_ids=list(set(rel.source_chunk_ids)),
            )

    def get_entity(self, name: str) -> Optional[Entity]:
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
        return [
            Entity(
                name=node,
                type=data.get("type", "CONCEPT"),
                description=data.get("description", ""),
                source_chunk_ids=data.get("source_chunk_ids", []),
            )
            for node, data in self.graph.nodes(data=True)
        ]

    def get_all_relationships(self) -> List[Relationship]:
        return [
            Relationship(
                source=u,
                target=v,
                relation_type=data.get("relation_type", "RELATED_TO"),
                description=data.get("description", ""),
                weight=data.get("weight", 1.0),
                source_chunk_ids=data.get("source_chunk_ids", []),
            )
            for u, v, data in self.graph.edges(data=True)
        ]

    def get_neighbors(self, entity_name: str) -> List[str]:
        if not self.graph.has_node(entity_name):
            return []
        return list(set(list(self.graph.successors(entity_name)) + list(self.graph.predecessors(entity_name))))

    def find_all_paths_between(self, sources: List[str], cutoff: int = 3) -> List[List[str]]:
        undirected = self.graph.to_undirected()
        paths = []
        valid_sources = [s for s in sources if undirected.has_node(s)]
        for i in range(len(valid_sources)):
            for j in range(i + 1, len(valid_sources)):
                u, v = valid_sources[i], valid_sources[j]
                try:
                    sp = nx.shortest_path(undirected, u, v)
                    if len(sp) - 1 <= cutoff:
                        paths.append(sp)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
        return paths

    def get_subgraph(
        self,
        seed_entities: List[str],
        max_hops: int = 1,
        max_nodes: int = 50,
    ) -> Tuple[List[Entity], List[Relationship]]:
        visited_nodes: Set[str] = set()
        current_frontier = set(n for n in seed_entities if self.graph.has_node(n))
        visited_nodes.update(current_frontier)

        for _ in range(max_hops):
            next_frontier = set()
            for node in current_frontier:
                for nb in self.get_neighbors(node):
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

        sub_entities = [self.get_entity(n) for n in visited_nodes if self.get_entity(n) is not None]  # type: ignore

        sub_relations = [
            Relationship(
                source=u,
                target=v,
                relation_type=data.get("relation_type", "RELATED_TO"),
                description=data.get("description", ""),
                weight=data.get("weight", 1.0),
                source_chunk_ids=data.get("source_chunk_ids", []),
            )
            for u, v, data in self.graph.edges(data=True)
            if u in visited_nodes and v in visited_nodes
        ]

        return sub_entities, sub_relations

    def find_path(self, source: str, target: str) -> Optional[List[str]]:
        undirected = self.graph.to_undirected()
        if undirected.has_node(source) and undirected.has_node(target):
            try:
                return nx.shortest_path(undirected, source, target)
            except nx.NetworkXNoPath:
                return None
        return None

    def get_stats(self) -> GraphStats:
        node_count = self.graph.number_of_nodes()
        edge_count = self.graph.number_of_edges()
        density = nx.density(self.graph) if node_count > 1 else 0.0

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
        nodes = [{"id": n, **data} for n, data in self.graph.nodes(data=True)]
        edges = [{"source": u, "target": v, **data} for u, v, data in self.graph.edges(data=True)]
        return {"nodes": nodes, "edges": edges}

    def from_dict(self, data: Dict[str, Any]) -> None:
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
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def load_json(self, file_path: str) -> None:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.from_dict(data)
