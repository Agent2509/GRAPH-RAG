"""Interactive high-fidelity physics network graph visualization using PyVis & Vis.js."""

from typing import Dict, List, Optional, Set
from pyvis.network import Network
from graph_rag.storage.graph_store import GraphStore

COMMUNITY_COLORS = [
    "#6366f1",  # Electric Indigo
    "#06b6d4",  # Cyan
    "#10b981",  # Emerald Green
    "#f59e0b",  # Amber Gold
    "#f43f5e",  # Radiant Rose
    "#8b5cf6",  # Violet Purple
    "#38bdf8",  # Neon Sky
    "#ec4899",  # Vivid Pink
    "#14b8a6",  # Vibrant Teal
    "#f97316",  # Bright Orange
]

TYPE_STYLES: Dict[str, Dict[str, str]] = {
    "ORGANIZATION": {"color": "#38bdf8", "border": "#0284c7", "shape": "dot", "icon": "🏢"},
    "COMPANY": {"color": "#38bdf8", "border": "#0284c7", "shape": "dot", "icon": "🏢"},
    "PERSON": {"color": "#f43f5e", "border": "#e11d48", "shape": "dot", "icon": "👤"},
    "DOCUMENT": {"color": "#06b6d4", "border": "#0891b2", "shape": "box", "icon": "📄"},
    "SECTION": {"color": "#06b6d4", "border": "#0891b2", "shape": "box", "icon": "📑"},
    "PAGE": {"color": "#0ea5e9", "border": "#0369a1", "shape": "box", "icon": "📄"},
    "TECHNOLOGY": {"color": "#10b981", "border": "#059669", "shape": "diamond", "icon": "⚙️"},
    "PRODUCT": {"color": "#10b981", "border": "#059669", "shape": "diamond", "icon": "📦"},
    "CONCEPT": {"color": "#a855f7", "border": "#7e22ce", "shape": "dot", "icon": "💡"},
    "REQUIREMENT": {"color": "#c084fc", "border": "#9333ea", "shape": "dot", "icon": "📋"},
    "LOCATION": {"color": "#f59e0b", "border": "#d97706", "shape": "star", "icon": "📍"},
    "EVENT": {"color": "#ec4899", "border": "#db2777", "shape": "triangle", "icon": "📅"},
}

DEFAULT_TYPE_STYLE = {"color": "#818cf8", "border": "#4f46e5", "shape": "dot", "icon": "🔹"}


class GraphVisualizer:
    """Renders sleek, interactive 2D physics network graphs."""

    def __init__(self, height: str = "680px", width: str = "100%"):
        self.height = height
        self.width = width

    def generate_html(
        self,
        graph_store: GraphStore,
        highlight_nodes: Optional[List[str]] = None,
        filter_community_id: Optional[int] = None,
        min_degree: int = 0,
        color_by: str = "type",
        node_limit: int = 500,
    ) -> str:
        """Generate high-fidelity interactive HTML network visualization."""
        net = Network(
            height=self.height,
            width=self.width,
            directed=True,
            bgcolor="#080c14",  # Obsidian dark canvas
            font_color="#ffffff",
        )

        # High-end physics and interaction configuration
        net.set_options("""
        {
          "nodes": {
            "borderWidth": 2,
            "borderWidthSelected": 4,
            "shadow": {
              "enabled": true,
              "color": "rgba(0, 0, 0, 0.75)",
              "size": 12,
              "x": 2,
              "y": 2
            },
            "font": {
              "color": "#ffffff",
              "size": 13,
              "face": "system-ui, -apple-system, sans-serif",
              "strokeWidth": 3,
              "strokeColor": "#080c14"
            }
          },
          "edges": {
            "color": {
              "color": "rgba(56, 189, 248, 0.38)",
              "highlight": "#00f0ff",
              "hover": "#38bdf8"
            },
            "font": {
              "color": "#94a3b8",
              "size": 10,
              "face": "system-ui, sans-serif",
              "strokeWidth": 2,
              "strokeColor": "#080c14",
              "align": "horizontal"
            },
            "smooth": {
              "type": "curvedCW",
              "roundness": 0.22
            },
            "arrows": {
              "to": {
                "enabled": true,
                "scaleFactor": 0.7
              }
            }
          },
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -3200,
              "centralGravity": 0.28,
              "springLength": 115,
              "springConstant": 0.04,
              "damping": 0.09,
              "avoidOverlap": 0.35
            },
            "maxVelocity": 38,
            "solver": "barnesHut",
            "timestep": 0.4,
            "stabilization": {
              "enabled": true,
              "iterations": 180
            }
          },
          "interaction": {
            "hover": true,
            "navigationButtons": false,
            "keyboard": true,
            "zoomView": true,
            "tooltipDelay": 100,
            "hideEdgesOnDrag": false
          }
        }
        """)

        highlight_set = set(highlight_nodes or [])
        graph = graph_store.graph

        if graph.number_of_nodes() == 0:
            return net.generate_html()

        degrees = dict(graph.degree())
        max_deg = max(degrees.values()) if degrees and max(degrees.values()) > 0 else 1

        # Candidate nodes filtered by community and degree
        candidate_nodes = []
        for node, data in graph.nodes(data=True):
            comm_id = data.get("community_id", 0)
            if filter_community_id is not None and comm_id != filter_community_id:
                continue
            deg = degrees.get(node, 0)
            if deg < min_degree:
                continue
            candidate_nodes.append((node, data, deg, comm_id))

        # Sort by degree descending and limit if necessary
        candidate_nodes.sort(key=lambda x: x[2], reverse=True)
        candidate_nodes = candidate_nodes[:node_limit]
        candidate_set = {x[0] for x in candidate_nodes}

        # Add nodes to PyVis network
        for node, data, deg, comm_id in candidate_nodes:
            raw_type = str(data.get("type", "CONCEPT")).strip().upper()
            type_style = TYPE_STYLES.get(raw_type, DEFAULT_TYPE_STYLE)
            desc = data.get("description", "No description available.")

            # Dynamic node sizing based on degree (scaled 18 to 46)
            size = 18 + int(28 * (deg / max_deg))

            # Color scheme selection
            if color_by == "community":
                bg_color = COMMUNITY_COLORS[comm_id % len(COMMUNITY_COLORS)]
                border_color = "#ffffff" if node in highlight_set else bg_color
            else:
                bg_color = type_style["color"]
                border_color = "#ffffff" if node in highlight_set else type_style["border"]

            shape = type_style.get("shape", "dot")

            # Enhanced HTML Tooltip Card
            tooltip = (
                f"<div style='font-family: system-ui; padding: 6px; font-size: 12px; color: #fff; max-width: 280px;'>"
                f"<div style='font-weight: bold; font-size: 14px; margin-bottom: 3px; color: #38bdf8;'>{type_style.get('icon', '')} {node}</div>"
                f"<div style='color: #94a3b8; margin-bottom: 4px;'>Type: <b style='color: #f8fafc;'>{raw_type}</b> • Degree: <b>{deg}</b> • Cluster: <b>{comm_id}</b></div>"
                f"<div style='font-size: 11px; line-height: 1.4; color: #cbd5e1;'>{desc[:220]}</div>"
                f"</div>"
            )

            is_hub = (deg >= 4)
            label_text = f"★ {node}" if is_hub else node

            net.add_node(
                node,
                label=label_text,
                title=tooltip,
                size=size + (10 if node in highlight_set else 0),
                shape=shape,
                color={
                    "background": bg_color,
                    "border": border_color,
                    "highlight": {
                        "background": "#00f0ff",
                        "border": "#ffffff",
                    },
                },
                borderWidth=4 if node in highlight_set else (3 if is_hub else 2),
                font={
                    "color": "#ffffff",
                    "size": 15 if is_hub else 12,
                    "strokeWidth": 4 if is_hub else 2,
                    "strokeColor": "#080c14",
                },
            )

        # Add edges
        added_edges: Set[tuple] = set()
        for u, v, data in graph.edges(data=True):
            if u not in candidate_set or v not in candidate_set:
                continue

            rel_type = data.get("relation_type", "RELATED_TO")
            desc = data.get("description", "")
            weight = data.get("weight", 1.0)

            edge_key = (u, v, rel_type)
            if edge_key in added_edges:
                continue
            added_edges.add(edge_key)

            edge_title = (
                f"<div style='font-family: system-ui; padding: 4px; font-size: 11px; color: #fff; max-width: 240px;'>"
                f"<b style='color: #38bdf8;'>{rel_type}</b>"
                f"<div style='color: #cbd5e1; margin-top: 2px;'>{desc}</div>"
                f"</div>"
            )

            net.add_edge(
                u,
                v,
                label=rel_type.lower().replace("_", " "),
                title=edge_title,
                value=min(weight, 4.0),
            )

        raw_html = net.generate_html()

        # Inject interactive click-to-focus neighborhood script
        focus_script = """
        <script type="text/javascript">
        if (typeof network !== 'undefined' && typeof nodes !== 'undefined') {
          network.on('click', function(params) {
            if (params.nodes.length > 0) {
              var selected = params.nodes[0];
              var connected = network.getConnectedNodes(selected);
              connected.push(selected);
              var updateArray = [];
              nodes.forEach(function(node) {
                if (connected.indexOf(node.id) !== -1) {
                  updateArray.push({id: node.id, opacity: 1.0, font: {color: '#ffffff'}});
                } else {
                  updateArray.push({id: node.id, opacity: 0.16, font: {color: 'rgba(255,255,255,0.18)'}});
                }
              });
              nodes.update(updateArray);
            } else {
              var resetArray = [];
              nodes.forEach(function(node) {
                resetArray.push({id: node.id, opacity: 1.0, font: {color: '#ffffff'}});
              });
              nodes.update(resetArray);
            }
          });
        }
        </script>
        """

        if "</body>" in raw_html:
            return raw_html.replace("</body>", f"{focus_script}\n</body>")
        return raw_html + focus_script
