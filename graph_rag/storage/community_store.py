"""Storage for hierarchical graph communities and their summary reports."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from graph_rag.models import Community


class CommunityStore:
    """Stores detected communities and associated summaries."""

    def __init__(self):
        self.communities: Dict[int, Community] = {}

    def add_community(self, community: Community) -> None:
        """Add or overwrite a community cluster."""
        self.communities[community.id] = community

    def get_community(self, community_id: int) -> Optional[Community]:
        """Retrieve community by ID."""
        return self.communities.get(community_id)

    def get_all_communities(self) -> List[Community]:
        """Return all stored communities."""
        return list(self.communities.values())

    def to_dict(self) -> List[Dict[str, Any]]:
        """Serialize communities to list of dictionaries."""
        return [c.model_dump() for c in self.communities.values()]

    def from_dict(self, data: List[Dict[str, Any]]) -> None:
        """Load communities from dictionary list."""
        self.communities.clear()
        for item in data:
            c = Community(**item)
            self.communities[c.id] = c

    def save_json(self, file_path: str) -> None:
        """Save communities to JSON."""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def load_json(self, file_path: str) -> None:
        """Load communities from JSON."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.from_dict(data)

    def __len__(self) -> int:
        return len(self.communities)
