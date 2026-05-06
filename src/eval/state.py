"""State persistence for cross-stage resource tracking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class StateManager:
    """Persists resource IDs and data between evaluation stages.

    State is stored as a JSON file so that setup-created resources
    (table names, record IDs, skill IDs) can be referenced during
    verify and teardown stages.
    """

    def __init__(self, state_file: str | Path = "results/state.json"):
        self.state_file = Path(state_file)
        self._state: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if self.state_file.exists():
            self._state = json.loads(self.state_file.read_text(encoding="utf-8"))

    def save(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(
            json.dumps(self._state, indent=2, default=str), encoding="utf-8"
        )

    def set(self, key: str, value: Any) -> None:
        self._state[key] = value
        self.save()

    def get(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)

    def get_all(self) -> dict[str, Any]:
        return dict(self._state)

    def clear(self) -> None:
        self._state = {}
        self.save()

    def set_resource(self, scenario_id: str, resource_type: str, resource_id: str) -> None:
        """Store a resource ID created by a scenario."""
        key = f"{scenario_id}.{resource_type}"
        self.set(key, resource_id)

    def get_resource(self, scenario_id: str, resource_type: str) -> str | None:
        """Retrieve a resource ID created by a scenario."""
        key = f"{scenario_id}.{resource_type}"
        return self.get(key)
