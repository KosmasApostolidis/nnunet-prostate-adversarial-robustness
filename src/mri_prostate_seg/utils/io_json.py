"""JSON read/write helpers used by experiment runners and CLI scripts."""

from __future__ import annotations

import json
import os
from typing import Any


def write_json(path: str, payload: Any, *, indent: int = 4) -> None:
    """Write ``payload`` to ``path`` as JSON, creating parent dirs if needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=indent)


def read_json(path: str) -> Any:
    """Read JSON from ``path`` and return the parsed object."""
    with open(path, "r") as fh:
        return json.load(fh)


def update_json(path: str, updates: dict[str, Any]) -> dict[str, Any]:
    """Merge ``updates`` into the dict stored at ``path`` and rewrite.

    If ``path`` does not exist it is created with ``updates`` as initial content.
    Returns the merged dict for caller convenience.
    """
    if os.path.exists(path):
        with open(path, "r") as fh:
            current = json.load(fh)
        if not isinstance(current, dict):
            raise TypeError(
                f"update_json expects a dict at the top level of {path}, got {type(current).__name__}"
            )
    else:
        current = {}

    current.update(updates)
    write_json(path, current)
    return current
