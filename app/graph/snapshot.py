"""Persist a target's graph to disk for safekeeping.

Every build writes a self-contained JSON snapshot (nodes + edges + stats +
summary) to CACHED_GRAPHS_DIR, named by target + timestamp so prior runs are
never overwritten.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import List

from .. import config
from ..schemas import GraphEdge, GraphNode


def _slug(name: str) -> str:
    s = re.sub(r"[^\w]+", "_", name.strip().lower()).strip("_")
    return s or "target"


def save_graph_snapshot(
    target_name: str,
    depth: int,
    nodes: List[GraphNode],
    edges: List[GraphEdge],
    stats: dict,
    summary: dict,
) -> str:
    """Write a JSON snapshot; return its path."""
    os.makedirs(config.CACHED_GRAPHS_DIR, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    fname = f"{_slug(target_name)}__depth{depth}__{ts}.json"
    path = os.path.join(config.CACHED_GRAPHS_DIR, fname)

    payload = {
        "target": target_name,
        "depth": depth,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stats": stats,
        "summary": summary,
        "nodes": [n.model_dump() for n in nodes],
        "edges": [e.model_dump(by_alias=True) for e in edges],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return path
