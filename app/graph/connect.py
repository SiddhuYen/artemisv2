"""Path-finding between two people (bidirectional, meet-in-the-middle).

Expand BOTH people's graphs depth-wise into one combined graph, then find the
best path connecting them over public person-person edges. Where their
neighborhoods overlap, a bridge node appears and a path exists.

No Claude verification — the path is evidence-grounded but unverified.
"""
from __future__ import annotations

import heapq
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import config
from ..models import Person, RelationshipEdge, Source
from ..utils.names import person_norm_key
from . import builder
from .expansion import expand_graph

# relationship strength multiplier (shared with candidate-path scoring)
REL_STRENGTH = {
    "cofounder": 1.0, "board_member": 0.95, "advisor": 0.9, "investor": 0.85,
    "employee": 0.8, "coworker": 0.8, "coauthor": 0.8, "appointee": 0.75,
    "faculty": 0.7, "student": 0.7, "author": 0.6, "speaker": 0.5,
    "interview": 0.5, "family_social": 0.45, "unknown": 0.4,
}
_STATUS_PENALTY = {"strong": 0.0, "candidate": 0.3, "raw": 1.0,
                   "weak": 2.0, "rejected": 12.0}

# A path may only traverse edges with a KNOWN relationship type and at least
# candidate-tier confidence. This stops weak co-occurrence noise (e.g. an
# 'unknown 0.35' bridge through a boilerplate/homonym node) from forming a
# path — the Fred→Cook run's junk routes were exactly such edges.
_PATH_WORTHY_STATUS = {"candidate", "strong", "accepted"}


def _path_worthy(e: RelationshipEdge) -> bool:
    return e.relationship_type != "unknown" and e.status in _PATH_WORTHY_STATUS


def _adjacency(db: Session):
    person_by_id = {p.id: p for p in db.execute(select(Person)).scalars()}
    src_by_id = {s.id: s for s in db.execute(select(Source)).scalars()}
    best: Dict[Tuple[str, str], RelationshipEdge] = {}
    for e in db.execute(
        select(RelationshipEdge).where(RelationshipEdge.person_b_id.isnot(None))
    ).scalars():
        a, b = e.person_a_id, e.person_b_id
        if not a or not b or a == b:
            continue
        if not _path_worthy(e):
            continue  # keep noise/unknown edges out of the pathable graph
        key = tuple(sorted((a, b)))
        cur = best.get(key)
        if cur is None or (e.confidence_raw or 0) > (cur.confidence_raw or 0):
            best[key] = e
    adj: Dict[str, List[Tuple[str, RelationshipEdge]]] = defaultdict(list)
    for (a, b), e in best.items():
        adj[a].append((b, e))
        adj[b].append((a, e))
    return adj, person_by_id, src_by_id


def _edge_cost(e: RelationshipEdge) -> float:
    conf = max(e.confidence_raw or 0.01, 0.01)
    return -math.log(conf) + _STATUS_PENALTY.get(e.status, 1.0)


def _best_path(adj, start: str, target: str, max_hops: int, excluded=None):
    """Best (max-confidence) path, optionally skipping `excluded` intermediate
    nodes so callers can find genuinely different routes."""
    excluded = excluded or set()
    if start == target:
        return [(start, None)]
    best_cost = {start: 0.0}
    heap = [(0.0, 0, start, [(start, None)])]
    while heap:
        cost, hops, node, path = heapq.heappop(heap)
        if node == target:
            return path
        if hops >= max_hops:
            continue
        for nbr, edge in adj.get(node, []):
            if nbr in excluded and nbr != target:
                continue
            nc = cost + _edge_cost(edge)
            if nbr not in best_cost or nc < best_cost[nbr]:
                best_cost[nbr] = nc
                heapq.heappush(heap, (nc, hops + 1, nbr, path + [(nbr, edge)]))
    return None


def _diverse_paths(adj, start: str, target: str, max_hops: int, k: int):
    """Up to k routes; each avoids all bridge (intermediate) nodes used by the
    earlier ones, so they're genuinely different."""
    paths = []
    excluded = set()
    for _ in range(k):
        hops = _best_path(adj, start, target, max_hops, excluded)
        if hops is None:
            break
        paths.append(hops)
        for pid, _edge in hops[1:-1]:  # exclude this route's bridges next time
            excluded.add(pid)
    return paths


def _score(edges: List[RelationshipEdge]) -> float:
    if not edges:
        return 1.0
    avg_conf = sum((e.confidence_raw or 0) for e in edges) / len(edges)
    avg_strength = sum(REL_STRENGTH.get(e.relationship_type, 0.4) for e in edges) / len(edges)
    return round(avg_conf * avg_strength, 3)


def connect_people(db: Session, name_a: str, name_b: str, depth: int = 2,
                   progress=None, context_a: str = "", context_b: str = "") -> dict:
    """Build both graphs, then return the best path between the two people.

    context_a / context_b disambiguate a non-notable person (e.g. "Indiana
    Pacers owner") so the search targets the right entity, not a famous namesake.
    """
    builder.reset_public_graph(db)

    # Point-to-point bridging wants STRONGEST expansion (toward shared, often
    # well-documented connections), not reachability (which walks both sides AWAY
    # from common ground). Force it for this build.
    #
    # SEPARATE per-side node budgets: side A is capped at `per_side`; then the cap
    # is raised to 2x so side B can add its own `per_side` on top — neither side
    # can starve the other. (node_count is global, so we raise the ceiling rather
    # than reset it between sides.)
    per_side = config.CONNECT_NODE_CAP_PER_SIDE
    prev_reach = config.EXPAND_PREFER_REACHABLE
    prev_cap = config.MAX_TOTAL_NODES
    config.EXPAND_PREFER_REACHABLE = False
    try:
        config.MAX_TOTAL_NODES = per_side
        if progress:
            progress(f"\n[1/2] building graph for {name_a} (depth {depth}, cap {per_side})…")
        expand_graph(db, name_a, depth, progress=progress, seed_context=context_a)

        config.MAX_TOTAL_NODES = 2 * per_side
        if progress:
            progress(f"\n[2/2] building graph for {name_b} (depth {depth}, +{per_side})…")
        expand_graph(db, name_b, depth, progress=progress, seed_context=context_b)
    finally:
        config.EXPAND_PREFER_REACHABLE = prev_reach
        config.MAX_TOTAL_NODES = prev_cap

    a = db.execute(
        select(Person).where(Person.norm_name == person_norm_key(name_a))
    ).scalar_one_or_none()
    b = db.execute(
        select(Person).where(Person.norm_name == person_norm_key(name_b))
    ).scalar_one_or_none()
    if a is None or b is None:
        missing = name_a if a is None else name_b
        return {"connected": False, "reason": f"'{missing}' not found in the graph"}

    adj, person_by_id, src_by_id = _adjacency(db)
    max_hops = 2 * depth + 1
    routes = _diverse_paths(adj, a.id, b.id, max_hops, config.CONNECT_MAX_PATHS)
    if not routes:
        return {
            "connected": False,
            "person_a": a.canonical_name, "person_b": b.canonical_name,
            "reason": f"no path within {max_hops} hops — their graphs don't overlap "
                      f"at depth {depth}. Try a higher depth.",
        }

    paths = []
    for hops in routes:
        path_nodes, edges_used, bridges = [], [], []
        for i, (pid, edge) in enumerate(hops):
            person = person_by_id.get(pid)
            label = person.canonical_name if person else pid
            node = {"label": label, "node_type": "public_person"}
            if edge is not None:
                edges_used.append(edge)
                src = src_by_id.get(edge.source_id)
                node["relationship_from_previous"] = edge.relationship_type
                node["confidence"] = edge.confidence_raw
                if edge.evidence_snippet:
                    node["evidence"] = edge.evidence_snippet
                if src and src.url:
                    node["source_url"] = src.url
            if 0 < i < len(hops) - 1:
                bridges.append(label)
            path_nodes.append(node)
        paths.append({"hops": len(hops) - 1, "score": _score(edges_used),
                      "bridges": bridges, "path": path_nodes})

    paths.sort(key=lambda p: p["score"], reverse=True)
    best = paths[0]
    return {
        "connected": True,
        "person_a": a.canonical_name,
        "person_b": b.canonical_name,
        # top-level mirrors the best path (back-compat)
        "hops": best["hops"], "score": best["score"],
        "bridges": best["bridges"], "path": best["path"],
        "paths": paths,  # all diverse routes, best first
        "warnings": ["Path is unverified", "Requires Claude verification before activation"],
    }
