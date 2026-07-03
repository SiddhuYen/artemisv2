"""Candidate-path generation: You -> Local Profile -> Public Person -> ... -> Target.

Best-path search over public person-person relationship_edges (max 4 public hops),
preferring higher-confidence, stronger, non-rejected edges. Produces
status='unverified' paths only — nothing is asserted as a real intro.
"""
from __future__ import annotations

import heapq
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import (
    CandidatePath,
    GraphMatch,
    LocalProfile,
    Person,
    RelationshipEdge,
    Source,
)

MAX_PUBLIC_HOPS = 4

# relationship strength multiplier (how much a relationship type "carries" an intro)
REL_STRENGTH = {
    "cofounder": 1.0, "board_member": 0.95, "advisor": 0.9, "investor": 0.85,
    "employee": 0.8, "coworker": 0.8, "coauthor": 0.8, "appointee": 0.75,
    "faculty": 0.7, "student": 0.7, "author": 0.6, "speaker": 0.5,
    "interview": 0.5, "family_social": 0.45, "unknown": 0.4,
}
_STATUS_PENALTY = {"strong": 0.0, "candidate": 0.3, "raw": 1.0,
                   "weak": 2.0, "rejected": 12.0}

WARNINGS = ["Path is unverified", "Requires Claude verification before activation"]


class _PublicEdges:
    def __init__(self, db: Session) -> None:
        self.person_by_id: Dict[str, Person] = {
            p.id: p for p in db.execute(select(Person)).scalars()
        }
        self.src_by_id: Dict[str, Source] = {
            s.id: s for s in db.execute(select(Source)).scalars()
        }
        # best (highest-confidence) edge between each unordered person pair
        best: Dict[Tuple[str, str], RelationshipEdge] = {}
        for e in db.execute(
            select(RelationshipEdge).where(RelationshipEdge.person_b_id.isnot(None))
        ).scalars():
            a, b = e.person_a_id, e.person_b_id
            if not a or not b or a == b:
                continue
            key = tuple(sorted((a, b)))
            cur = best.get(key)
            if cur is None or (e.confidence_raw or 0) > (cur.confidence_raw or 0):
                best[key] = e
        self.adj: Dict[str, List[Tuple[str, RelationshipEdge]]] = defaultdict(list)
        for (a, b), e in best.items():
            self.adj[a].append((b, e))
            self.adj[b].append((a, e))


def _edge_cost(e: RelationshipEdge) -> float:
    conf = max(e.confidence_raw or 0.01, 0.01)
    return -math.log(conf) + _STATUS_PENALTY.get(e.status, 1.0)


def _best_path(pe: _PublicEdges, start: str, target: str):
    """Return [(person_id, edge_used_to_reach_or_None), ...] or None."""
    if start == target:
        return [(start, None)]
    best_cost: Dict[str, float] = {start: 0.0}
    heap = [(0.0, 0, start, [(start, None)])]
    while heap:
        cost, hops, node, path = heapq.heappop(heap)
        if node == target:
            return path
        if hops >= MAX_PUBLIC_HOPS:
            continue
        for nbr, edge in pe.adj.get(node, []):
            ncost = cost + _edge_cost(edge)
            if nbr not in best_cost or ncost < best_cost[nbr]:
                best_cost[nbr] = ncost
                heapq.heappush(heap, (ncost, hops + 1, nbr, path + [(nbr, edge)]))
    return None


def _score(local_conf: float, edges: List[RelationshipEdge]) -> float:
    if not edges:
        return round(local_conf, 3)
    avg_conf = sum((e.confidence_raw or 0) for e in edges) / len(edges)
    avg_strength = sum(REL_STRENGTH.get(e.relationship_type, 0.4) for e in edges) / len(edges)
    return round(local_conf * avg_conf * avg_strength, 3)


def _build_path_json(pe, profile, match, target_id, hop_pairs) -> dict:
    nodes = [
        {"node_type": "you", "label": "You"},
        {"node_type": "local_profile", "label": profile.canonical_name,
         "reason": "Uploaded network contact"},
    ]
    edges_used: List[RelationshipEdge] = []
    for i, (pid, edge) in enumerate(hop_pairs):
        person = pe.person_by_id.get(pid)
        label = person.canonical_name if person else pid
        if i == 0:
            node = {"node_type": "public_person", "label": label,
                    "reason": f"Matched: {match.explanation}"}
        else:
            edges_used.append(edge)
            src = pe.src_by_id.get(edge.source_id) if edge else None
            node = {"node_type": "public_person", "label": label,
                    "reason": f"Public graph edge ({edge.relationship_type})"}
            if src and src.url:
                node["source_url"] = src.url
        if pid == target_id:
            node["reason"] += "  [target]"
        nodes.append(node)

    return {
        "status": "unverified",
        "score": _score(match.confidence, edges_used),
        "path": nodes,
        "warnings": list(WARNINGS),
    }, edges_used


def generate_paths_for_target(db: Session, target_id: str) -> List[CandidatePath]:
    """Recompute candidate paths from local matches to one target person."""
    db.query(CandidatePath).filter(CandidatePath.target_person_id == target_id).delete()
    db.flush()

    pe = _PublicEdges(db)
    if target_id not in pe.person_by_id:
        return []

    # only person-level matches yield paths (org_overlap is a near-miss bridge)
    matches = list(db.execute(
        select(GraphMatch).where(GraphMatch.public_person_id.isnot(None))
    ).scalars())
    profiles = {p.id: p for p in db.execute(select(LocalProfile)).scalars()}

    created: List[CandidatePath] = []
    for match in matches:
        profile = profiles.get(match.local_profile_id)
        if profile is None:
            continue
        hop_pairs = _best_path(pe, match.public_person_id, target_id)
        if hop_pairs is None:
            continue  # matched person not connected to target within hop limit
        path_json, edges_used = _build_path_json(pe, profile, match, target_id, hop_pairs)
        cp = CandidatePath(
            target_person_id=target_id,
            local_profile_id=profile.id,
            public_person_id=match.public_person_id,
            path_json=path_json,
            score=path_json["score"],
            status="unverified",
        )
        db.add(cp)
        created.append(cp)

    db.commit()
    created.sort(key=lambda c: c.score, reverse=True)
    return created
