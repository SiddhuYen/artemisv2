"""Turn ORM rows into the node/edge wire format required by the spec, plus a
graph summary (top entities / strongest edges / confidence distribution)."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import Organization, Person, RelationshipEdge, Source
from .schemas import GraphEdge, GraphNode


def serialize_nodes(db: Session) -> List[GraphNode]:
    nodes: List[GraphNode] = []
    for p in db.execute(select(Person)).scalars():
        nodes.append(GraphNode(id=p.id, label=p.canonical_name, kind="person"))
    for o in db.execute(select(Organization)).scalars():
        nodes.append(GraphNode(id=o.id, label=o.name, kind="organization", type=o.type))
    return nodes


def serialize_edges(db: Session) -> List[GraphEdge]:
    # preload sources for url lookup
    src_map = {s.id: s for s in db.execute(select(Source)).scalars()}
    edges: List[GraphEdge] = []
    for e in db.execute(select(RelationshipEdge)).scalars():
        to_id = e.person_b_id or e.organization_id
        if not to_id:
            continue
        src = src_map.get(e.source_id)
        edges.append(
            GraphEdge(
                id=e.id,
                **{"from": e.person_a_id},
                to=to_id,
                type=e.relationship_type,
                confidence=e.confidence_raw or 0.0,
                source_url=src.url if src else None,
                status=e.status,
                method=e.method,
                evidence=e.evidence_snippet,
                depth=e.depth or 0,
            )
        )
    return edges


# --- graph summary ---------------------------------------------------------
def _strongest_unique_edges(edges: List[GraphEdge]):
    """Collapse to one edge per (from, to, type), keeping the highest confidence."""
    best: Dict[tuple, GraphEdge] = {}
    for e in edges:
        key = (e.from_, e.to, e.type)
        cur = best.get(key)
        if cur is None or e.confidence > cur.confidence:
            best[key] = e
    return sorted(best.values(), key=lambda e: e.confidence, reverse=True)


def build_summary(nodes: List[GraphNode], edges: List[GraphEdge], top_n: int = 10) -> dict:
    label = {n.id: n.label for n in nodes}
    kind = {n.id: n.kind for n in nodes}

    tiers = {"strong": 0, "candidate": 0, "weak": 0, "other": 0}
    buckets = {"0.0-0.3": 0, "0.3-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
    for e in edges:
        if e.status in tiers:
            tiers[e.status] += 1
        else:
            tiers["other"] += 1
        c = e.confidence
        if c < 0.3:
            buckets["0.0-0.3"] += 1
        elif c < 0.6:
            buckets["0.3-0.6"] += 1
        elif c < 0.8:
            buckets["0.6-0.8"] += 1
        else:
            buckets["0.8-1.0"] += 1

    # per-node degree + strength (touch count + summed confidence)
    degree: Dict[str, int] = defaultdict(int)
    strength: Dict[str, float] = defaultdict(float)
    for e in edges:
        for nid in (e.from_, e.to):
            degree[nid] += 1
            strength[nid] += e.confidence

    def _rank(node_kind: str):
        ranked = sorted(
            (nid for nid in degree if kind.get(nid) == node_kind),
            key=lambda nid: (strength[nid], degree[nid]),
            reverse=True,
        )
        return [
            {"id": nid, "label": label.get(nid, nid),
             "degree": degree[nid], "strength": round(strength[nid], 2)}
            for nid in ranked[:top_n]
        ]

    strongest = [
        {
            "from": label.get(e.from_, e.from_),
            "to": label.get(e.to, e.to),
            "type": e.type,
            "confidence": e.confidence,
            "status": e.status,
            "source_url": e.source_url,
        }
        for e in _strongest_unique_edges(edges)[:top_n]
    ]

    return {
        "nodes": len(nodes),
        "edges": len(edges),
        "tiers": {"strong": tiers["strong"], "candidate": tiers["candidate"],
                  "weak": tiers["weak"], "other": tiers["other"]},
        "confidence_distribution": buckets,
        "top_people": _rank("person"),
        "top_organizations": _rank("organization"),
        "strongest_edges": strongest,
    }
