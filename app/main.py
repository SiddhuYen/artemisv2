"""FastAPI surface for Artemis V2.

Endpoints:
  POST /targets/search   run search -> extraction -> graph build -> expansion
  GET  /graph            full node/edge graph
  GET  /people           list discovered people
  GET  /edges            list relationship edges
  GET  /health           liveness + extractor mode
"""
from __future__ import annotations

import os
import threading

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import get_db, init_db
from .extraction import ollama_available
from .graph.expansion import expand_graph
from .models import (
    CandidatePath,
    GraphMatch,
    LocalProfile,
    Organization,
    Person,
    RelationshipEdge,
    Source,
)
from .network.ingest import ingest_csv
from .network.matching import run_matching
from .network.paths import generate_paths_for_target
from .schemas import GraphResponse, GraphStats, TargetSearchRequest
from .serializers import (
    build_summary,
    serialize_edges,
    serialize_neighborhood,
    serialize_nodes,
)

app = FastAPI(
    title="Artemis V2 — Public Relationship Graph Builder",
    version="0.1.0",
    description="Discovers public relationships between people/orgs from open sources. "
    "MVP scope: search -> extraction -> graph building -> expansion. "
    "No external-network matching and no Claude verification (deferred).",
)


@app.on_event("startup")
def _startup() -> None:
    init_db()


# Serialize graph builds so the config-global mutation inside connect_people()
# and concurrent writers can't race. Builds are minutes-long and Brave-rate-
# limited anyway, so serialization costs little and guarantees correctness.
# (Reads/pathfinding still run concurrently — WAL lets them proceed during a build.)
_BUILD_LOCK = threading.Lock()


# ONE shared global graph for the whole team: every run accumulates into it, and
# pathfinding runs over the union so a route can pass through people other runs
# discovered. get_db yields a session on the single default engine.
@app.get("/", include_in_schema=False)
def _root() -> RedirectResponse:
    return RedirectResponse(url="/ui/")


@app.get("/healthz", include_in_schema=False)
def healthz() -> dict:
    # cheap liveness probe for load balancers / Render health checks —
    # no dependency I/O (does NOT probe Ollama), so it can't hang or flap.
    return {"status": "ok"}


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "extractor": "ollama" if ollama_available() else "heuristic",
    }


@app.get("/status")
def status() -> dict:
    """Live service status for the UI — surfaces Brave search quota so the site
    can warn when results are degraded (Brave exhausted -> fallback provider)."""
    from .providers.brave import brave_status
    return {
        "brave": brave_status(),
        "extractor": "ollama" if ollama_available() else "heuristic",
    }


@app.post("/targets/search", response_model=GraphResponse)
def targets_search(req: TargetSearchRequest, db: Session = Depends(get_db)) -> GraphResponse:
    # ADDITIVE: accumulate the searched person into the shared global map (no
    # reset), then return only that person's neighborhood (not the whole map).
    with _BUILD_LOCK:
        stats = expand_graph(db, req.target_name, req.max_depth)
    nodes, edges = serialize_neighborhood(db, req.target_name, req.max_depth)
    return GraphResponse(
        graph_id="global",
        nodes=nodes,
        edges=edges,
        stats=GraphStats(**stats),
    )


@app.get("/graph", response_model=GraphResponse)
def get_graph(db: Session = Depends(get_db)) -> GraphResponse:
    stats = GraphStats(
        people_found=db.query(Person).count(),
        organizations_found=db.query(Organization).count(),
        edges_found=db.query(RelationshipEdge).count(),
        sources_fetched=db.query(Source).count(),
    )
    return GraphResponse(
        graph_id="global",
        nodes=serialize_nodes(db),
        edges=serialize_edges(db),
        stats=stats,
    )


@app.get("/people")
def list_people(db: Session = Depends(get_db)) -> list:
    out = []
    for p in db.execute(select(Person)).scalars():
        out.append(
            {
                "id": p.id,
                "canonical_name": p.canonical_name,
                "aliases": p.aliases or [],
                "metadata": p.meta or {},
                "created_at": p.created_at,
            }
        )
    return out


@app.get("/edges")
def list_edges(db: Session = Depends(get_db)) -> list:
    return [e.model_dump(by_alias=True) for e in serialize_edges(db)]


@app.get("/summary")
def graph_summary(db: Session = Depends(get_db)) -> dict:
    """Top people/orgs, strongest edges, and confidence distribution."""
    return build_summary(serialize_nodes(db), serialize_edges(db))


@app.post("/connect")
def connect(req: dict, db: Session = Depends(get_db)) -> dict:
    """Find a path between two people (builds both graphs, meets in the middle).
    Body: {"person_a": "...", "person_b": "...", "depth": 2}"""
    from .graph.connect import connect_people
    a = (req.get("person_a") or "").strip()
    b = (req.get("person_b") or "").strip()
    try:
        depth = max(1, min(int(req.get("depth", 2)), 3))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="depth must be an integer 1-3")
    if not a or not b:
        raise HTTPException(status_code=400, detail="person_a and person_b required")
    with _BUILD_LOCK:  # connect_people mutates config globals — serialize builds
        result = connect_people(db, a, b, depth,
                                context_a=(req.get("context_a") or "").strip(),
                                context_b=(req.get("context_b") or "").strip())
    result["graph_id"] = "global"
    return result


# ===========================================================================
# Local network matching (no Claude verification — candidate paths only)
# NOTE: these still use the DEFAULT engine (get_db), not the per-session graph.
# They're stage-2 (not in the beta UI); session-scoping them is future work.
# ===========================================================================
def _profile_dict(p: LocalProfile) -> dict:
    return {
        "id": p.id, "canonical_name": p.canonical_name, "aliases": p.aliases or [],
        "email": p.email, "linkedin_url": p.linkedin_url,
        "companies": p.companies or [], "titles": p.titles or [],
        "schools": p.schools or [], "locations": p.locations or [],
        "notes": p.notes,
    }


def _match_dict(m: GraphMatch) -> dict:
    return {
        "id": m.id, "local_profile_id": m.local_profile_id,
        "public_person_id": m.public_person_id, "public_org_id": m.public_org_id,
        "match_type": m.match_type, "confidence": m.confidence,
        "explanation": m.explanation,
    }


@app.post("/network/upload")
async def network_upload(file: UploadFile = File(...), db: Session = Depends(get_db)) -> dict:
    raw = await file.read()
    content = raw.decode("utf-8", errors="replace")
    stats = ingest_csv(db, content)
    return {"ingested": stats, "profiles_total": db.query(LocalProfile).count()}


@app.get("/network/profiles")
def network_profiles(db: Session = Depends(get_db)) -> list:
    return [_profile_dict(p) for p in db.execute(select(LocalProfile)).scalars()]


@app.post("/match/{target_person_id}")
def match_target(target_person_id: str, db: Session = Depends(get_db)) -> dict:
    target = db.get(Person, target_person_id)
    if target is None:
        raise HTTPException(status_code=404, detail="target person not found")
    matches = run_matching(db)
    paths = generate_paths_for_target(db, target_person_id)
    by_type: dict = {}
    for m in matches:
        by_type[m.match_type] = by_type.get(m.match_type, 0) + 1
    return {
        "target": target.canonical_name,
        "target_person_id": target_person_id,
        "matches": len(matches),
        "matches_by_type": by_type,
        "candidate_paths": len(paths),
        "note": "All candidate paths are UNVERIFIED. Claude verification not run.",
    }


@app.get("/matches")
def list_matches(db: Session = Depends(get_db)) -> list:
    return [_match_dict(m) for m in db.execute(select(GraphMatch)).scalars()]


@app.get("/candidate-paths")
def list_candidate_paths(db: Session = Depends(get_db)) -> list:
    rows = db.execute(select(CandidatePath).order_by(CandidatePath.score.desc())).scalars()
    return [
        {"id": c.id, "target_person_id": c.target_person_id, "score": c.score,
         "status": c.status, "path": c.path_json}
        for c in rows
    ]


@app.get("/candidate-paths/{path_id}")
def get_candidate_path(path_id: str, db: Session = Depends(get_db)) -> dict:
    c = db.get(CandidatePath, path_id)
    if c is None:
        raise HTTPException(status_code=404, detail="candidate path not found")
    return {"id": c.id, "target_person_id": c.target_person_id, "score": c.score,
            "status": c.status, "path": c.path_json}


# --- static frontend (mounted last so it never shadows the API routes) ------
_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/ui", StaticFiles(directory=_STATIC_DIR, html=True), name="ui")
