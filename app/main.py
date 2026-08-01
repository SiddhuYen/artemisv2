"""FastAPI surface for Artemis V2.

Every endpoint that builds a graph is asynchronous: it admits or refuses the
work immediately and returns a job id, because a build is minutes of live web
crawling and no managed host's proxy will hold a request open that long.

  POST /targets/search   build a target's neighborhood      -> {"job_id"}
  POST /discover         expand one person's network        -> {"job_id"}
  POST /connect          find a path between two people     -> {"job_id"}
  GET  /jobs/{id}        status, percent, queue position, result
  POST /jobs/{id}/cancel stop a running or queued job
  GET  /graph            full node/edge graph
  GET  /people           list discovered people
  GET  /edges            list relationship edges
  GET  /status           search providers, Claude stages, auth, build queue
  GET  /health           liveness + extractor mode
  GET  /healthz          bare liveness (never gated, no dependency I/O)
  POST /login /logout    exchange the shared token for a session cookie

Admission control lives in two layers: `auth` (who may call at all, and how
often) and `buildqueue` (how many builds run at once, and how many may wait).
"""
from __future__ import annotations

import os
import threading
import time
import uuid
from collections import defaultdict

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from . import auth, config
from .buildqueue import BUILDS, QueueFull
from .db import SessionLocal, get_boards_db, get_db, init_boards_db, init_db, safe_graph_id
from .extraction import claude_available
from .graph.expansion import expand_graph
from .models import (
    Board,
    BoardPage,
    CandidatePath,
    GraphMatch,
    LocalEdge,
    EnrichmentRun,
    LocalProfile,
    Person,
    Source,
)
from .network.cliques import materialize_contact_cliques
from .network.enrichment import RunConflict, cancel_run, plan_run, run_dict
from .network.executor import STARTABLE, claim_run, execute_run
from .network.ingest import backfill_graph_edges, ingest_csv, ingest_rows
from .network.owner import get_owner, owner_dict, upsert_owner
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
    init_boards_db()
    from .providers import cache as provider_cache
    provider_cache.purge_expired()  # bounds the SQLite cache file across restarts;
    # the CLI path already does this per-run (cli.py's run()), the server path never did


@app.on_event("shutdown")
def _shutdown() -> None:
    from .providers import browser
    browser.shutdown()  # no-op if Playwright was never launched


# ---------------------------------------------------------------------------
# Access control. One shared token gates the whole surface — UI and API — when
# config.ACCESS_TOKEN is set; unset, the app is open (a local checkout).
#
# Middleware rather than a per-route dependency so it also covers the mounted
# static UI and anything added later: a new endpoint is protected by default
# instead of protected only if someone remembered the dependency.
# ---------------------------------------------------------------------------
# Reachable without the token. /healthz so a load balancer can probe a locked
# app; the login route so there is a way to GET the form and POST the token.
_PUBLIC_PATHS = {"/healthz", "/login", "/logout"}


def _wants_html(request: Request) -> bool:
    return "text/html" in (request.headers.get("accept") or "")


@app.middleware("http")
async def _require_token(request: Request, call_next):
    if not auth.enabled() or request.url.path in _PUBLIC_PATHS:
        return await call_next(request)
    if auth.request_authenticated(request.headers, request.cookies):
        return await call_next(request)
    # A browser navigating to a page gets sent to the login form; an API client
    # gets a machine-readable 401 it can act on rather than an HTML redirect it
    # would have to sniff.
    if _wants_html(request):
        return RedirectResponse(url="/login", status_code=303)
    return JSONResponse({"detail": "authentication required"}, status_code=401)


def _client_key(request: Request) -> str:
    return auth.client_key(request.client.host if request.client else "", request.headers)


def _enforce_rate_limit(request: Request) -> None:
    """Charge one build token, or 429. Applied to the endpoints that spend
    money, never to reads — a throttled dashboard is just a broken dashboard."""
    allowed, retry_after = auth.build_limiter.check(_client_key(request))
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=(f"rate limit reached ({config.BUILD_RATE_LIMIT} builds per "
                    f"{int(config.BUILD_RATE_WINDOW_S // 60)} min) — "
                    f"retry in {int(retry_after)}s"),
            headers={"Retry-After": str(int(retry_after))},
        )


_LOGIN_PAGE = """<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Artemis — sign in</title>
<style>
  :root { color-scheme: light dark; }
  body { font: 15px/1.5 ui-sans-serif, system-ui, sans-serif; display: grid;
         place-items: center; min-height: 100vh; margin: 0; }
  form { display: grid; gap: .75rem; width: min(22rem, 90vw); }
  h1 { font-size: 1.1rem; margin: 0 0 .25rem; }
  input, button { font: inherit; padding: .6rem .7rem; border-radius: .4rem;
                  border: 1px solid color-mix(in srgb, currentColor 30%, transparent); }
  button { cursor: pointer; font-weight: 600; }
  .err { color: #c0392b; min-height: 1.2em; font-size: .9em; }
</style>
<form id="f">
  <h1>Artemis</h1>
  <label for="t">Access token</label>
  <input id="t" type="password" autocomplete="current-password" autofocus>
  <button type="submit">Sign in</button>
  <div class="err" id="e"></div>
</form>
<script>
document.getElementById('f').addEventListener('submit', async ev => {
  ev.preventDefault();
  const e = document.getElementById('e');
  e.textContent = '';
  const res = await fetch('/login', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ token: document.getElementById('t').value }),
  });
  if (res.ok) { location.href = '/ui/'; return; }
  let msg = 'Sign in failed.';
  try { msg = (await res.json()).detail || msg; } catch (_) {}
  e.textContent = msg;
});
</script>
"""


@app.get("/login", include_in_schema=False)
def login_page(request: Request):
    if auth.request_authenticated(request.headers, request.cookies):
        return RedirectResponse(url="/ui/", status_code=303)
    return HTMLResponse(_LOGIN_PAGE)


@app.post("/login")
def login(req: dict, request: Request) -> JSONResponse:
    """Exchange the shared token for a session cookie.

    Rate-limited per client independently of the build limiter: a public URL
    invites brute force, and a shared token has no lockout of its own.
    """
    if not auth.enabled():
        return JSONResponse({"ok": True, "auth_required": False})
    allowed, retry_after = auth.login_limiter.check(_client_key(request))
    if not allowed:
        raise HTTPException(status_code=429,
                            detail=f"too many attempts — retry in {int(retry_after)}s",
                            headers={"Retry-After": str(int(retry_after))})
    if not auth.token_ok(_str_field(req, "token")):
        raise HTTPException(status_code=401, detail="invalid token")
    resp = JSONResponse({"ok": True})
    resp.set_cookie(
        config.SESSION_COOKIE, auth.session_value(),
        max_age=config.SESSION_MAX_AGE_S,
        httponly=True,          # unreadable from JS, so an XSS can't lift it
        samesite="lax",
        secure=request.url.scheme == "https",
    )
    return resp


@app.post("/logout")
def logout() -> JSONResponse:
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(config.SESSION_COOKIE)
    return resp


# ---------------------------------------------------------------------------
# Background jobs — /discover and /connect can run minutes (live web search
# across multiple hops), far past what a single HTTP request should block on.
# Each POST spawns a worker thread and returns a job_id immediately; the UI
# polls GET /jobs/{id} for a real percent-complete (from expand_graph's
# on_step hop/node counters) instead of guessing at an indeterminate spinner.
# In-memory only — fine for a single-process deployment; a job that outlives
# the process (crash/restart) is simply gone, same as any other in-flight work.
# ---------------------------------------------------------------------------
_JOBS: dict = {}
_JOBS_LOCK = threading.Lock()
_JOB_TTL_S = 3600  # stale finished jobs are swept opportunistically, not held forever


class JobCancelled(Exception):
    """Raised inside worker threads when a user cancels a background job."""


def _gc_jobs() -> None:
    cutoff = time.time() - _JOB_TTL_S
    stale = [jid for jid, j in _JOBS.items()
             if j["status"] not in {"running", "cancelling"} and j["updated_at"] < cutoff]
    for jid in stale:
        del _JOBS[jid]


def _new_job(kind: str = "job", ticket=None) -> str:
    job_id = uuid.uuid4().hex
    with _JOBS_LOCK:
        _gc_jobs()
        _JOBS[job_id] = {
            "kind": kind, "status": "queued", "pct": 0, "message": "queued…",
            "result": None, "error": None, "updated_at": time.time(),
            "cancel_requested": False, "_cancel_event": threading.Event(),
            # The build-queue ticket this job is waiting on, so /jobs/{id} can
            # report a real place in line instead of an opaque "queued…".
            "_ticket": ticket,
        }
    return job_id


def _update_job(job_id: str, **fields) -> None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return
        if job.get("status") in {"cancelling", "cancelled"} and "status" not in fields:
            return
        if job.get("_cancel_event") and job["_cancel_event"].is_set() and fields.get("status") in {"done", "error"}:
            fields = {"status": "cancelled", "pct": job.get("pct", 0),
                      "message": "cancelled", "error": None, "result": None}
        if "pct" in fields:
            pct = max(0, min(100, int(fields["pct"] or 0)))
            if fields.get("status") != "done":
                pct = max(int(job.get("pct", 0) or 0), pct)
            fields["pct"] = pct
        job.update(fields, updated_at=time.time())


def _public_job(job: dict) -> dict:
    """The client-visible shape: drop internals, add live queue position."""
    out = {k: v for k, v in job.items()
           if k != "updated_at" and not k.startswith("_")}
    ticket = job.get("_ticket")
    # Computed on read, not stored: position changes as other builds finish,
    # and a value written at enqueue time would be stale by the first poll.
    out["queue_position"] = BUILDS.position(ticket) if ticket is not None else 0
    return out


def _get_job(job_id: str) -> dict:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown or expired job")
        return _public_job(job)


def _job_cancel_event(job_id: str) -> threading.Event:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown or expired job")
        return job["_cancel_event"]


def _check_job_cancelled(job_id: str) -> None:
    if _job_cancel_event(job_id).is_set():
        raise JobCancelled()


def _await_build_slot(job_id: str, ticket, check_cancel) -> None:
    """Block until this job's ticket reaches the head of the build queue.

    Reports its place in line while it waits, so a queued user sees "waiting —
    2 build(s) ahead" rather than a progress bar stuck at 0 that is
    indistinguishable from a hang.
    """
    def _tick() -> None:
        # Cancel first: a user who gave up while queued should stop here, not
        # after being handed a slot. Raising leaves the ticket for the caller's
        # finally-block to release.
        check_cancel()
        pos = BUILDS.position(ticket)
        if pos > 0:
            # Message only, no status — _update_job's guard drops a
            # status-less update once a job is cancelling, so this can never
            # resurrect a job the user just cancelled.
            _update_job(job_id, message=f"waiting — {pos} build(s) ahead")

    _tick()
    BUILDS.acquire(ticket, check_cancel=_tick)
    _update_job(job_id, status="running", message="starting…")


def _start_build_job(request: Request, kind: str, worker, args: tuple) -> dict:
    """Admit a build, or refuse it. Every money-spending endpoint goes through here.

    Two gates, in order, both answered on the request thread so the client gets
    an immediate verdict rather than a job id that will never run:
      1. per-client rate limit  -> 429, this caller has had enough for now
      2. queue capacity         -> 429, the server is saturated for everyone
    """
    _enforce_rate_limit(request)
    try:
        ticket = BUILDS.reserve()
    except QueueFull:
        stats = BUILDS.stats()
        raise HTTPException(
            status_code=429,
            detail=(f"server busy — {stats['running']} build(s) running, "
                    f"{stats['queued']} queued. Try again in a minute."),
            headers={"Retry-After": "60"},
        )
    job_id = _new_job(kind, ticket=ticket)
    threading.Thread(target=worker, daemon=True,
                     args=(job_id, ticket) + args).start()
    return {"job_id": job_id}


def _hop_fraction(hop: int, done: int, total: int, max_depth: int) -> float:
    """0..1 progress within a single-sided expand_graph run."""
    within_hop = (done / total) if total else 0.0
    return min(1.0, (hop + within_hop) / max(max_depth, 1))


def _str_field(req: dict, key: str) -> str:
    """Extract+strip a string field from a raw JSON request body.

    Endpoints below take `req: dict` (not a Pydantic model) so a client can
    send any JSON shape; a non-string value (e.g. {"person_a": 123}) must
    raise a clean 400 here rather than blow up on .strip() with a raw 500.
    """
    val = req.get(key)
    if val is None:
        return ""
    if not isinstance(val, str):
        raise HTTPException(status_code=400, detail=f"'{key}' must be a string")
    return val.strip()


def _owner_id(x_graph_id: str = Header(default="default", alias="X-Graph-Id")) -> str:
    """The per-browser id the frontend mints into localStorage. Scopes both the
    operator's own profile (/owner) and Boards; the discovery graph itself is
    shared. Defined up here because endpoints in several sections below take it
    as a dependency, and Depends() resolves at decoration time."""
    return safe_graph_id(x_graph_id)


@app.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    return _get_job(job_id)


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown or expired job")
        if job["status"] in {"done", "error", "cancelled"}:
            return {k: v for k, v in job.items()
                    if k != "updated_at" and not k.startswith("_")}
        job["_cancel_event"].set()
        job.update(status="cancelling", cancel_requested=True,
                   message="cancelling…", updated_at=time.time())
    return _get_job(job_id)


# ONE shared global graph for the whole team: every run accumulates into it, and
# pathfinding runs over the union so a route can pass through people other runs
# discovered. get_db yields a session on the single default engine.
@app.get("/", include_in_schema=False)
def _root() -> RedirectResponse:
    return RedirectResponse(url="/ui/")


@app.get("/healthz", include_in_schema=False)
def healthz() -> dict:
    # cheap liveness probe for load balancers / Render health checks —
    # no dependency I/O at all, so it can't hang or flap.
    return {"status": "ok"}


def _extraction_status() -> dict:
    """Which extractor is actually running, and which Claude stages are live.

    `claude_available()` only checks that a client can be constructed — no
    network call — so this stays cheap. `spacy_available()` does load the NER
    model on first call; that's a one-time cost the first build pays anyway,
    and it's why /healthz above stays separate.
    """
    from .extraction import spacy_available
    from .extraction.claude_client import credential_state
    from .extraction.entity_filter import is_filtering_active
    from .extraction.relation_classifier import is_active as classifier_active

    # `credential_state()` is the honest answer for display; `claude_available()`
    # is the optimistic one the call sites use. They differ only in the
    # 'unverified' case (a profile-based credential nothing has exercised yet).
    state = credential_state()
    usable = claude_available()
    if config.CLAUDE_EXTRACT and usable:
        extractor = "claude"
    elif spacy_available():
        extractor = "spacy-ner"
    else:
        extractor = "heuristic"
    return {
        "extractor": extractor,
        # Reports configuration only — never the key itself.
        "claude": {
            "configured": state == "configured",
            "credentials": state,  # configured | unverified | unavailable
            # Two models, deliberately: the batched stages run on a cheap one,
            # page-level extraction on the strong one. Reporting only the
            # latter would misdescribe what is actually running on most builds.
            "batch_model": config.CLAUDE_BATCH_MODEL,
            "extraction_model": config.CLAUDE_EXTRACT_MODEL,
            "extraction": bool(config.CLAUDE_EXTRACT and usable),
            "entity_filter": is_filtering_active(),
            "relation_classifier": classifier_active(),
        },
    }


@app.get("/health")
def health() -> dict:
    out = {"status": "ok"}
    out.update(_extraction_status())
    return out


@app.get("/status")
def status() -> dict:
    """Live service status for the UI. Search degrades only when BOTH paid
    providers are out: Serper (primary) -> Brave (backup) -> DuckDuckGo (free)."""
    from .providers.brave import brave_status
    from .providers.serper import serper_status
    serper, brave = serper_status(), brave_status()
    using = "serper" if serper["ok"] else ("brave" if brave["ok"] else "duckduckgo")
    out = {
        "serper": serper,
        "brave": brave,
        "search": {
            "ok": serper["ok"] or brave["ok"],
            "degraded": not (serper["ok"] or brave["ok"]),
            "using": using,
        },
        # So the UI can warn on an unprotected deployment, and show how busy
        # the server is before a user starts a build that would only queue.
        "auth": auth.status(),
        "builds": BUILDS.stats(),
    }
    out.update(_extraction_status())
    return out


def _run_target_search_job(job_id: str, ticket, target_name: str, max_depth: int) -> None:
    def check_cancel() -> None:
        _check_job_cancelled(job_id)

    def on_step(evt: dict) -> None:
        check_cancel()
        frac = _hop_fraction(evt["hop"], evt.get("done", 0), evt.get("total", 1), max_depth)
        _update_job(job_id, pct=int(min(97, frac * 100)),
                    message=f"hop {evt['hop']+1}/{max_depth} · "
                            f"{evt.get('done', 0)}/{evt.get('total', 1)} nodes")

    db = None
    try:
        db = SessionLocal()
        _await_build_slot(job_id, ticket, check_cancel)
        check_cancel()
        # ADDITIVE: accumulate the searched person into the shared global map
        # (no reset), then return only that person's neighborhood.
        stats = expand_graph(db, target_name, max_depth, on_step=on_step,
                             cancel_checker=check_cancel)
        check_cancel()
        nodes, edges = serialize_neighborhood(db, target_name, max_depth)
        result = GraphResponse(graph_id="global", nodes=nodes, edges=edges,
                               stats=GraphStats(**stats)).model_dump(by_alias=True)
        _update_job(job_id, status="done", pct=100, message="done", result=result)
    except JobCancelled:
        _update_job(job_id, status="cancelled", message="cancelled",
                    error=None, result=None)
    except Exception as exc:
        _update_job(job_id, status="error", error=str(exc))
    finally:
        BUILDS.release(ticket)
        if db is not None:
            db.close()


@app.post("/targets/search")
def targets_search(req: TargetSearchRequest, request: Request) -> dict:
    """Build a target's neighborhood into the shared graph, as a background job.

    BREAKING (was synchronous): this used to run the whole build inside the
    request and return a GraphResponse. A deep build takes minutes, which is
    past every managed host's proxy idle timeout — the response was frequently
    never delivered no matter how well the build went. It now behaves like
    /discover and /connect: returns {"job_id"} immediately, and the same
    GraphResponse body arrives as the job's `result`.
    """
    return _start_build_job(request, "targets_search", _run_target_search_job,
                            (req.target_name, req.max_depth))


@app.get("/graph", response_model=GraphResponse)
def get_graph(db: Session = Depends(get_db)) -> GraphResponse:
    # Derive people/org/edge counts from the lists we're building anyway
    # instead of re-querying each table -- only sources_fetched needs its
    # own count, since a fetched source with no resulting edge isn't
    # otherwise represented in `edges`.
    nodes = serialize_nodes(db)
    edges = serialize_edges(db)
    stats = GraphStats(
        people_found=sum(1 for n in nodes if n.kind == "person"),
        organizations_found=sum(1 for n in nodes if n.kind == "organization"),
        edges_found=len(edges),
        sources_fetched=db.query(Source).count(),
    )
    return GraphResponse(
        graph_id="global",
        nodes=nodes,
        edges=edges,
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


def _run_connect_job(job_id: str, ticket, a: str, b: str, depth: int,
                     context_a: str, context_b: str) -> None:
    from .graph.connect import connect_people
    state = {"a": {"hop": 0, "done": 0, "total": 1},
             "b": {"hop": 0, "done": 0, "total": 1}}
    state_lock = threading.Lock()

    def check_cancel() -> None:
        _check_job_cancelled(job_id)

    def on_step(evt: dict) -> None:
        check_cancel()
        side = evt.get("side", "a")
        with state_lock:
            s = state.setdefault(side, {"hop": 0, "done": 0, "total": 1})
            s["hop"] = evt["hop"]
            s["total"] = max(evt.get("total", 1), 1)
            s["done"] = evt.get("done", 0)
            frac = sum(_hop_fraction(v["hop"], v["done"], v["total"], depth)
                       for v in state.values()) / len(state)
            message = (f"[A] hop {state['a']['hop']+1}/{depth} · "
                       f"{state['a']['done']}/{state['a']['total']} nodes · "
                       f"[B] hop {state['b']['hop']+1}/{depth} · "
                       f"{state['b']['done']}/{state['b']['total']} nodes")
        _update_job(job_id, pct=int(min(97, frac * 100)),
                    message=message)

    db = None
    try:
        db = SessionLocal()
        _await_build_slot(job_id, ticket, check_cancel)
        check_cancel()
        result = connect_people(db, a, b, depth, context_a=context_a,
                                context_b=context_b, on_step=on_step,
                                cancel_checker=check_cancel)
        check_cancel()
        result["graph_id"] = "global"
        _update_job(job_id, status="done", pct=100, message="done", result=result)
    except JobCancelled:
        _update_job(job_id, status="cancelled", message="cancelled",
                    error=None, result=None)
    except Exception as exc:
        _update_job(job_id, status="error", error=str(exc))
    finally:
        # Outermost, not around the build alone: cancelling while QUEUED raises
        # inside _await_build_slot, before any inner block would run, and the
        # ticket would sit in the queue forever holding a slot nobody uses.
        # release() handles both states and is safe to call twice.
        BUILDS.release(ticket)
        if db is not None:
            db.close()


@app.post("/connect")
def connect(req: dict, request: Request) -> dict:
    """Kick off a path search between two people (builds both graphs, meets in
    the middle) as a background job; poll GET /jobs/{job_id} for progress and
    the eventual result. Body: {"person_a": "...", "person_b": "...", "depth": 2}"""
    a = _str_field(req, "person_a")
    b = _str_field(req, "person_b")
    try:
        depth = max(1, min(int(req.get("depth", 2)), 3))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="depth must be an integer 1-3")
    if not a or not b:
        raise HTTPException(status_code=400, detail="person_a and person_b required")
    return _start_build_job(
        request, "connect", _run_connect_job,
        (a, b, depth, _str_field(req, "context_a"), _str_field(req, "context_b")))


def _run_discover_job(job_id: str, ticket, name: str, depth: int) -> None:
    from .graph.connect import discover_person

    def check_cancel() -> None:
        _check_job_cancelled(job_id)

    def on_step(evt: dict) -> None:
        check_cancel()
        frac = _hop_fraction(evt["hop"], evt.get("done", 0), evt.get("total", 1), depth)
        _update_job(job_id, pct=int(min(97, frac * 100)),
                    message=f"hop {evt['hop']+1}/{depth} · "
                            f"{evt.get('done', 0)}/{evt.get('total', 1)} nodes")

    db = None
    try:
        db = SessionLocal()
        _await_build_slot(job_id, ticket, check_cancel)
        check_cancel()
        expand_graph(db, name, depth, on_step=on_step,
                     cancel_checker=check_cancel)
        check_cancel()
        result = discover_person(db, name, depth)
        check_cancel()
        result["graph_id"] = "global"
        _update_job(job_id, status="done", pct=100, message="done", result=result)
    except JobCancelled:
        _update_job(job_id, status="cancelled", message="cancelled",
                    error=None, result=None)
    except Exception as exc:
        _update_job(job_id, status="error", error=str(exc))
    finally:
        BUILDS.release(ticket)  # see _run_connect_job: covers the queued case too
        if db is not None:
            db.close()


@app.post("/discover")
def discover(req: dict, request: Request) -> dict:
    """Kick off expansion of one person's public network (ranked, within
    `depth` hops) as a background job; poll GET /jobs/{job_id} for progress
    and the eventual result. Body: {"person_name": "...", "depth": 2}"""
    name = _str_field(req, "person_name")
    try:
        depth = max(1, min(int(req.get("depth", 2)), 3))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="depth must be an integer 1-3")
    if not name:
        raise HTTPException(status_code=400, detail="person_name required")
    return _start_build_job(request, "discover", _run_discover_job, (name, depth))


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
        "notes": p.notes, "connected_on": p.connected_on, "created_at": p.created_at,
    }


def _match_dict(m: GraphMatch) -> dict:
    return {
        "id": m.id, "local_profile_id": m.local_profile_id,
        "public_person_id": m.public_person_id, "public_org_id": m.public_org_id,
        "match_type": m.match_type, "confidence": m.confidence,
        "explanation": m.explanation,
    }


@app.post("/network/upload")
async def network_upload(file: UploadFile = File(...), owner_name: str = Form(""),
                         db: Session = Depends(get_db)) -> dict:
    """`owner_name` (optional): whose contacts these are. When given, each
    imported contact also becomes a real linkedin_1st edge in the shared
    public graph — anchored to a Person node for `owner_name` — so /connect
    and /discover can route through it immediately. Without it, ingestion
    stays scoped to the private LocalProfile/LocalEdge tables, same as before."""
    chunks = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > config.MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="file too large")
        chunks.append(chunk)
    content = b"".join(chunks).decode("utf-8", errors="replace")
    # ingest_csv is synchronous, DB-bound work -- run it off the event loop so
    # a large CSV import doesn't stall every other concurrent request (incl. /health).
    stats = await run_in_threadpool(ingest_csv, db, content, owner_name=owner_name)
    return {"ingested": stats, "profiles_total": db.query(LocalProfile).count()}


@app.post("/network/profiles/backfill-graph-edges")
async def network_profiles_backfill(req: dict, db: Session = Depends(get_db)) -> dict:
    """Retroactively create linkedin_1st edges for every already-imported
    LocalProfile, for accounts that uploaded before `owner_name` was wired
    up on the frontend (or before this bridge existed at all) -- otherwise
    those contacts sit in LocalProfile forever, invisible to /connect and
    /discover. Idempotent; safe to call repeatedly (e.g. once per session)."""
    owner_name = (req.get("owner_name") or "").strip()
    if not owner_name:
        raise HTTPException(status_code=400, detail="owner_name required")
    count = await run_in_threadpool(backfill_graph_edges, db, owner_name)
    return {"graph_edges": count}


@app.post("/network/cliques")
async def network_cliques(owner_id: str = Depends(_owner_id),
                          db: Session = Depends(get_db)) -> dict:
    """Wave 0 of initial enrichment: derive org membership and small-employer
    coworker cliques from the already-imported contacts.

    Costs nothing — no searches, no page fetches, no Claude — so it is NOT a
    build and deliberately skips the BuildQueue admission path that /connect
    and /discover go through. Idempotent; the frontend calls it after an
    import, and calling it again just converges.

    A saved owner profile puts the operator into their own employer's and
    school's clusters too; without one they stay outside every org cluster."""
    counts = await run_in_threadpool(
        materialize_contact_cliques, db, None, get_owner(db, owner_id))
    return {"wave0": counts}


# ===========================================================================
# The operator's own identity, scoped by X-Graph-Id (see _owner_id).
# ===========================================================================
@app.get("/owner")
def read_owner(owner_id: str = Depends(_owner_id),
               db: Session = Depends(get_db)) -> dict:
    """The stored profile, or an unconfigured placeholder. Never 404s: "no
    profile yet" is a normal first-boot state, not a client error."""
    return owner_dict(get_owner(db, owner_id))


@app.put("/owner")
def write_owner(req: dict, owner_id: str = Depends(_owner_id),
                db: Session = Depends(get_db)) -> dict:
    """Save who the operator is. Body may carry any of name, company, title,
    school, linkedin_url, email; omitted fields keep their stored value.

    `company` and `school` are what let ranking's shared-affiliation boost
    actually fire — nothing was sending them before this existed.
    """
    fields = {k: _str_field(req, k) for k in
              ("name", "company", "title", "school", "linkedin_url", "email")
              if k in req}
    if not fields:
        raise HTTPException(status_code=400, detail="no recognized fields")
    existing = get_owner(db, owner_id)
    if "name" in fields and not fields["name"] and existing is None:
        raise HTTPException(status_code=400, detail="name required")
    return owner_dict(upsert_owner(db, owner_id, **fields))


# ===========================================================================
# Initial enrichment — the operator's own 2-layer network.
#
# Planning is free and separate from execution on purpose: POST /enrich/runs
# ranks the contacts and persists the plan without issuing a single search, so
# the operator can see who would be enriched, in what order, and who is
# excluded and why, before committing to hours of paid crawling.
# ===========================================================================
@app.post("/enrich/runs")
async def create_enrichment_run(req: dict, owner_id: str = Depends(_owner_id),
                                db: Session = Depends(get_db)) -> dict:
    """Plan a run over the imported contacts. Costs nothing; runs nothing.

    Body: {"owner_name": "...", "owner_company": "...", "owner_school": "...",
    "depth": 1, "budget_s": 0}
    """
    # The stored profile supplies whatever the request leaves out — that is the
    # point of having one. owner_company/owner_school in particular were never
    # sent by any caller, so before this the shared-affiliation boost in
    # ranking.score_contacts was dead code in practice.
    profile = get_owner(db, owner_id)
    owner_name = _str_field(req, "owner_name") or (profile.name if profile else "")
    if not owner_name:
        raise HTTPException(
            status_code=400,
            detail="owner_name required (or save a profile via PUT /owner)")
    company = _str_field(req, "owner_company") or (profile.company if profile else "") or ""
    school = _str_field(req, "owner_school") or (profile.school if profile else "") or ""
    try:
        depth = max(1, min(int(req.get("depth", 1)), 3))
        budget_s = max(0.0, float(req.get("budget_s", 0) or 0))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400,
                            detail="depth must be an integer 1-3 and budget_s a number")
    try:
        run = await run_in_threadpool(
            plan_run, db, owner_name, company, school, depth, budget_s)
    except RunConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return run_dict(db, run)


@app.get("/enrich/runs")
def list_enrichment_runs(db: Session = Depends(get_db)) -> list:
    runs = db.execute(
        select(EnrichmentRun).order_by(EnrichmentRun.created_at.desc()).limit(20)
    ).scalars()
    # The cached counters on the row are good enough for a list view — see
    # enrichment.tally on why they are never trusted for a single run's detail.
    return [{"id": r.id, "owner_name": r.owner_name, "state": r.state,
             "depth": r.depth, "counters": r.counters or {},
             "created_at": r.created_at, "finished_at": r.finished_at}
            for r in runs]


@app.get("/enrich/runs/{run_id}")
def get_enrichment_run(run_id: str, db: Session = Depends(get_db)) -> dict:
    run = db.get(EnrichmentRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="unknown enrichment run")
    return run_dict(db, run)


@app.post("/enrich/runs/{run_id}/start")
def start_enrichment_run(run_id: str, req: dict = None,
                         db: Session = Depends(get_db)) -> dict:
    """Begin (or resume) execution on a background thread; returns immediately.

    Body: {"limit": 30} — how many contacts THIS invocation covers. Defaults to
    config.ENRICH_WAVE1_SIZE (wave 1); pass 0 to run the plan to exhaustion.

    Deliberately NOT routed through _start_build_job: that admits a single
    build, whereas a run admits one background build PER CONTACT so interactive
    callers can interleave (see network/executor.py).
    """
    req = req or {}
    run = db.get(EnrichmentRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="unknown enrichment run")
    try:
        limit = int(req.get("limit", config.ENRICH_WAVE1_SIZE))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="limit must be an integer")

    # Claim BEFORE spawning: a conditional UPDATE, so a double-clicked start
    # button gets one worker and one 409 rather than two threads racing over
    # the same plan and paying twice per contact.
    if not claim_run(db, run_id):
        raise HTTPException(
            status_code=409,
            detail=f"run is {run.state}; only {' or '.join(STARTABLE)} can start")

    def _worker() -> None:
        # Its own Session: this outlives the request, and the request-scoped
        # one from Depends(get_db) is closed the moment we return.
        worker_db = SessionLocal()
        try:
            execute_run(worker_db, run_id, limit=max(0, limit), claimed=True)
        except Exception:
            pass  # executor already recorded the failure on the run row
        finally:
            worker_db.close()

    threading.Thread(target=_worker, daemon=True,
                     name=f"enrich-{run_id[:8]}").start()
    db.expire(run)
    return run_dict(db, run)


@app.post("/enrich/runs/{run_id}/pause")
def pause_enrichment_run(run_id: str, db: Session = Depends(get_db)) -> dict:
    """Stop after the contact currently in flight. Progress is kept, and
    POST .../start resumes from the next pending task."""
    run = db.get(EnrichmentRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="unknown enrichment run")
    if run.state == "running":
        run.state = "paused"
        db.commit()
    return run_dict(db, run)


@app.post("/enrich/runs/{run_id}/cancel")
def cancel_enrichment_run(run_id: str, db: Session = Depends(get_db)) -> dict:
    run = db.get(EnrichmentRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="unknown enrichment run")
    return run_dict(db, cancel_run(db, run))


@app.post("/network/contacts/import")
async def import_contacts(req: dict, db: Session = Depends(get_db)) -> dict:
    """Bulk-add already-parsed contacts (the phone/vCard import in the UI).

    The .vcf is parsed in the browser so the user can pick who to bring over —
    only the chosen cards arrive here, as rows shaped like a connections
    export, and go through the same ingestion as an uploaded CSV (de-dupe,
    "You" edge, optional public-graph edges when `owner_name` is given)."""
    contacts = req.get("contacts")
    if not isinstance(contacts, list) or not contacts:
        raise HTTPException(status_code=400, detail="contacts must be a non-empty list")
    if len(contacts) > config.MAX_IMPORT_CONTACTS:
        raise HTTPException(
            status_code=413,
            detail=f"too many contacts (max {config.MAX_IMPORT_CONTACTS} per import)")

    rows = []
    for c in contacts:
        if not isinstance(c, dict):
            raise HTTPException(status_code=400, detail="each contact must be an object")
        rows.append({
            "Name":          _str_field(c, "name"),
            "Company":       _str_field(c, "company"),
            "Position":      _str_field(c, "title"),
            "Email Address": _str_field(c, "email"),
            "Notes":         _str_field(c, "notes"),
        })

    stats = await run_in_threadpool(ingest_rows, db, rows,
                                    owner_name=_str_field(req, "owner_name"))
    return {"ingested": stats, "profiles_total": db.query(LocalProfile).count()}


@app.get("/network/profiles")
def network_profiles(db: Session = Depends(get_db)) -> list:
    return [_profile_dict(p) for p in db.execute(select(LocalProfile)).scalars()]


@app.post("/network/profiles")
def add_profile(req: dict, db: Session = Depends(get_db)) -> dict:
    """Manually add a single contact (the "+ Add Contact" UI action)."""
    from .utils.names import name_variants, person_norm_key
    name = _str_field(req, "name")
    if not name:
        raise HTTPException(status_code=400, detail="name required")
    company = _str_field(req, "company")
    title = _str_field(req, "title")
    school = _str_field(req, "school")
    profile = LocalProfile(
        canonical_name=name,
        norm_name=person_norm_key(name),
        aliases=sorted(v for v in name_variants(name) if v != name),
        email=_str_field(req, "email") or None,
        linkedin_url=_str_field(req, "linkedin_url") or None,
        companies=[company] if company else [],
        titles=[title] if title else [],
        schools=[school] if school else [],
        locations=[],
        notes=_str_field(req, "notes") or None,
        raw_row={},
    )
    db.add(profile)
    db.flush()
    db.add(LocalEdge(from_profile_id=None, to_profile_id=profile.id))
    db.commit()
    return _profile_dict(profile)


@app.delete("/network/profiles")
def clear_profiles(db: Session = Depends(get_db)) -> dict:
    """Wipe all uploaded/added contacts and anything derived from them
    (matches, candidate paths). Never touches the public discovery graph."""
    n = db.query(LocalProfile).count()
    db.query(CandidatePath).delete()
    db.query(GraphMatch).delete()
    db.query(LocalEdge).delete()
    db.query(LocalProfile).delete()
    db.commit()
    return {"cleared": n}


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


# ===========================================================================
# Boards — a user's manually-built canvas workspace (UI-only; never mutates
# the canonical discovery data above). Owner-scoped by X-Graph-Id (see
# _owner_id above). Each board holds one or more Pages, each an independent
# node/edge canvas.
# ===========================================================================
def _get_owned_board(db: Session, board_id: str, owner_id: str) -> Board:
    b = db.get(Board, board_id)
    if b is None or b.owner_id != owner_id:
        raise HTTPException(status_code=404, detail="board not found")
    return b


def _page_dict(p: BoardPage) -> dict:
    return {"id": p.id, "name": p.name, "position": p.position, "elements": p.elements or {}}


def _board_pages(db: Session, board_id: str) -> list[BoardPage]:
    return list(db.execute(
        select(BoardPage).where(BoardPage.board_id == board_id).order_by(BoardPage.position.asc())
    ).scalars())


def _board_summary(b: Board, seq: int, pages: list[BoardPage]) -> dict:
    nodes = sum(len((p.elements or {}).get("nodes") or []) for p in pages)
    edges = sum(len((p.elements or {}).get("edges") or []) for p in pages)
    return {
        "id": b.id, "seq": seq, "name": b.name, "status": b.status or "active",
        "created_at": b.created_at, "target_name": b.target_name, "target_org": b.target_org,
        "pages": len(pages), "nodes": nodes, "edges": edges,
        # first page's elements power the boards-list minimap preview
        "preview_elements": (pages[0].elements or {}) if pages else {},
    }


@app.post("/boards")
def create_board(req: dict, owner_id: str = Depends(_owner_id),
                  db: Session = Depends(get_boards_db)) -> dict:
    name = _str_field(req, "name")
    if not name:
        raise HTTPException(status_code=400, detail="name required")
    board = Board(
        owner_id=owner_id, name=name,
        target_name=_str_field(req, "target_name") or None,
        target_org=_str_field(req, "target_org") or None,
    )
    db.add(board)
    db.flush()
    page = BoardPage(board_id=board.id, name="Page 1", position=0, elements={})
    db.add(page)
    db.commit()
    # This board was just created, so it's necessarily the newest for this
    # owner -- a plain (indexed) owner_id count already equals its seq,
    # without the unindexed `created_at <=` string range-scan.
    seq = db.query(Board).filter(Board.owner_id == owner_id).count()
    return _board_summary(board, seq, [page])


@app.get("/boards")
def list_boards(owner_id: str = Depends(_owner_id), db: Session = Depends(get_boards_db)) -> list:
    rows = list(db.execute(
        select(Board).where(Board.owner_id == owner_id).order_by(Board.created_at.asc())
    ).scalars())
    # One query for all boards' pages instead of one per board (N+1).
    pages_by_board: dict[str, list] = defaultdict(list)
    if rows:
        board_ids = [b.id for b in rows]
        for p in db.execute(
            select(BoardPage).where(BoardPage.board_id.in_(board_ids))
            .order_by(BoardPage.board_id, BoardPage.position.asc())
        ).scalars():
            pages_by_board[p.board_id].append(p)
    summaries = [_board_summary(b, i + 1, pages_by_board.get(b.id, []))
                 for i, b in enumerate(rows)]
    summaries.reverse()  # newest first
    return summaries


@app.get("/boards/{board_id}")
def get_board(board_id: str, owner_id: str = Depends(_owner_id),
              db: Session = Depends(get_boards_db)) -> dict:
    b = _get_owned_board(db, board_id, owner_id)
    pages = _board_pages(db, board_id)
    return {"id": b.id, "name": b.name, "status": b.status or "active",
            "target_name": b.target_name, "target_org": b.target_org,
            "created_at": b.created_at, "pages": [_page_dict(p) for p in pages]}


@app.patch("/boards/{board_id}")
def update_board(board_id: str, req: dict, owner_id: str = Depends(_owner_id),
                  db: Session = Depends(get_boards_db)) -> dict:
    b = _get_owned_board(db, board_id, owner_id)
    if "status" in req:
        if req["status"] not in ("active", "archived"):
            raise HTTPException(status_code=400, detail="status must be 'active' or 'archived'")
        b.status = req["status"]
    if "name" in req:
        new_name = _str_field(req, "name")
        if new_name:
            b.name = new_name
    if "target_name" in req:
        b.target_name = _str_field(req, "target_name") or None
    if "target_org" in req:
        b.target_org = _str_field(req, "target_org") or None
    db.commit()
    seq = db.query(Board).filter(
        Board.owner_id == owner_id, Board.created_at <= b.created_at
    ).count()
    return _board_summary(b, seq, _board_pages(db, board_id))


@app.delete("/boards/{board_id}")
def delete_board(board_id: str, owner_id: str = Depends(_owner_id),
                  db: Session = Depends(get_boards_db)) -> dict:
    b = _get_owned_board(db, board_id, owner_id)
    db.query(BoardPage).filter(BoardPage.board_id == b.id).delete()
    db.delete(b)
    db.commit()
    return {"deleted": board_id}


# ---- Pages ------------------------------------------------------------------
@app.post("/boards/{board_id}/pages")
def create_page(board_id: str, req: dict, owner_id: str = Depends(_owner_id),
                 db: Session = Depends(get_boards_db)) -> dict:
    _get_owned_board(db, board_id, owner_id)
    existing = _board_pages(db, board_id)
    name = _str_field(req, "name") or f"Page {len(existing) + 1}"
    page = BoardPage(board_id=board_id, name=name, position=len(existing), elements={})
    db.add(page)
    db.commit()
    return _page_dict(page)


@app.patch("/boards/{board_id}/pages/{page_id}")
def update_page(board_id: str, page_id: str, req: dict, owner_id: str = Depends(_owner_id),
                 db: Session = Depends(get_boards_db)) -> dict:
    _get_owned_board(db, board_id, owner_id)
    page = db.get(BoardPage, page_id)
    if page is None or page.board_id != board_id:
        raise HTTPException(status_code=404, detail="page not found")
    if "name" in req:
        new_name = _str_field(req, "name")
        if new_name:
            page.name = new_name
    if "elements" in req:
        page.elements = req.get("elements") or {}
    db.commit()
    return _page_dict(page)


@app.delete("/boards/{board_id}/pages/{page_id}")
def delete_page(board_id: str, page_id: str, owner_id: str = Depends(_owner_id),
                 db: Session = Depends(get_boards_db)) -> dict:
    _get_owned_board(db, board_id, owner_id)
    pages = _board_pages(db, board_id)
    if len(pages) <= 1:
        raise HTTPException(status_code=400, detail="a board must keep at least one page")
    page = db.get(BoardPage, page_id)
    if page is None or page.board_id != board_id:
        raise HTTPException(status_code=404, detail="page not found")
    db.delete(page)
    db.commit()
    return {"deleted": page_id}


# --- static frontend (mounted last so it never shadows the API routes) ------
_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/ui", StaticFiles(directory=_STATIC_DIR, html=True), name="ui")
