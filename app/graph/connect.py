"""Path-finding between two people (bidirectional, meet-in-the-middle).

Expand BOTH people's graphs depth-wise into one combined graph, then find the
best path connecting them over public person-person edges. Where their
neighborhoods overlap, a bridge node appears and a path exists.

Candidate routes are Claude-verified hop by hop before being returned (see
hop_verify.verify) -- only the hops in a route actually found, not every edge
considered during search.
"""
from __future__ import annotations

import heapq
import math
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import unquote

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from .. import config
from ..extraction import bridge_strategy, extract, relation_classifier, spacy_extractor
from ..extraction.schemas import EdgeSignals, ExtractedEdge
from ..models import Organization, Person, RelationshipEdge, Source
from ..network.cliques import materialize_contact_cliques
from ..network.ingest import backfill_graph_edges
from ..network.owner import get_owner_by_name
from ..network.ranking import BridgeTarget, ScoredContact, score_contacts
from ..network.silo_weights import initial_weights
from ..silos import COLLEAGUE_SILO
from ..utils.htmltext import html_to_text
from ..utils.names import mention_patterns, person_norm_key
from . import builder, hop_verify
from .expansion import ORCH, expand_graph

# relationship strength multiplier (shared with candidate-path scoring)
REL_STRENGTH = {
    "linkedin_1st": 1.0, "podcast_guest": 1.0,
    "cofounder": 1.0, "board_member": 0.95, "advisor": 0.9, "investor": 0.85,
    "employee": 0.8, "coworker": 0.8, "coauthor": 0.8, "appointee": 0.75,
    "faculty": 0.7, "student": 0.7, "author": 0.6, "speaker": 0.5,
    "interview": 0.5, "family_social": 0.45, "unknown": 0.4,
}
_STATUS_PENALTY = {"strong": 0.0, "candidate": 0.3, "raw": 1.0,
                   "weak": 2.0, "rejected": 12.0}

# How many of the target's own employers to aim contact selection at. Past the
# best two or three the affiliations are stale or weakly evidenced, and each
# one widens the set of contacts that count as "shares an employer".
_TARGET_ORG_LIMIT = 3

# 'rejected' means an edge was reviewed (by a human or the LLM classifier) and
# marked false — that is the only status that means "not a real connection"
# rather than merely "weakly evidenced", so it's the only one excluded
# outright. Everything else (weak status, an untyped 'unknown' relationship)
# is priced instead via _STATUS_PENALTY / UNKNOWN_TYPE_SURCHARGE below.
#
# This used to be a hard filter requiring known-type + candidate-tier-or-above,
# meant to stop weak co-occurrence noise (e.g. an 'unknown 0.35' bridge through
# a boilerplate/homonym node) from forming a path — the Fred→Cook run's junk
# routes were exactly such edges. But on a real, sparsely-evidenced graph most
# edges land at 'weak' status or 'unknown' type (the Claude classifier that
# retypes 'unknown' edges only reaches candidate-tier+ edges — see
# _retype_unknown_edges), so the filter excluded ~80% of the graph and two
# people who WERE linked in the data routinely came back "not connected" with
# no path at all to show. A cost, unlike an exclusion, still prefers every
# better-evidenced route there is and only falls back to a noisy bridge when
# it's genuinely the only way across.
_UNTRAVERSABLE_STATUS = {"rejected"}
UNKNOWN_TYPE_SURCHARGE = 1.2  # extra cost for an untyped ('unknown') edge


def _untraversable(status: str, relationship_type: str, signals: Optional[dict]) -> bool:
    """Shared traversability rule for _path_worthy (full ORM row) and
    _route_exists (lean column-only SELECT, for the same reason no edge
    cost/fame penalty applies here either -- see both callers' docstrings).

    'rejected' status means an edge was reviewed and marked false -- the
    only status that means "not a real connection" outright (see the long
    comment on _UNTRAVERSABLE_STATUS for why weak/unknown edges otherwise
    stay traversable -- that fix, and the reasoning behind it, stays intact
    here unchanged).

    The second exclusion is narrower and different in KIND, not degree: an
    edge with NO real evidence the two names ever appeared together
    (signals.sentence_cooccurrence False) AND no explicit relationship
    keyword AND no assigned relationship type at all is not weak evidence of
    a real connection -- it's two coincidental mentions on the same fetched
    page glued into an edge by bare co-presence. Confirmed live: "Dream
    Sports Chief Technology Officer Amit Sharma..." and "...Mark Zuckerberg
    seeks to..." are two unrelated sentences from one page, persisted as a
    'directly connected' result at confidence 0.10. This does NOT touch the
    broader weak/unknown-type population (an edge with either cooccurrence
    OR an explicit keyword still passes) -- excluding by status/type alone
    was already tried and reverted for breaking real sparse-graph
    connectivity (see the block comment above); this is a much narrower cut.
    """
    if status in _UNTRAVERSABLE_STATUS:
        return True
    signals = signals or {}
    if (relationship_type == "unknown"
            and not signals.get("sentence_cooccurrence")
            and not signals.get("explicit_keyword_match")):
        return True
    return False


def _path_worthy(e: RelationshipEdge) -> bool:
    return not _untraversable(e.status, e.relationship_type, e.signals)


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
        if a not in person_by_id or b not in person_by_id:
            continue  # dangling edge — its endpoint was pruned after this edge was written
        if not _path_worthy(e):
            continue  # 'rejected' — a reviewed, confirmed-false edge
        key = tuple(sorted((a, b)))
        cur = best.get(key)
        if cur is None or (e.confidence_raw or 0) > (cur.confidence_raw or 0):
            best[key] = e
    adj: Dict[str, List[Tuple[str, RelationshipEdge]]] = defaultdict(list)
    for (a, b), e in best.items():
        adj[a].append((b, e))
        adj[b].append((a, e))
    degree = {pid: len(v) for pid, v in adj.items()}
    return adj, person_by_id, src_by_id, degree


def _org_affiliations(db: Session) -> Dict[str, Dict[str, Tuple[str, float]]]:
    """person_id -> {org_id: (org name, confidence)} over person->org edges.

    A person-person edge records WHAT the tie is ("coworker") but never WHERE
    it happened: organization_id is set only on person->org rows, never
    alongside person_b_id (see builder.add_edge_from_extraction). So the place
    two people share is recovered here from each side's OWN org edges and
    intersected, rather than read off the edge between them.
    """
    names = {o.id: o.name for o in db.execute(select(Organization)).scalars()}
    aff: Dict[str, Dict[str, Tuple[str, float]]] = defaultdict(dict)
    for e in db.execute(
        select(RelationshipEdge).where(RelationshipEdge.organization_id.isnot(None))
    ).scalars():
        name = names.get(e.organization_id)
        if not name or not e.person_a_id or not _path_worthy(e):
            continue
        conf = e.confidence_raw or 0.0
        cur = aff[e.person_a_id].get(e.organization_id)
        if cur is None or conf > cur[1]:
            aff[e.person_a_id][e.organization_id] = (name, conf)
    return aff


def _shared_orgs(aff, a_id: str, b_id: str, limit: int = 2) -> List[str]:
    """Orgs BOTH endpoints are affiliated with, best-evidenced first.

    Ranked by the weaker of the two sides' confidences: a place is only as
    good an answer to "where?" as the shakier of the two affiliations behind
    it.
    """
    a = aff.get(a_id) or {}
    b = aff.get(b_id) or {}
    shared = sorted(set(a) & set(b),
                    key=lambda oid: min(a[oid][1], b[oid][1]), reverse=True)
    return [a[oid][0] for oid in shared[:limit]]


def _edge_cost(e: RelationshipEdge) -> float:
    conf = max(e.confidence_raw or 0.01, 0.01)
    cost = -math.log(conf) + _STATUS_PENALTY.get(e.status, 1.0)
    if e.relationship_type == "unknown":
        cost += UNKNOWN_TYPE_SURCHARGE
    return cost


def _node_penalty(person_by_id, degree, person_id: str) -> float:
    """Cost added for routing THROUGH person_id (never applied to the final
    target — see _best_path). Fame: a real edge to a Wikidata-notable person is
    a poor bridge, they're unlikely to relay a stranger's intro. Mega-hub: a
    node with far more edges than typical shouldn't absorb every route."""
    p = person_by_id.get(person_id)
    fame = config.FAME_PENALTY if (p and p.wikidata_qid) else 0.0
    deg = degree.get(person_id, 0)
    hub = config.DEGREE_PENALTY_COEF * math.log(deg) if deg > config.MEGA_HUB_DEGREE else 0.0
    return fame + hub


def _best_path(adj, start: str, target: str, max_hops: int, excluded=None,
               person_by_id=None, degree=None):
    """Best (max-confidence) path, optionally skipping `excluded` intermediate
    nodes so callers can find genuinely different routes."""
    excluded = excluded or set()
    person_by_id = person_by_id or {}
    degree = degree or {}
    if start == target:
        return [(start, None)]
    counter_seed = 0
    best_cost = {start: 0.0}
    heap = [(0.0, 0, counter_seed, start, [(start, None)])]
    while heap:
        cost, hops, _t, node, path = heapq.heappop(heap)
        if node == target:
            return path
        if hops >= max_hops:
            continue
        for nbr, edge in adj.get(node, []):
            if nbr in excluded and nbr != target:
                continue
            penalty = 0.0 if nbr == target else (
                config.HOP_SURCHARGE + _node_penalty(person_by_id, degree, nbr))
            nc = cost + _edge_cost(edge) + penalty
            if nbr not in best_cost or nc < best_cost[nbr]:
                best_cost[nbr] = nc
                counter_seed += 1
                heapq.heappush(heap, (nc, hops + 1, counter_seed, nbr, path + [(nbr, edge)]))
    return None


def _diverse_paths(adj, start: str, target: str, max_hops: int, k: int,
                   person_by_id=None, degree=None):
    """Up to k routes; each avoids all bridge (intermediate) nodes used by the
    earlier ones, so they're genuinely different."""
    paths = []
    excluded = set()
    for _ in range(k):
        hops = _best_path(adj, start, target, max_hops, excluded, person_by_id, degree)
        if hops is None:
            break
        paths.append(hops)
        for pid, _edge in hops[1:-1]:  # exclude this route's bridges next time
            excluded.add(pid)
    return paths


def _verified_routes(db: Session, routes, person_by_id, cancel_checker=None):
    """Drop any candidate route with a hop that fails verification.

    Verifies only the hops in routes _diverse_paths already found -- not
    every edge considered during search -- so cost scales with what's shown
    to users. A rejected hop drops its WHOLE route rather than triggering a
    live re-search excluding just that edge: _diverse_paths already computed
    several genuinely different candidates (that's what "diverse" means
    here), so filtering the list it returns is enough without new search
    machinery."""
    kept = []
    for hops in routes:
        if cancel_checker:
            cancel_checker()
        ok = True
        for pid, edge in hops:
            if edge is None:
                continue
            a_name = person_by_id.get(edge.person_a_id)
            b_name = person_by_id.get(edge.person_b_id)
            a_name = a_name.canonical_name if a_name else edge.person_a_id
            b_name = b_name.canonical_name if b_name else edge.person_b_id
            if not hop_verify.verify(db, edge, a_name, b_name):
                ok = False
                break
        if ok:
            kept.append(hops)
    return kept


def _score(edges: List[RelationshipEdge]) -> float:
    if not edges:
        return 1.0
    avg_conf = sum((e.confidence_raw or 0) for e in edges) / len(edges)
    avg_strength = sum(REL_STRENGTH.get(e.relationship_type, 0.4) for e in edges) / len(edges)
    return round(avg_conf * avg_strength, 3)


_PROBE_ID_CHUNK = 500  # bound-parameter safety margin as the graph grows


def _route_exists(db: Session, name_a: str, name_b: str, max_hops: int) -> bool:
    """Does ANY traversable route within max_hops already exist? Bounded,
    indexed, hop-by-hop walk out of A, stopping the moment B is reached.

    Was: _adjacency() + _diverse_paths(k=1) -- an unfiltered SELECT across
    Person, Source, and RelationshipEdge, rebuilding the WHOLE graph's
    adjacency map from scratch, to answer a yes/no question this is called
    on every /connect request (once upfront, then again after every node
    expand_graph processes -- see should_stop). The graph is shared and
    additive, never reset, so that price only grew with every run the app
    had ever done -- including for two people who turn out not to be
    connected at all. Cost now scales with the neighborhood actually
    walked, not total graph size.

    Traversability is deliberately the same single rule _path_worthy applies
    ('rejected' is out, everything else -- weak-status and unknown-typed
    edges, and a NULL status -- is walkable) and nothing more: no edge cost,
    no fame/hub penalty, no route diversity. Ranking is the final scoring
    pass's job; a plain existence check doesn't need it. Matched in Python,
    not SQL: a NULL status is traversable to _path_worthy but would be
    silently dropped by `status NOT IN (...)`.

    Each far endpoint is joined back to `people`, exactly like _adjacency's
    person_by_id requirement -- without it, a pair of dangling edges could
    bridge a "route" through a person who no longer exists, and since a True
    here skips the live search entirely, that would report "already
    connected" and then have the final scoring pass return no path at all
    (same shape as the bug test_adjacency_skips_edges_with_a_missing_endpoint
    guards against, just reachable through this function instead)."""
    if db.in_transaction():
        db.rollback()
    a_id = db.execute(
        select(Person.id).where(Person.norm_name == person_norm_key(name_a))
    ).scalar_one_or_none()
    b_id = db.execute(
        select(Person.id).where(Person.norm_name == person_norm_key(name_b))
    ).scalar_one_or_none()
    if a_id is None or b_id is None:
        return False
    if a_id == b_id:
        return True

    frontier = {a_id}
    visited = {a_id}
    for _ in range(max_hops):
        if not frontier:
            return False
        next_frontier = set()
        ids = list(frontier)
        for i in range(0, len(ids), _PROBE_ID_CHUNK):
            chunk = ids[i:i + _PROBE_ID_CHUNK]
            rows = db.execute(
                select(RelationshipEdge.person_b_id, RelationshipEdge.status,
                      RelationshipEdge.relationship_type, RelationshipEdge.signals)
                .join(Person, Person.id == RelationshipEdge.person_b_id)
                .where(RelationshipEdge.person_a_id.in_(chunk))
            ).all() + db.execute(
                select(RelationshipEdge.person_a_id, RelationshipEdge.status,
                      RelationshipEdge.relationship_type, RelationshipEdge.signals)
                .join(Person, Person.id == RelationshipEdge.person_a_id)
                .where(RelationshipEdge.person_b_id.in_(chunk))
            ).all()
            for far_id, status, rtype, signals in rows:
                if _untraversable(status, rtype, signals):
                    continue
                if far_id == b_id:
                    return True
                if far_id not in visited:
                    next_frontier.add(far_id)
        visited |= next_frontier
        frontier = next_frontier
    return False


# When one side of a /connect pair is a public figure and the other isn't,
# expanding both to the same depth is disproportionate: the famous side
# balloons into a huge, expensive, slow-to-prune amount of data (see the
# Larry Ellison / Prantik Chakraborty case -- Ellison's expansion alone was
# large enough to collide with the other side's concurrent write and hit
# SQLite's busy_timeout), almost none of which is likely to be the actual
# bridge. A real path from an ordinary person to a public figure is far more
# likely to run through that figure's own well-documented immediate circle
# (leadership team, board seats, close associates) than to be found by
# exhaustively walking their entire network hop by hop -- the same
# "prefer_reachable" philosophy _ranked_expandable already applies when
# picking which frontier node to expand next, just applied one level higher,
# to which of the two starting people gets the full expansion.
SHALLOW_FAMOUS_DEPTH = 1

# SHALLOW_FAMOUS_DEPTH never scales with the caller's requested `depth` -- the
# famous side is always capped at exactly 1 hop, whether the request is depth
# 2 or depth 5. At depth=3 specifically that means the origin side's normal
# reach (3 hops) already dwarfs the famous side's fixed 1-hop reach, so the
# origin side gets ONE hop beyond `depth` to compensate. Deliberately scoped
# to exactly 3 (not depth>=3) -- the right scaling for 4+ hasn't been decided
# yet and generalizing without evidence would just be guessing at cost/value.
ORIGIN_EXTRA_HOP_AT_DEPTH = 3

# A trailing "of X" / "at X" / ", X" clause some names carry baked into one
# field instead of a separate context_a/context_b (e.g. "Larry Ellison of
# Oracle" typed as a single board-node name -- the frontend's Route panel has
# no separate company/context field at all, see _direct_pair_search's own
# context_a/context_b for the field that DOES exist server-side but isn't
# wired up from there). A raw Wikipedia title lookup on the combined string
# fails to match ("Larry Ellison of Oracle" isn't close enough to "Larry
# Ellison" for the notability check), silently disabling the asymmetric-depth
# mitigation below for exactly the famous-person case it exists for.
_TRAILING_CONTEXT_RE = re.compile(r",.*$|\s+(?:of|at|from|with)\s+.+$", re.IGNORECASE)


def _strip_trailing_context(name: str) -> str:
    stripped = _TRAILING_CONTEXT_RE.sub("", name).strip()
    return stripped or name


def _resolve_expansion_depths(name_a: str, name_b: str, depth: int) -> Tuple[int, int]:
    """(depth_a, depth_b) for _expand_both_concurrently.

    Symmetric (both at `depth`) unless EXACTLY one of the two is notable --
    if both are famous, or neither is, there's no clear asymmetry to
    exploit, so today's behavior stands. Notability check failing (e.g. a
    transient Wikipedia lookup error) degrades to symmetric too, same as
    any other best-effort signal in this codebase.

    Checks both the raw name and its context-stripped form (see
    _strip_trailing_context) in one batched lookup -- a person counts as
    notable if either resolves, so "Larry Ellison of Oracle" still gets
    caught even though the exact string never has its own Wikipedia page.

    See ORIGIN_EXTRA_HOP_AT_DEPTH: at depth=3 exactly, the non-famous
    (origin) side's full depth is bumped by one hop to partially offset the
    famous side's fixed 1-hop cap not scaling with `depth`.
    """
    stripped_a, stripped_b = _strip_trailing_context(name_a), _strip_trailing_context(name_b)
    try:
        notable = ORCH.notable_set(list({name_a, stripped_a, name_b, stripped_b}))
    except Exception:
        return depth, depth
    a_notable = name_a in notable or stripped_a in notable
    b_notable = name_b in notable or stripped_b in notable
    if a_notable == b_notable:
        return depth, depth
    shallow = min(SHALLOW_FAMOUS_DEPTH, depth)
    full = depth + 1 if depth == ORIGIN_EXTRA_HOP_AT_DEPTH else depth
    return (shallow, full) if a_notable else (full, shallow)


def _origin_is_operator(db: Session, origin_name: str, owner_name: str) -> bool:
    """Whether the origin is the person who actually uploaded the contacts.

    Two ways to know, both by normalized name so "siddhu yen" matches
    "Siddhu Yen": the caller said so (`owner_name`, which /connect forwards
    from the browser's stored operator identity), or a saved OwnerProfile
    matches the origin. Either is sufficient.

    Returns False when neither is available. That is the safe direction: the
    only thing gated on this claims the origin personally knows every imported
    contact, and asserting that about the wrong person writes first-degree ties
    into a shared graph that /connect will then route through as if real.
    """
    origin_key = person_norm_key(origin_name or "")
    if not origin_key:
        return False
    if person_norm_key(owner_name or "") == origin_key:
        return True
    profile = get_owner_by_name(db, origin_name)
    return profile is not None


def _ensure_origin_enriched(db: Session, origin_name: str, progress=None,
                            owner_name: str = "") -> dict:
    """Step 1 of every /connect: materialize the ORIGIN's own network.

    The origin's contacts are the operator's ground truth — the nodes most
    routes actually run through — but they only reach the shared graph via two
    derivations that, until now, ran solely as a side effect of importing a
    CSV: the linkedin_1st bridge (ingest.backfill_graph_edges) and wave 0's org
    membership and coworker cliques (cliques.materialize_contact_cliques). A
    connect whose operator imported contacts on another device, or before
    either derivation existed, was pathfinding over a graph that simply did not
    contain their own first degree.

    Making it step 1 of the walk rather than a step of import is safe because
    BOTH derivations are free and idempotent: no searches, no page fetches, no
    Claude, and stable synthetic source URLs so re-running converges instead of
    duplicating. The cost of doing it on every connect is a couple of local
    queries; the cost of NOT doing it is a "no path" answer produced by an
    absence in the graph rather than an absence in the world.

    Deliberately NOT the paid part of enrichment. Expanding the origin's
    contacts costs ~35 queries each and is target-dependent, so it belongs to
    the ranked bridge front (_bridge_contacts), which knows who is being
    reached. This establishes the foundation that front then walks out from.
    """
    counts = {"linkedin_1st_edges": 0, "wave0": {}}
    if not config.CONNECT_ENRICH_ORIGIN or not (origin_name or "").strip():
        return counts
    # Best-effort throughout: an origin with no imported contacts is the normal
    # case for a famous-to-famous connect, and a failure here must never cost
    # the caller a route the rest of the walk could still have found.
    try:
        # The contact bridge asserts that every imported contact is a FIRST-
        # DEGREE connection of the origin. That is only true when the origin is
        # the operator who uploaded them, and /connect's person_a is whichever
        # node happened to be tagged 📍 -- so it runs ONLY on an identity match.
        # For anyone else the contacts are simply not their connections, and
        # writing them anyway would invent ties the pathfinder then walks.
        if config.CONNECT_ORIGIN_BACKFILL and _origin_is_operator(
                db, origin_name, owner_name):
            counts["linkedin_1st_edges"] = backfill_graph_edges(db, origin_name)
        elif progress:
            progress(f"[origin] {origin_name} is not the contact owner — "
                     "skipping first-degree bridge")
        # Wave 0 is origin-independent -- it derives org membership and
        # coworker cliques from the contacts' own employer columns, asserting
        # nothing about the origin at all -- so it always runs. `owner` only
        # adds the origin to their own employer cluster when a saved profile
        # happens to match by name; None simply omits that.
        counts["wave0"] = materialize_contact_cliques(
            db, owner=get_owner_by_name(db, origin_name))
    except Exception as exc:
        if progress:
            progress(f"[origin] enrichment skipped ({exc.__class__.__name__})")
        # Guarded: this is cleanup on a path that is already degrading
        # gracefully, and letting a failed rollback raise here would replace a
        # skipped-but-harmless step with a dead /connect -- masking the real
        # error with a secondary one from the recovery.
        try:
            db.rollback()
        except Exception:
            pass
        return counts
    if progress and (counts["linkedin_1st_edges"] or counts["wave0"].get("cliques")):
        wave0 = counts["wave0"]
        progress(f"[origin] {counts['linkedin_1st_edges']} first-degree edges, "
                 f"{wave0.get('membership_edges', 0)} org memberships, "
                 f"{wave0.get('coworker_edges', 0)} coworker ties")
    return counts


def _bridge_target(db: Session, name: str, context: str) -> BridgeTarget:
    """Everything already known about the far endpoint, for ranking bridges.

    Strictly free: the employers come from org edges the graph already holds
    (or, on a cold graph, from nothing at all), and the silo weights are
    derived from that same text. No provider call, no Claude — this runs before
    the expansion it is meant to aim, so it cannot afford to cost anything.
    """
    companies: List[str] = []
    person = db.execute(
        select(Person).where(Person.norm_name == person_norm_key(name))
    ).scalar_one_or_none()
    if person is not None:
        aff = _org_affiliations(db).get(person.id, {})
        # Best-evidenced first: a weakly-evidenced employer is a weak thing to
        # aim a whole contact-selection pass at.
        companies = [n for n, _c in
                     sorted(aff.values(), key=lambda v: -v[1])][:_TARGET_ORG_LIMIT]
    if context and context not in companies:
        companies.insert(0, context)
    return BridgeTarget(
        name=name, context=context, companies=companies,
        silo_weights=initial_weights(companies=companies) if companies else {},
    )


def _bridge_contacts(db: Session, target: BridgeTarget,
                     limit: int, progress=None,
                     origin_name: str = "", origin_context: str = "") -> List[ScoredContact]:
    """The operator's own contacts most likely to bridge to `target`.

    This is the third expansion front, and the reason /connect no longer
    depends on a frozen batch: L1 is a set of known-real people whose ties the
    graph mostly hasn't explored, and WHICH of them is worth exploring is a
    function of who we're trying to reach — a question the cold-start batch
    ranking could not have asked, because at plan time no target existed.

    Contacts with no org context are skipped by score_contacts itself (a bare
    name can't be searched without attaching a namesake's network), so they
    never reach the front.

    Two stages, in this order for a reason. score_contacts is exact, free and
    handles the whole export; bridge_strategy then reasons over the top of that
    ranking about which overlap actually matters for THIS target. Reasoning
    only REORDERS — the queue is still `limit` long either way, so a good call
    front-loads the right contact and a bad one costs nothing beyond the call
    itself, because expansion runs the queue sequentially and stops the moment
    any of them closes the route.
    """
    if limit <= 0:
        return []
    # owner_name is what lets score_contacts drop the operator from their own
    # contact list ("the operator is not their own contact"). Omitting it left
    # owner_norm empty, so that guard compared every contact against "" and
    # never fired -- confirmed live, where the origin came back as the #1
    # bridge toward the target. Expanding them here is pure waste: front A is
    # already walking that exact node, so the slot bought a duplicate instead
    # of a route.
    scored = score_contacts(db, owner_name=origin_name, target=target)
    eligible = [c for c in scored if c.skip_reason is None]
    picked = eligible[:limit]
    decision = None

    if bridge_strategy.is_active() and len(eligible) > 1:
        shortlist = eligible[:max(limit, config.BRIDGE_SHORTLIST)]
        try:
            decision = bridge_strategy.choose(
                origin_name or "the operator", origin_context,
                target.name, target.context, sorted(target.orgs()), shortlist)
        except Exception:
            decision = None  # speculative front: never let ranking fail a connect
        if decision and decision["picks"]:
            promoted = [shortlist[i] for i in decision["picks"]]
            promoted_ids = {c.local_profile_id for c in promoted}
            # Everything not promoted keeps its deterministic order behind the
            # picks. This is the fallback that makes a wrong pick survivable.
            rest = [c for c in eligible if c.local_profile_id not in promoted_ids]
            picked = (promoted + rest)[:limit]

    if picked and progress:
        if decision:
            progress(f"  [strategy] {decision['angle']} — {decision['why']}")
        promoted_n = len(decision["picks"]) if decision else 0
        for rank, contact in enumerate(picked):
            why = ", ".join(contact.bridge_reasons) or "best available"
            mark = " ←first" if decision and rank < promoted_n else ""
            progress(f"  · {contact.display_name} ({contact.context}) — {why}{mark}")
    return picked


def _expand_bridge_contacts(WorkerSession, contacts: List[ScoredContact],
                            protected: set, progress, target_name: str,
                            target_context: str,
                            cancel_checker: Optional[Callable[[], None]] = None,
                            should_stop: Optional[Callable[[Session], bool]] = None) -> dict:
    """Expand each selected bridge contact, one hop, best-ranked first.

    SEQUENTIAL by design, unlike the two endpoint expansions that run beside
    it. Each expand_graph already runs its own per-node worker pool, so fanning
    the contacts out too would multiply the outbound request rate by the
    contact count — and the whole front is speculative. Running them in rank
    order and re-checking `should_stop` between each means the moment either
    endpoint expansion (or a bridge contact itself) completes the route, the
    remaining contacts are simply never paid for.

    Depth 1: the goal is to surface each contact's OWN ties so they can meet
    the target's expanding neighborhood, not to walk outward from them. Their
    hop-2 is the target side's job.

    Alpha runs on every contact (see the kwargs below) -- these are the nodes
    it was designed for, and unlike the endpoint walk there is no notability
    asymmetry to infer the flag from.
    """
    stats: Dict[str, dict] = {}
    for contact in contacts:
        worker_db = WorkerSession()
        try:
            if cancel_checker:
                cancel_checker()
            if should_stop and should_stop(worker_db):
                break
            if progress:
                progress(f"\n[bridge] {contact.display_name} "
                         f"({contact.context or 'no context'})…")
            kwargs = {
                "progress": progress,
                "seed_context": contact.context,
                "protected_norms": protected,
                "prefer_reachable": False,
                "silo_weights": contact.silo_weights or None,
                "target_person_name": target_name,
                "target_context": target_context,
                # Alpha (steps 4/5/6 and phase 4c) applies here for the same
                # reason it applies to the non-famous side of an asymmetric
                # walk: a bridge contact IS a normal person being searched
                # with a specific destination in mind, which is exactly the
                # case node profiling, search-strategy angle selection and the
                # targeted re-query were built for. On the endpoint walk the
                # flag is inferred from a notability asymmetry; here there is
                # nothing to infer -- the contact is never the famous side.
                #
                # Note target_person_name above is load-bearing for this, not
                # decorative: phase 4e needs BOTH a target and this flag, so
                # passing the target alone (as this front originally did) left
                # the strategy step permanently inert.
                "enhanced_professional_search": True,
            }
            if cancel_checker:
                kwargs["cancel_checker"] = cancel_checker
            if should_stop:
                kwargs["should_stop"] = should_stop
            stats[contact.norm_name] = expand_graph(
                worker_db, contact.display_name, 1, **kwargs)
        except Exception as exc:
            # One speculative contact failing is not a reason to fail the
            # /connect — unlike an ENDPOINT expansion, whose failure means
            # there is no graph to path over. Contrast _expand_both_concurrently,
            # which deliberately propagates.
            if progress:
                progress(f"  ⚠ bridge contact {contact.display_name} failed "
                         f"({exc.__class__.__name__}) — skipped")
        finally:
            worker_db.close()
    return stats


def _expand_both_concurrently(db: Session, name_a: str, name_b: str,
                              depth_a: int, depth_b: int,
                              protected: set, progress, context_a: str, context_b: str,
                              on_step: Optional[Callable[[dict], None]] = None,
                              cancel_checker: Optional[Callable[[], None]] = None,
                              should_stop: Optional[Callable[[Session], bool]] = None) -> dict:
    """Run both endpoints' expand_graph calls concurrently, each on its own
    Session (bound to the same engine as `db` — a Session isn't thread-safe to
    share). The two sides are fully independent expansions into the same
    shared graph; nothing about A needs B done first, so there's no reason
    the old "[1/2] then [2/2]" sequencing should hold up the wall clock.

    depth_a/depth_b are independent (see _resolve_expansion_depths) -- a
    famous endpoint gets a shallow, immediate-circle-only expansion while
    the other side gets the full requested depth, rather than both sides
    always expanding equally regardless of how disproportionate the
    resulting data volume would be.

    Either side's exception propagates via future.result() — same as an
    unhandled exception from a sequential call would have; this is not the
    place to silently swallow a genuine failure (contrast with expand_graph's
    OWN per-node worker pool, which deliberately skips a failed node rather
    than aborting the whole hop — a full endpoint failing is not that).

    Returns {"a": <side A's expand_graph stats>, "b": <side B's>} -- each
    includes visited_by_hop, so connect_people can show what was explored
    on BOTH sides even when they never actually met (see its "explored"
    field)."""
    engine = db.get_bind()
    WorkerSession = sessionmaker(bind=engine, autoflush=False,
                                 expire_on_commit=False, future=True)

    # Asymmetric depth (see _resolve_expansion_depths) means exactly one side
    # is famous and shallow -- the OTHER, full-depth side is the one actually
    # walking TOWARD a famous target, which is the situation
    # expansion._process_person's targeted-recheck phase (4c) exists for. No
    # asymmetry (both/neither notable) -> no clear famous target to walk
    # toward -> neither side gets it, same as today.
    enhanced_a = depth_a > depth_b
    enhanced_b = depth_b > depth_a
    # Mirror image, to the OTHER side: once the full-depth side's targeted
    # search has effectively concluded "the bridge is professional" (that's
    # what triggered the asymmetric depth to begin with), the famous side's
    # own limited 1-hop budget should spend itself on colleagues and board
    # seats, not family/friends silos -- see expansion.expand_graph's
    # `professional_only` and PROFESSIONAL_SILOS.
    professional_only_a = depth_a < depth_b
    professional_only_b = depth_b < depth_a

    def _run(name: str, context: str, label: str, side_depth: int,
             enhanced: bool, professional_only: bool,
             target_name: str, target_context: str) -> dict:
        worker_db = WorkerSession()
        try:
            if cancel_checker:
                cancel_checker()
            if progress:
                progress(f"\n[{label}] building graph for {name} (depth {side_depth})…")
            step_cb = (lambda evt, side=label.lower(): on_step({**evt, "side": side})) if on_step else None
            kwargs = {
                "progress": progress,
                "seed_context": context,
                "protected_norms": protected,
                "on_step": step_cb,
                # Point-to-point bridging wants STRONGEST expansion, not the
                # reachability walk /discover uses -- see connect_people. Passed
                # per call so a concurrent /discover build keeps its own mode.
                "prefer_reachable": False,
                "enhanced_professional_search": enhanced,
                "professional_only": professional_only,
                # Alpha step 6 (search_strategy): the OTHER endpoint's name/
                # context, so the non-famous side's strategy decision can
                # reason about who it's actually walking toward instead of
                # picking an angle in the abstract.
                "target_person_name": target_name,
                "target_context": target_context,
            }
            if cancel_checker:
                kwargs["cancel_checker"] = cancel_checker
            if should_stop:
                kwargs["should_stop"] = should_stop
            return expand_graph(worker_db, name, side_depth, **kwargs)
        finally:
            worker_db.close()

    # The third front: the operator's own contacts, ranked toward B. Selected
    # here (not by the caller) because it must happen INSIDE the "no route yet"
    # branch -- a /connect answered from the existing graph should stay free.
    #
    # The whole front is speculative, so nothing about it may be load-bearing:
    # if SELECTING contacts fails (as expanding one already can), fall back to
    # the plain two-endpoint walk rather than failing a /connect that would
    # otherwise have succeeded.
    try:
        bridges = _bridge_contacts(db, _bridge_target(db, name_b, context_b),
                                   config.CONNECT_BRIDGE_CONTACTS, progress=progress,
                                   origin_name=name_a, origin_context=context_a)
    except Exception as exc:
        if progress:
            progress(f"[bridge] contact ranking unavailable "
                     f"({exc.__class__.__name__}) — endpoints only")
        bridges = []
    if bridges and progress:
        progress(f"[bridge] {len(bridges)} contact(s) ranked toward {name_b}")
    # Endpoints AND bridge contacts must survive every side's noise prune: a
    # bridge contact deleted by side A's prune takes its freshly-discovered
    # ties with it, which is the whole point of having expanded it.
    protected = set(protected) | {c.norm_name for c in bridges}

    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {
            "a": ex.submit(_run, name_a, context_a, "A", depth_a, enhanced_a, professional_only_a,
                          name_b, context_b),
            "b": ex.submit(_run, name_b, context_b, "B", depth_b, enhanced_b, professional_only_b,
                          name_a, context_a),
        }
        if bridges:
            futures["bridge"] = ex.submit(
                _expand_bridge_contacts, WorkerSession, bridges, protected,
                progress, name_b, context_b, cancel_checker, should_stop)
        # Each side's own visited_by_hop (see expand_graph) -- so a caller
        # can show what was explored on BOTH sides even when the two never
        # actually met (see connect_people's "explored" field).
        return {side: f.result() for side, f in futures.items()}


_WIKIPEDIA_TITLE_RE = re.compile(r"wikipedia\.org/wiki/([^?#]+)")


def _split_sentences(text: str) -> List[str]:
    """Abbreviation-aware via spaCy when available (a naive '. '-based
    regex splitter breaks on "U.S."/"Dr."/etc, turning one real sentence
    into two fragments and silently pushing two co-mentioned names further
    apart than they really are in the text -- this is what originally hid
    the Redfield/Trump appointment sentence from even a 2-sentence window).
    Falls back to the regex splitter only when spaCy isn't installed.

    Delegates to spacy_extractor.split_sentences, which is the same two-step
    shared with extraction.subject_windows -- kept as a local alias because
    this name is what the rest of this module (and its tests) reach for."""
    return spacy_extractor.split_sentences(text)


def _sentence_windows(sentences: List[str], window: int = 2) -> List[str]:
    """Consecutive-sentence windows, joined into one string each.

    A relationship is often stated across a sentence boundary via a pronoun
    -- "Redfield became Director... He was appointed to the post by
    President Donald Trump..." -- which a single-sentence-only check misses
    entirely, since that second sentence never says "Redfield" by name.
    Requiring both names within a small window instead of one sentence
    catches this without needing real coreference resolution.
    """
    return [" ".join(sentences[i:i + window]) for i in range(len(sentences))]


def _name_mention_pattern(name: str, other_name: str = "") -> Tuple[re.Pattern, Optional[re.Pattern]]:
    """Returns (mention_pattern, conflict_pattern).

    mention_pattern matches either the full name or just its last token
    (surname) -- real prose re-mentions someone by surname alone after the
    first full mention ("Redfield" / "Trump", never "Robert R Redfield"
    again), and requiring the exact full name every time would miss almost
    every real sentence, including the one that actually states the
    relationship (see module docs: the Wikipedia ARTICLE BODY has "He was
    appointed to the post by President Donald Trump...", using surnames
    throughout).

    But a bare surname is genuinely ambiguous for anyone who shares it with
    someone else notable -- scanning a WHOLE article for "Trump" also
    matches "Ivanka Trump", "Trump Tower", "Fred Trump", none of which are
    Donald Trump. conflict_pattern matches a DIFFERENT full name sharing the
    same surname (a capitalized word immediately before it that isn't this
    person's own first name AND isn't a title/honorific like "President" --
    "President Trump" is the same Donald Trump, not a different person);
    a window matching that should be treated as probably about someone
    else, not silently trusted as a mention of this person. None when the
    name has no first name to compare against (a mononym), since there's
    nothing to distinguish it from in that case.

    other_name is the counterpart being searched for in the SAME pair search
    (e.g. name_b, when this is name_a's pattern). Its first name is excluded
    from the conflict check too -- otherwise, whenever the two people being
    searched for share a surname (spouses, siblings, parent/child -- exactly
    the family_social case this exists to support), the counterpart's own
    full-name mention ("Jane Smith" while building John Smith's pattern)
    would misfire the conflict check as if it named some unrelated third
    Smith, dropping every window that states the very relationship being
    searched for.

    Delegates to utils.names.mention_patterns, which is where this moved so
    extraction.subject_windows could ask the same question -- kept as a local
    alias because this name is what the rest of this module (and its tests)
    reach for.
    """
    return mention_patterns(name, other_name)


def _wikipedia_title_from_url(url: str) -> Optional[str]:
    m = _WIKIPEDIA_TITLE_RE.search(url)
    if not m:
        return None
    return unquote(m.group(1)).replace("_", " ")


def _fetch_result_text(res) -> str:
    """Full plain-text content for one search result.

    Detecting Wikipedia by URL, not `res.provider` -- a Wikipedia page that
    Serper (or Brave/DuckDuckGo) surfaces as an ordinary organic hit carries
    that provider's name, not "wikipedia" (`res.provider == "wikipedia"`
    only fires when the dedicated Wikipedia provider itself was queried
    directly, which never happens in this flow). Missing that meant every
    Wikipedia URL here got raw HTML-scraped instead, mangling the infobox
    into unparseable pseudo-sentences ("In office March 26, 2018 ... Deputy
    Anne Schuchat Preceded by...").

    Uses the full article body (wikipedia.article_text), not just the lead
    summary (wikipedia.summary) -- the summary is often too short to mention
    a secondary figure at all (Redfield's summary never says "Trump"), while
    the full article's body does, in plain, real sentences.
    """
    title = _wikipedia_title_from_url(res.url)
    if title:
        text = ORCH.wikipedia.article_text(title)
        if text:
            return text
    try:
        page = ORCH.fetch(res.url)
        return html_to_text(page.content) if page.content else res.snippet
    except Exception:
        return res.snippet


def _direct_pair_search(db: Session, name_a: str, name_b: str, context_a: str = "",
                        context_b: str = "",
                        cancel_checker: Optional[Callable[[], None]] = None) -> Tuple[bool, bool]:
    """Cheap first-pass: search for the two people TOGETHER and extract any
    edge found directly between them, before paying for a full bidirectional
    neighborhood walk.

    Generic per-person silo expansion (_expand_both_concurrently) rarely
    reaches a specific pair organically — a famous person's neighborhood is
    far too large to exhaustively walk within any reasonable depth/node
    budget, so a real, well-documented, easily-searchable fact (e.g. a
    government appointment) can go undiscovered even though it's one search
    away. This checks that directly: a single query naming both people.

    Returns (found, confident). `confident` is True only if at least one
    persisted edge reached better than 'weak' status.

    When Claude is configured, this scans every sentence (across every
    fetched result, not just one guess per page) that mentions BOTH people
    by name, and hands ALL of them to the Claude relationship classifier
    (extraction.relation_classifier) in one combined batched call -- real
    language understanding of the evidence, not a keyword-table guess that
    can confidently mislabel evidence it doesn't recognize (this is what
    produced the wrong 'interview' label on Redfield/Trump: a keyword table
    matched nothing, so a silo's intent_default guessed wrong -- and that
    would have mislabeled a tenth *result* exactly like it mislabeled the
    first, since more results doesn't fix a classification gap). Scanning
    every co-mentioning sentence (not just the first spaCy happened to
    associate with the entity) is what actually surfaces the sentence that
    states the relationship outright, wherever in the article it sits.

    Claude's own "unknown" verdict for a sentence is trusted as-is (not
    treated as a non-answer to fall back from) -- an honest "the evidence
    doesn't say" is strictly better than keeping a keyword-guessed label
    Claude explicitly declined to support.

    Falls back to the original single-extraction-per-page keyword guess
    (spaCy/heuristic + a silo's signal table) when Claude isn't configured
    at all, mirroring _retype_unknown_edges's own is_active() guard, so
    behavior degrades gracefully rather than losing edges outright.
    """
    query = f'"{name_a}" "{name_b}" {context_a} {context_b}'.strip()
    try:
        results = ORCH.search(query, is_person=True)
    except Exception:
        return False, False
    if not results:
        return False, False

    if relation_classifier.is_active():
        return _direct_pair_search_via_claude(db, name_a, name_b, query, results, cancel_checker)
    return _direct_pair_search_via_keywords(db, name_a, name_b, query, results, cancel_checker)


def _direct_pair_search_via_claude(db: Session, name_a: str, name_b: str, query: str,
                                   results, cancel_checker) -> Tuple[bool, bool]:
    a_pat, a_conflict = _name_mention_pattern(name_a, other_name=name_b)
    b_pat, b_conflict = _name_mention_pattern(name_b, other_name=name_a)

    # (Source, window) for every 2-sentence window, in every fetched result,
    # naming BOTH people -- not just spaCy's single nearest-sentence-per-
    # mention guess, which can land on an unrelated or garbled sentence
    # while the real, explicit one sits a paragraph away. Windows (not just
    # single sentences) catch a relationship stated across a sentence
    # boundary via a pronoun (see _sentence_windows). A window is skipped
    # if it contains a DIFFERENT full name sharing either person's surname
    # (e.g. "Ivanka Trump" when name_b is "Donald Trump") -- it's more
    # likely about that other person, not evidence naming the actual target.
    seen_windows = set()
    candidates: List[Tuple[Source, str]] = []
    for res in results:
        if cancel_checker:
            cancel_checker()
        text = _fetch_result_text(res)
        if not text:
            continue
        source = builder.save_source(db, res, query, text)
        for window in _sentence_windows(_split_sentences(text)):
            if window in seen_windows:
                continue
            if a_conflict and a_conflict.search(window):
                continue
            if b_conflict and b_conflict.search(window):
                continue
            if a_pat.search(window) and b_pat.search(window):
                seen_windows.add(window)
                candidates.append((source, window[:700]))

    if not candidates:
        return False, False

    # One combined call classifying EVERY candidate sentence -- batching is
    # relation_classifier.classify's own concern (config.CLAUDE_CLASSIFY_BATCH),
    # so this stays one logical call regardless of how many sentences matched.
    items = [{"a": name_a, "b": name_b, "evidence": sentence} for _src, sentence in candidates]
    verdicts = relation_classifier.classify(items)

    subject = builder.get_or_create_person(db, name_a)
    counterpart = builder.get_or_create_person(db, name_b)
    if subject is None or counterpart is None:
        return False, False

    found = False
    confident = False
    for (source, sentence), verdict in zip(candidates, verdicts):
        rtype = verdict.get("type", "unknown")
        conf = verdict.get("confidence", 0.0)
        if rtype != "unknown" and conf >= config.CLAUDE_CLASSIFY_MIN_CONF:
            final_type = rtype
            final_conf = round(min(conf, config.RELATION_CONF_CEILING), 3)
        else:
            # Claude looked at THIS sentence and couldn't confidently
            # support a specific type -- trust that verdict as-is rather
            # than inventing a keyword-guessed label it explicitly declined.
            final_type = "unknown"
            final_conf = round(conf, 3)
        edge = ExtractedEdge(
            person_a=name_a, person_b=name_b, other_kind="person",
            relationship_type=final_type,
            method="Claude relationship classification",
            evidence_snippet=sentence, source_url="",
            confidence_base=final_conf, confidence_adjusted=final_conf,
            signals=EdgeSignals(explicit_keyword_match=(final_type != "unknown")),
        )
        persisted = builder.add_edge_from_extraction(db, subject, edge, 0, source, counterpart)
        found = True
        # relationship_type != "unknown" is required, not just a status
        # check: nothing guarantees Claude pairs "unknown" with confidence
        # 0.0 (every real call this session happened to, but the schema
        # only says confidence is "how clearly the evidence supports the
        # type you picked" -- for type="unknown" that's not well-defined,
        # and a non-zero value would otherwise land above WEAK_MAX and get
        # reported as a confident match for a relationship we don't
        # actually know the type of).
        if (persisted is not None and persisted.relationship_type != "unknown"
                and persisted.status != "weak"):
            confident = True
    if found:
        # commit_with_retry, not a bare db.commit(): every edge above was
        # already durably persisted via add_edge_from_extraction's own
        # SAVEPOINT retry, so nothing new needs reapplying here -- this just
        # covers the rare case where this commit is itself this
        # transaction's first write (see builder.commit_with_retry).
        builder.commit_with_retry(db)
    return found, confident


def _direct_pair_search_via_keywords(db: Session, name_a: str, name_b: str, query: str,
                                     results, cancel_checker) -> Tuple[bool, bool]:
    """Degraded-mode fallback when Claude isn't configured: one
    spaCy/heuristic extraction pass per result, typed by a keyword-signal
    table instead of real reasoning.

    Uses COLLEAGUE_SILO, not SILOS[0] -- SILOS[0] ("news") has
    intent_default=True with default_relationship="interview", so any
    co-occurrence text that doesn't happen to contain one of its 6 keywords
    ("interview"/"podcast"/"named"/"joins board"/"board"/"appointed") would
    get confidently typed "interview" regardless of what the text actually
    says -- reproducing, in this fallback, the exact wrong-default bug this
    module exists to fix for the Claude path. COLLEAGUE_SILO carries broad
    signal coverage across most relationship types (same table as
    STRUCTURED_SILO) with intent_default=False, so a real keyword still gets
    typed correctly, and absent one it honestly falls through to "unknown"
    instead of guessing."""
    b_norm = person_norm_key(name_b)
    candidates: List[Tuple[ExtractedEdge, Source]] = []
    for res in results:
        if cancel_checker:
            cancel_checker()
        text = _fetch_result_text(res)
        if not text:
            continue
        source = builder.save_source(db, res, query, text)
        out = extract(name_a, text, COLLEAGUE_SILO, res.snippet, res.url)
        for edge in out.edges:
            if edge.other_kind != "person" or person_norm_key(edge.person_b) != b_norm:
                continue
            candidates.append((edge, source))

    if not candidates:
        return False, False

    found = False
    confident = False
    for edge, source in candidates:
        subject = builder.get_or_create_person(db, name_a)
        counterpart = builder.get_or_create_person(db, name_b)
        if subject is None or counterpart is None:
            continue
        persisted = builder.add_edge_from_extraction(db, subject, edge, 0, source, counterpart)
        found = True
        # Same reasoning as the Claude path: an "unknown"-typed edge must
        # never count as confident, regardless of its confidence_raw --
        # cooccurrence-driven confidence can land above WEAK_MAX even when
        # no keyword actually matched a relationship type.
        if (persisted is not None and persisted.relationship_type != "unknown"
                and persisted.status != "weak"):
            confident = True
    if found:
        # commit_with_retry, not a bare db.commit(): every edge above was
        # already durably persisted via add_edge_from_extraction's own
        # SAVEPOINT retry, so nothing new needs reapplying here -- this just
        # covers the rare case where this commit is itself this
        # transaction's first write (see builder.commit_with_retry).
        builder.commit_with_retry(db)
    return found, confident


def _build_explored(expand_stats: Optional[dict], name_a: str, name_b: str) -> Optional[dict]:
    """Each side's per-hop explored node names (see expand_graph's
    visited_by_hop), for a caller to visualize what Artemis actually looked
    at even when the two sides never met -- not just the found path, which
    doesn't exist in that case at all.

    None when no fresh expansion ran (a route was already known, or found
    via the cheap direct-pair check) -- nothing new was explored, so
    there's nothing to show beyond the route itself.
    """
    if not expand_stats:
        return None
    explored = {
        "a": {"seed": name_a, "by_hop": (expand_stats.get("a") or {}).get("visited_by_hop", {}),
              "boundary": (expand_stats.get("a") or {}).get("boundary", [])},
        "b": {"seed": name_b, "by_hop": (expand_stats.get("b") or {}).get("visited_by_hop", {}),
              "boundary": (expand_stats.get("b") or {}).get("boundary", [])},
    }
    # The bridge front is keyed by contact rather than by hop -- each contact is
    # its own depth-1 expansion, so there is no single seed to report. Present
    # only when contacts were actually expanded, so a caller can tell "no
    # contacts were worth it" apart from "the feature is off".
    bridge = expand_stats.get("bridge")
    if bridge:
        explored["bridge"] = {
            norm: {"by_hop": (stats or {}).get("visited_by_hop", {}),
                   "boundary": (stats or {}).get("boundary", [])}
            for norm, stats in bridge.items()
        }
    return explored


def connect_people(db: Session, name_a: str, name_b: str, depth: int = 2,
                   progress=None, context_a: str = "", context_b: str = "",
                   on_step: Optional[Callable[[dict], None]] = None,
                   cancel_checker: Optional[Callable[[], None]] = None,
                   owner_name: str = "") -> dict:
    """Build both graphs, then return the best path between the two people.

    context_a / context_b disambiguate a non-notable person (e.g. "Indiana
    Pacers owner") so the search targets the right entity, not a famous namesake.

    `on_step`, like expand_graph's own, reports structured per-side hop/node
    progress (each event tagged {"side": "a"|"b"}) instead of free-text lines.

    THREE fronts expand, not two: both endpoints, plus the operator's own
    contacts ranked toward name_b (see _bridge_contacts). That third front is
    what makes enrichment specific to this connect — the contacts worth
    searching depend on who is being reached, and a batch planned before any
    target existed could only ever have guessed. It runs solely on the
    no-route-yet path, so an answer available from the existing graph, or from
    the cheap direct-pair check, still costs nothing.

    `owner_name` is who the CALLER is, which is not necessarily name_a: step 1
    bridges the imported contacts to the origin as first-degree ties only when
    the two are the same person (see _origin_is_operator). Empty means "not
    stated", which suppresses that bridge rather than guessing.
    """
    # ADDITIVE: build both people INTO the shared global map (never reset), then
    # find a path over the WHOLE accumulated graph — a route may run through
    # people that OTHER runs discovered. Point-to-point bridging wants STRONGEST
    # expansion (toward shared, well-documented connections), not reachability;
    # request it per-call. This used to assign config.EXPAND_PREFER_REACHABLE
    # and restore it in a finally block, which made every concurrent build in
    # the API unsafe and is why they were serialized behind one lock.
    # Both endpoints must survive BOTH expansions' noise-shape prune, not just
    # their own — expand_graph's own seed exemption only covers the call it's
    # made on, and the second call's prune would otherwise be free to delete
    # the first call's seed as an ordinary (unprotected) node.
    both = {person_norm_key(name_a), person_norm_key(name_b)}
    max_hops = 2 * depth + 1
    route_found = threading.Event()

    def should_stop(check_db: Session) -> bool:
        if route_found.is_set():
            return True
        if _route_exists(check_db, name_a, name_b, max_hops):
            route_found.set()
            return True
        return False

    if cancel_checker:
        cancel_checker()
    if progress:
        progress("\n[known] checking what's already in the graph…")
    if _route_exists(db, name_a, name_b, max_hops):
        # Zero-cost first check: no search, no fetch, no extraction — just
        # whatever's already persisted, which now includes linkedin_1st edges
        # bridged in from uploaded contacts (see network/ingest.py) and
        # anything a prior /connect or /discover run already found (the graph
        # is shared and additive, never reset). A known first-degree
        # connection should never cost a live search to rediscover.
        route_found.set()
        if progress:
            progress("[known] already connected in the existing graph — skipping search entirely")
    else:
        # STEP 1: the origin's own initial enrichment, before a cent is spent.
        # Free and idempotent (see _ensure_origin_enriched), so it runs on
        # every connect rather than depending on whether an import or a batch
        # run ever happened to derive it.
        if progress:
            progress("\n[origin] initial enrichment for "
                     f"{name_a} — bridging their own network into the graph…")
        _ensure_origin_enriched(db, name_a, progress=progress,
                                owner_name=owner_name)

        # Re-check, because step 1 may have just built the answer: a contact of
        # the origin IS the target, or sits one coworker tie away from them.
        # The check costs nothing and short-circuits the entire paid walk, so
        # not repeating it here would mean paying for a route we already have.
        if cancel_checker:
            cancel_checker()
        if _route_exists(db, name_a, name_b, max_hops):
            route_found.set()
            if progress:
                progress("[origin] connected through the origin's own network — "
                         "no search needed")

    if not route_found.is_set():
        # _direct_pair_search already tries every returned result (not just
        # the first few) whenever what it's found so far is only weak, so by
        # the time it returns there's nothing more to gain from SEARCHING
        # further here. Whether it found something worth STOPPING for is a
        # different question, and `confident` is what answers it.
        found, confident = _direct_pair_search(db, name_a, name_b, context_a, context_b,
                                              cancel_checker=cancel_checker)
        if found and confident:
            route_found.set()
            if progress:
                progress("[direct] found a confident direct mention — "
                         "skipping full neighborhood expansion")
        elif found:
            # A WEAK mention is not a route. `confident` used to be logged and
            # discarded, so any co-mention at all cancelled both endpoint
            # expansions -- and a weak edge is frequently one _untraversable
            # rejects (untyped, no cooccurrence, no keyword) or one the
            # pathfinder simply cannot chain through. The walk then reported
            # "no path" in a couple of seconds having never expanded the far
            # endpoint at all: no silos, no Alpha, no bridge contacts.
            #
            # Keep the edge -- it is real evidence and may yet shorten a route
            # -- but carry on with the expansion it was standing in for.
            if progress:
                progress("[direct] found only a weak direct mention — keeping it "
                         "as evidence and continuing to full expansion")

    # Populated only when a fresh expansion actually ran (not when a route
    # was already known, or found via the cheap direct-pair check) -- see
    # _build_explored below and its use in both return paths.
    expand_stats: Optional[dict] = None
    if not route_found.is_set():
        if cancel_checker:
            cancel_checker()
        depth_a, depth_b = _resolve_expansion_depths(name_a, name_b, depth)
        if progress and (depth_a, depth_b) != (depth, depth):
            shallow_side = "A" if depth_a < depth_b else "B"
            progress(f"[reachable] side {shallow_side} is a public figure — "
                     f"capping their expansion to depth {min(depth_a, depth_b)} "
                     "(immediate circle only) instead of matching the other side")
        expand_stats = _expand_both_concurrently(
            db, name_a, name_b, depth_a, depth_b, both, progress,
            context_a, context_b, on_step=on_step,
            cancel_checker=cancel_checker,
            should_stop=should_stop)

    if cancel_checker:
        cancel_checker()
    a = db.execute(
        select(Person).where(Person.norm_name == person_norm_key(name_a))
    ).scalar_one_or_none()
    b = db.execute(
        select(Person).where(Person.norm_name == person_norm_key(name_b))
    ).scalar_one_or_none()
    if a is None or b is None:
        missing = name_a if a is None else name_b
        return {"connected": False, "reason": f"'{missing}' not found in the graph"}

    adj, person_by_id, src_by_id, degree = _adjacency(db)
    if cancel_checker:
        cancel_checker()
    routes = _diverse_paths(adj, a.id, b.id, max_hops, config.CONNECT_MAX_PATHS,
                            person_by_id, degree)
    if cancel_checker:
        cancel_checker()
    had_candidates = bool(routes)
    verified = config.CLAUDE_VERIFY_HOPS and hop_verify.claude_available()
    if verified:
        routes = _verified_routes(db, routes, person_by_id, cancel_checker)
    if cancel_checker:
        cancel_checker()
    if not routes:
        # A distinct reason when candidates existed but none survived
        # verification -- "try a higher depth" would be actively misleading
        # there, since more expansion won't fix a hop that failed on its
        # own evidence.
        reason = (
            f"{config.CONNECT_MAX_PATHS} candidate route(s) found within "
            f"{max_hops} hops, but none passed hop verification — the "
            "evidence didn't hold up on inspection."
            if had_candidates and verified else
            f"no path within {max_hops} hops — their graphs don't overlap "
            f"at depth {depth}. Try a higher depth."
        )
        return {
            "connected": False,
            "person_a": a.canonical_name, "person_b": b.canonical_name,
            "reason": reason,
            "explored": _build_explored(expand_stats, name_a, name_b),
        }

    org_aff = _org_affiliations(db)
    paths = []
    for hops in routes:
        if cancel_checker:
            cancel_checker()
        path_nodes, edges_used, bridges = [], [], []
        for i, (pid, edge) in enumerate(hops):
            person = person_by_id.get(pid)
            label = person.canonical_name if person else pid
            node = {"label": label, "node_type": "public_person"}
            if person and person.meta and person.meta.get("homonym_rejected"):
                node["homonym_flag"] = person.meta["homonym_rejected"]
            if edge is not None:
                edges_used.append(edge)
                src = src_by_id.get(edge.source_id)
                node["relationship_from_previous"] = edge.relationship_type
                node["confidence"] = edge.confidence_raw
                if edge.evidence_snippet:
                    node["evidence"] = edge.evidence_snippet
                if edge.method:
                    node["method"] = edge.method
                if src and src.url:
                    node["source_url"] = src.url
                if src and src.title:
                    node["source_title"] = src.title
                # "coworker" alone doesn't say coworker WHERE -- the org both
                # ends of this hop belong to is the missing half of the answer.
                # (i > 0 whenever an edge exists: the start node is the only
                # one carrying no incoming edge.)
                via = _shared_orgs(org_aff, hops[i - 1][0], pid) if i else []
                if via:
                    node["via_orgs"] = via
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
        "warnings": [] if verified else
                    ["Path is unverified", "Requires Claude verification before activation"],
        "explored": _build_explored(expand_stats, name_a, name_b),
    }


def discover_person(db: Session, name: str, depth: int = 2, limit: int = 100) -> dict:
    """Everyone reachable from `name` within `depth` hops, cheapest-first.

    Unlike connect_people (which needs a specific bridge to a specific target),
    this ranks the ENTIRE neighborhood — so a node's own penalty (fame, hub
    fan-out) is charged even on the nodes we ultimately list, not just the ones
    we route through. A celebrity three hops out should rank behind a
    close colleague one hop out, not ahead of them.
    """
    root = db.execute(
        select(Person).where(Person.norm_name == person_norm_key(name))
    ).scalar_one_or_none()
    if root is None:
        return {"found": False, "reason": f"'{name}' is not in the graph"}

    adj, person_by_id, src_by_id, degree = _adjacency(db)

    counter_seed = 0
    dist: Dict[str, float] = {root.id: 0.0}
    hops_to: Dict[str, int] = {root.id: 0}
    first_edge: Dict[str, RelationshipEdge] = {}
    heap = [(0.0, 0, counter_seed, root.id)]
    while heap:
        cost, hops, _t, node = heapq.heappop(heap)
        if cost > dist.get(node, float("inf")) or hops >= depth:
            continue
        for neighbor, edge in adj.get(node, []):
            penalty = _node_penalty(person_by_id, degree, neighbor)
            new_cost = cost + _edge_cost(edge) + penalty
            if new_cost < dist.get(neighbor, float("inf")):
                dist[neighbor] = new_cost
                hops_to[neighbor] = hops + 1
                first_edge[neighbor] = edge if node == root.id else first_edge.get(node)
                counter_seed += 1
                heapq.heappush(heap, (new_cost, hops + 1, counter_seed, neighbor))

    connections = []
    for pid, cost in sorted(dist.items(), key=lambda kv: (hops_to.get(kv[0], 0), kv[1])):
        if pid == root.id:
            continue
        person = person_by_id.get(pid)
        if person is None:
            continue
        edge = first_edge.get(pid)
        source = src_by_id.get(edge.source_id) if edge and edge.source_id else None
        connections.append({
            "id": pid,
            "label": person.canonical_name,
            "hops": hops_to.get(pid, 0),
            "cost": round(cost, 2),
            "relationship_type": edge.relationship_type if edge else "unknown",
            "confidence": round(edge.confidence_raw or 0.0, 2) if edge else 0.0,
            "source_url": source.url if source else "",
        })
        if len(connections) >= limit:
            break

    return {
        "found": True,
        "person": root.canonical_name,
        "connections": connections,
        "count": len(connections),
    }
