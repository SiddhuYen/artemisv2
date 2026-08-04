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
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple
from urllib.parse import unquote

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from .. import config
from ..extraction import (bridge_hypothesis, bridge_strategy, extract,
                          relation_classifier, route_adjudicator, spacy_extractor)
from ..extraction.entity_filter import is_filtering_active
from ..extraction.entity_filter import validate as filter_entities
from ..extraction.schemas import EdgeSignals, ExtractedEdge, ExtractionOutput
from ..models import LocalProfile, Organization, Person, RelationshipEdge, Source
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


# The edge type that asserts "I personally know this person because I uploaded
# them". Every other type is a claim about the world, sourced from a page anyone
# could read; this one is a claim about ONE operator's address book.
CONTACT_CLAIM_TYPE = "linkedin_1st"


def _contact_edge_gate(db: Session, operator_name: str):
    """(is_traversable(person_a_id, person_b_id, relationship_type)) for this operator.

    Uploaded connections are private to whoever uploaded them, and until now the
    graph could not express that. `local_profiles` carries owner_norm, but
    ingest.backfill_graph_edges converts a profile into a RelationshipEdge and
    the conversion DROPS the owner -- relationship_edges has no owner column --
    so after the bridge ran, an edge to someone else's contact was byte-for-byte
    identical to an edge to your own.

    Observed: 2,152 linkedin_1st edges on "Abhimanyu Sharma", 1,132 of them
    pointing at a second person's LinkedIn export that happened to share the
    database. Those outrank real ties (linkedin_1st is the strongest claim in
    the graph) AND suppress the search that would replace them, because
    _route_exists short-circuits the paid walk on any traversable route.

    Ownership is recovered at READ time by joining back to local_profiles on
    norm_name, rather than by backfilling a column onto millions of edges: the
    profile table is the record of who uploaded what, and it is already correct.

    Fails closed in both directions. An endpoint whose profile has NO owner
    (imported before ownership existed) is nobody's contact and is private to
    nobody -- so it is traversable by no one. And an unidentified caller
    (operator_name empty, e.g. a famous-to-famous connect) gets no contact edges
    at all, which is right: those edges assert a private relationship, and a
    caller who has not said who they are cannot be its owner.
    """
    owner_key = person_norm_key(operator_name or "")
    profile_owner: Dict[str, Optional[str]] = {}
    self_ids: set = set()
    for pid, norm, owner in db.execute(
        select(Person.id, Person.norm_name, LocalProfile.owner_norm)
        .join(LocalProfile, LocalProfile.norm_name == Person.norm_name)
    ).all():
        profile_owner[pid] = owner
        if owner_key and norm == owner_key:
            self_ids.add(pid)

    def traversable(a_id: str, b_id: str, rtype: Optional[str]) -> bool:
        if rtype != CONTACT_CLAIM_TYPE:
            return True
        for pid in (a_id, b_id):
            # The operator's OWN node is exempt, and this is not a formality:
            # an operator is very often a contact in somebody else's export, so
            # their person node maps to a profile owned by that other person
            # (or, for a pre-ownership import, by nobody). Checking it would
            # reject every edge they appear on -- including all of their own
            # contacts. Verified live: "Abhimanyu Sharma" is a row in
            # local_profiles with owner_norm NULL, and without this exemption
            # his 1,020 real contacts dropped to 10 reachable neighbours.
            if pid in self_ids:
                continue
            if pid in profile_owner and profile_owner[pid] != owner_key:
                return False
        return True

    return traversable


def _org_shaped_person_ids(db: Session) -> set:
    """Person rows that are really organizations, by name collision with `organizations`.

    An org that got minted into `people` is walkable as a human intermediary,
    and no evidence check can catch it: "Justin Hotard -> Hewlett Packard
    Enterprise" IS a true, well-sourced relationship, so hop verification
    confirms it (observed: verified_status='genuine', reason "worked at
    Hewlett Packard Enterprise in a leadership role"). The claim is right; the
    TYPE is wrong. A person-to-employer affiliation is being walked as though
    the employer were a person who knows people.

    Deliberately a collision test, not a name-shape test. utils.names'
    looks_like_org_name only matches legal suffixes and returns False for
    "Hewlett Packard Enterprise", "Goldman Sachs" and "Aruba Networks" -- it
    would catch almost none of these. A row existing in BOTH tables is the
    only signal available that does not require re-deciding what a company
    name looks like.

    Used to block PASS-THROUGH only, never to hide a node. The collision says
    the two tables disagree, not which one is wrong: "Arnold Schwarzenegger"
    and "Steve Nash" are also in this set, as real people with a junk org row
    of the same name. Excluding them as intermediates costs a route that ran
    through them; deleting them would lose the person.
    """
    return {
        pid for (pid,) in db.execute(
            select(Person.id).where(Person.norm_name.in_(select(Organization.norm_name)))
        ).all()
    }


def _adjacency(db: Session, operator_name: str = ""):
    """`operator_name` gates uploaded-connection edges to their owner. Empty
    means an unidentified caller, who owns none of them -- see
    _contact_edge_gate."""
    traversable = _contact_edge_gate(db, operator_name)
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
        if not traversable(a, b, e.relationship_type):
            continue  # somebody else's uploaded connection
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
                   person_by_id=None, degree=None, excluded_intermediates=None):
    """Up to k routes; each avoids all bridge (intermediate) nodes used by the
    earlier ones, so they're genuinely different.

    `excluded_intermediates` seeds that same exclusion set before the first
    route -- nodes that may be an endpoint but must never be routed THROUGH
    (see _org_shaped_person_ids). _best_path already exempts the target from
    exclusion, so seeding here blocks pass-through without hiding anyone.
    """
    paths = []
    excluded = set(excluded_intermediates or ())
    for _ in range(k):
        hops = _best_path(adj, start, target, max_hops, excluded, person_by_id, degree)
        if hops is None:
            break
        paths.append(hops)
        for pid, _edge in hops[1:-1]:  # exclude this route's bridges next time
            excluded.add(pid)
    return paths


def _rejection_notes(db: Session, person_by_id, limit: int = 8) -> List[str]:
    """The verifier's OWN words for why it threw hops out, most recent first.

    Fed to the adjudicator because "rejected" alone is not information -- the
    reason is. "The title alone does not establish that Paul Graham and Drew
    Houston actually know each other" tells a reader the walk was chasing a
    video billing, which is what makes "you never checked X against Y" the
    obvious next move.
    """
    rows = db.execute(
        select(RelationshipEdge)
        .where(RelationshipEdge.verified_status == "rejected")
        .order_by(RelationshipEdge.verified_at.desc())
        .limit(limit)
    ).scalars().all()
    notes = []
    for e in rows:
        a = person_by_id.get(e.person_a_id)
        b = person_by_id.get(e.person_b_id)
        if a is None or b is None:
            continue
        notes.append(f"{a.canonical_name} -[{e.relationship_type}]- "
                     f"{b.canonical_name}: {(e.verified_reason or '')[:180]}")
    return notes


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


def _route_exists(db: Session, name_a: str, name_b: str, max_hops: int,
                  operator_name: str = "") -> bool:
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

    # The same pass-through rule the scoring pass applies, for the same reason
    # #53's second gate exists: a cheap check that says "connected" where the
    # pathfinder then says "no path" produces the worst pair of outcomes at
    # once -- the expensive walk is skipped BECAUSE a route is believed found,
    # and then nothing is returned. An org-shaped node may still be an
    # endpoint, so this only stops the walk expanding THROUGH one.
    org_shaped = _org_shaped_person_ids(db) - {a_id, b_id}
    # The same ownership rule the scoring pass applies. Without it here the
    # cheap check would short-circuit the paid walk on an edge the pathfinder
    # will then refuse to walk -- the exact disagreement #53's second gate and
    # the org-shaped guard both exist to prevent.
    traversable = _contact_edge_gate(db, operator_name)

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
                select(RelationshipEdge.person_b_id, RelationshipEdge.person_a_id,
                       RelationshipEdge.status, RelationshipEdge.relationship_type,
                       RelationshipEdge.signals)
                .join(Person, Person.id == RelationshipEdge.person_b_id)
                .where(RelationshipEdge.person_a_id.in_(chunk))
            ).all() + db.execute(
                select(RelationshipEdge.person_a_id, RelationshipEdge.person_b_id,
                       RelationshipEdge.status, RelationshipEdge.relationship_type,
                       RelationshipEdge.signals)
                .join(Person, Person.id == RelationshipEdge.person_a_id)
                .where(RelationshipEdge.person_b_id.in_(chunk))
            ).all()
            for far_id, near_id, status, rtype, signals in rows:
                if _untraversable(status, rtype, signals):
                    continue
                if not traversable(far_id, near_id, rtype):
                    continue
                if far_id == b_id:
                    return True
                if far_id not in visited and far_id not in org_shaped:
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


def _notable_endpoints(name_a: str, name_b: str) -> Tuple[bool, bool]:
    """(a_notable, b_notable) -- is each endpoint an independently famous person?

    Checks both the raw name and its context-stripped form (see
    _strip_trailing_context) in one batched lookup -- a person counts as
    notable if either resolves, so "Larry Ellison of Oracle" still gets caught
    even though the exact string never has its own Wikipedia page.

    (False, False) when the lookup fails, so every caller degrades to the
    unenhanced, symmetric behavior rather than to a guess.
    """
    stripped_a, stripped_b = _strip_trailing_context(name_a), _strip_trailing_context(name_b)
    try:
        notable = ORCH.notable_set(list({name_a, stripped_a, name_b, stripped_b}))
    except Exception:
        return False, False
    return (name_a in notable or stripped_a in notable,
            name_b in notable or stripped_b in notable)


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
    a_notable, b_notable = _notable_endpoints(name_a, name_b)
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

    # Alpha (targeted recheck 4c, strategy angles 4e, the top-5 narrowing at
    # step 7) belongs to a side that is walking TOWARD a famous target -- that
    # is the situation expansion._process_person's targeted-recheck phase
    # exists for. So the question is simply "is the OTHER endpoint notable",
    # and it is asked directly.
    #
    # It used to be inferred from `depth_a > depth_b`. Depth asymmetry is set
    # by _resolve_expansion_depths only when EXACTLY ONE endpoint is notable,
    # so on a famous<->famous pair both differences were zero and Alpha
    # silently switched itself off on both sides -- for the pairs most likely
    # to need it. Sanjay Ghemawat <-> Larry Page resolved to (2, 2), so the
    # top-5 narrowing and every targeted phase were unreachable, and the walk
    # fell back to the generic 15-node beam with no targeted recheck at all.
    #
    # Neither notable is unchanged: no famous target to walk toward, no Alpha.
    a_notable, b_notable = _notable_endpoints(name_a, name_b)
    enhanced_a = b_notable
    enhanced_b = a_notable
    # Mirror image, to the OTHER side: once the full-depth side's targeted
    # search has effectively concluded "the bridge is professional" (that's
    # what triggered the asymmetric depth to begin with), the famous side's
    # own limited 1-hop budget should spend itself on colleagues and board
    # seats, not family/friends silos -- see expansion.expand_graph's
    # `professional_only` and PROFESSIONAL_SILOS.
    professional_only_a = depth_a < depth_b
    professional_only_b = depth_b < depth_a

    def _make_prober(far_name: str, far_context: str, label: str):
        """Ask each ranked frontier node whether IT reaches the far endpoint.

        Expansion walks outward and hopes the two frontiers meet. For a famous
        endpoint they cannot: SHALLOW_FAMOUS_DEPTH caps that side at one hop
        precisely because their neighborhood is too large to enumerate, so the
        meeting has to be found rather than walked into. Asking directly costs
        ONE search per node against ~35 to expand one, and a famous person's
        ties are the ones most likely to be written down and findable in a
        single query.

        Observed motivation: Charlie Warren -> Donald Trump returned a five-hop
        chain (through a video title typed 'family_social', and a venture firm
        held as a person) while never once asking whether Paul Graham, Drew
        Houston or Mark Zuckerberg is documented with Trump. The last of those
        is, on a widely-reported panel.

        Two gates, both about not wasting the search:
          - the frontier is passed through the entity filter first, so probes
            are not spent on "General Manager" or "Andreessen Horowitz";
          - only when the far endpoint is notable, since the whole argument is
            that a documented person answers in one query.

        Persists nothing on its own: every edge still comes out of
        _direct_pair_search reading a fetched page.
        """
        if not config.CONNECT_PROBE_FRONTIER or not far_name.strip():
            return None
        far_notable = (_notable_endpoints(far_name, far_name)[0]
                       if config.CONNECT_PROBE_ONLY_FAMOUS else True)
        if not far_notable:
            return None

        def probe(frontier: List[str]) -> None:
            names = [n for n in frontier[:max(0, config.CONNECT_PROBE_MAX_PER_HOP)]
                     if person_norm_key(n) != person_norm_key(far_name)]
            if not names:
                return
            real = filter_entities(names, "person") if is_filtering_active() else set(names)
            for who in names:
                if who not in real:
                    continue
                if cancel_checker:
                    cancel_checker()
                # Stop the moment the pair is connected -- by an earlier probe
                # in this same loop, or by the other side's concurrent walk.
                if should_stop is not None and should_stop(db):
                    return
                if progress:
                    progress(f"  ?[{label}] does {who} reach {far_name}?")
                try:
                    _direct_pair_search(db, who, far_name, "", far_context,
                                        cancel_checker=cancel_checker)
                except Exception:  # noqa: BLE001 -- a probe must not fail the walk
                    continue

        return probe

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
                # Each side probes toward the OTHER endpoint -- the one it is
                # trying to reach, not the one it is walking out from.
                "on_frontier": _make_prober(target_name, target_context, label),
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


# Stands in for a failed extraction so the harvest's name-collection pass
# can read .edges unconditionally.
_EMPTY_EXTRACTION = ExtractionOutput(extractor="none")


class _PairPage(NamedTuple):
    """One fetched result, kept whole so every consumer reads the same bytes."""
    source: Source
    text: str
    snippet: str
    url: str


class _PageExtractions:
    """extract() memoized per (page, subject).

    Two consumers now read the same pages -- the keyword path, asking "is B
    here", and the harvest, asking "who else is here" -- and both want the
    subject=A extraction of the same page. With per-source Claude extraction
    enabled that is a whole-page model call, so computing it twice would
    literally double the cost of a page for no new information.
    """

    def __init__(self, pages: List[_PairPage]):
        self._pages = pages
        self._cache: Dict[Tuple[int, str], object] = {}

    def get(self, idx: int, subject: str):
        """The extraction for one (page, subject), or None if it failed."""
        key = (idx, person_norm_key(subject))
        if key not in self._cache:
            page = self._pages[idx]
            try:
                self._cache[key] = extract(subject, page.text, COLLEAGUE_SILO,
                                           page.snippet, page.url)
            except Exception:
                self._cache[key] = None
        return self._cache[key]


def _direct_pair_search(db: Session, name_a: str, name_b: str, context_a: str = "",
                        context_b: str = "",
                        cancel_checker: Optional[Callable[[], None]] = None,
                        progress: Optional[Callable[[str], None]] = None) -> Tuple[bool, bool]:
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

    # Fetch ONCE, here, rather than inside each variant. Both variants need the
    # same page text, and so does the harvest below -- re-fetching per consumer
    # would pay for the same HTML two or three times.
    pages: List[_PairPage] = []
    for res in results:
        if cancel_checker:
            cancel_checker()
        text = _fetch_result_text(res)
        if not text:
            continue
        pages.append(_PairPage(source=builder.save_source(db, res, query, text),
                               text=text, snippet=res.snippet, url=res.url))
    if not pages:
        return False, False

    extractions = _PageExtractions(pages)
    if relation_classifier.is_active():
        found, confident = _direct_pair_search_via_claude(
            db, name_a, name_b, pages, cancel_checker)
    else:
        found, confident = _direct_pair_search_via_keywords(
            db, name_a, name_b, pages, extractions, cancel_checker)

    # Deliberately AFTER, and deliberately not folded into (found, confident):
    # the harvest answers "who else is standing here", which is a different
    # question from "are these two directly tied". Letting it move `confident`
    # would short-circuit the paid walk on the strength of an edge that does
    # not involve the target at all -- see _route_exists's use in
    # connect_people, where a confident direct hit skips expansion entirely.
    _harvest_pair_page_entities(db, name_a, name_b, pages, extractions,
                                cancel_checker, progress)
    return found, confident


def _direct_pair_search_via_claude(db: Session, name_a: str, name_b: str,
                                   pages: List[_PairPage],
                                   cancel_checker) -> Tuple[bool, bool]:
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
    for page in pages:
        if cancel_checker:
            cancel_checker()
        source = page.source
        for window in _sentence_windows(_split_sentences(page.text)):
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


def _direct_pair_search_via_keywords(db: Session, name_a: str, name_b: str,
                                     pages: List[_PairPage],
                                     extractions: "_PageExtractions",
                                     cancel_checker) -> Tuple[bool, bool]:
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
    for idx, page in enumerate(pages):
        if cancel_checker:
            cancel_checker()
        out = extractions.get(idx, name_a)
        if out is None:
            continue
        for edge in out.edges:
            # Only the A-B edge here -- everyone ELSE this extraction found is
            # not thrown away any more, but claimed by _harvest_pair_page_entities,
            # which owns the "who else is on this page" question for both paths.
            if edge.other_kind != "person" or person_norm_key(edge.person_b) != b_norm:
                continue
            candidates.append((edge, page.source))

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


def _harvest_pair_page_entities(db: Session, name_a: str, name_b: str,
                                pages: List[_PairPage],
                                extractions: "_PageExtractions",
                                cancel_checker, progress=None) -> int:
    """Record the OTHER people named on the pair-search pages. Returns edges written.

    The pages are already fetched and already saved as Sources; until now both
    direct-pair paths read each one for a single A-B fact and discarded
    everyone else on it. The keyword path did so most visibly -- it ran a full
    extraction and dropped every edge whose counterpart was not exactly B --
    and the Claude path never even looked, since it only built windows naming
    both endpoints.

    That discard is what makes a three-hop pair report "no connection". A query
    naming both endpoints returns pages about the world they share, so the
    people on them are precisely the plausible intermediaries; Sanjay Ghemawat
    and Larry Page are two hops apart through Jeff Dean, who is named on the
    Google-engineering pages this search already downloads.

    Runs per ENDPOINT, not just the origin: an intermediary is only useful if
    it can be reached from both directions, and extracting solely around A
    would build half a bridge. The A-B edge itself is skipped -- the callers
    own that question, and writing it here too would double-persist it and
    muddy the (found, confident) verdict they return.

    Best-effort throughout. This is an opportunistic bonus on top of a search
    that has already produced its real answer, so a failure here must cost the
    caller nothing.
    """
    if not config.CONNECT_HARVEST_PAIR_PAGES or not pages:
        return 0
    endpoint_norms = {person_norm_key(name_a), person_norm_key(name_b)}
    written = 0
    # Same gate expansion applies to ITS counterparts (expansion._process_person:
    # "dropped X -- not a real person"). This path did not have it, and it is the
    # one that mints a Person per extracted name: without it the harvest is a
    # fast way to fill `people` with companies, which then get walked as human
    # intermediaries (see _org_shaped_person_ids). Batched and cached, so the
    # marginal cost is one classification per never-before-seen name.
    proposed = {edge.person_b
                for idx in range(min(len(pages), max(0, config.CONNECT_HARVEST_MAX_PAGES)))
                for subject in (name_a, name_b)
                for edge in ((extractions.get(idx, subject) or _EMPTY_EXTRACTION).edges)
                if edge.other_kind == "person"}
    real_people = filter_entities(sorted(proposed), "person") if proposed else set()
    # Highest-ranked results only: relevance falls off down the list, and every
    # page here costs one per-source extraction call PER endpoint.
    for idx, page in enumerate(pages[:max(0, config.CONNECT_HARVEST_MAX_PAGES)]):
        for subject in (name_a, name_b):
            if cancel_checker:
                cancel_checker()
            # Memoized: on the keyword path subject=A was already extracted for
            # the A-B question, so that half of this costs nothing.
            out = extractions.get(idx, subject)
            if out is None:
                continue
            person = builder.get_or_create_person(db, subject)
            if person is None:
                continue
            for edge in out.edges:
                if edge.other_kind != "person":
                    continue
                if person_norm_key(edge.person_b) in endpoint_norms:
                    continue      # the A-B edge belongs to the caller
                if (is_filtering_active() and not edge.signals.trusted
                        and edge.person_b not in real_people):
                    continue      # a company, a section heading, a job title
                try:
                    counterpart = builder.get_or_create_person(db, edge.person_b)
                    if counterpart is None:
                        continue   # node cap, or a name that isn't one
                    if builder.add_edge_from_extraction(
                            db, person, edge, 0, page.source, counterpart) is not None:
                        written += 1
                except Exception:
                    continue
    if written:
        try:
            builder.commit_with_retry(db)
        except Exception:
            return written
        if progress:
            progress(f"[direct] harvested {written} edge(s) to other people named "
                     f"on the {min(len(pages), config.CONNECT_HARVEST_MAX_PAGES)} "
                     "page(s) already fetched for the pair query")
    return written


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
        if _route_exists(check_db, name_a, name_b, max_hops, owner_name):
            route_found.set()
            return True
        return False

    if cancel_checker:
        cancel_checker()
    if progress:
        progress("\n[known] checking what's already in the graph…")
    if _route_exists(db, name_a, name_b, max_hops, owner_name):
        # Zero-cost first check: no search, no fetch, no extraction — just
        # whatever's already persisted, which now includes linkedin_1st edges
        # bridged in from uploaded contacts (see network/ingest.py) and
        # anything a prior /connect or /discover run already found (the graph
        # is shared and additive, never reset). A known first-degree
        # connection should never cost a live search to rediscover.
        route_found.set()
        if progress:
            progress("[known] already connected in the existing graph — skipping search entirely")
    def _run_origin_enrichment() -> None:
        """The origin's own network, derived into the shared graph.

        Nominally free -- no search, no fetch, no Claude, just derivations over
        already-imported rows -- which is why it used to run first, ahead of
        every paid stage. In practice materialize_contact_cliques resolves one
        Person per contact in a Python loop, so on a 2,153-contact export over
        an 84ms link it is ~20 minutes of round trips, and it ran even for an
        origin with no relationship to those contacts at all.

        Its position is now after the cheap searches (see the cascade below).
        Two consequences, both wanted: a route the searches can answer never
        pays those 20 minutes, and a route through the operator's OWN contacts
        pays one or two searches it did not strictly need before finding them.
        The second is the smaller number by orders of magnitude.
        """
        if progress:
            progress("\n[origin] initial enrichment for "
                     f"{name_a} — bridging their own network into the graph…")
        _ensure_origin_enriched(db, name_a, progress=progress,
                                owner_name=owner_name)
        # Re-check, because this may have just built the answer: a contact of
        # the origin IS the target, or sits one coworker tie away from them.
        if cancel_checker:
            cancel_checker()
        if _route_exists(db, name_a, name_b, max_hops, owner_name):
            route_found.set()
            if progress:
                progress("[origin] connected through the origin's own network — "
                         "no search needed")

    # The paid stages, as callables rather than inline blocks: the checks above
    # can decide a route already exists and skip them, and if that decision is
    # later overturned by hop verification they have to be runnable a second
    # time. See the resume block below.
    def _run_direct_pair() -> None:
        # _direct_pair_search already tries every returned result (not just
        # the first few) whenever what it's found so far is only weak, so by
        # the time it returns there's nothing more to gain from SEARCHING
        # further here. Whether it found something worth STOPPING for is a
        # different question, and `confident` is what answers it.
        found, confident = _direct_pair_search(db, name_a, name_b, context_a, context_b,
                                              cancel_checker=cancel_checker,
                                              progress=progress)
        # TWO gates, because neither implies the other. `confident` judges the
        # EVIDENCE; _route_exists judges whether the final scoring pass can
        # actually walk what was persisted, using that pass's own
        # _untraversable rule rather than a proxy for it. A confident edge is
        # Claude-typed and so normally clears _untraversable -- but it can
        # still be unwalkable for a reason the evidence knows nothing about,
        # most importantly when a concurrent prune deleted one of its endpoints
        # and left the edge dangling (_route_exists joins both ends back to
        # `people` for exactly this).
        #
        # Getting this wrong produces the worst possible pair of outcomes at
        # once: the expensive expansion is skipped BECAUSE a route is believed
        # found, and then the pathfinder, judging by the stricter rule, reports
        # "no path". Checking with the pathfinder's own rule closes that by
        # construction. It is the cheap bounded neighbor walk -- no adjacency
        # rebuild -- and runs only on this branch.
        if found and confident and _route_exists(db, name_a, name_b, max_hops, owner_name):
            route_found.set()
            if progress:
                progress("[direct] found a confident direct mention — "
                         "skipping full neighborhood expansion")
        elif found and confident:
            if progress:
                progress("[direct] confident direct mention found, but the "
                         "pathfinder can't walk it — continuing to full expansion")
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

    def _run_expansion() -> dict:
        if cancel_checker:
            cancel_checker()
        depth_a, depth_b = _resolve_expansion_depths(name_a, name_b, depth)
        if progress and (depth_a, depth_b) != (depth, depth):
            shallow_side = "A" if depth_a < depth_b else "B"
            progress(f"[reachable] side {shallow_side} is a public figure — "
                     f"capping their expansion to depth {min(depth_a, depth_b)} "
                     "(immediate circle only) instead of matching the other side")
        return _expand_both_concurrently(
            db, name_a, name_b, depth_a, depth_b, both, progress,
            context_a, context_b, on_step=on_step,
            cancel_checker=cancel_checker,
            should_stop=should_stop)

    def _run_bridge_hypothesis() -> None:
        """Ask who might stand between them, then check each name by search.

        Sits between the pair search and the expansion because that is where it
        pays: the pair search has just established the two are not documented
        TOGETHER, which is the question this stage is for. One model call plus
        up to two searches per name, against expansion's ~35 queries per node
        across two neighborhoods.

        The model names candidates; SEARCH decides. Each name is run through
        the same _direct_pair_search used above, so every edge that lands is
        read off a fetched page -- a wrong guess costs a search, never an
        invented connection. See extraction/bridge_hypothesis.
        """
        candidates = bridge_hypothesis.propose(name_a, name_b, context_a, context_b)
        if not candidates:
            if progress:
                progress("[bridge] no documented intermediary proposed — "
                         "continuing to full expansion")
            return
        if progress:
            progress("[bridge] proposed intermediaries: "
                     + "; ".join(f"{c['name']} ({c['why']})" for c in candidates))
        for cand in candidates:
            if cancel_checker:
                cancel_checker()
            who = cand["name"]
            # Both halves, because half a bridge is not one: an intermediary
            # documented with A but not with B leaves the pair as far apart as
            # before. Run unconditionally rather than short-circuiting on the
            # first half, so the second half's edge is persisted for the
            # pathfinder even when the first was only weak.
            _direct_pair_search(db, name_a, who, context_a, "",
                                cancel_checker=cancel_checker, progress=progress)
            _direct_pair_search(db, who, name_b, "", context_b,
                                cancel_checker=cancel_checker, progress=progress)
            # The pathfinder's own rule decides whether that actually built a
            # route -- not the searches' own `found`, which says only that
            # something was written.
            if _route_exists(db, name_a, name_b, max_hops, owner_name):
                route_found.set()
                if progress:
                    progress(f"[bridge] {who} connects them — "
                             "skipping full neighborhood expansion")
                return
        if progress:
            progress("[bridge] no proposed intermediary was borne out by search — "
                     "continuing to full expansion")

    # The cascade, cheapest first. Each stage runs only if the ones before it
    # did not answer, so the common cases never reach the expensive ones:
    #
    #   0. _route_exists          free       already in the graph?
    #   1. _run_direct_pair       1 search   are they documented together?
    #   2. _run_bridge_hypothesis 1 call     who stands between them?
    #                             + <=2 searches per name
    #   3. _run_origin_enrichment free*      the operator's own network
    #   4. _run_expansion         ~35 queries/node, two neighborhoods
    #
    # Origin enrichment used to be step 1 on the grounds that it is free. Its
    # implementation is not (see _run_origin_enrichment), and putting ~20
    # minutes of round trips in front of a single search meant a pair the
    # search could answer in seconds waited for work irrelevant to it. It stays
    # ahead of the expansion, which is what it actually exists to inform.
    if not route_found.is_set():
        _run_direct_pair()

    if not route_found.is_set():
        _run_bridge_hypothesis()

    if not route_found.is_set():
        _run_origin_enrichment()

    # Populated only when a fresh expansion actually ran (not when a route
    # was already known, or found via the cheap direct-pair check) -- see
    # _build_explored below and its use in both return paths.
    expand_stats: Optional[dict] = None
    if not route_found.is_set():
        expand_stats = _run_expansion()

    # Defensive, and deliberately documented as such: this drops any pending
    # read state before the scoring pass, so that pass starts from the graph as
    # it stands after the expansion's own _prune_invalid_nodes has deleted junk
    # nodes on other Sessions.
    #
    # It is here because ObjectDeletedError ("Instance '<Person ...>' has been
    # deleted, or its row is otherwise not present") was seen live at exactly
    # this point -- after the stop condition was met, so a route that HAD been
    # found was returned to the caller as a hard error.
    #
    # What this comment will NOT claim is the mechanism, because the obvious
    # two were tested and neither reproduces:
    #   * "the Session is reading a stale pre-cleanup snapshot" is false. Both
    #     backends run READ COMMITTED, so this Session sees another Session's
    #     committed insert/delete on its very next statement, with no rollback
    #     needed. Verified directly.
    #   * "rollback expires the identity map, then a deleted row raises on
    #     refresh" also did not reproduce: a clean instance survived expire +
    #     delete + attribute access without raising.
    # So the real trigger is still unidentified, and the honest reading of this
    # call is a cheap guard, not a root-cause fix. If the error resurfaces,
    # capture the traceback rather than trusting this to have handled it --
    # tests/test_stale_session_after_expansion.py pins only that the final read
    # is served from a clean Session, which is all that is actually verified.
    #
    # expunge_all WITHOUT a rollback, deliberately. Pairing the two (as the
    # first version of this did) expires every instance and then detaches it,
    # so any object a caller still holds raises DetachedInstanceError on its
    # next attribute access -- trading one error for another, at a point whose
    # whole purpose is to stop a successful search from surfacing as a failure.
    # The rollback was also the half with nothing to show for it: at READ
    # COMMITTED it changes no visibility (see
    # test_a_session_is_not_reading_a_stale_snapshot). expunge_all alone gives
    # the property actually wanted here -- the scoring pass below builds its
    # objects fresh instead of reusing whatever the expansion left mapped.
    db.expunge_all()

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

    def _routes_now():
        """(surviving routes, how many candidates there were, person_by_id, …).

        Recomputed rather than cached because the resume below can persist new
        edges, and because verification MUTATES the graph -- a rejected hop is
        written back as status='rejected', which _path_worthy then treats as
        untraversable. The second call therefore cannot re-propose a route the
        first one just disproved, which is what bounds the resume.
        """
        adjacency, by_id, src, deg = _adjacency(db, owner_name)
        if cancel_checker:
            cancel_checker()
        found = _diverse_paths(adjacency, a.id, b.id, max_hops,
                               config.CONNECT_MAX_PATHS, by_id, deg,
                               excluded_intermediates=_org_shaped_person_ids(db))
        if cancel_checker:
            cancel_checker()
        candidate_count = len(found)
        was_verified = config.CLAUDE_VERIFY_HOPS and hop_verify.claude_available()
        if was_verified:
            found = _verified_routes(db, found, by_id, cancel_checker)
        if cancel_checker:
            cancel_checker()
        return found, candidate_count, was_verified, adjacency, by_id, src, deg

    routes, n_candidates, verified, adj, person_by_id, src_by_id, degree = _routes_now()

    # The short-circuit is not self-correcting, and that is what this fixes.
    # Every check above ("already in the graph", "connected through the
    # origin's own network") skips the paid walk on the strength of edges
    # nothing has inspected yet. When verification then rejects all of them,
    # the skip was made on a false premise -- and until now the walk simply
    # reported "no connection", having never searched at all.
    #
    # Observed: Sanjay Ghemawat -> Larry Page. A 0.39-confidence heuristic edge
    # (Ghemawat -> Eric Schmidt, off a PDF whose evidence sentence names
    # Schmidt, Page and Brin but not Ghemawat) made a two-hop route appear to
    # exist. The walk was skipped, verification correctly rejected the edge,
    # and the caller got "no connection" for $0.0007 and zero searches.
    #
    # Runs at most once, and only when nothing was searched. If expansion had
    # already run, its edges are the ones being rejected and repeating it would
    # rediscover the same pages. Bounded further by the fact that verification
    # persists its rejections, so the disproved route is gone from the graph
    # before the retry re-reads it.
    if not routes and n_candidates and verified and expand_stats is None:
        if progress:
            progress(f"\n[verify] all {n_candidates} candidate route(s) were "
                     "rejected on their own evidence — the earlier 'already "
                     "connected' shortcut was wrong. Resuming the search it skipped…")
        route_found.clear()
        _run_direct_pair()
        if not route_found.is_set():
            expand_stats = _run_expansion()
        # The same guard the first scoring read gets, for the same reason: this
        # resume ran a fresh expansion, whose _prune_invalid_nodes deletes junk
        # nodes on other Sessions, and _routes_now() below is a scoring read. The
        # expunge_all above happened BEFORE that expansion, so it does not cover
        # it. See its comment for what is and isn't claimed.
        db.expunge_all()
        routes, n_after, verified, adj, person_by_id, src_by_id, degree = _routes_now()
        n_candidates = max(n_candidates, n_after)

    # LAST STOP BEFORE "no connection". Everything above has run out, but by now
    # this function is holding exactly the context needed to judge whether it
    # stopped too early: who was explored on each side, what routes were
    # proposed, and the verifier's own words for rejecting them. Nothing ever
    # looked at that before answering. Charlie Warren -> Donald Trump is the
    # case -- the walk quit while Sam Altman sat unexpanded in the graph with 34
    # edges, and neither "both are Y Combinator" nor "Altman has met Trump
    # repeatedly" was ever a query.
    #
    # The model's two moves are priced differently, and that asymmetry is what
    # makes this safe: a probe is one search and may name anyone, because the
    # search decides; an expansion is ~35 searches and may only name nodes this
    # walk already ranked, by index into the shortlist handed to it. So it
    # steers spending it cannot invent, and every edge still arrives through the
    # ordinary search path. See extraction/route_adjudicator.
    adjudication: Optional[dict] = None
    if not routes and route_adjudicator.is_active():
        explored = _build_explored(expand_stats, name_a, name_b) or {}

        def _side(side: str, endpoint_id: str) -> List[str]:
            """That side's people: whatever expansion explored, plus the
            endpoint's own neighbours (which is all there is when the walk was
            short-circuited before expanding). Ranked by degree WITHIN this
            side, never against the graph at large -- ranking globally handed
            the model the 30 most-connected nodes in the database, which is
            search history plus junk, and produced pairings of Charlie Warren
            with Lip-Bu Tan and Arnold Schwarzenegger."""
            names = {n for hop in (explored.get(side, {}).get("by_hop", {}) or {}).values()
                     for n in hop}
            ids = {pid for pid, p in person_by_id.items() if p.canonical_name in names}
            ids |= {nbr for nbr, _e in adj.get(endpoint_id, [])}
            ids -= {a.id, b.id}
            ranked = [person_by_id[pid].canonical_name
                      for pid in sorted(ids, key=lambda p: -degree.get(p, 0))
                      if pid in person_by_id][:40]
            if is_filtering_active() and ranked:
                real = filter_entities(ranked, "person")
                ranked = [n for n in ranked if n in real]
            return ranked[:25]

        left = _side("a", a.id)
        # name_b leads the right-hand list deliberately: "does one of these
        # people know the TARGET" is the question that closes the gap in one
        # hop, and it is only askable if the target is on the list.
        right = [b.canonical_name] + [n for n in _side("b", b.id)
                                      if n != b.canonical_name]
        if left and right:
            if progress:
                progress("\n[adjudicate] no route survived — asking whether any of "
                         f"{len(left)} people around {a.canonical_name} could know "
                         f"{b.canonical_name} or the {len(right) - 1} people near them…")
            adjudication = route_adjudicator.decide(
                name_a, name_b, context_a, context_b,
                left=left, right=right,
                rejected=_rejection_notes(db, person_by_id))

    if adjudication and adjudication["action"] != "none":
        if progress:
            progress(f"[adjudicate] {adjudication['why']}")
        for pair in adjudication["pairs"]:
            if cancel_checker:
                cancel_checker()
            if progress:
                progress(f"  ?[adjudicate] {pair['a']} — {pair['b']}")
            try:
                _direct_pair_search(db, pair["a"], pair["b"],
                                    cancel_checker=cancel_checker)
            except Exception:  # noqa: BLE001 -- a last-resort pass must not raise
                continue
        for who in adjudication["expand"]:
            if cancel_checker:
                cancel_checker()
            if progress:
                progress(f"  ⟳[adjudicate] expanding {who}")
            try:
                expand_graph(db, who, 1, progress=progress,
                             prefer_reachable=False,
                             cancel_checker=cancel_checker)
            except Exception:  # noqa: BLE001
                continue
        db.expunge_all()
        routes, n_after, verified, adj, person_by_id, src_by_id, degree = _routes_now()
        n_candidates = max(n_candidates, n_after)

    if not routes:
        # A distinct reason when candidates existed but none survived
        # verification -- "try a higher depth" would be actively misleading
        # there, since more expansion won't fix a hop that failed on its
        # own evidence.
        reason = (
            f"{n_candidates} candidate route(s) found within "
            f"{max_hops} hops, but none passed hop verification — the "
            "evidence didn't hold up on inspection."
            if n_candidates and verified else
            f"no path within {max_hops} hops — their graphs don't overlap "
            f"at depth {depth}. Try a higher depth."
        )
        out = {
            "connected": False,
            "person_a": a.canonical_name, "person_b": b.canonical_name,
            "reason": reason,
            "explored": _build_explored(expand_stats, name_a, name_b),
        }
        if adjudication:
            # What the last pass concluded and what it spent, so the answer
            # reports an exhausted search rather than a bare refusal.
            out["adjudication"] = adjudication
            if adjudication["action"] != "none":
                out["reason"] = (
                    f"{reason} Then followed up on: {adjudication['why']} "
                    f"({len(adjudication['pairs'])} pair search(es), "
                    f"{len(adjudication['expand'])} expansion(s)) — still nothing.")
        return out

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
                # The handle an operator needs to say "this hop is wrong"
                # (POST /edges/reject). Without it a caller looking at a bogus
                # hop can only describe it by endpoint names, which is ambiguous
                # the moment two people have more than one edge between them.
                node["edge_id"] = edge.id
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
