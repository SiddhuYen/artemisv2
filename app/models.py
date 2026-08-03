"""ORM models for the relationship graph.

Enum-like columns are stored as TEXT for SQLite friendliness; the allowed
values are documented by the constant tuples below and validated in the
schema / builder layers.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy.types import JSON

from .db import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --- controlled vocabularies ----------------------------------------------
ORG_TYPES = ("company", "nonprofit", "school", "government", "event", "unknown")

RELATIONSHIP_TYPES = (
    "coworker", "cofounder", "board_member", "advisor", "investor",
    "employee", "speaker", "author", "student", "faculty",
    "family_social", "interview", "coauthor", "appointee",
    "podcast_guest",  # host<->guest, asserted by a podcast RSS feed episode
    "linkedin_1st",   # verified direct connection, supplied by a LinkedIn export
    "unknown",
)

# Person<->person types where (a, b) asserts exactly what (b, a) asserts, so
# the two orientations are ONE connection and must never become two rows.
#
# This is a dedup vocabulary, not a graph-traversal one -- connect.py already
# walks every edge in both directions (see _adjacency). It exists because the
# stored orientation is an accident of WHICH endpoint was expanded first: a
# walk that expands A extracts "A coworker B" from a page, and a later walk
# that expands B extracts "B coworker A" from the SAME page. Both are the same
# fact. While `processed` froze a node after one expansion that second walk
# almost never happened; now that coverage-based reuse re-searches a node
# whenever a new question is asked of it (see graph/expansion.py), it is the
# normal case, and without symmetric dedup every re-enrichment would mirror
# the existing graph into a duplicate half.
#
# Directional types are deliberately absent: "A interview B" (A interviewed B)
# and "A investor B" say something the reverse does not, so collapsing their
# orientations would destroy the fact rather than dedupe it.
SYMMETRIC_RELATIONSHIP_TYPES = frozenset({
    "coworker", "cofounder", "coauthor", "family_social",
    "podcast_guest", "linkedin_1st",
})

# Organization<->organization facts (e.g. two firms co-invested in the same
# funding round). Deliberately NEVER used to bridge two PEOPLE by chaining
# person->org->org->person -- see Organization.meta["co_investments"] and
# graph.builder.record_coinvestment. Kept as a separate vocabulary so it can
# never be confused with a person-level RELATIONSHIP_TYPES value.
ORG_RELATIONSHIP_TYPES = ("co_investor",)

EDGE_STATUSES = ("weak", "candidate", "strong", "raw", "rejected", "accepted")

PROVIDERS = ("duckduckgo", "wikipedia", "scrape")


class Person(Base):
    __tablename__ = "people"

    id = Column(String, primary_key=True, default=_uuid)
    canonical_name = Column(String, nullable=False)
    norm_name = Column(String, index=True, unique=True, nullable=False)
    # Wikidata QID (e.g. "Q265852") when the person resolves to a real Wikidata
    # human entity — an AUTHORITATIVE identity anchor. Two different notable
    # people with the same name have distinct QIDs, so they never merge into a
    # single false-bridge node (the core homonym-disambiguation signal).
    wikidata_qid = Column(String, index=True, nullable=True)
    # 1 once this person's OWN searches have been run (they've been "expanded").
    # Lets a later/deeper run REUSE their persisted neighbors instead of
    # re-searching, and continue outward from the frontier (incremental deepening
    # across runs and across teammates in the shared map).
    processed = Column(Integer, default=0, nullable=False)
    aliases = Column(JSON, default=list)
    meta = Column("metadata", JSON, default=dict)
    created_at = Column(String, default=lambda: _now().isoformat())


class Organization(Base):
    __tablename__ = "organizations"

    id = Column(String, primary_key=True, default=_uuid)
    name = Column(String, nullable=False)
    norm_name = Column(String, index=True, unique=True, nullable=False)
    type = Column(String, default="unknown")  # one of ORG_TYPES
    meta = Column("metadata", JSON, default=dict)
    created_at = Column(String, default=lambda: _now().isoformat())


class Source(Base):
    __tablename__ = "sources"

    id = Column(String, primary_key=True, default=_uuid)
    url = Column(String, index=True)
    title = Column(String)
    snippet = Column(Text)
    full_text = Column(Text, nullable=True)
    provider = Column(String)  # one of PROVIDERS
    query_used = Column(String)
    created_at = Column(String, default=lambda: _now().isoformat())


class RelationshipEdge(Base):
    __tablename__ = "relationship_edges"

    id = Column(String, primary_key=True, default=_uuid)
    person_a_id = Column(String, ForeignKey("people.id"), nullable=False, index=True)
    person_b_id = Column(String, ForeignKey("people.id"), nullable=True, index=True)
    organization_id = Column(String, ForeignKey("organizations.id"), nullable=True, index=True)
    relationship_type = Column(String, default="unknown")  # one of RELATIONSHIP_TYPES
    method = Column(Text)            # how the relationship was inferred
    evidence_snippet = Column(Text)
    source_id = Column(String, ForeignKey("sources.id"), nullable=True)
    confidence_base = Column(Float, default=0.0)
    confidence_raw = Column(Float, default=0.0)  # the adjusted/final confidence
    signals = Column(JSON, default=dict)         # EdgeSignals dump
    depth = Column(Integer, default=0)
    status = Column(String, default="weak")  # one of EDGE_STATUSES (tier)
    created_at = Column(String, default=lambda: _now().isoformat())

    # Path-assembly-time hop verification (see graph.hop_verify): does THIS
    # edge's own evidence actually support the claimed relationship, judged
    # independently of whatever path it's being walked in (the cached verdict
    # is reused across every path this edge appears in, so it can't depend on
    # a specific path's surrounding context -- see hop_verify's docstring).
    # One of "genuine" / "rejected", or None if never checked.
    verified_status = Column(String, nullable=True)
    verified_at = Column(String, nullable=True)
    verified_reason = Column(Text, nullable=True)

    person_a = relationship("Person", foreign_keys=[person_a_id])
    person_b = relationship("Person", foreign_keys=[person_b_id])
    organization = relationship("Organization", foreign_keys=[organization_id])
    source = relationship("Source", foreign_keys=[source_id])


# ===========================================================================
# Local network (uploaded CSV) — stage: graph matching (no Claude yet)
# ===========================================================================

MATCH_TYPES = ("exact_name", "name_company", "name_school", "org_overlap", "fuzzy_name")
PATH_STATUSES = ("unverified", "verified", "rejected")  # only 'unverified' is set here


class LocalProfile(Base):
    __tablename__ = "local_profiles"

    id = Column(String, primary_key=True, default=_uuid)
    canonical_name = Column(String, nullable=False)
    norm_name = Column(String, index=True)  # person_norm_key, for matching
    aliases = Column(JSON, default=list)
    email = Column(String, nullable=True, index=True)
    linkedin_url = Column(String, nullable=True)
    companies = Column(JSON, default=list)
    titles = Column(JSON, default=list)
    schools = Column(JSON, default=list)
    locations = Column(JSON, default=list)
    notes = Column(Text, nullable=True)
    raw_row = Column(JSON, default=dict)
    connected_on = Column(String, nullable=True)  # from the CSV's "Connected On" column, if present
    created_at = Column(String, default=lambda: _now().isoformat())


class LocalEdge(Base):
    __tablename__ = "local_edges"

    id = Column(String, primary_key=True, default=_uuid)
    # from_profile_id NULL == "You" (the network owner)
    from_profile_id = Column(String, ForeignKey("local_profiles.id"), nullable=True)
    to_profile_id = Column(String, ForeignKey("local_profiles.id"), nullable=False)
    edge_type = Column(String, default="uploaded_network")
    confidence = Column(Float, default=1.0)
    source = Column(String, default="uploaded_csv")
    created_at = Column(String, default=lambda: _now().isoformat())


class GraphMatch(Base):
    __tablename__ = "graph_matches"

    id = Column(String, primary_key=True, default=_uuid)
    local_profile_id = Column(String, ForeignKey("local_profiles.id"), nullable=False, index=True)
    public_person_id = Column(String, ForeignKey("people.id"), nullable=True, index=True)
    public_org_id = Column(String, ForeignKey("organizations.id"), nullable=True, index=True)
    match_type = Column(String)  # one of MATCH_TYPES
    confidence = Column(Float, default=0.0)
    explanation = Column(Text)
    created_at = Column(String, default=lambda: _now().isoformat())


class CandidatePath(Base):
    __tablename__ = "candidate_paths"

    id = Column(String, primary_key=True, default=_uuid)
    target_person_id = Column(String, ForeignKey("people.id"), nullable=False, index=True)
    local_profile_id = Column(String, ForeignKey("local_profiles.id"), nullable=False)
    public_person_id = Column(String, ForeignKey("people.id"), nullable=True)
    path_json = Column(JSON, default=dict)
    score = Column(Float, default=0.0)
    status = Column(String, default="unverified")  # NEVER 'accepted' at this stage
    created_at = Column(String, default=lambda: _now().isoformat())


# ===========================================================================
# Initial enrichment — building the operator's own 2-layer network.
#
# L1 (your contacts) is asserted for free by the export. L2 (who they know) is
# the only layer that costs searches, and at ~35 queries/person a 1,000-contact
# export is tens of thousands of provider calls and hours of wall clock. So the
# work is ranked, budgeted, and executed in waves.
#
# This lives in the DB rather than in main.py's in-memory _JOBS because a run
# outlives the process: it takes hours, and a deploy or restart mid-run must
# resume rather than start over. Person.processed already records "this node
# was expanded", but it cannot express "probed, no web footprint, don't retry"
# or scope progress to a particular run — hence a real task table.
# ===========================================================================

class OwnerProfile(Base):
    """Who the operator is — persisted server-side rather than in localStorage.

    Everything before this took `owner_name` as a per-request parameter that
    the browser had to remember to send, which is why contacts imported before
    the frontend was wired up have no graph edges at all (see
    ingest.backfill_graph_edges). It also meant the operator's OWN employer and
    school were unavailable to ranking, so the shared-affiliation boost in
    ranking.score_contacts could never fire in practice.

    Scoped by `owner_id` (the X-Graph-Id header, same as Boards) rather than
    assumed singleton, so a second operator on one deployment does not silently
    overwrite the first. The discovery graph itself is still shared.

    Note this does NOT help disambiguate the operator's own Person node: the
    homonym guard only engages when a Wikidata QID is supplied, and an ordinary
    operator has none.
    """
    __tablename__ = "owner_profiles"

    id = Column(String, primary_key=True, default=_uuid)
    owner_id = Column(String, index=True, unique=True, nullable=False)
    name = Column(String, nullable=False)
    company = Column(String, nullable=True)
    title = Column(String, nullable=True)
    school = Column(String, nullable=True)
    linkedin_url = Column(String, nullable=True)
    email = Column(String, nullable=True)
    created_at = Column(String, default=lambda: _now().isoformat())
    updated_at = Column(String, default=lambda: _now().isoformat())


ENRICHMENT_RUN_STATES = ("planned", "running", "paused", "done", "cancelled", "failed")

ENRICHMENT_TASK_STATES = (
    "pending",       # ranked, not yet attempted
    "enriching",     # expand_graph in flight
    "done",          # expanded (or already processed by an earlier run/teammate)
    "probed_empty",  # cheap probe found no web footprint — full sweep skipped
    "skipped",       # never eligible (see EnrichmentTask.skip_reason)
    "failed",        # attempted and errored; retried by _next_task up to
                     # config.ENRICH_MAX_ATTEMPTS, then terminal
)

# Why a contact was excluded from enrichment before spending anything on it.
SKIP_REASONS = (
    "no_context",    # bare name, no company/title/school — see ranking.py
    "generic_only",  # only a generic employer ("Self-Employed") to go on
)


class EnrichmentRun(Base):
    __tablename__ = "enrichment_runs"

    id = Column(String, primary_key=True, default=_uuid)
    owner_name = Column(String, nullable=False)
    # The operator's own affiliations, used only to boost contacts who share
    # them (see ranking.score_contacts). Free-text, optional.
    owner_company = Column(String, nullable=True)
    owner_school = Column(String, nullable=True)
    state = Column(String, default="planned")  # one of ENRICHMENT_RUN_STATES
    depth = Column(Integer, default=1)         # hops to expand each contact
    budget_s = Column(Float, default=0.0)      # 0 = unbounded
    counters = Column(JSON, default=dict)      # per-state task tallies, cached
    error = Column(Text, nullable=True)
    started_at = Column(String, nullable=True)
    finished_at = Column(String, nullable=True)
    created_at = Column(String, default=lambda: _now().isoformat())


# A task is usually one CONTACT to expand. Wave 2 adds ORG tasks: expanding an
# employer several contacts share, once, reaches that organization's public
# neighbourhood for one contact's worth of queries. Same table so org sweeps
# inherit the ranking, resume, budget and cancel machinery unchanged.
TASK_KINDS = ("contact", "org")


class EnrichmentTask(Base):
    __tablename__ = "enrichment_tasks"

    id = Column(String, primary_key=True, default=_uuid)
    run_id = Column(String, ForeignKey("enrichment_runs.id"), nullable=False, index=True)
    kind = Column(String, default="contact", nullable=False)  # one of TASK_KINDS
    local_profile_id = Column(String, ForeignKey("local_profiles.id"), nullable=True)
    # Resolved lazily: the Person may not exist yet when the run is planned
    # (contacts imported without owner_name have no graph node). norm_name is
    # the stable key; person_id is a convenience filled in at execution time.
    person_id = Column(String, ForeignKey("people.id"), nullable=True)
    display_name = Column(String, nullable=False)
    norm_name = Column(String, index=True, nullable=False)
    # Disambiguating context passed to expand_graph as seed_context — usually
    # the employer. Enriching a bare "John Smith" with no context is worse than
    # skipping it: it grafts some unrelated notable namesake's network onto the
    # operator's graph.
    context = Column(String, nullable=True)
    score = Column(Float, default=0.0)
    # Silo key -> weight, computed from this contact's export row when the run
    # is planned (network/silo_weights.initial_weights). Decides which silos
    # run for them and how many queries each gets. Stored rather than recomputed
    # so a plan is inspectable, reproducible, and later tunable.
    silo_weights = Column(JSON, default=dict)
    rank = Column(Integer, default=0, index=True)  # 1-based execution order
    state = Column(String, default="pending", index=True)  # ENRICHMENT_TASK_STATES
    skip_reason = Column(String, nullable=True)    # one of SKIP_REASONS
    attempts = Column(Integer, default=0)
    last_error = Column(Text, nullable=True)
    updated_at = Column(String, default=lambda: _now().isoformat())
    created_at = Column(String, default=lambda: _now().isoformat())


# ===========================================================================
# Boards — a user's manually-built canvas workspace (UI-only concept; never
# mutates the canonical discovery data above). Scoped by owner_id, the same
# per-browser id the frontend already sends as X-Graph-Id. Each board can hold
# several independent canvas Pages (tabs), each with its own node/edge layout.
# ===========================================================================

BOARD_STATUSES = ("active", "archived")


class Board(Base):
    __tablename__ = "boards"

    id = Column(String, primary_key=True, default=_uuid)
    owner_id = Column(String, index=True, nullable=False)
    name = Column(String, nullable=False)
    target_name = Column(String, nullable=True)
    target_org = Column(String, nullable=True)
    status = Column(String, default="active")  # one of BOARD_STATUSES
    created_at = Column(String, default=lambda: _now().isoformat())


class BoardPage(Base):
    __tablename__ = "board_pages"

    id = Column(String, primary_key=True, default=_uuid)
    board_id = Column(String, ForeignKey("boards.id"), nullable=False, index=True)
    name = Column(String, default="Page 1")
    position = Column(Integer, default=0)   # tab ordering
    elements = Column(JSON, default=dict)   # {"nodes": [...], "edges": [...], "centerId": ...}
    created_at = Column(String, default=lambda: _now().isoformat())
