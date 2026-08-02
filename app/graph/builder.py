"""Graph persistence: dedup-aware upserts for people, orgs, sources, edges.

Dedup keys:
  - people -> person_norm_key (normalised, middle-initials stripped); surface
              variants are auto-stored as aliases.
  - orgs   -> org_norm_key (normalised, trailing legal/structural suffix stripped).
  - sources-> url
  - edges  -> (person_a, counterpart, relationship_type, source_url) -> discard dup.

Every edge carries source URL + evidence snippet + base & adjusted confidence +
signals + status tier. No relationship is ever auto-set to 'accepted'
(verification is deferred).
"""
from __future__ import annotations

import random
import time
from typing import Callable, Optional, TypeVar

from sqlalchemy import func, or_, select
from sqlalchemy.exc import IntegrityError, OperationalError, PendingRollbackError
from sqlalchemy.orm import Session

from .. import config
from ..extraction import tier
from ..extraction.schemas import ExtractedEdge
from ..models import (
    CandidatePath,
    GraphMatch,
    Organization,
    Person,
    RelationshipEdge,
    Source,
)
from ..providers.base import SearchResult
from ..utils.names import (
    is_noise_name,
    looks_like_person_name,
    name_variants,
    org_norm_key,
    person_norm_key,
)
from . import disambiguate


def _strip_nul(s: Optional[str]) -> Optional[str]:
    """Drop embedded NUL bytes (0x00) from scrape/extraction-derived text.

    SQLite stores them without complaint; Postgres's text type rejects them
    outright, raising ValueError at flush time. A NUL can enter any string
    that traces back to a scraped page or an LLM-extracted name/snippet, not
    just Source.full_text — so every write site that persists such a string
    needs this, not just the search-result save path.
    """
    return s.replace("\x00", "") if s else s


class SharedGraphResetError(RuntimeError):
    """Raised when something tries to wipe a graph that isn't private."""


def graph_is_shared() -> bool:
    """True when the graph lives in a server database rather than a private
    local SQLite file.

    The whole codebase was written when the graph was a file on one laptop,
    where wiping it costs its owner a rebuild and nobody else anything. On a
    team's shared Postgres the same call destroys everyone's work at once, so
    the destructive paths below ask this first.
    """
    return config.IS_POSTGRES


def reset_public_graph(db: Session, force: bool = False) -> None:
    """Clear the PUBLIC graph + derived matches/paths, preserving the uploaded
    local network (local_profiles / local_edges). Children first for FK safety.

    Refuses to run against a SHARED graph unless `force` is passed. This is a
    deliberate chokepoint: every destructive path in the app funnels through
    here, and two of them are easy to trigger without meaning to --
    `python -m app.cli "Some Name"` resets by default (--keep is opt-IN), and
    org_discovery calls this as routine scratch cleanup. Against a private
    SQLite file both are harmless; against a team database either one silently
    deletes every collaborator's graph, including whatever the deployment
    accumulated. Callers that genuinely mean it pass force=True.
    """
    if graph_is_shared() and not force:
        raise SharedGraphResetError(
            "refusing to wipe a shared graph database — this would delete "
            "every collaborator's data, not just yours. Accumulate instead "
            "(the CLI's --keep), or pass force=True if you really mean it."
        )
    db.query(CandidatePath).delete()
    db.query(GraphMatch).delete()
    db.query(RelationshipEdge).delete()
    db.query(Source).delete()
    db.query(Organization).delete()
    db.query(Person).delete()
    db.commit()


# --- node-count cap --------------------------------------------------------
def node_count(db: Session) -> int:
    # Cap on PEOPLE only — they're the expandable, path-relevant nodes.
    # Orgs are cheap leaf metadata and shouldn't starve the person budget.
    return db.scalar(select(func.count()).select_from(Person)) or 0


def at_node_cap(db: Session) -> bool:
    return node_count(db) >= config.MAX_TOTAL_NODES


# --- entity upserts --------------------------------------------------------
def get_or_create_person(db: Session, name: str, qid: Optional[str] = None,
                         allow_create: bool = True,
                         identity_text: Optional[str] = None) -> Optional[Person]:
    """Resolve a person node, disambiguating homonyms by Wikidata QID.

    Identity rules:
      - qid given: same-QID node wins (authoritative merge across name variants);
        a name-match with NO qid ADOPTS this qid -- UNLESS the node's own
        already-accumulated evidence conflicts with `identity_text` (see
        _homonym_conflict), in which case it's treated like case 3 below; a
        name-match with a DIFFERENT qid is a distinct person (a homonym) ->
        a separate, QID-suffixed node.
      - no qid: fall back to the normalized-name key (today's behavior).

    `identity_text` is a short description of the identity being merged onto
    an existing same-named node -- the candidate side of the homonym check.
    With a `qid`, it's the QID's own identity (e.g. a Wikipedia summary).
    WITHOUT a qid -- counterpart resolution during edge persistence, which
    never has a QID to key off -- it's whatever the caller knows about this
    SPECIFIC mention (typically the edge's own evidence sentence); a rejected
    merge there gets a domain-keyed node instead of a QID-keyed one, since
    there's no QID to disambiguate by (see the no-QID branch below). Omitting
    it entirely reverts to the old unconditional-adopt behavior -- there is
    nothing to check the name against.

    This split exists because of a live miss: a real "Donald Trump" (US
    president) node in the graph silently absorbed an unrelated academic
    coauthor sharing the same bare name, because counterpart resolution
    never passed anything to compare against at all -- the QID path's guard
    only ever protected the SUBJECT side of identity resolution, never an
    arbitrary discovered counterpart.
    """
    name = _strip_nul(name)
    norm = person_norm_key(name)
    if not norm:
        return None

    if qid:
        # 1) authoritative: an existing node already carrying this QID
        by_qid = db.execute(
            select(Person).where(Person.wikidata_qid == qid)
        ).scalar_one_or_none()
        if by_qid:
            _merge_aliases(by_qid, name)
            return by_qid
        # 2) a name-match with no QID yet -> normally it's this same person;
        #    adopt the QID -- unless the homonym guard says otherwise.
        by_name = db.execute(
            select(Person).where(Person.norm_name == norm)
        ).scalar_one_or_none()
        adopt = by_name is not None and not by_name.wikidata_qid
        if adopt and _homonym_conflict(db, by_name, identity_text):
            adopt = False
        if adopt:
            by_name.wikidata_qid = qid
            _merge_aliases(by_name, name)
            return by_name
        # 3) name-match exists but with a DIFFERENT QID, or the homonym guard
        #    just rejected adopting this QID onto it -> genuine homonym.
        #    Give the newcomer a QID-disambiguated key so they don't collide.
        if not allow_create:
            return None
        node_norm = norm if by_name is None else f"{norm}#{qid}"
        return _new_person_or_existing(db, name, node_norm, qid)

    # no QID: plain name-key dedup, homonym-gated when the caller gave an
    # identity_text to check (see this function's docstring). A rejected
    # merge gets a DOMAIN-keyed node, not a hash of the raw text -- so every
    # future sighting of the same KIND of conflicting mention (e.g. another
    # academic coauthor named "Donald Trump") converges onto the SAME
    # secondary node instead of fragmenting into a new one per differently-
    # worded sentence. Silent (no separation) whenever domains_of() can't
    # anchor the candidate side, same conservative default _homonym_conflict
    # already has -- an unrecognizable signal is not evidence of a conflict.
    existing = db.execute(
        select(Person).where(Person.norm_name == norm)
    ).scalar_one_or_none()
    node_norm = norm
    if existing is not None and _homonym_conflict(db, existing, identity_text):
        # domain_conflict (called inside _homonym_conflict) only ever returns
        # True when BOTH sides' domains_of() are non-empty -- so `domains`
        # here is guaranteed non-empty too; the guard below is a defensive
        # invariant check, not a reachable branch.
        domains = sorted(disambiguate.domains_of(identity_text or ""))
        if domains:
            node_norm = f"{norm}#{'+'.join(domains)}"
            distinct = db.execute(
                select(Person).where(Person.norm_name == node_norm)
            ).scalar_one_or_none()
            existing = distinct  # None if this specific conflicting identity hasn't been seen before
    if existing:
        _merge_aliases(existing, name)
        return existing
    if not allow_create:
        return None
    return _new_person_or_existing(db, name, node_norm, None)


def _existing_evidence_signal(db: Session, person: Person) -> str:
    """A few of `person`'s own already-persisted edge-evidence snippets --
    used as an independent professional-domain signal by _homonym_conflict.

    Deliberately NOT a fresh search for `person`'s name: a search run to
    justify adopting `identity_text` could be the SAME search that produced
    `identity_text` in the first place, which would let a single fame-
    dominated lookup confirm itself. These snippets instead come from
    whatever OTHER subject's search first discovered this person as a
    counterpart, before this identity was ever proposed for it.
    """
    rows = db.execute(
        select(RelationshipEdge.evidence_snippet)
        .where(or_(RelationshipEdge.person_a_id == person.id,
                   RelationshipEdge.person_b_id == person.id))
        .order_by(RelationshipEdge.confidence_raw.desc(), RelationshipEdge.id)
        .limit(config.IDENTITY_SIGNAL_MAX_SNIPPETS)
    ).scalars()
    return " ".join(s for s in rows if s)


def _homonym_conflict(db: Session, existing: Person,
                      identity_text: Optional[str]) -> bool:
    """True when `existing`'s own accumulated evidence clearly anchors in a
    different professional world than `identity_text` (a description of the
    identity about to be adopted onto it) -- see graph.disambiguate.

    Silent (False) whenever there's nothing to compare: no identity_text was
    given, the feature is disabled, or `existing` has no prior evidence yet
    (a brand-new node adopts freely, same as before this guard existed).
    Records a `homonym_rejected` note in `existing.meta` on a real conflict --
    advisory only, not a permanent block: it doesn't stop `existing` from
    continuing to accumulate its own edges, and a later direct assignment of
    `wikidata_qid` (bypassing this path entirely, same as case 1 above) can
    still confirm the identity if a human decides the rejection was wrong.
    """
    if not identity_text or not config.IDENTITY_VERIFY_ENABLED:
        return False
    signal = _existing_evidence_signal(db, existing)
    if not signal:
        return False
    if not disambiguate.domain_conflict(signal, identity_text):
        return False
    meta = dict(existing.meta or {})
    meta["homonym_rejected"] = {"identity_text": identity_text[:300]}
    # flush_in_savepoint, not a bare `existing.meta = meta`: leaving this
    # mutation pending is what made the NEXT savepoint entry (usually
    # _new_person_or_existing's, a few lines later in both callers) flush it
    # pre-SAVEPOINT, where a lock is unrecoverable and costs the whole node.
    # See flush_in_savepoint's docstring for the full failure shape.
    flush_in_savepoint(db, lambda: setattr(existing, "meta", meta))
    return True


def _new_person(db: Session, name: str, norm: str, qid: Optional[str]) -> Person:
    person = Person(
        canonical_name=name.strip(),
        norm_name=norm,
        wikidata_qid=qid,
        aliases=sorted(v for v in name_variants(name) if v != name.strip()),
        meta={},
    )
    db.add(person)
    db.flush()
    return person


def _is_deadlock(exc: OperationalError) -> bool:
    """Postgres SQLSTATE 40P01 (deadlock_detected). Two concurrent inserts can
    deadlock fighting over the same b-tree index page even when their VALUES
    don't conflict at all (see ix_people_norm_name/ix_organizations_norm_name)
    -- unlike a duplicate-key IntegrityError, this means Postgres itself
    aborted the transaction to break the cycle, so there's no row to look up:
    the whole transaction has to be rolled back and retried from here, not
    just re-selected. SQLite has no equivalent (its writer/writer contention
    is handled at the connection layer in db.py, as a retry-in-place, because
    SQLite's busy-timeout failure doesn't poison the transaction the way a
    real deadlock does)."""
    return getattr(getattr(exc, "orig", None), "pgcode", None) == "40P01"


# Same shape as the HTTP retry backoff (config.HTTP_BACKOFF_BASE/JITTER,
# see providers/base.py) but kept local: this waits out a DB transaction,
# not a network call, and an immediate retry mostly just re-collides -- the
# other side of a deadlock is often mid-way through several more seconds of
# its own research before it commits (observed directly: two real co-hosts,
# each repeatedly discovering the other as a counterpart, kept deadlocking on
# the SAME norm_name for multiple retries in a row with no delay).
_DEADLOCK_BACKOFF_BASE = 0.3   # seconds
_DEADLOCK_BACKOFF_JITTER = 0.4


def _deadlock_backoff(attempt: int) -> None:
    time.sleep(_DEADLOCK_BACKOFF_BASE * (2 ** attempt) + random.uniform(0, _DEADLOCK_BACKOFF_JITTER))


def _is_locked(exc: Exception) -> bool:
    """SQLite 'database is locked' -- surfaces once busy_timeout's own
    in-driver wait (5s, see db.py's _tune_sqlite) is exhausted while a write
    is still contending for the file lock. The module docstring on
    _is_deadlock claims SQLite's writer/writer contention is "handled at the
    connection layer... as a retry-in-place" -- true only up to that 5s
    ceiling. A slow bulk write (e.g. expansion._prune_invalid_nodes deleting
    hundreds of Claude-flagged junk-organization edges after researching a
    very public figure) can genuinely still be contending past it, and nothing
    retries beyond busy_timeout's own wait -- confirmed live, repeatedly,
    against this exact code path.

    Also matches a PendingRollbackError wrapping the same message: confirmed
    live that a lock during db.commit()'s internal flush sometimes surfaces
    this way instead of a bare OperationalError -- PendingRollbackError has
    no `.orig` at all, so `getattr(exc, "orig", exc)` falls back to the
    exception itself, and its own str() already embeds "Original exception
    was: ... database is locked", so the same substring check still matches.
    Every caller must catch BOTH exception types for this to matter --
    catching only OperationalError re-raises a PendingRollbackError
    immediately, unretried, which is exactly what shipped and failed live."""
    msg = str(getattr(exc, "orig", exc) or "")
    return "database is locked" in msg.lower()


def _is_transient(exc: Exception) -> bool:
    """Either flavor of "retry me": a SQLite lock timeout (_is_locked) or a
    Postgres deadlock (_is_deadlock). Every retry site below should check
    this, not _is_deadlock alone -- _is_deadlock's own docstring is explicit
    that it only recognizes Postgres's SQLSTATE, so on SQLite it is always
    False and a caller checking only it re-raises a lock immediately, with
    zero retries. That gap was real, not hypothetical: _new_person_or_existing
    and get_or_create_org (below) shipped with exactly that gap -- their
    SAVEPOINT/backoff scaffolding only ever fired for a deadlock, never for
    the SQLite lock this whole file is otherwise built to survive -- until
    this function unified the two checks.

    Takes Exception, not OperationalError, because _is_locked also matches a
    PendingRollbackError (see its docstring) -- callers must catch
    `except (OperationalError, PendingRollbackError)`, not OperationalError
    alone, or exactly this scenario re-raises unretried. Confirmed live: a
    lock during a multi-row bulk delete's db.commit() surfaced as
    PendingRollbackError, not OperationalError, and every retry site here was
    (at the time) only catching the latter."""
    return _is_locked(exc) or _is_deadlock(exc)


def flush_in_savepoint(db: Session, mutate: Callable[[], None],
                       _retries: int = 5) -> None:
    """Apply `mutate` and flush it INSIDE a SAVEPOINT, retrying on a transient
    lock/deadlock (see _is_transient).

    Exists because of a specific, confirmed failure shape: `Session.begin_nested()`
    flushes any PENDING state BEFORE it establishes the SAVEPOINT (SQLAlchemy's
    SessionTransaction._take_snapshot), and it does so regardless of
    autoflush=False. So an uncommitted ORM mutation left lying in the session
    gets written by the NEXT unrelated savepoint entry -- and if that write hits
    a lock, the savepoint never gets created, SQLAlchemy deactivates the
    transaction, and every retry that follows dies instantly on
    PendingRollbackError ("first issue Session.rollback()"). Worse, that error's
    text embeds the original "database is locked", so _is_locked keeps calling
    it transient and the caller burns its whole retry budget in milliseconds
    before dropping the node. Observed live: a lock on _homonym_conflict's
    leftover `people.metadata` UPDATE, flushed by _new_person_or_existing's
    savepoint, silently dropped nodes from a /connect walk.

    The cure is to never LEAVE a mutation pending. Applying it inside the
    savepoint means a lock rolls back only to the savepoint -- the recoverable
    shape that save_source and add_edge_from_extraction already survive
    routinely -- instead of poisoning the whole session.

    NOT a substitute for commit_with_retry: this leaves the change uncommitted
    (flushed only), so it still belongs to the caller's transaction and is
    still discarded if that transaction rolls back. That's deliberate -- these
    are advisory writes that should follow the fate of the work around them.
    """
    for attempt in range(_retries + 1):
        try:
            with db.begin_nested():
                mutate()
                db.flush()
            return
        except (OperationalError, PendingRollbackError) as exc:
            if not _is_transient(exc) or attempt >= _retries:
                raise
            _deadlock_backoff(attempt)


_T = TypeVar("_T")


def commit_with_retry(db: Session, apply: Optional[Callable[[], _T]] = None,
                      _retries: int = 5) -> Optional[_T]:
    """Commit, retrying with backoff on a transient SQLite lock or Postgres
    deadlock (see _is_transient).

    This is NOT a bare `db.commit()` retry. Confirmed empirically (see
    tests/test_commit_retry.py): after a failed flush/commit, the Session's
    DBAPI transaction is aborted and unusable until db.rollback() runs -- and
    rollback() reverts any already-persistent object's pending attribute
    change back to its last-committed value, and expunges any brand-new
    object that was never successfully flushed. So a second db.commit() with
    no rollback() in between either raises again or (if there was nothing
    left pending) silently "succeeds" having committed nothing. `apply`
    exists to make the retry actually correct: it's called again before every
    attempt -- including the first -- to (re)establish whatever this commit
    is meant to persist, so attempt 2 redoes the same work attempt 1 lost to
    rollback() instead of committing an empty transaction.

    `apply` must therefore be safe to call more than once (idempotent
    mutation, or re-add()-ing already-constructed objects -- both are true of
    every call site this is used from). Its return value (if any) is returned
    once the commit that followed it actually succeeds.
    """
    for attempt in range(_retries + 1):
        try:
            result = apply() if apply is not None else None
            db.commit()
            return result
        except (OperationalError, PendingRollbackError) as exc:
            db.rollback()
            if not _is_transient(exc) or attempt >= _retries:
                raise
            _deadlock_backoff(attempt)
    return None  # unreachable (loop either returns or raises)


def delete_relationship_edges_with_retry(db: Session, condition, _retries: int = 5) -> int:
    """Bulk-delete RelationshipEdge rows matching `condition`, retrying with
    backoff on a transient SQLite lock or Postgres deadlock (see _is_transient)
    instead of letting a slow prune delete surface as a hard job failure.

    Wrapped in a SAVEPOINT (db.begin_nested), not a plain retry: this runs
    mid-transaction, after edges earlier in the same node's processing were
    already added to the session (see expansion._prune_invalid_nodes's own
    docstring on why it flushes first before this delete) -- a plain retry
    would need a rollback() that reverts that already-accumulated work too,
    not just this delete. Mirrors _new_person_or_existing's SAVEPOINT
    reasoning for the exact same "don't lose a sibling's pending work" hazard.

    Returns the number of rows deleted.
    """
    for attempt in range(_retries + 1):
        try:
            with db.begin_nested():
                result = db.query(RelationshipEdge).filter(condition).delete(
                    synchronize_session=False)
            return result
        except (OperationalError, PendingRollbackError) as exc:
            if not _is_transient(exc) or attempt >= _retries:
                raise
            _deadlock_backoff(attempt)
    return 0  # unreachable (loop either returns or raises), keeps type checkers happy


def _new_person_or_existing(db: Session, name: str, norm: str,
                            qid: Optional[str], _retries: int = 5) -> Person:
    """_new_person, tolerant of a concurrent insert of the same norm_name
    racing in from another worker's own Session -- nodes within a hop, and the
    two connect_people sides, now run on separate threads/sessions (see
    expand_graph / connect.py), and each independently checks-then-creates.
    Two workers can both miss an existing row and both attempt to insert the
    same norm_name; norm_name is unique at the DB level (see Person.norm_name
    in models.py), so the loser's flush raises IntegrityError instead of
    silently duplicating the person. Recovering by re-selecting is correct,
    not just safe: the winner's committed row IS the person both callers
    wanted.

    Also tolerant of a genuine Postgres deadlock (see _is_deadlock) between
    two DIFFERENT norm_names landing on the same index page -- there, a
    re-select can legitimately still come back empty (neither side "won" the
    row we wanted), so this retries the whole insert attempt itself, bounded,
    rather than assuming an IntegrityError-shaped resolution.

    Wrapped in a SAVEPOINT (db.begin_nested), not a plain db.rollback(): this
    is called mid-way through _process_person, AFTER the subject person has
    already been resolved and flushed (uncommitted) in this SAME session/
    transaction -- a plain rollback() here reverts the WHOLE transaction, not
    just this insert, silently wiping out that already-flushed subject too.
    That's not hypothetical: it's exactly what caused a downstream
    ForeignKeyViolation ("person_a_id ... is not present in table people")
    the first time this was tried with a bare rollback() -- the subject's own
    row had been rolled back out from under it by ITS COUNTERPART's recovery
    path. A SAVEPOINT scopes the rollback to just this insert."""
    try:
        with db.begin_nested():
            return _new_person(db, name, norm, qid)
    except IntegrityError:
        existing = db.execute(
            select(Person).where(Person.norm_name == norm)
        ).scalar_one_or_none()
        if existing is None:
            raise  # the constraint fired on something else -- don't swallow that
        _merge_aliases(existing, name)
        return existing
    except (OperationalError, PendingRollbackError) as exc:
        if not _is_transient(exc) or _retries <= 0:
            raise
        _deadlock_backoff(5 - _retries)
        return _new_person_or_existing(db, name, norm, qid, _retries - 1)


def _merge_aliases(person: Person, surface: str) -> None:
    aliases = set(person.aliases or [])
    for v in name_variants(surface):
        if v and v != person.canonical_name:
            aliases.add(v)
    # Prefer the longest surface form as the canonical display name — but never
    # promote a scraped-chrome surface ("Bill Gates - Wikipedia") over it. That
    # string shares this person's norm_name (chrome is stripped before keying),
    # so silently adopting it as canonical_name makes the shape check that
    # protects this SAME node from _prune_invalid_nodes start failing on it.
    stripped = surface.strip()
    if (len(stripped) > len(person.canonical_name)
            and not is_noise_name(stripped) and looks_like_person_name(stripped)):
        aliases.add(person.canonical_name)
        person.canonical_name = stripped
        aliases.discard(person.canonical_name)
    if aliases != set(person.aliases or []):
        person.aliases = sorted(aliases)


def get_or_create_org(
    db: Session, name: str, org_type: str = "unknown", allow_create: bool = True,
    _retries: int = 5,
) -> Optional[Organization]:
    name = _strip_nul(name)
    norm = org_norm_key(name)
    if not norm:
        return None
    existing = db.execute(
        select(Organization).where(Organization.norm_name == norm)
    ).scalar_one_or_none()
    if existing:
        if existing.type == "unknown" and org_type != "unknown":
            existing.type = org_type
        return existing
    if not allow_create:
        return None
    org = Organization(name=name.strip(), norm_name=norm, type=org_type, meta={})
    try:
        # SAVEPOINT, not a plain flush()+rollback(): this can run mid-way
        # through _process_person, after the subject person was already
        # flushed (uncommitted) in this same session -- see the matching
        # comment on _new_person_or_existing for why a bare rollback() here
        # would silently wipe that out from under its own caller.
        #
        # db.add(org) belongs INSIDE this block, not before it -- confirmed
        # by a failing test (tests/test_commit_retry.py) written to exercise
        # the retry path: with add() outside the savepoint, a failed attempt
        # leaves `org` added-but-unflushed at the OUTER transaction scope
        # (rollback-to-savepoint only undoes what happened INSIDE the
        # savepoint), and the recursive retry below constructs and adds a
        # SECOND org with the same norm_name on top of it. Both then land in
        # the same flush once one attempt finally succeeds, and the real
        # norm_name UNIQUE constraint fires -- not a deadlock/lock retry at
        # all, an IntegrityError this function was never meant to hit here.
        with db.begin_nested():
            db.add(org)
            db.flush()
    except IntegrityError:
        # same race as _new_person_or_existing: another worker's session won
        # the insert for this norm_name first.
        existing = db.execute(
            select(Organization).where(Organization.norm_name == norm)
        ).scalar_one_or_none()
        if existing is None:
            raise
        if existing.type == "unknown" and org_type != "unknown":
            existing.type = org_type
        return existing
    except (OperationalError, PendingRollbackError) as exc:
        # see _new_person_or_existing / _is_transient: either a Postgres
        # deadlock between two DIFFERENT norm_names on the same index page
        # (a re-select can legitimately still miss, so retry the whole
        # attempt) or a SQLite lock timeout -- bounded either way.
        if not _is_transient(exc) or _retries <= 0:
            raise
        _deadlock_backoff(5 - _retries)
        return get_or_create_org(db, name, org_type, allow_create, _retries - 1)
    return org


def save_source(
    db: Session, result: SearchResult, query_used: str, full_text: Optional[str] = None,
    _retries: int = 5,
) -> Source:
    # Scraped page text (and, rarely, the query string that produced it, or
    # even the URL) occasionally carries a stray NUL byte (encoding noise,
    # binary content leaking through, or a NUL-codepoint escape in an
    # upstream API's JSON). SQLite stores it without complaint — Postgres's text type
    # flat-out rejects embedded NULs, raising ValueError at flush time. Strip
    # rather than reject: a lost null byte costs nothing, a failed source
    # save costs the whole node's research. Stripped BEFORE the dedup lookup
    # below so a NUL-bearing and NUL-free copy of the same URL match.
    url = _strip_nul(result.url)
    query_used = _strip_nul(query_used)
    # `url` has no DB-level uniqueness constraint (it's just indexed), so this
    # check-then-insert isn't atomic. A handful of enrichment sources reuse one
    # fixed URL across every person (e.g. openalex.org for coauthors_enrichment),
    # so under real concurrent writers (Postgres; SQLite's single-writer lock
    # happened to serialize this away) two people's threads can both miss each
    # other's uncommitted insert and each create a row. `.first()` rather than
    # `.scalar_one_or_none()` tolerates that instead of crashing on it — these
    # are just cached source snippets, a rare duplicate is harmless.
    existing = db.execute(
        select(Source).where(Source.url == url)
    ).scalars().first()
    if existing:
        if full_text and not existing.full_text:
            existing.full_text = _strip_nul(full_text)
        return existing

    title = _strip_nul(result.title) or ""
    snippet = _strip_nul(result.snippet) or ""
    full_text_clean = _strip_nul(full_text)
    provider = result.provider

    # SAVEPOINT + retry, not a bare add()+flush(): this is the single most
    # frequently executed write in the whole app (one call per fetched search
    # result) and it runs mid-transaction, before the per-node's own final
    # commit -- confirmed live, this exact statement is what first surfaced
    # "database is locked" with zero retry protection (see PR history). A
    # fresh Source is constructed on every attempt rather than reusing the
    # same instance: a failed flush() expunges a never-persisted object from
    # the session (confirmed empirically, see tests/test_commit_retry.py),
    # so retrying with the SAME instance would just re-add a still-transient
    # object -- reconstructing is simpler than reasoning about whether that's
    # safe, and it's what _new_person_or_existing already does for the same
    # reason.
    for attempt in range(_retries + 1):
        try:
            source = Source(
                url=url, title=title, snippet=snippet, full_text=full_text_clean,
                provider=provider, query_used=query_used,
            )
            with db.begin_nested():
                db.add(source)
                db.flush()
            return source
        except (OperationalError, PendingRollbackError) as exc:
            if not _is_transient(exc) or attempt >= _retries:
                raise
            _deadlock_backoff(attempt)
    raise AssertionError("unreachable")  # loop either returns or raises


# --- org<->org facts (never bridged into the person graph) -----------------
def record_coinvestment(db: Session, orgs: list, company: str,
                        source_url: str = "") -> int:
    """Record, on each firm, that it co-invested with the others in `company`.

    Deliberately an ORGANIZATION-level fact ONLY, stored in `Organization.meta`
    — never a RelationshipEdge, and never used to infer a tie between one
    firm's employee and another's (no "employee of Org A -> Org A co-invested
    with Org B -> employee/CEO of Org B" chaining). Pathfinding
    (graph.connect._adjacency) reads RelationshipEdge rows exclusively, so a
    fact stored here is structurally inert for person-to-person routing —
    it can never become a bridge between two people.

    Returns the number of (firm, other-firm) pairs newly recorded.
    """
    orgs = [o for o in orgs if o is not None]
    if len(orgs) < 2:
        return 0
    recorded = 0
    for org in orgs:
        others = [o for o in orgs if o.id != org.id]
        meta = dict(org.meta or {})
        book = dict(meta.get("co_investments") or {})
        for other in others:
            entry = book.setdefault(other.norm_name,
                                    {"firm": other.name, "rounds": []})
            if not any(r.get("company") == company for r in entry["rounds"]):
                entry["rounds"].append({"company": company,
                                        "source_url": source_url})
                recorded += 1
        book_changed = meta.get("co_investments") != book
        meta["co_investments"] = book
        if book_changed:
            org.meta = meta
    return recorded


# --- status policy ---------------------------------------------------------
def derive_status(relationship_type: str, confidence: float) -> str:
    """Confidence tier (weak/candidate/strong); family_social never reaches strong."""
    t = tier(confidence)
    if relationship_type == "family_social" and t == "strong":
        return "candidate"
    return t


# --- edge upsert -----------------------------------------------------------
def add_edge_from_extraction(
    db: Session,
    subject: Person,
    edge: ExtractedEdge,
    depth: int,
    source: Optional[Source],
    counterpart,  # Person | Organization
    _retries: int = 5,
) -> Optional[RelationshipEdge]:
    """Persist one ExtractedEdge, applying the (a,b,type,source_url) dedup rule."""
    is_person = edge.other_kind == "person"
    other_id = counterpart.id if is_person else None
    org_id = counterpart.id if not is_person else None
    source_id = source.id if source else None
    conf = edge.confidence_adjusted

    # Dedup rule: same (person_a, counterpart, relationship_type, source_url).
    existing = db.execute(
        select(RelationshipEdge).where(
            RelationshipEdge.person_a_id == subject.id,
            RelationshipEdge.person_b_id == other_id,
            RelationshipEdge.organization_id == org_id,
            RelationshipEdge.relationship_type == edge.relationship_type,
            RelationshipEdge.source_id == source_id,
        )
    ).scalar_one_or_none()
    if existing:
        if conf > (existing.confidence_raw or 0):
            existing.confidence_raw = conf
            existing.confidence_base = edge.confidence_base
            existing.status = derive_status(edge.relationship_type, conf)
            existing.signals = edge.signals.model_dump()
        return existing

    method = _strip_nul(edge.method)
    evidence_snippet = _strip_nul(edge.evidence_snippet)
    confidence_base = round(edge.confidence_base, 3)
    confidence_raw = round(conf, 3)
    signals = edge.signals.model_dump()
    status = derive_status(edge.relationship_type, conf)

    # SAVEPOINT + retry: same reasoning as save_source, and just as hot a
    # path -- one call per persisted relationship, mid-transaction, before
    # the per-node's own final commit. A fresh RelationshipEdge is built on
    # every attempt for the same reason save_source rebuilds its object: a
    # failed flush() expunges a never-persisted instance from the session
    # (confirmed empirically), so reusing it on retry wouldn't reattach it.
    for attempt in range(_retries + 1):
        try:
            row = RelationshipEdge(
                person_a_id=subject.id,
                person_b_id=other_id,
                organization_id=org_id,
                relationship_type=edge.relationship_type,
                method=method,
                evidence_snippet=evidence_snippet,
                source_id=source_id,
                confidence_base=confidence_base,
                confidence_raw=confidence_raw,
                signals=signals,
                depth=depth,
                status=status,
            )
            with db.begin_nested():
                db.add(row)
                db.flush()
            return row
        except (OperationalError, PendingRollbackError) as exc:
            if not _is_transient(exc) or attempt >= _retries:
                raise
            _deadlock_backoff(attempt)
    raise AssertionError("unreachable")  # loop either returns or raises
